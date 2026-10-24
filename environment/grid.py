"""Discretised operational environment for the UAV mission.

The environment is a 2-D grid of uniform cells.  It stores:

* ``obstacle_mask`` - impassable terrain / buildings;
* ``nofly_mask``    - restricted airspace that a UAV must never enter;
* ``hazard_map``    - continuous hazard intensity in ``[0, 1]``;
* ``targets``       - weighted disaster-surveillance targets;
* ``starts``        - deterministic drone launch cells.

The whole map is generated from a single seed so experiments are reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import EnvironmentConfig
from environment.targets import Target, build_priority_map


class EnvironmentGenerationError(RuntimeError):
    """Raised when a seed cannot produce a usable, connected map."""


@dataclass
class GridEnvironment:
    """A fully materialised hazard grid."""

    obstacle_mask: np.ndarray
    nofly_mask: np.ndarray
    hazard_map: np.ndarray
    targets: list[Target]
    starts: list[tuple[int, int]]
    seed: int
    priority_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.obstacle_mask = np.asarray(self.obstacle_mask, dtype=bool)
        self.nofly_mask = np.asarray(self.nofly_mask, dtype=bool)
        self.hazard_map = np.asarray(self.hazard_map, dtype=float)
        if self.obstacle_mask.shape != self.nofly_mask.shape:
            raise ValueError("obstacle_mask and nofly_mask must have equal shapes")
        if self.obstacle_mask.shape != self.hazard_map.shape:
            raise ValueError("hazard_map must match the mask shape")
        if self.hazard_map.ndim != 2:
            raise ValueError("hazard_map must be two-dimensional")
        self.targets = list(self.targets)
        self.starts = [tuple(p) for p in self.starts]
        if self.priority_map is None:
            self.priority_map = build_priority_map(
                self.width, self.height, self.targets
            )
        else:
            self.priority_map = np.asarray(self.priority_map, dtype=float)
            if self.priority_map.shape != self.hazard_map.shape:
                raise ValueError("priority_map must match the hazard map shape")

    @property
    def width(self) -> int:
        return int(self.hazard_map.shape[1])

    @property
    def height(self) -> int:
        return int(self.hazard_map.shape[0])

    @property
    def forbidden_mask(self) -> np.ndarray:
        """Cells a UAV may never occupy (obstacles or no-fly zones)."""

        return self.obstacle_mask | self.nofly_mask

    @property
    def free_mask(self) -> np.ndarray:
        return ~self.forbidden_mask

    @property
    def free_cell_count(self) -> int:
        return int(self.free_mask.sum())

    @property
    def priority_mass(self) -> float:
        return float(self.priority_map.sum())

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not bool(self.forbidden_mask[y, x])

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "targets": [t.to_dict() for t in self.targets],
            "starts": [list(s) for s in self.starts],
            "obstacle_cells": int(self.obstacle_mask.sum()),
            "nofly_cells": int(self.nofly_mask.sum()),
            "hazard_mean": float(self.hazard_map.mean()),
            "hazard_max": float(self.hazard_map.max()),
            "free_cells": self.free_cell_count,
            "priority_mass": self.priority_mass,
        }


def _blobs(
    rng: np.random.Generator,
    height: int,
    width: int,
    n_blobs: int,
    radius_range: tuple[int, int],
) -> np.ndarray:
    """Paint circular blobs onto a boolean mask."""

    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    for _ in range(n_blobs):
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        radius = int(rng.integers(radius_range[0], radius_range[1] + 1))
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    return mask


def _largest_free_component(free_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the largest 4-connected free region as ``(x, y)`` cells."""

    height, width = free_mask.shape
    visited = np.zeros_like(free_mask, dtype=bool)
    best: list[tuple[int, int]] = []

    for y0 in range(height):
        for x0 in range(width):
            if not free_mask[y0, x0] or visited[y0, x0]:
                continue
            queue: deque[tuple[int, int]] = deque([(x0, y0)])
            visited[y0, x0] = True
            component: list[tuple[int, int]] = []
            while queue:
                x, y = queue.popleft()
                component.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and free_mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if len(component) > len(best):
                best = component
    return best


def _sample_spread(
    rng: np.random.Generator,
    cells: list[tuple[int, int]],
    count: int,
    min_distance: int,
    avoid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Sample ``count`` cells that are at least ``min_distance`` apart."""

    avoid = avoid or set()
    chosen: list[tuple[int, int]] = []
    candidates = [c for c in cells if c not in avoid]
    if len(candidates) < count:
        return None

    for spacing in (min_distance, max(1, min_distance // 2), 1):
        chosen = []
        pool = candidates[:]
        rng.shuffle(pool)
        for cell in pool:
            if all(
                max(abs(cell[0] - c[0]), abs(cell[1] - c[1])) >= spacing
                for c in chosen
            ):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
        if len(chosen) == count:
            return chosen
    return None


def _hazard_field(
    rng: np.random.Generator,
    cfg: EnvironmentConfig,
) -> np.ndarray:
    """Sum of Gaussian hazard pockets, clipped to ``[0, 1]``."""

    height, width = cfg.height, cfg.width
    yy, xx = np.ogrid[:height, :width]
    hazard = np.zeros((height, width), dtype=float)
    for _ in range(cfg.n_hazard_blobs):
        cy = float(rng.uniform(0, height))
        cx = float(rng.uniform(0, width))
        sigma = float(rng.uniform(*cfg.hazard_sigma))
        amplitude = float(rng.uniform(*cfg.hazard_amplitude))
        hazard += amplitude * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
        )
    return np.clip(hazard, 0.0, 1.0)


def generate_environment(
    cfg: EnvironmentConfig,
    n_drones: int,
) -> GridEnvironment:
    """Build a deterministic environment for ``n_drones`` UAVs.

    The generator retries internally when a random seed produces an
    unusable layout (too few free cells, disconnected starts/targets).  If no
    acceptable layout is found, :class:`EnvironmentGenerationError` is raised
    rather than silently returning a broken map.
    """

    if n_drones < 1:
        raise ValueError("n_drones must be >= 1")
    if cfg.n_targets < 0:
        raise ValueError("n_targets must be >= 0")
    if cfg.width < 5 or cfg.height < 5:
        raise ValueError("grid must be at least 5 x 5")

    rng = np.random.default_rng(cfg.seed)

    for _attempt in range(40):
        obstacles = _blobs(
            rng, cfg.height, cfg.width, cfg.n_obstacle_blobs, cfg.obstacle_radius
        )
        nofly = _blobs(
            rng, cfg.height, cfg.width, cfg.n_nofly_blobs, cfg.nofly_radius
        )
        free = ~(obstacles | nofly)
        if int(free.sum()) < max(40, 0.30 * free.size):
            continue

        component = _largest_free_component(free)
        if len(component) < max(n_drones + cfg.n_targets + 5, 0.30 * free.sum()):
            continue

        starts = _sample_spread(rng, component, n_drones, cfg.min_start_spacing)
        if starts is None:
            continue

        targets_cells = _sample_spread(
            rng,
            component,
            cfg.n_targets,
            cfg.min_target_spacing,
            avoid=set(starts),
        )
        if targets_cells is None:
            continue

        targets = [
            Target(
                target_id=f"T{i}",
                x=int(x),
                y=int(y),
                priority=round(float(rng.uniform(*cfg.target_priority)), 3),
            )
            for i, (x, y) in enumerate(targets_cells)
        ]
        hazard = _hazard_field(rng, cfg)
        return GridEnvironment(
            obstacle_mask=obstacles,
            nofly_mask=nofly,
            hazard_map=hazard,
            targets=targets,
            starts=starts,
            seed=cfg.seed,
            priority_map=build_priority_map(
                cfg.width,
                cfg.height,
                targets,
                radius=cfg.target_zone_radius,
            ),
        )

    raise EnvironmentGenerationError(
        "could not generate a usable environment from seed "
        f"{cfg.seed}; try another seed or fewer/smaller obstacle blobs"
    )
