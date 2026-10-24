"""Core domain model: targets, priority map, grid, drones and chromosomes.

This module groups the data structures that the rest of the engine uses.
Keeping them together makes the assignment easy to follow: a chromosome
stores deployment cells, waypoints, speed modes and sensor modes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from config import EnvironmentConfig, FleetConfig


# ======================================================================
# Targets and priority map
# ======================================================================

"""Disaster-surveillance targets."""


@dataclass(frozen=True)
class Target:
    """A point of interest that a UAV sensor may detect.

    ``priority`` is a relative importance weight in ``(0, 1]``.  Each target
    also owns a Gaussian priority zone used by the area-coverage metric.
    """

    target_id: str
    x: int
    y: int
    priority: float

    def to_dict(self) -> dict:
        return asdict(self)


def total_priority(targets: list[Target]) -> float:
    """Return the summed priority of ``targets`` (0.0 for an empty list)."""

    return float(sum(t.priority for t in targets))


def build_priority_map(
    width: int,
    height: int,
    targets: list[Target],
    radius: float = 3.0,
) -> np.ndarray:
    """Return a normalised priority-zone map around the targets.

    Each target owns a circular priority zone.  The zone is normalised to one
    unit of mass before it is multiplied by the target priority, so edge
    truncation does not change the target's importance.  A sensor footprint
    covering the whole zone therefore contributes exactly its full priority.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    yy, xx = np.ogrid[:height, :width]
    field = np.zeros((height, width), dtype=float)
    for target in targets:
        zone = (
            (xx - target.x) ** 2 + (yy - target.y) ** 2
            <= radius * radius
        )
        cells = int(zone.sum())
        if cells:
            field[zone] += target.priority / cells
    return field

# ======================================================================
# Grid and environment generation
# ======================================================================

"""Discretised operational environment for the UAV mission.

The environment is a 2-D grid of uniform cells.  It stores:

* ``obstacle_mask`` - impassable terrain / buildings;
* ``nofly_mask``    - restricted airspace that a UAV must never enter;
* ``hazard_map``    - continuous hazard intensity in ``[0, 1]``;
* ``targets``       - weighted disaster-surveillance targets;
* ``starts``        - deterministic drone launch cells.

The whole map is generated from a single seed so experiments are reproducible.
"""


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

# ======================================================================
# Drone model
# ======================================================================

"""UAV fleet model and normalised performance assumptions.

All values are deliberately *simulation units*.  They are not datasheet
parameters for a real drone.  Keeping them normalised makes the trade-offs
between speed, energy, sensor use, and time explicit.
"""


# Speed modes: 0 = slow, 1 = cruise, 2 = fast.
# ``cells_per_time`` is a normalised ground speed in grid cells per time unit.
SPEED_MODES: dict[int, dict[str, float | str]] = {
    0: {"name": "slow", "cells_per_time": 0.70},
    1: {"name": "cruise", "cells_per_time": 1.00},
    2: {"name": "fast", "cells_per_time": 1.50},
}
CRUISE_SPEED = float(SPEED_MODES[1]["cells_per_time"])

# Sensor modes: 0 = off, 1 = medium, 2 = high.
# ``radius`` is in grid cells (Euclidean distance from a sampled route point).
# ``energy_per_time`` is added while a leg is flown with that sensor mode.
SENSOR_MODES: dict[int, dict[str, float | str]] = {
    0: {"name": "off", "radius": 0.0, "energy_per_time": 0.0},
    1: {"name": "medium", "radius": 1.6, "energy_per_time": 0.35},
    2: {"name": "high", "radius": 3.2, "energy_per_time": 0.85},
}

# Energy per grid cell at cruise speed.  Movement energy scales with the square
# of the speed ratio (a simple aerodynamic-style drag assumption).
MOVE_ENERGY_PER_CELL = 1.0
SPEED_DRAG_EXPONENT = 2.0


@dataclass(frozen=True)
class Drone:
    """A single UAV and its hard mission limits."""

    drone_id: int
    start: tuple[int, int]
    max_flight_range: float
    battery_capacity: float

    def to_dict(self) -> dict:
        return asdict(self)


def speed_of(mode: int) -> float:
    """Return normalised speed for a mode, validating the mode index."""

    if mode not in SPEED_MODES:
        raise ValueError(f"invalid speed mode {mode!r}")
    return float(SPEED_MODES[mode]["cells_per_time"])


def sensor_radius(mode: int) -> float:
    if mode not in SENSOR_MODES:
        raise ValueError(f"invalid sensor mode {mode!r}")
    return float(SENSOR_MODES[mode]["radius"])


def sensor_energy_rate(mode: int) -> float:
    if mode not in SENSOR_MODES:
        raise ValueError(f"invalid sensor mode {mode!r}")
    return float(SENSOR_MODES[mode]["energy_per_time"])


def build_fleet(starts: list[tuple[int, int]], cfg: FleetConfig) -> list[Drone]:
    """Create ``cfg.n_drones`` drones at the supplied launch cells."""

    if len(starts) != cfg.n_drones:
        raise ValueError(
            f"expected {cfg.n_drones} start positions, got {len(starts)}"
        )
    return [
        Drone(
            drone_id=i,
            start=(int(x), int(y)),
            max_flight_range=cfg.max_flight_range,
            battery_capacity=cfg.battery_capacity,
        )
        for i, (x, y) in enumerate(starts)
    ]


def modes_table() -> dict:
    """Human-readable mode table for docs and JSON metadata."""

    return {
        "speed_modes": {k: dict(v) for k, v in SPEED_MODES.items()},
        "sensor_modes": {k: dict(v) for k, v in SENSOR_MODES.items()},
        "move_energy_per_cell": MOVE_ENERGY_PER_CELL,
        "speed_drag_exponent": SPEED_DRAG_EXPONENT,
    }

# ======================================================================
# Routes and chromosome
# ======================================================================

"""Chromosome representation for the fleet strategy.

A chromosome has one :class:`DroneRoute` per UAV.  Each route is a fixed-length
sequence of :class:`Waypoint` genes.  A gene stores:

* integer ``x`` / ``y`` grid coordinates (the waypoint the drone flies to);
* a categorical ``speed`` mode (0 slow, 1 cruise, 2 fast);
* a categorical ``sensor`` power mode (0 off, 1 medium, 2 high).

A second, shorter segment stores each drone's deployment cell.  It is evolved
alongside the routes, so the GA chooses where the fleet starts as well as where
it flies.  Settings on waypoint *i* govern the leg flown from the previous
point to waypoint *i*.

Waypoints are immutable (``frozen=True``), which makes it safe for crossover to
reuse them while still building independent route lists.  :meth:`copy` returns
a fully independent chromosome.
"""


@dataclass(frozen=True)
class Waypoint:
    x: int
    y: int
    speed: int = 1
    sensor: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValueError("waypoint coordinates must be integers")
        if self.speed not in SPEED_MODES:
            raise ValueError(f"invalid speed mode {self.speed!r}")
        if self.sensor not in SENSOR_MODES:
            raise ValueError(f"invalid sensor mode {self.sensor!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DroneRoute:
    waypoints: list[Waypoint] = field(default_factory=list)

    def copy(self) -> DroneRoute:
        return DroneRoute(waypoints=list(self.waypoints))

    def to_dict(self) -> dict:
        return {"waypoints": [wp.to_dict() for wp in self.waypoints]}


@dataclass
class Chromosome:
    """Complete fleet strategy: deployment cells plus one route per drone."""

    routes: list[DroneRoute] = field(default_factory=list)
    deployments: list[tuple[int, int]] = field(default_factory=list)

    def copy(self) -> Chromosome:
        return Chromosome(
            routes=[route.copy() for route in self.routes],
            deployments=list(self.deployments),
        )

    def start_for(
        self, drone_id: int, fallback: tuple[int, int]
    ) -> tuple[int, int]:
        """Return the evolved deployment cell, or a fixed fallback."""

        if drone_id < len(self.deployments):
            return self.deployments[drone_id]
        return fallback

    def signature(self) -> tuple:
        """Stable hashable representation used for evaluation caching."""

        routes = tuple(
            tuple(
                (wp.x, wp.y, wp.speed, wp.sensor)
                for wp in route.waypoints
            )
            for route in self.routes
        )
        return routes, tuple(self.deployments)

    def validate_structure(
        self,
        n_drones: int,
        n_waypoints: int,
        width: int | None = None,
        height: int | None = None,
        require_deployments: bool = False,
    ) -> None:
        """Raise ``ValueError`` if the chromosome is structurally invalid.

        Bounds are checked only when ``width`` / ``height`` are provided.
        ``Waypoint`` itself validates coordinates are integers and modes are
        valid.  Older tests and callers may omit deployments; the evaluator
        then falls back to the environment's fixed launch cells.
        """

        if not isinstance(self.routes, list):
            raise ValueError("routes must be a list")
        if len(self.routes) != n_drones:
            raise ValueError(
                f"expected {n_drones} routes, found {len(self.routes)}"
            )
        if self.deployments:
            if len(self.deployments) != n_drones:
                raise ValueError(
                    f"expected {n_drones} deployments, "
                    f"found {len(self.deployments)}"
                )
            for i, (x, y) in enumerate(self.deployments):
                if not isinstance(x, int) or not isinstance(y, int):
                    raise ValueError(f"deployment {i} must use integer cells")
                if width is not None and not 0 <= x < width:
                    raise ValueError(f"deployment {i} x={x} out of bounds")
                if height is not None and not 0 <= y < height:
                    raise ValueError(f"deployment {i} y={y} out of bounds")
        elif require_deployments:
            raise ValueError("deployments are required")
        for i, route in enumerate(self.routes):
            if not isinstance(route, DroneRoute):
                raise ValueError(f"route {i} is not a DroneRoute")
            if len(route.waypoints) != n_waypoints:
                raise ValueError(
                    f"route {i} has {len(route.waypoints)} waypoints, "
                    f"expected {n_waypoints}"
                )
            for j, wp in enumerate(route.waypoints):
                if not isinstance(wp, Waypoint):
                    raise ValueError(f"gene {i}:{j} is not a Waypoint")
                if width is not None and not 0 <= wp.x < width:
                    raise ValueError(f"waypoint {i}:{j} x={wp.x} out of bounds")
                if height is not None and not 0 <= wp.y < height:
                    raise ValueError(f"waypoint {i}:{j} y={wp.y} out of bounds")

    def to_dict(self) -> dict:
        return {
            "deployments": [list(cell) for cell in self.deployments],
            "routes": [route.to_dict() for route in self.routes],
        }

    @staticmethod
    def from_dict(data: dict) -> Chromosome:
        routes = [
            DroneRoute(
                waypoints=[
                    Waypoint(
                        x=int(wp["x"]),
                        y=int(wp["y"]),
                        speed=int(wp.get("speed", 1)),
                        sensor=int(wp.get("sensor", 1)),
                    )
                    for wp in route["waypoints"]
                ]
            )
            for route in data["routes"]
        ]
        return Chromosome(
            routes=routes,
            deployments=[
                (int(x), int(y))
                for x, y in data.get("deployments", [])
            ],
        )


def fleet_for_chromosome(
    base_fleet: list[Drone],
    chromosome: Chromosome,
) -> list[Drone]:
    """Return drones whose starts honour the evolved deployment genes."""

    return [
        Drone(
            drone_id=drone.drone_id,
            start=chromosome.start_for(drone.drone_id, drone.start),
            max_flight_range=drone.max_flight_range,
            battery_capacity=drone.battery_capacity,
        )
        for drone in base_fleet
    ]
