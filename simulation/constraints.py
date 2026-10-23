"""Grid geometry and hard-constraint checks.

This module deliberately contains no genetic-algorithm logic.  It provides
small helpers that the evaluator, the repair operator, and the tests share:

* dense sampling of a straight flight leg;
* line-of-sight / forbidden-cell checks;
* nearest free-cell search.

Dense sampling (every 0.5 cell) is used so a leg is checked along its whole
length, not just at its endpoints.  A route is legal only when every sampled
cell is inside the grid and outside obstacles and no-fly zones.
"""

from __future__ import annotations

import math

import numpy as np

from environment.grid import GridEnvironment

SAMPLE_STEP = 0.5


def segment_samples(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: float = SAMPLE_STEP,
) -> np.ndarray:
    """Return integer cells sampled densely along a straight segment.

    The endpoints are always included.  Sample density is independent of leg
    length, which makes hazard integration and detection checks consistent.
    """

    distance = math.hypot(x1 - x0, y1 - y0)
    if distance == 0.0:
        return np.array([[int(x0), int(y0)]], dtype=int)
    n = max(1, int(math.ceil(distance / step)))
    ts = np.linspace(0.0, 1.0, n + 1)
    xs = np.rint(x0 + (x1 - x0) * ts).astype(int)
    ys = np.rint(y0 + (y1 - y0) * ts).astype(int)
    return np.stack((xs, ys), axis=1)


def segment_distance(x0: int, y0: int, x1: int, y1: int) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def samples_in_bounds(env: GridEnvironment, samples: np.ndarray) -> bool:
    return bool(
        np.all(samples[:, 0] >= 0)
        and np.all(samples[:, 0] < env.width)
        and np.all(samples[:, 1] >= 0)
        and np.all(samples[:, 1] < env.height)
    )


def segment_is_clear(
    env: GridEnvironment,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    """True if the whole straight leg is in bounds and over free cells."""

    samples = segment_samples(x0, y0, x1, y1)
    if not samples_in_bounds(env, samples):
        return False
    xs = samples[:, 0]
    ys = samples[:, 1]
    return not bool(np.any(env.forbidden_mask[ys, xs]))


def nearest_free_cell(
    env: GridEnvironment,
    x: int,
    y: int,
    max_radius: int = 25,
) -> tuple[int, int] | None:
    """Find the closest free cell to ``(x, y)`` by Chebyshev rings."""

    if env.is_free(x, y):
        return int(x), int(y)
    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx, ny = x + dx, y + dy
                if env.is_free(nx, ny):
                    return int(nx), int(ny)
    return None


def route_is_clear(
    env: GridEnvironment,
    start: tuple[int, int],
    waypoints: list[tuple[int, int]],
) -> bool:
    prev_x, prev_y = start
    for x, y in waypoints:
        if not env.is_free(x, y):
            return False
        if not segment_is_clear(env, prev_x, prev_y, x, y):
            return False
        prev_x, prev_y = x, y
    return True
