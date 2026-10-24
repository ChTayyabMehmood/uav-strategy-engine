"""Constraint repair for routes and chromosomes.

Repair is intentionally kept outside the GA's genetic operators.  The GA
produces a candidate; this module makes a best effort to turn it into a
feasible straight-leg route.  If local repair cannot fix a route, the route is
regenerated from scratch (a documented fallback).  The evaluator still performs
an independent feasibility check afterwards, so repair can never silently mark
an invalid route as valid.
"""

from __future__ import annotations

import math

import numpy as np

from environment.grid import GridEnvironment
from environment.targets import Target
from simulation.constraints import (
    nearest_free_cell,
    segment_is_clear,
    segment_samples,
)
from uav.chromosome import Chromosome, DroneRoute, Waypoint
from uav.drone import Drone

MAX_RAY_DISTANCE = 22.0


def _ray_cell(
    env: GridEnvironment,
    origin: tuple[int, int],
    angle: float,
    max_distance: float,
    step: float = 0.5,
) -> tuple[int, int] | None:
    """March from ``origin`` and return the farthest clear cell on the ray."""

    ox, oy = origin
    dx, dy = math.cos(angle), math.sin(angle)
    last: tuple[int, int] | None = None
    t = step
    while t <= max_distance + 1e-9:
        x = int(round(ox + dx * t))
        y = int(round(oy + dy * t))
        if not env.is_free(x, y):
            break
        last = (x, y)
        t += step
    return last


def sample_area_waypoint(
    env: GridEnvironment,
    origin: tuple[int, int],
    rng: np.random.Generator,
    targets: list[Target] | None = None,
) -> tuple[int, int] | None:
    """Sample one reachable waypoint using a clear straight ray.

    With probability 0.6 the ray points toward a randomly chosen target (with a
    small angular jitter), which seeds the population with useful exploration.
    Otherwise the direction is random.  The returned cell always has a clear
    line of sight from ``origin``.
    """

    targets = targets or []
    for _attempt in range(50):
        if targets and rng.random() < 0.6:
            target = targets[int(rng.integers(0, len(targets)))]
            angle = math.atan2(target.y - origin[1], target.x - origin[0])
            angle += float(rng.uniform(-0.15, 0.15))
            max_distance = min(
                math.hypot(target.x - origin[0], target.y - origin[1]),
                MAX_RAY_DISTANCE * 1.4,
            )
        else:
            angle = float(rng.uniform(-math.pi, math.pi))
            max_distance = float(rng.uniform(4.0, MAX_RAY_DISTANCE))

        cell = _ray_cell(env, origin, angle, max_distance)
        if cell is None or cell == origin:
            # Try a short free step if the ray was immediately blocked.
            cell = _ray_cell(env, origin, angle, 2.0)
        if cell is None or cell == origin:
            continue
        if segment_is_clear(env, origin[0], origin[1], cell[0], cell[1]):
            return cell
    return None


def random_valid_route(
    env: GridEnvironment,
    start: tuple[int, int],
    n_waypoints: int,
    rng: np.random.Generator,
    targets: list[Target] | None = None,
) -> DroneRoute | None:
    """Build a random straight-leg route that is clear by construction."""

    waypoints: list[Waypoint] = []
    previous = start
    for _ in range(n_waypoints):
        cell = sample_area_waypoint(env, previous, rng, targets)
        if cell is None:
            return None
        waypoints.append(
            Waypoint(
                x=int(cell[0]),
                y=int(cell[1]),
                speed=int(rng.integers(0, 3)),
                sensor=int(rng.integers(0, 3)),
            )
        )
        previous = cell
    return DroneRoute(waypoints=waypoints)


def _repair_cell(
    env: GridEnvironment,
    previous: tuple[int, int],
    current: tuple[int, int],
    following: tuple[int, int] | None,
    max_radius: int,
) -> tuple[int, int] | None:
    """Find a nearby cell that makes both adjacent legs clear."""

    cx, cy = current
    for radius in range(0, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if not env.is_free(nx, ny):
                    continue
                if not segment_is_clear(
                    env, previous[0], previous[1], nx, ny
                ):
                    continue
                if following is not None and not segment_is_clear(
                    env, nx, ny, following[0], following[1]
                ):
                    continue
                return int(nx), int(ny)
    return None


def repair_route(
    env: GridEnvironment,
    route: DroneRoute,
    start: tuple[int, int],
    rng: np.random.Generator,
    targets: list[Target] | None = None,
    max_radius: int = 8,
) -> DroneRoute | None:
    """Repair a route in place conceptually and return a new route or ``None``.

    Steps:
    1. Clamp waypoints into the grid.
    2. Snap forbidden waypoints to the nearest free cell.
    3. Nudge waypoints until both adjacent legs are clear.
    4. Verify the complete route; regenerate if local repair failed.
    """

    waypoints = [wp for wp in route.waypoints]

    for i, wp in enumerate(waypoints):
        if not env.in_bounds(wp.x, wp.y):
            x = min(max(wp.x, 0), env.width - 1)
            y = min(max(wp.y, 0), env.height - 1)
            waypoints[i] = Waypoint(x, y, wp.speed, wp.sensor)
        if not env.is_free(waypoints[i].x, waypoints[i].y):
            free = nearest_free_cell(env, waypoints[i].x, waypoints[i].y)
            if free is None:
                return None
            waypoints[i] = Waypoint(
                free[0], free[1], waypoints[i].speed, waypoints[i].sensor
            )

    for _pass in range(4):
        changed = False
        for i, wp in enumerate(waypoints):
            previous = (
                (waypoints[i - 1].x, waypoints[i - 1].y)
                if i > 0
                else start
            )
            following = (
                (waypoints[i + 1].x, waypoints[i + 1].y)
                if i + 1 < len(waypoints)
                else None
            )
            current = (wp.x, wp.y)
            first_ok = segment_is_clear(
                env, previous[0], previous[1], current[0], current[1]
            )
            second_ok = following is None or segment_is_clear(
                env, current[0], current[1], following[0], following[1]
            )
            if first_ok and second_ok:
                continue
            replacement = _repair_cell(
                env, previous, current, following, max_radius
            )
            if replacement is not None:
                waypoints[i] = Waypoint(
                    replacement[0], replacement[1], wp.speed, wp.sensor
                )
                changed = True
        if not changed:
            break

    previous = start
    for wp in waypoints:
        if not segment_is_clear(env, previous[0], previous[1], wp.x, wp.y):
            break
        previous = (wp.x, wp.y)
    else:
        return DroneRoute(waypoints=waypoints)

    # Local repair failed: documented fallback is a fresh valid route.
    for _attempt in range(30):
        route_candidate = random_valid_route(
            env, start, len(route.waypoints), rng, targets
        )
        if route_candidate is not None:
            return route_candidate
    return None


def repair_chromosome(
    env: GridEnvironment,
    chromosome: Chromosome,
    drones: list[Drone],
    rng: np.random.Generator,
) -> Chromosome | None:
    """Repair every route of a chromosome, or return ``None`` if impossible."""

    if len(chromosome.routes) != len(drones):
        return None
    if chromosome.deployments:
        if len(chromosome.deployments) != len(drones):
            return None
        starts = list(chromosome.deployments)
    else:
        starts = [drone.start for drone in drones]
    for start in starts:
        if not env.is_free(*start):
            return None
    routes: list[DroneRoute] = []
    for route, start in zip(chromosome.routes, starts):
        repaired = repair_route(env, route, start, rng, env.targets)
        if repaired is None:
            return None
        routes.append(repaired)
    return Chromosome(
        routes=routes,
        deployments=list(chromosome.deployments),
    )


def visited_cells(
    env: GridEnvironment,
    start: tuple[int, int],
    route: DroneRoute,
) -> list[tuple[int, int]]:
    """Return every sampled cell on a route (useful for coverage maps)."""

    cells: list[tuple[int, int]] = []
    previous = start
    for wp in route.waypoints:
        samples = segment_samples(previous[0], previous[1], wp.x, wp.y)
        cells.extend((int(x), int(y)) for x, y in samples)
        previous = (wp.x, wp.y)
    return cells
