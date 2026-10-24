"""Spatially meaningful mutation and constraint repair hook."""

from __future__ import annotations

import numpy as np

from environment.grid import GridEnvironment
from simulation.repair import (
    repair_chromosome,
    sample_area_waypoint,
)
from uav.chromosome import Chromosome, Waypoint, fleet_for_chromosome
from uav.drone import Drone

MOVE_PROBABILITY = 0.60
SPEED_PROBABILITY = 0.20
DEPLOYMENT_RATE = 0.05


def mutate(
    chromosome: Chromosome,
    env: GridEnvironment,
    drones: list[Drone],
    rng: np.random.Generator,
    mutation_rate: float,
    max_step: int = 4,
    deployment_rate: float = DEPLOYMENT_RATE,
) -> Chromosome:
    """Mutate a copy of ``chromosome`` with per-waypoint probability.

    Three mutation kinds are supported:

    * move the waypoint (local jitter, or occasionally a new clear ray);
    * change its speed mode;
    * change its sensor mode.

    Speed/sensor modes always change to a *different* valid mode.  The child is
    repaired before being returned; the parent object is never modified.
    """

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1]")

    child = chromosome.copy()
    if mutation_rate == 0.0:
        return child
    if not child.deployments:
        child.deployments = [drone.start for drone in drones]

    for i in range(len(child.deployments)):
        if rng.random() >= deployment_rate:
            continue
        moved = _nearby_free_cell(env, child.deployments[i], rng)
        if moved is not None:
            child.deployments[i] = moved

    fleet = fleet_for_chromosome(drones, child)
    for drone, route in zip(fleet, child.routes):
        previous = drone.start
        for i, wp in enumerate(route.waypoints):
            if rng.random() >= mutation_rate:
                previous = (wp.x, wp.y)
                continue

            draw = float(rng.random())
            if draw < MOVE_PROBABILITY:
                if rng.random() < 0.15:
                    cell = sample_area_waypoint(
                        env, previous, rng, env.targets
                    )
                else:
                    dx = int(rng.integers(-max_step, max_step + 1))
                    dy = int(rng.integers(-max_step, max_step + 1))
                    if dx == 0 and dy == 0:
                        dx = 1
                    x = min(max(wp.x + dx, 0), env.width - 1)
                    y = min(max(wp.y + dy, 0), env.height - 1)
                    cell = (x, y)
                if cell is not None:
                    wp = Waypoint(cell[0], cell[1], wp.speed, wp.sensor)
            elif draw < MOVE_PROBABILITY + SPEED_PROBABILITY:
                new_speed = int((wp.speed + int(rng.integers(1, 3))) % 3)
                wp = Waypoint(wp.x, wp.y, new_speed, wp.sensor)
            else:
                new_sensor = int((wp.sensor + int(rng.integers(1, 3))) % 3)
                wp = Waypoint(wp.x, wp.y, wp.speed, new_sensor)

            route.waypoints[i] = wp
            previous = (wp.x, wp.y)

    repaired = repair_chromosome(env, child, drones, rng)
    if repaired is None:
        # Documented fallback: keep the unmutated parent rather than emit an
        # invalid candidate.
        return chromosome.copy()
    return repaired


def _nearby_free_cell(
    env: GridEnvironment,
    cell: tuple[int, int],
    rng: np.random.Generator,
    radius: int = 10,
) -> tuple[int, int] | None:
    """Sample a free deployment cell near ``cell``."""

    x, y = cell
    for _ in range(24):
        nx = int(np.clip(x + rng.integers(-radius, radius + 1), 0, env.width - 1))
        ny = int(np.clip(y + rng.integers(-radius, radius + 1), 0, env.height - 1))
        if env.is_free(nx, ny):
            return nx, ny
    ys, xs = np.nonzero(env.free_mask)
    if len(xs) == 0:
        return None
    index = int(rng.integers(0, len(xs)))
    return int(xs[index]), int(ys[index])
