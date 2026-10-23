"""Spatially meaningful mutation and constraint repair hook."""

from __future__ import annotations

import numpy as np

from environment.grid import GridEnvironment
from simulation.repair import (
    repair_chromosome,
    sample_area_waypoint,
)
from uav.chromosome import Chromosome, Waypoint
from uav.drone import Drone

MOVE_PROBABILITY = 0.60
SPEED_PROBABILITY = 0.20


def mutate(
    chromosome: Chromosome,
    env: GridEnvironment,
    drones: list[Drone],
    rng: np.random.Generator,
    mutation_rate: float,
    max_step: int = 4,
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

    for drone, route in zip(drones, child.routes):
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
