"""Population initialisation."""

from __future__ import annotations

import numpy as np

from environment.grid import GridEnvironment
from simulation.repair import random_valid_route
from uav.chromosome import Chromosome, DroneRoute
from uav.drone import Drone


def random_chromosome(
    env: GridEnvironment,
    drones: list[Drone],
    n_waypoints: int,
    rng: np.random.Generator,
) -> Chromosome | None:
    """Build one random, straight-leg-feasible chromosome."""

    routes: list[DroneRoute] = []
    for drone in drones:
        route = random_valid_route(
            env, drone.start, n_waypoints, rng, env.targets
        )
        if route is None:
            return None
        routes.append(route)
    return Chromosome(routes=routes)


def initialize_population(
    size: int,
    env: GridEnvironment,
    drones: list[Drone],
    n_waypoints: int,
    rng: np.random.Generator,
    max_attempts: int = 60,
) -> list[Chromosome]:
    """Create a diverse initial population of valid chromosomes.

    Diversity comes from random ray directions and random speed/sensor modes.
    If a chromosome cannot be built after ``max_attempts``, a RuntimeError is
    raised - this indicates an over-constrained map rather than silently
    producing invalid individuals.
    """

    if size < 1:
        raise ValueError("population size must be >= 1")
    population: list[Chromosome] = []
    for _ in range(size):
        for _attempt in range(max_attempts):
            chromosome = random_chromosome(
                env, drones, n_waypoints, rng
            )
            if chromosome is not None:
                population.append(chromosome)
                break
        else:
            raise RuntimeError(
                "could not initialise a valid chromosome; the environment "
                "may be too constrained"
            )
    return population
