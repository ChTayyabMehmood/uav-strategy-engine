"""Spatial crossover for fleet chromosomes.

The operator exchanges a suffix of the waypoint sequence for the *same drone*
in two parents.  This is a one-point crossover on compatible route segments:
the prefix comes from one parent and the suffix from the other, so good partial
routes can be recombined without mixing drones that have different start cells.

Waypoints are immutable, so children can reference the same ``Waypoint`` objects
without risk of aliasing.  The route lists themselves are always new.  The
result may contain a discontinuity at the cut; the GA calls the repair operator
afterwards.
"""

from __future__ import annotations

import numpy as np

from uav.chromosome import Chromosome, DroneRoute


def one_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: np.random.Generator,
    crossover_rate: float = 0.9,
) -> tuple[Chromosome, Chromosome]:
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1]")
    if rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()
    if len(parent_a.routes) != len(parent_b.routes):
        return parent_a.copy(), parent_b.copy()

    routes_a: list[DroneRoute] = []
    routes_b: list[DroneRoute] = []
    for route_a, route_b in zip(parent_a.routes, parent_b.routes):
        n = len(route_a.waypoints)
        if n != len(route_b.waypoints) or n < 2:
            routes_a.append(route_a.copy())
            routes_b.append(route_b.copy())
            continue
        if rng.random() < 0.5:
            cut = int(rng.integers(1, n))
            routes_a.append(
                DroneRoute(
                    waypoints=list(route_a.waypoints[:cut])
                    + list(route_b.waypoints[cut:])
                )
            )
            routes_b.append(
                DroneRoute(
                    waypoints=list(route_b.waypoints[:cut])
                    + list(route_a.waypoints[cut:])
                )
            )
        else:
            routes_a.append(route_a.copy())
            routes_b.append(route_b.copy())

    return Chromosome(routes=routes_a), Chromosome(routes=routes_b)
