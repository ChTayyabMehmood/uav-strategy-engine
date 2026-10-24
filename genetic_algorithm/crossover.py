"""Segment-swapping crossover for fleet chromosomes.

A chromosome is a sequence of three drone blocks.  Each block contains one
deployment cell and one route, so swapping a suffix of blocks never separates a
route from the deployment that was evolved with it.  This is a spatial,
multi-segment one-point crossover: good fleet combinations survive intact,
while different deployment/route combinations are explored.
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
    """Swap a suffix of drone blocks between two parents."""

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1]")
    if rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()
    if len(parent_a.routes) != len(parent_b.routes):
        return parent_a.copy(), parent_b.copy()
    if bool(parent_a.deployments) != bool(parent_b.deployments):
        return parent_a.copy(), parent_b.copy()

    n_blocks = len(parent_a.routes)
    if n_blocks < 2:
        return parent_a.copy(), parent_b.copy()
    cut = int(rng.integers(1, n_blocks))

    routes_a = [
        route.copy()
        for route in parent_a.routes[:cut] + parent_b.routes[cut:]
    ]
    routes_b = [
        route.copy()
        for route in parent_b.routes[:cut] + parent_a.routes[cut:]
    ]
    deployments_a: list[tuple[int, int]] = []
    deployments_b: list[tuple[int, int]] = []
    if parent_a.deployments:
        deployments_a = (
            list(parent_a.deployments[:cut])
            + list(parent_b.deployments[cut:])
        )
        deployments_b = (
            list(parent_b.deployments[:cut])
            + list(parent_a.deployments[cut:])
        )

    return (
        Chromosome(routes=routes_a, deployments=deployments_a),
        Chromosome(routes=routes_b, deployments=deployments_b),
    )
