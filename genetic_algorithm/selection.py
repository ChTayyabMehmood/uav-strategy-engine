"""Tournament selection.

A tournament of size ``k`` samples ``k`` distinct candidates uniformly and
returns the one with the best (feasible, fitness) key.  Larger ``k`` increases
selection pressure: the probability that the single best candidate in the whole
population wins a tournament is ``k / N``.  A value of ``k = 2`` is weak
pressure; ``k = 5`` is noticeably greedy.
"""

from __future__ import annotations

import numpy as np

from simulation.evaluator import EvaluationResult
from uav.chromosome import Chromosome


def tournament_select(
    population: list[Chromosome],
    evaluations: list[EvaluationResult],
    rng: np.random.Generator,
    tournament_size: int,
) -> Chromosome:
    if len(population) != len(evaluations):
        raise ValueError("population and evaluations must be aligned")
    if len(population) == 0:
        raise ValueError("cannot select from an empty population")
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")

    k = min(tournament_size, len(population))
    indices = rng.choice(len(population), size=k, replace=False)
    best = max(
        indices,
        key=lambda i: (
            evaluations[i].feasible,
            evaluations[i].fitness,
        ),
    )
    return population[int(best)]
