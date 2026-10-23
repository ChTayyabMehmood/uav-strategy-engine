"""Random-search baseline using the same environment, repair, and fitness."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config import FitnessWeights
from environment.grid import GridEnvironment
from genetic_algorithm.population import random_chromosome
from simulation.evaluator import EvaluationResult, MissionEvaluator
from simulation.repair import repair_chromosome
from uav.chromosome import Chromosome
from uav.drone import Drone


@dataclass
class RandomSearchResult:
    best_chromosome: Chromosome
    best_evaluation: EvaluationResult
    evaluations: int
    feasible_count: int
    feasibility_rate: float
    runtime_seconds: float
    seed: int
    history: list[tuple[int, float | None]]


def run_random_search(
    env: GridEnvironment,
    drones: list[Drone],
    weights: FitnessWeights,
    n_waypoints: int,
    budget: int,
    seed: int,
    time_reference: float = 120.0,
    progress=None,
) -> RandomSearchResult:
    """Evaluate ``budget`` random repaired candidates and keep the best.

    Candidates use exactly the same generator and feasibility rules as the GA's
    initial population.  Every candidate is spatially repaired when possible;
    energy/range violations are still scored as infeasible by the evaluator.
    """

    if budget < 1:
        raise ValueError("budget must be >= 1")

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = MissionEvaluator(
        env, drones, weights, time_reference=time_reference
    )

    best_chromosome: Chromosome | None = None
    best_evaluation: EvaluationResult | None = None
    feasible_count = 0
    history: list[tuple[int, float | None]] = []
    best_feasible_fitness: float | None = None

    report_every = max(1, budget // 10)
    for evaluation_index in range(1, budget + 1):
        chromosome = None
        for _attempt in range(40):
            candidate = random_chromosome(
                env, drones, n_waypoints, rng
            )
            if candidate is None:
                continue
            repaired = repair_chromosome(env, candidate, drones, rng)
            if repaired is not None:
                chromosome = repaired
                break
        if chromosome is None:
            raise RuntimeError(
                "random search could not generate a valid candidate"
            )

        result = evaluator.evaluate(chromosome)
        if result.feasible:
            feasible_count += 1
            if (
                best_feasible_fitness is None
                or result.fitness > best_feasible_fitness
            ):
                best_feasible_fitness = result.fitness
        if best_evaluation is None or (
            result.feasible,
            result.fitness,
        ) > (
            best_evaluation.feasible,
            best_evaluation.fitness,
        ):
            best_chromosome = chromosome.copy()
            best_evaluation = result

        history.append((evaluation_index, best_feasible_fitness))
        if progress is not None and evaluation_index % report_every == 0:
            progress(evaluation_index, best_feasible_fitness)

    if best_chromosome is None or best_evaluation is None:
        raise RuntimeError("random search produced no candidates")

    return RandomSearchResult(
        best_chromosome=best_chromosome,
        best_evaluation=best_evaluation,
        evaluations=budget,
        feasible_count=feasible_count,
        feasibility_rate=feasible_count / budget,
        runtime_seconds=time.perf_counter() - start,
        seed=seed,
        history=history,
    )
