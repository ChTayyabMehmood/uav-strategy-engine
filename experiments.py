"""Random-search baseline and hyperparameter sensitivity experiments."""

from __future__ import annotations

import csv
import itertools
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from config import FitnessWeights, GAConfig
from genetic import GeneticAlgorithm, random_chromosome
from model import Chromosome, Drone, GridEnvironment
from simulation import EvaluationResult, MissionEvaluator, repair_chromosome


# ======================================================================
# Random-search baseline
# ======================================================================

"""Random-search baseline using the same environment, repair, and fitness."""


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

# ======================================================================
# Sensitivity experiment
# ======================================================================

"""Hyperparameter sensitivity experiments.

Two modes are supported:

* one-factor-at-a-time (default): vary population size, mutation rate, and
  tournament size independently while everything else stays at the base
  configuration;
* full factorial (``full_factorial=True``): every combination of the three
  parameter value lists.  With the defaults that is 27 runs per seed.

All runs share the same environment instance and fitness weights.
"""


DEFAULT_VALUES: dict[str, list[float]] = {
    "population_size": [50, 100, 200],
    "mutation_rate": [0.10, 0.20, 0.30],
    "tournament_size": [2, 3, 5],
}

DETAIL_FIELDS = [
    "parameter",
    "value",
    "combo",
    "seed",
    "population_size",
    "generations",
    "mutation_rate",
    "tournament_size",
    "best_fitness",
    "coverage",
    "detection_rate",
    "total_energy",
    "mission_time",
    "hazard_exposure",
    "runtime_seconds",
    "evaluations",
    "feasible",
    "termination_reason",
]

SUMMARY_FIELDS = [
    "parameter",
    "value",
    "combo",
    "n_runs",
    "mean_best_fitness",
    "std_best_fitness",
    "mean_coverage",
    "mean_detection_rate",
    "mean_total_energy",
    "mean_mission_time",
    "mean_hazard_exposure",
    "mean_runtime_seconds",
    "mean_evaluations",
    "feasible_rate",
]


def _run_specs(
    values: dict[str, list[float]],
    full_factorial: bool,
) -> list[dict[str, float]]:
    if full_factorial:
        names = list(values.keys())
        specs = []
        for combination in itertools.product(*(values[name] for name in names)):
            setting = dict(zip(names, combination))
            specs.append(setting)
        return specs
    specs = []
    for name, candidates in values.items():
        for candidate in candidates:
            specs.append({name: candidate})
    return specs


def run_sensitivity(
    env: GridEnvironment,
    drones: list[Drone],
    weights: FitnessWeights,
    base_config: GAConfig,
    n_waypoints: int,
    seeds: list[int],
    output_dir: str | Path,
    generations: int | None = None,
    values: dict[str, list[float]] | None = None,
    full_factorial: bool = False,
    time_reference: float = 120.0,
    progress=None,
) -> tuple[list[dict], list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = values or DEFAULT_VALUES.copy()
    if not seeds:
        raise ValueError("at least one seed is required")
    base = (
        replace(base_config, generations=generations)
        if generations is not None
        else base_config
    )

    specs = _run_specs(values, full_factorial)
    total_runs = len(specs) * len(seeds)
    rows: list[dict] = []
    completed = 0

    for spec in specs:
        if full_factorial:
            combo = ",".join(f"{k}={v}" for k, v in spec.items())
            parameter = "full_factorial"
            value = combo
        else:
            parameter, value = next(iter(spec.items()))
            combo = f"{parameter}={value}"

        for seed in seeds:
            config = replace(base, seed=seed, **spec)
            result = GeneticAlgorithm(
                env,
                drones,
                config,
                weights,
                n_waypoints=n_waypoints,
                time_reference=time_reference,
            ).run()
            evaluation = result.best_evaluation
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "combo": combo,
                    "seed": seed,
                    "population_size": config.population_size,
                    "generations": config.generations,
                    "mutation_rate": config.mutation_rate,
                    "tournament_size": config.tournament_size,
                    "best_fitness": evaluation.fitness,
                    "coverage": evaluation.coverage,
                    "detection_rate": evaluation.detection_rate,
                    "total_energy": evaluation.total_energy,
                    "mission_time": evaluation.mission_time,
                    "hazard_exposure": evaluation.hazard_exposure,
                    "runtime_seconds": result.runtime_seconds,
                    "evaluations": result.evaluation_requests,
                    "feasible": evaluation.feasible,
                    "termination_reason": result.termination_reason,
                }
            )
            completed += 1
            if progress is not None:
                progress(completed, total_runs, combo, seed, evaluation.fitness)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["combo"], []).append(row)
    summary: list[dict] = []
    for combo, group in grouped.items():
        first = group[0]
        fitness_values = [row["best_fitness"] for row in group]
        summary.append(
            {
                "parameter": first["parameter"],
                "value": first["value"],
                "combo": combo,
                "n_runs": len(group),
                "mean_best_fitness": statistics.fmean(fitness_values),
                "std_best_fitness": (
                    statistics.stdev(fitness_values)
                    if len(fitness_values) > 1
                    else 0.0
                ),
                "mean_coverage": statistics.fmean(
                    row["coverage"] for row in group
                ),
                "mean_detection_rate": statistics.fmean(
                    row["detection_rate"] for row in group
                ),
                "mean_total_energy": statistics.fmean(
                    row["total_energy"] for row in group
                ),
                "mean_mission_time": statistics.fmean(
                    row["mission_time"] for row in group
                ),
                "mean_hazard_exposure": statistics.fmean(
                    row["hazard_exposure"] for row in group
                ),
                "mean_runtime_seconds": statistics.fmean(
                    row["runtime_seconds"] for row in group
                ),
                "mean_evaluations": statistics.fmean(
                    row["evaluations"] for row in group
                ),
                "feasible_rate": statistics.fmean(
                    1.0 if row["feasible"] else 0.0 for row in group
                ),
            }
        )

    _write_csv(output_dir / "sensitivity_results.csv", DETAIL_FIELDS, rows)
    _write_csv(
        output_dir / "sensitivity_summary.csv", SUMMARY_FIELDS, summary
    )
    return rows, summary


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        round(value, 6)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in row.items()
                }
            )
