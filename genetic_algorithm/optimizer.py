"""Reusable genetic-algorithm optimizer for fleet strategy search."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass

import numpy as np

from config import FitnessWeights, GAConfig
from environment.grid import GridEnvironment
from genetic_algorithm.crossover import one_point_crossover
from genetic_algorithm.mutation import mutate
from genetic_algorithm.population import initialize_population, random_chromosome
from genetic_algorithm.selection import tournament_select
from simulation.evaluator import EvaluationResult, MissionEvaluator
from simulation.repair import repair_chromosome
from uav.chromosome import Chromosome
from uav.drone import Drone

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    generation: int
    feasible_count: int
    best_feasible_fitness: float | None
    average_feasible_fitness: float | None
    best_coverage: float | None
    best_detection_rate: float | None
    best_total_energy: float | None
    best_normalized_energy: float | None
    best_mission_time: float | None
    best_hazard_exposure: float | None
    best_normalized_risk: float | None
    best_penalized_fitness: float

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "feasible_count": self.feasible_count,
            "best_feasible_fitness": self.best_feasible_fitness,
            "average_feasible_fitness": self.average_feasible_fitness,
            "best_coverage": self.best_coverage,
            "best_detection_rate": self.best_detection_rate,
            "best_total_energy": self.best_total_energy,
            "best_normalized_energy": self.best_normalized_energy,
            "best_mission_time": self.best_mission_time,
            "best_hazard_exposure": self.best_hazard_exposure,
            "best_normalized_risk": self.best_normalized_risk,
            "best_penalized_fitness": self.best_penalized_fitness,
        }


@dataclass
class OptimizationResult:
    best_chromosome: Chromosome
    best_evaluation: EvaluationResult
    initial_best_chromosome: Chromosome
    initial_best_evaluation: EvaluationResult
    history: list[GenerationStats]
    termination_reason: str
    generations_run: int
    evaluation_requests: int
    unique_evaluations: int
    cache_hits: int
    runtime_seconds: float
    population_size: int
    feasible_candidates_final: int
    feasible_evaluations: int
    feasibility_rate: float


class GeneticAlgorithm:
    """Tournament-selection GA with elitism, spatial operators, and repair."""

    def __init__(
        self,
        env: GridEnvironment,
        drones: list[Drone],
        config: GAConfig,
        weights: FitnessWeights,
        n_waypoints: int = 6,
        time_reference: float = 120.0,
        progress=None,
    ) -> None:
        self.env = env
        self.drones = list(drones)
        self.config = config
        self.weights = weights
        self.n_waypoints = int(n_waypoints)
        self.time_reference = time_reference
        self.progress = progress
        self.logger = LOGGER

        if self.n_waypoints < 1:
            raise ValueError("n_waypoints must be >= 1")
        if config.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if config.generations < 1:
            raise ValueError("generations must be >= 1")
        if config.tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if config.elitism < 0:
            raise ValueError("elitism must be >= 0")
        if config.elitism > config.population_size:
            raise ValueError("elitism cannot exceed population_size")
        if not 0.0 <= config.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= config.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")

        self.evaluator = MissionEvaluator(
            env, self.drones, weights, time_reference=time_reference
        )
        self._cache: dict[tuple, EvaluationResult] = {}
        self._requests = 0
        self._unique = 0
        self._cache_hits = 0
        self._feasible_unique = 0

    # ------------------------------------------------------------------
    # Evaluation and statistics
    # ------------------------------------------------------------------
    def _evaluate(self, chromosome: Chromosome) -> EvaluationResult:
        self._requests += 1
        signature = chromosome.signature()
        if self.config.cache_evaluations and signature in self._cache:
            self._cache_hits += 1
            return self._cache[signature]
        result = self.evaluator.evaluate(chromosome)
        if self.config.cache_evaluations:
            self._cache[signature] = result
        self._unique += 1
        if result.feasible:
            self._feasible_unique += 1
        return result

    @staticmethod
    def _best_pair(
        population: list[Chromosome],
        evaluations: list[EvaluationResult],
    ) -> tuple[Chromosome, EvaluationResult]:
        index = max(
            range(len(population)),
            key=lambda i: (
                evaluations[i].feasible,
                evaluations[i].fitness,
            ),
        )
        return population[index], evaluations[index]

    def _stats(
        self,
        generation: int,
        population: list[Chromosome],
        evaluations: list[EvaluationResult],
    ) -> GenerationStats:
        feasible = [
            (i, ev) for i, ev in enumerate(evaluations) if ev.feasible
        ]
        best_penalized = max(ev.fitness for ev in evaluations)
        if feasible:
            _, best = max(feasible, key=lambda pair: pair[1].fitness)
            avg = statistics.fmean(ev.fitness for _, ev in feasible)
            return GenerationStats(
                generation=generation,
                feasible_count=len(feasible),
                best_feasible_fitness=best.fitness,
                average_feasible_fitness=avg,
                best_coverage=best.coverage,
                best_detection_rate=best.detection_rate,
                best_total_energy=best.total_energy,
                best_normalized_energy=best.normalized_energy,
                best_mission_time=best.mission_time,
                best_hazard_exposure=best.hazard_exposure,
                best_normalized_risk=best.normalized_risk,
                best_penalized_fitness=best_penalized,
            )
        return GenerationStats(
            generation=generation,
            feasible_count=0,
            best_feasible_fitness=None,
            average_feasible_fitness=None,
            best_coverage=None,
            best_detection_rate=None,
            best_total_energy=None,
            best_normalized_energy=None,
            best_mission_time=None,
            best_hazard_exposure=None,
            best_normalized_risk=None,
            best_penalized_fitness=best_penalized,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> OptimizationResult:
        start = time.perf_counter()
        rng = np.random.default_rng(self.config.seed)

        population = initialize_population(
            self.config.population_size,
            self.env,
            self.drones,
            self.n_waypoints,
            rng,
        )
        n_waypoints = self.n_waypoints
        evaluations = [self._evaluate(c) for c in population]
        history = [self._stats(0, population, evaluations)]
        initial_best_chromosome, initial_best_evaluation = self._best_pair(
            population, evaluations
        )
        if self.progress is not None:
            self.progress(0, history[-1])

        best_so_far = history[0].best_feasible_fitness
        stagnation = 0
        termination_reason = "generation_limit"

        for generation in range(1, self.config.generations):
            next_population: list[Chromosome] = []

            feasible_indices = [
                i for i, ev in enumerate(evaluations) if ev.feasible
            ]
            if self.config.elitism > 0 and feasible_indices:
                ranked = sorted(
                    feasible_indices,
                    key=lambda i: evaluations[i].fitness,
                    reverse=True,
                )
                for index in ranked[: self.config.elitism]:
                    next_population.append(population[index].copy())
            elif self.config.elitism > 0:
                self.logger.info(
                    "generation %d: no feasible candidates, elitism skipped",
                    generation,
                )

            while len(next_population) < self.config.population_size:
                parent_a = tournament_select(
                    population,
                    evaluations,
                    rng,
                    self.config.tournament_size,
                )
                parent_b = tournament_select(
                    population,
                    evaluations,
                    rng,
                    self.config.tournament_size,
                )
                child_a, child_b = one_point_crossover(
                    parent_a,
                    parent_b,
                    rng,
                    self.config.crossover_rate,
                )
                for child in (child_a, child_b):
                    if len(next_population) >= self.config.population_size:
                        break
                    mutated = mutate(
                        child,
                        self.env,
                        self.drones,
                        rng,
                        self.config.mutation_rate,
                    )
                    repaired = repair_chromosome(
                        self.env, mutated, self.drones, rng
                    )
                    if repaired is None:
                        repaired = random_chromosome(
                            self.env,
                            self.drones,
                            n_waypoints,
                            rng,
                        )
                    if repaired is None:
                        repaired = parent_a.copy()
                    next_population.append(repaired)

            population = next_population
            evaluations = [self._evaluate(c) for c in population]
            history.append(self._stats(generation, population, evaluations))
            if self.progress is not None:
                self.progress(generation, history[-1])

            current_best = history[-1].best_feasible_fitness
            if current_best is not None and (
                best_so_far is None or current_best > best_so_far + 1e-9
            ):
                best_so_far = current_best
                stagnation = 0
            else:
                stagnation += 1

            if (
                self.config.stagnation_limit is not None
                and stagnation >= self.config.stagnation_limit
            ):
                termination_reason = "stagnation"
                break

        best_chromosome, best_evaluation = self._best_pair(
            population, evaluations
        )
        if not best_evaluation.feasible:
            termination_reason = "no_feasible_solution"

        return OptimizationResult(
            best_chromosome=best_chromosome,
            best_evaluation=best_evaluation,
            initial_best_chromosome=initial_best_chromosome,
            initial_best_evaluation=initial_best_evaluation,
            history=history,
            termination_reason=termination_reason,
            generations_run=history[-1].generation + 1,
            evaluation_requests=self._requests,
            unique_evaluations=self._unique,
            cache_hits=self._cache_hits,
            runtime_seconds=time.perf_counter() - start,
            population_size=self.config.population_size,
            feasible_candidates_final=history[-1].feasible_count,
            feasible_evaluations=self._feasible_unique,
            feasibility_rate=(
                self._feasible_unique / self._unique
                if self._unique
                else 0.0
            ),
        )
