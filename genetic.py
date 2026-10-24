"""Genetic algorithm: initialisation, selection, operators and optimizer."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from config import FitnessWeights, GAConfig
from model import Chromosome, Drone, DroneRoute, GridEnvironment, Waypoint, fleet_for_chromosome
from simulation import (
    EvaluationResult,
    MissionEvaluator,
    random_valid_route,
    repair_chromosome,
    sample_area_waypoint,
)


# ======================================================================
# Population initialisation
# ======================================================================

"""Population initialisation."""


def sample_deployments(
    env: GridEnvironment,
    n_drones: int,
    rng: np.random.Generator,
    min_distance: int = 5,
) -> list[tuple[int, int]] | None:
    """Choose spread-out free cells for the fleet deployment segment."""

    ys, xs = np.nonzero(env.free_mask)
    if len(xs) < n_drones:
        return None
    cells = list(zip(xs.tolist(), ys.tolist()))
    for spacing in (min_distance, max(1, min_distance // 2), 1):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(
                max(abs(cell[0] - c[0]), abs(cell[1] - c[1])) >= spacing
                for c in chosen
            ):
                chosen.append(cell)
                if len(chosen) == n_drones:
                    return chosen
    return None


def random_chromosome(
    env: GridEnvironment,
    drones: list[Drone],
    n_waypoints: int,
    rng: np.random.Generator,
) -> Chromosome | None:
    """Build one random, straight-leg-feasible chromosome."""

    starts = sample_deployments(env, len(drones), rng)
    if starts is None:
        return None
    routes: list[DroneRoute] = []
    for start in starts:
        route = random_valid_route(
            env, start, n_waypoints, rng, env.targets
        )
        if route is None:
            return None
        routes.append(route)
    return Chromosome(routes=routes, deployments=starts)


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

# ======================================================================
# Tournament selection
# ======================================================================

"""Tournament selection.

A tournament of size ``k`` samples ``k`` distinct candidates uniformly and
returns the one with the best (feasible, fitness) key.  Larger ``k`` increases
selection pressure: the probability that the single best candidate in the whole
population wins a tournament is ``k / N``.  A value of ``k = 2`` is weak
pressure; ``k = 5`` is noticeably greedy.
"""


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

# ======================================================================
# Crossover
# ======================================================================

"""Segment-swapping crossover for fleet chromosomes.

A chromosome is a sequence of three drone blocks.  Each block contains one
deployment cell and one route, so swapping a suffix of blocks never separates a
route from the deployment that was evolved with it.  This is a spatial,
multi-segment one-point crossover: good fleet combinations survive intact,
while different deployment/route combinations are explored.
"""


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

# ======================================================================
# Mutation
# ======================================================================

"""Spatially meaningful mutation and constraint repair hook."""


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

# ======================================================================
# GA optimizer
# ======================================================================

"""Reusable genetic-algorithm optimizer for fleet strategy search."""


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
    generation_best_chromosomes: list[Chromosome] = field(default_factory=list)
    generation_best_evaluations: list[EvaluationResult] = field(default_factory=list)


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
        if not 0.0 <= config.immigrant_rate <= 1.0:
            raise ValueError("immigrant_rate must be in [0, 1]")
        if config.immigrant_patience < 0:
            raise ValueError("immigrant_patience must be >= 0")

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
        generation_best_chromosomes = [initial_best_chromosome.copy()]
        generation_best_evaluations = [initial_best_evaluation]
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
            if (
                self.config.immigrant_rate > 0.0
                and self.config.immigrant_patience > 0
                and stagnation >= self.config.immigrant_patience
            ):
                n_immigrants = max(
                    1,
                    int(round(
                        self.config.immigrant_rate * len(population)
                    )),
                )
                worst = sorted(
                    range(len(population)),
                    key=lambda i: (
                        evaluations[i].feasible,
                        evaluations[i].fitness,
                    ),
                )[:n_immigrants]
                injected = 0
                for index in worst:
                    candidate = random_chromosome(
                        self.env,
                        self.drones,
                        n_waypoints,
                        rng,
                    )
                    if candidate is not None:
                        population[index] = candidate
                        injected += 1
                if injected:
                    self.logger.info(
                        "generation %d: injected %d random immigrants",
                        generation,
                        injected,
                    )
                    stagnation = 0
            evaluations = [self._evaluate(c) for c in population]
            history.append(self._stats(generation, population, evaluations))
            gen_best_chromosome, gen_best_evaluation = self._best_pair(
                population, evaluations
            )
            generation_best_chromosomes.append(gen_best_chromosome.copy())
            generation_best_evaluations.append(gen_best_evaluation)
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
            generation_best_chromosomes=generation_best_chromosomes,
            generation_best_evaluations=generation_best_evaluations,
        )
