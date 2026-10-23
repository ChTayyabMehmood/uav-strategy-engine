"""End-to-end tests for the genetic-algorithm optimizer."""

from __future__ import annotations

import pytest

from config import EnvironmentConfig, FitnessWeights, FleetConfig, GAConfig
from environment import generate_environment
from genetic_algorithm import GeneticAlgorithm
from uav import Drone, build_fleet


def build_scenario(
    seed: int = 21,
    n_drones: int = 2,
    waypoints: int = 3,
) -> tuple:
    env = generate_environment(
        EnvironmentConfig(
            seed=seed,
            width=18,
            height=18,
            n_targets=6,
            n_obstacle_blobs=4,
            n_nofly_blobs=2,
            n_hazard_blobs=3,
        ),
        n_drones,
    )
    fleet = FleetConfig(n_drones=n_drones, waypoints_per_route=waypoints)
    drones = build_fleet(env.starts, fleet)
    return env, drones, fleet


def test_small_run_completes_and_records_history() -> None:
    env, drones, fleet = build_scenario()
    config = GAConfig(
        population_size=8,
        generations=4,
        mutation_rate=0.2,
        tournament_size=3,
        elitism=1,
        seed=5,
    )
    result = GeneticAlgorithm(
        env, drones, config, FitnessWeights(), n_waypoints=fleet.waypoints_per_route
    ).run()
    assert len(result.history) == 4
    assert result.generations_run == 4
    assert result.evaluation_requests == 8 * 4
    assert result.unique_evaluations <= result.evaluation_requests
    assert result.best_evaluation.fitness == pytest.approx(
        max(
            stats.best_feasible_fitness
            for stats in result.history
            if stats.best_feasible_fitness is not None
        )
    )
    assert result.termination_reason == "generation_limit"


def test_elitism_keeps_best_feasible_non_decreasing() -> None:
    env, drones, fleet = build_scenario(seed=22)
    config = GAConfig(
        population_size=10,
        generations=6,
        mutation_rate=0.05,
        tournament_size=2,
        elitism=2,
        seed=6,
    )
    result = GeneticAlgorithm(
        env, drones, config, FitnessWeights(), n_waypoints=fleet.waypoints_per_route
    ).run()
    feasible_best = [
        stats.best_feasible_fitness
        for stats in result.history
        if stats.best_feasible_fitness is not None
    ]
    assert feasible_best
    assert all(
        b >= a - 1e-9 for a, b in zip(feasible_best, feasible_best[1:])
    )


def test_same_seed_is_reproducible() -> None:
    env, drones, fleet = build_scenario(seed=23)
    config = GAConfig(
        population_size=8,
        generations=3,
        mutation_rate=0.1,
        seed=7,
    )
    kwargs = dict(
        env=env,
        drones=drones,
        config=config,
        weights=FitnessWeights(),
        n_waypoints=fleet.waypoints_per_route,
    )
    first = GeneticAlgorithm(**kwargs).run()
    second = GeneticAlgorithm(**kwargs).run()
    assert first.best_chromosome.signature() == second.best_chromosome.signature()
    assert first.best_evaluation.fitness == pytest.approx(
        second.best_evaluation.fitness
    )


def test_no_feasible_solution_is_handled_explicitly() -> None:
    env = generate_environment(
        EnvironmentConfig(
            seed=24,
            width=15,
            height=15,
            n_targets=4,
            n_obstacle_blobs=3,
            n_nofly_blobs=1,
            n_hazard_blobs=2,
        ),
        1,
    )
    drones = [
        Drone(
            drone_id=0,
            start=env.starts[0],
            max_flight_range=0.001,
            battery_capacity=0.001,
        )
    ]
    config = GAConfig(
        population_size=6,
        generations=2,
        mutation_rate=0.2,
        seed=8,
    )
    result = GeneticAlgorithm(
        env, drones, config, FitnessWeights(), n_waypoints=2
    ).run()
    assert result.termination_reason == "no_feasible_solution"
    assert not result.best_evaluation.feasible
    assert all(stats.feasible_count == 0 for stats in result.history)


def test_generation_count_one() -> None:
    env, drones, fleet = build_scenario(seed=25)
    result = GeneticAlgorithm(
        env,
        drones,
        GAConfig(population_size=6, generations=1, seed=9),
        FitnessWeights(),
        n_waypoints=fleet.waypoints_per_route,
    ).run()
    assert len(result.history) == 1
    assert result.generations_run == 1
    assert result.termination_reason == "generation_limit"
