"""Tests for selection, crossover, mutation, and repair integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config import EnvironmentConfig, FleetConfig
from environment import generate_environment
from genetic_algorithm.crossover import one_point_crossover
from genetic_algorithm.mutation import mutate
from genetic_algorithm.selection import tournament_select
from simulation.repair import repair_chromosome
from uav import build_fleet


@pytest.fixture
def scenario():
    env = generate_environment(EnvironmentConfig(seed=11, width=20, height=20), 2)
    fleet = FleetConfig(n_drones=2, waypoints_per_route=4)
    drones = build_fleet(env.starts, fleet)
    return env, drones, fleet


def make_parents(scenario, seed: int):
    from genetic_algorithm.population import initialize_population

    env, drones, fleet = scenario
    rng = np.random.default_rng(seed)
    return initialize_population(
        2, env, drones, fleet.waypoints_per_route, rng
    )


def test_crossover_preserves_structure_and_independence(scenario) -> None:
    env, drones, fleet = scenario
    parent_a, parent_b = make_parents(scenario, 1)
    rng = np.random.default_rng(3)
    child_a, child_b = one_point_crossover(parent_a, parent_b, rng, 1.0)
    for child in (child_a, child_b):
        child.validate_structure(2, fleet.waypoints_per_route)
        assert len(child.routes) == 2
        assert len(child.routes[0].waypoints) == fleet.waypoints_per_route
    child_a.routes[0].waypoints.append(
        child_a.routes[0].waypoints[0]
    )
    assert len(parent_a.routes[0].waypoints) == fleet.waypoints_per_route
    assert child_a is not parent_a
    repaired = repair_chromosome(env, child_b, drones, rng)
    assert repaired is not None
    repaired.validate_structure(2, fleet.waypoints_per_route)


def test_crossover_rate_zero_returns_independent_copies(scenario) -> None:
    parent_a, parent_b = make_parents(scenario, 2)
    rng = np.random.default_rng(4)
    child_a, child_b = one_point_crossover(parent_a, parent_b, rng, 0.0)
    assert child_a.signature() == parent_a.signature()
    assert child_b.signature() == parent_b.signature()
    assert child_a is not parent_a


def test_mutation_rate_zero_changes_nothing(scenario) -> None:
    env, drones, _fleet = scenario
    parent, _ = make_parents(scenario, 5)
    rng = np.random.default_rng(6)
    child = mutate(parent, env, drones, rng, mutation_rate=0.0)
    assert child.signature() == parent.signature()
    assert child is not parent


def test_mutation_does_not_modify_parent_and_repairs(scenario) -> None:
    env, drones, _fleet = scenario
    parent, _ = make_parents(scenario, 7)
    before = parent.signature()
    rng = np.random.default_rng(8)
    child = mutate(parent, env, drones, rng, mutation_rate=1.0)
    assert parent.signature() == before
    assert child.signature() != before
    repaired = repair_chromosome(env, child, drones, rng)
    assert repaired is not None


def test_tournament_prefers_best_when_all_are_sampled() -> None:
    population = ["low", "high"]
    evaluations = [
        SimpleNamespace(feasible=True, fitness=1.0),
        SimpleNamespace(feasible=True, fitness=9.0),
    ]
    rng = np.random.default_rng(2)
    winner = tournament_select(population, evaluations, rng, 2)
    assert winner == "high"


def test_tournament_prefers_feasible_over_infeasible() -> None:
    population = ["infeasible", "feasible"]
    evaluations = [
        SimpleNamespace(feasible=False, fitness=500.0),
        SimpleNamespace(feasible=True, fitness=-50.0),
    ]
    rng = np.random.default_rng(2)
    assert tournament_select(population, evaluations, rng, 2) == "feasible"


def test_invalid_mutation_rate_rejected(scenario) -> None:
    env, drones, _fleet = scenario
    parent, _ = make_parents(scenario, 9)
    with pytest.raises(ValueError):
        mutate(parent, env, drones, np.random.default_rng(1), 1.5)
