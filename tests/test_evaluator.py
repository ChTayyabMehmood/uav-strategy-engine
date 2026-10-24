"""Mission-evaluator tests with small hand-built environments."""

from __future__ import annotations

import numpy as np
import pytest

from config import FitnessWeights
from environment import GridEnvironment, Target
from simulation import MissionEvaluator
from uav import Chromosome, DroneRoute, Waypoint, build_fleet
from config import FleetConfig


def make_env(
    targets: list[Target] | None = None,
    nofly: np.ndarray | None = None,
    hazard: np.ndarray | None = None,
    starts: list[tuple[int, int]] | None = None,
) -> GridEnvironment:
    width, height = 10, 10
    return GridEnvironment(
        obstacle_mask=np.zeros((height, width), dtype=bool),
        nofly_mask=(
            nofly
            if nofly is not None
            else np.zeros((height, width), dtype=bool)
        ),
        hazard_map=(
            hazard
            if hazard is not None
            else np.zeros((height, width), dtype=float)
        ),
        targets=[] if targets is None else targets,
        starts=starts or [(0, 0), (9, 0)],
        seed=0,
    )


def evaluator_for(
    env: GridEnvironment,
    n_drones: int = 2,
    battery: float = 260.0,
    max_range: float = 110.0,
) -> MissionEvaluator:
    fleet = FleetConfig(
        n_drones=n_drones,
        waypoints_per_route=1,
        max_flight_range=max_range,
        battery_capacity=battery,
    )
    drones = build_fleet(env.starts[:n_drones], fleet)
    return MissionEvaluator(env, drones, FitnessWeights(), time_reference=120.0)


def route(*points: tuple[int, int, int, int]) -> DroneRoute:
    return DroneRoute([Waypoint(x, y, speed, sensor) for x, y, speed, sensor in points])


def test_priority_weighted_coverage() -> None:
    targets = [
        Target("T0", 2, 2, 0.5),
        Target("T1", 8, 8, 1.0),
    ]
    env = make_env(targets=targets)
    chromosome = Chromosome(
        [
            route((2, 2, 1, 2)),
            route((8, 8, 1, 2)),
        ]
    )
    result = evaluator_for(env).evaluate(chromosome)
    assert result.feasible
    assert result.coverage == pytest.approx(1.0)
    assert result.detection_rate == pytest.approx(1.0)
    assert set(result.covered_targets) == {"T0", "T1"}


def test_partial_coverage_is_priority_weighted() -> None:
    targets = [
        Target("T0", 2, 2, 0.5),
        Target("T1", 8, 8, 1.0),
    ]
    env = make_env(targets=targets)
    chromosome = Chromosome(
        [
            route((2, 2, 1, 2)),
            route((9, 9, 1, 0)),
        ]
    )
    result = evaluator_for(env).evaluate(chromosome)
    assert result.coverage == pytest.approx(0.5 / 1.5)
    assert result.detection_rate == pytest.approx(0.5)


def test_duplicate_detections_count_once() -> None:
    targets = [Target("T0", 5, 5, 1.0)]
    env = make_env(targets=targets, starts=[(0, 5), (9, 5)])
    chromosome = Chromosome(
        [
            route((5, 5, 1, 2)),
            route((5, 5, 1, 2)),
        ]
    )
    result = evaluator_for(env).evaluate(chromosome)
    assert result.coverage == pytest.approx(1.0)
    assert result.detection_rate == pytest.approx(1.0)
    assert result.covered_targets == ["T0"]


def test_deployment_genes_control_the_start_cell() -> None:
    env = make_env(targets=[], starts=[(0, 0), (9, 0)])
    chromosome = Chromosome(
        [
            route((2, 0, 1, 0)),
            route((2, 0, 1, 0)),
        ],
        deployments=[(0, 0), (5, 0)],
    )
    result = evaluator_for(env).evaluate(chromosome)
    assert [item.start for item in result.per_drone] == [(0, 0), (5, 0)]
    assert result.per_drone[1].distance == pytest.approx(3.0)


def test_empty_targets_are_safe() -> None:
    env = make_env(targets=[])
    chromosome = Chromosome([route((2, 2, 1, 2)), route((8, 8, 1, 2))])
    result = evaluator_for(env).evaluate(chromosome)
    assert result.coverage == 0.0
    assert result.detection_rate == 0.0
    assert result.feasible


def test_energy_and_time_for_known_leg() -> None:
    env = make_env(targets=[], starts=[(0, 0)])
    chromosome = Chromosome([route((2, 0, 1, 0))])
    result = evaluator_for(env, n_drones=1).evaluate(chromosome)
    assert result.total_distance == pytest.approx(2.0)
    assert result.total_energy == pytest.approx(2.0)
    assert result.mission_time == pytest.approx(2.0)
    assert result.normalized_energy == pytest.approx(2.0 / 260.0)
    assert result.normalized_time == pytest.approx(2.0 / 120.0)


def test_sensor_energy_is_added() -> None:
    env = make_env(targets=[], starts=[(0, 0)])
    chromosome = Chromosome([route((2, 0, 1, 1))])
    result = evaluator_for(env, n_drones=1).evaluate(chromosome)
    assert result.total_energy == pytest.approx(2.0 + 2.0 * 0.35)


def test_hazard_exposure_and_normalised_risk() -> None:
    env = make_env(
        targets=[],
        starts=[(0, 0)],
        hazard=np.full((10, 10), 0.5),
    )
    chromosome = Chromosome([route((2, 0, 1, 0))])
    result = evaluator_for(env, n_drones=1).evaluate(chromosome)
    assert result.hazard_exposure == pytest.approx(1.0)
    assert result.normalized_risk == pytest.approx(0.5)


def test_metrics_are_non_negative() -> None:
    env = make_env(targets=[Target("T0", 3, 3, 1.0)])
    result = evaluator_for(env).evaluate(
        Chromosome([route((3, 3, 0, 2)), route((1, 4, 2, 1))])
    )
    assert result.total_energy >= 0
    assert result.mission_time >= 0
    assert result.hazard_exposure >= 0
    assert result.normalized_risk >= 0


def test_nofly_route_cannot_be_feasible() -> None:
    nofly = np.zeros((10, 10), dtype=bool)
    nofly[0, 1] = True
    env = make_env(targets=[], nofly=nofly, starts=[(0, 0)])
    chromosome = Chromosome([route((2, 0, 1, 1))])
    result = evaluator_for(env, n_drones=1).evaluate(chromosome)
    assert not result.feasible
    assert any(v.kind == "no_fly_zone" for v in result.violations)
    assert result.penalty > 0


def test_range_and_battery_violations() -> None:
    env = make_env(targets=[], starts=[(0, 0)])
    chromosome = Chromosome([route((5, 0, 1, 0))])
    evaluator = evaluator_for(
        env, n_drones=1, battery=1.0, max_range=1.0
    )
    result = evaluator.evaluate(chromosome)
    kinds = {v.kind for v in result.violations}
    assert {"battery", "max_range"} <= kinds
    assert not result.feasible


def test_invalid_structure_is_detected() -> None:
    env = make_env(targets=[], starts=[(0, 0)])
    evaluator = evaluator_for(env, n_drones=1)
    wrong_count = Chromosome([])
    result = evaluator.evaluate(wrong_count)
    assert not result.feasible
    assert result.violations[0].kind == "invalid_structure"

    out_of_bounds = Chromosome([route((20, 0, 1, 1))])
    result = evaluator.evaluate(out_of_bounds)
    assert not result.feasible
    assert result.violations[0].kind == "invalid_structure"


def test_evaluation_is_deterministic() -> None:
    env = make_env(targets=[Target("T0", 3, 3, 1.0)])
    chromosome = Chromosome([route((3, 3, 1, 2)), route((9, 9, 1, 1))])
    evaluator = evaluator_for(env)
    a = evaluator.evaluate(chromosome).to_dict()
    b = evaluator.evaluate(chromosome).to_dict()
    assert a == b
