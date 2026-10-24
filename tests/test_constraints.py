"""Hard-constraint and repair tests."""

from __future__ import annotations

import numpy as np

from model import GridEnvironment, Target
from simulation import (
    nearest_free_cell,
    route_is_clear,
    segment_is_clear,
    segment_samples,
)
from simulation import random_valid_route, repair_route
from model import DroneRoute, Waypoint


def make_wall_env() -> GridEnvironment:
    width, height = 10, 10
    obstacles = np.zeros((height, width), dtype=bool)
    nofly = np.zeros((height, width), dtype=bool)
    nofly[:, 5] = True  # vertical restricted-airspace wall
    return GridEnvironment(
        obstacle_mask=obstacles,
        nofly_mask=nofly,
        hazard_map=np.zeros((height, width)),
        targets=[Target("T0", 2, 2, 1.0)],
        starts=[(0, 5), (9, 5)],
        seed=1,
    )


def test_segment_sampling_includes_endpoints() -> None:
    samples = segment_samples(0, 0, 4, 0)
    assert tuple(samples[0]) == (0, 0)
    assert tuple(samples[-1]) == (4, 0)
    assert {tuple(s) for s in samples} >= {(0, 0), (2, 0), (4, 0)}


def test_segment_crossing_nofly_is_detected() -> None:
    env = make_wall_env()
    assert not segment_is_clear(env, 4, 5, 6, 5)
    assert segment_is_clear(env, 1, 1, 3, 3)


def test_waypoint_inside_nofly_is_repaired() -> None:
    env = make_wall_env()
    route = DroneRoute([Waypoint(5, 5), Waypoint(4, 5)])
    repaired = repair_route(env, route, start=(3, 5), rng=np.random.default_rng(1))
    assert repaired is not None
    coords = [(wp.x, wp.y) for wp in repaired.waypoints]
    assert route_is_clear(env, (3, 5), coords)
    assert all(not env.nofly_mask[y, x] for x, y in coords)


def test_crossing_route_is_repaired_or_regenerated() -> None:
    env = make_wall_env()
    route = DroneRoute([Waypoint(9, 5)])
    repaired = repair_route(env, route, start=(0, 5), rng=np.random.default_rng(2))
    assert repaired is not None
    coords = [(wp.x, wp.y) for wp in repaired.waypoints]
    assert route_is_clear(env, (0, 5), coords)
    # A regenerated route stays on the start side of the wall.
    assert all(x < 5 for x, _ in coords)


def test_out_of_bounds_waypoint_is_repaired() -> None:
    env = make_wall_env()
    route = DroneRoute([Waypoint(20, 20), Waypoint(2, 2)])
    repaired = repair_route(env, route, start=(1, 1), rng=np.random.default_rng(3))
    assert repaired is not None
    coords = [(wp.x, wp.y) for wp in repaired.waypoints]
    assert route_is_clear(env, (1, 1), coords)
    assert all(env.in_bounds(x, y) for x, y in coords)


def test_random_valid_route_is_valid() -> None:
    env = make_wall_env()
    rng = np.random.default_rng(4)
    route = random_valid_route(env, (1, 1), 5, rng, env.targets)
    assert route is not None
    assert len(route.waypoints) == 5
    coords = [(wp.x, wp.y) for wp in route.waypoints]
    assert route_is_clear(env, (1, 1), coords)


def test_nearest_free_cell() -> None:
    env = make_wall_env()
    assert nearest_free_cell(env, 5, 5) == (4, 4)
    assert nearest_free_cell(env, 0, 0) == (0, 0)
