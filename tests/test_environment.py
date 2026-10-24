"""Tests for deterministic environment generation."""

from __future__ import annotations

import numpy as np
import pytest

from config import EnvironmentConfig
from environment import EnvironmentGenerationError, generate_environment


def test_grid_dimensions_and_targets_are_valid() -> None:
    env = generate_environment(EnvironmentConfig(seed=7, width=30, height=30), 3)
    assert env.obstacle_mask.shape == (30, 30)
    assert env.nofly_mask.shape == (30, 30)
    assert env.hazard_map.shape == (30, 30)
    assert env.hazard_map.min() >= 0.0
    assert env.hazard_map.max() <= 1.0
    assert len(env.starts) == 3
    assert len(env.targets) == 14
    for target in env.targets:
        assert env.in_bounds(target.x, target.y)
        assert not env.forbidden_mask[target.y, target.x]
        assert 0.0 < target.priority <= 1.0
    for x, y in env.starts:
        assert env.is_free(x, y)


def test_priority_map_covers_target_zones() -> None:
    env = generate_environment(
        EnvironmentConfig(seed=7, width=30, height=30), 3
    )
    assert env.priority_map.shape == (30, 30)
    assert env.priority_mass > 0.0
    assert env.priority_map.sum() == pytest.approx(env.priority_mass)
    for target in env.targets:
        assert env.priority_map[target.y, target.x] > 0.0


def test_same_seed_is_reproducible() -> None:
    cfg = EnvironmentConfig(seed=123, width=25, height=25)
    a = generate_environment(cfg, 2)
    b = generate_environment(cfg, 2)
    assert np.array_equal(a.obstacle_mask, b.obstacle_mask)
    assert np.array_equal(a.nofly_mask, b.nofly_mask)
    assert np.array_equal(a.hazard_map, b.hazard_map)
    assert a.starts == b.starts
    assert [t.to_dict() for t in a.targets] == [t.to_dict() for t in b.targets]


def test_different_seed_changes_map() -> None:
    a = generate_environment(EnvironmentConfig(seed=1, width=25, height=25), 2)
    b = generate_environment(EnvironmentConfig(seed=2, width=25, height=25), 2)
    assert not np.array_equal(a.obstacle_mask, b.obstacle_mask)


def test_out_of_bounds_queries_are_safe() -> None:
    env = generate_environment(EnvironmentConfig(seed=4, width=20, height=20), 1)
    assert not env.in_bounds(-1, 0)
    assert not env.in_bounds(0, 20)
    assert not env.is_free(20, 20)


def test_impossible_map_raises() -> None:
    cfg = EnvironmentConfig(
        seed=3,
        width=6,
        height=6,
        n_obstacle_blobs=80,
        obstacle_radius=(3, 4),
        n_nofly_blobs=80,
        nofly_radius=(3, 4),
    )
    with pytest.raises(EnvironmentGenerationError):
        generate_environment(cfg, 2)
