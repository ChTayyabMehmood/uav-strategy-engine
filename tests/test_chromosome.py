"""Chromosome structure, copying, and serialisation tests."""

from __future__ import annotations

import pytest

from model import Chromosome, DroneRoute, Waypoint


def make_chromosome() -> Chromosome:
    return Chromosome(
        routes=[
            DroneRoute(
                [
                    Waypoint(1, 2, speed=0, sensor=1),
                    Waypoint(3, 4, speed=1, sensor=2),
                    Waypoint(5, 6, speed=2, sensor=0),
                ]
            ),
            DroneRoute(
                [
                    Waypoint(7, 8),
                    Waypoint(9, 0),
                    Waypoint(2, 5, speed=0, sensor=0),
                ]
            ),
        ]
    )


def test_structure_is_valid() -> None:
    chromosome = make_chromosome()
    chromosome.validate_structure(2, 3, width=10, height=10)
    assert chromosome.signature()[0][0][0] == (1, 2, 0, 1)


def test_copy_is_independent() -> None:
    original = make_chromosome()
    clone = original.copy()
    clone.routes[0].waypoints[0] = Waypoint(9, 9, 2, 2)
    clone.routes.append(DroneRoute([]))
    assert original.routes[0].waypoints[0] == Waypoint(1, 2, 0, 1)
    assert len(original.routes) == 2
    assert original.signature() != clone.signature()


def test_deployment_genes_round_trip() -> None:
    chromosome = make_chromosome()
    chromosome.deployments = [(1, 2), (3, 4)]
    restored = Chromosome.from_dict(chromosome.to_dict())
    assert restored.deployments == [(1, 2), (3, 4)]
    assert restored.signature() == chromosome.signature()


def test_wrong_route_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_chromosome().validate_structure(3, 3)


def test_wrong_waypoint_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_chromosome().validate_structure(2, 4)


def test_out_of_bounds_is_rejected_when_bounds_given() -> None:
    with pytest.raises(ValueError):
        make_chromosome().validate_structure(2, 3, width=5, height=5)


def test_invalid_modes_are_rejected() -> None:
    with pytest.raises(ValueError):
        Waypoint(1, 1, speed=3)
    with pytest.raises(ValueError):
        Waypoint(1, 1, sensor=-1)


def test_non_integer_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError):
        Waypoint(1.5, 1)  # type: ignore[arg-type]


def test_serialisation_round_trip() -> None:
    original = make_chromosome()
    restored = Chromosome.from_dict(original.to_dict())
    assert restored.signature() == original.signature()
