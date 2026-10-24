"""Central configuration objects for the UAV strategy engine.

All numeric values in this module are *simulation assumptions*, not claims about
real UAV hardware.  They are normalised so that experiments are reproducible
and easy to reason about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EnvironmentConfig:
    """Parameters used to generate the 50 x 50 hazard grid.

    Coordinate convention: ``x`` is the column, ``y`` is the row.  Arrays are
    indexed ``array[y, x]``.  ``(0, 0)`` is the top-left cell.  This convention
    is used consistently in the evaluator and in every Matplotlib figure.
    """

    seed: int = 42
    width: int = 50
    height: int = 50
    n_targets: int = 14

    n_obstacle_blobs: int = 12
    obstacle_radius: tuple[int, int] = (2, 5)

    n_nofly_blobs: int = 7
    nofly_radius: tuple[int, int] = (2, 4)

    n_hazard_blobs: int = 8
    hazard_sigma: tuple[float, float] = (3.5, 7.0)
    hazard_amplitude: tuple[float, float] = (0.45, 0.90)

    target_priority: tuple[float, float] = (0.5, 1.0)
    target_zone_radius: float = 3.0
    min_start_spacing: int = 6
    min_target_spacing: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FleetConfig:
    """Fleet-level simulation parameters (normalised units)."""

    n_drones: int = 3
    waypoints_per_route: int = 6
    max_flight_range: float = 110.0
    battery_capacity: float = 260.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FitnessWeights:
    """Weights for the scalarised multi-objective fitness function.

    The assignment's illustrative formula is used as the default.  Changing
    these weights changes which trade-off the GA prefers; they are exposed here
    and recorded in every output artifact.
    """

    coverage: float = 100.0
    detection: float = 50.0
    energy: float = 20.0
    time: float = 15.0
    risk: float = 30.0

    # Hard-constraint handling: infeasible candidates receive a large penalty
    # so they never outrank feasible candidates, but feasibility is also stored
    # independently of this scalar (see EvaluationResult.feasible).
    infeasible_base_penalty: float = 1000.0
    violation_penalty: float = 50.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GAConfig:
    """Genetic-algorithm hyperparameters (mutable for experiments)."""

    population_size: int = 60
    generations: int = 50
    mutation_rate: float = 0.2
    tournament_size: int = 3
    crossover_rate: float = 0.9
    elitism: int = 2
    stagnation_limit: int | None = None
    immigrant_rate: float = 0.1
    immigrant_patience: int = 4
    seed: int = 42
    cache_evaluations: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationConfig:
    """Complete configuration bundle saved with every run."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    ga: GAConfig = field(default_factory=GAConfig)
    time_reference: float = 120.0

    def to_dict(self) -> dict:
        return asdict(self)
