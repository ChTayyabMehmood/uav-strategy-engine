"""UAV fleet model and normalised performance assumptions.

All values are deliberately *simulation units*.  They are not datasheet
parameters for a real drone.  Keeping them normalised makes the trade-offs
between speed, energy, sensor use, and time explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from config import FleetConfig

# Speed modes: 0 = slow, 1 = cruise, 2 = fast.
# ``cells_per_time`` is a normalised ground speed in grid cells per time unit.
SPEED_MODES: dict[int, dict[str, float | str]] = {
    0: {"name": "slow", "cells_per_time": 0.70},
    1: {"name": "cruise", "cells_per_time": 1.00},
    2: {"name": "fast", "cells_per_time": 1.50},
}
CRUISE_SPEED = float(SPEED_MODES[1]["cells_per_time"])

# Sensor modes: 0 = off, 1 = medium, 2 = high.
# ``radius`` is in grid cells (Euclidean distance from a sampled route point).
# ``energy_per_time`` is added while a leg is flown with that sensor mode.
SENSOR_MODES: dict[int, dict[str, float | str]] = {
    0: {"name": "off", "radius": 0.0, "energy_per_time": 0.0},
    1: {"name": "medium", "radius": 1.6, "energy_per_time": 0.35},
    2: {"name": "high", "radius": 3.2, "energy_per_time": 0.85},
}

# Energy per grid cell at cruise speed.  Movement energy scales with the square
# of the speed ratio (a simple aerodynamic-style drag assumption).
MOVE_ENERGY_PER_CELL = 1.0
SPEED_DRAG_EXPONENT = 2.0


@dataclass(frozen=True)
class Drone:
    """A single UAV and its hard mission limits."""

    drone_id: int
    start: tuple[int, int]
    max_flight_range: float
    battery_capacity: float

    def to_dict(self) -> dict:
        return asdict(self)


def speed_of(mode: int) -> float:
    """Return normalised speed for a mode, validating the mode index."""

    if mode not in SPEED_MODES:
        raise ValueError(f"invalid speed mode {mode!r}")
    return float(SPEED_MODES[mode]["cells_per_time"])


def sensor_radius(mode: int) -> float:
    if mode not in SENSOR_MODES:
        raise ValueError(f"invalid sensor mode {mode!r}")
    return float(SENSOR_MODES[mode]["radius"])


def sensor_energy_rate(mode: int) -> float:
    if mode not in SENSOR_MODES:
        raise ValueError(f"invalid sensor mode {mode!r}")
    return float(SENSOR_MODES[mode]["energy_per_time"])


def build_fleet(starts: list[tuple[int, int]], cfg: FleetConfig) -> list[Drone]:
    """Create ``cfg.n_drones`` drones at the supplied launch cells."""

    if len(starts) != cfg.n_drones:
        raise ValueError(
            f"expected {cfg.n_drones} start positions, got {len(starts)}"
        )
    return [
        Drone(
            drone_id=i,
            start=(int(x), int(y)),
            max_flight_range=cfg.max_flight_range,
            battery_capacity=cfg.battery_capacity,
        )
        for i, (x, y) in enumerate(starts)
    ]


def modes_table() -> dict:
    """Human-readable mode table for docs and JSON metadata."""

    return {
        "speed_modes": {k: dict(v) for k, v in SPEED_MODES.items()},
        "sensor_modes": {k: dict(v) for k, v in SENSOR_MODES.items()},
        "move_energy_per_cell": MOVE_ENERGY_PER_CELL,
        "speed_drag_exponent": SPEED_DRAG_EXPONENT,
    }
