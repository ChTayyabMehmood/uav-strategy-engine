"""Chromosome representation for the fleet strategy.

A chromosome has one :class:`DroneRoute` per UAV.  Each route is a fixed-length
sequence of :class:`Waypoint` genes.  A gene stores:

* integer ``x`` / ``y`` grid coordinates (the waypoint the drone flies to);
* a categorical ``speed`` mode (0 slow, 1 cruise, 2 fast);
* a categorical ``sensor`` power mode (0 off, 1 medium, 2 high).

The drone's launch cell is *not* a gene: the evaluator always starts at the
drone's fixed start position.  Settings on waypoint *i* govern the leg flown
from the previous point to waypoint *i*.

Waypoints are immutable (``frozen=True``), which makes it safe for crossover to
reuse them while still building independent route lists.  :meth:`copy` returns
a fully independent chromosome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from uav.drone import SENSOR_MODES, SPEED_MODES


@dataclass(frozen=True)
class Waypoint:
    x: int
    y: int
    speed: int = 1
    sensor: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValueError("waypoint coordinates must be integers")
        if self.speed not in SPEED_MODES:
            raise ValueError(f"invalid speed mode {self.speed!r}")
        if self.sensor not in SENSOR_MODES:
            raise ValueError(f"invalid sensor mode {self.sensor!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DroneRoute:
    waypoints: list[Waypoint] = field(default_factory=list)

    def copy(self) -> DroneRoute:
        return DroneRoute(waypoints=list(self.waypoints))

    def to_dict(self) -> dict:
        return {"waypoints": [wp.to_dict() for wp in self.waypoints]}


@dataclass
class Chromosome:
    """Complete fleet strategy: one route per drone."""

    routes: list[DroneRoute] = field(default_factory=list)

    def copy(self) -> Chromosome:
        return Chromosome(routes=[route.copy() for route in self.routes])

    def signature(self) -> tuple:
        """Stable hashable representation used for evaluation caching."""

        return tuple(
            tuple(
                (wp.x, wp.y, wp.speed, wp.sensor)
                for wp in route.waypoints
            )
            for route in self.routes
        )

    def validate_structure(
        self,
        n_drones: int,
        n_waypoints: int,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Raise ``ValueError`` if the chromosome is structurally invalid.

        Bounds are checked only when ``width`` / ``height`` are provided.
        ``Waypoint`` itself validates coordinates are integers and modes are
        valid.
        """

        if not isinstance(self.routes, list):
            raise ValueError("routes must be a list")
        if len(self.routes) != n_drones:
            raise ValueError(
                f"expected {n_drones} routes, found {len(self.routes)}"
            )
        for i, route in enumerate(self.routes):
            if not isinstance(route, DroneRoute):
                raise ValueError(f"route {i} is not a DroneRoute")
            if len(route.waypoints) != n_waypoints:
                raise ValueError(
                    f"route {i} has {len(route.waypoints)} waypoints, "
                    f"expected {n_waypoints}"
                )
            for j, wp in enumerate(route.waypoints):
                if not isinstance(wp, Waypoint):
                    raise ValueError(f"gene {i}:{j} is not a Waypoint")
                if width is not None and not 0 <= wp.x < width:
                    raise ValueError(f"waypoint {i}:{j} x={wp.x} out of bounds")
                if height is not None and not 0 <= wp.y < height:
                    raise ValueError(f"waypoint {i}:{j} y={wp.y} out of bounds")

    def to_dict(self) -> dict:
        return {"routes": [route.to_dict() for route in self.routes]}

    @staticmethod
    def from_dict(data: dict) -> Chromosome:
        routes = [
            DroneRoute(
                waypoints=[
                    Waypoint(
                        x=int(wp["x"]),
                        y=int(wp["y"]),
                        speed=int(wp.get("speed", 1)),
                        sensor=int(wp.get("sensor", 1)),
                    )
                    for wp in route["waypoints"]
                ]
            )
            for route in data["routes"]
        ]
        return Chromosome(routes=routes)
