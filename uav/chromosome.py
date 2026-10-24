"""Chromosome representation for the fleet strategy.

A chromosome has one :class:`DroneRoute` per UAV.  Each route is a fixed-length
sequence of :class:`Waypoint` genes.  A gene stores:

* integer ``x`` / ``y`` grid coordinates (the waypoint the drone flies to);
* a categorical ``speed`` mode (0 slow, 1 cruise, 2 fast);
* a categorical ``sensor`` power mode (0 off, 1 medium, 2 high).

A second, shorter segment stores each drone's deployment cell.  It is evolved
alongside the routes, so the GA chooses where the fleet starts as well as where
it flies.  Settings on waypoint *i* govern the leg flown from the previous
point to waypoint *i*.

Waypoints are immutable (``frozen=True``), which makes it safe for crossover to
reuse them while still building independent route lists.  :meth:`copy` returns
a fully independent chromosome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from uav.drone import SENSOR_MODES, SPEED_MODES, Drone


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
    """Complete fleet strategy: deployment cells plus one route per drone."""

    routes: list[DroneRoute] = field(default_factory=list)
    deployments: list[tuple[int, int]] = field(default_factory=list)

    def copy(self) -> Chromosome:
        return Chromosome(
            routes=[route.copy() for route in self.routes],
            deployments=list(self.deployments),
        )

    def start_for(
        self, drone_id: int, fallback: tuple[int, int]
    ) -> tuple[int, int]:
        """Return the evolved deployment cell, or a fixed fallback."""

        if drone_id < len(self.deployments):
            return self.deployments[drone_id]
        return fallback

    def signature(self) -> tuple:
        """Stable hashable representation used for evaluation caching."""

        routes = tuple(
            tuple(
                (wp.x, wp.y, wp.speed, wp.sensor)
                for wp in route.waypoints
            )
            for route in self.routes
        )
        return routes, tuple(self.deployments)

    def validate_structure(
        self,
        n_drones: int,
        n_waypoints: int,
        width: int | None = None,
        height: int | None = None,
        require_deployments: bool = False,
    ) -> None:
        """Raise ``ValueError`` if the chromosome is structurally invalid.

        Bounds are checked only when ``width`` / ``height`` are provided.
        ``Waypoint`` itself validates coordinates are integers and modes are
        valid.  Older tests and callers may omit deployments; the evaluator
        then falls back to the environment's fixed launch cells.
        """

        if not isinstance(self.routes, list):
            raise ValueError("routes must be a list")
        if len(self.routes) != n_drones:
            raise ValueError(
                f"expected {n_drones} routes, found {len(self.routes)}"
            )
        if self.deployments:
            if len(self.deployments) != n_drones:
                raise ValueError(
                    f"expected {n_drones} deployments, "
                    f"found {len(self.deployments)}"
                )
            for i, (x, y) in enumerate(self.deployments):
                if not isinstance(x, int) or not isinstance(y, int):
                    raise ValueError(f"deployment {i} must use integer cells")
                if width is not None and not 0 <= x < width:
                    raise ValueError(f"deployment {i} x={x} out of bounds")
                if height is not None and not 0 <= y < height:
                    raise ValueError(f"deployment {i} y={y} out of bounds")
        elif require_deployments:
            raise ValueError("deployments are required")
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
        return {
            "deployments": [list(cell) for cell in self.deployments],
            "routes": [route.to_dict() for route in self.routes],
        }

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
        return Chromosome(
            routes=routes,
            deployments=[
                (int(x), int(y))
                for x, y in data.get("deployments", [])
            ],
        )


def fleet_for_chromosome(
    base_fleet: list[Drone],
    chromosome: Chromosome,
) -> list[Drone]:
    """Return drones whose starts honour the evolved deployment genes."""

    return [
        Drone(
            drone_id=drone.drone_id,
            start=chromosome.start_for(drone.drone_id, drone.start),
            max_flight_range=drone.max_flight_range,
            battery_capacity=drone.battery_capacity,
        )
        for drone in base_fleet
    ]
