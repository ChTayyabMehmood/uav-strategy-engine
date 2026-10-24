"""Mission simulation: segment checks, repair, evaluation and fitness."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from config import FitnessWeights
from model import (
    CRUISE_SPEED,
    MOVE_ENERGY_PER_CELL,
    SPEED_DRAG_EXPONENT,
    Chromosome,
    Drone,
    DroneRoute,
    GridEnvironment,
    Target,
    Waypoint,
    fleet_for_chromosome,
    sensor_energy_rate,
    sensor_radius,
    speed_of,
)


# ======================================================================
# Segment checks
# ======================================================================

"""Grid geometry and hard-constraint checks.

This module deliberately contains no genetic-algorithm logic.  It provides
small helpers that the evaluator, the repair operator, and the tests share:

* dense sampling of a straight flight leg;
* line-of-sight / forbidden-cell checks;
* nearest free-cell search.

Dense sampling (every 0.5 cell) is used so a leg is checked along its whole
length, not just at its endpoints.  A route is legal only when every sampled
cell is inside the grid and outside obstacles and no-fly zones.
"""


SAMPLE_STEP = 0.5


def segment_samples(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: float = SAMPLE_STEP,
) -> np.ndarray:
    """Return integer cells sampled densely along a straight segment.

    The endpoints are always included.  Sample density is independent of leg
    length, which makes hazard integration and detection checks consistent.
    """

    distance = math.hypot(x1 - x0, y1 - y0)
    if distance == 0.0:
        return np.array([[int(x0), int(y0)]], dtype=int)
    n = max(1, int(math.ceil(distance / step)))
    ts = np.linspace(0.0, 1.0, n + 1)
    xs = np.rint(x0 + (x1 - x0) * ts).astype(int)
    ys = np.rint(y0 + (y1 - y0) * ts).astype(int)
    return np.stack((xs, ys), axis=1)


def segment_distance(x0: int, y0: int, x1: int, y1: int) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def samples_in_bounds(env: GridEnvironment, samples: np.ndarray) -> bool:
    return bool(
        np.all(samples[:, 0] >= 0)
        and np.all(samples[:, 0] < env.width)
        and np.all(samples[:, 1] >= 0)
        and np.all(samples[:, 1] < env.height)
    )


def segment_is_clear(
    env: GridEnvironment,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    """True if the whole straight leg is in bounds and over free cells."""

    samples = segment_samples(x0, y0, x1, y1)
    if not samples_in_bounds(env, samples):
        return False
    xs = samples[:, 0]
    ys = samples[:, 1]
    return not bool(np.any(env.forbidden_mask[ys, xs]))


def nearest_free_cell(
    env: GridEnvironment,
    x: int,
    y: int,
    max_radius: int = 25,
) -> tuple[int, int] | None:
    """Find the closest free cell to ``(x, y)`` by Chebyshev rings."""

    if env.is_free(x, y):
        return int(x), int(y)
    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx, ny = x + dx, y + dy
                if env.is_free(nx, ny):
                    return int(nx), int(ny)
    return None


def route_is_clear(
    env: GridEnvironment,
    start: tuple[int, int],
    waypoints: list[tuple[int, int]],
) -> bool:
    prev_x, prev_y = start
    for x, y in waypoints:
        if not env.is_free(x, y):
            return False
        if not segment_is_clear(env, prev_x, prev_y, x, y):
            return False
        prev_x, prev_y = x, y
    return True

# ======================================================================
# Constraint repair
# ======================================================================

"""Constraint repair for routes and chromosomes.

Repair is intentionally kept outside the GA's genetic operators.  The GA
produces a candidate; this module makes a best effort to turn it into a
feasible straight-leg route.  If local repair cannot fix a route, the route is
regenerated from scratch (a documented fallback).  The evaluator still performs
an independent feasibility check afterwards, so repair can never silently mark
an invalid route as valid.
"""


MAX_RAY_DISTANCE = 22.0


def _ray_cell(
    env: GridEnvironment,
    origin: tuple[int, int],
    angle: float,
    max_distance: float,
    step: float = 0.5,
) -> tuple[int, int] | None:
    """March from ``origin`` and return the farthest clear cell on the ray."""

    ox, oy = origin
    dx, dy = math.cos(angle), math.sin(angle)
    last: tuple[int, int] | None = None
    t = step
    while t <= max_distance + 1e-9:
        x = int(round(ox + dx * t))
        y = int(round(oy + dy * t))
        if not env.is_free(x, y):
            break
        last = (x, y)
        t += step
    return last


def sample_area_waypoint(
    env: GridEnvironment,
    origin: tuple[int, int],
    rng: np.random.Generator,
    targets: list[Target] | None = None,
) -> tuple[int, int] | None:
    """Sample one reachable waypoint using a clear straight ray.

    With probability 0.6 the ray points toward a randomly chosen target (with a
    small angular jitter), which seeds the population with useful exploration.
    Otherwise the direction is random.  The returned cell always has a clear
    line of sight from ``origin``.
    """

    targets = targets or []
    for _attempt in range(50):
        if targets and rng.random() < 0.6:
            target = targets[int(rng.integers(0, len(targets)))]
            angle = math.atan2(target.y - origin[1], target.x - origin[0])
            angle += float(rng.uniform(-0.15, 0.15))
            max_distance = min(
                math.hypot(target.x - origin[0], target.y - origin[1]),
                MAX_RAY_DISTANCE * 1.4,
            )
        else:
            angle = float(rng.uniform(-math.pi, math.pi))
            max_distance = float(rng.uniform(4.0, MAX_RAY_DISTANCE))

        cell = _ray_cell(env, origin, angle, max_distance)
        if cell is None or cell == origin:
            # Try a short free step if the ray was immediately blocked.
            cell = _ray_cell(env, origin, angle, 2.0)
        if cell is None or cell == origin:
            continue
        if segment_is_clear(env, origin[0], origin[1], cell[0], cell[1]):
            return cell
    return None


def random_valid_route(
    env: GridEnvironment,
    start: tuple[int, int],
    n_waypoints: int,
    rng: np.random.Generator,
    targets: list[Target] | None = None,
) -> DroneRoute | None:
    """Build a random straight-leg route that is clear by construction."""

    waypoints: list[Waypoint] = []
    previous = start
    for _ in range(n_waypoints):
        cell = sample_area_waypoint(env, previous, rng, targets)
        if cell is None:
            return None
        waypoints.append(
            Waypoint(
                x=int(cell[0]),
                y=int(cell[1]),
                speed=int(rng.integers(0, 3)),
                sensor=int(rng.integers(0, 3)),
            )
        )
        previous = cell
    return DroneRoute(waypoints=waypoints)


def _repair_cell(
    env: GridEnvironment,
    previous: tuple[int, int],
    current: tuple[int, int],
    following: tuple[int, int] | None,
    max_radius: int,
) -> tuple[int, int] | None:
    """Find a nearby cell that makes both adjacent legs clear."""

    cx, cy = current
    for radius in range(0, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if not env.is_free(nx, ny):
                    continue
                if not segment_is_clear(
                    env, previous[0], previous[1], nx, ny
                ):
                    continue
                if following is not None and not segment_is_clear(
                    env, nx, ny, following[0], following[1]
                ):
                    continue
                return int(nx), int(ny)
    return None


def repair_route(
    env: GridEnvironment,
    route: DroneRoute,
    start: tuple[int, int],
    rng: np.random.Generator,
    targets: list[Target] | None = None,
    max_radius: int = 8,
) -> DroneRoute | None:
    """Repair a route in place conceptually and return a new route or ``None``.

    Steps:
    1. Clamp waypoints into the grid.
    2. Snap forbidden waypoints to the nearest free cell.
    3. Nudge waypoints until both adjacent legs are clear.
    4. Verify the complete route; regenerate if local repair failed.
    """

    waypoints = [wp for wp in route.waypoints]

    for i, wp in enumerate(waypoints):
        if not env.in_bounds(wp.x, wp.y):
            x = min(max(wp.x, 0), env.width - 1)
            y = min(max(wp.y, 0), env.height - 1)
            waypoints[i] = Waypoint(x, y, wp.speed, wp.sensor)
        if not env.is_free(waypoints[i].x, waypoints[i].y):
            free = nearest_free_cell(env, waypoints[i].x, waypoints[i].y)
            if free is None:
                return None
            waypoints[i] = Waypoint(
                free[0], free[1], waypoints[i].speed, waypoints[i].sensor
            )

    for _pass in range(4):
        changed = False
        for i, wp in enumerate(waypoints):
            previous = (
                (waypoints[i - 1].x, waypoints[i - 1].y)
                if i > 0
                else start
            )
            following = (
                (waypoints[i + 1].x, waypoints[i + 1].y)
                if i + 1 < len(waypoints)
                else None
            )
            current = (wp.x, wp.y)
            first_ok = segment_is_clear(
                env, previous[0], previous[1], current[0], current[1]
            )
            second_ok = following is None or segment_is_clear(
                env, current[0], current[1], following[0], following[1]
            )
            if first_ok and second_ok:
                continue
            replacement = _repair_cell(
                env, previous, current, following, max_radius
            )
            if replacement is not None:
                waypoints[i] = Waypoint(
                    replacement[0], replacement[1], wp.speed, wp.sensor
                )
                changed = True
        if not changed:
            break

    previous = start
    for wp in waypoints:
        if not segment_is_clear(env, previous[0], previous[1], wp.x, wp.y):
            break
        previous = (wp.x, wp.y)
    else:
        return DroneRoute(waypoints=waypoints)

    # Local repair failed: documented fallback is a fresh valid route.
    for _attempt in range(30):
        route_candidate = random_valid_route(
            env, start, len(route.waypoints), rng, targets
        )
        if route_candidate is not None:
            return route_candidate
    return None


def repair_chromosome(
    env: GridEnvironment,
    chromosome: Chromosome,
    drones: list[Drone],
    rng: np.random.Generator,
) -> Chromosome | None:
    """Repair every route of a chromosome, or return ``None`` if impossible."""

    if len(chromosome.routes) != len(drones):
        return None
    if chromosome.deployments:
        if len(chromosome.deployments) != len(drones):
            return None
        starts = list(chromosome.deployments)
    else:
        starts = [drone.start for drone in drones]
    for start in starts:
        if not env.is_free(*start):
            return None
    routes: list[DroneRoute] = []
    for route, start in zip(chromosome.routes, starts):
        repaired = repair_route(env, route, start, rng, env.targets)
        if repaired is None:
            return None
        routes.append(repaired)
    return Chromosome(
        routes=routes,
        deployments=list(chromosome.deployments),
    )


def visited_cells(
    env: GridEnvironment,
    start: tuple[int, int],
    route: DroneRoute,
) -> list[tuple[int, int]]:
    """Return every sampled cell on a route (useful for coverage maps)."""

    cells: list[tuple[int, int]] = []
    previous = start
    for wp in route.waypoints:
        samples = segment_samples(previous[0], previous[1], wp.x, wp.y)
        cells.extend((int(x), int(y)) for x, y in samples)
        previous = (wp.x, wp.y)
    return cells

# ======================================================================
# Mission evaluation
# ======================================================================

"""Mission evaluator: simulate a chromosome and compute all objectives.

The evaluator is deliberately separate from the GA.  It returns a structured
result containing every metric, every constraint violation, and an independent
feasibility flag.  A chromosome is never treated as feasible merely because its
scalar fitness is high.
"""


@dataclass(frozen=True)
class Violation:
    kind: str
    drone_id: int | None
    detail: str
    magnitude: float = 1.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "drone_id": self.drone_id,
            "detail": self.detail,
            "magnitude": float(self.magnitude),
        }


@dataclass
class DroneEvaluation:
    drone_id: int
    start: tuple[int, int]
    distance: float
    duration: float
    energy: float
    hazard_exposure: float
    detected_targets: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "drone_id": self.drone_id,
            "start": list(self.start),
            "distance": float(self.distance),
            "duration": float(self.duration),
            "energy": float(self.energy),
            "hazard_exposure": float(self.hazard_exposure),
            "detected_targets": list(self.detected_targets),
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class EvaluationResult:
    fitness: float
    raw_fitness: float
    penalty: float
    coverage: float
    detection_rate: float
    total_energy: float
    normalized_energy: float
    mission_time: float
    normalized_time: float
    total_distance: float
    hazard_exposure: float
    normalized_risk: float
    covered_targets: list[str]
    n_targets: int
    feasible: bool
    violations: list[Violation]
    per_drone: list[DroneEvaluation]
    covered_cells: int = 0
    free_cells: int = 0
    surveyed_fraction: float = 0.0
    coverage_mask: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "fitness": float(self.fitness),
            "raw_fitness": float(self.raw_fitness),
            "penalty": float(self.penalty),
            "coverage": float(self.coverage),
            "detection_rate": float(self.detection_rate),
            "total_energy": float(self.total_energy),
            "normalized_energy": float(self.normalized_energy),
            "mission_time": float(self.mission_time),
            "normalized_time": float(self.normalized_time),
            "total_distance": float(self.total_distance),
            "hazard_exposure": float(self.hazard_exposure),
            "normalized_risk": float(self.normalized_risk),
            "covered_targets": list(self.covered_targets),
            "n_targets": self.n_targets,
            "feasible": bool(self.feasible),
            "violations": [v.to_dict() for v in self.violations],
            "per_drone": [d.to_dict() for d in self.per_drone],
            "covered_cells": int(self.covered_cells),
            "free_cells": int(self.free_cells),
            "surveyed_fraction": float(self.surveyed_fraction),
        }


class MissionEvaluator:
    """Evaluate complete fleet chromosomes on a fixed environment."""

    def __init__(
        self,
        env: GridEnvironment,
        drones: list[Drone],
        weights: FitnessWeights,
        time_reference: float = 120.0,
    ) -> None:
        self.env = env
        self.drones = list(drones)
        self.weights = weights
        self.time_reference = float(time_reference)
        self.target_ids = [t.target_id for t in env.targets]
        self.target_x = np.array([t.x for t in env.targets], dtype=float)
        self.target_y = np.array([t.y for t in env.targets], dtype=float)
        self.priority_map = np.asarray(env.priority_map, dtype=float)
        self.total_priority_mass = float(self.priority_map.sum())
        self.free_cells = env.free_cell_count
        self.total_capacity = float(
            sum(d.battery_capacity for d in self.drones)
        )
        self._sensor_offsets: dict[float, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _invalid_result(self, message: str) -> EvaluationResult:
        violation = Violation("invalid_structure", None, message, 1.0)
        penalty = (
            self.weights.infeasible_base_penalty
            + self.weights.violation_penalty
        )
        return EvaluationResult(
            fitness=-penalty,
            raw_fitness=0.0,
            penalty=penalty,
            coverage=0.0,
            detection_rate=0.0,
            total_energy=0.0,
            normalized_energy=0.0,
            mission_time=0.0,
            normalized_time=0.0,
            total_distance=0.0,
            hazard_exposure=0.0,
            normalized_risk=0.0,
            covered_targets=[],
            n_targets=len(self.env.targets),
            feasible=False,
            violations=[violation],
            per_drone=[],
            covered_cells=0,
            free_cells=self.free_cells,
            surveyed_fraction=0.0,
            coverage_mask=np.zeros_like(self.env.free_mask),
        )

    def _leg_detections(
        self,
        samples: np.ndarray,
        radius: float,
    ) -> set[int]:
        """Indices of targets within ``radius`` of any sampled leg cell."""

        if radius <= 0.0 or len(self.target_ids) == 0:
            return set()
        dx = samples[:, 0][:, None] - self.target_x[None, :]
        dy = samples[:, 1][:, None] - self.target_y[None, :]
        hit = np.any(dx * dx + dy * dy <= radius * radius, axis=0)
        return {int(i) for i in np.flatnonzero(hit)}

    def _offsets(self, radius: float) -> np.ndarray:
        """Integer cell offsets inside a sensor disk."""

        key = round(float(radius), 6)
        cached = self._sensor_offsets.get(key)
        if cached is not None:
            return cached
        r = int(np.ceil(radius))
        dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
        keep = dx * dx + dy * dy <= radius * radius
        offsets = np.stack((dx[keep], dy[keep]), axis=1)
        self._sensor_offsets[key] = offsets
        return offsets

    def _mark_sensor_footprint(
        self,
        mask: np.ndarray,
        samples: np.ndarray,
        radius: float,
    ) -> None:
        """Union the sensor disks around the rounded sampled path cells."""

        if radius <= 0.0 or len(samples) == 0:
            return
        points = np.rint(samples).astype(int)
        points = np.unique(points, axis=0)
        offsets = self._offsets(radius)
        px = points[:, 0][:, None] + offsets[None, :, 0]
        py = points[:, 1][:, None] + offsets[None, :, 1]
        inside = (
            (px >= 0)
            & (px < self.env.width)
            & (py >= 0)
            & (py < self.env.height)
        )
        if not bool(inside.any()):
            return
        px = px[inside]
        py = py[inside]
        free = self.env.free_mask[py, px]
        mask[py[free], px[free]] = True

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, chromosome: Chromosome) -> EvaluationResult:
        n_waypoints = (
            len(chromosome.routes[0].waypoints)
            if chromosome.routes
            else 0
        )
        try:
            chromosome.validate_structure(
                n_drones=len(self.drones),
                n_waypoints=n_waypoints,
                width=self.env.width,
                height=self.env.height,
                require_deployments=bool(chromosome.deployments),
            )
        except ValueError as exc:
            return self._invalid_result(str(exc))

        fleet = fleet_for_chromosome(self.drones, chromosome)
        for drone in fleet:
            if not self.env.is_free(*drone.start):
                return self._invalid_result(
                    f"deployment {drone.drone_id} at {drone.start} "
                    "is outside free airspace"
                )

        covered: set[int] = set()
        coverage_mask = np.zeros((self.env.height, self.env.width), dtype=bool)
        violations: list[Violation] = []
        per_drone: list[DroneEvaluation] = []

        for drone, route in zip(fleet, chromosome.routes):
            previous = drone.start
            distance = 0.0
            duration = 0.0
            energy = 0.0
            risk = 0.0
            detected: set[int] = set()
            drone_violations: list[Violation] = []

            for wp in route.waypoints:
                samples = segment_samples(previous[0], previous[1], wp.x, wp.y)
                length = segment_distance(
                    previous[0], previous[1], wp.x, wp.y
                )
                speed = speed_of(wp.speed)
                duration += length / speed
                speed_ratio = speed / CRUISE_SPEED
                energy += (
                    length
                    * MOVE_ENERGY_PER_CELL
                    * speed_ratio**SPEED_DRAG_EXPONENT
                )
                energy += (length / speed) * sensor_energy_rate(wp.sensor)

                xs = samples[:, 0]
                ys = samples[:, 1]
                inside = (
                    (xs >= 0)
                    & (xs < self.env.width)
                    & (ys >= 0)
                    & (ys < self.env.height)
                )
                n_out = int((~inside).sum())
                if n_out:
                    drone_violations.append(
                        Violation(
                            "out_of_bounds",
                            drone.drone_id,
                            f"leg to ({wp.x}, {wp.y}) leaves the grid",
                            float(n_out),
                        )
                    )

                if bool(inside.any()):
                    vx = xs[inside]
                    vy = ys[inside]
                    nofly_count = int(self.env.nofly_mask[vy, vx].sum())
                    obstacle_count = int(self.env.obstacle_mask[vy, vx].sum())
                    if nofly_count:
                        drone_violations.append(
                            Violation(
                                "no_fly_zone",
                                drone.drone_id,
                                f"leg to ({wp.x}, {wp.y}) crosses restricted airspace",
                                float(nofly_count),
                            )
                        )
                    if obstacle_count:
                        drone_violations.append(
                            Violation(
                                "obstacle",
                                drone.drone_id,
                                f"leg to ({wp.x}, {wp.y}) crosses blocked terrain",
                                float(obstacle_count),
                            )
                        )
                    valid_samples = np.stack((vx, vy), axis=1)
                    risk += (
                        float(self.env.hazard_map[vy, vx].mean()) * length
                    )
                    radius = sensor_radius(wp.sensor)
                    detected |= self._leg_detections(valid_samples, radius)
                    self._mark_sensor_footprint(
                        coverage_mask, valid_samples, radius
                    )

                distance += length
                previous = (wp.x, wp.y)

            if distance > drone.max_flight_range + 1e-9:
                drone_violations.append(
                    Violation(
                        "max_range",
                        drone.drone_id,
                        f"distance {distance:.2f} exceeds range "
                        f"{drone.max_flight_range:.2f}",
                        distance - drone.max_flight_range,
                    )
                )
            if energy > drone.battery_capacity + 1e-9:
                drone_violations.append(
                    Violation(
                        "battery",
                        drone.drone_id,
                        f"energy {energy:.2f} exceeds capacity "
                        f"{drone.battery_capacity:.2f}",
                        energy - drone.battery_capacity,
                    )
                )

            covered |= detected
            violations.extend(drone_violations)
            per_drone.append(
                DroneEvaluation(
                    drone_id=drone.drone_id,
                    start=drone.start,
                    distance=distance,
                    duration=duration,
                    energy=energy,
                    hazard_exposure=risk,
                    detected_targets=[
                        self.target_ids[i] for i in sorted(detected)
                    ],
                    violations=drone_violations,
                )
            )

        observed_priority = float(
            self.priority_map[coverage_mask].sum()
        )
        coverage = (
            observed_priority / self.total_priority_mass
            if self.total_priority_mass > 0.0
            else 0.0
        )
        covered_cells = int(coverage_mask.sum())
        surveyed_fraction = (
            covered_cells / self.free_cells if self.free_cells else 0.0
        )
        detection_rate = (
            len(covered) / len(self.env.targets)
            if self.env.targets
            else 0.0
        )

        total_energy = float(sum(d.energy for d in per_drone))
        total_distance = float(sum(d.distance for d in per_drone))
        mission_time = max((d.duration for d in per_drone), default=0.0)
        hazard_exposure = float(sum(d.hazard_exposure for d in per_drone))

        normalized_energy = (
            total_energy / self.total_capacity
            if self.total_capacity > 0.0
            else 0.0
        )
        normalized_time = (
            mission_time / self.time_reference
            if self.time_reference > 0.0
            else 0.0
        )
        normalized_risk = (
            hazard_exposure / total_distance
            if total_distance > 0.0
            else 0.0
        )

        raw_fitness = (
            self.weights.coverage * coverage
            + self.weights.detection * detection_rate
            - self.weights.energy * normalized_energy
            - self.weights.time * normalized_time
            - self.weights.risk * normalized_risk
        )
        if violations:
            total_magnitude = sum(min(v.magnitude, 10.0) for v in violations)
            penalty = (
                self.weights.infeasible_base_penalty
                + self.weights.violation_penalty * len(violations)
                + self.weights.violation_penalty * 0.5 * total_magnitude
            )
        else:
            penalty = 0.0
        fitness = raw_fitness - penalty

        return EvaluationResult(
            fitness=float(fitness),
            raw_fitness=float(raw_fitness),
            penalty=float(penalty),
            coverage=float(coverage),
            detection_rate=float(detection_rate),
            total_energy=total_energy,
            normalized_energy=float(normalized_energy),
            mission_time=float(mission_time),
            normalized_time=float(normalized_time),
            total_distance=total_distance,
            hazard_exposure=hazard_exposure,
            normalized_risk=float(normalized_risk),
            covered_targets=[
                self.target_ids[i] for i in sorted(covered)
            ],
            n_targets=len(self.env.targets),
            feasible=not violations,
            violations=violations,
            per_drone=per_drone,
            covered_cells=covered_cells,
            free_cells=self.free_cells,
            surveyed_fraction=float(surveyed_fraction),
            coverage_mask=coverage_mask,
        )
