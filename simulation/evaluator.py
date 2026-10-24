"""Mission evaluator: simulate a chromosome and compute all objectives.

The evaluator is deliberately separate from the GA.  It returns a structured
result containing every metric, every constraint violation, and an independent
feasibility flag.  A chromosome is never treated as feasible merely because its
scalar fitness is high.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from config import FitnessWeights
from environment.grid import GridEnvironment
from simulation.constraints import segment_distance, segment_samples
from uav.chromosome import Chromosome, fleet_for_chromosome
from uav.drone import (
    CRUISE_SPEED,
    MOVE_ENERGY_PER_CELL,
    SPEED_DRAG_EXPONENT,
    Drone,
    sensor_energy_rate,
    sensor_radius,
    speed_of,
)


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
