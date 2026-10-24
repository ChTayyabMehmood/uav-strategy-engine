"""Mission map: environment, targets, launch cells, and evolved routes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from environment.grid import GridEnvironment
from simulation.evaluator import EvaluationResult
from uav.chromosome import Chromosome
from uav.drone import Drone

DRONE_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]


def route_coordinates(
    start: tuple[int, int], route
) -> list[tuple[int, int]]:
    return [start] + [(wp.x, wp.y) for wp in route.waypoints]


def plot_mission_map(
    env: GridEnvironment,
    chromosome: Chromosome,
    evaluation: EvaluationResult,
    drones: list[Drone],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = (-0.5, env.width - 0.5, env.height - 0.5, -0.5)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    hazard_image = ax.imshow(
        env.hazard_map,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        alpha=0.55,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        zorder=1,
    )

    overlay = np.zeros((env.height, env.width, 4), dtype=float)
    overlay[env.obstacle_mask] = (0.12, 0.12, 0.14, 0.85)
    overlay[env.nofly_mask] = (0.86, 0.10, 0.10, 0.60)
    ax.imshow(
        overlay,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        zorder=2,
    )

    detected = set(evaluation.covered_targets)
    starts = (
        [item.start for item in evaluation.per_drone]
        if evaluation.per_drone
        else [drone.start for drone in drones]
    )
    for drone_id, (start, route, color) in enumerate(
        zip(starts, chromosome.routes, DRONE_COLORS * 2)
    ):
        coordinates = route_coordinates(start, route)
        xs, ys = zip(*coordinates)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=3.2,
            alpha=0.95,
            zorder=4,
            label=f"Drone {drone_id}",
        )
        ax.scatter(
            [start[0]],
            [start[1]],
            marker="^",
            s=120,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )

    for target in env.targets:
        is_detected = target.target_id in detected
        ax.scatter(
            [target.x],
            [target.y],
            marker="*",
            s=70 + 150 * target.priority,
            color="#25a244" if is_detected else "#f2f2f2",
            edgecolors="black",
            linewidths=0.7,
            zorder=7,
        )
        label_offset = (3, -10) if target.y < 1.5 else (3, 3)
        ax.annotate(
            target.target_id,
            (target.x, target.y),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=6.5,
            color="black",
            zorder=8,
        )

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(0, env.width, 5))
    ax.set_yticks(range(0, env.height, 5))
    ax.grid(color="white", alpha=0.15, linewidth=0.5)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    ax.set_aspect("equal")

    handles = [
        Patch(facecolor="#1f1f24", label="Obstacle"),
        Patch(facecolor="#db1a1a", alpha=0.7, label="No-fly zone"),
        Patch(facecolor=plt.cm.YlOrRd(0.65), alpha=0.7, label="Hazard"),
        Line2D(
            [],
            [],
            marker="*",
            color="none",
            markerfacecolor="#25a244",
            markeredgecolor="black",
            markersize=13,
            label="Detected target",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color="none",
            markerfacecolor="#f2f2f2",
            markeredgecolor="black",
            markersize=13,
            label="Undetected target",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color="gray",
            linestyle="none",
            markersize=9,
            label="Deployment cell",
        ),
    ]
    for drone_id, color in enumerate(
        DRONE_COLORS[: len(chromosome.routes)]
    ):
        handles.append(
            Line2D([], [], color=color, label=f"Drone {drone_id}")
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.85,
        fontsize=7.5,
        ncol=2,
    )

    fig.colorbar(
        hazard_image,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        label="Hazard intensity",
    )

    if title is None:
        title = (
            "Evolved UAV mission - "
            f"fitness {evaluation.fitness:.2f}"
        )
    ax.set_title(title, fontsize=12, pad=10)
    stats = (
        f"coverage {evaluation.coverage:.1%} | "
        f"detection {evaluation.detection_rate:.1%} | "
        f"energy {evaluation.total_energy:.1f} | "
        f"time {evaluation.mission_time:.1f} | "
        f"risk {evaluation.normalized_risk:.2f} | "
        f"feasible: {evaluation.feasible}"
    )
    fig.text(0.5, 0.015, stats, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
