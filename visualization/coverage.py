"""Coverage visualisation: surveyed cells and detected vs missed targets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment.grid import GridEnvironment
from simulation.evaluator import EvaluationResult
from simulation.repair import visited_cells
from uav.chromosome import Chromosome
from uav.drone import Drone


def plot_coverage_overview(
    env: GridEnvironment,
    chromosome: Chromosome,
    evaluation: EvaluationResult,
    drones: list[Drone],
    output_path: str | Path,
    title: str = "Survey coverage",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = (-0.5, env.width - 0.5, env.height - 0.5, -0.5)

    visit_counts = np.zeros((env.height, env.width), dtype=float)
    for drone, route in zip(drones, chromosome.routes):
        for x, y in visited_cells(env, drone.start, route):
            if env.in_bounds(x, y):
                visit_counts[y, x] += 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    image = axes[0].imshow(
        visit_counts,
        cmap="viridis",
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )
    axes[0].set_title("Cell visitation density (sampled route cells)")
    axes[0].set_xlabel("x (column)")
    axes[0].set_ylabel("y (row)")
    axes[0].set_aspect("equal")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03, label="Visits")

    axes[1].imshow(
        env.hazard_map,
        cmap="Greys",
        alpha=0.35,
        origin="upper",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    detected = set(evaluation.covered_targets)
    for target in env.targets:
        is_detected = target.target_id in detected
        axes[1].scatter(
            [target.x],
            [target.y],
            marker="*",
            s=90 + 170 * target.priority,
            color="#25a244" if is_detected else "#c62828",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        label_offset = (3, -10) if target.y < 1.5 else (3, 3)
        axes[1].annotate(
            target.target_id,
            (target.x, target.y),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=6.5,
        )
    for drone, route in zip(drones, chromosome.routes):
        xs = [drone.start[0]] + [wp.x for wp in route.waypoints]
        ys = [drone.start[1]] + [wp.y for wp in route.waypoints]
        axes[1].plot(xs, ys, color="#1f77b4", linewidth=0.9, alpha=0.7)
    axes[1].set_title(
        "Target status (green = detected, red = missed)",
        fontsize=10.5,
    )
    axes[1].set_xlabel("x (column)")
    axes[1].set_ylabel("y (row)")
    axes[1].set_xlim(-0.5, env.width - 0.5)
    axes[1].set_ylim(env.height - 0.5, -1.2)
    axes[1].set_aspect("equal")

    fig.suptitle(
        f"{title} - {evaluation.coverage:.1%} priority-weighted coverage",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
