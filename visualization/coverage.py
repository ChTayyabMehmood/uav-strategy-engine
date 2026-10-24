"""Coverage visualisation: priority zones surveyed by the fleet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment.grid import GridEnvironment
from simulation.evaluator import EvaluationResult
from simulation.repair import visited_cells
from uav.chromosome import Chromosome
from uav.drone import Drone

ROUTE_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]


def _starts(
    drones: list[Drone], evaluation: EvaluationResult
) -> list[tuple[int, int]]:
    if evaluation.per_drone:
        return [item.start for item in evaluation.per_drone]
    return [drone.start for drone in drones]


def _draw_routes(
    ax: plt.Axes,
    chromosome: Chromosome,
    starts: list[tuple[int, int]],
    alpha: float = 0.9,
    linewidth: float = 1.0,
) -> None:
    for drone_id, (route, start, color) in enumerate(
        zip(chromosome.routes, starts, ROUTE_COLORS * 2)
    ):
        xs = [start[0]] + [wp.x for wp in route.waypoints]
        ys = [start[1]] + [wp.y for wp in route.waypoints]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            marker="o",
            markersize=1.8,
            label=f"Drone {drone_id}",
        )
        ax.scatter(
            [start[0]],
            [start[1]],
            marker="^",
            s=38,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
        )


def plot_coverage_overview(
    env: GridEnvironment,
    chromosome: Chromosome,
    evaluation: EvaluationResult,
    drones: list[Drone],
    output_path: str | Path,
    title: str = "Survey coverage",
) -> Path:
    """Plot the priority-zone coverage map and the detected-target map."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = (-0.5, env.width - 0.5, env.height - 0.5, -0.5)
    starts = _starts(drones, evaluation)
    mask = evaluation.coverage_mask
    if mask is None:
        mask = np.zeros((env.height, env.width), dtype=bool)

    visit_counts = np.zeros((env.height, env.width), dtype=float)
    for start, route in zip(starts, chromosome.routes):
        for x, y in visited_cells(env, start, route):
            if env.in_bounds(x, y):
                visit_counts[y, x] += 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))

    priority = axes[0].imshow(
        env.priority_map,
        cmap="Greys",
        origin="upper",
        extent=extent,
        alpha=0.55,
        interpolation="nearest",
    )
    overlay = np.zeros((env.height, env.width, 4), dtype=float)
    overlay[mask] = (0.16, 0.68, 0.30, 0.45)
    axes[0].imshow(
        overlay,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        zorder=2,
    )
    _draw_routes(axes[0], chromosome, starts)
    axes[0].set_title("Priority-zone coverage (green = surveyed)", fontsize=10.5)
    axes[0].set_xlabel("x (column)")
    axes[0].set_ylabel("y (row)")
    axes[0].set_aspect("equal")
    fig.colorbar(priority, ax=axes[0], fraction=0.046, pad=0.03, label="Priority")

    visit_image = axes[1].imshow(
        visit_counts,
        cmap="viridis",
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )
    detected = set(evaluation.covered_targets)
    for target in env.targets:
        is_detected = target.target_id in detected
        axes[1].scatter(
            [target.x],
            [target.y],
            marker="*",
            s=70 + 150 * target.priority,
            color="#25a244" if is_detected else "#c62828",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )
    axes[1].set_title("Cell visitation density and target status", fontsize=10.5)
    axes[1].set_xlabel("x (column)")
    axes[1].set_ylabel("y (row)")
    axes[1].set_xlim(-0.5, env.width - 0.5)
    axes[1].set_ylim(env.height - 0.5, -0.5)
    axes[1].set_aspect("equal")
    fig.colorbar(visit_image, ax=axes[1], fraction=0.046, pad=0.03, label="Visits")

    fig.suptitle(
        f"{title} - {evaluation.coverage:.1%} priority-zone coverage",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_coverage_evolution(
    env: GridEnvironment,
    chromosomes: list[Chromosome],
    evaluations: list[EvaluationResult],
    output_path: str | Path,
    title: str = "Coverage across generations",
) -> Path:
    """Show how the best route's surveyed priority area evolves."""

    if len(chromosomes) != len(evaluations):
        raise ValueError("chromosomes and evaluations must be aligned")
    if not chromosomes:
        raise ValueError("at least one generation is required")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(chromosomes)
    indices = sorted({0, n // 3, (2 * n) // 3, n - 1})
    extent = (-0.5, env.width - 0.5, env.height - 0.5, -0.5)

    fig, axes = plt.subplots(1, len(indices), figsize=(12.0, 3.2))
    if len(indices) == 1:
        axes = [axes]
    for ax, index in zip(axes, indices):
        evaluation = evaluations[index]
        mask = evaluation.coverage_mask
        if mask is None:
            mask = np.zeros((env.height, env.width), dtype=bool)
        ax.imshow(
            env.priority_map,
            cmap="Greys",
            origin="upper",
            extent=extent,
            alpha=0.55,
            interpolation="nearest",
        )
        overlay = np.zeros((env.height, env.width, 4), dtype=float)
        overlay[mask] = (0.16, 0.68, 0.30, 0.50)
        ax.imshow(
            overlay,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            zorder=2,
        )
        starts = _starts([], evaluation)
        _draw_routes(ax, chromosomes[index], starts, linewidth=0.8)
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(env.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Gen {index} | {evaluation.coverage:.1%}",
            fontsize=9,
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
