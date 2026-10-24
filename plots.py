"""All report figures: mission maps, convergence, coverage and sensitivity."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from model import Chromosome, Drone, GridEnvironment
from simulation import EvaluationResult, visited_cells


# ======================================================================
# Mission map
# ======================================================================

"""Mission map: environment, targets, launch cells, and evolved routes."""


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

# ======================================================================
# Convergence plot
# ======================================================================

"""Fitness convergence plot: best and average feasible fitness."""


def plot_convergence(
    history,
    output_path: str | Path,
    title: str = "GA fitness convergence",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generations = [stats.generation for stats in history]
    best = [
        stats.best_feasible_fitness
        if stats.best_feasible_fitness is not None
        else float("nan")
        for stats in history
    ]
    average = [
        stats.average_feasible_fitness
        if stats.average_feasible_fitness is not None
        else float("nan")
        for stats in history
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        generations,
        best,
        color="#1f77b4",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Best feasible fitness",
    )
    ax.plot(
        generations,
        average,
        color="#ff7f0e",
        marker="s",
        markersize=3.0,
        linewidth=1.5,
        linestyle="--",
        label="Mean feasible fitness",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    if all(value != value for value in best):  # all NaN
        ax.text(
            0.5,
            0.5,
            "No feasible solutions were found",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#b00020",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_best_so_far(
    history: list[tuple[int, float | None]],
    output_path: str | Path,
    title: str = "Random search best-so-far fitness",
) -> Path:
    """Plot a baseline's best feasible fitness against candidate count."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluations = [item[0] for item in history]
    values = [
        item[1] if item[1] is not None else float("nan") for item in history
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(evaluations, values, color="#2ca02c", linewidth=1.8)
    ax.set_xlabel("Candidate evaluations")
    ax.set_ylabel("Best feasible fitness so far")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if all(value != value for value in values):
        ax.text(
            0.5,
            0.5,
            "No feasible candidate found",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#b00020",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path

# ======================================================================
# Coverage plots
# ======================================================================

"""Coverage visualisation: priority zones surveyed by the fleet."""


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

# ======================================================================
# Comparison plot
# ======================================================================

"""Comparison chart for the GA versus random-search baseline."""


def plot_algorithm_comparison(
    rows: list[dict],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = sorted({int(row["seed"]) for row in rows})
    methods = ["genetic_algorithm", "random_search"]
    labels = {"genetic_algorithm": "GA", "random_search": "Random search"}
    colors = {"genetic_algorithm": "#1f77b4", "random_search": "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.36
    x = np.arange(len(seeds))

    for offset, method in enumerate(methods):
        fitness = [
            next(
                row["best_fitness"]
                for row in rows
                if row["method"] == method and int(row["seed"]) == seed
            )
            for seed in seeds
        ]
        coverage = [
            next(
                row["coverage"]
                for row in rows
                if row["method"] == method and int(row["seed"]) == seed
            )
            for seed in seeds
        ]
        positions = x + (offset - 0.5) * width
        axes[0].bar(
            positions,
            fitness,
            width=width,
            color=colors[method],
            label=labels[method],
        )
        axes[1].bar(
            positions,
            coverage,
            width=width,
            color=colors[method],
            label=labels[method],
        )

    for ax, ylabel, title in (
        (axes[0], "Best feasible fitness", "Best fitness by seed"),
        (axes[1], "Priority-weighted coverage", "Coverage by seed"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {seed}" for seed in seeds])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle("GA versus random search (equal evaluation budget)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path

# ======================================================================
# Sensitivity plot
# ======================================================================

"""Sensitivity summary plot (one-factor-at-a-time runs only)."""


PARAMETER_ORDER = ["population_size", "mutation_rate", "tournament_size"]


def plot_sensitivity_summary(
    summary: list[dict],
    output_path: str | Path,
) -> Path | None:
    parameters = [
        name
        for name in PARAMETER_ORDER
        if any(row["parameter"] == name for row in summary)
    ]
    if not parameters:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(parameters), figsize=(4.8 * len(parameters), 4.4)
    )
    if len(parameters) == 1:
        axes = [axes]

    for ax, parameter in zip(axes, parameters):
        rows = [row for row in summary if row["parameter"] == parameter]
        rows.sort(key=lambda row: float(row["value"]))
        values = [float(row["value"]) for row in rows]
        means = [row["mean_best_fitness"] for row in rows]
        errors = [row["std_best_fitness"] for row in rows]
        ax.errorbar(
            values,
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            color="#1f77b4",
            linewidth=1.8,
        )
        ax.set_xlabel(parameter.replace("_", " "))
        ax.set_ylabel("Mean best feasible fitness")
        ax.set_title(parameter.replace("_", " ").title())
        ax.grid(alpha=0.25)
        if parameter == "mutation_rate":
            ax.set_xscale("log")

    fig.suptitle("Hyperparameter sensitivity (mean +/- std across seeds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
