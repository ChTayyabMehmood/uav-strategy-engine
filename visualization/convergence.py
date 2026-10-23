"""Fitness convergence plot: best and average feasible fitness."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


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
