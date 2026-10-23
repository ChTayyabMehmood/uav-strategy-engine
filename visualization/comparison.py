"""Comparison chart for the GA versus random-search baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
