"""Sensitivity summary plot (one-factor-at-a-time runs only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

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
