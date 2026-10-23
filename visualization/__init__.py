"""Visualisation package.

Matplotlib is forced to the non-interactive Agg backend because every figure is
written to a file (this also keeps headless runs safe).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from visualization.comparison import plot_algorithm_comparison
from visualization.convergence import plot_best_so_far, plot_convergence
from visualization.coverage import plot_coverage_overview
from visualization.mission_map import DRONE_COLORS, plot_mission_map
from visualization.sensitivity import plot_sensitivity_summary

__all__ = [
    "DRONE_COLORS",
    "plot_algorithm_comparison",
    "plot_best_so_far",
    "plot_convergence",
    "plot_coverage_overview",
    "plot_mission_map",
    "plot_sensitivity_summary",
]
