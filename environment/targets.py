"""Disaster-surveillance targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Target:
    """A point of interest that a UAV sensor may detect.

    ``priority`` is a relative importance weight in ``(0, 1]``.  Each target
    also owns a Gaussian priority zone used by the area-coverage metric.
    """

    target_id: str
    x: int
    y: int
    priority: float

    def to_dict(self) -> dict:
        return asdict(self)


def total_priority(targets: list[Target]) -> float:
    """Return the summed priority of ``targets`` (0.0 for an empty list)."""

    return float(sum(t.priority for t in targets))


def build_priority_map(
    width: int,
    height: int,
    targets: list[Target],
    radius: float = 3.0,
) -> np.ndarray:
    """Return a normalised priority-zone map around the targets.

    Each target owns a circular priority zone.  The zone is normalised to one
    unit of mass before it is multiplied by the target priority, so edge
    truncation does not change the target's importance.  A sensor footprint
    covering the whole zone therefore contributes exactly its full priority.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    yy, xx = np.ogrid[:height, :width]
    field = np.zeros((height, width), dtype=float)
    for target in targets:
        zone = (
            (xx - target.x) ** 2 + (yy - target.y) ** 2
            <= radius * radius
        )
        cells = int(zone.sum())
        if cells:
            field[zone] += target.priority / cells
    return field
