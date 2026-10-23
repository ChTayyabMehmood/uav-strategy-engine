"""Disaster-surveillance targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Target:
    """A point of interest that a UAV sensor may detect.

    ``priority`` is a relative importance weight in ``(0, 1]``.  Priority-weighted
    coverage is the sum of priorities of detected targets divided by the sum of
    all priorities.
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
