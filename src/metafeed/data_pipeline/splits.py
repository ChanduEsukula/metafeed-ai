"""Deterministic chronological assignment of recommendation events."""

from dataclasses import dataclass
import math

from metafeed.data_pipeline.kuairec import KuaiRecInteraction


@dataclass(frozen=True)
class TemporalBoundaries:
    train_end: float
    validation_end: float

    def __post_init__(self):
        if not (
            math.isfinite(self.train_end)
            and math.isfinite(self.validation_end)
        ):
            raise ValueError("Boundaries must be finite")

        if self.train_end >= self.validation_end:
            raise ValueError(
                "train_end must precede validation_end"
            )


def assign_split(
    event: KuaiRecInteraction,
    boundaries: TemporalBoundaries,
) -> str:
    if event.timestamp < boundaries.train_end:
        return "train"

    if event.timestamp < boundaries.validation_end:
        return "validation"

    return "test"
