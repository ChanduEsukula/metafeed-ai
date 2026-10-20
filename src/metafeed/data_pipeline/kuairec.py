"""Validated, memory-bounded reader for KuaiRec interactions."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REQUIRED_COLUMNS = {
    "user_id",
    "video_id",
    "play_duration",
    "video_duration",
    "timestamp",
    "watch_ratio",
}


@dataclass(frozen=True, slots=True)
class KuaiRecInteraction:
    user_id: int
    video_id: int
    play_duration: float
    video_duration: float
    timestamp: float
    watch_ratio: float


class KuaiRecDataError(ValueError):
    """Raised when an interaction record is invalid."""


def parse_interaction(
    row: dict[str, str],
    line_number: int,
) -> KuaiRecInteraction:
    try:
        user_id = int(row["user_id"])
        video_id = int(row["video_id"])
        play = float(row["play_duration"])
        duration = float(row["video_duration"])
        timestamp = float(row["timestamp"])
        ratio = float(row["watch_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise KuaiRecDataError(
            f"Invalid record at CSV line {line_number}"
        ) from exc

    values = (play, duration, timestamp, ratio)

    if user_id < 0 or video_id < 0:
        raise KuaiRecDataError(
            f"Negative identifier at CSV line {line_number}"
        )

    if not all(math.isfinite(value) for value in values):
        raise KuaiRecDataError(
            f"Non-finite value at CSV line {line_number}"
        )

    if play < 0 or duration <= 0 or ratio < 0:
        raise KuaiRecDataError(
            f"Invalid duration or ratio at CSV line {line_number}"
        )

    return KuaiRecInteraction(
        user_id=user_id,
        video_id=video_id,
        play_duration=play,
        video_duration=duration,
        timestamp=timestamp,
        watch_ratio=ratio,
    )


def iter_kuairec(
    path: Path,
    limit: int | None = None,
) -> Iterator[KuaiRecInteraction]:
    """Stream validated records without materializing the full CSV."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise KuaiRecDataError(
                f"Missing required columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            if limit is not None and line_number - 2 >= limit:
                break

            yield parse_interaction(row, line_number)
