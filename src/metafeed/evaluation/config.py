
"""Configuration for reproducible KuaiRec offline evaluation."""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from metafeed.data_pipeline.splits import TemporalBoundaries


@dataclass(frozen=True)
class EvaluationConfig:
    source_file: Path
    boundaries: TemporalBoundaries
    positive_threshold: float
    split: str
    k_values: tuple[int, ...]
    random_seed: int

    def is_positive(self, watch_ratio: float) -> bool:
        return (
            math.isfinite(watch_ratio)
            and watch_ratio >= self.positive_threshold
        )


def _utc_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if parsed.tzinfo is None:
        raise ValueError("Evaluation boundaries must include a timezone")

    return parsed.astimezone(timezone.utc).timestamp()


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load and validate the KuaiRec evaluation configuration."""

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if data["split_strategy"] != "global_chronological":
        raise ValueError("Unsupported split strategy")

    target = data["target_definition"]

    if target["field"] != "watch_ratio" or target["operator"] != ">=":
        raise ValueError("Unsupported target definition")

    threshold = float(target["threshold"])

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Positive threshold must be finite and nonnegative")

    evaluation = data["evaluation"]

    if evaluation["split"] not in {"validation", "test"}:
        raise ValueError("Unsupported evaluation split")

    expected_policies = {
        "candidate_catalog": "training_observed_items",
        "history_exclusion": "prior_positive_interactions",
        "cold_start_policy": "exclude_and_report",
        "missing_user_policy": "exclude_and_report",
        "negative_sampling": "none",
    }

    for key, expected in expected_policies.items():
        if evaluation[key] != expected:
            raise ValueError(f"Unsupported evaluation policy: {key}")

    k_values = tuple(evaluation["k_values"])

    if (
        not k_values
        or any(type(k) is not int or k <= 0 for k in k_values)
        or len(set(k_values)) != len(k_values)
    ):
        raise ValueError("k_values must contain unique positive integers")

    seed = evaluation["random_seed"]

    if type(seed) is not int or seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")

    return EvaluationConfig(
        source_file=Path(data["source_file"]),
        boundaries=TemporalBoundaries(
            train_end=_utc_timestamp(data["train_end_utc"]),
            validation_end=_utc_timestamp(data["validation_end_utc"]),
        ),
        positive_threshold=threshold,
        split=evaluation["split"],
        k_values=k_values,
        random_seed=seed,
    )
