
"""Leakage-safe KuaiRec popularity evaluation."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.data_pipeline.splits import assign_split
from metafeed.evaluation.config import EvaluationConfig
from metafeed.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)
from metafeed.evaluation.popularity import PopularityBaseline


@dataclass(frozen=True)
class EvaluationResult:
    evaluated_users: int
    cold_start_users: int
    empty_ground_truth_users: int
    unavailable_positive_items: int

    distinct_target_positive_pairs: int
    in_catalog_positive_pairs: int
    eligible_positive_pairs: int
    previously_seen_positive_pairs: int

    split_events: dict[str, int]
    metrics: dict[str, float]


class RankingModel(Protocol):
    catalog: set[int]
    training_users: set[int]

    def recommend(self, user_id: int, k: int, excluded_items: set[int]) -> list[int]: ...


def evaluate_popularity(
    path: Path,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate using training-only scores and complete earlier history."""

    model = PopularityBaseline()

    model.fit(
        iter_kuairec(path),
        config.boundaries,
        config.positive_threshold,
    )

    return evaluate_ranker(path, config, model)


def evaluate_ranker(
    path: Path,
    config: EvaluationConfig,
    model: RankingModel,
) -> EvaluationResult:
    """Shared population, exclusions and metrics for fitted ranking models.

    The caller must verify catalog/users come from the complete training split.
    This is the original popularity evaluation loop, unchanged.
    """
    histories: dict[int, set[int]] = defaultdict(set)
    ground_truth: dict[int, set[int]] = defaultdict(set)

    target_users: set[int] = set()
    unavailable: set[tuple[int, int]] = set()

    split_events = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    # ---------------------------------------------------------
    # Pass 2: Build historical exclusions and held-out targets
    # ---------------------------------------------------------

    for event in iter_kuairec(path):
        split = assign_split(event, config.boundaries)
        split_events[split] += 1

        is_positive = config.is_positive(event.watch_ratio)

        if split == "train":
            if is_positive:
                histories[event.user_id].add(event.video_id)
            continue

        if config.split == "test" and split == "validation":
            if is_positive:
                histories[event.user_id].add(event.video_id)
            continue

        if split != config.split:
            continue

        target_users.add(event.user_id)

        if not is_positive:
            continue

        if event.video_id not in model.catalog:
            unavailable.add(
                (event.user_id, event.video_id)
            )
        else:
            ground_truth[event.user_id].add(
                event.video_id
            )

    # ---------------------------------------------------------
    # Evaluation coverage accounting
    # ---------------------------------------------------------

    in_catalog_positive_pairs = sum(
        len(items)
        for items in ground_truth.values()
    )

    distinct_target_positive_pairs = (
        len(unavailable)
        + in_catalog_positive_pairs
    )

    previously_seen_positive_pairs = sum(
        len(
            items & histories.get(user_id, set())
        )
        for user_id, items in ground_truth.items()
    )

    eligible_positive_pairs = (
        in_catalog_positive_pairs
        - previously_seen_positive_pairs
    )

    # ---------------------------------------------------------
    # Initialize ranking metric accumulators
    # ---------------------------------------------------------

    totals = {
        f"{metric}@{k}": 0.0
        for k in config.k_values
        for metric in (
            "Recall",
            "NDCG",
            "HitRate",
        )
    }

    evaluated = 0
    cold_start = 0
    empty_ground_truth = 0

    # ---------------------------------------------------------
    # Evaluate eligible users
    # ---------------------------------------------------------

    for user_id in sorted(target_users):

        if user_id not in model.training_users:
            cold_start += 1
            continue

        excluded = histories.get(
            user_id,
            set(),
        )

        relevant = (
            ground_truth.get(user_id, set())
            - excluded
        )

        if not relevant:
            empty_ground_truth += 1
            continue

        recommendations = model.recommend(
            user_id,
            k=max(config.k_values),
            excluded_items=excluded,
        )

        evaluated += 1

        for k in config.k_values:
            totals[f"Recall@{k}"] += recall_at_k(
                recommendations,
                relevant,
                k,
            )

            totals[f"NDCG@{k}"] += ndcg_at_k(
                recommendations,
                relevant,
                k,
            )

            totals[f"HitRate@{k}"] += hit_rate_at_k(
                recommendations,
                relevant,
                k,
            )

    # ---------------------------------------------------------
    # Macro-average metrics across evaluated users
    # ---------------------------------------------------------

    metrics = {
        name: (
            total / evaluated
            if evaluated
            else 0.0
        )
        for name, total in totals.items()
    }

    # ---------------------------------------------------------
    # Return evaluation results
    # ---------------------------------------------------------

    return EvaluationResult(
        evaluated_users=evaluated,
        cold_start_users=cold_start,
        empty_ground_truth_users=empty_ground_truth,
        unavailable_positive_items=len(unavailable),
        distinct_target_positive_pairs=(
            distinct_target_positive_pairs
        ),
        in_catalog_positive_pairs=(
            in_catalog_positive_pairs
        ),
        eligible_positive_pairs=(
            eligible_positive_pairs
        ),
        previously_seen_positive_pairs=(
            previously_seen_positive_pairs
        ),
        split_events=split_events,
        metrics=metrics,
    )
