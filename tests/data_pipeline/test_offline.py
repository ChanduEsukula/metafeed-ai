
"""Tests for leakage-safe KuaiRec offline evaluation."""

import csv
from pathlib import Path

import pytest

from metafeed.data_pipeline.splits import TemporalBoundaries
from metafeed.evaluation.config import EvaluationConfig
from metafeed.evaluation.offline import evaluate_popularity


def make_config(split="test"):
    return EvaluationConfig(
        source_file=Path("unused.csv"),
        boundaries=TemporalBoundaries(100, 200),
        positive_threshold=1.0,
        split=split,
        k_values=(1, 2),
        random_seed=42,
    )


def write_events(path, events):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "user_id",
            "video_id",
            "play_duration",
            "video_duration",
            "timestamp",
            "watch_ratio",
        ])

        for user, item, timestamp, ratio in events:
            writer.writerow([
                user,
                item,
                1000,
                1000,
                timestamp,
                ratio,
            ])


def test_test_split_history_and_metrics(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (2, 20, 20, 1.0),
        (3, 20, 30, 1.0),
        (1, 20, 210, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.evaluated_users == 1
    assert result.metrics["Recall@1"] == 1.0
    assert result.metrics["NDCG@1"] == 1.0
    assert result.metrics["HitRate@1"] == 1.0

    assert result.split_events == {
        "train": 3,
        "validation": 0,
        "test": 1,
    }


def test_validation_history_excludes_test_repeats(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (2, 20, 20, 1.0),
        (1, 20, 150, 1.0),
        (1, 20, 210, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.evaluated_users == 0
    assert result.empty_ground_truth_users == 1


def test_cold_start_user_reported(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (99, 10, 210, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.cold_start_users == 1
    assert result.evaluated_users == 0


def test_future_only_item_unavailable(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (1, 99, 210, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.unavailable_positive_items == 1
    assert result.empty_ground_truth_users == 1


def test_repeatability(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (2, 20, 20, 1.0),
        (1, 20, 210, 1.0),
    ])

    first = evaluate_popularity(path, make_config())
    second = evaluate_popularity(path, make_config())

    assert first == second


def test_unavailable_positive_pairs_are_distinct(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (1, 99, 210, 1.0),
        (1, 99, 220, 1.0),
        (1, 98, 230, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.unavailable_positive_items == 2
    assert result.empty_ground_truth_users == 1
    assert result.evaluated_users == 0


def test_validation_positive_history_removes_ground_truth(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (2, 20, 20, 1.0),
        (1, 20, 150, 1.0),
        (1, 20, 210, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.evaluated_users == 0
    assert result.empty_ground_truth_users == 1


def test_catalog_contains_training_negative_only_items(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (2, 20, 20, 0.2),
        (1, 20, 210, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.evaluated_users == 1
    assert result.unavailable_positive_items == 0
    assert result.metrics["HitRate@1"] == 1.0


def test_split_boundary_events(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 99, 1.0),
        (2, 20, 99, 1.0),
        (1, 20, 100, 1.0),
        (1, 20, 200, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.split_events == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }

    assert result.evaluated_users == 0
    assert result.empty_ground_truth_users == 1


def test_positive_pair_coverage_accounting(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (1, 10, 10, 1.0),
        (2, 20, 20, 1.0),
        (2, 30, 30, 0.2),
        (1, 20, 150, 1.0),
        (1, 10, 210, 1.0),
        (1, 20, 220, 1.0),
        (1, 30, 230, 1.0),
        (1, 99, 240, 1.0),
        (1, 99, 250, 1.0),
    ])

    result = evaluate_popularity(path, make_config())

    assert result.distinct_target_positive_pairs == 4
    assert result.in_catalog_positive_pairs == 3
    assert result.unavailable_positive_items == 1
    assert result.previously_seen_positive_pairs == 2
    assert result.eligible_positive_pairs == 1
    assert result.evaluated_users == 1

    assert (
        result.distinct_target_positive_pairs
        == result.in_catalog_positive_pairs
        + result.unavailable_positive_items
    )

    assert (
        result.in_catalog_positive_pairs
        == result.previously_seen_positive_pairs
        + result.eligible_positive_pairs
    )
