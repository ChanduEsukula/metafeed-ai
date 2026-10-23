
"""Tests for KuaiRec PyTorch training data."""

import csv

import pytest

from metafeed.data_pipeline.splits import TemporalBoundaries
from metafeed.training.kuairec_dataset import (
    KuaiRecPositiveDataset,
    iter_unique_positive_batches,
    load_training_data,
)


BOUNDARIES = TemporalBoundaries(100, 200)


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
            writer.writerow(
                [user, item, 1000, 1000, timestamp, ratio]
            )


@pytest.fixture
def training_data(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (20, 200, 10, 1.0),
        (10, 100, 20, 1.2),
        (20, 200, 30, 1.5),
        (30, 300, 40, 0.2),
        (40, 400, 100, 1.0),
        (50, 500, 200, 1.0),
    ])

    return load_training_data(path, BOUNDARIES)


def test_training_only_pairs(training_data):
    assert len(training_data.positive_pairs) == 3


def test_deterministic_user_mapping(training_data):
    assert training_data.user_to_index == {
        10: 0,
        20: 1,
        30: 2,
    }


def test_deterministic_item_mapping(training_data):
    assert training_data.item_to_index == {
        100: 0,
        200: 1,
        300: 2,
    }


def test_repeated_positive_events_preserved(training_data):
    assert training_data.positive_pairs.count((1, 1)) == 2


def test_negative_only_item_in_catalog(training_data):
    assert 300 in training_data.item_to_index


def test_dataset_length(training_data):
    dataset = KuaiRecPositiveDataset(training_data)
    assert len(dataset) == 3


def test_dataset_getitem(training_data):
    dataset = KuaiRecPositiveDataset(training_data)
    assert dataset[0] == (1, 1)


def test_unique_batch_identifiers(training_data):
    batches = list(
        iter_unique_positive_batches(
            training_data,
            batch_size=3,
        )
    )

    for users, items in batches:
        assert len(users) == len(set(users.tolist()))
        assert len(items) == len(set(items.tolist()))


def test_all_training_pairs_emitted(training_data):
    batches = list(
        iter_unique_positive_batches(
            training_data,
            batch_size=3,
        )
    )

    emitted = [
        pair
        for users, items in batches
        for pair in zip(users.tolist(), items.tolist())
    ]

    assert emitted == list(training_data.positive_pairs)


def test_invalid_batch_size(training_data):
    with pytest.raises(ValueError):
        list(iter_unique_positive_batches(training_data, 0))


def test_reproducible_mapping(tmp_path):
    path = tmp_path / "events.csv"

    write_events(path, [
        (20, 200, 10, 1.0),
        (10, 100, 20, 1.0),
    ])

    first = load_training_data(path, BOUNDARIES)
    second = load_training_data(path, BOUNDARIES)

    assert first == second


def test_reservoir_reproducible_and_not_prefix(tmp_path):
    from metafeed.training.kuairec_dataset import sample_training_data
    path = tmp_path / "events.csv"
    events = [(u, u, u, 1.0) for u in range(100)]
    write_events(path, events)
    first = sample_training_data(path, BOUNDARIES, 10, 42)
    second = sample_training_data(path, BOUNDARIES, 10, 42)
    third = sample_training_data(path, BOUNDARIES, 10, 43)
    assert first == second
    assert first.data.positive_pairs != third.data.positive_pairs
    assert first.data.positive_pairs != tuple((u, u) for u in range(10))
    assert len(first.data.positive_pairs) == 10
    assert first.diagnostics["training_positive_events"] == 100
    assert first.diagnostics["known_positive_pairs"] == 100
    assert first.diagnostics["sampled_users"] == 10


def test_complete_history_and_maps_ignore_heldout(tmp_path):
    from metafeed.training.kuairec_dataset import sample_training_data
    path = tmp_path / "events.csv"
    write_events(path, [
        (1, 10, 99, 1), (1, 20, 1, 1), (2, 30, 90, 1),
        (3, 40, 80, 0.5), (999, 999, 100, 1), (888, 888, 200, 1),
        (1, 999, 120, 1),
    ])
    sampled = sample_training_data(path, BOUNDARIES, 1, 42)
    assert sampled.data.user_to_index == {1: 0, 2: 1, 3: 2}
    assert sampled.data.item_to_index == {10: 0, 20: 1, 30: 2, 40: 3}
    assert sampled.known_positives == {0: frozenset({0, 1}), 1: frozenset({2})}
    assert len(sampled.data.positive_pairs) == 1
    assert sampled.diagnostics["training_events"] == 4
    assert sampled.diagnostics["known_positive_pairs"] == 3


def test_reservoir_keeps_duplicate_events_and_days(tmp_path):
    from collections import Counter
    from metafeed.training.kuairec_dataset import sample_training_data
    path = tmp_path / "events.csv"
    write_events(path, [(1, 10, 1, 1), (1, 10, 1, 1), (2, 20, 90000, 1)])
    sampled = sample_training_data(path, TemporalBoundaries(100000, 200000), 10, 42)
    assert Counter(sampled.data.positive_pairs) == Counter({(0, 0): 2, (1, 1): 1})
    assert sampled.diagnostics["sampled_events_by_utc_day"] == {0: 2, 1: 1}
    assert sampled.diagnostics["training_positive_events_by_utc_day"] == {0: 2, 1: 1}
    assert sampled.diagnostics["known_positive_pairs"] == 2


@pytest.mark.parametrize("max_events,threshold", [(0, 1), (1, float("nan")), (1, -1)])
def test_reservoir_rejects_invalid_parameters(tmp_path, max_events, threshold):
    from metafeed.training.kuairec_dataset import sample_training_data
    with pytest.raises(ValueError):
        sample_training_data(tmp_path / "unused.csv", BOUNDARIES, max_events, 42, threshold)


def test_full_catalog_matches_popularity_baseline(tmp_path):
    from metafeed.data_pipeline.kuairec import iter_kuairec
    from metafeed.evaluation.popularity import PopularityBaseline
    from metafeed.training.kuairec_dataset import sample_training_data
    path = tmp_path / "events.csv"
    write_events(path, [(1, 10, 1, 1), (2, 20, 2, 0), (3, 30, 100, 1)])
    baseline = PopularityBaseline()
    baseline.fit(iter_kuairec(path), BOUNDARIES)
    sampled = sample_training_data(path, BOUNDARIES, 1, 42)
    assert set(sampled.data.item_to_index) == baseline.catalog
    assert set(sampled.data.user_to_index) == baseline.training_users
