
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
