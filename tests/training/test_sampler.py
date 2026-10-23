
"""Tests for deterministic recommendation batching."""

from collections import Counter

import pytest

from metafeed.training.sampler import (
    batch_diagnostics,
    iter_training_batches,
)


def sample_pairs():
    return [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 40),
        (3, 20),
        (3, 50),
        (1, 10),
    ]


def test_all_events_preserved():
    pairs = sample_pairs()
    batches = list(iter_training_batches(pairs, 3, seed=42))

    flattened = [pair for batch in batches for pair in batch]

    assert Counter(flattened) == Counter(pairs)


def test_unique_items_per_batch():
    batches = iter_training_batches(sample_pairs(), 3, 42)

    for batch in batches:
        items = [item for _, item in batch]
        assert len(items) == len(set(items))


def test_repeated_users_allowed():
    pairs = [(1, 10), (1, 20), (1, 30)]
    batches = list(iter_training_batches(pairs, 3, 42))

    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_same_seed_reproducible():
    first = list(iter_training_batches(sample_pairs(), 3, 42))
    second = list(iter_training_batches(sample_pairs(), 3, 42))

    assert first == second


def test_batch_capacity_respected():
    batches = iter_training_batches(sample_pairs(), 2, 42)
    assert all(0 < len(batch) <= 2 for batch in batches)


def test_single_item_still_preserves_events():
    pairs = [(1, 10), (2, 10), (3, 10)]

    batches = list(iter_training_batches(pairs, 8, 42))

    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert Counter(pair for b in batches for pair in b) == Counter(pairs)


def test_empty_dataset():
    assert list(iter_training_batches([], 4, 42)) == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        list(iter_training_batches(sample_pairs(), batch_size, 42))


def test_invalid_epoch():
    with pytest.raises(ValueError):
        list(iter_training_batches(sample_pairs(), 2, 42, epoch=-1))


def test_batch_diagnostics():
    batches = [
        [(1, 10), (1, 20), (2, 30)],
    ]
    positives = {
        1: frozenset({10, 20}),
        2: frozenset({30}),
    }

    result = batch_diagnostics(batches, positives)

    assert result.interactions == 3
    assert result.batches == 1
    assert result.mean_batch_size == 3
    assert result.valid_negative_fraction == pytest.approx(4 / 6)


def test_diagnostics_missing_target():
    with pytest.raises(ValueError, match="Target missing"):
        batch_diagnostics(
            [[(1, 10)]],
            {1: frozenset({20})},
        )


def test_empty_diagnostics():
    result = batch_diagnostics([], {})

    assert result.interactions == 0
    assert result.batches == 0
    assert result.valid_negative_fraction == 0
