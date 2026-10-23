
import math

import pytest

from metafeed.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
)


# ============================================================
# Recall@K Tests
# ============================================================

def test_perfect_recall():
    assert recall_at_k([1, 2, 3], {1, 2}, 2) == 1.0


def test_partial_recall():
    assert recall_at_k([1, 3, 4], {1, 2}, 2) == 0.5


# ============================================================
# NDCG@K Tests
# ============================================================

def test_perfect_ndcg():
    assert ndcg_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)


def test_partial_ndcg():
    score = ndcg_at_k([3, 1, 2], {1, 2}, 3)
    assert 0 < score < 1


def test_ndcg_exact_ordering():
    expected = (
        1 / math.log2(3)
        + 1 / math.log2(4)
    ) / (
        1 + 1 / math.log2(3)
    )

    assert ndcg_at_k([3, 1, 2], {1, 2}, 3) == pytest.approx(expected)


def test_ndcg_is_bounded():
    assert 0 <= ndcg_at_k([3, 1, 2], {1, 2}, 3) <= 1


# ============================================================
# HitRate@K Tests
# ============================================================

def test_hit_rate_with_relevant_recommendation():
    assert hit_rate_at_k([5, 10, 30], {10, 20}, 3) == 1.0


def test_hit_rate_without_relevant_recommendation():
    assert hit_rate_at_k([5, 6, 7], {10, 20}, 3) == 0.0


def test_hit_rate_respects_k():
    assert hit_rate_at_k([5, 10, 30], {10}, 1) == 0.0
    assert hit_rate_at_k([5, 10, 30], {10}, 2) == 1.0


def test_hit_rate_empty_relevant_set():
    assert hit_rate_at_k([1, 2, 3], set(), 3) == 0.0


def test_hit_rate_rejects_invalid_k():
    with pytest.raises(ValueError):
        hit_rate_at_k([1], {1}, 0)


def test_hit_rate_rejects_duplicate_recommendations():
    with pytest.raises(ValueError, match="unique"):
        hit_rate_at_k([1, 1], {1}, 2)


def test_hit_rate_empty_recommendations():
    assert hit_rate_at_k([], {1, 2}, 10) == 0.0


# ============================================================
# Shared Edge Cases
# ============================================================

def test_no_relevant_items():
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, 0)

    with pytest.raises(ValueError):
        ndcg_at_k([1], {1}, -1)


@pytest.mark.parametrize(
    "metric",
    [recall_at_k, ndcg_at_k, hit_rate_at_k],
)
def test_duplicate_recommendations_rejected(metric):
    with pytest.raises(ValueError, match="unique"):
        metric([1, 1], {1}, 2)
