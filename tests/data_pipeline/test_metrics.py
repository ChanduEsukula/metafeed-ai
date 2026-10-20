import pytest

from metafeed.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
)


def test_perfect_recall():
    assert recall_at_k([1, 2, 3], {1, 2}, 2) == 1.0


def test_partial_recall():
    assert recall_at_k([1, 3, 4], {1, 2}, 2) == 0.5


def test_no_relevant_items():
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0


def test_perfect_ndcg():
    assert ndcg_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)


def test_partial_ndcg():
    score = ndcg_at_k([3, 1, 2], {1, 2}, 3)
    assert 0 < score < 1


def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, 0)

    with pytest.raises(ValueError):
        ndcg_at_k([1], {1}, -1)
