from pathlib import Path

import pytest

from metafeed.recommender import (
    RecommendationEngine,
    load_dataset,
)
from metafeed.schemas import Dataset


SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "src/metafeed/data/sample.json"
)


@pytest.fixture
def engine():
    return RecommendationEngine(load_dataset(SAMPLE))


def test_popularity_ranking(engine):
    results = engine.recommend("alice")

    assert [
        (item.item_id, item.score)
        for item in results
    ] == [
        ("video_003", 2),
        ("video_004", 1),
        ("video_005", 0),
        ("video_006", 0),
    ]


def test_cold_start_user(engine):
    results = engine.recommend("eve")
    assert results[0].item_id == "video_001"
    assert results[0].score == 3


def test_limit(engine):
    assert len(engine.recommend("alice", 2)) == 2


def test_unknown_user(engine):
    with pytest.raises(KeyError):
        engine.recommend("unknown")


def test_invalid_limits(engine):
    for value in (0, -1, 101):
        with pytest.raises(ValueError):
            engine.recommend("alice", value)


def test_stable_tie_order(engine):
    results = engine.recommend("eve")
    assert results[1].item_id == "video_002"
    assert results[2].item_id == "video_003"
    assert results == engine.recommend("eve")


def test_exhausted_catalog():
    dataset = Dataset.model_validate({
        "users": ["a"],
        "items": ["x"],
        "interactions": [
            {"user_id": "a", "item_id": "x"}
        ],
    })
    assert RecommendationEngine(dataset).recommend("a") == []


def test_duplicate_interactions_rejected():
    with pytest.raises(ValueError):
        Dataset.model_validate({
            "users": ["a"],
            "items": ["x"],
            "interactions": [
                {"user_id": "a", "item_id": "x"},
                {"user_id": "a", "item_id": "x"},
            ],
        })


def test_unknown_interaction_user_rejected():
    with pytest.raises(ValueError):
        Dataset.model_validate({
            "users": ["a"],
            "items": ["x"],
            "interactions": [
                {"user_id": "unknown", "item_id": "x"}
            ],
        })
