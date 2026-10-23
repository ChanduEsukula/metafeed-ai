
from dataclasses import replace

import pytest

from metafeed.data_pipeline.kuairec import KuaiRecInteraction
from metafeed.data_pipeline.splits import TemporalBoundaries
from metafeed.evaluation.popularity import PopularityBaseline


BOUNDARIES = TemporalBoundaries(100, 200)


def event(user, item, timestamp, ratio=1.0):
    return KuaiRecInteraction(
        user_id=user,
        video_id=item,
        play_duration=1000,
        video_duration=1000,
        timestamp=timestamp,
        watch_ratio=ratio,
    )


def test_training_only_popularity():
    model = PopularityBaseline()

    model.fit([
        event(1, 10, 10),
        event(2, 20, 20),
        event(3, 20, 110),
        event(4, 20, 210),
    ], BOUNDARIES)

    assert model.counts[10] == 1
    assert model.counts[20] == 1


def test_repeat_events_count_toward_popularity():
    model = PopularityBaseline()

    model.fit([
        event(1, 10, 10),
        event(1, 10, 20),
        event(2, 20, 30),
    ], BOUNDARIES)

    assert model.counts[10] == 2
    assert model.counts[20] == 1


def test_zero_positive_items_remain_in_catalog():
    model = PopularityBaseline()

    model.fit([
        event(1, 10, 10, ratio=0.2),
        event(2, 20, 20),
    ], BOUNDARIES)

    assert model.catalog == {10, 20}
    assert model.counts[10] == 0


def test_training_history_exclusion():
    model = PopularityBaseline()

    model.fit([
        event(1, 10, 10),
        event(2, 10, 20),
        event(2, 20, 30),
    ], BOUNDARIES)

    assert model.recommend(1, k=5) == [20]


def test_deterministic_tie_breaking():
    model = PopularityBaseline()

    model.fit([
        event(1, 30, 10),
        event(2, 20, 20),
        event(3, 10, 30),
    ], BOUNDARIES)

    assert model.recommend(1, k=3) == [10, 20]


def test_explicit_exclusion():
    model = PopularityBaseline()

    model.fit([
        event(1, 10, 10),
        event(2, 20, 20),
        event(3, 30, 30),
    ], BOUNDARIES)

    assert model.recommend(
        1, k=3, excluded_items={20}
    ) == [10, 30]


def test_unknown_user_rejected():
    model = PopularityBaseline()
    model.fit([event(1, 10, 10)], BOUNDARIES)

    with pytest.raises(KeyError):
        model.recommend(999)


def test_invalid_k_rejected():
    model = PopularityBaseline()
    model.fit([event(1, 10, 10)], BOUNDARIES)

    with pytest.raises(ValueError):
        model.recommend(1, k=0)


def test_repeated_fit_resets_state():
    model = PopularityBaseline()

    model.fit([event(1, 10, 10)], BOUNDARIES)
    model.fit([event(2, 20, 20)], BOUNDARIES)

    assert model.catalog == {20}
    assert model.counts[10] == 0
    assert 1 not in model.training_users


def test_no_future_items_enter_catalog():
    model = PopularityBaseline()

    model.fit([
        event(1, 10, 10),
        event(1, 99, 150),
        event(1, 88, 250),
    ], BOUNDARIES)

    assert model.catalog == {10}
    assert 99 not in model.catalog
    assert 88 not in model.catalog
