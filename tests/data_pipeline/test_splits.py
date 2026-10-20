from dataclasses import replace

import pytest

from metafeed.data_pipeline.kuairec import KuaiRecInteraction
from metafeed.data_pipeline.splits import (
    TemporalBoundaries,
    assign_split,
)


@pytest.fixture
def sample_event():
    return KuaiRecInteraction(
        user_id=1,
        video_id=10,
        play_duration=5000,
        video_duration=10000,
        timestamp=100.0,
        watch_ratio=0.5,
    )


def test_train_assignment(sample_event):
    boundaries = TemporalBoundaries(200, 300)
    assert assign_split(sample_event, boundaries) == "train"


def test_validation_assignment(sample_event):
    event = replace(sample_event, timestamp=200)
    boundaries = TemporalBoundaries(200, 300)

    assert assign_split(event, boundaries) == "validation"


def test_test_assignment(sample_event):
    event = replace(sample_event, timestamp=300)
    boundaries = TemporalBoundaries(200, 300)

    assert assign_split(event, boundaries) == "test"


def test_invalid_order_rejected():
    with pytest.raises(ValueError):
        TemporalBoundaries(300, 200)


def test_equal_boundaries_rejected():
    with pytest.raises(ValueError):
        TemporalBoundaries(200, 200)


def test_nonfinite_boundaries_rejected():
    with pytest.raises(ValueError):
        TemporalBoundaries(float("nan"), 300)
