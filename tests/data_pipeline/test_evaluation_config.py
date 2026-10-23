
from pathlib import Path

import pytest

from metafeed.evaluation.config import load_evaluation_config


CONFIG_PATH = Path("configs/datasets/kuairec.json")


def test_load_evaluation_config():
    config = load_evaluation_config(CONFIG_PATH)

    assert config.split == "test"
    assert config.k_values == (10,)
    assert config.random_seed == 42
    assert config.positive_threshold == 1.0


def test_positive_threshold():
    config = load_evaluation_config(CONFIG_PATH)

    assert config.is_positive(1.0)
    assert config.is_positive(1.5)
    assert not config.is_positive(0.99)


def test_chronological_boundaries():
    config = load_evaluation_config(CONFIG_PATH)

    assert config.boundaries.train_end < config.boundaries.validation_end


@pytest.mark.parametrize("value", [
    float("nan"),
    float("inf"),
    float("-inf"),
])
def test_nonfinite_watch_ratio_is_not_positive(value):
    config = load_evaluation_config(CONFIG_PATH)

    assert not config.is_positive(value)
