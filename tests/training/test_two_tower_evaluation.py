"""Learned ranking and population parity; run separately from FAISS."""

import csv
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.data_pipeline.splits import TemporalBoundaries
from metafeed.evaluation.config import EvaluationConfig
from metafeed.evaluation.offline import evaluate_popularity, evaluate_ranker
from metafeed.evaluation.popularity import PopularityBaseline
from metafeed.evaluation.two_tower import TwoTowerRanker
from metafeed.models.two_tower import TwoTowerModel


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / "events.csv"
    rows = [
        (1, 10, 1, 1), (2, 20, 2, 1), (3, 30, 3, 0.2),
        (4, 40, 4, 1), (5, 10, 5, 0.2),
        (1, 20, 100, 1), (1, 10, 101, 1), (2, 20, 102, 1),
        (3, 40, 103, 1), (99, 20, 104, 1), (4, 999, 105, 1),
        (4, 999, 106, 1), (5, 30, 107, 1), (1, 40, 200, 1),
    ]
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "video_id", "play_duration", "video_duration", "timestamp", "watch_ratio"])
        for user, item, timestamp, ratio in rows:
            writer.writerow([user, item, 1000, 1000, timestamp, ratio])
    config = EvaluationConfig(source, TemporalBoundaries(100, 200), 1.0, "validation", (1, 10), 42)
    model = TwoTowerModel(5, 4, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = {
        "model_state_dict": model.state_dict(), "embedding_dim": 2,
        "user_map": {5: 0, 4: 1, 3: 2, 2: 3, 1: 4},
        "item_map": {40: 0, 10: 1, 30: 2, 20: 3},
        # Sample provenance must never restrict candidates or eligible users.
        "sampling": {"sampled_users": 1, "sampled_items": 1},
        "dataset_config": {
            "split_strategy": "global_chronological",
            "train_end_utc": "1970-01-01T00:01:40Z",
            "validation_end_utc": "1970-01-01T00:03:20Z",
            "target_definition": {"field": "watch_ratio", "operator": ">=", "threshold": 1.0},
        },
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    return source, config, checkpoint, path


@pytest.mark.parametrize("split", ["validation", "test"])
def test_population_coverage_parity(experiment, split):
    source, config, _, path = experiment
    config = replace(config, split=split)
    ranker = TwoTowerRanker(path, config)
    baseline = PopularityBaseline()
    baseline.fit(iter_kuairec(source), config.boundaries, config.positive_threshold)
    ranker.validate_population(baseline.training_users, baseline.catalog)
    popularity = evaluate_popularity(source, config)
    learned = evaluate_ranker(source, config, ranker)
    assert {k: v for k, v in asdict(popularity).items() if k != "metrics"} == {
        k: v for k, v in asdict(learned).items() if k != "metrics"
    }
    if split == "validation":
        assert learned.evaluated_users == 3
        assert learned.cold_start_users == 1
        assert learned.empty_ground_truth_users == 2
        assert learned.unavailable_positive_items == 1
        assert learned.distinct_target_positive_pairs == 7
        assert learned.previously_seen_positive_pairs == 2
        assert learned.eligible_positive_pairs == 4  # includes cold user, preserving benchmark semantics
        assert learned.metrics["Recall@10"] == 1
    assert evaluate_ranker(source, config, ranker) == learned


def test_ties_exclusions_and_unsampled_ids(experiment):
    _, config, _, path = experiment
    ranker = TwoTowerRanker(path, config)
    assert ranker.recommend(5, 10, set()) == [10, 20, 30, 40]
    assert ranker.recommend(1, 10, {10, 20}) == [30, 40]
    assert ranker.recommend(3, 10, ranker.catalog) == []
    with pytest.raises(KeyError):
        ranker.recommend(99, 10, set())
    with pytest.raises(ValueError):
        ranker.recommend(1, 0, set())


def test_normalization_and_exact_inner_products(experiment):
    _, config, checkpoint, path = experiment
    checkpoint["model_state_dict"]["user_embedding.weight"].fill_(0)
    checkpoint["model_state_dict"]["user_embedding.weight"][4] = torch.tensor([3., 0.])
    checkpoint["model_state_dict"]["item_embedding.weight"][:] = torch.tensor([
        [-1., 0.], [2., 0.], [0., 4.], [100., 100.],
    ])
    torch.save(checkpoint, path)
    ranker = TwoTowerRanker(path, config)
    assert ranker.recommend(1, 4, set()) == [10, 20, 30, 40]
    torch.testing.assert_close(torch.linalg.vector_norm(ranker.item_vectors, dim=1), torch.ones(4))
    assert not ranker.model.training


@pytest.mark.parametrize("damage", ["missing_weight", "bad_shape", "nonfinite", "bad_map", "missing_metadata"])
def test_incompatible_checkpoints_rejected(experiment, damage):
    _, config, checkpoint, path = experiment
    if damage == "missing_weight":
        checkpoint["model_state_dict"].pop("user_embedding.weight")
    elif damage == "bad_shape":
        checkpoint["embedding_dim"] = 3
    elif damage == "nonfinite":
        checkpoint["model_state_dict"]["item_embedding.weight"][0, 0] = float("nan")
    elif damage == "bad_map":
        checkpoint["user_map"][1] = 0
    else:
        checkpoint.pop("dataset_config")
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        TwoTowerRanker(path, config)


def test_protocol_and_catalog_mismatch_rejected(experiment):
    _, config, _, path = experiment
    with pytest.raises(ValueError, match="protocol"):
        TwoTowerRanker(path, replace(config, positive_threshold=0.5))
    ranker = TwoTowerRanker(path, config)
    with pytest.raises(ValueError, match="complete training"):
        ranker.validate_population({1}, {10})


def test_restricted_loading_and_no_rng_or_checkpoint_mutation(experiment, monkeypatch):
    _, config, checkpoint, path = experiment
    original = path.read_bytes()
    rng = torch.get_rng_state().clone()
    load = torch.load
    calls = []
    def restricted(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", restricted)
    ranker = TwoTowerRanker(path, config)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert torch.equal(rng, torch.get_rng_state())
    assert path.read_bytes() == original
    for name, value in checkpoint["model_state_dict"].items():
        assert torch.equal(ranker.model.state_dict()[name], value)


def test_validation_cli_report_fingerprints_and_exclusive_output(experiment, tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import json
    import sys

    source, config, checkpoint, checkpoint_path = experiment
    script = Path(__file__).resolve().parents[2] / "scripts/evaluate_two_tower.py"
    spec = importlib.util.spec_from_file_location("evaluate_two_tower", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = json.loads((script.parents[1] / "configs/datasets/kuairec.json").read_text())
    data.update(checkpoint["dataset_config"])
    data["source_file"] = str(source)
    data["evaluation"]["k_values"] = [1, 10]
    assert data["evaluation"]["split"] == "test"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    checkpoint["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    torch.save(checkpoint, checkpoint_path)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(sys, "argv", [
        str(script), "--checkpoint", str(checkpoint_path), "--config", str(config_path),
        "--output", str(output),
    ])
    runner.main()
    report = json.loads(output.read_text())
    assert report["split"] == "validation"
    assert report["effective_evaluation_config"]["split"] == "validation"
    assert report["population_coverage_parity"] is True
    assert report["reports"]["two_tower"]["evaluated_users"] == 3
    assert json.loads(config_path.read_text())["evaluation"]["split"] == "test"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == original
    checkpoint["source_sha256"] = "wrong_dataset"
    torch.save(checkpoint, checkpoint_path)
    sys.argv[-1] = str(tmp_path / "mismatched.json")
    with pytest.raises(ValueError, match="fingerprint"):
        runner.main()
    assert not (tmp_path / "mismatched.json").exists()
