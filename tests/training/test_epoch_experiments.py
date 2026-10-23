"""Tiny local experiments only; no real-data training or FAISS imports."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from metafeed.evaluation.config import load_evaluation_config
from metafeed.evaluation.experiments import evaluate_checkpoints, rank_checkpoint_results

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_experiment(tmp_path, monkeypatch):
    training = load_script("train_two_tower")
    source = tmp_path / "events.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "video_id", "play_duration", "video_duration", "timestamp", "watch_ratio"])
        for user, item, timestamp, ratio in [
            (1, 10, 1, 1), (999, 999, 100, 1), (2, 20, 2, 1),
            (3, 30, 3, 1), (1, 20, 4, 1), (1, 10, 5, 1),
            (4, 40, 6, 0.5), (1, 99, 200, 1),
            (1, 30, 100, 1), (2, 40, 110, 1), (3, 30, 120, 1),
        ]:
            writer.writerow([user, item, 1000, 1000, timestamp, ratio])
    config_data = json.loads(training.CONFIG_PATH.read_text())
    config_data.update({
        "source_file": str(source),
        "train_end_utc": "1970-01-01T00:01:40Z",
        "validation_end_utc": "1970-01-01T00:03:20Z",
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    monkeypatch.setattr(training, "CONFIG_PATH", config_path)

    def run(epochs, name=None, save=True):
        output = tmp_path / f"{name or f'run{epochs}'}.pt"
        args = [
            "train_two_tower.py", "--epochs", str(epochs), "--max-events", "4",
            "--batch-size", "3", "--embedding-dim", "4", "--seed", "42",
            "--output", str(output),
        ]
        if save:
            args.append("--save-every-epoch")
        monkeypatch.setattr(sys, "argv", args)
        training.main()
        return output
    return training, source, config_path, run


@pytest.mark.parametrize("epoch,name", [(1, "experiment.epoch-0001.pt"), (10, "experiment.epoch-0010.pt")])
def test_epoch_checkpoint_names(epoch, name):
    training = load_script("train_two_tower")
    assert training.epoch_checkpoint_path(Path("out/experiment.pt"), epoch) == Path("out") / name
    with pytest.raises(ValueError):
        training.epoch_checkpoint_path(Path("experiment.pt"), 0)


def test_three_five_ten_epochs_have_identical_prefixes(tiny_experiment, monkeypatch):
    training, source, _, run = tiny_experiment
    seen_history = []
    original_train_epoch = training.train_epoch
    def inspect_epoch(model, optimizer, pairs, known, batch_size, seed, epoch):
        seen_history.append(known)
        return original_train_epoch(model, optimizer, pairs, known, batch_size, seed, epoch)
    monkeypatch.setattr(training, "train_epoch", inspect_epoch)
    outputs = [run(epochs) for epochs in (3, 5, 10)]
    checkpoints = [torch.load(training.epoch_checkpoint_path(output, 3), weights_only=True) for output in outputs]
    reference = checkpoints[0]
    for output, checkpoint, budget in zip(outputs, checkpoints, (3, 5, 10)):
        assert checkpoint["completed_epochs"] == 3
        assert checkpoint["epochs"] == budget
        assert checkpoint["sampling"] == reference["sampling"]
        assert checkpoint["losses"] == reference["losses"]
        assert checkpoint["seed"] == 42
        assert checkpoint["user_map"] == {1: 0, 2: 1, 3: 2, 4: 3}
        assert checkpoint["item_map"] == {10: 0, 20: 1, 30: 2, 40: 3}
        for key, value in checkpoint["model_state_dict"].items():
            assert torch.equal(value, reference["model_state_dict"][key])
        report = json.loads(output.with_suffix(".json").read_text())
        assert len(report["epoch_checkpoints"]) == budget
        assert len(report["epoch_diagnostics"]) == budget
        assert all(d["event_coverage"] == 1 for d in report["epoch_diagnostics"])
        assert all("cuda_peak_allocated_bytes" in d and "events_per_second" in d for d in report["epoch_diagnostics"])
        assert all(training.epoch_checkpoint_path(output, epoch).exists() for epoch in range(1, budget + 1))
        assert training.restore_checkpoint(output)[1]["completed_epochs"] == budget
    assert len(seen_history) == 18
    assert all(lookup == {0: frozenset({0, 1}), 1: frozenset({1}), 2: frozenset({2})} for lookup in seen_history)


def test_epoch_collision_preserves_artifacts_before_training(tiny_experiment, monkeypatch):
    training, _, config_path, run = tiny_experiment
    sentinel = config_path.parent / "collision.epoch-0002.pt"
    sentinel.write_bytes(b"existing checkpoint")
    monkeypatch.setattr(training, "sample_training_data", lambda *args: pytest.fail("Must fail before scan"))
    with pytest.raises(FileExistsError):
        run(3, "collision")
    assert sentinel.read_bytes() == b"existing checkpoint"
    assert not sentinel.with_name("collision.epoch-0001.pt").exists()


def test_optional_epoch_saving_and_snapshot_isolation(tiny_experiment):
    training, _, _, run = tiny_experiment
    output = run(3, save=False)
    assert output.exists()
    assert not training.epoch_checkpoint_path(output, 1).exists()
    model = training.TwoTowerModel(2, 2, 4)
    device = next(model.parameters()).device
    rng = torch.get_rng_state().clone()
    snapshot = training.training_snapshot(model, {"embedding_dim": 4}, {1: 0, 2: 1}, {1: 0, 2: 1}, 2, [])
    before = snapshot["model_state_dict"]["user_embedding.weight"].clone()
    with torch.no_grad():
        model.user_embedding.weight.add_(1)
    assert next(model.parameters()).device == device
    assert torch.equal(before, snapshot["model_state_dict"]["user_embedding.weight"])
    assert torch.equal(rng, torch.get_rng_state())


def result(name, ndcg, recall=0.5, epoch=1):
    return {"checkpoint": name, "NDCG@10": ndcg, "Recall@10": recall,
            "HitRate@10": 0.5, "completed_epochs": epoch, "evaluated_users": 10}


def test_best_checkpoint_uses_ndcg_only_and_stable_ties():
    rows = [result("high_recall", 0.1, 1), result("best", 0.3, 0.2),
            result("later", 0.3, 1, 3), result("z", 0.3, 1)]
    ranked = rank_checkpoint_results(rows)
    assert [row["checkpoint"] for row in ranked] == ["best", "z", "later", "high_recall"]
    assert rank_checkpoint_results(list(reversed(rows))) == ranked
    assert "validation_rank" not in rows[0]


@pytest.mark.parametrize("rows", [[], [result("bad", float("nan"))],
    [result("duplicate", 0.2), result("duplicate", 0.3)],
    [{**result("empty", 0.1), "evaluated_users": 0}]])
def test_invalid_selection_results_rejected(rows):
    with pytest.raises(ValueError):
        rank_checkpoint_results(rows)


def test_validation_aggregation_and_artifact_preservation(tiny_experiment, monkeypatch):
    import metafeed.evaluation.experiments as experiments
    training, source, config_path, run = tiny_experiment
    output = run(3)
    checkpoints = [training.epoch_checkpoint_path(output, epoch) for epoch in range(1, 4)]
    original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    original_evaluate = experiments.evaluate_ranker
    calls = []
    def inspect_split(path, config, ranker):
        calls.append((config.split, type(ranker).__name__))
        return original_evaluate(path, config, ranker)
    monkeypatch.setattr(experiments, "evaluate_ranker", inspect_split)
    report = evaluate_checkpoints(checkpoints, source, load_evaluation_config(config_path))
    assert calls == [("validation", "PopularityBaseline")] + [("validation", "TwoTowerRanker")] * 3
    assert report["split"] == "validation"
    assert report["best_checkpoint"] == report["comparison"][0]
    assert len(report["comparison"]) == 3
    for row in report["comparison"]:
        assert row["population_coverage_parity"]
        assert row["evaluated_users"] == report["popularity"]["evaluated_users"] == 2
        assert row["sample_sha256"] == report["comparison"][0]["sample_sha256"]
        assert all(metric in row for metric in ("Recall@10", "NDCG@10", "HitRate@10"))
    assert report["population_coverage_counters"]["cold_start_users"] == 1
    assert report["population_coverage_counters"]["empty_ground_truth_users"] == 1
    assert original == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    runner = load_script("validate_two_tower_checkpoints")
    comparison = config_path.parent / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "validate_two_tower_checkpoints.py", "--checkpoints", str(output.with_name("run3.epoch-*.pt")),
        "--config", str(config_path), "--output", str(comparison),
    ])
    runner.main()
    saved = comparison.read_bytes()
    assert len(json.loads(saved)["comparison"]) == 3
    with pytest.raises(FileExistsError):
        runner.main()
    assert comparison.read_bytes() == saved


def test_mixed_sampling_and_duplicate_paths_rejected(tiny_experiment):
    training, source, config_path, run = tiny_experiment
    output = run(3)
    first, second = [training.epoch_checkpoint_path(output, epoch) for epoch in (1, 2)]
    checkpoint = torch.load(second, weights_only=True)
    checkpoint["sampling"]["sample_sha256"] = "different-sample"
    changed = config_path.parent / "different.pt"
    training.save_checkpoint(checkpoint, changed)
    with pytest.raises(ValueError, match="different sample"):
        evaluate_checkpoints([first, changed], source, load_evaluation_config(config_path))
    with pytest.raises(ValueError, match="unique checkpoint"):
        evaluate_checkpoints([first, first], source, load_evaluation_config(config_path))


def test_original_final_only_checkpoint_metadata_is_supported(tiny_experiment):
    training, source, config_path, run = tiny_experiment
    output = run(3)
    checkpoint = torch.load(output, weights_only=True)
    for key in ("completed_epochs", "learning_rate", "temperature"):
        checkpoint.pop(key)
    legacy = config_path.parent / "legacy.pt"
    training.save_checkpoint(checkpoint, legacy)
    report = evaluate_checkpoints([legacy, output], source, load_evaluation_config(config_path))
    assert all(row["completed_epochs"] == 3 for row in report["comparison"])
    assert all(row["learning_rate"] == 0.001 and row["temperature"] == 0.1 for row in report["comparison"])
