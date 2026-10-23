"""Integration tests for the real-data script without FAISS imports."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import torch

from metafeed.models.two_tower import TwoTowerModel
from metafeed.training.losses import build_positive_lookup
from metafeed.training.sampler import iter_training_batches

spec = importlib.util.spec_from_file_location(
    "train_two_tower", Path(__file__).resolve().parents[2] / "scripts/train_two_tower.py"
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def run_epoch(pairs, lookup=None, batch_size=4, epoch=0):
    torch.manual_seed(42)
    model = TwoTowerModel(2, 3, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stats = training.train_epoch(
        model, optimizer, pairs,
        lookup if lookup is not None else build_positive_lookup(tuple(pairs)),
        batch_size, 42, epoch,
    )
    return model, optimizer, before, stats


def test_duplicate_event_coverage_and_skipped_tail():
    pairs = [(0, 0)] * 3 + [(1, 1)]
    model, optimizer, before, stats = run_epoch(pairs, batch_size=2)
    assert stats["visited_events"] == 4
    assert stats["event_coverage"] == 1
    assert stats["effective_training_events"] == 2
    assert stats["effective_training_coverage"] == 0.5
    assert stats["optimizer_steps"] == 1
    assert stats["skipped_batches"] == 2
    assert stats["min_batch_size"] == 1
    assert stats["mean_batch_size"] == pytest.approx(4 / 3)
    assert stats["valid_negative_fraction"] == 1
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("pairs", [[(0, 0)] * 3, [(0, 0), (0, 1), (0, 2)]])
def test_no_valid_negatives_never_step(pairs):
    model, optimizer, before, stats = run_epoch(pairs)
    assert stats["event_coverage"] == 1
    assert stats["effective_training_coverage"] == 0
    assert stats["optimizer_steps"] == 0
    assert stats["skipped_batches"] == stats["batches"]
    assert stats["valid_negative_fraction"] == 0
    assert stats["mean_training_loss"] is None
    assert not optimizer.state
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_fully_masked_rows_not_counted_as_effective():
    pairs = [(0, 0), (0, 1), (1, 2)]
    lookup = {0: frozenset({0, 1, 2}), 1: frozenset({2})}
    _, _, _, stats = run_epoch(pairs, lookup)
    assert stats["optimizer_steps"] == 1
    assert stats["effective_training_events"] == 1
    assert stats["effective_training_coverage"] == pytest.approx(1 / 3)
    assert stats["valid_negative_fraction"] == pytest.approx(2 / 6)


def test_repeated_training_reproducible():
    pairs = [(0, 0), (0, 1), (1, 2), (1, 1), (0, 0)]
    first, _, _, stats = run_epoch(pairs, batch_size=3)
    second, _, _, repeated = run_epoch(pairs, batch_size=3)
    timing_keys = {"elapsed_seconds", "events_per_second", "effective_events_per_second",
                   "process_peak_rss_bytes"}
    assert {k: v for k, v in stats.items() if k not in timing_keys} == {
        k: v for k, v in repeated.items() if k not in timing_keys
    }
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_checkpoint_round_trip_and_no_overwrite(tmp_path):
    model, _, _, stats = run_epoch([(0, 0), (1, 1)])
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "user_map": {10: 0, 20: 1}, "item_map": {30: 0, 40: 1, 50: 2},
        "embedding_dim": 4, "seed": 42, "losses": [stats["mean_training_loss"]],
        "training_events": 2, "epoch_diagnostics": [stats],
    }
    output = tmp_path / "checkpoint.pt"
    training.save_checkpoint(checkpoint, output)
    original = output.read_bytes()
    restored, metadata = training.restore_checkpoint(output)
    assert metadata["epoch_diagnostics"] == [stats]
    assert metadata["user_map"] == checkpoint["user_map"]
    users, items = torch.tensor([0, 1]), torch.tensor([0, 1])
    torch.testing.assert_close(restored(users, items), model(users, items), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        training.save_checkpoint({}, output)
    assert output.read_bytes() == original


def test_checkpoint_rejects_incompatible_weights(tmp_path):
    path = tmp_path / "bad.pt"
    training.save_checkpoint({
        "model_state_dict": {}, "user_map": {0: 0},
        "item_map": {0: 0}, "embedding_dim": 4,
    }, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        training.restore_checkpoint(path)


def test_epoch_shuffle_preserves_events_and_input():
    pairs = [(u, i) for u in range(4) for i in range(5)] + [(0, 0)] * 3
    original = pairs.copy()
    first = list(iter_training_batches(pairs, 4, 42, epoch=0))
    second = list(iter_training_batches(pairs, 4, 42, epoch=1))
    assert first != second
    for batches in (first, second):
        assert Counter(p for batch in batches for p in batch) == Counter(pairs)
        assert all(len({i for _, i in batch}) == len(batch) for batch in batches)
    assert pairs == original


def test_cli_uses_only_training_positives(tmp_path, monkeypatch):
    import csv
    import json
    import sys

    config = json.loads(training.CONFIG_PATH.read_text())
    cutoff = training.load_evaluation_config(training.CONFIG_PATH).boundaries
    source = tmp_path / "interactions.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "user_id", "video_id", "play_duration", "video_duration",
            "timestamp", "watch_ratio",
        ])
        for user, item, timestamp, ratio in [
            (999, 999, cutoff.train_end, 1),
            (10, 20, cutoff.train_end - 1, 1),
            (888, 888, cutoff.validation_end, 1),
            (30, 40, cutoff.train_end - 2, 1),
            (777, 777, cutoff.train_end - 3, 0.5),
            (10, 20, cutoff.train_end - 4, 1),
        ]:
            writer.writerow([user, item, ratio * 1000, 1000, timestamp, ratio])
    config["source_file"] = str(source)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(training, "CONFIG_PATH", config_path)
    output = tmp_path / "trained.pt"
    monkeypatch.setattr(sys, "argv", [
        "train_two_tower.py", "--max-events", "3", "--epochs", "1",
        "--batch-size", "2", "--embedding-dim", "4", "--output", str(output),
    ])
    training.main()
    _, checkpoint = training.restore_checkpoint(output)
    assert checkpoint["user_map"] == {10: 0, 30: 1, 777: 2}
    assert checkpoint["item_map"] == {20: 0, 40: 1, 777: 2}
    assert checkpoint["training_events"] == 3
    assert checkpoint["epoch_diagnostics"][0]["visited_events"] == 3
    assert checkpoint["positive_lookup_scope"] == "complete_training_split"
    original = output.read_bytes()
    monkeypatch.setattr(training, "sample_training_data", lambda *args: pytest.fail("Should fail before reading"))
    with pytest.raises(FileExistsError):
        training.main()
    assert output.read_bytes() == original


@pytest.mark.parametrize("name", ["cpu", "cuda"])
def test_device_selection(name, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert training.select_device(name).type == name


def test_unavailable_cuda_fails_explicitly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        training.select_device("cuda")


def test_invalid_device():
    with pytest.raises(ValueError, match="Unsupported device"):
        training.select_device("tpu")


def test_epoch_reports_timing_and_memory():
    _, _, _, stats = run_epoch([(0, 0), (1, 1)])
    assert stats["elapsed_seconds"] > 0
    assert stats["events_per_second"] == pytest.approx(2 / stats["elapsed_seconds"])
    assert stats["effective_events_per_second"] > 0
    assert stats["cuda_peak_allocated_bytes"] is None
    assert stats["cuda_peak_reserved_bytes"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_actual_cuda_step():
    training.configure_reproducibility(42)
    model = TwoTowerModel(2, 2, 4).to(training.select_device("cuda"))
    optimizer = torch.optim.Adam(model.parameters())
    stats = training.train_epoch(
        model, optimizer, [(0, 0), (1, 1)],
        {0: frozenset({0}), 1: frozenset({1})}, 2, 42, 0,
    )
    assert stats["optimizer_steps"] == 1
    assert stats["cuda_peak_allocated_bytes"] > 0
    assert stats["cuda_peak_reserved_bytes"] >= stats["cuda_peak_allocated_bytes"]


def test_prepare_only_reports_without_training(tmp_path, monkeypatch):
    import sys
    import json
    from metafeed.data_pipeline.splits import TemporalBoundaries
    from metafeed.training.kuairec_dataset import sample_training_data

    source = tmp_path / "source.csv"
    source.write_text(
        "user_id,video_id,play_duration,video_duration,timestamp,watch_ratio\n"
        "1,10,1000,1000,1,1\n1,20,1000,1000,2,1\n"
    )
    prepared = sample_training_data(source, TemporalBoundaries(100, 200), 1, 42)
    monkeypatch.setattr(training, "sample_training_data", lambda *args: prepared)
    monkeypatch.setattr(training, "TwoTowerModel", lambda **kwargs: pytest.fail("No training"))
    output, report = tmp_path / "model.pt", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "train_two_tower.py", "--prepare-only", "--max-events", "1", "--source", str(source),
        "--output", str(output), "--report", str(report),
    ])
    training.main()
    assert not output.exists()
    data = json.loads(report.read_text())
    assert data["sampling"]["known_positive_pairs"] == 2
    assert data["source_sha256"]
    assert data["config_sha256"]
    assert data["deterministic_algorithms"] is True
    original = report.read_bytes()
    with pytest.raises(FileExistsError, match="Report already exists"):
        training.main()
    assert report.read_bytes() == original


def test_cuda_unavailable_fails_before_dataset_scan(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "sample_training_data", lambda *args: pytest.fail("No dataset scan"))
    monkeypatch.setattr(sys, "argv", [
        "train_two_tower.py", "--device", "cuda", "--output", str(tmp_path / "model.pt"),
    ])
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        training.main()
