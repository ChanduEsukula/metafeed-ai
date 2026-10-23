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
    assert stats == repeated
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
    assert checkpoint["user_map"] == {10: 0, 30: 1}
    assert checkpoint["item_map"] == {20: 0, 40: 1}
    assert checkpoint["training_events"] == 3
    assert checkpoint["epoch_diagnostics"][0]["visited_events"] == 3
    assert checkpoint["positive_lookup_scope"] == "sampled_training_positives_only"
    original = output.read_bytes()
    monkeypatch.setattr(training, "iter_kuairec", lambda _: pytest.fail("Should fail before reading"))
    with pytest.raises(FileExistsError):
        training.main()
    assert output.read_bytes() == original
