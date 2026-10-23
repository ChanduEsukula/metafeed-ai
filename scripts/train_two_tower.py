
"""Small reproducible KuaiRec two-tower training experiment."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from metafeed.evaluation.config import load_evaluation_config
from metafeed.models.two_tower import TwoTowerModel
from metafeed.training.kuairec_dataset import sample_training_data
from metafeed.training.losses import masked_contrastive_loss
from metafeed.training.sampler import batch_diagnostics, iter_training_batches


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/datasets/kuairec.json"
DEFAULT_OUTPUT = ROOT / "data/processed/kuairec/models"


def select_device(name):
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device: {name}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime or use --device cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def configure_reproducibility(seed):
    # Must precede CUDA initialization. Fail rather than silently use nondeterminism.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def process_peak_rss_bytes():
    """Process lifetime high-water RSS, not per-epoch incremental memory."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, AttributeError):
        return None


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_epoch(model, optimizer, pairs, known_positives, batch_size, seed, epoch):
    """Visit every event; update only batches with usable negatives.

    Effective events are rows with at least one unmasked off-diagonal
    negative, not all rows in an optimizer batch. Loss remains the existing
    mean over all rows in that batch (fully masked rows contribute zero).
    """
    device = next(model.parameters()).device
    model.train()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sizes = []
    losses = []
    visited = effective = valid = possible = skipped = 0
    for batch in iter_training_batches(pairs, batch_size, seed, epoch):
        stats = batch_diagnostics([batch], known_positives)
        sizes.append(len(batch))
        visited += len(batch)
        valid += stats.valid_negatives
        possible += stats.possible_negatives
        if stats.valid_negatives == 0:
            skipped += 1
            continue

        users = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)
        items = torch.tensor([i for _, i in batch], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_contrastive_loss(
            model.encode_users(users), model.encode_items(items),
            users, items, known_positives,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        effective += stats.effective_interactions

    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "events_per_second": visited / elapsed if elapsed else 0.0,
        "effective_events_per_second": effective / elapsed if elapsed else 0.0,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "epoch": epoch + 1,
        # Arithmetic mean of optimizer-step batch means; null if no updates.
        "mean_training_loss": sum(losses) / len(losses) if losses else None,
        "visited_events": visited,
        "event_coverage": visited / len(pairs) if pairs else 0.0,
        "effective_training_events": effective,
        "effective_training_coverage": effective / len(pairs) if pairs else 0.0,
        # Sizes and negative fraction include skipped batches.
        "batches": len(sizes),
        "mean_batch_size": visited / len(sizes) if sizes else 0.0,
        "min_batch_size": min(sizes, default=0),
        "valid_negative_fraction": valid / possible if possible else 0.0,
        "skipped_batches": skipped,
        "optimizer_steps": len(losses),
    }


def save_checkpoint(checkpoint, output):
    """Exclusive creation prevents overwrites, including concurrent writers."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(checkpoint, handle)


def epoch_checkpoint_path(output: Path, epoch: int) -> Path:
    """Run-specific stem plus a sortable, one-based epoch identifier."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    return output.with_name(f"{output.stem}.epoch-{epoch:04d}{output.suffix or '.pt'}")


def training_snapshot(model, metadata, user_map, item_map, event_count, diagnostics):
    """Copy to CPU for serialization without moving the active training model."""
    return {
        **metadata,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "user_map": user_map, "item_map": item_map,
        "losses": [stats["mean_training_loss"] for stats in diagnostics],
        "training_events": event_count,
        "epoch_diagnostics": list(diagnostics),
        "completed_epochs": len(diagnostics),
        "epoch_checkpoints": metadata.get("epoch_checkpoints", [])[:len(diagnostics)],
        "sampler": "deterministic_unique_items",
    }


def restore_checkpoint(output):
    """Use restricted loading and validate weights in a fresh CPU model."""
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    model = TwoTowerModel(
        len(checkpoint["user_map"]), len(checkpoint["item_map"]),
        checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "two_tower_smoke.pt")
    parser.add_argument("--source", type=Path, help="Override configured CSV path")
    parser.add_argument("--report", type=Path, help="Exclusive JSON report; defaults beside checkpoint")
    parser.add_argument("--prepare-only", action="store_true", help="Assess preprocessing without training")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Save exclusive <output-stem>.epoch-NNNN.pt snapshots")
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".json")
    if report_path.resolve() == args.output.resolve():
        parser.error("Report and checkpoint paths must differ")
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")


    if min(
        args.max_events,
        args.epochs,
        args.batch_size,
        args.embedding_dim,
    ) <= 0:
        parser.error("Training parameters must be positive")

    epoch_outputs = (
        [epoch_checkpoint_path(args.output, epoch) for epoch in range(1, args.epochs + 1)]
        if args.save_every_epoch and not args.prepare_only else []
    )
    if not args.prepare_only:
        for output_path in [args.output, *epoch_outputs]:
            if report_path.resolve() == output_path.resolve():
                parser.error("Report and checkpoint paths must differ")
            if output_path.exists():
                raise FileExistsError(f"Checkpoint already exists: {output_path}")

    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must be in [0, 2**32)")
    started = time.perf_counter()
    configure_reproducibility(args.seed)
    device = select_device(args.device)
    config = load_evaluation_config(CONFIG_PATH)
    path = args.source or ROOT / config.source_file
    prepared = sample_training_data(
        path, config.boundaries, args.max_events, args.seed, config.positive_threshold,
    )
    data = prepared.data
    if not data.positive_pairs:
        raise RuntimeError("No positive training interactions found")
    pairs = data.positive_pairs
    user_map, item_map = data.user_to_index, data.item_to_index
    known_positives = prepared.known_positives
    with path.open("rb") as handle:
        source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "sampling": prepared.diagnostics,
        "positive_lookup_scope": "complete_training_split",
        "selection": prepared.diagnostics["strategy"],
        "source_sha256": source_sha256,
        "effective_source": str(path),
        "dataset_config": json.loads(CONFIG_PATH.read_text()),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "preprocessing_seconds": time.perf_counter() - started,
        "preprocessing_peak_rss_bytes": process_peak_rss_bytes(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "code_sha256": {
            str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in [Path(__file__), *sorted((ROOT / "src/metafeed/training").glob("*.py")),
                         ROOT / "src/metafeed/models/two_tower.py"]
        },
        "command": sys.argv,
        "seed": args.seed, "device": str(device),
        "batch_size": args.batch_size, "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "save_every_epoch": args.save_every_epoch,
        "epoch_checkpoints": [str(path) for path in epoch_outputs],
        "learning_rate": 0.001, "temperature": 0.1,
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)
    if args.prepare_only:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return

    model = TwoTowerModel(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    diagnostics = []
    for epoch in range(args.epochs):
        stats = train_epoch(
            model, optimizer, pairs, known_positives,
            args.batch_size, args.seed, epoch,
        )
        diagnostics.append(stats)
        print(json.dumps(stats, sort_keys=True), flush=True)
        if args.save_every_epoch:
            save_checkpoint(
                training_snapshot(model, metadata, user_map, item_map, len(pairs), diagnostics),
                epoch_outputs[epoch],
            )

    losses = [stats["mean_training_loss"] for stats in diagnostics]
    output = args.output

    checkpoint = training_snapshot(
        model, metadata, user_map, item_map, len(pairs), diagnostics,
    )

    save_checkpoint(checkpoint, output)
    restored_model, restored = restore_checkpoint(output)
    assert restored["training_events"] == len(pairs)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(
            tensor.cpu(), restored_model.state_dict()[name], rtol=0, atol=0,
        )

    metadata.update({
        "epoch_diagnostics": diagnostics,
        "completed_epochs": len(diagnostics),
        "total_elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "checkpoint": str(output), "checkpoint_restoration": "PASS",
    })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print("Device:", device)
    print("Positive events:", len(pairs))
    print("Users:", len(user_map))
    print("Items:", len(item_map))
    print("Mean batch size:", round(sum(s["visited_events"] for s in diagnostics) / sum(s["batches"] for s in diagnostics), 2))
    print("Loss history:", [round(x, 6) if x is not None else None for x in losses])
    print("Checkpoint:", output)
    print("Checkpoint restoration: PASS")


if __name__ == "__main__":
    main()
