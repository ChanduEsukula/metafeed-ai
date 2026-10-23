
"""Small reproducible KuaiRec two-tower training experiment."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.evaluation.config import load_evaluation_config
from metafeed.models.two_tower import TwoTowerModel
from metafeed.training.sampler import batch_diagnostics, iter_training_batches
from metafeed.training.losses import (
    build_positive_lookup,
    masked_contrastive_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/datasets/kuairec.json"
DEFAULT_OUTPUT = ROOT / "data/processed/kuairec/models"


def train_epoch(model, optimizer, pairs, known_positives, batch_size, seed, epoch):
    """Visit every event; update only batches with usable negatives.

    Effective events are rows with at least one unmasked off-diagonal
    negative, not all rows in an optimizer batch. Loss remains the existing
    mean over all rows in that batch (fully masked rows contribute zero).
    """
    device = next(model.parameters()).device
    model.train()
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

    return {
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
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "two_tower_smoke.pt")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Checkpoint already exists: {args.output}")

    if min(
        args.max_events,
        args.epochs,
        args.batch_size,
        args.embedding_dim,
    ) <= 0:
        parser.error("Training parameters must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_evaluation_config(CONFIG_PATH)
    path = ROOT / config.source_file

    rows = []

    for event in iter_kuairec(path):
        if event.timestamp >= config.boundaries.train_end:
            continue

        if not config.is_positive(event.watch_ratio):
            continue

        rows.append((event.user_id, event.video_id))

        if len(rows) >= args.max_events:
            break

    if not rows:
        raise RuntimeError("No positive training interactions found")

    user_ids = sorted({user for user, _ in rows})
    item_ids = sorted({item for _, item in rows})

    user_map = {user: idx for idx, user in enumerate(user_ids)}
    item_map = {item: idx for idx, item in enumerate(item_ids)}

    pairs = [
        (user_map[user], item_map[item])
        for user, item in rows
    ]

    # Sample-only lookup: full-data training needs ALL pre-cutoff positives,
    # including events omitted from any training sample. Never use held-out data.
    known_positives = build_positive_lookup(tuple(pairs))

    device = torch.device(args.device)

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

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

    losses = [stats["mean_training_loss"] for stats in diagnostics]
    output = args.output

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "user_map": user_map,
        "item_map": item_map,
        "embedding_dim": args.embedding_dim,
        "seed": args.seed,
        "losses": losses,
        "training_events": len(rows),
        "epoch_diagnostics": diagnostics,
        "batch_size": args.batch_size,
        "sampler": "deterministic_unique_items",
        "positive_lookup_scope": "sampled_training_positives_only",
        "selection": "first eligible positive events in source order",
        "dataset_config": json.loads(CONFIG_PATH.read_text()),
        "device": str(device),
    }

    save_checkpoint(checkpoint, output)
    restored_model, restored = restore_checkpoint(output)
    assert restored["training_events"] == len(rows)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(
            tensor.cpu(), restored_model.state_dict()[name], rtol=0, atol=0,
        )

    print("Device:", device)
    print("Positive events:", len(rows))
    print("Users:", len(user_map))
    print("Items:", len(item_map))
    print("Mean batch size:", round(sum(s["visited_events"] for s in diagnostics) / sum(s["batches"] for s in diagnostics), 2))
    print("Loss history:", [round(x, 6) if x is not None else None for x in losses])
    print("Checkpoint:", output)
    print("Checkpoint restoration: PASS")


if __name__ == "__main__":
    main()
