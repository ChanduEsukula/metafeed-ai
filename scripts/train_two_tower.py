
"""Small reproducible KuaiRec two-tower training experiment."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.evaluation.config import load_evaluation_config
from metafeed.models.two_tower import TwoTowerModel
from metafeed.training.losses import (
    build_positive_lookup,
    masked_contrastive_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/datasets/kuairec.json"
DEFAULT_OUTPUT = ROOT / "data/processed/kuairec/models"


def make_batch(
    pairs,
    start,
    batch_size,
):
    """Select unique users and items from a shuffled pool."""

    selected = []
    users = set()
    items = set()

    for user, item in pairs[start:]:
        if user in users or item in items:
            continue

        selected.append((user, item))
        users.add(user)
        items.add(item)

        if len(selected) == batch_size:
            break

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

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

    losses = []
    batch_sizes = []

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(pairs)

        epoch_losses = []

        for start in range(0, len(pairs), args.batch_size):
            batch = make_batch(pairs, start, args.batch_size)

            if len(batch) < 2:
                continue

            users = torch.tensor(
                [user for user, _ in batch],
                dtype=torch.long,
                device=device,
            )
            items = torch.tensor(
                [item for _, item in batch],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad()

            loss = masked_contrastive_loss(
                model.encode_users(users),
                model.encode_items(items),
                users,
                items,
                known_positives,
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            batch_sizes.append(len(batch))

        if not epoch_losses:
            raise RuntimeError("No usable training batches")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(mean_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={mean_loss:.6f} "
            f"batches={len(epoch_losses)}",
            flush=True,
        )

    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)

    output = DEFAULT_OUTPUT / "two_tower_smoke.pt"

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "user_map": user_map,
        "item_map": item_map,
        "embedding_dim": args.embedding_dim,
        "seed": args.seed,
        "losses": losses,
        "training_events": len(rows),
    }

    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")

    torch.save(checkpoint, output)

    restored = torch.load(
        output,
        map_location="cpu",
        weights_only=True,
    )

    assert restored["training_events"] == len(rows)

    print("Device:", device)
    print("Positive events:", len(rows))
    print("Users:", len(user_map))
    print("Items:", len(item_map))
    print("Mean batch size:", round(sum(batch_sizes) / len(batch_sizes), 2))
    print("Loss history:", [round(x, 6) for x in losses])
    print("Checkpoint:", output)
    print("Checkpoint restoration: PASS")


if __name__ == "__main__":
    main()
