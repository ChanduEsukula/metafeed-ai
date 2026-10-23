
"""Training-only KuaiRec adapter for PyTorch recommendation models."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.data_pipeline.splits import (
    TemporalBoundaries,
    assign_split,
)


@dataclass(frozen=True)
class TrainingData:
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]
    positive_pairs: tuple[tuple[int, int], ...]

    @property
    def num_users(self) -> int:
        return len(self.user_to_index)

    @property
    def num_items(self) -> int:
        return len(self.item_to_index)


def load_training_data(
    path: Path,
    boundaries: TemporalBoundaries,
    positive_threshold: float = 1.0,
) -> TrainingData:
    """Load training-only positive pairs with deterministic ID maps."""

    if not 0 <= positive_threshold < float("inf"):
        raise ValueError("Invalid positive threshold")

    users: set[int] = set()
    items: set[int] = set()
    raw_positive_pairs: list[tuple[int, int]] = []

    for event in iter_kuairec(path):
        if assign_split(event, boundaries) != "train":
            continue

        users.add(event.user_id)
        items.add(event.video_id)

        if event.watch_ratio >= positive_threshold:
            raw_positive_pairs.append(
                (event.user_id, event.video_id)
            )

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(sorted(users))
    }

    item_to_index = {
        item_id: index
        for index, item_id in enumerate(sorted(items))
    }

    positive_pairs = tuple(
        (user_to_index[user], item_to_index[item])
        for user, item in raw_positive_pairs
    )

    return TrainingData(
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        positive_pairs=positive_pairs,
    )


class KuaiRecPositiveDataset(Dataset):
    """Return encoded positive user-item interactions."""

    def __init__(self, data: TrainingData) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data.positive_pairs)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.data.positive_pairs[index]


def iter_unique_positive_batches(
    data: TrainingData,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic batches with unique users and items.

    This prevents duplicate IDs within each batch, but does not
    eliminate all known-positive false negatives.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    used_users: set[int] = set()
    used_items: set[int] = set()

    batch: list[tuple[int, int]] = []

    for user, item in data.positive_pairs:
        if (
            user in used_users
            or item in used_items
            or len(batch) >= batch_size
        ):
            if batch:
                yield (
                    torch.tensor(
                        [u for u, _ in batch],
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        [i for _, i in batch],
                        dtype=torch.long,
                    ),
                )

            batch = []
            used_users.clear()
            used_items.clear()

        batch.append((user, item))
        used_users.add(user)
        used_items.add(item)

    if batch:
        yield (
            torch.tensor(
                [u for u, _ in batch],
                dtype=torch.long,
            ),
            torch.tensor(
                [i for _, i in batch],
                dtype=torch.long,
            ),
        )


@dataclass(frozen=True)
class SampledTrainingData:
    data: TrainingData
    known_positives: dict[int, frozenset[int]]
    diagnostics: dict


def sample_training_data(
    path: Path,
    boundaries: TemporalBoundaries,
    max_events: int,
    seed: int,
    positive_threshold: float = 1.0,
) -> SampledTrainingData:
    """Uniform event reservoir across the COMPLETE positive training split.

    Each positive event (including repeats) has equal selection probability.
    Memory is O(sample size + distinct training-positive pairs + catalog).
    All training observations define ID maps; held-out rows define nothing.
    Coverage of every user/item/day is measured, not guaranteed by sampling.
    Reproducibility requires the same source bytes, order, seed and settings.
    """
    import hashlib
    import json
    import random
    from collections import Counter, defaultdict

    if max_events <= 0:
        raise ValueError("max_events must be positive")
    if not 0 <= positive_threshold < float("inf"):
        raise ValueError("Invalid positive threshold")

    rng = random.Random(seed)
    users, items = set(), set()
    history = defaultdict(set)
    reservoir = []
    days = Counter()
    train_events = positives = 0
    for index, event in enumerate(iter_kuairec(path)):
        if assign_split(event, boundaries) != "train":
            continue
        train_events += 1
        users.add(event.user_id)
        items.add(event.video_id)
        if event.watch_ratio < positive_threshold:
            continue
        positives += 1
        history[event.user_id].add(event.video_id)
        days[int(event.timestamp // 86400)] += 1
        row = (event.user_id, event.video_id, event.timestamp, index)
        if len(reservoir) < max_events:
            reservoir.append(row)
        else:
            slot = rng.randrange(positives)
            if slot < max_events:
                reservoir[slot] = row

    user_map = {user: index for index, user in enumerate(sorted(users))}
    item_map = {item: index for index, item in enumerate(sorted(items))}
    pairs = tuple((user_map[u], item_map[i]) for u, i, _, _ in reservoir)
    # Reuse mapped item integers across users rather than allocate one per pair.
    known = {
        user_map[user]: frozenset(item_map[item] for item in positive_items)
        for user, positive_items in history.items()
    }
    sampled_users = {u for u, _, _, _ in reservoir}
    sampled_items = {i for _, i, _, _ in reservoir}
    positive_items = set().union(*history.values()) if history else set()
    sample_days = Counter(int(t // 86400) for _, _, t, _ in reservoir)
    diagnostics = {
        "strategy": "uniform_positive_event_reservoir_v1",
        "seed": seed,
        "max_events": max_events,
        "training_events": train_events,
        "training_positive_events": positives,
        "known_positive_pairs": sum(map(len, known.values())),
        "training_users": len(users), "training_items": len(items),
        "positive_users": len(history), "positive_items": len(positive_items),
        "sampled_events": len(pairs),
        "sampled_users": len(sampled_users), "sampled_items": len(sampled_items),
        "user_coverage": len(sampled_users) / len(users) if users else 0.0,
        "item_coverage": len(sampled_items) / len(items) if items else 0.0,
        "positive_user_coverage": len(sampled_users) / len(history) if history else 0.0,
        "positive_item_coverage": len(sampled_items) / len(positive_items) if positive_items else 0.0,
        "training_positive_events_by_utc_day": dict(sorted(days.items())),
        "sampled_events_by_utc_day": dict(sorted(sample_days.items())),
        "sample_sha256": hashlib.sha256(
            json.dumps(reservoir, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return SampledTrainingData(
        TrainingData(user_map, item_map, pairs), known, diagnostics,
    )
