
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
