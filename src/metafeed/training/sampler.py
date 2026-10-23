
"""Deterministic, coverage-preserving recommendation batches."""

import random
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

Pair = tuple[int, int]


@dataclass(frozen=True)
class BatchDiagnostics:
    interactions: int
    batches: int
    min_batch_size: int
    max_batch_size: int
    mean_batch_size: float
    valid_negative_fraction: float
    valid_negatives: int
    possible_negatives: int
    effective_interactions: int


def iter_training_batches(
    pairs: Sequence[Pair],
    batch_size: int,
    seed: int,
    epoch: int = 0,
) -> Iterator[list[Pair]]:
    """Yield every event once, with unique item IDs per batch.

    Users may repeat. Repeated events remain separate training rows.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")

    rng = random.Random(seed + epoch)
    order = list(range(len(pairs)))
    rng.shuffle(order)

    by_item: dict[int, deque[Pair]] = defaultdict(deque)

    for index in order:
        user, item = pairs[index]
        by_item[item].append((user, item))

    active = deque(by_item.keys())

    while active:
        batch: list[Pair] = []

        for _ in range(min(batch_size, len(active))):
            item = active.popleft()
            pair = by_item[item].popleft()
            batch.append(pair)

            if by_item[item]:
                active.append(item)

        yield batch


def batch_diagnostics(
    batches: Sequence[Sequence[Pair]],
    known_positives: Mapping[int, frozenset[int]],
) -> BatchDiagnostics:
    """Summarize event visits and usable off-diagonal negatives.

    Effective interactions count anchor rows with at least one valid
    negative. The negative fraction is weighted by all off-diagonal
    slots, including slots in batches without valid negatives.
    """

    sizes = [len(batch) for batch in batches]
    interactions = sum(sizes)

    valid = 0
    possible = 0
    effective = 0

    for batch in batches:
        for row, (user, target) in enumerate(batch):
            row_valid = 0
            positives = known_positives.get(user, frozenset())

            if target not in positives:
                raise ValueError("Target missing from known positives")

            for column, (_, candidate_item) in enumerate(batch):
                if row == column:
                    continue

                possible += 1
                if candidate_item not in positives:
                    valid += 1
                    row_valid += 1
            effective += int(row_valid > 0)

    return BatchDiagnostics(
        interactions=interactions,
        batches=len(batches),
        min_batch_size=min(sizes, default=0),
        max_batch_size=max(sizes, default=0),
        mean_batch_size=(
            interactions / len(batches) if batches else 0.0
        ),
        valid_negative_fraction=(
            valid / possible if possible else 0.0
        ),
        valid_negatives=valid,
        possible_negatives=possible,
        effective_interactions=effective,
    )
