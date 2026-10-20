"""Deterministic per-user interaction sampling."""

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def sample_by_user(
    records: Iterable[T],
    user_id: callable,
    max_per_user: int,
    seed: int = 42,
) -> list[T]:
    """Sample up to max_per_user records per user."""
    if max_per_user <= 0:
        raise ValueError("max_per_user must be positive")

    groups = defaultdict(list)

    for record in records:
        groups[user_id(record)].append(record)

    rng = random.Random(seed)
    result = []

    for user in sorted(groups):
        group = groups[user]

        if len(group) > max_per_user:
            group = rng.sample(group, max_per_user)

        result.extend(group)

    return result


def reservoir_sample_by_user(
    records: Iterable[T],
    user_id: callable,
    max_per_user: int,
    seed: int = 42,
) -> dict[int, list[T]]:
    """Bounded-memory reservoir sampling within each user."""
    if max_per_user <= 0:
        raise ValueError("max_per_user must be positive")

    rng = random.Random(seed)
    reservoirs = defaultdict(list)
    seen = defaultdict(int)

    for record in records:
        user = user_id(record)
        seen[user] += 1

        bucket = reservoirs[user]

        if len(bucket) < max_per_user:
            bucket.append(record)
        else:
            index = rng.randrange(seen[user])
            if index < max_per_user:
                bucket[index] = record

    return dict(reservoirs)
