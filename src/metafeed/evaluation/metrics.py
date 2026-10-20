"""Recommendation ranking metrics for held-out relevant items."""

import math
from collections.abc import Sequence


def recall_at_k(
    recommended: Sequence[int],
    relevant: set[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant:
        return 0.0

    retrieved = set(recommended[:k])
    return len(retrieved & relevant) / len(relevant)


def ndcg_at_k(
    recommended: Sequence[int],
    relevant: set[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )

    ideal_hits = min(k, len(relevant))

    idcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0
