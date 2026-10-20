"""Binary recommendation ranking metrics."""

import math
from collections.abc import Sequence


def _validate_recommendations(
    recommended: Sequence[int],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(recommended) != len(set(recommended)):
        raise ValueError("recommendations must contain unique item IDs")


def recall_at_k(
    recommended: Sequence[int],
    relevant: set[int],
    k: int,
) -> float:
    _validate_recommendations(recommended, k)

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: Sequence[int],
    relevant: set[int],
    k: int,
) -> float:
    _validate_recommendations(recommended, k)

    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )

    idcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(k, len(relevant)))
    )

    return dcg / idcg
