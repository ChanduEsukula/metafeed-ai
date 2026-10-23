
"""Known-positive-aware contrastive loss for two-tower retrieval."""

from collections.abc import Mapping
import torch
from torch.nn import functional as F


def build_positive_lookup(
    positive_pairs: tuple[tuple[int, int], ...],
) -> dict[int, frozenset[int]]:
    """Deduplicate training positives for each encoded user."""

    lookup: dict[int, set[int]] = {}

    for user_id, item_id in positive_pairs:
        lookup.setdefault(user_id, set()).add(item_id)

    return {
        user_id: frozenset(items)
        for user_id, items in lookup.items()
    }


def masked_contrastive_loss(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    known_positives: Mapping[int, frozenset[int]],
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy with other known positives excluded.

    Every row's diagonal is the positive target.
    Other items known positive for that user are masked.
    Duplicate positive item IDs in a batch are rejected.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if (
        user_vectors.ndim != 2
        or item_vectors.ndim != 2
        or user_vectors.shape != item_vectors.shape
    ):
        raise ValueError("Embedding batches must have matching 2D shapes")

    batch_size = user_vectors.shape[0]

    if batch_size == 0:
        raise ValueError("Batch cannot be empty")

    if (
        user_ids.ndim != 1
        or item_ids.ndim != 1
        or user_ids.numel() != batch_size
        or item_ids.numel() != batch_size
    ):
        raise ValueError("ID batches must match embedding batch size")

    users = [int(x) for x in user_ids.detach().cpu().tolist()]
    items = [int(x) for x in item_ids.detach().cpu().tolist()]

    if len(set(items)) != batch_size:
        raise ValueError("Batch must contain unique item IDs")

    logits = (user_vectors @ item_vectors.T) / temperature

    mask = torch.zeros(
        (batch_size, batch_size),
        dtype=torch.bool,
        device=logits.device,
    )

    for row, user_id in enumerate(users):
        positives = known_positives.get(user_id, frozenset())

        if items[row] not in positives:
            raise ValueError("Target pair missing from known positives")

        for column, item_id in enumerate(items):
            if row != column and item_id in positives:
                mask[row, column] = True

    logits = logits.masked_fill(
        mask,
        torch.finfo(logits.dtype).min,
    )

    labels = torch.arange(
        batch_size,
        device=logits.device,
    )

    return F.cross_entropy(logits, labels)
