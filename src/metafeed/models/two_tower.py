
"""PyTorch two-tower recommendation model."""

import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerModel(nn.Module):
    """Learn normalized user and item representations."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        if num_users <= 0 or num_items <= 0:
            raise ValueError("User and item counts must be positive")

        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.user_embedding = nn.Embedding(
            num_users, embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_items, embedding_dim
        )

        nn.init.normal_(
            self.user_embedding.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def encode_users(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.user_embedding(user_ids)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_items(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.item_embedding(item_ids)
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return paired user-item similarity scores."""
        users = self.encode_users(user_ids)
        items = self.encode_items(item_ids)

        return (users * items).sum(dim=-1)

    def contrastive_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """In-batch contrastive loss for positive interactions."""

        if user_ids.ndim != 1 or positive_item_ids.ndim != 1:
            raise ValueError("Expected one-dimensional ID tensors")

        if user_ids.shape[0] != positive_item_ids.shape[0]:
            raise ValueError("Batch sizes must match")

        if user_ids.numel() == 0:
            raise ValueError("Batch cannot be empty")

        users = self.encode_users(user_ids)
        items = self.encode_items(positive_item_ids)

        logits = users @ items.T
        logits = logits / self.temperature

        labels = torch.arange(
            user_ids.shape[0],
            device=user_ids.device,
        )

        return F.cross_entropy(logits, labels)
