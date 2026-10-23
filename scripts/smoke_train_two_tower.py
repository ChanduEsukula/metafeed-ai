
"""Small CPU/MPS smoke test for two-tower training."""

import torch

from metafeed.models.two_tower import TwoTowerModel
from metafeed.training.losses import (
    build_positive_lookup,
    masked_contrastive_loss,
)


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    pairs = (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    )

    lookup = build_positive_lookup(pairs)

    model = TwoTowerModel(
        num_users=4,
        num_items=4,
        embedding_dim=16,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    users = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
        device=device,
    )

    items = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
        device=device,
    )

    initial_loss = None
    final_loss = None

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        user_vectors = model.encode_users(users)
        item_vectors = model.encode_items(items)

        loss = masked_contrastive_loss(
            user_vectors,
            item_vectors,
            users,
            items,
            lookup,
        )

        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()

        final_loss = loss.item()

    print("Device:", device)
    print("Initial loss:", round(initial_loss, 6))
    print("Final loss:", round(final_loss, 6))

    assert final_loss < initial_loss, (
        "Training failed to reduce loss"
    )

    print("Two-tower smoke training: PASS")


if __name__ == "__main__":
    main()
