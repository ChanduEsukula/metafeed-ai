
import pytest
import torch

from metafeed.training.losses import (
    build_positive_lookup,
    masked_contrastive_loss,
)


def test_lookup_deduplicates_repeated_events():
    lookup = build_positive_lookup((
        (1, 10),
        (1, 10),
        (1, 20),
    ))

    assert lookup[1] == frozenset({10, 20})


def test_known_positive_is_masked():
    users = torch.tensor([1, 2])
    items = torch.tensor([10, 20])

    user_vectors = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    item_vectors = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    lookup = {
        1: frozenset({10, 20}),
        2: frozenset({20}),
    }

    loss = masked_contrastive_loss(
        user_vectors,
        item_vectors,
        users,
        items,
        lookup,
    )

    expected = torch.nn.functional.cross_entropy(
        torch.tensor([
            [10.0, torch.finfo(torch.float32).min],
            [0.0, 10.0],
        ]),
        torch.tensor([0, 1]),
    )

    assert torch.allclose(loss, expected)


def test_missing_target_positive_rejected():
    with pytest.raises(ValueError, match="Target pair"):
        masked_contrastive_loss(
            torch.eye(2),
            torch.eye(2),
            torch.tensor([1, 2]),
            torch.tensor([10, 20]),
            {1: frozenset({10}), 2: frozenset()},
        )


def test_duplicate_items_rejected():
    with pytest.raises(ValueError, match="unique"):
        masked_contrastive_loss(
            torch.eye(2),
            torch.eye(2),
            torch.tensor([1, 2]),
            torch.tensor([10, 10]),
            {
                1: frozenset({10}),
                2: frozenset({10}),
            },
        )


def test_single_example_has_finite_loss():
    loss = masked_contrastive_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1]),
        torch.tensor([10]),
        {1: frozenset({10})},
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_gradients_are_finite():
    user_vectors = torch.randn(3, 8, requires_grad=True)
    item_vectors = torch.randn(3, 8, requires_grad=True)

    loss = masked_contrastive_loss(
        user_vectors,
        item_vectors,
        torch.tensor([1, 2, 3]),
        torch.tensor([10, 20, 30]),
        {
            1: frozenset({10, 20}),
            2: frozenset({20}),
            3: frozenset({30}),
        },
    )

    loss.backward()

    assert torch.isfinite(user_vectors.grad).all()
    assert torch.isfinite(item_vectors.grad).all()


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError):
        masked_contrastive_loss(
            torch.eye(2),
            torch.eye(2),
            torch.tensor([1, 2]),
            torch.tensor([10, 20]),
            {1: frozenset({10}), 2: frozenset({20})},
            temperature=0,
        )
