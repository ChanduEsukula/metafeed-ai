
import pytest
import torch

from metafeed.models.two_tower import TwoTowerModel


def test_model_output_shape():
    model = TwoTowerModel(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])

    scores = model(users, items)

    assert scores.shape == (3,)


def test_embedding_dimensions():
    model = TwoTowerModel(10, 20, embedding_dim=32)

    users = model.encode_users(torch.tensor([1, 2]))
    items = model.encode_items(torch.tensor([3, 4]))

    assert users.shape == (2, 32)
    assert items.shape == (2, 32)


def test_embeddings_are_normalized():
    model = TwoTowerModel(10, 20)

    vectors = model.encode_users(torch.tensor([1, 2]))

    norms = torch.linalg.vector_norm(vectors, dim=1)

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_similarity_is_bounded():
    model = TwoTowerModel(10, 20)

    scores = model(
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
    )

    assert torch.all(scores <= 1.00001)
    assert torch.all(scores >= -1.00001)


def test_contrastive_loss_is_finite():
    model = TwoTowerModel(10, 20)

    loss = model.contrastive_loss(
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4, 5]),
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_training_updates_embeddings():
    torch.manual_seed(42)

    model = TwoTowerModel(10, 20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])

    before = model.user_embedding.weight.detach().clone()

    loss = model.contrastive_loss(users, items)
    loss.backward()
    optimizer.step()

    after = model.user_embedding.weight.detach()

    assert not torch.equal(before, after)


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        TwoTowerModel(0, 20)

    with pytest.raises(ValueError):
        TwoTowerModel(10, 0)

    with pytest.raises(ValueError):
        TwoTowerModel(10, 20, embedding_dim=0)

    with pytest.raises(ValueError):
        TwoTowerModel(10, 20, temperature=0)


def test_invalid_batch_shape():
    model = TwoTowerModel(10, 20)

    with pytest.raises(ValueError):
        model.contrastive_loss(
            torch.tensor([[1, 2]]),
            torch.tensor([3, 4]),
        )


def test_invalid_batch_size():
    model = TwoTowerModel(10, 20)

    with pytest.raises(ValueError):
        model.contrastive_loss(
            torch.tensor([1, 2]),
            torch.tensor([3]),
        )


def test_empty_batch_rejected():
    model = TwoTowerModel(10, 20)

    with pytest.raises(ValueError):
        model.contrastive_loss(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
