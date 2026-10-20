import numpy as np
import pytest

from metafeed.retrieval.faiss_index import VideoVectorIndex


def test_exact_search_matches_numpy():
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)

    video_ids = [101, 202, 303]
    query = np.array([1.0, 0.2, 0.0], dtype=np.float32)

    index = VideoVectorIndex(dimension=3)
    index.build(video_ids, vectors)

    results = index.search(query, k=3)

    scores = vectors @ query
    expected = np.argsort(-scores)

    assert [video_id for video_id, _ in results] == [
        video_ids[i] for i in expected
    ]

    np.testing.assert_allclose(
        [score for _, score in results],
        scores[expected],
        atol=1e-6,
    )


def test_duplicate_video_ids_rejected():
    index = VideoVectorIndex(dimension=2)

    with pytest.raises(ValueError, match="duplicate"):
        index.build(
            [10, 10],
            [[1.0, 0.0], [0.0, 1.0]],
        )


def test_invalid_embeddings_rejected():
    index = VideoVectorIndex(dimension=2)

    with pytest.raises(ValueError):
        index.build([1], [[float("nan"), 0.0]])


def test_invalid_query_rejected():
    index = VideoVectorIndex(dimension=2)
    index.build([1], [[1.0, 0.0]])

    with pytest.raises(ValueError):
        index.search([float("inf"), 0.0], k=1)


def test_save_load_parity(tmp_path):
    index = VideoVectorIndex(dimension=2)
    index.build(
        [101, 202],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    before = index.search([0.9, 0.1], k=2)

    index.save(tmp_path)
    restored = VideoVectorIndex.load(tmp_path)

    assert restored.search([0.9, 0.1], k=2) == before


def test_search_k_larger_than_catalog():
    index = VideoVectorIndex(dimension=2)
    index.build([101], [[1.0, 0.0]])

    assert len(index.search([1.0, 0.0], k=10)) == 1


def test_rebuilding_index_replaces_old_vectors():
    index = VideoVectorIndex(dimension=2)

    index.build([101], [[1.0, 0.0]])
    index.build([202], [[0.0, 1.0]])

    assert index.index.ntotal == 1
    assert index.search([0.0, 1.0], k=1)[0][0] == 202


def test_nonpositive_k_rejected():
    index = VideoVectorIndex(dimension=2)
    index.build([101], [[1.0, 0.0]])

    for k in [0, -1]:
        with pytest.raises(ValueError, match="positive"):
            index.search([1.0, 0.0], k=k)
