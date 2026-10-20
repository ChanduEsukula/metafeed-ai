"""Exact inner-product retrieval with explicit video IDs."""

from pathlib import Path

import faiss
import numpy as np


class VideoVectorIndex:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.video_ids = np.empty(0, dtype=np.int64)

    def build(self, video_ids, embeddings):
        ids = np.asarray(video_ids, dtype=np.int64)
        vectors = np.asarray(embeddings, dtype=np.float32)

        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("video_ids must be a nonempty 1D array")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate video IDs")
        if vectors.shape != (len(ids), self.dimension):
            raise ValueError("embedding shape mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must be finite")

        new_index = faiss.IndexFlatIP(self.dimension)
        new_index.add(np.ascontiguousarray(vectors))

        self.index = new_index
        self.video_ids = ids.copy()

    def search(self, user_vector, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        if self.index.ntotal == 0:
            raise ValueError("index is empty")

        vector = np.asarray(user_vector, dtype=np.float32)

        if vector.shape != (self.dimension,):
            raise ValueError("query vector shape mismatch")
        if not np.isfinite(vector).all():
            raise ValueError("query must be finite")

        count = min(k, self.index.ntotal)

        scores, positions = self.index.search(
            np.ascontiguousarray(vector.reshape(1, -1)),
            count,
        )

        return [
            (int(self.video_ids[pos]), float(score))
            for pos, score in zip(positions[0], scores[0])
        ]

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.index.ntotal == 0:
            raise ValueError("cannot save an empty index")

        faiss.write_index(self.index, str(path / "videos.faiss"))
        np.save(path / "video_ids.npy", self.video_ids)

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        index = faiss.read_index(str(path / "videos.faiss"))
        ids = np.load(path / "video_ids.npy", allow_pickle=False)

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("expected IndexFlatIP")
        if ids.ndim != 1 or len(ids) != index.ntotal:
            raise ValueError("invalid saved video mapping")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate saved video IDs")

        instance = cls(index.d)
        instance.index = index
        instance.video_ids = ids.astype(np.int64, copy=True)

        return instance
