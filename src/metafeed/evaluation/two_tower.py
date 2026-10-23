"""CPU exact-scoring adapter for safely restored two-tower checkpoints."""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from metafeed.evaluation.config import EvaluationConfig
from metafeed.models.two_tower import TwoTowerModel


def _validate_map(mapping, name):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Invalid {name}")
    if any(type(key) is not int or key < 0 for key in mapping):
        raise ValueError(f"Invalid raw IDs in {name}")
    if any(type(index) is not int for index in mapping.values()):
        raise ValueError(f"Invalid indices in {name}")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError(f"Noncontiguous or duplicate indices in {name}")


class TwoTowerRanker:
    """Score all training items, including IDs absent from optimization.

    Never substitute new/random embeddings for IDs missing from a checkpoint.
    Incompatible catalogs must fail, rather than change the evaluation population.
    """

    def __init__(self, checkpoint_path: Path, config: EvaluationConfig):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for name in ("user_map", "item_map"):
            _validate_map(checkpoint[name], name)
        dimension = checkpoint["embedding_dim"]
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("Invalid embedding dimension")
        self.user_map = checkpoint["user_map"]
        self.item_map = checkpoint["item_map"]
        self.catalog = set(self.item_map)
        self.training_users = set(self.user_map)
        metadata = checkpoint.get("dataset_config")
        if not metadata:
            raise ValueError("Checkpoint lacks dataset protocol metadata")
        target = metadata["target_definition"]
        if (
            metadata["split_strategy"] != "global_chronological"
            or datetime.fromisoformat(metadata["train_end_utc"].replace("Z", "+00:00")).timestamp()
            != config.boundaries.train_end
            or datetime.fromisoformat(metadata["validation_end_utc"].replace("Z", "+00:00")).timestamp()
            != config.boundaries.validation_end
            or target != {"field": "watch_ratio", "operator": ">=", "threshold": config.positive_threshold}
        ):
            raise ValueError("Checkpoint training protocol does not match evaluation")
        self.source_sha256 = checkpoint.get("source_sha256")
        self.sampling = checkpoint.get("sampling", {})
        self.experiment_settings = {
            key: checkpoint.get(key)
            for key in ("seed", "embedding_dim", "batch_size", "training_events",
                        "epochs", "completed_epochs", "learning_rate", "temperature",
                        "positive_lookup_scope", "selection")
        }
        self.epoch_diagnostics = checkpoint.get("epoch_diagnostics", [])
        # Preserve the caller's RNG: constructing a model must not affect it.
        with torch.random.fork_rng(devices=[]):
            self.model = TwoTowerModel(len(self.user_map), len(self.item_map), dimension)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Nonfinite checkpoint weights")
        self.raw_items = np.array(sorted(self.catalog), dtype=np.int64)
        with torch.inference_mode():
            self.item_vectors = self.model.encode_items(torch.tensor(
                [self.item_map[int(item)] for item in self.raw_items], dtype=torch.long,
            ))

    def validate_population(self, training_users: set[int], catalog: set[int]):
        if self.training_users != training_users or self.catalog != catalog:
            raise ValueError("Checkpoint maps do not match complete training population/catalog")

    @torch.inference_mode()
    def recommend(self, user_id: int, k: int, excluded_items: set[int]) -> list[int]:
        if k <= 0:
            raise ValueError("k must be positive")
        if user_id not in self.user_map:
            raise KeyError("Unknown training user")
        user_vector = self.model.encode_users(torch.tensor([self.user_map[user_id]]))[0]
        scores = (self.item_vectors @ user_vector).numpy()
        # raw_items is ascending, so stable sorting explicitly breaks ties by raw ID.
        order = np.argsort(-scores, kind="stable")
        result = []
        for index in order:
            item = int(self.raw_items[index])
            if item not in excluded_items:
                result.append(item)
                if len(result) == k:
                    break
        return result
