from collections import Counter
from pathlib import Path

from metafeed.schemas import Dataset, Recommendation


class RecommendationEngine:
    def __init__(self, dataset: Dataset):
        self.users = set(dataset.users)
        self.items = dataset.items

        self.popularity = Counter(
            event.item_id for event in dataset.interactions
        )

        self.history = {
            user_id: set() for user_id in dataset.users
        }

        for event in dataset.interactions:
            self.history[event.user_id].add(event.item_id)

    def recommend(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[Recommendation]:
        if user_id not in self.users:
            raise KeyError("Unknown user")

        if not 1 <= limit <= 100:
            raise ValueError("Invalid limit")

        candidates = [
            item_id
            for item_id in self.items
            if item_id not in self.history[user_id]
        ]

        ranked = sorted(
            candidates,
            key=lambda item_id: (
                -self.popularity[item_id],
                item_id,
            ),
        )

        return [
            Recommendation(
                item_id=item_id,
                score=self.popularity[item_id],
            )
            for item_id in ranked[:limit]
        ]


def load_dataset(path: Path) -> Dataset:
    try:
        return Dataset.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load dataset: {path}"
        ) from exc
