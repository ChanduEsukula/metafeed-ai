
"""Training-only popularity baseline for offline evaluation."""

from collections import Counter
from collections.abc import Iterable

from metafeed.data_pipeline.kuairec import KuaiRecInteraction
from metafeed.data_pipeline.splits import TemporalBoundaries, assign_split


class PopularityBaseline:
    """Rank training-observed items using training positive-event counts."""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.catalog: set[int] = set()
        self.training_users: set[int] = set()
        self.training_history: dict[int, set[int]] = {}

    def fit(
        self,
        interactions: Iterable[KuaiRecInteraction],
        boundaries: TemporalBoundaries,
        positive_threshold: float = 1.0,
    ) -> None:
        if not 0 <= positive_threshold < float("inf"):
            raise ValueError("Invalid positive threshold")

        self.counts.clear()
        self.catalog.clear()
        self.training_users.clear()
        self.training_history.clear()

        for event in interactions:
            if assign_split(event, boundaries) != "train":
                continue

            self.catalog.add(event.video_id)
            self.training_users.add(event.user_id)

            if event.watch_ratio >= positive_threshold:
                self.counts[event.video_id] += 1
                self.training_history.setdefault(
                    event.user_id, set()
                ).add(event.video_id)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        excluded_items: set[int] | None = None,
    ) -> list[int]:
        if k <= 0:
            raise ValueError("k must be positive")

        if user_id not in self.training_users:
            raise KeyError("Unknown training user")

        excluded = (
            self.training_history.get(user_id, set())
            if excluded_items is None
            else excluded_items
        )

        ranked = sorted(
            self.catalog - excluded,
            key=lambda item_id: (-self.counts[item_id], item_id),
        )

        return ranked[:k]
