"""Audit repeated interactions and watch-ratio consistency."""

from collections import Counter
from pathlib import Path

from metafeed.data_pipeline.kuairec import iter_kuairec

DATA = Path(
    "data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv"
)


def main():
    total = 0
    repeated_events = 0
    pair_counts = Counter()

    ratio_mismatches = 0
    max_ratio_error = 0.0
    tolerance = 1e-5

    for event in iter_kuairec(DATA):
        total += 1

        pair = (event.user_id, event.video_id)

        if pair_counts[pair] > 0:
            repeated_events += 1

        pair_counts[pair] += 1

        expected = event.play_duration / event.video_duration
        error = abs(expected - event.watch_ratio)

        max_ratio_error = max(max_ratio_error, error)

        if error > tolerance * max(1.0, abs(expected)):
            ratio_mismatches += 1

    repeated_pairs = sum(
        count > 1 for count in pair_counts.values()
    )

    print("\n=== KuaiRec Interaction Quality Audit ===")
    print(f"Total events: {total:,}")
    print(f"Unique user-video pairs: {len(pair_counts):,}")
    print(f"Events beyond first per pair: {repeated_events:,}")
    print(f"Pairs appearing more than once: {repeated_pairs:,}")
    print(f"Watch-ratio mismatches: {ratio_mismatches:,}")
    print(f"Maximum absolute ratio error: {max_ratio_error:.10f}")


if __name__ == "__main__":
    main()
