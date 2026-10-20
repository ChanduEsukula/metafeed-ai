"""Export chronological, per-user sampled KuaiRec data."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.data_pipeline.sampling import reservoir_sample_by_user

SOURCE = Path("data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv")
OUT = Path("data/processed/kuairec_colab_v2")

def timestamp(date):
    return datetime.fromisoformat(date).replace(
        tzinfo=timezone.utc
    ).timestamp()

TRAIN_END = timestamp("2020-08-10")
VALIDATION_END = timestamp("2020-08-31")


def events_between(start, end):
    for event in iter_kuairec(SOURCE):
        if start <= event.timestamp < end:
            yield event


def write_split(name, groups):
    path = OUT / f"{name}.csv"
    counts = Counter()
    rows = 0

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["user_id", "video_id", "timestamp", "label"]
        )

        for user in sorted(groups):
            for event in sorted(
                groups[user],
                key=lambda e: (e.timestamp, e.video_id),
            ):
                label = int(event.watch_ratio >= 1.0)
                writer.writerow([
                    event.user_id,
                    event.video_id,
                    event.timestamp,
                    label,
                ])
                counts[label] += 1
                rows += 1

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "rows": rows,
        "labels": dict(counts),
        "users": len(groups),
        "sha256": checksum,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = reservoir_sample_by_user(
        events_between(float("-inf"), TRAIN_END),
        user_id=lambda e: e.user_id,
        max_per_user=100,
        seed=42,
    )

    validation = reservoir_sample_by_user(
        events_between(TRAIN_END, VALIDATION_END),
        user_id=lambda e: e.user_id,
        max_per_user=30,
        seed=43,
    )

    train_stats = write_split("train", train)
    validation_stats = write_split("validation", validation)

    manifest = {
        "version": 2,
        "train_end_utc": "2020-08-10T00:00:00Z",
        "validation_end_utc": "2020-08-31T00:00:00Z",
        "seed": 42,
        "label_definition": "watch_ratio >= 1.0",
        "selection": "per-user reservoir sampling",
        "train": train_stats,
        "validation": validation_stats,
    }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
