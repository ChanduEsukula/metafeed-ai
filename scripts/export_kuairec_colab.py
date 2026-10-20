"""Export a training-only KuaiRec sample for Colab."""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from metafeed.data_pipeline.kuairec import iter_kuairec

SOURCE = Path(
    "data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv"
)
OUTPUT = Path("data/processed/kuairec_colab_train.csv")
MANIFEST = Path("data/processed/kuairec_colab_manifest.json")

TRAIN_END = datetime(
    2020, 8, 10, tzinfo=timezone.utc
).timestamp()

MAX_ROWS = 100_000


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows = 0

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "video_id", "label"])

        for event in iter_kuairec(SOURCE):
            if event.timestamp >= TRAIN_END:
                continue

            label = int(event.watch_ratio >= 1.0)
            writer.writerow([event.user_id, event.video_id, label])

            counts[label] += 1
            rows += 1

            if rows >= MAX_ROWS:
                break

    manifest = {
        "source": "KuaiRec 2.0 big_matrix.csv",
        "split": "training_only",
        "train_end_utc": "2020-08-10T00:00:00Z",
        "selection": "first eligible rows in source order",
        "rows": rows,
        "label_definition": "watch_ratio >= 1.0",
        "label_counts": dict(counts),
        "purpose": "TorchRec GPU smoke training",
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print("Export:", OUTPUT)
    print("Rows:", rows)
    print("Labels:", dict(counts))
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
