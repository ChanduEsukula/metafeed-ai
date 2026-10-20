import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = Path(
    "data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv"
)

def main():
    total = 0
    users = set()
    videos = set()
    missing = Counter()
    invalid = 0
    rewatched = 0
    earliest = float("inf")
    latest = float("-inf")

    with DATASET.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        print("Dataset columns:", reader.fieldnames)

        for row in reader:
            total += 1
            users.add(row["user_id"])
            videos.add(row["video_id"])

            for key, value in row.items():
                if value is None or not value.strip():
                    missing[key] += 1

            try:
                play = float(row["play_duration"])
                duration = float(row["video_duration"])
                ratio = float(row["watch_ratio"])
                timestamp = float(row["timestamp"])

                if (
                    not all(map(math.isfinite, (
                        play, duration, ratio, timestamp
                    )))
                    or play < 0
                    or duration <= 0
                    or ratio < 0
                ):
                    invalid += 1
                    continue

                if ratio > 1:
                    rewatched += 1

                earliest = min(earliest, timestamp)
                latest = max(latest, timestamp)

            except (ValueError, TypeError):
                invalid += 1

    print("\n=== KuaiRec 2.0 Audit ===")
    print(f"Interactions: {total:,}")
    print(f"Unique users: {len(users):,}")
    print(f"Unique videos: {len(videos):,}")
    print(f"Missing values: {dict(missing)}")
    print(f"Invalid numeric rows: {invalid:,}")
    print(f"Watch ratio above 1: {rewatched:,}")

    if earliest != float("inf"):
        print(
            "Earliest interaction:",
            datetime.fromtimestamp(
                earliest, timezone.utc
            ).isoformat()
        )
        print(
            "Latest interaction:",
            datetime.fromtimestamp(
                latest, timezone.utc
            ).isoformat()
        )

if __name__ == "__main__":
    main()
