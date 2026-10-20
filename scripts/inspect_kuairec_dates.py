from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from metafeed.data_pipeline.kuairec import iter_kuairec

DATA = Path(
    "data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv"
)

counts = Counter()
total = 0

for event in iter_kuairec(DATA):
    day = datetime.fromtimestamp(
        event.timestamp, timezone.utc
    ).date().isoformat()

    counts[day] += 1
    total += 1

running = 0
cutoffs = {}
targets = {
    "train_70": total * 0.70,
    "validation_85": total * 0.85,
}

print("=== KuaiRec UTC Daily Distribution ===")

for day in sorted(counts):
    running += counts[day]

    for name, target in targets.items():
        if name not in cutoffs and running >= target:
            cutoffs[name] = day

print("Total interactions:", f"{total:,}")
print("Observed days:", len(counts))
print("70% boundary day:", cutoffs["train_70"])
print("85% boundary day:", cutoffs["validation_85"])

print("\nLast 15 observed days:")
for day in sorted(counts)[-15:]:
    print(day, f"{counts[day]:,}")
