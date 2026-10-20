"""Audit sampled versus full-history KuaiRec video coverage."""

import csv
from datetime import datetime, timezone
from pathlib import Path

from metafeed.data_pipeline.kuairec import iter_kuairec

source = Path("data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv")
processed = Path("data/processed/kuairec_colab_v2")

train_cutoff = datetime(
    2020, 8, 10, tzinfo=timezone.utc
).timestamp()

def read_video_ids(path):
    with path.open(newline="") as handle:
        return {
            int(row["video_id"])
            for row in csv.DictReader(handle)
        }

sampled_train = read_video_ids(processed / "train.csv")
validation = read_video_ids(processed / "validation.csv")

full_history = set()

for event in iter_kuairec(source):
    if event.timestamp < train_cutoff:
        full_history.add(event.video_id)

artificial_cold = (
    validation - sampled_train
) & full_history

genuine_unseen = validation - full_history

print("=== MetaFeed AI Historical Catalog Audit ===")
print("Full-history training videos:", len(full_history))
print("Sampled training videos:", len(sampled_train))
print("Validation videos:", len(validation))
print("Videos absent from sampled training:",
      len(validation - sampled_train))
print("Absent due to sampling:",
      len(artificial_cold))
print("Absent from all pre-cutoff history:",
      len(genuine_unseen))

assert len(sampled_train) == 7042
assert len(validation) == 4716
assert len(validation - sampled_train) == 1762
assert artificial_cold.isdisjoint(genuine_unseen)

print("Historical catalog audit: SUCCESS")
