
"""Reproducible KuaiRec offline benchmark runner."""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from metafeed.evaluation.config import load_evaluation_config
from metafeed.evaluation.offline import evaluate_popularity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/datasets/kuairec.json"
DEFAULT_OUTPUT = ROOT / "data/processed/kuairec/evaluation"


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def peak_memory_mib() -> float:
    """Return peak process RSS in MiB."""
    peak = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    divisor = (
        1024 * 1024
        if sys.platform == "darwin"
        else 1024
    )

    return round(peak / divisor, 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the KuaiRec offline popularity benchmark."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--run-id",
        choices=("run1", "run2", "coverage"),
        required=True,
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_evaluation_config(config_path)

    source_path = config.source_file

    if not source_path.is_absolute():
        source_path = ROOT / source_path

    source_path = source_path.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    output_dir = args.output_dir.resolve()
    output_path = output_dir / f"{args.run_id}.json"

    if output_path.exists():
        raise FileExistsError(
            f"Report already exists: {output_path}"
        )

    print(
        "MetaFeed AI — KuaiRec Offline Benchmark",
        flush=True,
    )
    print(f"Run: {args.run_id}", flush=True)
    print(f"Dataset: {source_path}", flush=True)
    print(f"Evaluation split: {config.split}", flush=True)
    print("Evaluating full dataset...", flush=True)

    start = time.perf_counter()

    result = evaluate_popularity(
        source_path,
        config,
    )

    elapsed = time.perf_counter() - start

    report = {
        "protocol": "fixed_period_held_out_v1",
        "dataset": "KuaiRec 2.0",
        "evaluation_split": config.split,
        "positive_threshold": config.positive_threshold,
        "k_values": list(config.k_values),
        "train_end": config.boundaries.train_end,
        "validation_end": config.boundaries.validation_end,
        "dataset_sha256": sha256_file(source_path),
        "config_sha256": sha256_file(config_path),
        "split_events": result.split_events,
        "evaluated_users": result.evaluated_users,
        "cold_start_users": result.cold_start_users,
        "empty_ground_truth_users": (
            result.empty_ground_truth_users
        ),
        "unavailable_positive_user_item_pairs": (
            result.unavailable_positive_items
        ),

        # Ground-truth and catalog coverage statistics
        "distinct_target_positive_pairs": (
            result.distinct_target_positive_pairs
        ),
        "in_catalog_positive_pairs": (
            result.in_catalog_positive_pairs
        ),
        "eligible_positive_pairs": (
            result.eligible_positive_pairs
        ),
        "previously_seen_positive_pairs": (
            result.previously_seen_positive_pairs
        ),

        "metrics": result.metrics,
        "runtime_seconds": round(elapsed, 3),
        "peak_rss_mib": peak_memory_mib(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "run_id": args.run_id,
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
