"""Validation-first, full-catalog two-tower/popularity parity benchmark."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import torch

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.evaluation.config import load_evaluation_config
from metafeed.evaluation.offline import evaluate_ranker
from metafeed.evaluation.popularity import PopularityBaseline
from metafeed.evaluation.two_tower import TwoTowerRanker

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets/kuairec.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Report already exists: {args.output}")
    # Validation only: the repository's default test config is not mutated.
    config = replace(load_evaluation_config(args.config), split="validation")
    source = args.source or ROOT / config.source_file
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    dataset_hash = sha256(source)
    model = TwoTowerRanker(args.checkpoint, config)
    if not model.source_sha256 or model.source_sha256 != dataset_hash:
        raise ValueError("Checkpoint dataset fingerprint does not match source")
    baseline = PopularityBaseline()
    print("Preparing full training catalog and baseline statistics...", flush=True)
    baseline.fit(iter_kuairec(source), config.boundaries, config.positive_threshold)
    model.validate_population(baseline.training_users, baseline.catalog)
    shared_seconds = time.perf_counter() - started
    reports = {}
    for name, ranker in (("popularity", baseline), ("two_tower", model)):
        print(f"Evaluating {name} on validation...", flush=True)
        start = time.perf_counter()
        result = evaluate_ranker(source, config, ranker)
        reports[name] = {
            **asdict(result), "evaluation_seconds": time.perf_counter() - start,
            "process_lifetime_peak_rss_mib": peak_rss_mib(),
        }
        print(json.dumps(reports[name], sort_keys=True), flush=True)
    population = lambda r: {k: v for k, v in r.items() if k not in {
        "metrics", "evaluation_seconds", "process_lifetime_peak_rss_mib",
    }}
    if population(reports["popularity"]) != population(reports["two_tower"]):
        raise RuntimeError("Population/coverage parity failed")
    report = {
        "protocol": "fixed_period_held_out_v1", "split": "validation",
        "source_sha256": dataset_hash, "checkpoint_sha256": sha256(args.checkpoint),
        "config_sha256": sha256(args.config), "dataset_config": json.loads(args.config.read_text()),
        "effective_evaluation_config": {**asdict(config), "source_file": str(source)},
        "catalog_items": len(model.catalog), "training_users": len(model.training_users),
        "checkpoint_sampling": model.sampling, "population_coverage_parity": True,
        "shared_preparation_seconds": shared_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "reports": reports, "device": "cpu", "torch_num_threads": 1,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "candidate_policy": "full training catalog minus prior positives; no negative sampling",
        "ranking": "normalized exact inner product; ties ascending raw video ID",
        "command": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Parity PASS. Report: {args.output}")


if __name__ == "__main__":
    main()
