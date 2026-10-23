"""Rank saved two-tower epochs by validation NDCG@10; never evaluate test."""

import argparse
import glob
import json
from pathlib import Path
import platform
import sys

import torch

from metafeed.evaluation.config import load_evaluation_config
from metafeed.evaluation.experiments import evaluate_checkpoints, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths or quoted glob patterns")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets/kuairec.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Exclusive JSON comparison table")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Comparison already exists: {args.output}")
    checkpoints = []
    for pattern in args.checkpoints:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"No checkpoints match: {pattern}")
        checkpoints.extend(Path(path) for path in matched)
    config = load_evaluation_config(args.config)
    source = args.source or ROOT / config.source_file
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report = evaluate_checkpoints(checkpoints, source, config)
    report.update({
        "config_sha256": sha256_file(args.config), "command": sys.argv,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "device": "cpu", "torch_num_threads": 1,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(f"Best checkpoint by validation NDCG@10: {report['best_checkpoint']['checkpoint']}")
    print(f"Comparison: {args.output}")


if __name__ == "__main__":
    main()
