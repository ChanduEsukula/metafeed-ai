"""Validation-only checkpoint comparisons using the parity-tested evaluator."""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from metafeed.data_pipeline.kuairec import iter_kuairec
from metafeed.evaluation.offline import evaluate_ranker
from metafeed.evaluation.popularity import PopularityBaseline
from metafeed.evaluation.two_tower import TwoTowerRanker


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def population_counters(result):
    return {key: value for key, value in asdict(result).items() if key != "metrics"}


def controlled_settings(ranker):
    """Epoch budget may vary; sample, seed, architecture and optimizer must match."""
    settings = ranker.experiment_settings
    sample = ranker.sampling
    required = ("seed", "embedding_dim", "batch_size", "training_events")
    if any(type(settings.get(key)) is not int for key in required):
        raise ValueError("Checkpoint lacks deterministic experiment settings")
    if not sample.get("sample_sha256") or sample.get("sampled_events") != settings["training_events"]:
        raise ValueError("Checkpoint lacks consistent sample provenance")
    if settings["positive_lookup_scope"] != "complete_training_split":
        raise ValueError("Complete training-positive history is required")
    if settings["selection"] != "uniform_positive_event_reservoir_v1":
        raise ValueError("Expected deterministic positive-event reservoir")
    return {
        **{key: settings[key] for key in required},
        "sample_sha256": sample["sample_sha256"],
        "sample_max_events": sample.get("max_events"),
        "mapping_sha256": hashlib.sha256(json.dumps({
            "users": sorted(ranker.user_map.items()),
            "items": sorted(ranker.item_map.items()),
        }, separators=(",", ":")).encode()).hexdigest(),
        "selection": settings["selection"],
        "positive_lookup_scope": settings["positive_lookup_scope"],
        # Original archived checkpoints used these fixed values before recording them.
        "learning_rate": settings["learning_rate"] if settings["learning_rate"] is not None else 0.001,
        "temperature": settings["temperature"] if settings["temperature"] is not None else 0.1,
    }


def rank_checkpoint_results(rows):
    """NDCG@10 descending; exact ties prefer earlier epoch, then identifier.

    Recall/HitRate are reported, never used to break selection ties.
    """
    if not rows:
        raise ValueError("No checkpoint results to select")
    identifiers = [row["checkpoint"] for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("Duplicate checkpoint identifiers")
    for row in rows:
        if row["evaluated_users"] <= 0:
            raise ValueError("Cannot select a checkpoint without eligible validation users")
        for metric in ("NDCG@10", "Recall@10", "HitRate@10"):
            value = row[metric]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid validation metric: {metric}")
    ordered = sorted(rows, key=lambda row: (
        -row["NDCG@10"], row["completed_epochs"], row["checkpoint"],
    ))
    return [{**row, "validation_rank": index} for index, row in enumerate(ordered, 1)]


def evaluate_checkpoints(checkpoints, source, config):
    """Fit popularity once; evaluate each immutable checkpoint on validation.

    Reuses the original population/candidate/metric loop; no gradient updates,
    held-out features, test metrics, or changes to training data preparation.
    """
    paths = [Path(path).resolve() for path in checkpoints]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide unique checkpoint paths")
    config = replace(config, split="validation")
    if 10 not in config.k_values:
        raise ValueError("Validation selection requires metrics at K=10")
    started = time.perf_counter()
    source_hash = sha256_file(source)
    baseline = PopularityBaseline()
    baseline.fit(iter_kuairec(source), config.boundaries, config.positive_threshold)
    shared_seconds = time.perf_counter() - started
    print("Evaluating popularity on validation...", flush=True)
    baseline_start = time.perf_counter()
    baseline_result = evaluate_ranker(source, config, baseline)
    baseline_seconds = time.perf_counter() - baseline_start
    counters = population_counters(baseline_result)
    reference_settings = None
    rows = []
    for path in paths:
        print(f"Evaluating validation checkpoint: {path.name}", flush=True)
        checkpoint_start = time.perf_counter()
        ranker = TwoTowerRanker(path, config)
        if ranker.source_sha256 != source_hash:
            raise ValueError("Checkpoint dataset fingerprint does not match source")
        ranker.validate_population(baseline.training_users, baseline.catalog)
        settings = controlled_settings(ranker)
        if reference_settings is not None and settings != reference_settings:
            raise ValueError("Checkpoints have different sample/seed/architecture/optimizer settings")
        reference_settings = settings
        epoch = ranker.experiment_settings["completed_epochs"]
        diagnostics = ranker.epoch_diagnostics
        if epoch is None and diagnostics:
            epoch = diagnostics[-1]["epoch"]  # compatible with original final-only checkpoints
        if type(epoch) is not int or epoch <= 0 or [d["epoch"] for d in diagnostics] != list(range(1, epoch + 1)):
            raise ValueError("Checkpoint completed epoch disagrees with diagnostics")
        budget = ranker.experiment_settings["epochs"]
        if type(budget) is not int or budget <= 0 or epoch > budget:
            raise ValueError("Checkpoint epoch exceeds requested epoch budget")
        result = evaluate_ranker(source, config, ranker)
        if population_counters(result) != counters:
            raise RuntimeError("Population/coverage parity failed")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows.append({
            "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            "completed_epochs": epoch, "requested_epochs": ranker.experiment_settings["epochs"],
            **settings, **{metric: result.metrics[metric] for metric in ("Recall@10", "NDCG@10", "HitRate@10")},
            "evaluated_users": result.evaluated_users,
            "population_coverage_parity": True,
            "evaluation_seconds": time.perf_counter() - checkpoint_start,
            "process_lifetime_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
            "training_epoch_diagnostics": diagnostics,
        })
        del ranker
    ranked = rank_checkpoint_results(rows)
    return {
        "split": "validation", "selection_metric": "NDCG@10",
        "tie_breaking": "earlier completed epoch, then checkpoint identifier; secondary metrics not used",
        "source_sha256": source_hash,
        "effective_config": {**asdict(config), "source_file": str(source)},
        "shared_preparation_seconds": shared_seconds,
        "popularity": {**asdict(baseline_result), "evaluation_seconds": baseline_seconds},
        "population_coverage_counters": counters,
        "comparison": ranked, "best_checkpoint": ranked[0],
        "total_elapsed_seconds": time.perf_counter() - started,
    }
