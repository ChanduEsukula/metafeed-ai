# T4 pilot: held-out validation against popularity

The checkpoint from `artifacts/colab/metafeed_t4_pilot.zip` was evaluated on
KuaiRec validation without retraining. The archive and its checkpoint bytes were
preserved. A copy was extracted to ignored
`artifacts/colab/t4_evaluation_input.pt` using exclusive creation.

## Protocol and reproduction

```sh
.venv/bin/python scripts/evaluate_two_tower.py \
  --checkpoint artifacts/colab/t4_evaluation_input.pt \
  --output artifacts/colab/t4_validation_parity.json
```

Use a new output filename for another run; existing reports are refused.
The runner is validation-only even when the input dataset configuration names
test; it records the effective configuration and leaves the file unchanged.
No real-data test metrics were computed or used to select settings.

`offline.evaluate_ranker()` contains the original popularity evaluation loop.
`evaluate_popularity()` still fits its existing training-only baseline and calls
that loop. The new runner uses the same loop for both fitted rankers. It verifies
the checkpoint's full-source fingerprint, train/validation cutoffs, positive
threshold, and exact training-user/catalog membership before evaluation.

- Training: timestamps before 2020-08-10 UTC; validation: before 2020-08-31 UTC.
- Positive: `watch_ratio >= 1.0`.
- Catalog: all 7,904 training-observed items; user maps: all 7,174 training users.
- Candidates: full training catalog minus each user's complete prior positives;
  no sampled negatives. Validation excludes train positives; the shared evaluator
  retains train + validation history exclusion for eventual test evaluation.
- Ground truth: distinct target positives in catalog minus earlier positives.
- Exclude/report cold-start users, then empty eligible ground truth, exactly as
  in the existing benchmark. All target-split users, including negative-only
  users, enter the coverage accounting.
- Ranking: normalized float32 embeddings, exact CPU inner products, ties by
  ascending raw video ID. One user scored at a time, all items cached.
- Metrics: existing Recall@10, NDCG@10, HitRate@10, macro-averaged over eligible users.

Loading uses `torch.load(..., weights_only=True, map_location='cpu')`, validates
ID maps/protocol/finite weights, and strictly loads the model. Missing IDs are
rejected rather than replaced. Users/items absent from optimization remain in
the full population/catalog with their checkpoint embeddings.

## Measured validation results

The checkpoint is the fixed 5,000-event, three-epoch T4 pilot (32 dimensions,
seed 42). Evaluation used CPU, one PyTorch thread, with no FAISS dependency.

| Metric | Popularity | Two-tower |
|---|---:|---:|
| Recall@10 | 0.0011955474721206088 | 0.0013684148085270392 |
| NDCG@10 | 0.008194794910701011 | 0.008187366911328740 |
| HitRate@10 | 0.08922248973233253 | 0.07718453476844639 |
| Evaluated users | 7,061 | 7,061 |
| Evaluation seconds, including CSV/history scan | 42.213 | 32.533 |
| Process lifetime peak RSS, MiB | 654.047 | 654.047 |

Shared preparation (source hashing, restoration, baseline training statistics,
complete-catalog verification) took 30.730 seconds. Total runtime was 105.484
seconds. RSS is the cumulative high-water mark of the same process, including
both rankers and shared preparation, not independently attributed model memory.
These timings do not isolate scoring speed or project T4 evaluation performance.

Parity passed for **every non-metric field** of `EvaluationResult`:

| Coverage counter | Both rankers |
|---|---:|
| Cold-start users | 2 |
| Empty eligible ground-truth users | 31 |
| Distinct target positive pairs | 648,113 |
| Unavailable positive user-item pairs | 158,536 |
| In-catalog positive pairs | 489,577 |
| Previously seen positive pairs | 52,223 |
| Eligible positive pairs | 437,354 |
| Train / validation / test event counts | 8,787,755 / 2,130,071 / 1,612,980 |

The existing pair counters precede cold-user exclusion; their semantics are
preserved. Test events are counted for split integrity, not evaluated here.

## Verification and limitations

Separate macOS processes, with no OpenMP override:

```sh
.venv/bin/python -m pytest tests/training tests/models -q
# 78 passed, 1 skipped (CUDA hardware unavailable)
.venv/bin/python -m pytest tests/retrieval -q
# 8 passed
.venv/bin/python -m pytest tests/data_pipeline tests/test_api.py tests/test_recommender.py -q
# 78 passed, 1 existing dependency deprecation warning
```

New tests cover validation/test population parity on synthetic data, exclusions,
cold/empty users, duplicate targets, unsampled and negative-only training IDs,
exact normalized scores, raw-ID ties, deterministic repeated evaluation, malformed
maps/weights/protocols, restricted loading, checkpoint/RNG preservation, source
fingerprint mismatches, and validation-only/exclusive CLI reports.

Recall is higher, HitRate is lower, and NDCG is slightly lower. This single small
pilot does not establish overall improvement or statistical significance. Only
3,179 users and 2,314 items appeared in optimization; the remaining training-known
IDs stay in evaluation. Cross-hardware/version bitwise equality is not promised.
No hyperparameters were selected using test results.

Provenance:

- Source SHA-256: `4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9`.
- Evaluated checkpoint SHA-256: `d0347d6607efe99ae2972683b259cf433724f66be4f201225110f8376696a586`.
- Archive SHA-256 before/after: `4394e3c69443c303fd38ce50f45514b136e0d65dd3c0a583010692a9a9b028b0`.

Full machine-readable results are in ignored
`artifacts/colab/t4_validation_parity.json`. No experiment artifacts, dataset,
checkpoint, dependencies, or default dataset configuration were modified.
