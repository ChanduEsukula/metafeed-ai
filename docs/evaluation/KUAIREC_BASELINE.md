# KuaiRec 2.0 Offline Popularity Baseline

## Protocol

- Dataset: KuaiRec 2.0 big_matrix.csv
- Task: Fixed-period held-out recommendation evaluation
- Training cutoff: 2020-08-10 00:00:00 UTC
- Validation cutoff: 2020-08-31 00:00:00 UTC
- Positive label: watch_ratio >= 1.0
- Evaluation split: Test
- Candidate catalog: Items observed in training
- Popularity: Positive training-event counts, including repeated events
- History exclusions: Positive events in training and validation
- Ground truth: Distinct in-catalog test positives not previously seen
- Ranking: Popularity descending, item ID ascending for ties
- Negative sampling: None
- Metrics: User-macro Recall@10, NDCG@10, HitRate@10
- Cold-start users: Excluded and reported

## Dataset Splits

| Split | Interactions |
|---|---:|
| Train | 8,787,755 |
| Validation | 2,130,071 |
| Test | 1,612,980 |
| Total | 12,530,806 |

## Coverage

| Population | Count |
|---|---:|
| Distinct positive test user-item pairs | 434,372 |
| Pairs outside training catalog | 180,853 |
| In-catalog positive pairs | 253,519 |
| Previously seen in-catalog pairs | 23,124 |
| Remaining eligible positive pairs | 230,395 |
| Evaluated users | 6,851 |
| Cold-start users | 2 |
| Users with empty eligible ground truth | 41 |

Coverage counts are pair-level statistics across target users,
including cold-start users. Ranking metrics are calculated
only over evaluated warm-start users.

## Popularity Baseline Results

| Metric | Result |
|---|---:|
| Recall@10 | 0.0006709216527837271 |
| NDCG@10 | 0.002553498612838516 |
| HitRate@10 | 0.03138227995913005 |

## Execution

- Machine: Apple Silicon Mac
- Python: 3.13.7
- Coverage run runtime: 67.903 seconds
- Coverage run peak RSS: 577.72 MiB
- Test suite: 86 passed, 1 dependency warning

## Reproducibility

Two complete evaluations produced identical metrics,
split counts, user counts, and dataset/config fingerprints.

Dataset SHA-256:
4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9

Configuration SHA-256:
195ddccb1234461305055ce5a1099c7f395437f35b04aee285fb67c8b474fc01

Reproduction:

    python scripts/run_kuairec_benchmark.py --run-id run1
    python scripts/run_kuairec_benchmark.py --run-id run2

Use a fresh output directory when repeating runs.
The runner intentionally refuses to overwrite reports.

## Limitations

- Scores represent warm-catalog, fixed-period offline evaluation.
- Items first appearing after the training cutoff cannot be recommended.
- Positive test items previously observed by the user are excluded.
- Candidate ranking uses all eligible training-catalog items.
- Results are not online engagement or CTR measurements.
- No learned personalization model is evaluated yet.
- The original raw dataset is not committed to Git.
