# KuaiRec 2.0 — Dataset and Evaluation Card

## Source

KuaiRec 2.0: https://github.com/chongminggao/KuaiRec

Local source: big_matrix.csv

Raw data is not committed to Git.

## Verified local observations

- Interactions: 12,530,806
- Unique users: 7,176
- Unique videos: 10,728
- Missing fields detected: 0
- Invalid numeric rows detected: 0
- Watch ratios above 1: 4,237,441
- Observed days: 56

## Chronological split

| Split | UTC interval | Interactions |
| --- | --- | ---: |
| Train | Before 2020-08-10 | 8,787,755 |
| Validation | 2020-08-10 through 2020-08-30 | 2,130,071 |
| Test | 2020-08-31 onward | 1,612,980 |

The boundaries were chosen using the temporal
distribution of interactions, not model performance.

## Evaluation safeguards

- Do not use future interactions to construct past features.
- Do not treat unobserved videos as proven dislikes.
- Do not use an interaction's watch duration as an input
  for predicting that same interaction's engagement.
- Fit popularity statistics and learned parameters on
  training data only.
- Investigate repeated user-video interactions.
- Report cold-start users and items separately.
- Do not claim offline metrics measure production uplift.
- Keep the small_matrix data separate from big_matrix.

## Outstanding work

- Verify dataset redistribution terms.
- Audit duplicate user-video interactions.
- Audit watch-ratio consistency.
- Define engagement targets.
- Measure unique user/item overlap across splits.
- Develop leakage-safe recommendation evaluation.

## Interaction-quality audit

Full `big_matrix.csv` streaming audit:

- Total events: 12,530,806
- Unique user-video pairs: 10,300,969
- Events beyond first observation per pair: 2,229,837
- User-video pairs observed more than once: 1,841,544
- Watch-ratio inconsistencies above relative tolerance 1e-5: 49
- Maximum absolute watch-ratio error: 0.0000958008

Repeated interactions are preserved as separate events.
They must not be globally deduplicated before temporal evaluation.

The small ratio discrepancies require further inspection.
Do not assume they indicate corrupted data.

Historical features and popularity counts must use only events
available before the relevant prediction cutoff.
