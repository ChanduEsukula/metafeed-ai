# Validation-driven 50,000-event two-tower experiments

This workflow prepares 3-, 5-, and 10-epoch experiments. It does not report any
new real-data experiment results or claim improved recommendation quality.

Use the updated working tree on `feat/pytorch-two-tower-retrieval`, transferred
using the source-only procedure in [COLAB_TWO_TOWER.md](COLAB_TWO_TOWER.md).
Cloning an older branch revision will omit uncommitted changes. Select an NVIDIA
T4 runtime, use the existing PyTorch/NumPy environment, capture versions and
`nvidia-smi`, and make the existing full KuaiRec 2.0 CSV available as documented
there. Do not upgrade packages or install TorchRec. Existing archives are inputs,
not destinations for new writes.

## Fixed configuration

Use seed 42, 50,000 positive events, batch size 64, embedding dimension 32,
Adam learning rate 0.001 and temperature 0.1. These match the settings in the
existing archived 50k experiment. The CLI's sampler, complete training-positive
lookup, all-training-observation ID maps, and train cutoff are unchanged.

The archived 50k sample fingerprint is
`60000f6cd4af64d17c8dca4c5c907115ed6232367a8a22922bd4bde624496dd7`.
Check that each new training report has this fingerprint for the same source
bytes and fixed configuration. It contains 50,000 events, 6,719 sampled users
and 4,596 sampled items; full maps contain 7,174 users and 7,904 items. Coverage
counts describe the existing sample, not measured results of new training runs.

First assess preprocessing in Colab without training:

```sh
cd /content/metafeed-ai
OMP_NUM_THREADS=1 python scripts/train_two_tower.py \
  --prepare-only --max-events 50000 --seed 42 --device cpu \
  --batch-size 64 --embedding-dim 32 \
  --report artifacts/validation-v1-preparation.json
```

Run shell blocks in Colab `%%bash` cells. Inspect host memory/time, then begin
with the bounded 3-epoch experiment. Every destination below is exclusively
created; choose a new run label if any already exists.

```sh
cd /content/metafeed-ai
OMP_NUM_THREADS=1 python scripts/train_two_tower.py \
  --device cuda --max-events 50000 --epochs 3 --batch-size 64 \
  --embedding-dim 32 --seed 42 --save-every-epoch \
  --output data/processed/kuairec/models/validation-v1-e3.pt \
  --report artifacts/validation-v1-e3-training.json
```

After inspecting GPU memory, effective coverage, steps and throughput, use the
same command with `--epochs 5` and `validation-v1-e5` output/report names, then
`--epochs 10` and `validation-v1-e10` names. These are independent runs from the
same initialization and sample, not resumed jobs. For example:

```sh
cd /content/metafeed-ai
for epochs in 5 10; do
  OMP_NUM_THREADS=1 python scripts/train_two_tower.py \
    --device cuda --max-events 50000 --epochs "$epochs" --batch-size 64 \
    --embedding-dim 32 --seed 42 --save-every-epoch \
    --output "data/processed/kuairec/models/validation-v1-e${epochs}.pt" \
    --report "artifacts/validation-v1-e${epochs}-training.json" || exit 1
done
```

Alternatively, a single 10-epoch trajectory with snapshots supplies the epoch-3,
epoch-5 and epoch-10 states, avoiding duplicate training prefixes. Compare all
its epochs with the validation runner. Do not treat the independent runs'
identical prefixes as independent statistical replications.

## Snapshot safety and diagnostics

`--save-every-epoch` is optional. For output `validation-v1-e3.pt`, snapshots are:

- `validation-v1-e3.epoch-0001.pt`
- `validation-v1-e3.epoch-0002.pt`
- `validation-v1-e3.epoch-0003.pt`

The original final-output behavior remains: `validation-v1-e3.pt` is also saved.
It duplicates the final epoch's weights, so use the `epoch-*.pt` glob for
comparisons. All planned filenames are checked before data loading, and each
write uses exclusive creation to handle concurrent writers safely. Restarting
with the same output stem refuses existing artifacts; it never replaces them.
These are model snapshots, not optimizer/RNG resume checkpoints.

Snapshots copy tensors to CPU without moving or modifying the training model,
consuming RNG state, or changing the optimizer schedule. They retain source,
config/code/sample fingerprints, maps, seed, architecture, optimizer settings,
and all completed epoch diagnostics. `epochs` is the requested budget;
`completed_epochs` is the actual snapshot epoch. GPU memory allocated/reserved,
training loss, event/effective coverage, valid-negative fraction, skipped batches,
optimizer steps, elapsed time and throughput remain recorded per epoch. Epoch
training time excludes snapshot I/O; total runtime includes it. Final checkpoint
restoration is checked using restricted loading and exact tensor comparison.

No validation/test events enter sampled optimization rows, maps, or known-positive
masking. Sampling scans only training-positive events for selection and retains
all distinct training-positive relationships, including unsampled ones. All
sampled events are visited once per epoch, with unique items in every batch.

## Evaluate and select using validation only

```sh
cd /content/metafeed-ai
python scripts/validate_two_tower_checkpoints.py \
  --checkpoints 'data/processed/kuairec/models/validation-v1-e*.epoch-*.pt' \
  --output artifacts/validation-v1-comparison.json
```

A quoted glob is expanded by the runner; explicit checkpoint paths also work.
Every matched checkpoint is evaluated. Missing patterns, duplicate paths,
existing output files, incompatible maps/protocols/source hashes, inconsistent
sample/seed/architecture/batch/optimizer settings, and failed population parity
cause failure without a comparison being published.

The runner always forces **validation**, even though the repository dataset
configuration defaults to test; it exposes no test-evaluation switch. Popularity
is fitted/evaluated once. Each checkpoint uses the existing parity-tested
`evaluate_ranker()` loop, exact normalized CPU scores, identical full catalog,
prior-positive exclusions, cold/empty users, and ranking metrics. It asserts all
population/coverage counters match popularity. Training-known IDs absent from
optimization remain in the evaluation population and candidate catalog.

The JSON `comparison` array is the requested table: checkpoint path and SHA-256,
completed/requested epochs, model settings, sample fingerprint, Recall@10,
NDCG@10, HitRate@10, evaluated users, parity result, evaluation time, lifetime
peak process RSS, and epoch training diagnostics. `popularity` preserves the
baseline metrics/counters; `best_checkpoint` identifies the selected row.

Selection sorts **validation NDCG@10 descending**. Exact ties prefer the earlier
completed epoch, then the checkpoint identifier lexicographically. Recall and
HitRate are secondary reporting metrics and never break NDCG ties. Undefined,
nonfinite/out-of-range metrics or zero eligible users cannot select a winner.
No winner is copied over an existing artifact. The original final-only checkpoint
format is supported when its complete diagnostics/settings are present; the
completed epoch is read from its last diagnostic. Original missing learning-rate/
temperature metadata uses the historical fixed values 0.001/0.1.

Keep the test set untouched while selecting duration. Validation selection alone
is not evidence of test performance, statistical significance, or production
uplift. More epochs can overfit. Repeated checkpoint evaluation currently scans
the CSV for each model; shared target-history caching is not implemented. Memory
is bounded to one loaded ranker plus baseline/history state. RSS is a process
lifetime high-water mark, not independently attributed model memory. CUDA runs
and cross-version/hardware bitwise equivalence remain to be verified in Colab.

Save new reports/checkpoints to your own Drive before the runtime ends.
`data/processed/` and `artifacts/` stay Git-ignored. Do not alter prior archives,
force-add datasets, or launch these 50k training jobs locally.
