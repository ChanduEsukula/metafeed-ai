# Controlled KuaiRec two-tower experiment on Colab T4

This is a sampled training experiment, not a recommendation-quality result.
No TorchRec, FAISS, pretrained weights, or paid GPU is required for training.
Local macOS tests must keep PyTorch and FAISS in separate Python processes.

## Source and runtime

Select a GPU runtime in Colab (`Runtime > Change runtime type`) and verify that
`nvidia-smi` identifies an NVIDIA T4. GPU types/availability vary; stop if the
required T4 is unavailable. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Use the exact working tree from `feat/pytorch-two-tower-retrieval`, including
uncommitted changes. Cloning the branch alone does not include local edits.
For example, locally create a source-only archive without committing:

```sh
tar --exclude=.git --exclude=.venv --exclude=data --exclude=artifacts \
    --exclude=__pycache__ --exclude=.pytest_cache \
    -czf /tmp/metafeed-two-tower-source.tgz \
    pyproject.toml src scripts configs tests docs
shasum -a 256 /tmp/metafeed-two-tower-source.tgz
```

Upload that archive to the runtime and extract into `/content/metafeed-ai`.
Record its hash with the experiment. Do not include datasets or checkpoints.

Use Colab's existing CUDA-enabled PyTorch and NumPy; do not upgrade them or
install TorchRec. Check Python is within the repository's supported range
(currently >=3.11,<3.14). In a notebook cell:

```python
import os, sys, torch, numpy
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
assert (3, 11) <= sys.version_info[:2] < (3, 14)
assert torch.cuda.is_available(), 'Select a GPU runtime'
assert 'T4' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(sys.version, torch.__version__, numpy.__version__, torch.version.cuda)
```

From `/content/metafeed-ai`, install only the local package:

```sh
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
mkdir -p artifacts
python -m pip freeze --all > artifacts/colab-environment.txt
nvidia-smi > artifacts/nvidia-smi.txt
```

The training path needs PyTorch and NumPy; the `--no-deps` install intentionally
does not change the runtime. The package's API/dev dependencies are needed only
for their corresponding test suites. If a required package is missing, stop
and record a justified installation rather than upgrading the runtime wholesale.
For repeat runs use the same captured environment, source archive, dataset hash,
GPU model and command. Cross-device/version bitwise equivalence is not promised;
see [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Dataset access

Obtain KuaiRec **2.0** from the [official dataset project](https://github.com/chongminggao/KuaiRec)
under its access/usage terms, or upload your existing local copy. This procedure
does not download it automatically. Use full `big_matrix.csv`, not the small
matrix or the old sampled Colab exports. Upload to runtime storage, or mount
your Drive in a notebook cell and copy the file to local runtime storage:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```sh
mkdir -p 'data/raw/kuairec/KuaiRec 2.0/data'
cp '/content/drive/MyDrive/KuaiRec 2.0/data/big_matrix.csv' \
   'data/raw/kuairec/KuaiRec 2.0/data/big_matrix.csv'
```

Adjust the Drive path to your existing file. An alternative location can be
passed with `--source '/absolute/path/big_matrix.csv'`. The report records that
path and the complete source SHA-256. Training uses the existing dataset config:
train before 2020-08-10 UTC; validation before 2020-08-31 UTC; test thereafter;
positive means `watch_ratio >= 1.0`.

## Assess before increasing scale

Run preparation only first; it scans the CSV completely, but performs no model
training and saves no checkpoint. A uniform reservoir samples at most 5,000
positive **events**, without replacement, over all training positives. Repeated
events remain distinct. It avoids the old source-prefix/user bias, but does not
guarantee every user, item or day appears. Inspect the measured coverage.

```sh
OMP_NUM_THREADS=1 python scripts/train_two_tower.py \
  --prepare-only --device cpu --max-events 5000 --seed 42 \
  --report artifacts/t4-preparation-seed42.json
```

The report includes complete and sampled per-UTC-day counts; training and
positive user/item counts; sampled user/item coverage; complete positive-pair
count; sample fingerprint; source fingerprint; preprocessing seconds; and process
peak RSS. Memory retains only the reservoir plus full distinct positive history
and training catalog, not all training events. Full history still grows with
unique positive pairs. Assess the observed memory against the actual runtime's
available host RAM; do not extrapolate GPU memory from host RSS.

Then run a bounded T4 pilot:

```sh
OMP_NUM_THREADS=1 python scripts/train_two_tower.py \
  --device cuda --max-events 5000 --epochs 3 --batch-size 64 \
  --embedding-dim 32 --seed 42 \
  --output data/processed/kuairec/models/two_tower_t4_n5000_seed42.pt \
  --report artifacts/t4-n5000-seed42.json
```

Every sampled event is visited exactly once per epoch. Batch items are unique;
users may repeat. Known-positive masking uses **all positives before train_end**,
including unsampled events. Validation/test positives never enter that lookup.
ID maps use all training observations, including negative-only users/items.

Each epoch reports loss (mean of optimizer-step batch means), all-event coverage,
effective anchor-row coverage, valid off-diagonal-negative fraction, skipped
batches, optimizer steps, elapsed seconds, visited/effective events per second,
process peak RSS and CUDA peak allocated/reserved bytes. Effective rows have at
least one unmasked negative; other targets can still appear as negative candidates
for other rows. A batch with no valid negatives never calls the optimizer.
Timing synchronizes the device; CUDA peaks reset per epoch, whereas process RSS
is a lifetime high-water mark. Null GPU memory on CPU/MPS means unavailable.

Repeat with distinct `--output` and `--report` names. Compare source/sample hashes,
ID maps, coverage, step counts and weights/losses in the same environment; exclude
timings/memory peaks from deterministic comparisons. Existing paths are refused,
including at exclusive file creation. Restoration uses `weights_only=True` and
strictly reloads a fresh CPU model, comparing every tensor exactly. Checkpoints
preserve earlier fields and add config, sampling, source/code hashes, versions,
CUDA/device details, and diagnostics. These are restoration artifacts, not yet
complete optimizer/RNG resume snapshots.

Only after inspecting pilot host/GPU memory, effective throughput, masked-negative
fraction and elapsed time should you consider a larger *bounded* sample, e.g.
50,000 events with a new output name. Do not launch full-data training yet.
Copy reports/checkpoints to your own Drive before the runtime expires. All paths
under `data/raw/`, `data/processed/`, and `artifacts/` are Git-ignored; never add
raw data/checkpoints with `git add -f`.

## Evaluation contract: identical to popularity

Model evaluation is designed against `evaluate_popularity()` in
`src/metafeed/evaluation/offline.py`, not sampled validation batches. Before
claiming improvement, implement a shared evaluation path or adapter and assert
identical population/coverage counters against the existing benchmark:

| Component | Required trained-model policy |
|---|---|
| Source/config | Same full CSV hash and `configs/datasets/kuairec.json` |
| Splits/positive threshold | Exactly the existing chronological boundaries and threshold |
| Parameters/history | Train parameters only on training data; no validation/test statistics |
| Catalog/maps | All training-observed items/users, including negative-only ones |
| History exclusion | Validation: complete train positives; test: complete train + validation positives |
| Ground truth | Distinct positive items in target split, intersect training catalog, subtract prior positives |
| Population | All users observed in target split; exclude/report users absent from training, then empty eligible ground truth |
| Candidates | Full training catalog minus that user's prior positives; no sampled negatives |
| Ranking | Exact inner-product scores of normalized towers, descending; ties by ascending raw item ID |
| Metrics | Existing Recall@10, NDCG@10, HitRate@10; macro-average over identical eligible users |
| Diagnostics | Same cold-start/empty-ground-truth/unavailable/repeated-positive counters and split counts |

Score the full catalog in bounded user batches; FAISS is unnecessary for this
catalog size. Never restrict candidates or eligible users to sampled/trained
embeddings. Report untrained user/item coverage separately: training-known but
unsampled IDs retain their initialized embeddings and must remain in the matched
population. Do not use batch contrastive loss as a ranking metric. Validation can
select settings; test is reserved for the final fixed experiment.

The current evaluator is hard-coded to popularity. A learned-model scoring
adapter and parity tests remain to be implemented; no trained-model Recall/NDCG/
HitRate or improvement claim has been produced by this work.

## Local assessment for this change

Two independent CPU runs used 5,000 sampled events, one epoch, batch size 64,
embedding dimension 32, seed 42 and `OMP_NUM_THREADS=1`. They scanned the full
local CSV but trained only the bounded sample. Both produced identical sample
fingerprints, maps, losses and model tensors. Checkpoint restoration passed.

- Full train: 8,787,755 observations, 3,027,364 positive events,
  2,726,352 distinct positive pairs, 7,174 users and 7,904 catalog items.
- Sample: 3,179 users (44.31%), 2,314 items (29.28% of catalog;
  30.24% of positive items), 19 of 32 positive UTC days. Sparse positive
  days are not guaranteed to be represented by a uniform event reservoir.
- Epoch: loss 5.2317305073, visited 5,000/5,000, effective anchors 4,998/5,000,
  valid-negative fraction 0.7472575565, 80 optimizer steps, one skipped batch,
  81 total batches, mean size 61.7284, minimum size 2.
- Preprocessing: 30.39–31.03 seconds; preprocessing peak RSS 426.91–517.86 MiB.
  Total process peak RSS 451.19–549.22 MiB, including training/restoration.
- Epoch elapsed: 0.198–0.244 seconds; visited throughput about 20,467–25,270
  events/second. These are local CPU observations, not T4 projections.
- Source SHA-256: `4ee1a72ff9ce4e86ef50525da9dfd7aefcefa717b37c9d570010f689c42f4ef9`.
- Sample SHA-256: `3a812c26644f45a60f25d7c14e352117e1694edb3e9b7bce6f83f304460e2e9c`.

Reports/checkpoints are under ignored `data/processed/kuairec/models/`, named
`two_tower_complete_history_cpu_seed42` and
`two_tower_complete_history_cpu_seed42_repeat` (`.json` / `.pt`). No CUDA device
was available locally. Run the hardware-dependent CUDA test and pilot on T4;
CUDA memory/runtime and recommendation quality remain unmeasured.
