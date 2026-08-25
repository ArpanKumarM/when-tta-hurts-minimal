# when-tta-hurts-minimal

A compact, standalone reproduction of "When Test-Time Augmentation Hurts:
A Controlled Study in Medical Image Classification": naive mixed-policy
test-time augmentation (TTA) is measured against clean inference across
a 39-cell matrix over PathMNIST, BloodMNIST, and DermaMNIST (SmallCNN
with BatchNorm/GroupNorm at 28/64/128px, plus a ResNet-18 positive
control), with a matched-vs-unmatched training-policy comparison and a
paired-bootstrap / McNemar / Benjamini-Hochberg statistical layer.

## File map

```
config.py      39-cell matrix, frozen hyperparameters, dataset checksums/URLs, bootstrap seeds
data.py        dataset download+checksum, MedMNIST loading, augmentation policy, deterministic TTA views
model.py       SmallCNN, ResNet-18 (small-input adaptation)
train.py       training loop (Adam, cosine schedule, early stopping)
evaluate.py    clean + 100-view TTA inference, aggregation, BN-adaptation, metrics
analyze.py     paired bootstrap, McNemar, BH-FDR, difference-in-differences, tables
reproduce.py   asset acquisition + checksum verification + exact-result comparison
manifest.json  release-archive filenames/sizes/hashes/URLs for packaged checkpoints+predictions
results/       summary.json, CSV tables, PDF/PNG figures (tracked in git)
assets/        data/checkpoints/predictions (git-ignored; downloaded or generated locally)
```

## Installation

```bash
uv sync --frozen
```

## Exact reproduction

```bash
uv run python reproduce.py
```

Uses the canonical checkpoints and predictions under `assets/` if
present locally; otherwise downloads and checksum-verifies the release
archives listed in `manifest.json` first. Recomputes every preregistered
and secondary statistic from those canonical predictions and compares
the result against the committed `results/summary.json` field-by-field.
Exits 0 and prints `PASS` on an exact match, exits 1 and prints the
specific mismatch(es) otherwise. No dataset, checkpoint, or training run
is required for this path.

## Retraining

```bash
uv run python train.py --run-id <run-id>
uv run python train.py --all
```

Trains one or all 39 matrix cells from scratch (downloading and
checksum-verifying the required MedMNIST datasets on first use) and
writes checkpoints under `assets/checkpoints/`.

## Validation evaluation

```bash
uv run python evaluate.py --run-id <run-id> --split validation
uv run python evaluate.py --all --split validation
```

## Test evaluation

```bash
uv run python evaluate.py --run-id <run-id> --split test
uv run python evaluate.py --all --split test
```

Writes `predictions.npz` and `metrics.json` under `assets/predictions/`.

## Analysis

```bash
uv run python analyze.py
```

Reads `assets/predictions/`, recomputes every preregistered and
secondary statistic, and writes `results/summary.json` and
`results/tables/*.csv`. These are the values verified exactly against
the canonical generation-2 results (see `reproduce.py`). The PDF/PNG
files under `results/figures/` are copied byte-identical from the
canonical evidence package rather than re-rendered by `analyze.py` in
this version; `matplotlib` is pinned as a dependency for a future
figure-regeneration pass but is not currently invoked by any script
here.

## Runtime and storage

Measured on Apple M3 Pro (18GB unified memory), PyTorch MPS backend.
Training: ~3.3 min/run at 28px, ~14 min/run at 64px, up to ~90 min/run
at 128px. Full 39-cell retraining is on the order of several hours.
`reproduce.py` (statistics only, no training/inference) runs in well
under a minute. Packaged checkpoint+prediction release archives total
approximately 800MB.

## Hardware determinism

Exact-result reproduction (`reproduce.py`) recomputes purely numerical
statistics (bootstrap resampling, McNemar, BH-FDR, DiD) from
already-computed prediction arrays using NumPy only -- this is
deterministic given a fixed seed, independent of CPU/CUDA/MPS backend.
Retraining (`train.py`) and re-evaluating (`evaluate.py`) involve
PyTorch model forward/backward passes; bitwise-identical weights or
logits across different backends (CPU vs. CUDA vs. MPS) or hardware are
**not** guaranteed, since floating-point reduction order differs by
backend. Same-hardware/same-software-stack reproduction is expected to
be deterministic to the degree that backend supports it. This does not
change the frozen training/evaluation protocol itself -- only whether a
retrain reproduces bit-identical weights.

## Dataset and artifact download behavior

`reproduce.py` and `train.py`/`evaluate.py` use local `assets/` when
present. If checkpoints or predictions are absent, `reproduce.py`
downloads the release archives listed in `manifest.json`, verifies each
archive's SHA-256 before extraction, and rejects any archive entry that
would extract outside its target directory or that is a symlink/hardlink.
MedMNIST datasets themselves are never repackaged or redistributed by
this repository -- `data.py` downloads them directly from the official
MedMNIST Zenodo record (`https://zenodo.org/records/10519652`) and
verifies each file's official MD5 checksum before use.

## Main result boundaries

- 30 distinct unmatched-policy base cells were evaluated under the
  fixed mixed-policy, N=50 naive-TTA condition.
- All 30 showed a negative delta accuracy (TTA harmed every cell).
- Matched-policy (train/test augmentation-policy-matched) mitigation is
  supported only by a secondary, non-preregistered, fixed-model
  difference-in-differences comparison -- not a preregistered
  cross-condition test.
- No broad, population-level, or general-medical-imaging claim is made
  or supported by this evidence; findings are scoped to the specific
  datasets, architectures, resolutions, and seeds evaluated here.

## Dataset licensing

PathMNIST and BloodMNIST: CC BY 4.0. DermaMNIST: CC BY-NC 4.0
(non-commercial use only) -- any redistribution or downstream use of
DermaMNIST-derived results must retain this restriction. None of these
datasets are intended for clinical use.
