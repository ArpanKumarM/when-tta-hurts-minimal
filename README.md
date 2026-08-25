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
analyze.py     paired bootstrap, McNemar, BH-FDR, difference-in-differences, tables, figures
reproduce.py   asset acquisition + checksum verification + exact-output regeneration and verification
manifest.json  release-archive and canonical-output filenames/sizes/hashes
results/       summary.json, CSV tables, PDF/PNG figures (tracked in git)
assets/        data/checkpoints/predictions (git-ignored; downloaded or generated locally)
```

## Required tools

- Git
- [GitHub CLI](https://cli.github.com/) (`gh`)
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.12 (installed automatically by `uv` if not already present)

## Authentication (private repository)

This repository and its release assets are private. Before cloning or
running `reproduce.py`, authenticate the GitHub CLI once:

```bash
gh auth login
```

## Fresh-clone reproduction

```bash
gh repo clone <owner>/when-tta-hurts-minimal
cd when-tta-hurts-minimal
uv sync --frozen
uv run python reproduce.py
```

## What `reproduce.py` does

1. If `assets/checkpoints/` and `assets/predictions/` are not already
   present locally, downloads the eight release archives via
   `gh release download` (requires `gh auth login`; fails closed with a
   short message if not authenticated), verifies each archive's SHA-256
   against `manifest.json`, and extracts them (rejecting any archive
   entry that would extract outside its target directory or that is a
   symlink/hardlink).
2. Recomputes the scientific summary (paired bootstrap, McNemar,
   Benjamini-Hochberg, difference-in-differences) from the 39 canonical
   prediction arrays, and regenerates `results/summary.json`, all seven
   CSV tables, and all five figures (PDF + PNG) from that recomputed
   data -- nothing is copied from a prior run.
3. Verifies all 18 regenerated files against the SHA-256 hashes recorded
   in `manifest.json`, printing `PASS` and exiting 0 on an exact match,
   or printing the specific mismatch(es) and exiting 1 otherwise.

Running it twice in a row produces byte-identical output both times.

## Optional: training, evaluation, analysis

Retrain one or all 39 matrix cells from scratch (downloads and
checksum-verifies the required MedMNIST datasets on first use):

```bash
uv run python train.py --run-id <run-id>
uv run python train.py --all
```

Evaluate a trained checkpoint on the validation or test split:

```bash
uv run python evaluate.py --run-id <run-id> --split validation
uv run python evaluate.py --all --split validation
uv run python evaluate.py --run-id <run-id> --split test
uv run python evaluate.py --all --split test
```

Recompute the summary/tables/figures directly, without the
download/verification steps `reproduce.py` performs:

```bash
uv run python analyze.py
```

## Size and runtime

Release archives: ~825MB download, ~830MB extracted (39 checkpoints,
39 prediction files). `reproduce.py` (statistics + figure regeneration
only, no training/inference) runs in well under a minute once assets
are present. Measured on Apple M3 Pro (18GB unified memory), PyTorch
MPS backend: full retraining of all 39 cells takes on the order of
several hours (~3.3 min/run at 28px, ~14 min/run at 64px, up to ~90
min/run at 128px).

## Exact reproduction vs. retraining

`reproduce.py` regenerates the canonical summary, tables, and figures
from the already-computed checkpoints and prediction arrays; this is
purely numerical (NumPy bootstrap/statistics plus deterministic
Matplotlib rendering) and produces byte-identical output on repeated
runs, independent of hardware. Retraining (`train.py`) and
re-evaluating (`evaluate.py`) involve PyTorch forward/backward passes;
bitwise-identical weights or logits across different backends (CPU vs.
CUDA vs. MPS) or hardware are **not** guaranteed, since floating-point
reduction order differs by backend. Same-hardware/same-software-stack
retraining is expected to be deterministic to the degree the backend
supports. This distinction does not change the training/evaluation
protocol itself -- only whether a retrain reproduces bit-identical
weights.

## Output files

- `results/summary.json` -- every preregistered (H1/H2/H3/BLOCK_C) and
  secondary cross-condition (H1/H2/H3) statistic.
- `results/tables/*.csv` -- the seven evidence tables (design
  classification, unmatched-policy cells, matched-policy comparison,
  normalization DiD, resolution DiD, BLOCK_C, claim adjudication).
- `results/figures/*.{pdf,png}` -- the five corresponding figures.

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

MedMNIST datasets are downloaded directly from the official Zenodo
record, never repackaged or redistributed by this repository.
PathMNIST and BloodMNIST: CC BY 4.0. DermaMNIST: CC BY-NC 4.0
(non-commercial use only) -- any redistribution or downstream use of
DermaMNIST-derived results must retain this restriction. None of these
datasets are intended for clinical use.
