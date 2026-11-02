# Agent onboarding — Neurobridge_SSL

This file is for **AI coding agents** (and humans) who need to work in this repo without a long discovery phase. For a short human intro, see [`README.md`](README.md).

**Writing paper text?** Read [`.agents/skills/paper-prose/SKILL.md`](.agents/skills/paper-prose/SKILL.md) first. The authors reject LLM-sounding headings and sentences, and that file lists the patterns to avoid.

---

## What this repo does

**Decode images from EEG** on **THINGS-EEG-2** in the **inter-subject** setting: train on subjects 1–9 (leave one out), test on the held-out subject (LOSO over 10 subjects).

Pipeline in one sentence:

1. **Frozen** image embeddings (precomputed ViT features).
2. **Trainable** EEG encoder → optional projector → shared alignment space.
3. **Contrastive** EEG–image loss with **grouped multi-positive** batches (same image, different subjects).
4. At test time: **200-way closed-set retrieval**; optional **transductive adaptation** (SAW / CSLS / Sinkhorn / soft Procrustes) for much higher accuracy.

A parallel track exists for **THINGS-MEG** (`scripts/things_meg/`, `preprocess_meg.py`) and **AllJoined** (`scripts/alljoined/`). The mainline work is **THINGS-EEG**.
NeurIPS 2026 Track 1 sealed-phase work (Alljoined-1.6M, new subjects + new images) is evaluated only under the 4×5-subject cross-validation protocol in `neurips_challenge/CV4_PROTOCOL.md` (`cv4.sh`, `cv4_summary.py`). The current pipeline and the best ensemble (optx5, cv4 44.25) are described in [`neurips_challenge/README.md`](neurips_challenge/README.md).

---

## Environment

Always activate the project venv before Python:

```bash
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
```

- Dependencies: [`requirements.txt`](requirements.txt) (PyTorch 2.6 + CUDA 12.6, `open_clip_torch`, `transformers`, `mne`, etc.).
- `source .venv/bin/activate` can resolve to a different venv on this machine — prefer calling
  **`.venv/bin/python`** directly. `pytest` is **not** installed.
- **Do not commit** `data/`, `results/`, `*.pth` (see [`.gitignore`](.gitignore)).
  `scripts/` **is** tracked as of 2026-09-15 — it used to be gitignored, and work was lost that way.
- Workspace rules: [`.cursor/rules/guidelines.mdc`](.cursor/rules/guidelines.mdc) — prefer **small, surgical diffs** and **verify** changes.

---

## Repository layout

| Path | Role |
|------|------|
| [`train.py`](train.py) | Main training entrypoint (~2.1k lines): LOSO loop, loss, eval during train, checkpoints |
| [`evaluate.py`](evaluate.py) | Standalone eval from `checkpoint_test_best.pth` + SATTC-style refinement |
| [`compute_avg_results.py`](compute_avg_results.py) | Aggregate per-subject `result.csv` → `inter_subject_summary.csv` |
| [`extract_feature.py`](extract_feature.py) | Precompute image features (CLIP, OpenCLIP, InternViT, DINOv2, …) |
| [`preprocess_eeg.py`](preprocess_eeg.py) / [`preprocess_meg.py`](preprocess_meg.py) | Dataset preprocessing (not the daily train path) |
| [`module/dataset.py`](module/dataset.py) | `EEGPreImageDataset` — EEG + image features, caching |
| [`module/loss.py`](module/loss.py) | `ContrastiveLoss` (1-pos and multi-pos) |
| [`module/sampler.py`](module/sampler.py) | `GroupedImageBatchSampler` |
| [`module/util.py`](module/util.py) | Retrieval, SAW, CSLS, Sinkhorn, Procrustes, whitening |
| [`module/projector.py`](module/projector.py) | `direct` / `linear` / `bottleneck` projectors |
| [`module/eeg_encoder/model.py`](module/eeg_encoder/model.py) | TSConv, EEGNet, EEGConformer, … |
| [`module/eeg_encoder/atm/`](module/eeg_encoder/atm/) | ATMS encoder |
| [`scripts/things_eeg/`](scripts/things_eeg/) | Bash sweeps + Python experiment drivers (28 `.sh`, 47 files at top level) |
| [`scripts/things_meg/`](scripts/things_meg/) | MEG analogues |
| [`data/`](data/) | Image features, etc. (gitignored; large) |
| [`results/`](results/) | Experiment outputs (gitignored) |
| [`papers/`](papers/) | Markdown notes on related methods (SATTC, SSL, Sparse CLIP, …) |
| [`CVPR_paper/`](CVPR_paper/) | CVPR 2027 rewrite (transductive-calibration paradigm paper): `paper.tex` skeleton, `checklist.md` (decisions, protocol P-clean, experiments), `EXPERIMENT_LOG.md` |

---

## Data paths (this machine)

Paths are **hardcoded in scripts**; override with env vars where noted.

| Resource | Typical path |
|----------|----------------|
| Preprocessed EEG (250 Hz, 250 samples) | `/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/` |
| Per-subject layout | `{eeg_data_dir}/sub-01/` … `sub-10/` with `train.npy` / `test.npy` **or** NICE-EEG names `preprocessed_eeg_training.npy` / `preprocessed_eeg_test.npy` |
| Image features | `/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/` |
| Default image encoder dir | `InternViT-6B_layer28_mean_8bit` (also layers 25, 27, 29, 31 for ablations) |
| THINGS images (for `extract_feature.py`) | Often `/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set` |

`module/dataset.py` supports both this repo’s `.npy` layout and NICE-EEG dict format; it writes processed EEG caches under `{eeg_data_dir}/.cache/neurobridge_eeg/`.

**Train EEG tensor shapes** (after processing):

- Train: `(num_objects, num_images_per_object, num_reps, channels, time)` — usually averaged reps → `(objects, images, channels, time)`.
- Test: 200 objects × 1 image × reps → **200-way** retrieval.

Default time window: **250 samples** (`--time_window 0 250`).

---

## Model architecture

```
EEG (B, C, T)
  → eeg_encoder (TSConv, EEGNet, EEGConformer, TSConv_parameterizable, ATM, …)
  → backbone dim (--eeg_backbone_dim, default = image feature dim)
  → eeg_projector (--projector linear|bottleneck|direct, --feature_dim)
  → L2-normalized alignment embedding

Image features (precomputed, frozen)
  → img_projector (same type/dim as EEG side)
  → contrastive target
```

Important flags:

- **`--eeg_backbone_dim`**: width of encoder output. When **smaller than** image dim, only the EEG path is bottlenecked (“projector-only” sweeps fix backbone e.g. at 64/128/1024 and sweep `--feature_dim`).
- **`--feature_dim`**: alignment space dimension (typical strong values: **64**, **128**).
- **`--projector direct`**: no separate alignment dim; backbone dim is the alignment dim.

Encoders are selected with `--eeg_encoder_type` (default in argparse is `EEGProject`; **scripts usually use `TSConv`**).

---

## Training (`train.py`)

### LOSO loop (standard)

For held-out subject `S`, train on all other IDs in `{1..10}`:

```bash
python train.py \
  --train_subject_ids 1 2 3 4 5 6 7 8 9 \
  --test_subject_ids 10 \
  --output_name sub-10 \
  --output_dir ./results/things_eeg/inter-subjects/my_run \
  ...
```

Reference driver: [`scripts/things_eeg/inter-subjects.sh`](scripts/things_eeg/inter-subjects.sh).

### Mainline recipe (strong inter-subject baseline)

Flags commonly used together:

```text
--eeg_encoder_type TSConv
--projector linear
--feature_dim 64          # or 128
--multi_positive_loss
--grouped_batch_sampler
--samples_per_image 9
--data_average
--img_l2norm
--softplus              # learnable temperature via softplus
--batch_size 1024
--learning_rate 3e-4
--num_epochs 50
--image_feature_dir .../InternViT-6B_layer28_mean_8bit
```

### Optional training extras

| Flag | Effect |
|------|--------|
| `--subject_mixup_mode raw_eeg\|embedding` | Same-stimulus cross-subject convex mixup |
| `--eval_mode saw_csls` | Use SAW+CSLS at **train-time** test eval (not full SATTC loop) |
| `--select_best_on val` + `--val_subject_id N` | Checkpoint selection on a val subject (cleaner than test) |
| `--subject_probe_holdout` | Linear probe: can subject ID be read from backbone vs align space? |
| `--save_weights` | Write `checkpoint_test_best.pth` (needed for `evaluate.py`) |

### Outputs per run

Under `{output_dir}/{timestamp}-{output_name}/` (e.g. `20260519-122936-sub-10/`):

| File | Content |
|------|---------|
| `train_config.json` | Full args snapshot |
| `train.log` | Text log |
| `result.csv` | top1/top5, best epoch (evaluated on test or per `select_best_on`) |
| `checkpoint_test_best.pth` | If `--save_weights` |
| `training_metrics.png` | Loss/acc curves |
| TensorBoard | `events.out.tfevents.*` |

**Idempotency:** If a folder with the same `output_name` suffix exists and already has `result.csv`, `train.py` **exits without overwriting**. Incomplete dirs may be deleted and re-run.

Aggregate subjects:

```bash
python compute_avg_results.py --result_dir RUN_DIR --output_name inter_subject_summary.csv
```

---

## Evaluation

### During training

`train.py` runs retrieval on the test subject each epoch using `module.util.retrieve_all` and `--eval_mode` (`plain_cosine`, `saw`, `csls`, `saw_csls`). This is **not** the full iterative SATTC pipeline; use `evaluate.py` for that.

### Standalone (`evaluate.py`)

Loads a finished checkpoint and can apply **full refinement** (soft Procrustes + Sinkhorn, etc.):

```bash
python evaluate.py \
  --checkpoint_dir results/.../20260519-122936-sub-10 \
  --output_dir results/.../eval_out \
  --output_name sub-10-sattc \
  --eval_mode saw_csls \
  --sattc_soft_procrustes \
  --sattc_sinkhorn \
  ...
```

Requires `checkpoint_test_best.pth` and `train_config.json` in `checkpoint_dir`. Merges optional `evaluate_config.json` and CLI overrides.

Core geometry utilities live in [`module/util.py`](module/util.py): `subject_adaptive_whiten`, `csls_scores`, `sinkhorn_normalize`, `fit_soft_assignment_procrustes`, `apply_orthogonal_map`, `retrieve_all`.

---

## Metric convention

The reported accuracy is **"best top1 acc"** — the *best-epoch* top-1 on the LOSO held-out
subject (200-way, plain cosine unless a transductive family is stated). `result.csv` also has
`top1 acc` (**last** epoch); do not confuse them — they can differ by 10+ points. `checkpoint_test_best.pth`
is the best-top1 epoch, so `evaluate.py` on it reproduces the per-subject `best top1 acc` exactly.
"Avg." is the mean over the 10 held-out subjects.

## Finding the current best run

`results/` holds **thousands** of run dirs — do not eyeball them, and do not spawn one Python
process per file (too slow). The current best base model + config is tracked in agent memory
(`current-best-model`). To rescan from scratch, walk `Average` rows in one process:

```python
import glob, os, pandas as pd
best = []
for f in glob.glob('results/things_eeg/**/inter_subject_summary.csv', recursive=True):
    df = pd.read_csv(f)
    t = [c for c in df.columns if 'top1' in c.lower()]
    s = [c for c in df.columns if c.lower() in ('sub', 'subject')]
    if not t or not s: continue
    r = df[df[s[0]].astype(str).str.contains('Average', case=False, na=False)]
    if len(r): best.append((float(r.iloc[0][t[0]]), os.path.dirname(f)))
best.sort(reverse=True); print(*best[:20], sep='\n')
```

Remember the top hits are usually **transductive** (SATTC/TTA) numbers, not plain base models —
check `train_config.json` (`eval_mode`, `subject_mixup_mode`, `eeg_encoder_type`) before trusting one.

## Result families (do not conflate)

**Critical for correct claims:**

| Family | Typical top-1 | What it is |
|--------|----------------|------------|
| Compact inter-subject training | ~25–32% | e.g. `featdim_128`, no heavy TTA |
| Mixup / training regularization | mid-30s | e.g. `mixup_*` summaries |
| **Transductive SATTC / SAGE-TTA** on 200 test pairs | see below | SAW + CSLS + Sinkhorn + iterative Procrustes on **full** test fold |
| Split-test transfer | ~49% → ~59% | Fit adaptation on half of test, apply to other half; code in tag `pre-cleanup-2026-11-02` |
| Score ensembling (inductive) | ~44% honest | nested-LOFO member selection; code in tag `pre-cleanup-2026-11-02` |

**Which transductive number to quote.** The `~71%` April 2026 sweep figure appears nowhere in the paper. Current numbers:

| Number | What it is |
|---|---|
| **77.2** | NeurIPS paper headline — ATM after SAGE-TTA; **single seed**, best of 5 image encoders, TTA config tuned on TSConv's test set, test-selected checkpoint |
| 68.1 / 68.8 / 68.3 | TSConv + SubjectMix + TTA: 3-seed paper mean / seed 3300 in May / same seed re-run in July (code-path drift). Config = argmax of **test** top-1; checkpoint selected on test |
| 72.75 | same, with test-oracle hyperparameters (upper bound, not quotable) |
| 67.30 | same checkpoint, joint-argmax LOSO re-selection over a grid centred on the test optimum |
| **63.6** | LOSO-val (val-selected checkpoint + val-selected TTA, but k and τ frozen at test-tuned values) — the least leaky NeurIPS number |

None of these is fully clean. The CVPR rewrite (`CVPR_paper/`) regenerates every number under
protocol P-clean (`CVPR_paper/checklist.md`); log new runs in `CVPR_paper/EXPERIMENT_LOG.md`.

**Caveats for agents writing docs or papers:**

1. Many runs use `--select_best_on test` (checkpoint picked on held-out test subject) — others use `--val_subject_id`. State which protocol a number used.
2. Every transductive number is **transductive** (uses all 200 unlabeled test pairs jointly), not plain inductive retrieval, and assumes a closed known candidate pool.
3. `results/` and `data/` are local/NAS paths — may be missing on a fresh clone.

Example result roots on this machine:

- `results/things_eeg/inter-subjects/` — LOSO sessions, sweeps (`projector_only_bb*`, `tsconv_*`, `sattc_*`, …)
- Sweep summaries: `session_summary.csv`, `sweep_summary.csv`, `inter_subject_summary.csv`

---

## Checkpoints: what still exists (read before assuming a `.pth` is there)

A weight cleanup in September 2026 removed 227 GB of checkpoints. **`.pth` only** was deleted —
every `result.csv`, `train_config.json`, `train.log`, `training_metrics.png`, tfevents and
`*summary*.csv` was kept, so all 48,567 run records are intact and every published number is still
readable. `results/` is now ~169 GB with 3,690 `.pth`.

**[`PROTECTED.md`](PROTECTED.md) is the authority on what must not be deleted** — every path backing
a SAGE-paper (NeurIPS sub 28770) or rebuttal claim, plus the live experimental heads. Read it before
deleting anything under `results/`.

Three facts that follow, and that change how you plan work:

1. **The SATTC/SAGE paper track cannot be replayed from disk — its base checkpoints were deleted.**
   The 70 `sattc_*` and 5 `tta_*` sessions (31,150 runs) are evaluate-only runs: they hold no weights
   of their own and loaded the base models in `inter-subjects/tsconv_dropout_sweep_20260429-{190741,170207}`
   and `eeg_encoder_architecture_sweep_mixup_20260429-221100`, whose `.pth` were **deleted in the
   September 2026 cleanup**. The same `tsconv_dropout_sweep_20260429-190741/...seed3300` base is the
   `DEFAULT_SOURCE_RUN_DIR` of every `scripts/things_eeg/tta_rebuttal/` experiment. So the 68.x/77.2
   numbers exist only in `result.csv`; re-deriving them means retraining. Surviving clean weights:
   `tsconv_val_best_20260501-170237` (LOSO-val TSConv, 3 seeds) and `honest_ensemble/*_valcon` (ValCon).
2. **The ensemble line's score dumps and code were deleted in the 2026-11-02 cleanup**; the code is
   in tag `pre-cleanup-2026-11-02`.
3. **Most ablation sweeps are now weight-free** (`eeg_encoder_architecture_sweep_*`,
   `projector_only_bb*`, `tsconv_dropout_sweep_*`, `multihead`, `mixup_*`,
   `multipos_loss_sweep`, `holdout_probe_*`). Their numbers are in `result.csv`; re-running
   `evaluate.py` on them is not possible.

---

## Scripts cheat sheet

| Script | Purpose |
|--------|---------|
| [`inter-subjects.sh`](scripts/things_eeg/inter-subjects.sh) | Canonical LOSO × `feature_dim` sweep |
| [`inter-subject-tsconv-val-best.sh`](scripts/things_eeg/inter-subject-tsconv-val-best.sh) | TSConv + cross-subject mixup, LOSO-val checkpoint selection |
| [`inter-subject-SAATC.sh`](scripts/things_eeg/inter-subject-SAATC.sh) | SAGE-TTA (SATTC) evaluation sweep over 10 LOSO checkpoints |
| [`progressive_sattc_candidate_sweep.py`](scripts/things_eeg/progressive_sattc_candidate_sweep.py) | SATTC hyperparameter / sample-count sweeps |

The sparse CLIP, score-ensemble, reconstruction/caption, transfer and intra-subject lines were abandoned; their code is in tag `pre-cleanup-2026-11-02`.

Bash scripts usually: `source .venv/bin/activate`, `cd` repo root, set `EEG_DATA_DIR` / `IMAGE_FEATURE_DIR` / `DEVICE`, loop subjects 1–10, call `compute_avg_results.py` after each config.

---

## Image features

Precompute with [`extract_feature.py`](extract_feature.py) (`--model_type internvit|open_clip|clip|...`). Features are stored as `.npy` per image under `data/things_eeg/image_feature/<encoder_name>/`.

Strong default: **InternViT-6B layer 28**, mean pool, 8-bit (`InternViT-6B_layer28_mean_8bit`). Layer ablations: 25, 27, 29, 31.

---

## Papers and notes

[`papers/`](papers/) holds markdown digests (not build artifacts): `SATTC.md`, `SSL.md`, `Neurobridge.md`, `sparse_clip.md`, `subject_aware_alignment.md`, etc. Use the [arxiv-reader skill](.cursor/skills/arxiv-reader/SKILL.md) when fetching new arXiv sources.

---

## Legacy / low priority

- [`module/image_augmentation.py`](module/image_augmentation.py), [`module/eeg_augmentation.py`](module/eeg_augmentation.py) — used for aug feature paths, not mainline
- [`fuse_feature.py`](fuse_feature.py) — ancillary (used by `scripts/things_meg/image_feature_extract.sh`)
- README: “older utilities for augmentation and feature extraction are not the main path”

## EEG foundation-model encoders (LaBraM / CBraMod)

`--eeg_encoder_type LaBraM` or `CBraMod` full-fine-tunes a pretrained EEG foundation model in place
of TSConv ([`module/eeg_encoder/foundation.py`](module/eeg_encoder/foundation.py), driver
[`inter-subject-foundation.sh`](scripts/things_eeg/inter-subject-foundation.sh)). This answers
reviewer vxam-Q1; recorded results are in `results/things_eeg/foundation/` +
`labram_2x2_summary.csv` (plain / SAGE-TTA: nomix 27.05 -> 49.95, mix 31.85 -> 63.10).

Pretrained weights come from the Hub ids `braindecode/labram-pretrained` and
`braindecode/cbramod-pretrained`, already in `/nasbrain/p20fores/.cache/huggingface`. Export
`HF_HOME` to that path (the driver does it for you).

Two details that are easy to get wrong:

- The models consume **200 samples**, not our 250. The wrapper treats both as the same one-second
  window and interpolates 250 Hz -> 200 Hz; train.py has no switch for it. `foundation.py` also
  implements `resample='crop'`, which is **wrong** — it collapses sub-01 from 30.0 to 3.0 top-1.
- LaBraM needs `ch_names` for a 63-channel subset, otherwise it demands its full 128-channel
  canonical order. The wrapper passes `THINGS_EEG2_CH_NAMES`; all 63 map case-insensitively.

Reconstructed 2026-09-15 after the original was lost (see [`PROTECTED.md`](PROTECTED.md)) and
verified against the surviving checkpoints — `evaluate.py` reproduces all four arms exactly on
sub-01: LaBraM_mix 30.00/59.50, LaBraM_nomix 25.50/57.00, CBraMod_mix 11.50/30.50,
CBraMod_nomix 8.50/28.00. There is no separate FM TTA driver: `evaluate.py` with the published
SATTC flags covers it.

## Known gaps

- There is no test suite. The six root `test_*.py` self-checks were deleted 2026-09-15 (recoverable
  from git history if needed). `pytest` is not installed.
- `run_smooth_kernel_sweep.sh` requires `run_smooth_paperbase.sh` to have run first: the latter
  creates the `$SMOOTH_WORKTREE` worktree and patches `SMOOTH_K` into its `train.py`.

---

## Agent workflow checklist

1. **Activate** `.venv` before any `python` / `python3` command.
2. **Read** `train_config.json` in an existing run before changing hyperparameters.
3. **Prefer** editing `train.py` / `module/*` only when the task requires it; many experiments are **script-only**.
4. **Do not** assume `data/` or `results/` exist in git — check NAS paths.
5. **Match** existing bash patterns: `set -e`, `REPO_ROOT`, `SESSION_DIR`, `compute_avg_results.py` aggregation.
6. **Preserve** idempotent run directories (`result.csv` guard).
7. **Distinguish** training accuracy vs SATTC-adapted accuracy when reporting or plotting.

---

## Quick command reference

```bash
# One subject, minimal (paths must exist)
source .venv/bin/activate
python train.py \
  --train_subject_ids 1 2 3 4 5 6 7 8 9 --test_subject_ids 10 \
  --output_name sub-10 --output_dir ./results/things_eeg/inter-subjects/debug \
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
  --image_feature_dir ./data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
  --eeg_encoder_type TSConv --projector linear --feature_dim 64 \
  --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 \
  --data_average --img_l2norm --softplus --device cuda:0 --seed 2099

# Full LOSO session
bash scripts/things_eeg/inter-subjects.sh

# Aggregate
python compute_avg_results.py --result_dir RESULTS/SESSION/featdim_64 --output_name inter_subject_summary.csv
```

---

*Last updated from repo state: 2 November 2026 (code cleanup of abandoned lines and dead train.py
flags; pre-cleanup tree in tag `pre-cleanup-2026-11-02`). Update this file when default paths, main recipes, or
entrypoints change materially.*
