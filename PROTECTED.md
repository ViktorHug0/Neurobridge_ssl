# Protected assets — do not delete

Established 2026-09-15 during the repo weight cleanup. Anything listed here backs a claim in the
**SAGE paper (NeurIPS sub 28770)** or its **author response / rebuttal**, or is the live head of an
ongoing line. Check this file before any deletion round.

Rule applied throughout the cleanup: **`.pth` only is ever deleted.** `result.csv`,
`train_config.json`, `train.log`, `training_metrics.png`, tfevents and every `*summary*.csv` are
kept everywhere, so all recorded numbers stay readable even where weights are gone.

## Results — protected (831 checkpoints, 75.6 GB)

| Path | pth / GB | Backs |
|---|---|---|
| `results/things_eeg/subjectmix_rebuttal/` | 245 / 15.6 | vxam SubjectMix justification; `smooth_kernel_sweep/` = smoothing controls |
| `results/things_eeg/intra-subjects/` | 101 / 7.8 | intra-vs-inter bars; `TTA/` holds the 85.2 inductive intra bar |
| `results/things_eeg/foundation/` | 24 / 2.5 | vxam-Q1 LaBraM + CBraMod full-FT 2×2 (`labram_2x2_summary.csv`) |
| `results/alljoined/` | 112 / 4.4 | **paper dataset 2 of 3** (AllJoined-1.6M, appendix `alljoined-meg-results`); also 6S7i-Q3 subject variability (n=6, CV 0.179→0.379) |
| `results/things_meg/` | 30 / 1.9 | **paper dataset 3 of 3** (THINGS-MEG, 4 subjects/271 ch; appendix `alljoined-meg-results`) — the "generalizes across recording modalities" claim rests on this |
| `results/things_eeg/tta_rebuttal/` | 0 / 0 | Exp A/B, honest TTA selection, open-gallery, partial coverage (+ablations), streaming refit, alpha sweeps |
| `results/things_eeg/nsd_sage/` | 0 / 0 | `nsd_sage_technical_report.md`; SAGE→fMRI transfer |
| `results/things_eeg/inter-subjects/sattc_*`, `tta_*` | 0 / 0 | **the entire SATTC paper track — 75 sessions, 31,150 runs, already weightless** (ran without `--save_weights`) |
| `results/things_eeg/synthetic_subjects/promising_allfolds/group_e100/seed3300` | 10 / 0.6 | current best base model, 37.0 |
| `results/things_eeg/synthetic_subjects/ensemble_screen/dumps/` | 0 / 1.2 GB npz | 2,010 score dumps = 235 arms × 10 folds; the whole ensemble analysis replays from here |
| `results/things_eeg/synthetic_subjects/layer_sweep/` | 122 / 5.2 | arm-naming reference (`*_ch17` = electrode ablation) |
| `results/things_eeg/ensemble50_testselected/` | 192 / 9.7 | 16-arm ledger, `rule_final_ledger16.json` |
| `results/things_eeg/inter-subject-sparse/` | 25 / 29.9 | sparse winner `fd32768_learned_confidence` + controls (curated 2026-09-14) |

Also protected inside otherwise-cleared `inter-subjects/`:

- `eeg_encoder_sweep_20260426-184949/ATM_seed3300` — ledger arm `atm_iv`
- `eeg_encoder_architecture_sweep_mixup_20260429-221100/ViT-H-14_layer10_mean_ATM_seed3300_mixup` — `atm_vith`
- `eeg_encoder_architecture_sweep_mixup_20260429-221100/ViT-bigG-14_layer27_mean_TSConv_seed3300_mixup` — `tsconv_bigg`
- `eeg_encoder_architecture_sweep_mixup_20260429-221100/EVA02-E-14_layer35_mean_TSConv_seed3300_mixup` — `tsconv_eva`
- `tsconv_val_best_20260501-170237/param_k30_pool51_do050_featdim512_seed330{0,1,2}_valbest` — live ValCon/gate pool roots (`validation/pool.json`, `gate_pool.json`); `honest_ensemble` refits point here as `val_root`

### Raw/preprocessed data — protected results, re-derivable inputs

`data/things_meg/` (47 GB) and `data/alljoined_final/` (9.7 GB) are **inputs**, not records. The
paper's numbers live in `results/things_meg/` and `results/alljoined/`, which are protected above.
The inputs are needed only to re-train, and are re-obtainable: `scripts/things_meg/download_meg.sh`
/ `scripts/alljoined/`, plus the paper's appendix documents the exact preprocessing (MEG: 0–1000 ms
epochs, 0.1–100 Hz band-pass, downsample to 200 Hz, baseline correction, repetition averaging,
float16). Deleting them costs a re-download and re-preprocess, not a result — but do it as a
deliberate decision, not as cleanup.

## Scripts — protected

| Path | Backs |
|---|---|
| `scripts/things_eeg/tta_rebuttal/` (27 files) | every response-period experiment |
| `scripts/things_eeg/inter-subject-SAATC*.sh` | SATTC/SAGE training + sweeps |
| `scripts/nsd_sage/` (13 files) | SAGE→fMRI transfer |
| `scripts/alljoined/` | AllJoined LOSO |
| `scripts/things_eeg/run_subjectmix_controls.sh`, `run_smooth_paperbase.sh`, `run_smooth_kernel_sweep.sh`, `run_mixcontrols_paperbase.sh`, `subjectmix_rebuttal.sh`, `mixcontrols*.sbatch` | SubjectMix / smoothing controls (moved from repo root 2026-09-15) |
| `module/util.py` | `sinkhorn_normalize`, `csls_scores`, `fit_soft_assignment_procrustes` — the method itself |
| `evaluate.py`, `train.py`, `compute_avg_results.py` | entrypoints |
| `neurips_author_response_period/` (19 files) | paper.tex, answers.tex, reviews, deployment case, NSD report |

## The FM experiment's code — lost, then rebuilt (2026-09-15)

These three files were **never committed** (`scripts/` was in `.gitignore`) and vanished from disk
with no git history, leaving the vxam-Q1 LaBraM result without a code path:

- `module/eeg_encoder/foundation.py` — LaBraM/CBraMod encoder wrapper
- `scripts/things_eeg/inter-subject-foundation.sh` — training driver
- `scripts/things_eeg/tta_rebuttal/run_fm_tta.py` — TTA driver

**Rebuilt and verified.** The first two are restored; the third is unnecessary (`evaluate.py` with
the published SATTC flags does FM TTA). The architecture was not guessed — the surviving
`model_state_dict` pins it exactly (221 params, zero shape mismatches against
`Labram(n_times=200, n_chans=63, n_outputs=0, patch_size=200)` plus `head` Linear(200, 1024); CBraMod
the same with `return_encoder_output=True`), and the recipe came from `train_config.json`
(alpha 0.8 CLIP+MSE, lr 2e-3→1e-5 cosine, wd 0.05, 50 ep, warmup 5, `subject_mixup_mode=raw_eeg`
α=0.5). `evaluate.py` now reproduces **all four arms exactly** on sub-01:

| arm | recorded | replay |
|---|---|---|
| LaBraM_mix | 30.00 / 59.50 | 30.00 / 59.50 |
| LaBraM_nomix | 25.50 / 57.00 | 25.50 / 57.00 |
| CBraMod_mix | 11.50 / 30.50 | 11.50 / 30.50 |
| CBraMod_nomix | 8.50 / 28.00 | 8.50 / 28.00 |

LaBraM_mix also reproduces on sub-02 (41.50/74.50) and sub-10 (44.50/78.00).

The one thing the weights could not determine was how a 250-sample epoch became the 200 the models
consume; `--fm_resample resample` (250 Hz → 200 Hz over the same one-second window) reproduces the
numbers, while `crop` gives 3.0 instead of 30.0. Objective and full-FT decision follow AVDE
(Du/Dai et al. 2026, `papers/shallow_alignment.md`) and Liu et al. ICLR 2026 respectively.

**Lesson:** `.gitignore` excludes `scripts/`, `results/`, `data/`, `output/`, `slides/`, `graphs/`.
Only 35 of ~180 files in `scripts/` are tracked (they predate the ignore rule). Everything else there
is one `rm` from permanent loss. Commit or un-ignore `scripts/` before the next cleanup round.
