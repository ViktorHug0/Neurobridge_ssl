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
| `results/alljoined/` | 112 / 4.4 | 6S7i-Q3 subject variability (n=6, CV 0.179→0.379) |
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

## ⚠ Already lost — the FM experiment's code

These were **never committed** (`scripts/` is in `.gitignore`) and are gone from disk with no git
history. The vxam-Q1 LaBraM result (2×2: nomix 27.05→49.95, mix 31.85→63.10) is no longer
reproducible from code:

- `module/eeg_encoder/foundation.py` — LaBraM/CBraMod encoder wrapper
- `scripts/things_eeg/inter-subject-foundation.sh` — training driver
- `scripts/things_eeg/tta_rebuttal/run_fm_tta.py` — TTA driver

**Recoverable:** all 24 checkpoints and `train_config.json` survive, and the configs record the full
recipe (`eeg_encoder_type=LaBraM|CBraMod`, alpha 0.8 CLIP+MSE, lr 2e-3→1e-5 cosine, wd 0.05, 50ep,
warmup 5, `subject_mixup_mode=raw_eeg` alpha 0.5). Rewriting the encoder wrapper against those
configs would restore reproducibility.

**Lesson:** `.gitignore` excludes `scripts/`, `results/`, `data/`, `output/`, `slides/`, `graphs/`.
Only 35 of ~180 files in `scripts/` are tracked (they predate the ignore rule). Everything else there
is one `rm` from permanent loss. Commit or un-ignore `scripts/` before the next cleanup round.
