# NeurIPS 2026 Track 1: the best ensemble

The submission-candidate pipeline for the sealed phase. The cohort is unseen Alljoined people and unseen images,
32-channel Emotiv, 120 Hz grader input, and DINOv2-giant targets. Everything is evaluated under the 4×5-subject
cross-validation described below.

**Best ensemble (optx5): 44.25** official subject-aggregated top-5, mean over the 20 cv4 test people
(folds 47.90 / 42.40 / 46.70 / 40.00). Five members, every one trained with channel-operator transfer
(`--op_transfer_prob 0.5`):

| recipe | model |
|---|---|
| `reve_d6_optx_things` | REVE architecture, depth 6, random init |
| `reve_scratch_optx_things` | REVE architecture, full depth, random init |
| `rmix_reg_optx_things` | TokenREVE, mixed electrode+time tokens, 4 registers |
| `sqf_dbn_optx_things` | FastTSConvSqueezeformer, per-dataset BatchNorm |
| `tsc_dbn40_bn64_optx_things` | TSConv_parameterizable, per-dataset BatchNorm, 64-d bottleneck head |

The members are combined at test time by the lean adaptation, `score_track1.batch_transform_wknn`. It works per
64-trial predict() batch, without labels and without state:
1. Each member's output is scaled by its batch mean norm.
2. Each trial is centred on its 4 most similar batch-mates in a within-person-whitened EEG fingerprint space.
3. The members are summed.
4. One Ledoit-Wolf shrink is applied with d = D / (D + B).

Without operator transfer, the same five members score 44.02 (`*_things` recipes). Everything else that was tried,
and what it scored, is in `HISTORY.md`. The code (and the full experiment logs) is in git tag `pre-cleanup-2026-11-02`.

## Pipeline

```bash
# data (once): Alljoined 120 Hz epochs, THINGS-EEG2 120 Hz epochs exported as Alljoined sub-21..30, DINOv2 targets
.venv/bin/python neurips_challenge/build_alljoined_cache.py
.venv/bin/python neurips_challenge/build_120hz_cache.py   # matched neuralbench's own loader element for element (validator in the tag)
.venv/bin/python neurips_challenge/export_things_to_alljoined.py
.venv/bin/python neurips_challenge/build_dinov2_targets.py
# then build the THINGS training caches on the NAS and touch <data>/.cache/things_caches_ok (cv4.sh checks it)

# train + score the members (one run per recipe and fold; several workers can share the queue)
bash neurips_challenge/cv4.sh --list
bash neurips_challenge/align_dyn.sh                          # worker on neurips_challenge/queue.txt
.venv/bin/python neurips_challenge/cv4_summary.py           # per-recipe epoch budgets and scores

# ensemble score under the lean adaptation
.venv/bin/python neurips_challenge/tta_cache.py reve_d6_optx_things rmix_reg_optx_things sqf_dbn_optx_things \
    tsc_dbn40_bn64_optx_things reve_scratch_optx_things
.venv/bin/python neurips_challenge/lean_tta.py               # -> 44.25
```

`cv4.sh` → `alljoined_run.sh` (stages the EEG memmaps via `stage_eeg_cache.py`, runs `train.py`, then `score_track1.py`).

## Files

| file | role |
|---|---|
| `cv4.sh`, `alljoined_run.sh`, `align_dyn.sh`, `queue.txt` | the 10 recipes, the folds, the run wrapper, the queue worker |
| `score_track1.py` | grader replica (per-epoch / subject-agg / instance-agg; `--self_test`), fingerprint + lean adaptation |
| `tta_cache.py` | per-trial member outputs on the sealed-condition stream (shuffled, mixed-people 64-trial batches) |
| `lean_tta.py` | ensemble score with the deployable adaptation and the official metric |
| `official_metrics.py` | verbatim copy of the competition's metric code |
| `cv4_summary.py` | protocol tables, epoch budgets, paired decisions |
| `build_*.py`, `export_things_to_alljoined.py` | input data |
| `submissions/` | the September warm-up uploads (zips only), not this ensemble |

## Protocol (cv4)

- **Data:** Alljoined-1.6M (NEMAR nm000134 v1.0.3, 20 people; sub-08 `ses-02old` excluded), the grader's preprocessing
  (neuralbench `EegExtractor` 120 Hz, 0.1-75 Hz, 50/60 Hz notch, RobustScaler; epochs -0.2..0.8 s, baseline, clamp ±20),
  in `/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/sub-XX/{train,test}.npy`. THINGS-EEG2 sub-1..10
  are exported there as sub-21..30 (32 Emotiv channels, same images). Single trials only; targets `dinov2-giant_track1`.
- **Folds (frozen, never edit):** 1 = 2 4 5 11 15, 2 = 7 14 16 17 20, 3 = 3 8 10 12 19, 4 = 1 6 9 13 18 (snake-dealt by
  difficulty). Each trains on the other 15 Alljoined people + sub-21..30. The hard people (9, 17, 19, quiet 15) stay in.
- **Score:** per test person, average the 80 repeats of each image, rank by cosine against the 200 test targets, top-5;
  then average over people. Every run also scores its test people every epoch on a seeded half of their trials
  (`sealed_per_epoch.jsonl`); a recipe's epoch budget = the best epoch of that curve (test-people selection of one
  hyperparameter, accepted: the final model is trained with that budget). No validation split: held-out concepts do not
  track new-person performance.
- **Noise and rules** (`cv4_summary.py --ref <baseline>`): paired per-person SD 1.8-2.4 (same model) / 2.3-4.1 (other
  family), so 20 people resolve ~1 point. Screen on folds 1-2: drop if gain <= 0, promote if >= +1 and >= 6/10 people
  improve. Adopt on 4 folds if gain >= +1.0 (same arm) / +1.5 (other arm) and Wilcoxon p < 0.1. For ensemble levers the
  judge is the 4-fold swap into the lean-TTA ensemble (`lean_tta.py`). A new recipe gets a new name in `cv4.sh`; never
  edit an existing line (results are keyed by name). `cv4_summary.py` flags UNDERTRAINED recipes (last epoch best by > 0.3).
- **Compute:** 3 RTX 3080 in total incl. the interactive node; sbatch with `-p Brain_GPU -A brain -q low
  --gres=gpu:rtx3080:1 -c 8 --mem=30G`. Runs claim themselves (`.claim/job`), so several `align_dyn.sh` workers can share
  the queue. EEG memmaps are staged to `/tmp/$USER-aj-$SLURM_JOB_ID` once per job.
- **Final sealed model:** retrain each member on all 20 Alljoined people + sub-21..30, stopping at its cv4 budget:
  `SPLIT=final/<recipe> TRAIN_SUBJECTS="$(seq -s ' ' 1 30)" TEST_SUBJECTS=1 bash neurips_challenge/alljoined_run.sh <arm>
  <recipe flags> --select_best_on fixed --stop_after_epoch <budget>` (the test person is only a smoke score). Before
  submitting, check: fewer repeats per image (1-20), mixed-person batches of other sizes, channel order permuted by name.

## Two constraints that shape everything

- **Raw outputs are ranked against raw DINOv2 candidates.** The grader can't apply an image-side projector, so only
  `--projector direct --feature_dim 1536` (or a bottleneck EEG head) produces a leaderboard-valid number.
  `score_track1.py` flags anything else.
- **The metric is subject-aggregated.** Predictions of an image's repeats are averaged within each person, ranked
  by cosine, and the per-person top-5 is averaged. Single-epoch numbers (the warm-up leaderboard) are a different
  measurement.
