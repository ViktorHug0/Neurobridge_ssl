# NeurIPS 2026 Track 1: the best ensemble

The submission-candidate pipeline for the sealed phase. The cohort is unseen Alljoined people and unseen images,
32-channel Emotiv, 120 Hz grader input, and DINOv2-giant targets. Everything is evaluated under the 4×5-subject
cross-validation in `CV4_PROTOCOL.md`.

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

Without operator transfer, the same five members score 44.02 (`*_things` recipes). The history of everything else
that was tried, and why it failed, is in `AUG_LOG.md`, `TTA_LOG.md`, `ARCH_LOG.md`, `ALIGN_LOG.md`, `ROUTING_LOG.md`
and `SUBJECT_LOG.md`. The code of those experiments is in git tag `pre-cleanup-2026-11-02`.

## Pipeline

```bash
# data (once): Alljoined 120 Hz epochs, THINGS-EEG2 120 Hz epochs exported as Alljoined sub-21..30, DINOv2 targets
.venv/bin/python neurips_challenge/build_alljoined_cache.py
.venv/bin/python neurips_challenge/build_120hz_cache.py && .venv/bin/python neurips_challenge/validate_120hz_cache.py
.venv/bin/python neurips_challenge/export_things_to_alljoined.py
.venv/bin/python neurips_challenge/build_dinov2_targets.py
# then build the THINGS training caches on the NAS and touch <data>/.cache/things_caches_ok (cv4.sh checks it)

# train + score the members (one run per recipe and fold; several workers can share the queue)
bash neurips_challenge/cv4.sh --list
Q=neurips_challenge/overnight/queue.txt bash neurips_challenge/overnight/align_dyn.sh
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
| `cv4.sh`, `alljoined_run.sh`, `overnight/align_dyn.sh`, `overnight/queue.txt` | the 10 recipes, the folds, the run wrapper, the queue worker |
| `score_track1.py` | grader replica (per-epoch / subject-agg / instance-agg; `--self_test`), fingerprint + lean adaptation |
| `tta_cache.py` | per-trial member outputs on the sealed-condition stream (shuffled, mixed-people 64-trial batches) |
| `lean_tta.py` | ensemble score with the deployable adaptation and the official metric |
| `official_metrics.py` | verbatim copy of the competition's metric code |
| `cv4_summary.py` | protocol tables, epoch budgets, paired decisions |
| `build_*.py`, `export_things_to_alljoined.py`, `validate_120hz_cache.py` | input data |
| `submissions/` | the September warm-up uploads (zips + build scripts), not this ensemble |

## Two constraints that shape everything

- **Raw outputs are ranked against raw DINOv2 candidates.** The grader can't apply an image-side projector, so only
  `--projector direct --feature_dim 1536` (or a bottleneck EEG head) produces a leaderboard-valid number.
  `score_track1.py` flags anything else.
- **The metric is subject-aggregated.** Predictions of an image's repeats are averaged within each person, ranked
  by cosine, and the per-person top-5 is averaged. Single-epoch numbers (the warm-up leaderboard) are a different
  measurement.
