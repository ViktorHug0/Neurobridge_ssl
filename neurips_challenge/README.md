# NeurIPS 2026 Track 1 — runs from this repo

Scope: train Neurobridge models on THINGS-EEG-2 with **this repo's EEG preprocessing** but the
**challenge's DINOv2 target space**, and score them with a faithful local copy of the Codabench
grader. The point is to bisect why `/nasbrain/p20fores/Neurips_challenge` produces numbers that
don't look like anything we've measured here.

Not in scope yet: Alljoined, the sealed-phase 32-channel montage, packaging a `submission.py`.

## The two metrics (do not mix them up)

| name | what it is | where |
|---|---|---|
| **Codabench warm-up** | cosine top-1/top-5 per **single EEG epoch** against the unique test targets, no averaging | leaderboard columns `top5_acc` / `top1_acc` |
| **NeuralBench reference** | same, but predictions averaged within each (subject, image) first | `test/full_retrieval/top5_acc_subject-agg` |

The challenge repo's `results/summary.csv` reports both: `top5_raw` is the leaderboard proxy,
`top5_subject_agg` is the reference. TSConv there scores **0.173 / 0.581**. This repo's usual
"best top5 acc" is the *subject-aggregated* kind, because `train.py:1932` hardcodes test-set
repetition averaging — so our historical numbers are never comparable to the leaderboard.

## The projector constraint (found while building the scorer)

The grader ranks the model's **raw 1536-D output** against **raw DINOv2 candidates** by cosine.
It never sees, and cannot apply, an image-side projector. So a checkpoint is only scoreable under
competition rules when both projectors are identities: `--projector direct --feature_dim 1536`.

This disqualifies most of this repo's history as-is. Every strong Neurobridge model
(`--projector linear --feature_dim 64/128/512`) ranks in a learned alignment space that exists on
both sides of the loss; folding the image projector into the EEG head reproduces the
*inner-product* ranking but not the *cosine* ranking, because the grader divides by each
candidate's norm. `score_track1.py` still scores such checkpoints — in their own space — but
stamps `leaderboard_valid: false` on the report.

Same reasoning kills test-time transduction (SAW / CSLS / Sinkhorn / Procrustes): the grader
does its own ranking from per-epoch predictions, so none of the SATTC machinery reaches it.

## Pipeline

1. **Targets — done.** `build_dinov2_targets.py` re-keys the start-kit's exca cache into
   `data/things_eeg/image_feature/dinov2-giant_track1/{image_train,image_test}.npy`
   (1654×10×1536 / 200×1×1536, 99 MB). All 16,740 images found, 200 unique test targets.
   Bit-identical to the grader's space, no GPU.
2. **Train — drafted.** `run_track1_tsconv.sh`: pooled all-10 subjects, **63 channels**, TSConv,
   this repo's recipe (grouped sampler + multi-positive, symmetric loss, softplus temperature,
   50 epochs, batch 1024, lr 3e-4), `--projector direct --feature_dim 1536`. Checkpoint selection
   is `--val_concept_ratio 0.1 --select_best_on val`, not our usual best-on-test, so the number
   is leaderboard-honest.
3. **Score — drafted.** `score_track1.py` streams the un-averaged test set one subject at a time
   (~1 GB each; storing all 160k predictions would need 10 GB) and emits per-epoch, subject-agg
   and instance-agg top-1/top-5. `--self_test` checks the metric math without a checkpoint.
4. **Verify the scorer — done.** `verify_grader.py` feeds identical synthetic predictions to
   `score_track1.py` and to NeuralBench's own `TopkAcc` + `agg_retrieval_preds` + `agg_per_group`,
   and requires agreement at all three aggregation levels. Run it with the *challenge* venv:

   ```
   /nasbrain/p20fores/Neurips_challenge/.venv/bin/python neurips_challenge/verify_grader.py
   ```

   Result: exact agreement (per_epoch 0.008375/0.036187 on 16,000 queries; subject_agg
   0.015500/0.076000 on 2,000; instance_agg 0.180000/0.395000 on 200). The noise scale is tuned so
   no level saturates — a level pinned at 0 or 1 would agree even if mis-wired.

   Corroboration from the start-kit side: in `results/summary.csv`, every row with
   `codabench_proxy_source = native_120hz_test` has `codabench_proxy_top5 == top5_raw` exactly
   (0.1221, 0.1787, 0.1764, 0.1734, 0.1651, 0.1454), confirming that NeuralBench's
   `test/full_retrieval/top5_acc` is the Codabench warm-up metric.

## Known deltas vs the challenge harness

Kept deliberately (this repo's recipe, the thing we're testing):

- 250 Hz / 250 samples / window 0–1.0 s vs 120 Hz / 120 samples / −0.2–0.8 s
- MVNN or channel-wise z-score (offline) vs RobustScaler-on-raw + clamp 20
- no band-pass or notch in `preprocess_eeg.py`
- checkpoint selection, batch sampler, loss symmetry (see below)

Not yet reproducible here, would need code:

- `ClipLoss(symmetric=False)` — `module/loss.py:178` always averages both directions
- `gradient_clip_val: 1.0` — no `clip_grad_norm_` anywhere in `train.py`
- OneCycleLR per optimizer step — `--lr_scheduler cosine` steps per epoch
- validation = 20% of *timelines* (whole subject×session recordings), monitored on a
  **within-batch** top-5

Reproducible but non-obvious:

- fixed logit scale 1.0 → `--init_temperature 1.0` with no `--t_learnable` / `--softplus`
- image side has **no** projector → `--projector direct`
- challenge batches count duplicate stimuli as negatives → omit `--grouped_batch_sampler`
  and `--multi_positive_loss`

## Files

- `build_dinov2_targets.py` — step 1, run once. Done.
- `run_track1_tsconv.sh` — run 1: train then score.
- `score_track1.py` — the grader replica. `--self_test` for the metric math.
- `verify_grader.py` — cross-checks that replica against NeuralBench's own metric code.
  Needs the challenge venv.
- `README.md` — this file.

---

# Sweep results (2026-09-22)

Twelve cells: {TSConv, EEGProject, ATM} x {250 Hz ours, 120 Hz NeuralBench} x {averaged, un-averaged}.
Pooled all-10 subjects, 63 channels, 50 epochs, seed 33, `--projector direct --feature_dim 1536`,
checkpoint selected on `--val_concept_ratio 0.1`. Every cell scored with the Codabench protocol
(`score_track1.py`): per-epoch cosine top-5 against the 200 unique DINOv2 targets.

## Averaged half — complete

Two input columns, and only one of them counts. **Official** is the 120 Hz tensor Codabench
actually serves (`track1_score.json`); **native** is the run's own training preprocessing
(`track1_score_native.json`) and is a diagnostic, not a leaderboard number.

| trained on | scored on | TSConv | EEGProject | ATM |
|---|---|---|---|---|
| 250 Hz (MVNN, 0-1.0 s) | native 250 Hz | 0.1725 | 0.1232 | 0.2636 |
| 250 Hz (MVNN, 0-1.0 s) | **official 120 Hz** | 0.0144 | 0.0182 | 0.0190 |
| 120 Hz (NeuralBench) | **official 120 Hz** | 0.1221 | 0.0881 | **0.1856** |

Chance is 0.025, so every 250 Hz model is **below chance** on the input it would receive. The only
submittable numbers in this table are the bottom row. Never quote a native score as a result.

Start-kit reference, `Track1TSConv` on its own harness: **0.1734** (subject-agg 0.5815).

Two patterns, each holding across all three encoders:

1. **250 Hz > 120 Hz.** This repo's preprocessing is not the weak link. Part of the 120 Hz deficit
   for the conv encoders is structural: TSConv's flattened width is 1440 at 250 samples but 400 at
   120, a 3.6x capacity cut. ATM is immune by construction -- its iTransformer embeds each channel
   through `Linear(seq_len, d_model)` with d_model fixed at 250.
2. **ATM >> TSConv > EEGProject**, at both rates, *while its subject conditioning is inactive*
   (ids 1-10 against an embedding table sized 0-9, so every batch falls back to the shared token).

Best *native* score is `250hz_avg / ATM` at 0.2636, but it collapses to 0.0190 on official input:
naive linear resampling does not bridge 250 Hz/0-1.0 s to 120 Hz/-0.2-0.8 s. Our best submittable
result is `120hz_avg / ATM` at **0.1856**, against Track1TSConv's 0.1454-0.1787 range.

## Un-averaged half — cancelled 2026-09-22 10:55

Stopped after 2 of 6 runs (`120hz_noavg` TSConv and EEGProject); `250hz_noavg` never started.

| 120 Hz | avg per-epoch | no-avg per-epoch | avg subj-agg | no-avg subj-agg | avg best ep | no-avg best ep |
|---|---|---|---|---|---|---|
| TSConv | 0.1221 | 0.1140 | 0.4595 | 0.3235 | 29 | 7 |
| EEGProject | 0.0881 | 0.0837 | 0.3625 | 0.2545 | 7 | 2 |

Averaging won both comparable pairs: a small per-epoch loss without it (~0.5 pp) but a large
subject-aggregated loss (11-14 pp). Caveat worth keeping: both un-averaged runs peaked within the
first few epochs, so part of that gap may be noisier checkpoint selection rather than a real
disadvantage. The ATM pair and all three 250 Hz pairs were never run.

## Why it was cancelled — validation cost

`120hz_noavg` runs at **~7 min/epoch, of which only 41 s is training**. The rest is the per-epoch
validation pass: `--val_concept_ratio 0.1` over un-averaged data is 66,000 samples versus 16,500
averaged, streamed from the NAS mmap every epoch. So a cell costs ~17.5 h rather than ~2.5 h, and
`250hz_noavg` (2x the data) cannot finish inside its 20 h walltime.

Two levers, neither applied without a decision:

- **Lower `--val_concept_ratio`** for these cells. Changes *which* concepts are held out, so the
  arms would train on different data -- `train.py:1778` keeps the seed constant precisely to stop
  that. Breaks the comparison worse than the slow run does.
- **Add a validate-every-N-epochs flag.** A code change; leaves the selection protocol intact
  apart from granularity, and is the option I would take.

## Memory, for future sizing

Peak RSS is far above the data size and is a **one-time ramp, not per-epoch growth** (54450 passed
epoch 11 at 110 GB after a 2-epoch dummy peaked at 93.8 GB).

| cell | train cache | measured peak |
|---|---|---|
| `120hz_avg` | 4.7 GB | 23.3 GB |
| `120hz_noavg` | 20 GB | 93.8 GB |
| `250hz_noavg` | 41.7 GB | >56 GB (OOM), untested above |

`module/dataset.py` keeps the materialised array after writing a cold cache instead of reopening
it as mmap, which is why cold runs cost more than warm ones -- but that does not explain ~74 GB of
overhead on a 20 GB mmap. Unresolved.

---

# Checkpoint selection: val loss is the wrong criterion (2026-09-22)

`train.py` ranks checkpoints by the selection split's **loss** (`train.py:3211`). On all eight
completed sweep runs, the val-loss minimum sits well below the val-accuracy maximum on the same
run's own test top-5 -- both criteria are val-only, so this is not test leakage, just a better
reading of the same signal.

| run | argmin val_loss | argmax val_top5 | gap |
|---|---|---|---|
| 120 Hz ATM | ep15 -> 15.30 | ep46 -> 19.40 | +4.10 |
| 120 Hz EEGProject | ep7 -> 6.35 | ep24 -> 11.70 | +5.35 |
| 120 Hz TSConv | ep29 -> 10.00 | ep46 -> 12.95 | +2.95 |
| 120 Hz noavg EEGProject | ep2 -> 3.05 | ep12 -> 6.75 | +3.70 |
| 120 Hz noavg TSConv | ep7 -> 4.95 | ep48 -> 6.60 | +1.65 |
| 250 Hz ATM | ep21 -> 26.10 | ep48 -> 28.65 | +2.55 |
| 250 Hz EEGProject | ep3 -> 11.60 | ep47 -> 16.65 | +5.05 |
| 250 Hz TSConv | ep9 -> 14.55 | ep39 -> 17.80 | +3.25 |

8/8, by 1.65-5.35 points. The contrastive loss keeps rising once the model gets confident, long
after ranking is still improving. REVE shows it cleanly: val_loss bottoms at epoch 6 (4.056) and
climbs monotonically to 5.164 by epoch 18 while test top-5 goes 14.40 -> 17.25.

So every number in the sweep table above is from an under-selected checkpoint, and the early "best
epochs" (3, 7, 9) were this artefact, not real saturation.

Fix: `--select_best_metric {loss,top5}`, default `loss` so no existing result changes meaning.
Both challenge scripts pass `top5`. Note this also changes what `--early_stop_patience` monitors,
since early stopping reads the same selection scalar.

## Consequence for a long run

`checkpoint_test_best.pth` is only rewritten when selection improves. Once val loss has turned
upward for good, the remaining epochs write nothing -- the file at the end of the job is the one
from the val-loss minimum. Check `epoch_metrics.csv` before waiting out a long run.
