# Augmentation / regularisation on the existing encoders: experiment log

Goal (user, 2026-10-26): no new architectures, no TTA work, no KD. Improve the five non-KD members (REVE-d6, SQF-dbn,
TSC-bn64, rmix-reg, REVE-scratch, all + THINGS) with data augmentation / invariance / regularisation so that the best
5-member non-KD ensemble reaches 47% (lean TTA pipeline: 64-d alpha+beta fingerprint, K 4, c 1, d = p/(p+n); current
44.05 replay, perfect-grouping ceiling of that set 44.93).

GPUs: 3 RTX 3080 incl. the interactive node (now sl-tp-br-514, job 64659) -> 2 sbatch workers + 1 local.
Workers: `Q=neurips_challenge/overnight/aug_queue.txt neurips_challenge/overnight/align_dyn.sh` (queue re-read per run).
Screen = cv4 fold 1 (5 subjects, sealed rule, half trials, per-epoch) vs the base recipe's fold-1 curve; promote to
4 folds -> tta_cache -> substitute into the 5-set replay (the only number that counts).

## Already dead on cv4 (ALIGN_LOG / ARCH_LOG / CV4_PROTOCOL; do not repeat)
mixup (REVE -0.7), raw SubjectMix (pairs screen, no THINGS: TSC -0.25, SQF -1.75), SCORE multi-positive (all 3
encoders), MixStyle, latency / montage jitter, channel dropout (REVE ~0), gain aug, stochastic depth (-1.2), wd .2,
EMA (SQF artefact), repetition averaging, blurred / multi-layer / soft targets, MMD, FiLM + subject dropout, CAR,
THINGS->Alljoined CORAL, EEG1 / Alljoined-1 data, global negatives, TTA (shift/gain/chdrop).

## Round 1 (2026-10-26 10:15): subject-style augmentations, REVE-d6 fold 1 (base fold-1 best 29.4)
Partners always from the same dataset (Alljoined 1-20 vs THINGS 21-30).
- `reve_d6_nswap_things` noise transplant (`--noise_swap_prob .5`, new): x' = own evoked mean (valid repeats) + another
  subject's single-trial residual (trial - its own repeat mean, random image), scaled to single-trial noise level.
  Rationale: person identity lives in the background alpha/beta topography (TTA_LOG); this keeps the label-carrying
  evoked part and swaps the background.
- `reve_d6_optx_things` channel-operator transfer (`--op_transfer_prob .5`, existing CVPR code + same-dataset partners):
  batch re-rendered through R_t^1/2 R_s^-1/2.
- `reve_d6_xmix_things` SubjectMix (`--xsubj_mix_prob .5`): Beta(1,1) mix with a same-image trial of another subject.
- CPU side probe: cross-subject ridge per frequency band (scratchpad band_probe.py) -> is the image signal outside alpha/beta?
- Fixes on launch: op_transfer crashed under bf16 autocast (einsum dtype) -> cast back; the CPU band probe read 15 GB of
  raw float32 files and evicted the local trainer's page cache (1 it/s) -> killed, re-warmed, probe now reads the
  trainer's own fp16 caches.
- Literature scan (subagent): neurips_challenge/AUG_LITERATURE.md (top: hyperalignment conversion, SAM, SWAD, Fishr,
  FACT alpha/beta amplitude mix; no 2025-26 paper reports a train-time input aug that helps zero-shot cross-subject
  EEG->image retrieval).
- Hyperalignment conversion (neurips_challenge/hyperalign_maps.py, train.py --hyperalign_prob): DROPPED before running.
  The cross-subject cross-covariance of repeat-averaged training responses is 2-4 dimensional (singular values
  1, .56, .30, .16, then <.07 for 1->3; THINGS 21->22: 1, .06, ...). Full Procrustes is therefore a random rotation
  (trace/32 ~ .1, ||R-I|| ~ 7.6 = random), and regularised toward I (lam >= .5 sv0) it is ~identity (diag >= .99):
  either destructive or a no-op. Side finding: sub-09 vs sub-06 cross-covariance has NEGATIVE trace (-.15 of the sv sum)
  -> sub-09's image-locked topography is inverted/displaced relative to the others (the known sub-09 anomaly).
- Band probe (fold 1, cross-subject ridge on 15 training subjects' repeat means, centred, top-5 of 200, chance 2.5):
  full 12.3 | <4 Hz 12.6 | 4-8 6.8 | 8-13 5.8 | 13-30 4.0 | 30+ 2.4 | <8 11.2 | all but 8-30 11.3.
  The cross-subject image signal is delta/theta; alpha/beta (the person fingerprint) carry little of it.
  -> `reve_d6_bswap_things` (`--band_swap_prob .5`, new): the trial's 8-30 Hz content replaced by that of a random trial
  of another same-dataset subject (label-preserving identity-band swap). Queued with `reve_d6_swa4_things`
  (`--swa_from_epoch 4`, new: uniform average of every step's weights from epoch 4 instead of the .9998 EMA; SWAD).
- reve_d6_nswap_things KILLED at e4 (21.0 23.6 25.7 25.7 vs base 20.9 25.5 27.7 27.8: matched -1.47). Swapping the
  whole non-evoked part (incl. delta noise) for another person's hurts; the narrower 8-30 Hz band swap took its GPU.
- reve_d6_bswap_things KILLED at e4 (21.1 24.4 26.3 26.2 vs base 20.9 25.5 27.7 27.8: matched -0.97). Even the
  label-poor 8-30 Hz band cannot be swapped between people without cost; with noise transplant (-1.47), replacing a
  person's own non-evoked content hurts REVE.
- Ensemble reference (aug_ens.py, base 5-set, lean TTA): folds 1-4 = 47.20 / 41.80 / 46.60 / 40.50, mean 44.02.
- reve_d6_xmix_things DONE fold 1: best 29.1 vs 29.4 (-0.3); early lead (+3.0 e1, +1.9 e2) fades to 0 by e7.
  Ensemble swap (aug_ens.py, fold 1): 47.20 -> 47.40 (+0.2) = noise. SubjectMix (same-dataset partners) NEUTRAL.
- reve_d6_optx_things DONE fold 1: best 31.0 @ e10 vs 29.4 (+1.6; matched +0.96), still rising at the stop epoch.
  Ensemble swap fold 1: 47.20 -> 47.50 (+0.3): the ensemble absorbs ~80% of a member's single gain (as KD did).
  Queued: reve_d6_optx folds 2-4 (confirmation); recipes ready: *_optx_things for the other 4 members,
  reve_d6_optxe2 (geodesic extrapolation u~U(0,2)), reve_d6_optx14 (stop e14).
- WHERE THE ENSEMBLE LOSES (base 5-set, lean TTA, per subject, all folds):
  9:10 17:15 19:24 15:32 16:38 7:39 8:40 1:42 11:42 6:46 5:48 10:50 4:51 13:51 18:54 20:54 12:55 14:62 2:64 3:65.
  Lifting sub-09 / sub-17 to ~45 alone = +3 on the mean. Zero-lag image-locked cross-covariance with the group:
  sub-09 / sub-17 r ~ 0.00 vs ~0.025 for normal subjects, and no channel permutation recovers it (Hungarian ~0) ->
  not a montage swap. Pixel-level lag correlation too noisy (|r| < .01 everywhere; sub-09 uniformly negative).
  Decoder-based lag scan running (tta_cache --shifts -12 -6 6 12 24 on reve_d6_things).
- Decoder-based diagnosis (reve_d6_things, lean TTA single member, per subject, all 20):
  input shift -12/-6/0/+6/+12/+24 samples (120 Hz) -> mean 1.8 / 14.8 / 39.4 / 20.2 / 3.8 / 0.6; EVERY subject peaks at
  0 (sub-09 9.5 -> 5.0/5.5 at +-6; sub-17 9.0 -> 7.0/3.5). No latency offset in the weak subjects. Negated input:
  every subject ~1 (sub-09 1.0). No polarity inversion either. sub-09/17 are intrinsically hard (also the weakest under
  pooled training, CV4_PROTOCOL §3), not mis-recorded -> no input transform rescues them.
  Side note: REVE halves at a 50 ms shift, i.e. very time-locked (the +-1/2-sample TTA probe was flat, so natural
  latency spread is not the issue).
- reve_d6_swa4_things DONE fold 1: plateau 28.3-28.7 from e6, best 28.7 vs 29.4 (-0.7, matched -0.14). Uniform weight
  averaging over the high-LR phase does not beat the .9998 EMA. DROPPED.
- LEAVE-ONE-OUT of the base 5-set (lean TTA, 20 subj; all 5 = 44.02): -REVE-d6 43.58 | -SQF 43.38 | -TSC 42.95 |
  -rmix-reg 43.88 | -REVE-scratch 43.40. The weakest single (TSC) matters most and the best single (rmix-reg) least:
  the ensemble is limited by error CORRELATION (3 of 5 are REVE-family), not member quality. => change of strategy:
  augmentations / input views that make a member DIFFERENT, evaluated in the rmix-reg slot (aug_ens.py
  rmix_reg_things=<variant>). First: reve_d6_lp8 (input < 8 Hz).
- reve_d6_sam1_things DONE fold 1 (SAM rho .1, stale ascent): best 29.2 vs 29.4 (-0.2, matched -0.25). NEUTRAL, dropped.
- ENSEMBLE SWAPS fold 1 (base 47.20; slot REVE-d6 | slot rmix-reg): SAM 48.50 | 48.50; optx 47.50 | 48.20;
  xmix 47.40 | 47.80. SAM is neutral single (-0.2) yet +1.3 in the ensemble -> folds 2-4 queued first.
- reve_d6_optx_things fold 2 DONE: best 29.7 vs 27.6 (+2.1). Single over folds 1-2: +1.85 (promote by the single rule).
  ENSEMBLE folds 1-2 (base 44.50): REVE-d6 slot 44.40 (-0.10; fold 2 41.80 -> 41.30), rmix-reg slot 44.95 (+0.45).
  A +1.85 single gain gives ~0 in the ensemble. optx folds 3-4 deprioritised behind SAM.
- reve_d6_lp8_things DONE fold 1: best 25.8 vs 29.4 (-3.6). Ensemble swap fold 1 (base 47.20): REVE-d6 slot 46.80,
  rmix-reg slot 46.40, REVE-scratch slot 46.20. The < 8 Hz view loses more signal than its diversity adds. DEAD;
  tsc_dbn40_bn64_lp8 killed at e22 (matched -1.6).
- reve_d6_sam1_things fold 2 DONE: best 28.5 vs 27.6 (+0.9). Ensemble fold 2 (base 41.80): REVE slot 41.60, rmix slot
  41.40. Folds 1-2 REVE slot: 44.50 -> 45.05 (+0.55), all from fold 1. Probably swap noise: the seed control
  (reve_d6_s7_things) calibrates it; SAM folds 3-4 + SAM on TSC/SQF/REVE-scratch/rmix-reg running/queued.
- tsc_dbn40_bn64_sam1_things DONE fold 1: best 29.3 vs 29.3 (0.0). Ensemble fold 1: TSC slot 47.10 (-0.1); TSC-SAM +
  REVE-SAM together 47.90 (< REVE-SAM alone 48.50) -> not additive, consistent with swap noise.
- SEED CONTROL reve_d6_s7_things fold 1: best 29.2 vs 29.4. Ensemble fold 1: REVE slot 47.00 (-0.2), rmix slot 47.10.
  A reseed moves the fold-1 ensemble by ~0.2: SAM's +1.3 on fold 1 exceeds it, but fold 2 was -0.2 -> folds 3-4 decide.
- sqf_dbn_sam1_things DONE fold 1: best 27.1 vs 28.5 (-1.4, matched -0.11).
  Ensemble fold 1: SQF slot 46.70 (-0.5). SAM on SQF dropped; optx folds 3-4 dropped (ensemble ~0).
- Composition probes fold 1 (base 47.20): rmix slot <- TSC-SAM 47.90, <- SQF-SAM 46.90; rmix <- TSC-SAM + REVE <- REVE-SAM
  47.50. Fold-1 swaps scatter by +-0.7 with no consistent direction: only 4-fold swaps can resolve effects this small.
- reve_d6_sam1_things fold 3 DONE: best 27.1 vs 26.9. Ensemble fold 3: 46.60 -> 45.90 (-0.7). Folds 1-3: 45.20 -> 45.33
  (+0.13). SAM's fold-1 +1.3 was noise. rmix_reg_sam1 dropped.
- rmix_reg_sam1 (claimed before the drop) and reve_scratch_sam1 (e3, ~45 min/epoch) KILLED: SAM settled as noise.
  Remaining: reve_d6_sam1 fold 4 (completes the 4-fold SAM swap), reve_d6_vrex1 fold 1.
- reve_d6_sam1_things 4 FOLDS: single best-epoch deltas -0.2 / +0.9 / +0.2 / -0.2. ENSEMBLE (20 subj): 44.02 -> 44.10
  (+0.08; per fold +1.3 / -0.2 / -0.7 / -0.1). SAM CLOSED: no ensemble effect.
- reve_d6_vrex1_things KILLED e2 (20.5 vs 25.5 at e2, matched -4.05): V-REx (lambda 1) hurts badly.
- DATA-DIVERSITY swaps (existing 4-fold runs trained WITHOUT THINGS, 20 subj, base 44.02): REVE-d6 slot <- reve_d6 43.35,
  rmix slot <- reve_d6 43.67; SQF slot <- sqf_b26 42.75, rmix <- sqf_b26 42.35; TSC slot <- tsc_b11 43.43, rmix <- tsc_b11
  42.57; rmix <- tsc_dbn40_things (no bottleneck) 43.22. All negative.

## Round summary (2026-10-26)
Nothing beat the base 5-set beyond reseed noise (+-0.2 per fold swap). Best 4-fold: SAM-REVE 44.10 vs 44.02.
Dead: noise transplant, alpha/beta band swap, SubjectMix (same-dataset), hyperalignment conversion (maps random or I),
SWA, SAM (REVE/TSC/SQF), V-REx, input low-pass < 8 Hz, data-diverse (no-THINGS) members. Operator transfer: +1.85
single (folds 1-2) but ~0 in the ensemble.
Why: (1) the ensemble is limited by error correlation, not member quality (LOO: -TSC -1.07, -rmix-reg -0.14; ~80% of a
member's single gain vanishes); (2) the mean is held down by intrinsically hard subjects (sub-09 10, sub-17 15) whose
image-locked signal is not shared at any channel mapping, lag or polarity; (3) cross-subject image information is
< 8 Hz (probe), so input-side invariance tricks either destroy signal or change nothing. 47% not reached.
- SWAP SCAN (aug_swapscan.py, folds 1-2, base 44.50): 14 existing training variants of the member families (CAR, CORAL,
  mixup, concept-negative masking, bottleneck widths, seed replicates, filters, optx, SAM) x 5 slots = 70 swaps. Top:
  +0.55 (tsc_dbn40_bn128 in rmix slot; SAM-REVE in REVE slot), +0.45, +0.40 (CAR in REVE slot), then <= +0.3. The max of
  70 draws at +-0.5 noise: no real signal. Loss-geometry diversity runs launched (reve_d6_siglip, reve_d6_mse1, folds 1-2;
  workers 64938 / 64939).
- Alljoined-1 revisited: the duplicate-path control (tsc_dbn40_bn64_dupctl, sub-01 trials via the extra-trial path) is
  -0.22 matched -> the path is fine and the aj1 -9.8 is a real data effect; not reopened (49.5k COCO trials < THINGS).
  Local worker: tsc_dbn40_bn64_siglip folds 1-2 (loss-geometry diversity in the conv family).
- reve_d6_mse1_things KILLED e6 (.9 InfoNCE + .1 MSE: matched -2.73).
- reve_d6_siglip_things fold 1 DONE: best 26.9 vs 29.4 (-2.5). Ensemble swap fold 1 (base 47.20): REVE slot 46.60, rmix
  slot 47.30, REVE-scratch slot 45.80. Loss-geometry diversity does not help; SigLIP fold 2 and TSC-SigLIP stopped.
  All GPU work stopped (2026-10-26 evening). Goal (47%) not reached; best 4-fold 44.10 (= base 44.02 within noise).

## Round 3 (2026-10-26 late): negative correlation (diversity regulariser; admissibility = user's call)
The ensemble is correlation-limited, so the last lever targets it directly: `reve_d6_nc_things` = REVE-d6 + penalty
lambda * sum_{j != true} p_student(j) p_others(j) over the batch images (tau .02), where p_others = softmax of the
mean prediction of the 4 OTHER members (REVE-d6, SQF, TSC, REVE-scratch; export_teacher_cv4.py -> teachers/fold<k>_4m.npy)
on each training trial. It repels the student from the others' MISTAKES and never imitates them (not distillation, but
it does use the other members' training-set predictions). Target slot: rmix-reg. Folds 1-2 launched.
- reve_d6_nc_things (lambda 1) folds 1-2 DONE: single best 29.0 / 28.2 vs 29.4 / 27.6. ENSEMBLE folds 1-2 (base 44.50):
  rmix slot 44.25, REVE-d6 slot 44.50, REVE-scratch slot 44.35 (rmix slot per fold +0.8 / -1.3). Negative correlation
  at lambda 1 does not decorrelate enough to matter. lambda 3 (reve_d6_nc3_things) fold 1 running locally.
- SATURATION (folds 1-2, lean TTA): base 5 = 44.50; base 5 + SAM/optx/NC REVE variants (8) = 44.15; all 17 cached
  non-KD members = 43.95; 5 + 2 conv extras = 43.80. Earlier: 10-member incl. KD 44.0-44.5, KD trio 44.1. Every
  combination lands at 44-45: the limit is shared by the whole model family (common errors from the same
  person-averaged inputs), not member diversity or count. A lever must change WHAT information is extracted.

## 2026-10-27 CorrCA probe DEAD; uniform-lever test queued
- Linear cross-subject ridge probe, fold 1, input projected on the top-k image-specific CorrCA components
  (training subjects only): k4 10.9, k8 11.7, k12 12.1, k16 12.2, k24 12.2 vs full 12.3 -> no gain at any k. Not run on GPU.
- Target-side levers are already dead from ALIGN_LOG (blur/view targets -0.9, soft targets -4, multi-layer heads ~0).
- Single swaps get ~80% absorbed. Open question: does the SAME lever on ALL 5 members cut the shared error?
  Op transfer (REVE-d6 +1.6/+2.1 single) queued on tsc/sqf/rmix/reve_scratch, folds 1-2 (workers 65024/65025 + local).
  Judge: aug_ens.py 1,2 with all five optx members vs 44.50.
- NC lambda 3 fold 1: single 27.7 vs 29.4 (-1.7); ensemble f1 47.20 -> 47.60 (rmix slot) / 48.00 (REVE-d6 slot).
  Within the +-0.7 single-fold scatter; lambda 1 had the same f1 bump then -1.3 on f2 -> fold 2 queued last.
- 2026-10-27 optx on all members, fold 1 singles: TSC 29.0 (-0.3), SQF 29.6 (+1.1), rmix 30.7 (+0.2).
  Ensemble f1 single swaps: TSC 47.80, SQF 47.50, rmix 48.10, REVE-d6 47.50; 4 optx members together 48.20 (+1.0 vs 47.20).
  Gains partly stack (unlike single-lever swaps before). Folds 3-4 queued for all 5 optx members.
- All-optx, all 5 members, fold 1: 47.90 (+0.7). REVE-scratch optx single f1 30.6 (+1.5).
- Folds 1-2 with 4 optx members (REVE-scratch f2 pending): 44.50 -> 45.30 (f1 +1.0, f2 +0.6); best 2-fold of the round
  (previous max 45.05). Singles f2: TSC +1.2, SQF +0.6, rmix +1.2, REVE-d6 +2.1. Every optx single is >= base except TSC f1 (-0.3).
  Op transfer is the one lever that helps every encoder; the ensemble keeps ~half of it. Folds 3-4 training.

## 2026-10-27 op-transfer variants (user: keep exploring this avenue today)
New flags (train.py): --op_transfer_bands <edges Hz> (per-band covariances, shared partners), --op_transfer_cross_dataset
(Alljoined <-> THINGS partners), --op_transfer_rotate (control: partner spectrum, random orientation). Self-checked
(transferred cov == partner cov; self-partners reproduce input). REVE-d6 fold-1 screens queued after the 4-fold verdict
runs: optxe2 (geodesic extrapolation u~U(0,2)), optxfb (0/4/8/13/30/61 Hz), optxrot, optxp1 (p=1), optxxd.
Reference: reve_d6_optx f1 31.0 / f2 29.7 (base 29.4 / 27.6).

## RESULT 2026-10-27 16:20: all-optx 4-fold verdict = +0.23 (mitigated)
All 5 members with --op_transfer_prob .5: 44.02 -> 44.25 (f1 47.20->47.90, f2 41.80->42.40, f3 46.60->46.70, f4 40.50->40.00).
Single-slot swaps over 4 folds: REVE-d6 43.80, TSC 43.93, SQF 43.68, rmix 43.98, REVE-scratch 43.85 (all <= 44.02).
Singles vs base (best epoch): REVE-d6 +1.6/+2.1/+0.5/f4, SQF +1.1/+0.6/+1.7/+1.0, TSC -0.3/+1.2/0/-0.1,
rmix +0.2/+1.2/+0.4/f4, REVE-scratch +1.5/+1.8/+0.1/f4. Folds 1-2 overstated it (+0.8); folds 3-4 give +0.1/-0.5.
- Subset scan (31 mixes of base/optx per slot, 4 folds; SELECTED -> optimistic): best 44.62 (optx in REVE-d6, TSC, rmix
  slots); all-optx 44.25; every mix 44.25-44.62.
- Overlap with the TTA: plain 5-member ensemble (no centring/shrink) base 32.95 -> optx 33.40 (+0.45); lean TTA 44.02 -> 44.25
  (+0.23). The person-centring absorbs about half; the ensemble average absorbs half of the rest (singles ~+0.8).
- Variant screens fold 1 vs reve_d6_optx at matched epochs (early): optxe2 -1.6 (5 ep), optxfb -2.3 (3 ep), optxrot -0.9 (4 ep).

## 2026-10-27 night: op-transfer variants done; oracle analysis; neighbour batches
Variants, REVE-d6 fold 1 (standard optx: single 31.0, ensemble 47.50; base 29.4 / 47.20):
| variant | single | ens f1 |
| optxe2 (geodesic u~U(0,2)) | 30.1 | 47.80 |
| optxfb (per-band 0/4/8/13/30/61 Hz) | 30.0 | 47.20 |
| optxrot (CONTROL: random orientation) | 30.5 | 48.00 |
| optxp1 (p = 1) | 30.5 | 48.00 |
| optxxd (cross-dataset partners) | 29.6 | 47.30 |
All within the +-0.7 single-fold scatter. The random-orientation control = real transfer -> op transfer acts as a
generic spatial-mixing robustness augmentation, not a subject model. NC lambda 3, folds 1-2: rmix slot 44.80 (+0.3),
REVE-d6 slot 44.70 (+0.2) -> neutral.
Oracle (base 5, lean TTA, 4 folds, scratchpad oracle.py): members 39.4 / 34.0 / 34.2 / 40.5 / 40.4, ensemble 44.02,
any-member top-5 57.6, >=2 members 46.1. Of ensemble misses, 75% no member hits, 17% exactly one, 8% two or more.
Hit correlation: REVE-d6 / rmix / REVE-scratch .74-.75 among themselves, every pair with SQF or TSC .52-.58 -> three of the
five slots are near-copies of one view. Per-image hit rate is shared across people (split-half r .33-.52).
=> Change WHAT the REVE-family members learn. Temperature is absorbed (unnormalised EEG sets its own scale), so:
neighbour batches (--neighbour_batch_prob p, module/sampler.py NeighbourBatchSampler): with prob p a batch = 128 distinct
images from one of 129 spherical-k-means clusters of the DINOv2 targets (in-batch cosine .758 vs .675 random) -> fine-
grained negatives. Queued fold 1: reve_d6_nb5, reve_scratch_nb5, reve_d6_nb1, rmix_reg_nb5 (workers 65400/65401 + local).
- 2026-10-28 01:40 neighbour batches fold 1: reve_d6_nb5 single 27.7 (-1.7) -> ensemble 47.90 (REVE-d6 slot), 47.80 (rmix),
  47.50 (REVE-scratch); reve_d6_nb1 27.1 (-2.3) -> 47.20 / 47.50 / 46.80. nb5 is the first lever with single DOWN and ensemble UP.
  Fold-1 hit phi: nb5 vs REVE-scratch/rmix .68-.69 (base REVE-d6 .71-.73, optx .72-.75); vs SQF/TSC .41/.43 (base .44/.49).
  Queued: reve_d6_nb5 folds 2-4, reve_d6_optxnb5 folds 1-2; rmix_reg_nb5 / reve_scratch_nb5 fold 1 running.
- 03:30 nb5 fold 2: reve_d6_nb5 single 28.1 (+0.5) but ensemble 41.80 -> 41.20; folds 1-2 44.55 (+0.05) -> fold-1 gain was
  scatter. rmix_reg_nb5 f1 single 28.6 (-1.9), ensemble 48.20 in its slot; REVE-d6 nb5 + rmix nb5 together f1 47.50 (no
  stacking). reve_scratch_nb5 f1 still training. Leaning neutral; folds 3-4 + optxnb5 finishing for a clean verdict.
- reve_scratch_nb5 f1: single 27.4 (-1.7); ensemble own slot 46.90 (-0.3), REVE-d6 slot 47.30, rmix slot 47.60 -> neutral/negative.

## RESULT 2026-10-28 06:40: neighbour batches DEAD
reve_d6_nb5 4 folds: 44.02 -> 43.83 (f1 47.90, f2 41.20, f3 46.10, f4 40.10); singles -1.7/+0.5/+1.6/-0.5.
reve_d6_optxnb5 folds 1-2: 44.50 -> 43.90. reve_scratch_nb5 f1 46.90, rmix_reg_nb5 f1 48.20 (own slot), nb5 pair f1 47.50.
Round status: best 4-fold no-KD ensemble = 44.25 (all-optx); every lever lands 43.8-44.6. Oracle 57.6 but 75% of misses
are hit by no member; REVE-family members are near-copies (hit phi .75). Queue empty, GPUs idle.
- 07:00 queued optx on the full 20-epoch cosine (reve_d6_optx20 f1/f2, rmix_reg_optx20 f1/f2; workers 65513/65514 + local): augmented REVE curves were still rising at the e10 stop (reve_d6_optx f1 best = e10).

## 2026-10-28 training audit (user: check epochs / lr / schedule / regularisation per member)
All 5: one-cycle, lr 1e-3, wd 0.05. REVE family (REVE-d6, rmix, REVE-scratch): 20-epoch cycle STOPPED at 10 (lr still
~5e-4, never annealed), EMA .9998; peaks e5-e10, f2/f4 decline after ~e6. SQF/TSC: full 40-ep anneal, no EMA, batch 1024,
very noisy mid-cycle (+-3/epoch), best at e34-e40 (still rising). Already dead: drop-path .1/.2 (-2.3/-3.6), wd .2, EMA on
SQF/TSC (+-0.2 at budget). Fold-1 screens queued: reve_d6_oc10 (full anneal over 10), reve_d6_oc20 (full 20), reve_d6_lr5
(peak 5e-4), tsc_dbn60_bn64, sqf_dbn60 (+ the running optx20 runs = optx on the full 20-epoch anneal).
- 10:20 optx20 (full 20-epoch one-cycle): REVE-d6 f1 peaks 30.0 (e13) then collapses to 22.4 (e18); f2 29.7 (e6-9) -> 21.3
  (e17). The low-lr anneal OVERFITS scratch REVE-d6 (same as LaBraM in CV4_PROTOCOL) -> the e10 stop is a needed early stop.
  rmix_reg optx20 f1: best 31.8 at e13 (+1.1 vs e10-stop optx), then 27.9 by e20 -> maybe stop ~13; fold 2 running.
  reve_d6_oc20 dropped from the queue (answered). aug_status.py fixed for runs longer than their base.
- optx20 ensemble: reve_d6_optx20 folds 1-2 44.75 (vs optx 44.40, base 44.50; sealed_best = e13/e9, collapse to 20.6 by e20); rmix_reg_optx20 f1 48.30 (best single swap so far on f1; sealed_best e13). rmix_reg_optx20 f2 running.

## RESULT 2026-10-28 23:50: training audit -> current schedules are near-optimal
Fold-1 singles vs base: reve_d6_oc10 (full anneal over 10) 27.6 (-1.8), reve_d6_lr5 (peak 5e-4) 28.1 (-1.3),
sqf_dbn60 26.4 (-2.1), tsc_dbn60_bn64 29.5 (+0.2). Full 20-ep anneal collapses REVE-d6 (30 -> 16.6-20.6 by e20).
rmix_reg_optx20 (stop 20, sealed_best picks ~e13): singles +1.3/+1.2/+1.3/+0.4 -> ensemble 4 folds 44.23 (+0.21);
with optx in all other slots 43.98. Single-model gains are again absorbed by the ensemble.
- 23:55 audit 2 queued (fold 1): reve_d6_wd01, reve_d6_bs256, tsc_dbn40_bn64_bs256, sqf_dbn_bs256. rmix_reg_optx20 per fold f1 48.30 f2 41.70 f3 46.00 f4 40.90; mix rmix20+reve_d6_optx+tsc_optx 44.45 (selected); tsc_dbn60 f1 ens 46.80.

## 2026-10-29 00:30 fusion-side levers (user: "try whatever is the best way to 47"), all-optx set, 4 folds
- Agreement-weighted trial aggregation (y_i *= exp(g * mean pairwise member cos)): g 0/1/2/4/8 -> 44.25/44.47/44.25/43.97/
  42.18; base set 44.02/43.90/44.00/43.58. Member agreement does not mark decodable trials. DEAD.
- Nested per-member weights (coordinate ascent on 2 folds, scored on the other 2, 4 splits): held-out vs equal +0.10,
  -0.30, +0.60, +0.15 (mean +0.14); fitted weights flip between splits. DEAD.
- TTA_LOG T48: perfect person grouping caps these members at 44.9-45.1; T41: cross-person same-image pooling would
  give 49.6-53.8 but needs image labels (label-free pooling fails: same-image pair precision ~1%).
- 01:20 audit 2 fold 1: reve_d6_wd01 27.3 (-2.1) DEAD; reve_d6_bs256 29.4 (+0.0), ensemble 48.20 -> folds 2-4 queued; tsc_dbn40_bn64_bs256 killed e23 (matched -2.69); sqf_dbn_bs256 running.
- 03:20 reve_d6_bs256 singles f1-3 +0.0/+0.8/+2.9 (f4 running); sqf_dbn_bs256 killed e25 (-6.44). Queued rmix_reg_bs256 / reve_scratch_bs256 folds 1-2.

## RESULT 2026-10-29 ~05:00: REVE-d6 batch 256 = better singles, WORSE ensemble -> DEAD; round closed
reve_d6_bs256 singles +0.0/+0.8/+2.9/+0.2 (mean +1.0) -> ensemble 4 folds 43.85 (-0.17); f3 (+2.9 single) 46.60 -> 45.80.
With optx in the other 4 slots 43.97. rmix/reve_scratch bs256 stopped (matched -1.0/-1.1). TSC/SQF bs256 killed, wd .01 dead.
Training audit conclusion: the members are properly trained; single-model gains of +1 to +3 do not survive the ensemble.
Best no-KD 5-member 4-fold ensemble of the whole round: 44.25 (all-optx). 47 not reached.

## 2026-10-29 member-pool check (user: "try whatever is the best way to 47")
4-fold swap scan of 15 cached non-KD candidates (incl. non-REVE-family ATM, dualstream, LaBraM scratch/pretrained,
pretrained REVE) into every slot: best 44.23 (rmix_reg_optx20, rmix slot); non-REVE members <= +0.08.
Nested greedy <=5 over the 19-member pool (choose on 2 folds, score on 2, 4 splits): held-out vs base5
42.00/43.55, 44.05/44.50, 41.70/41.15, 46.65/46.90 -> mean -0.42. Member selection on 10 people does not transfer.
=> No route to 47 inside the constraints (no KD, no new encoders, no new data, <= 5 members). Best: 44.25 (all-optx).

## 2026-10-29 decorrelated-encoder search (user-approved): rescore existing encoders by ENSEMBLE value, fold 1
(scratchpad decor_screen.py; lean-TTA single, hit phi vs the 5 members, rescue = share of ensemble misses it hits; members
rescue 5.1/9.7/10.2/6.1/7.4.) Base fold 1 = 47.20.
- Decorrelated families are weak: mixer 24.6 (phi .29-.36), eegnet 28.8, fbtn 27.8, conformer 22.2, gct 31.3 -> swaps 43-46.5.
- REVE-like (sw0/sw2 Bridge, src, rct, reve_z): single 38-42 but phi .7-.75 with the REVE family. sw0 4 folds 43.77 -> dead.
- TSConvPlus hk13 / hsm2: single 36.1/36.7, phi .46-.54 vs REVE but .73-.74 vs TSC; best swaps 48.1 (hk13 SQF slot,
  hsm2 rmix slot). Only fold 1 existed -> folds 2-4 training (workers 65865/65866 + local).
- ATM: rescue 11.0, swaps 47.7 on f1, but 4 folds <= 43.8 (earlier scan).
  NOTE: the fold-2 watcher's tta_cache may snapshot hk13/hsm2 folds 3-4 mid-run -> delete tta_cache/{hk13,hsm2}_things_f{3,4}.npy and recache before any 4-fold verdict.

## 2026-10-29 new-architecture search (user: "build the new architecture that, added as a 6th member, yields the highest gains")
Criterion: fold-1 gain of base5 + candidate (6th member, lean TTA) + hit phi vs members; then 4 folds for the best.
Never-tried families via braindecode 1.6.1: generic wrapper ensemble_experiments/architectures/braindecode_encoders.py
(registered as eeg_encoder_type Braindecode, arch_kwargs {"model": ...}; per-trial per-channel z-score input), recipe
pattern bd_<Model>_things (batch 512, 20-ep one-cycle). 26/33 models build on 32x120. Fold-1 screens queued: EEGNeX, BIOT
(spectrogram tokens), ATCNet, CTNet, MEDFormer, EEGInceptionERP, TIDNet, IFNet, TSception, EEGSimpleConv, Deep4Net, MSVTNet.
- 6th-member gain bar (fold 1, base5 47.20; CPU run, the GPU one OOMed beside training): hk13 +1.20, hsm2 +0.70,
  tsc_optx +0.70, atm +0.20, sqf_optx +0.20 (rescue 14.0), plab +0.10, fbtn -0.90, gct -1.00, mixer -1.00, eegnet -3.20.
  Weak decorrelated members HURT as a 6th; a useful one needs lean-TTA single >= ~35 AND phi <= ~.5 vs the REVE family.
- 13:45 TSConvPlus fold 2: hk13 swaps 41.1-42.0 (base 41.80), folds 1-2 <= 44.70; 6th-member f2 hk13 +0.30, hsm2 +0.60 (f1 +1.20/+0.70). bd_BIOT killed (chance: STFT tokens do not learn on 120-sample z-scored input). bd screens running; watcher screens each on CPU.
- 15:45 braindecode models are weak on this task: EEGNeX 18.3 raw (lean-TTA single 27.5, phi .30-.42, +6th +0.00),
  CTNet 16.8 at e8, ATCNet 4.6 at e8, BIOT chance -> stopped; EEGInceptionERP dropped. Remaining bd screens: MEDFormer,
  TIDNet, IFNet, TSception, EEGSimpleConv, Deep4Net, MSVTNet.
- New designed member GRUEncoder (ensemble_experiments/architectures/recurrent_encoders.py, 3.3M): per-trial z-score ->
  64 spatial filters -> depthwise smoothing -> per-dataset BN -> 2-layer BiGRU -> attention mean+std pool. gru_things /
  gruw_things (128 filters, hidden 384) at the front of the queue.

## 2026-10-29 evening: NEW GOAL (user): design new architectures until the 6-member ensemble (base 5 + 1, no KD) reaches 47
Metric: neurips_challenge/add6.py (6-member lean-TTA ensemble on base5 and all-optx5). TSConvPlus hk13 folds 1-3:
base5 45.20 -> 45.50, optx5 45.67 -> 45.43 -> stopped. Off-the-shelf braindecode screens dropped (user: own designs).
Own designs (ensemble_experiments/architectures/own_encoders.py): FourierFormer (frequency-token transformer, 5.6M),
DilatedConvNeXt1D (deep dilated ConvNeXt-1D, 1.5M); GRUEncoder f1: raw 25.7, lean 35.5, phi .38-.57, +6th -0.90.
- gruw (GRU 128 filters, hidden 384) f1: raw 25.0 (peak e10, overfits to 20.6 by e20), lean 34.4, phi .37-.49 vs members,
  rescue 12.5 (members 5-10), +6th +0.80, swaps 48.4 (REVE-d6) / 48.2 (rmix) / 48.3 (scratch) vs 47.20 -> best new-design
  lead. Queued gruw f2 + gruwr (dropout .4, op transfer .5, 12-ep cycle). Running: ff (FourierFormer), dcn, dcnph (phase).
- 20:45 user: 4 GPUs total (3 sbatch + 514). Worker 66211 added. GRUEncoder gained cell (gru/lstm) and bidirectional options; queued lstmwr, gruwrc (causal) after gruw f2 / gruwr. Early (vs TSC matched): ff +2.0 (e5), dcn +0.2 (e6), dcnph -2.4 (e5).
- 21:35 ff f1: raw 24.8 but lean 39.9, phi .47-.62, rescue 11.9, +6th +0.60, swaps 46.6-48.0. Queued ffr (dropout .2 + optx) f1, ff f2. gruwr e3 +2.0 vs TSC matched.
- 22:05 gruwr e7 27.9 (+1.2 vs TSC matched): regularisation fixed the GRU overfit. dcnph stopped (e20 -3.98). dcn e21 -1.40 (kept for the anneal).
- 22:50 gruwr (dropout .4 + op transfer + 12-ep): raw 28.7 (+1.3 matched vs TSC), lean 39.3 BUT phi .62-.63 vs REVE family
  (gruw .37-.49) -> +6th -0.10. Stronger AND more REVE-like. Hypothesis: op transfer pulls a new member toward the
  members' shared invariances. Cancelled gruwr f2-4, lstmwr, gruwrc. Queued no-optx: gruwd (dropout .4, 12-ep), ffd
  (d 384, depth 8), ff16 (0-16 Hz), ff40 (0-40 Hz). ffr (ff + optx) running = the same test on FourierFormer.
- 23:10 dcn f1: raw 26.7, lean 41.0, phi .40-.63, rescue 11.9, +6th +0.20. gruw f2: lean 30.7, phi .40-.55, +6th -0.60
  (f1 +0.80) -> gruw DEAD (mean ~+0.1). Remaining lead: ff (+0.60 f1; f2 queued). Pattern so far: single new members add
  <= ~0.6 as a 6th; the target needs +3.
- 23:50 ff f2: raw 26.3 (+0.3 vs TSC), lean 33.1, phi .67-.72 vs REVE family, +6th +0.00 (f1 +0.60). ffr (ff + optx) f1:
  lean 36.0, +6th -0.90 -> op transfer hurts new members (2nd time). PATTERN: fold-1 6th-member gains shrink on fold 2
  (hk13 +1.2 -> +0.3, gruw +0.8 -> -0.6, ff +0.6 -> 0.0) => real 6th gains ~0-0.3. Screens now judged on folds 1+2.
- 00:15 diagnosis: 17 non-KD members together = 43.95 (saturation) -> 47 needs a much STRONGER 6th (lean ~45+), not more decorrelation. New design SpecPatchFormer (electrode x 5-Hz-band complex tokens, 160 tokens, 5.5M) queued f1+f2.
- 01:00 gruwd f1 +6th +0.00 (dead); ffd 25.1 / ff16 25.2 raw (frequency family caps ~25). New: DualDomainFormer (time-patch + frequency tokens), queued f1+f2 on local worker.
- 01:10 ffd/ff16/ff40 f1: lean 38.4/39.3/36.7, phi .44-.67, +6th -1.10/-1.00/-1.10 -> FourierFormer family DEAD (ff's f1 +0.60 was a draw; f2 0.00).
- 01:35 spf e4 f1 25.7 (+2.95 vs TSC matched, -3.2 vs REVE-d6), f2 e3 24.2 (+5.9 vs TSC). ddf stopped (e4 22.4).
- 01:45 queued spfb (d 384, depth 8) f1+f2 and spf3 (3-Hz bands) f1; workers restarted (local 514i, 66242) after the ddf kills emptied the queue.
- 03:00 spf f1: lean 40.1, phi .45-.68, +6th -0.90; f2: lean 36.7, phi .53-.72, +6th 0.00 -> DEAD (frequency REVE = REVE-like errors). spfb/spf f3-4 cancelled.
- 03:20 late-window specialists (time_window 30-120 = 0.25-1 s; force non-early-visual information): grulate (gruw trunk) and dcnlate (dcn trunk), folds 1+2 on all 4 GPUs. repeat-mean objective NOT retried (L2 lever, dead earlier).
- 05:50 grulate f1/f2: lean 30.2/26.2, phi .37-.53 (most decorrelated yet) but +6th -0.20/-0.50 -> DEAD (late window
  too weak). ~15 own designs, none >= +0.3 as 6th member on both folds.
- BOOSTING member (classic way to build a complementary learner): --boost_gamma g weights each trial by (1 - p_ens(true
  image | batch))^g, p_ens from the 4 base members' TRAINING predictions (teachers/fold{K}_4m.npy, the NC file); targets
  unchanged (no imitation), needs --multi_positive_loss (only loss path with sample weights). FLAG FOR USER: uses
  members' training predictions as weights -> user to confirm it is not "KD". dcn_bst2_things f1+f2 running.
- 06:10 boosting: softmax p_true was flat (median .005, mean (1-p)^2 .983 -> no reweighting); per-trial teacher-row reads ran 1.08 s/it (killed). Now neurips_challenge/boost_weights.py precomputes 1 - rank/n of the true image over the 16540-image training gallery (rank/n median .05, 90% .50 = chance-level noisy trials), weight (rank/n)^gamma; dcn_bst5 / dcn_bst10 (gamma .5 / 1) f1+f2, 6-10 min/epoch.
- 09:35 per-trial boosting DEAD: dcn_bst5 single 15.6/11.9 (+6th -0.7/-1.3), dcn_bst10 chance (-3.9/-12.0): per-trial
  hardness is mostly single-trial noise, up-weighting it learns noise. dcnlate +6th -0.1/-1.1 DEAD.
  Next: IMAGE-level boosting (hardness = mean rank/n over all trials of the image, 10-90% .115-.204; boost_weights.py
  --image -> fold<k>_4m_pimg.npy), dcn_ibst20 / dcn_ibst40 (gamma 2 / 4), folds 1+2.
- 13:00 image-level boosting: dcn_ibst20 f1 lean 38.0, phi .32-.47, rescue 16.9 (best ever), +6th +1.40, swaps 47.3-49.2;
  f2 lean 30.3, phi .37-.55, rescue 9.3, +6th -0.80 -> 2-fold ~+0.3. ibst40 f1 -0.30 (too weak, 18.2 raw). Reported to
  user (~18 own designs, none reliably > +0.3; 47 needs +3). No reply -> continuing with 4-fold screening:
  ibst20 folds 3-4, ibst10 folds 1-4.

2026-10-30 13:55 -- image-boost 4-fold screen. cv4.sh bug: when the `_pimg/_ptrue` file was missing, the branch exported
teachers/fold<k>_4m.npy but never ran boost_weights.py on it, so the fold-3/4 workers sat in the 120 x 60 s wait. I ran
boost_weights.py 3/4 --image by hand (rank/n median 0.16, as on folds 1-2); cv4.sh now runs it after the export.
dcn_ibst10 (gamma 1) at epochs 8-9: matched -1.3 (f1) / -0.1 (f2) vs the TSC reference, against -4.1 / -3.9 for ibst20 -> keep.

# 2026-10-30 14:40 -- user: boosting CANCELLED (ibst20 f3/f4 and ibst10 f1/f2 killed mid-run). Goal: <= 6 non-KD members
at 47 via three routes (audit + literature scan, see chat): (1) target diversity, (2) rich fingerprint, (3) cross-person
same-image pooling break-even.
Route 1 ceiling (CPU): no-intercept ridge (lam .01 x mean eigenvalue) from L2-normalised training-image features of another
target to the grader layer (DINOv2-g L26 mean) keeps top-5 of the TRUE test-image features: L13 .99, L16 1.00, L40 1.00,
RN50 .95 (L8 .90 with intercept). Mapping does not cap a member. cv4.sh: <member>_t<L13|L16|L20|L33|L40|RN50>_things =
the member trained on that target (RN50: --feature_dim 1024); tta_cache.py bridge() maps its outputs to the grader layer
(linear, so norm = vote weight and the person centring carry over). Screens f1-2: reve_d6_tL40, tsc_dbn40_bn64_tRN50,
then reve_d6_tL13, sqf_dbn_tL40, tsc_dbn40_bn64_tL40, reve_d6_tRN50. Kill rule: phi > .55 vs SQF/TSC, or +6th <= +0.3 on both folds.
NOTE: these runs' per-epoch sealed scores are in their OWN target space (gallery = that target's test features).
Route 2 RESULT (lean TTA, 4 folds, per fold f1/f2/f3/f4): rich fingerprint (T9: base+fine+calpha+cbeta+cgamma+pgamma) vs
default: base5 43.85 vs 44.02 (47.30/41.10/45.60/41.40), optx5 44.15 vs 44.25 (48.10/41.80/45.90/40.80). DEAD at 5
members (its +0.5 was a 10-member effect); recheck once on the final 6-set.
Route 3 RESULT (pair_breakeven.py, scratchpad; base5 lean TTA 44.02, 4 folds; simulated detector, precision p / recall r,
y_i += b mean(proposed mates) after the shrink):
| r, b | p 1 | .5 | .3 | .2 | .1 | .05 |
|---|---|---|---|---|---|---|
| 1, .5 | 48.28 | 46.33 | 44.98 | 42.95 | 40.95 | 40.03 |
| 1, 1 | 50.90 | 46.93 | 43.78 | 40.12 | 37.05 | 34.75 |
| .3, .5 | 45.20 | 44.68 | 44.15 | 43.85 | 42.20 | 41.08 |
| .3, 1 | 46.33 | 44.70 | 44.18 | 42.30 | 39.45 | 37.33 |
Break-even p ~.25-.3 at FULL recall; 47 needs p >= .5 at full recall. Output-cosine pairs reach ~1% (chance .5%, T41):
a ~60x lift over chance at full recall would be needed. Detecting same-image pairs across people from single trials is
harder than the retrieval itself. DEAD.
Route 1 first results (folds 1-2 unless noted; lean TTA; base5 44.50 / optx5 45.15 on f1-2, f1 47.20 / 47.90):
| member | own-space single | bridged single lam .01 / .1 / 1 / 10 | best +6th (base5 / optx5) | phi vs 5 members |
|---|---|---|---|---|
| tsc_bn64 on CLIP RN50 (f1) | 25.7 | 10.2 / 13.1 / 15.7 / 15.0 | 39.30 / 38.60 (lam 1) = -7.9 | .19-.23 (f1), .21-.29 (f2) |
| reve_d6 on DINOv2 L40 | 29.25 | 21.9 / 25.95 / 27.2 / 25.15 | 41.20 / 42.00 (lam .1) = -3.3 | - |
Errors ARE decorrelated (phi ~.2) but these targets are WEAK on their own (25.7 / 29.3 vs grader-space singles 31-44),
and the clean-image bridge loses a further 2 (L40) to 10 (RN50) points on noisy predictions. tta_cache.py now also saves
the own-space output (<recipe>_f<k>_raw.npy); bridge_lam.py / ownspace.py (scratchpad) refit / score offline.
16:45 user: align only to targets that are strong by themselves = intermediate layers (Shallow Alignment): CLIP ViT-H
runs killed, RN50 / L40 / L13 / VITH variants dropped; queue = reve_d6 on InternViT L28 (running), DINOv2 L20, bigG L27,
InternViT L24, then sqf_dbn IV28, reve_d6 L33, tsc_bn64 IV28 (cv4.sh: _tIV<n>, _tBIGG).
17:05 target choice checked against Shallow Alignment (arXiv 2601.21948) Table 4/5 best layers (own-gallery THINGS-EEG,
intra top-1; inter top-1/top-5): InternViT L28 (60%) 82.6; 21.8/49.4 (best of 8; layers 24-28 "most favorable"),
EVA-02 L35 79.0; 19.3/46.2, ViT-H L11 76.8 (not extracted), DINOv2 L16 (38.5%) 75.9; 18.7/46.4, bigG L27 75.2; 20.8/47.0;
final layers 6-14 inter top-1. In-house THINGS sweep agrees (InternViT L23-35 solos 32-37, L28 best 37.05). Queue: IV28,
L20 (running), bigG L27, DINOv2 L16, EVA L35, IV24, sqf IV28, tsc IV28; L33 dropped (past the peak).
18:05 RESULT reve_d6_tIV28 (InternViT L28 target, f1-2): own-space lean single **43.75** (best single member so far; REVE-d6
in grader space ~39.5 on f1-2; training curve +3.7/+5.5 over reve_d6 at matched epochs). Clean-image ridge bridge: single
17.8 / 23.1 / 25.4 / 23.4 (lam .01/.1/1/10), +6th 39.3-41.6 vs optx5 45.15: the bridge loses ~18 points. Codebook bridge
(softmax over training images): 3% top-5 even on TRUE test features (test concepts are novel) -> must be linear.
Per-TRIAL stacking bridge (ridge of grader targets on the member's training-trial predictions): single 6.6-7.2 for any lam.
Not an index bug (checked); single-trial predictions are too noisy (cos with own target .035 vs .031 random, in-sample
top-5 of 16540 = 3.3%), so the fit is dominated by the shared component, which the TTA removes. Refit on CENTRED
per-image means (~36 trials/image, 600k trials) running.
19:00 user idea (b) launched: reve_d6_xIV28w{1,2} = REVE-d6 on the grader layer + a head aligned to the TRUE InternViT L28
embedding (train.py --auxiliary_fixed_targets: identity image side), weight 1 / 2; only the grader head is kept. (a) warm
start = the IV28 member with head' = W^T head (neurips_challenge/warmstart_ckpt.py), 4 epochs lr 2e-4 (reve_d6_wsIV28).
Bigg/EVA dropped (user: DINOv2 + InternViT intermediate layers only until positive). Claim race: 66269 took over the local
worker's L20 f1 run (16:50) and deleted its dir; both killed, L20 f1 requeued.
19:45 IV28 bridges, all DEAD (f1-2, optx5 45.15): image-mean centred stacking (600k trials, 36/image) single 13.9/16.3/20.45
(lam .01/.1/1), +6th <= 36.25 (in-sample image means are cleaner than test predictions); top-k PCA of the IV space
(k 16/32/64/128) single 14.4/17.8/16.2/16.6, +6th 37.4-40.0. No linear map carries the member's 43.75 own-space
strength into the grader layer, although the same maps carry clean image features at 100% top-5. Value of the IV28
target must come through training: (b) xIV28w1/w2 and (a) wsIV28 (head init = clean ridge lam 1).
22:45 RESULTS (f1-2): DINOv2 L16 member own-space 38.1 (not stronger than REVE-d6 ~39.5 in grader space), bridged single
10.4/11.4/16.0/23.6 (lam .01/.1/1/10), +6th <= 40.80 vs optx5 45.15. Train-then-bridge is DEAD for every target tried:
the bridge loses 14-18 points of single-member score even between two DINOv2 layers. Target diversity can only enter
through training: (a) wsIV28 (singles 26.8 / 25.7, grader space; FT erodes after epoch 1) and (b) xIV28w1/w2.
BUG: xIV28w* runs trained fine but score_track1.build_modules took the concatenated aux-target width (4736) -> the
queue retried scoring ~350 times on 66268/66269 (~2 h of 2 GPUs lost). Fixed (primary target_dims[0]). Watchers' pgrep
guards matched their own command lines (stalled 1.5 h); guards removed.
23:10 RESULT (a) and (b) on REVE (f1-2, lean TTA, grader space; base5 44.50 / optx5 45.15):
| member | single | phi vs REVE-d6/SQF/TSC/rmix/scratch (f1) | base5+1 | optx5+1 | best optx swap |
|---|---|---|---|---|---|
| reve_d6_wsIV28 (warm start) | 37.20 | .65/.43/.38/.60/.64 | 44.05 | 44.55 | 44.90 (for rmix) |
| reve_d6_xIV28w1 (user idea b) | 38.70 | .68/.40/.41/.64/.65 | 44.45 | 44.70 | 44.90 (for REVE-d6) |
Both as strong as REVE-d6 (39.45 on f1-2) but just another REVE copy (phi .65 with the REVE family): no ensemble gain.
Reading: the IV28 member's own-space 43.75 is mostly an easier GALLERY (InternViT spreads the 200 test images further
apart), not more EEG information: in the grader space it lands at REVE-d6's level; DINOv2 L16 in its own space 38.1.
Given a fixed grader target, error correlation follows the ARCHITECTURE, not the training target.
Next: (b) on the decorrelated slots (TSC, SQF + fixed IV28 head, w1) as slot swaps; reve_d6_xIV28w2 finishing.
00:25 reve_d6_xIV28w2 (IV head weight 2): single 37.35, base5+1 43.80, optx5+1 43.65, best optx swap 44.40 (REVE-d6);
phi .61/.38/.38/.56/.60 (f1). Heavier IV weight lowers phi slightly but costs strength: worse than w1. DEAD.
TSC / SQF + IV28 head (w1), matched-epoch vs their references at ep 26 / 11-17: TSC -0.64/-0.07, SQF -0.58/+0.16 (neutral);
ensemble verdict (swap into the TSC / SQF slot) pending (watcher). Queue drains after these; workers will exit.
2026-11-02 RESULT (b) on the decorrelated slots (f1-2; base5 44.50 / optx5 45.15):
tsc_dbn40_bn64_xIV28w1: single 34.00, base5+1 44.75, optx5+1 45.35 (+0.20), optx swaps 45.15/45.35/45.15/44.85/44.95
sqf_dbn_xIV28w1: single 34.30, base5+1 44.60, optx5+1 44.55, swaps 44.20-44.75. Noise level -> route 1 CLOSED in every form
(bridge, warm start, fixed InternViT head on REVE / TSC / SQF).
2026-11-02 user: "try whatever you want to create a 6-member ensemble (no KD)", 3 GPUs (514 + 2 sbatch: 66637, 66638).
Queue: hsm2 f3/f4 + hk13 f4 (best 6th-member screens: hsm2 +0.7/+0.6 on f1-2) to complete 4 folds; then a second seed
(--seed 7, cv4.sh _sd<n>_ suffix) of the decorrelated optx members TSC and SQF as the 6th member, 4 folds.
2026-11-02 IV28 TTA-then-bridge (lean centring + LW shrink in the InternViT space, then ridge to the grader layer): single 24.05 (lam 1) / 19.45 (lam 10), optx5 + it 42.15 / 38.80 vs 45.15. DEAD: the bridge loss is not the post-bridge shrink.
2026-11-02 nested greedy over 12 4-fold non-KD members (optx5 + plain SQF/TSC/REVE-scratch + rmixd8, tsc_dbn40, dualstream,
sqf_b26), <= 6 members, choose on 2 folds / score the other 2: held-out vs optx5 -1.35 (12->34), -1.10 (34->12),
+0.05 (13->24), +0.35 (24->13) = mean -0.51. Greedy on ALL 20 people (optimistic) stops at 4 members (rmix_reg_optx, tsc_optx,
sqf_optx, reve_d6_optx) = 44.55; a 5th/6th lowers it. The existing pool's ceiling is ~44.5.
2026-11-02 RESULT 4-fold 6th member (base5 44.02 / optx5 44.25): hsm2 43.93 / 43.97 (per fold optx5+1 48.10/42.40/45.80/39.60),
hk13 44.25 / 44.23 (48.10/42.60/45.60/40.60). Both DEAD: their f1-2 gains (+0.7/+0.6, +1.2/+0.3) were fold optimism.
Queue: second seeds (sd7) of tsc_optx and sqf_optx, 4 folds (balanced 2 REVE + 2 TSC + 2 SQF set).
2026-11-02 RESULT second seeds (sd7) of the decorrelated optx members, 4 folds (optx5 44.25):
optx5 + tsc_optx_sd7 44.12; optx5 + sqf_optx_sd7 44.35 (+0.10, noise); balanced 6 (REVE-d6, rmix-reg, 2 TSC, 2 SQF; no
REVE-scratch) 43.88; same without the 2nd SQF 43.85. Seeds of the decorrelated families add nothing either.
CONCLUSION of the 2026-10-29..11-02 search for a <= 6-member non-KD ensemble at 47: not reachable with the current member
families on cv4. Best = optx5 44.25 (44.35 with a 2nd SQF seed, within noise). Every 6th member, swap, re-selection
(nested -0.51; optimistic ceiling 44.55), target-diversity form (bridge / warm start / fixed InternViT head), rich
fingerprint and cross-person pooling lands at 43.9-44.6.
