# Subject-key experiments, inductive setting — running log

Follow-up of `scripts/mi/MI_LOG.md` (2026-11-10: every LOSO encoder decodes conditionally on the subject's
image-independent mean ERP, the "key"). Started 2026-11-17. Scope: THINGS-EEG2, leave-one-subject-out, 200-way,
**inductive only** (no gallery-level adaptation; the only thing taken from the new subject is its unlabelled mean ERP).

Goals (user, 2026-11-17):
1. One encoder instead of an ensemble: distil the key-substituted ensemble into a single larger model.
2. Then look for further levers that follow from the key mechanism.

Layout: code `scripts/subject_key/` (`sk_lib.py` on top of `scripts/mi/mi_lib.py`), outputs `results/things_eeg/subject_key/`,
logs `logs/subject_key/`, training runs `results/things_eeg/subject_key/train/<arm>/`.
GPU: one rtx3080, Slurm job `key_gpu` whose body is `worker.sh` (runs `queue.txt` line by line; add with `q.sh <name> <cmd>`);
small evaluations next to it with `run.sh kXX.py`.

Conventions
- Protocol for every new run: validation concepts (`--val_concept_ratio 0.1`, 165 of 1,654 training concepts held out for
  checkpoint selection, same concepts in every arm). In such a run `checkpoint_test_best.pth` is the validation-selected
  epoch and `checkpoint_testsel_best.pth` the test-selected one. Numbers are validation-selected unless marked "test-sel.".
- "own" = the held-out subject's trials as they are. "sub" = key substitution, label-free: ATM takes its attention map from the
  subject's own key and its readout is fed the mean training key K; other encoders get the key moved halfway to K.
  The new subject's key is the mean of its 200 test trials (80-repetition averages); K comes from training images.
- Screening folds: 1, 5, 7 (teacher gain over the best member +12 / +3 / +2, close to the 10-fold mean); a lever is kept
  for the 10 folds only if the three agree.

## K01 teacher candidates (17 Nov, 17:20) — `k01_teacher.py`, `k01_teacher.csv`
Top-1 on the held-out subject, 10 folds, uniform mean of cosine score matrices.

| pool | members | own | sub |
|---|---|---|---|
| validation-selected | ATM-128 seed 3300 / 3301 / 3302 | 33.5 / 31.4 / 31.8 | 42.4 / 41.0 / 42.0 |
| | TSConv | 32.4 | 35.3 |
| | LaBraM (pretrained) | 25.7 | 34.0 |
| | ATM + TSConv | 38.5 | 44.0 |
| | ATM + TSConv + LaBraM | 35.4 | 46.5 |
| | ATM x2 + LaBraM | 34.3 | 47.6 |
| | **ATM x2 + TSConv + LaBraM** | 35.9 | **48.25** |
| test-selected | ATM / Squeezeformer / TSConv / rmix / LaBraM | 35.5 / 36.5 / 36.0 / 31.0 / 31.8 | 44.5 / 39.3 / 37.4 / 38.6 / 40.0 |
| | ATM + TSConv + Squeezeformer (MI_LOG E16) | 43.9 | 49.05 |
| | ATM + Squeezeformer + LaBraM | 40.7 | 50.55 |
| | ATM + TSConv + Squeezeformer + rmix | 44.8 | 51.0 |

- E16 reproduced (43.9 -> 49.05). Under validation selection the same kind of ensemble sits about 2 points lower.
- With substitution the best partners of ATM are the token encoders (LaBraM, rmix), not the conv encoders: the members that
  depend most on the key gain most from the substitution and are the most complementary afterwards.
- No validation-selected Squeezeformer exists. Teacher for the distillation = ATM x2 (seeds 3300, 3302) + TSConv + LaBraM,
  all `*_vc`: 48.25 substituted, per fold 60 57 46 36 43 48 46 33 50 63.

## K02 is the teacher informative on its own training data? (17:25) — `k02_teacher_insample.py`, fold 1
Distillation runs on training batches (113 images x 9 subjects, 4-repetition averages), where the teachers are in-sample.
113-way top-1 inside a batch:

| | single members | ensemble |
|---|---|---|
| training images, clean | 72.6 - 73.5 | 83.7 |
| training images, pairwise SubjectMix | 77.1 - 81.2 | 89.1 |
| held-out concepts (seen subjects), clean | 41.4 - 54.4 | 61.5 |

The members are far from saturated on their training trials and the ensemble is 10 points above each of them, so
in-sample soft targets carry information. Ensemble softmax at tau 0.05: entropy 2.4 nats of 4.7, top probability 0.41
(tau 0.02: 0.81; tau 0.1: 0.13). Distillation default: tau 0.05.

## Distillation code (17:30)
`train.py --kd_teacher <file> --kd_lambda 1 --kd_tau 0.05` (`module/teacher.py`). The teacher file
(`export_teacher.py t4 <fold> <runs>`, 1.5 min per fold, 0.4 GB) holds each member's embedding of every clean training
trial and training image. The student adds KL(softmax(mean teacher cosine / tau) || softmax(student cosine / tau)) over
the batch images to its contrastive loss; a SubjectMix pair gets the same mix of its two trials' teacher cosines.
Launcher `train_arm.sh <arm> "<folds>" <atm|tsconv|sqf|rmix> <flags>` (FOLD in a flag = the fold number).
- First version ran the four teachers online on the mixed batch: 114 s per epoch (3.2 h per ATM fold), dropped.
- Costs without it: ATM-128 63 s per epoch (105 min per fold, GPU saturated, 8.6 GB), TSConv about 35 s per epoch.
- Queue (17:40): kd_tsconv, kd_rmix, base_rmix, kd_atm, each on folds 1, 5, 7.

## K04 a spatial operator read from the keys (17:42) — `k04_keyop.py`, TSConv validation-selected, 10 folds — NEGATIVE
Idea: the key might tell how a subject's head mixes the sources, so a 63x63 operator W fitted on keys alone
(W m_from ~ m_to, ridge toward the identity, strength a) could re-mix the CONTENT. Test on seen subjects first, where a
decoder exists behind every key: content of s re-mixed toward training subject t, under t's key.

| | own key | wrong key, content untouched | content re-mixed, a = 0.01 / 0.1 / 1 / 3 |
|---|---|---|---|
| seen subjects | 65.9 | 47.3 | 10.2 / 18.4 / 32.7 / 40.6 |
| held-out (toward K) | 32.4 | 35.2 (usual substitution) | 11.4 / 20.1 / 30.0 / 33.4 |

Every strength is worse than leaving the content alone, and the curve only climbs back as W goes to the identity. The key
does not carry a usable linear re-mixing of the content. Closed.

## K05 which key for the readout of ATM? (17:45) — `k05_keychoice.py`, atm128_vc, 3 seeds, 10 folds
Attention map always from the new subject's own key; only the key fed to the readout changes.

| readout key | seed 3300 | 3301 | 3302 | mean |
|---|---|---|---|---|
| own (no substitution) | 33.4 | 31.4 | 31.9 | 32.2 |
| K = mean of the 9 training keys | 42.4 | 40.9 | 42.0 | 41.8 |
| midpoint own / K | 41.7 | 39.8 | 40.4 | 40.6 |
| nearest training key (correlation) | 37.6 | 36.3 | 38.0 | 37.3 |
| least-squares combination of training keys | 41.9 | 39.5 | 40.2 | 40.5 |
| **convex combination closest to the new key ("mixture key")** | **44.6** | **41.9** | **43.4** | **43.3** |
| softmax(correlation / 0.05) weights | 44.0 | 42.1 | 43.4 | 43.1 |
| mean of the 9 per-key score matrices | 42.0 | 40.7 | 41.7 | 41.4 |
| best single training key (oracle) | 43.7 | 41.7 | 42.4 | 42.6 |

- The mixture key (weights on the simplex, fitted to the new subject's key, label-free) gains +1.5 over K on the 3-seed
  mean, 20 of 30 cells up, 9 down; over the 10 subjects p = 0.10. It beats the oracle single key. Small and not yet
  established; plausible reading: SubjectMix trained the readout on convex mixtures of subjects, so a convex mixture of
  keys is the in-distribution way to describe a new subject (the unconstrained combination is worse than K).
- Score-level ensembling over keys does not help. Keep K as the default; re-test the mixture key on the final encoder.

## K06 the attention / readout dissociation in a token encoder (17:47) — `k06_token_targeted.py`, rmix test-selected, 10 folds
ATM: the attention map wants the new subject's own key, the readout wants K (MI_LOG E13b). Same test in rmix (TokenREVE,
6 blocks): pass A sets the attention weights of every block, pass B carries values / feed-forward / readout.

| stream input \ attention weights from | the stream input itself | the own-key pass | own-key pass, blocks 0-2 only / 3-5 only |
|---|---|---|---|
| own key | 31.0 | – (K-key attention: 28.9) | |
| midpoint | 38.5 | **40.05** | 39.6 / 39.2 |
| K | 36.45 | 39.7 | 38.45 / 37.35 |

Same sign as in ATM at every cell: attention computed under the own key is better (+1.6 at the midpoint, +3.3 under K),
attention computed under K hurts the own-key stream (-2.1). The effect is spread over the six blocks and is smaller than
in ATM (+4.2 there), where the attention acts on the raw electrodes. The dissociation is a property of attention over
electrode tokens, not of ATM alone. Best label-free rmix: 31.0 -> 40.05.

## K07 headroom of the key lever in ATM: an ORACLE readout key (17:52) — `k07_oracle_key.py`, atm128_vc seed 3300, folds 1 3 5 7
Only the 63x250 additive readout key is trained (from K), with LABELS on the new subject's training-image trials
(contrastive cross-entropy through the frozen model, batch 128, Adam; attention map from the subject's own key),
then applied to its test trials. Supervised adaptation, not a method: it measures what a better key could give.

| fold | own key | K | oracle key, 100 / 300 / 600 steps |
|---|---|---|---|
| 1 | 40.0 | 48.5 | 55.5 / 58.0 / 62.0 |
| 3 | 28.5 | 44.5 | 48.0 / 50.5 / 54.0 |
| 5 | 40.0 | 39.5 | 48.5 / 51.0 / 56.0 |
| 7 | 27.5 | 44.5 | 48.0 / 59.0 / 59.5 |
| mean | 34.0 | 44.25 | 50.0 / 54.6 / **57.9** |

- A frozen ATM reaches 58 on a new subject (+13.6 over K, still rising) when it is handed the right additive key:
  the key lever is far from exhausted by K (or by the mixture key, +1.5). Seen subjects with their own key: 72.
- The gap "new subject under K -> seen subject" (28 points) is thus at least half a matter of WHICH KEY the readout gets,
  i.e. a per-subject input offset, not of the content path. (TSConv, MI_LOG E08: free additive pattern +9.8.)
- Next: find that key without labels (K08: self-supervised agreement between repetitions; predicting it from the own key).

## K08 fitting the key without labels: agreement between repetitions (18:00) — `k08_selfsup_key.py` — NEGATIVE
Same parameter as K07 (the 63x250 readout key, frozen ATM), objective = symmetric InfoNCE between two half-averages of
the repetitions of each (unknown) image. Folds 1 3 5 7, K = 44.25.
- On the subject's training-image trials (2 + 2 repetitions): 38.5 after 25 steps, 22.3 after 100, 29.5 after 600.
- On its test trials (40 + 40 repetitions): the loss is already ~0.02 at step 25; accuracy stays at K (48.5 -> 48, 44.5 -> 43).
Repetitions already agree with each other under K; what is wrong is where the embeddings sit relative to the images,
and agreement between repetitions carries no information about that. Closed.

## K09 / K10 anchored output calibration (18:05-18:13) — `k09_output_moments.py`, `k10_calib.py`, 10 folds — MAIN RESULT SO FAR
Shift the new subject's (L2-normalised) embeddings by one vector so that their mean equals an anchor:
`e <- e - mean(e over the subject's unlabelled trials) + anchor`. No labels, no gallery, no key substitution, no training.
Anchor = mean embedding of the training subjects on training images ("EEG anchor") or the mean IMAGE embedding of the
training images ("image anchor"). Own key everywhere unless stated. `own-centre` = subtract the own mean, no anchor.

| encoder | plain | key substitution | calibrated, EEG anchor | calibrated, image anchor | substitution + calibration |
|---|---|---|---|---|---|
| ATM-128 vc, seeds 3300 / 3301 / 3302 | 33.4 / 31.4 / 31.9 | 42.4 / 40.9 / 42.0 | 48.3 / 46.9 / 48.1 | **48.9 / 48.3 / 48.6** | 45.3 / 44.8 / 46.3 |
| rmix (test-sel.) | 31.0 | 38.5 | 44.9 | 44.4 | 44.8 |
| LaBraM vc | 25.7 | 34.0 | 38.9 | 40.3 | 38.2 |
| EEGProject vc | 22.1 | 27.8 | 35.1 | 35.6 | 36.0 |
| EEGNet vc | 21.3 | 28.5 | 33.2 | 32.0 | 34.4 |
| Conformer vc | 23.4 | 28.5 | 33.0 | 32.4 | 34.2 |
| Squeezeformer (test-sel.) | 36.5 | 39.1 | 42.0 | 41.0 | 41.2 |
| TSConv vc | 32.4 | 35.2 | 37.2 | 37.0 | 36.7 |

- **One mean shift is worth +15.5 for ATM (32.2 -> 47.7, 3 seeds; 48.6 with the image anchor)**, more than the key
  substitution (+9.6) and as much as the 4-member substituted ensemble of K01 (48.25). It beats the substitution for
  every encoder; the two do not add up (ATM: substitution then calibration 45.5 < calibration alone).
- The gain follows the key dependence of MI_LOG E09 (conv encoders +5, token / attention encoders +13 to +15): for a new
  subject the key-conditional decoder outputs the right geometry around the wrong centre. This is the "informative but
  mis-calibrated" output of MI_LOG E12, and its first-order part needs no gallery.
- Anchoring matters: subtracting the own mean without an anchor gives 38.4 (ATM 3300). Adding the per-dimension std: +0.5.
- Mean from DISJOINT trials at another noise level (the subject's training-image trials, 4 repetitions; EEG anchor also
  at 4 repetitions) applied to the 80-repetition test queries: ATM 42.9 / 39.5 / 40.7. The mean embedding depends on the
  number of repetitions, so the calibration trials must be as clean as the queries (see K11).
- Fewer repetitions per query (mean from the same r-repetition queries), ATM 3300: r=1 6.8 -> 6.4, r=4 16.1 -> 18.9,
  r=16 27.2 -> 36.9, r=80 33.4 -> 48.3. Same law as the substitution: it repairs calibration, which binds once trials are clean.
- MI_LOG E02 had this for TSConv only (+3.9 at the output) and filed moment matching as minor: TSConv is the encoder
  where it matters least.

## K11 where the calibration mean may come from (18:14) — `k11_calib_source.py`, image anchor, 10 folds
| encoder | plain | own 200 queries | queries of the OTHER 100 images | 5 / 10 / 20 / 50 / 100 other queries | running mean (causal), all / after the 20th query |
|---|---|---|---|---|---|
| ATM-128 vc 3300 | 33.4 | 48.9 | 48.6 | 36.7 / 42.4 / 45.8 / 47.9 / 49.1 | 47.2 / 48.3 |
| ATM-128 vc 3301 | 31.4 | 48.3 | 47.9 | 36.4 / 41.9 / 44.9 / 47.1 / 47.5 | 46.8 / 47.8 |
| ATM-128 vc 3302 | 31.9 | 48.6 | 48.6 | 36.6 / 42.6 / 45.3 / 47.7 / 48.4 | 47.1 / 48.3 |
| rmix | 31.0 | 44.4 | 44.2 | 33.6 / 39.1 / 41.9 / 43.2 / 44.0 | 42.7 / 43.8 |
| LaBraM vc | 25.7 | 40.3 | 39.8 | 30.0 / 35.0 / 37.3 / 39.5 / 40.5 | 38.8 / 40.0 |
| TSConv vc | 32.4 | 37.0 | 36.6 | 28.1 / 31.5 / 34.1 / 36.2 / 36.6 | 35.3 / 36.2 |
| Squeezeformer | 36.5 | 41.0 | 41.2 | 31.0 / 35.8 / 38.3 / 40.4 / 41.4 | 39.3 / 40.2 |

- The mean is a property of the subject, not of the scored queries: estimated on other images it gives the same accuracy.
- About 20 unlabelled trials give most of it, 50 all of it; with fewer than ~10 a conv encoder loses (the estimate needs
  shrinkage toward "no shift" there).
- Fully causal (each query calibrated with the queries seen so far): ATM 47.1, 3 seeds, cold start included.
- **Single-encoder inductive number to beat from here: ATM-128, validation-selected, 3 seeds: 32.2 plain, 41.8 key
  substitution, 48.6 calibrated (48.4 on other images, 47.1 causal).** Published best 35.3.

## K12 ensembles of calibrated members (18:20) — `k12_cal_ensemble.py`, image anchor, 10 folds
| pool | members | calibrated top-1 |
|---|---|---|
| validation-selected | ATM-128 seed 3300 / 3302 / 3301 | 48.8 / 48.7 / 48.4 |
| | LaBraM | 40.1 |
| | ATM x2 | 51.35 |
| | ATM x2 + LaBraM | 53.15 |
| | ATM x3 + LaBraM | 53.8 |
| | **ATM x3 + TSConv + LaBraM** | **54.15** |
| test-selected | ATM / LaBraM / rmix / Squeezeformer | 50.2 / 46.4 / 44.4 / 41.0 |
| | ATM + rmix | 54.8 |
| | ATM + Squeezeformer + rmix | 56.9 |

- With calibration the ensemble is worth +5.5 over one ATM (substitution era: +6). Two ATM seeds alone give +2.7: a
  good part of the ensemble gain is seed averaging of ATM, which a distilled ATM can hope to absorb.
- **Target for one distilled encoder: 54 (validation-selected ensemble); floor: 48.6 (one calibrated ATM).**

## Distillation, first read (18:17) — `k03_eval.py kd_tsconv tsconv_vc/seed3300`, fold 1 only
TSConv student, teacher t4 (ATM x2 + TSConv + LaBraM), tau 0.05, lambda 1, validation-selected:
own 42.5 -> 46.0, substituted 44.5 -> 48.0, calibrated 44.0 -> 47.5 (test-selected 43.0 / 44.5 / 43.5 -> 45.5 / 49.5 / 50.0).
+3.5 in every scoring mode on this fold; folds 5 and 7 running.

## K13 what a wrong key breaks, and what calibration restores (18:22) — `k13_seen_calib.py`, SEEN subjects, 200 test images
| encoder | own key | own key + calibration | wrong key | wrong key + calibration | no key | no key + calibration |
|---|---|---|---|---|---|---|
| ATM-128 vc | 73.3 | 79.9 | 38.3 | 47.0 | 19.1 | 49.6 |
| TSConv vc | 65.9 | 69.4 | 47.3 | 52.0 | 38.9 | 53.6 |
| rmix | 68.6 | 83.8 | 21.5 | 33.1 | 24.9 | 42.5 |
| LaBraM vc | 65.2 | 76.9 | 23.9 | 31.2 | 12.9 | 32.1 |

- Calibration also helps subjects the model WAS trained on, on new images: +6.6 (ATM), +15 (rmix), +12 (LaBraM). So the
  offset it removes has two sources: the test condition (80-repetition averages of unseen images against 4-repetition
  training trials) for everyone, and the subject for a new one. Of ATM's +15.5 on a new subject, about 7 is the test condition.
  (Lead outside this log's scope: the same shift should be free accuracy in the intra-subject benchmark.)
- A seen subject under a wrong key, calibrated (47.0), sits where a calibrated NEW subject sits (48.6). The remaining 30
  points to a correctly keyed seen subject (79.9) are not an offset: they are the key-indexed decoder itself (K07: an
  oracle key recovers at least 14 of them through the input).
- Removing the key and calibrating is no worse than a wrong key and calibrating: once the centre is repaired, which
  wrong key the readout gets hardly matters.

## K14 second-order calibration (18:25) — `k14_calib_cov.py`, 10 folds
Whiten with the subject's own covariance, re-colour with the anchor's, shrinkage s, blended with the mean-only shift by b.

| | ATM 3300 | ATM 3301 | rmix | TSConv |
|---|---|---|---|---|
| mean only, image anchor / EEG anchor | 48.9 / 48.3 | 48.3 / 46.9 | 44.4 / 44.9 | 37.0 / 37.2 |
| half blend, EEG anchor, s 0.6 | 50.3 | 49.6 | 45.3 | 36.7 |
| half blend, image anchor, s 0.6 | 49.5 | 50.2 | 44.2 | 34.6 |
| full re-colouring (b 1), best | 49.3 | 48.8 | 42.2 | 33.7 |

Half-strength covariance matching adds about +1.5 to ATM, nothing to rmix, and costs TSConv. Not robust across encoders;
the mean shift stays the default. Kept as an ATM-specific option to re-test on the final encoder.

## K12b ATM on other image spaces as ensemble members (18:30) — `k11_calib_source.py`, `k12_cal_ensemble.py` pool `vc_img`
Calibrated singles, validation-selected: ATM on InternViT L28 48.9 (the base), bigG L27 46.2, InternViT L20 45.7,
DINOv2 L40 28.6, InternViT L44 24.2, CLIP-H 20.4. No better base on disk. As ensemble members the image spaces add more
than seeds do:

| members (calibrated, validation-selected) | top-1 |
|---|---|
| ATM L28 x2 seeds | 51.35 |
| ATM L28 + ATM L20 | 53.3 |
| ATM L28 + ATM bigG + ATM L20 | 54.35 |
| ATM L28 + ATM bigG + ATM L20 + LaBraM | **55.9** |
| ATM L28 x2 + ATM bigG + ATM L20 + LaBraM | 56.05 |

Teacher `t6` = ATM L28 seeds 3300, 3302 + ATM bigG + ATM L20 + LaBraM + TSConv (exported for folds 1, 5, 7); the queued
ATM and rmix students use it (the TSConv student used `t4`). Loader fix: `sk_lib.load` now reads each arm's own image features.

## K15 the part of the offset that seen subjects share is the repetition count (18:40) — `k15_condition_shift.py`
Queries = mean of the first r repetitions. Plain -> calibrated (image anchor).

| | r = 4 (as in training) | r = 16 | r = 80 |
|---|---|---|---|
| ATM, seen subjects | 35.4 -> 36.1 | 64.6 -> 68.2 | 72.6 -> 80.8 |
| ATM, new subject | 16.1 -> 18.6 | 27.2 -> 37.1 | 33.4 -> 48.9 |
| rmix, seen subjects | 39.9 -> 39.8 | 64.7 -> 71.2 | 64.5 -> 83.6 |
| rmix, new subject | 13.7 -> 16.5 | 24.8 -> 32.3 | 31.0 -> 44.4 |
| TSConv, seen subjects | 24.8 -> 23.4 | 51.9 -> 51.9 | 67.3 -> 69.7 |
| TSConv, new subject | 10.5 -> 10.5 | 21.9 -> 24.5 | 32.4 -> 37.0 |

- Seen subjects gain nothing at 4 repetitions, the condition the models were trained in, and the gain grows with the
  number of repetitions: the encoders are trained on 4-repetition averages and tested on 80-repetition averages, and
  their output centre moves with the noise level. Without calibration rmix does not improve at all from 16 to 80
  repetitions on seen subjects (64.7 -> 64.5); with it, 71.2 -> 83.6.
- A new subject gains about twice what a seen subject gains at every r (ATM +2.5 / +9.9 / +15.5 against +0.7 / +3.6 /
  +8.2): its offset has the condition part plus its own.
- Consequences: (1) the calibration is a correction of the benchmark's train / test repetition mismatch as much as of
  the subject; any THINGS-EEG2 model scored on 80-repetition queries should gain from it, intra-subject included (not
  tested here); (2) calibration trials must match the queries' repetition count (K10: 4-repetition trials give half the gain).

## K16 side check: re-centring on intra-subject models (18:42) — `k16_intra_calib.py`, cvpr/intra, 10 subjects
One-cycle ATM trained and tested within a subject (17 electrodes): ATM-128 87.75 -> 88.8 (causal running mean 87.05),
ATM fd512 86.9 -> 89.3 (causal 86.9). K15's prediction holds weakly: +1.0 and +2.4 with the mean over all 200
queries, nothing as a running mean. A one-subject model moves its centre much less between 4 and 80 repetitions than a
9-subject model does on its seen subjects (+8). Not pursued.

## K17 limit of the calibration: the mean must come from a representative sample of images (18:47) — `k17_calib_bias.py`
The 200 test images are split into k clusters of the image space (k-means); the queries of one cluster are scored
against the full gallery, with the subject mean estimated either on all 200 queries or on that cluster's queries only.

| | plain | mean from all 200 queries | mean from the cluster only, k = 2 / 4 / 8 |
|---|---|---|---|
| ATM-128 vc 3300 | 33.4 | 48.9 | 31.8 / 26.0 / 20.7 |
| rmix | 31.0 | 44.4 | 26.8 / 21.4 / 16.3 |

A mean taken on a semantically narrow stream contains that stream's content, and removing it is worse than no
calibration. The shift has to be estimated on unlabelled trials that sample the image distribution broadly (K11: 20-50
random ones are enough, and they need not be the scored ones); once estimated it can be applied to any stream (the
"all" column is exactly that: narrow streams, shift from a broad sample). To be stated with every calibrated number.

## K18 where to re-centre inside ATM (18:52) — `k18_layer_calib.py`, atm128_vc seed 3300, 10 folds
The new subject's mean activation at one stage is replaced by the training subjects' mean activation at that stage;
the rest of the network runs on. "then output" = followed by the usual output calibration.

| stage re-centred | alone | then output |
|---|---|---|
| none | 33.4 | 48.85 |
| input embedding (= the key, as in the substitution) | 39.9 | 41.6 |
| after attention | 39.3 | 45.9 |
| after feed-forward | 44.5 | 48.3 |
| encoder output | 45.2 | 48.8 |
| conv readout, after the temporal / spatial nonlinearity | 47.7 / 47.5 | 49.3 / 48.5 |
| after the residual head block | 48.35 | 50.1 |
| output of the projector | 48.2 | 48.8 |

- The later the re-centring, the better, monotonically from the input (40) to the output (48). Replacing the key at
  the input, which is what the substitution does, is the worst place: it also changes what every later nonlinearity
  computes, whereas the network run on the subject's OWN key produces a correct geometry that only needs moving.
- Re-centring an early stage and then the output is worse than the output alone (input: 41.6): the early replacement
  destroys information the output shift cannot give back. A hidden shift after the residual block plus the output
  shift gains +1.25, within what K14 gets from second moments; not a new lever.
- For the paper: one curve, input (key substitution) to output (calibration), says where the subject effect should be removed.

## K19 embedding-space averaging instead of input averaging (18:55) — `k19_embed_avg.py`, atm128_vc 3300 — NEUTRAL, and it corrects K15's reading
A query = mean embedding of 80 / g input averages of g repetitions (g = 80 is the benchmark, g = 4 the training condition).

| g | new subject, plain -> calibrated | seen subject, plain -> calibrated |
|---|---|---|
| 80 | 33.4 -> 48.9 | 72.8 -> 80.5 |
| 16 | 33.4 -> 49.1 | 75.3 -> 80.2 |
| 4 | 31.1 -> 49.0 | 76.4 -> 80.1 |
| 1 | 22.7 -> 40.2 | 57.2 -> 72.3 |

- Feeding the encoder its training condition (g = 4) and averaging the embeddings recovers half of the seen-subject
  gain without any calibration (72.8 -> 76.4) but nothing for a new subject (31.1), and after calibration every g from
  4 to 80 lands on the same number. No lever here; the benchmark's input averaging stays.
- Correction to K15: the offset is not created by cleaner inputs. It is there at 4 repetitions too (g = 4: calibration
  still +3.7 on seen subjects once 20 embeddings are averaged); it is small against the trial noise of one 4-repetition
  query and becomes the dominant error once the noise is averaged out, in the input or in the embedding.

## Distillation into TSConv, folds 1 5 7 (19:18) — `k03_eval.py kd_tsconv tsconv_vc/seed3300`
Teacher t4, tau 0.05, lambda 1. Held-out top-1, baseline -> student.

| selection | own key | substituted | calibrated | calibrated per fold (1 / 5 / 7) |
|---|---|---|---|---|
| validation | 33.7 -> 35.5 | 36.5 -> 36.5 | 36.5 -> 37.3 | 44.0 -> 47.5 / 28.5 -> 31.5 / 37.0 -> 33.0 |
| test | 35.8 -> 37.3 | 37.8 -> 39.8 | 38.2 -> 42.7 | 43.5 -> 50.0 / 35.5 -> 36.5 / 35.5 -> 41.5 |

- Positive on two folds, negative on the third with validation selection; +4.5 calibrated with test selection. The
  validation-selected epoch of fold 7 is 8.5 points below the test-selected one for the student (33.0 vs 41.5): the
  concept-validation loss picks epochs noisily, more so for a distilled student.
- TSConv is not the candidate (calibrated 37-43 against 48.6 for ATM). Read as: distillation is not harmful and
  probably worth a few points; the decision is taken on the ATM students (kdft_atm running, then kd_rmix, kd_atm).

## Fine-tuning ATM with distillation, first fold (19:56) — `kdft_atm`, fold 5 (folds 7, 1 to come)
ATM seed 3300 (validation-selected) + 25 epochs at lr 1e-4 with the t6 teacher (tau 0.05, lambda 1), fixed epochs saved.
Calibrated top-1 / mean rank of the target: start 49.5 / 2.88; epoch 10 51.0 / 2.66; epoch 25 49.0 / 2.77;
validation-selected 51.5 / 2.66; test-selected 47.5 / 2.79. No movement outside the noise of one fold (a 200-query
fold has a standard error of 3.5 points on top-1; `k03_eval.py` now also prints the mean rank of the target, which is
steadier). Single-head distillation into ATM looks like a wash so far.

## K20 / K21 one frozen trunk, several distilled linear heads (19:58-20:02) — `k20_linear_heads.py`, `k21_heads.py`, folds 1 5 7
Trunk = ONE frozen ATM (atm128_vc seed 3300, the 1440-d input of its head). For each teacher member k, a linear head is
fitted in closed form (ridge regression on the 148,860 clean training trials) from the trunk feature to member k's EEG
embedding. A new subject is scored by each head against member k's own image embeddings, each head calibrated (K10),
and the heads' cosine matrices are averaged. One encoder forward; no gradient step.

| readout of the single frozen trunk | top-1 | mean rank of the target |
|---|---|---|
| its own trained head | 49.7 | 3.11 |
| one distilled head (mimics itself) | 50.2 | 3.06 |
| **3 heads: ATM teachers on InternViT L28, bigG L27, InternViT L20** | **55.5** | 2.65 |
| 4 heads: + a second L28 seed | 55.0 | 2.62 |
| 6 heads: + LaBraM + TSConv | 54.5 | 2.61 |
| the 6 teacher models themselves (six encoders) | 56.7 | 2.07 |
| 6 heads fitted to the IMAGE embeddings instead of the teachers' EEG embeddings | 37.0 | 5.23 |
| 6 distilled heads without calibration | 40.3 | – |

- **A single encoder with three distilled linear heads recovers most of what the six-model ensemble adds
  (49.7 -> 55.5 of 56.7 on these folds)**, where single-head distillation gave nothing. What the ensemble contributes is
  mostly a set of different TARGET SPACES for the same EEG features, and those can be read from one trunk.
- A head that mimics LaBraM from the ATM trunk scores 45.0 on new subjects; LaBraM itself, calibrated, 40.1 (10 folds).
- The teachers' EEG embeddings are the right regression targets; regressing onto image embeddings fails (37.0).
  Calibration is still needed (40.3 without).
- From the 128-d trunk output instead of the 1440-d feature: 6 heads 53.3.
- Next: 10 folds (queued: teacher export for the 7 other folds, then `k21_heads.py` on all folds); then train trunk and
  heads together.

## K22 / K23 the distilled heads: ridge strength and nonlinear heads (20:10-20:15) — `k22_heads_ridge.py`, `k23_mlp_heads.py`, CPU
Three heads (ATM teachers on L28, bigG, L20), frozen ATM trunk, folds 1 5 7.
- Ridge strength (x mean eigenvalue) 0.003 / 0.01 / 0.03 / 0.1 / 0.3 / 1 / 3: top-1 54.2 / 54.3 / 54.3 / 55.5 / 55.5 /
  54.7 / 52.5, mean rank 2.67 / 2.67 / 2.65 / 2.65 / 2.68 / 2.77 / 3.00. Flat over two orders of magnitude; 0.1 kept.
- Heads trained by gradient on the cached trunk features (1 - cosine to the teacher embedding, 8 epochs), folds 5 and 7:
  linear 50.3 (rank 3.19), 2-layer MLP 48.5 (3.32), the encoders' own head shape (residual GELU + LayerNorm) 50.0 (3.29);
  ridge on the same two folds 50.75 (3.23). Fold 1: 62-64.5 against 65.0. Input dropout 0.5 does not help.
  Nonlinear heads add nothing: on a frozen trunk the closed-form linear heads are the ceiling. More can only come
  from more spaces or from training the trunk with the heads.

## Plan change (20:20)
Single-head distillation is dropped as the main route (kd_atm, kd_rmix moved to `deferred.txt`). Queue: teacher export
for the 7 other folds -> `k21_heads.py` on 10 folds -> ep_pc (person-centred training, EEGProject) -> kdft_atm fold 1 ->
`msft_atm` / `ms_atm`: ONE ATM trained with three heads at once (main space L28 + bigG + L20, `train.py --aux_targets
<dirs> --aux_dim 128 --aux_lambda 1`, heads saved, scored with `k24_multispace.py`), fine-tuned 25 epochs from the
single-space ATM and from scratch. That removes the teachers altogether if it matches the distilled heads.

## Fine-tuning ATM with single-head distillation: wash (20:28) — `kdft_atm`, folds 5 and 7
Calibrated top-1 / mean rank: start 46.0 / 3.70; epoch 10 47.25 / 3.63; epoch 25 46.0 / 3.79; validation-selected
46.75 / 3.60. Nothing. Fold 1 still queued to complete the three folds.

## K21 on 10 folds, three trunks (20:34-20:50) — `TRUNK=<run> k21_heads.py 1,...,10`, `k21_heads_<trunk>.csv` — ENCODER RESULT
One frozen ATM (validation-selected) + closed-form distilled linear heads + calibration. Held-out top-1 (mean rank).

| trunk (ATM-128, InternViT L28) | its own head | 3 heads: L28 + bigG + L20 | 4 heads: + 2nd L28 seed | 6 heads: + LaBraM, TSConv | six-model teacher |
|---|---|---|---|---|---|
| seed 3300 | 48.85 (3.04) | **53.65** (2.59) | 53.65 | 53.2 (2.58) | 56.1 (2.18) |
| seed 3301 | 48.25 (3.11) | **51.3** (2.70) | 51.9 | 51.1 (2.70) | 56.1 |
| seed 3302 | 48.6 (3.05) | **52.7** (2.72) | 53.3 | 52.9 (2.66) | 56.1 |
| mean | 48.6 | **52.6** | 52.95 | 52.4 | 56.1 |

- Three heads against the trunk's own head, per held-out subject (seed 3300): +8.0 +5.0 +5.5 +3.5 +5.0 +5.0 +4.5 +2.5 -0.5
  +9.5; paired p = 0.0004 (seed 3301: p = 0.014; seed 3302: p = 0.0007).
- **One encoder forward, three linear heads: 52.6 (3 seeds), +4.0 over the single head, about half of what six
  separate encoders add (56.1), best published 35.3.** Student size: the ATM trunk (0.9 M parameters) + 3 x (1440 x 128).
- Heads for LaBraM / TSConv spaces do not add; a second seed of the same space adds +0.3. What counts is the number of
  distinct IMAGE spaces, which is the cheap axis to extend (needs one ATM teacher per space).
- Per-head (seed 3300): L28 48.6, bigG 44.75, L20 44.5, LaBraM 40.4, TSConv 42.0.
- Status of the "one encoder" question: answered in a first form (distillation into several heads works, into one head
  does not). Open: does training the trunk with the heads (ms_atm, msft_atm) beat the frozen trunk, and does it need
  the teachers at all.

## K21b more image spaces as heads (20:58) — `TEACHER=t9 k21_heads.py 1,5,7`
Three more ATM teachers exist under validation selection, on weak spaces (calibrated singles 20-29): CLIP-H, DINOv2
L40, InternViT L44. As distilled heads on the frozen trunk they score 15.8 / 25.8 / 21.7 alone, and added to the three
good heads (55.5, rank 2.65): + CLIP-H 52.8 (2.87), + DINOv2 55.2 (2.72), + L44 51.3 (2.82), all six spaces 48.8 (3.23).
A head only helps if its space aligns well by itself (L28 48.6, bigG 44.8, L20 44.5). Candidates left: other InternViT
mid layers, EVA02; each needs a teacher (10 ATM folds) in the distilled version, or just one more head in the jointly
trained version (ms_atm).

## Person-centred training (21:02) — `ep_pc` (EEGProject, `--person_centre_train`), fold 1 — NEGATIVE, stopped
Train as you test: every training output is re-centred per subject inside the batch; at test the new subject is
re-centred on the training subjects' mean. EEGProject fold 1, baseline -> person-centred, calibrated top-1 / mean rank:
validation-selected 43.0 / 4.36 -> 43.5 / 5.49; test-selected 47.5 / 4.13 -> 44.0 / 4.86; fixed epochs 30 / 40 / 50:
40.0 / 43.5 / 41.5, rank 4.8-5.3. Top-1 equal, ranking 20-25% worse at every checkpoint. The model does not need the
centring during training to benefit from it at test. Folds 5 and 7 cancelled. kdft_atm fold 1 deferred (two neutral folds).

## Standing numbers (21:10) — from the caches of K21, validation-selected, 3 ATM seeds, 10 held-out subjects
| | top-1 | top-5 | top-1 per held-out subject 1..10 |
|---|---|---|---|
| best published inductive (Structured Visual Target Learning, arXiv 2609.36971) | 35.3 | 65.6 | |
| ATM-128, plain | 32.2 | – | |
| ATM-128, calibrated | 48.6 | 80.8 | 55.7 48.8 51.2 35.8 48.8 48.0 43.5 42.5 48.3 63.0 |
| ATM-128 trunk + 3 distilled heads, calibrated | **52.6** | **83.3** | 62.5 52.5 55.0 38.8 54.5 53.5 48.2 44.2 47.0 69.3 |
| six separate encoders, calibrated (not a single model) | 56.1 | – | |

Head weights are not sensitive: (1, 1, 1) 53.65 / 51.35 / 52.65 for the three trunks, (2, 1, 1) 53.6 / 51.9 / 53.4.
Leaving one head out (seed 3300): without bigG 51.05, without L20 51.9, without L28 50.25.
All calibrated numbers use the mean of the subject's 200 unlabelled test queries (K11: other images 48.4, causal 47.1
for the single head; K17: the sample must be semantically broad).

## One ATM trained with three heads, no teacher (18 Nov, 10:10) — `k24_multispace.py msft_atm ms_atm`, folds 1 5 7
`train.py --aux_targets <bigG L27> <InternViT L20> --aux_dim 128 --aux_lambda 1`: the main head (InternViT L28) plus one
linear head per extra image space, each with its own learned image projection, trained together. Scored in the three
spaces, each calibrated, cosine matrices averaged. Calibrated top-1 (mean rank of the target):

| model | main head alone | all three heads |
|---|---|---|
| single-space ATM (atm128_vc 3300) | 49.7 (3.11) | – |
| frozen ATM + 3 distilled heads (K21) | – | 55.7 (2.64) |
| msft_atm: single-space ATM fine-tuned 25 epochs with the two extra heads, validation-selected / epoch 25 | 49.3 / 49.5 | 53.7 (2.55) / 54.0 (2.58) |
| ms_atm: trained from scratch with the three heads, validation-selected / test-selected | 49.5 / 52.0 | 54.5 (2.80) / 54.2 (2.62) |

- A single ATM trained once on three image spaces gives +4 to +5 over the single-space ATM with no teacher at all, at
  the level of the distilled heads within the noise of three folds (top-1 1.2 to 2 lower, mean rank equal).
- The extra heads do not improve the main head (49.3-49.5): the gain is in reading several spaces, not in a better trunk.
- 25 epochs of fine-tuning are as good as 100 epochs from scratch.
- Queued: the 7 other folds for both (msft first, about 3 h; then ms_atm).

## rmix as the student, first read (10:14) — `TRUNK=rmix/seed3300 k21_heads.py 1,5,7` (test-selected trunk, indicative)
Closed-form distilled heads (teacher t6) on the frozen rmix trunk (TokenREVE, 5.3 M parameters; the 256-d mean token).
Calibrated top-1 (mean rank): its own head 43.3 (3.41); one head mimicking an ATM teacher 45.2; **three heads
(L28 + bigG + L20) 51.0 (3.30)**; four ATM heads 51.7; six heads 50.0. On the same folds the ATM trunk gives 49.7 -> 55.5.
The larger token encoder gains more from the heads (+7.7 against +5.8) but stays 4.5 below the small ATM: the heads
cannot give rmix what its trunk lacks on new subjects.

## Queue (10:12)
rmix students under the validation protocol, folds 1 5 7, 27 min per fold each: `base_rmix` (baseline and trunk),
`kd_rmix` (single head, ensemble-mean distillation, the form used in the challenge), `ms_rmix` (three heads, no
teacher), `hydra_rmix` (three heads, each distilled from its own teacher member: `--kd_members 0 2 3`, new
`train.py` code). Then `msft_atm` on the 7 other folds (10-fold number of the teacher-free ATM) and `hydraft_atm`
(the ATM fine-tune with per-head distillation). `ms_atm` from scratch on 10 folds dropped (25 epochs of fine-tuning
do as well). The partial fold `msft_atm/_aborted_sub02_*` can be deleted.

## K26 the three-head readout under each calibration source (10:25) — `k26_heads_calib_source.py`, 3 trunk seeds x 10 folds
| subject mean estimated on | seed 3300 | 3301 | 3302 | mean |
|---|---|---|---|---|
| nothing (no calibration) | 36.0 | 35.9 | 34.5 | 35.5 |
| its 200 queries | 53.65 | 51.3 | 52.7 | **52.55** |
| the queries of the other 100 images | 52.55 | 51.4 | 52.3 | 52.1 |
| 20 / 50 other queries | 49.5 / 51.8 | 48.4 / 50.7 | 49.2 / 51.5 | 49.0 / 51.35 |
| the queries seen so far (causal), all / after the 20th | 51.4 / 52.7 | 50.1 / 51.3 | 51.1 / 52.4 | 50.9 / 52.1 |

The multi-head readout keeps its gain under every source: 52.1 with a mean from other images, 50.9 fully causal with
the cold start included (single head: 48.4 and 47.1). Without calibration the three heads give 35.5 (single head 32.2).

## User steer (10:26)
"If you have a way of directly training a multi-head ATM model that performs as well as a distilled student that's
even better: simpler, more practical." -> the directly trained multi-space ATM (`--aux_targets ... --aux_dim 128`) is the
target method; the distilled heads are the reference to match (3 folds: 55.7 distilled, 54.0-54.5 direct).
Plan: (1) `msft6_atm`: six heads (L28 + bigG L27 + L20 + L36 + EVA02 L35 + DINOv2 L20; plain single-space ATM on
those spaces: 35.2 / 31.5 / 31.2 / 33.7 / 31.8 / 28.8 on 3 folds), fine-tune variant, folds 1 5 7, to see whether more
spaces close the gap; (2) the rmix students; (3) the chosen direct recipe from scratch on all 10 folds.
Banks of the extra spaces are kept in half precision when `--aux_dim` is set (GPU memory).

## rmix under the validation protocol: baseline and distilled heads (11:20) — `base_rmix`, `TRUNK=base_rmix k21_heads.py 1,5,7`
- `base_rmix` (TokenREVE mixed tokens, the cvpr/loso recipe, validation concepts), folds 1 5 7, calibrated top-1 /
  mean rank: validation-selected 42.5 / 3.80, epoch 30 44.8 / 3.80, test-selected 44.2 / 3.54 (plain 25.2). Early
  stopping ends every fold at epoch 34; 21 min per fold.
- Three closed-form distilled heads (t6: ATM teachers on L28, bigG, L20) on that frozen trunk: **47.0** (own head
  42.5; four ATM heads 46.5; six heads 46.2). On the test-selected cvpr/loso rmix: 43.3 -> 51.0.
- So an rmix student built the same way as the ATM student gains +4.5 but ends below the single-head ATM (49.7) and
  far below the ATM student (55.7) on the same folds. Trained rmix students follow (kd_rmix, ms_rmix, hydra_rmix).

## User steer (11:30): "let's focus on ATM"
rmix students removed from the queue (kd_rmix, ms_rmix, hydra_rmix -> `deferred.txt`). What stands for rmix: distilled
heads 42.5 -> 47.0 (validation-selected trunk), below the single-head ATM. Queue: msft6_atm (running) -> the directly
trained multi-head ATM from scratch on the 7 other folds (recipe = 3 or 6 heads, decided on msft6) -> msft_atm on the
7 other folds -> hydraft_atm.

## More heads in the directly trained ATM (11:38) — `msft6_atm`, fold 1 — NEGATIVE, stopped
Six heads trained together (L28 main + bigG L27 + L20 + L36 + EVA02 L35 + DINOv2 L20), 25-epoch fine-tune.
Fold 1, calibrated top-1 (mean rank), epoch 25 / validation-selected:
all six heads 59.5 (1.59) / 58.5 (1.54); the first three heads of the same model 64.0 (1.50) / 63.0 (1.47);
the three-head model `msft_atm` 63.0 (1.47) / 62.0 (1.38); distilled three heads 65.0 (1.49).
Single heads: L28 58.5, bigG 54.0, L20 53.0, L36 52.0, EVA02 51.0, DINOv2 51.0.
The three added spaces lower the average by 4.5 points on this fold and training them next to the others neither helps
nor hurts the first three. Same lesson as K21b: three spaces (L28, bigG L27, L20) is the recipe. Folds 5, 7 cancelled.

## K27 second-order calibration for the three heads (11:45) — `k27_heads_cov.py`, 3 trunk seeds x 10 folds — no gain
Mean shift -> half-strength covariance matching (K14), per head: single head 48.6 -> 49.6; three heads 52.55 -> 52.7
top-1 with a worse mean rank (2.67 -> 2.82). The extra heads already bring what the covariance step brought. Mean shift kept.

## Why three spaces help: the heads' errors (11:48) — cached head scores, trunk seed 3300, 10 folds
- Correlation of the hit / miss patterns of two heads on the same trunk (phi): two seeds of the same space 0.73;
  two different image spaces 0.47-0.54. The cosine matrices themselves correlate 0.98 (same space) and 0.92-0.94.
- Of the queries, 27.0% are right for all three heads, 64.4% for at least one; the average gets 53.6% (best single 48.6).
  A head on another image space fails on different queries although it reads the same EEG features.

## K28 calibration from very few trials, with shrinkage (12:05) — `k28_few_trials.py`, three heads, trunk seed 3300
`e <- e - n / (n + n0) * (mean of n calibration trials - anchor)`; the other queries are scored; 10 folds x 10 draws.
No calibration: 36.0.

| n calibration trials | n0 = 0 (plain mean) | n0 = 2 | n0 = 5 | n0 = 10 |
|---|---|---|---|---|
| 3 | 33.0 | 41.8 | 42.0 | 40.3 |
| 5 | 40.0 | 44.3 | 44.3 | 42.6 |
| 10 | 45.7 | 47.3 | 47.6 | 46.0 |
| 20 | 49.4 | 49.8 | 49.7 | 48.9 |
| 50 | 51.8 | 51.9 | 51.9 | 51.4 |

With the estimated shift shrunk by n / (n + 5), the calibration helps from the third unlabelled trial on (+6) and
never hurts; the plain mean needs about 5 trials to break even. Use n0 = 5 in the causal / few-trial protocols.

## Status (12:05)
Interactive node restarted by the user at 11:59; the `key_gpu` job and its queue were not affected.
Running: `ms_atm` from scratch on folds 2 3 4 6 8 9 10 (fold 2 at epoch 23 at 12:00; about 80 min per fold, done
around 21:00). Then `msft_atm` on the same folds, then `hydraft_atm` (folds 1 5 7).

## Queue (12:15)
After `ms_atm` on the 7 other folds (about 21:00): `hydraft_atm` (25-epoch fine-tune with the two extra heads AND each
head distilled from its teacher member, folds 1 5 7: does distillation during training beat direct training?), then
`msL_atm` (the three-head ATM with a larger encoder: 2 attention layers, feed-forward width 1024, from scratch, folds
1 5 7: does capacity help the multi-head model?). `msft_atm` on the 7 other folds dropped (direct training is the target).
Earlier capacity facts for single-head ATM: output width 512 / trunk 1024 is worse than 128 / 128 (30.3 vs 35.2 plain).

## K29 key fine-tuning, launched (12:45) — `k29_key_tuning.py`, CPU job `key_tune` (Mee_Commun_CPU), folds 1 3 5 7
User's question: can tuning ONLY the key bring a new subject toward within-subject accuracy from little data?
Frozen ATM (atm128_vc 3300), the 63x250 readout key is the only parameter, calibration data = n of the subject's
training-image trials (n = 50, 200, 1000, 5000), init = K or the subject's own key, 300 steps.
(a) labelled (few-shot); (b) unlabelled: the set of the n calibration images is known but not the pairing, pseudo-labels
by one-to-one assignment of re-centred embeddings to the images, re-assigned every 30 steps.
Scored on the 200 test queries, plain and re-centred. K07 (all 16,540 labelled trials, 600 steps, from K): 44 -> 58 plain.
K08 (agreement between repetitions, no images at all) already failed.

## Exchange with the challenge session (13:00)
Report of the re-centring and key levers sent to `neurobridge-ssl-80` at the user's request. Its reply: no clash in the
shared code (my train.py flags are in its running jobs and train normally). In the challenge regime (single trials,
64-trial batches mixing ~11 people, r = 3) both halves of the re-centring were already tested: removing the batch or the
oracle person mean costs 15-20 % (MI_NEURIPS_LOG N06), shifting toward the mean trained-image target gives +2-4 % on
plain members and nothing once soft-label members are in. Consistent with the law found here (gain ~0 at r = 1).
Untried strict-setting variant noted: key-free training scored with the AVERAGE training key subtracted at test.

## Untested combinations: plan and first result (13:20)
User: "when the main runs are done please try the promising combinations that have not been tested yet"; and
"do not combine our test-time adaptations with GALA or this kind of stuff for now" (nothing gallery-based below).

Done, CPU: **shrinkage inside the causal running mean** (`e_t - t/(t+n0) (running mean - anchor)`, 3 trunk seeds x 10 folds x 5
orders). Three heads, all queries / the first 20 / after the 20th: n0 = 0 50.9 / 39.6 / 52.1; n0 = 2 51.4 / 44.8 / 52.1;
n0 = 5 51.3 / 45.1 / 52.0; n0 = 10 50.9 / 43.7 / 51.6. Single head, n0 = 5: 47.4 / 42.4 / 47.9 (n0 = 0: 47.1).
Shrinkage repairs the cold start (+5.5 on the first 20 queries) and costs nothing later. Causal default: n0 = 5.

Evaluation-only, CPU jobs `key_cpuA` / `key_cpuB` (Mee_Commun_CPU):
- `k31_heads_few_reps.py`: three heads at 1 / 4 / 16 / 80 repetitions per query, plain and re-centred.
- `k30_fixed_shift.py`: a strictly inductive shift (training-set gap between the image and EEG clouds, no data of
  the new subject), on the benchmark query and on the mean embedding of 4-repetition sub-averages.
- `k24_multispace.py msft_atm ms_atm`: now also prints the heads WITHOUT re-centring (strict setting, direct model).
- `TRUNK=atm128/seed3300 k21_heads.py`: three distilled heads on a TEST-selected trunk (comparable to the old 35.8).
- `key_tune` (K29): key fine-tuning, few labelled / unlabelled-with-image-set, with and without re-centring.

Training, GPU queue after `ms_atm` (7 folds): `hydraft_atm` (per-head distillation during training) -> `ep_dm`,
`tsconv_dm` (key-free training, to be scored with re-centring: does it stack?) -> `msL_atm` (larger ATM, three heads)
-> `ms_atm_dm` (three-head ATM trained key-free). All on folds 1 5 7; expected to finish tomorrow around 11:00.
Not planned: group SubjectMix on ATM (two clean TSConv pairs showed a wash, memory note of October).
After the 10-fold direct result: two more seeds of the direct three-head ATM if it holds.

## K31 three heads at fewer repetitions per query (13:35) — `k31_heads_few_reps.py`, trunk seed 3300, 10 folds
| repetitions per query | single head | single + re-centring | three heads | three heads + re-centring |
|---|---|---|---|---|
| 1 | 6.2 | 5.3 | 7.0 | 6.6 |
| 4 | 15.8 | 18.4 | 16.5 | 19.1 |
| 16 | 26.9 | 37.4 | 28.1 | 39.7 |
| 80 | 33.0 | 48.6 | 36.0 | 53.65 |

Both levers grow with the repetition count and vanish on single trials: the heads add +0.7 at 4 repetitions, +2.3 at 16,
+5 at 80 (on top of re-centring). They are levers of the low-noise regime, like the re-centring (K15-K19).

## Direct three-head ATM without re-centring (13:35) — `k24_multispace.py msft_atm ms_atm`, folds 1 5 7, strict setting
Validation-selected, no data of the new subject: main head 34.2 (`ms_atm`) / 35.3 (`msft_atm`), three heads 37.5 / 37.3.
Same folds, same seed: single ATM 35.7, three DISTILLED heads on it 38.7. So in the strict setting the three heads give
about +2 to +3 whether trained directly or distilled; the direct model's main head is not better than a plain ATM.

## K32 centre or geometry? (13:45) — `k32_centre_vs_geometry.py`, ATM seeds 3300 / 3301, 10 folds, 2 seen subjects per fold
User (12:30): the claim "the own key gives the right geometry around the wrong centre" is vague; it needs measuring.
Normalised embeddings of the 200 test queries against their image embeddings. `rot` / `lin` = ORACLE output maps fitted with
labels on 4,000 of the subject's training-image trials (centred clouds), applied to the test queries after re-centring.

| condition | plain | re-centred | offset share | CKA | pair cos (centred) | + oracle rotation | + oracle ridge map |
|---|---|---|---|---|---|---|---|
| new subject, own key | 33.4 / 31.4 | 48.9 / 48.3 | 32 % / 33 % | 0.73 | 0.46 | **71.9 / 71.3** | 55.3 / 55.1 |
| new subject, key K (substitution) | 42.4 / 40.9 | 45.9 / 45.7 | 17 % / 18 % | 0.71 | 0.44 | 72.2 / 71.2 | 56.2 / 55.6 |
| seen subject, own key | 73.9 / 72.9 | 81.0 / 80.9 | 25 % / 24 % | 0.75 | 0.58 | 83.7 / 82.7 | 67.4 |
| seen subject, wrong key | 40.1 / 39.7 | 46.1 / 45.6 | 20 % | 0.70 | 0.44 | 73.0 / 72.5 | 61.4 |

Share of the embedding change caused by a key swap that is one constant vector: new subject (own -> K) 70 %; seen subject
(own -> wrong) 38 %.

Reading (replaces the vague claim):
- The SHAPE of a new subject's cloud is almost that of a seen subject (CKA 0.73 against 0.75), whatever key is fed.
- What differs is where the cloud sits and how it is ORIENTED: a third of the squared EEG-image distance is one constant
  vector (re-centring, +15.5), and the centred cloud is turned against the image cloud (pair cosine 0.46 against 0.58).
- One labelled orthogonal map undoes most of the turn: 48.9 -> 71.9, within 12 points of a seen subject (83.7), and it
  brings a seen subject under a wrong key to the same place (73.0). A seen subject under its own key gains 2.7: the map
  is not a generic train-to-test fix. After the rotation the key fed no longer matters (71.9 own, 72.2 K).
- So new subject = same shape, wrong centre (+15.5), wrong orientation (+23), residual 12 not linear at the output.
- For a new subject the key moves the output mostly as one constant vector (70 %), which is why substitution and
  re-centring overlap and do not stack; for a seen subject the key does much more than that (38 %).
- The ridge map is worse than the rotation (55): fitted on noisy 4-repetition trials it shrinks directions; the rotation
  cannot. Oracle input key (K07, all 16,540 labels) reached 58: the output rotation is the better place to adapt.
- Caveats: the rotation uses LABELS of the new subject (few-shot, a third setting). One model family (ATM). Finding it
  without labels is what Procrustes-on-the-gallery methods do; not run here (user: no gallery-based combination for now).
Next: K33, the rotation from 20 to 4,000 labelled trials, with an identity prior.

## K30 a strictly inductive fixed shift (14:00) — `k30_fixed_shift.py`, 10 folds — DEAD END
Shift estimated on training data only (training-image mean minus the training subjects' mean embedding), added to every query.
| | plain | + fixed shift | + own re-centring |
|---|---|---|---|
| ATM 3300 / 3301 / 3302 | 33.4 / 31.4 / 31.9 | 33.3 / 32.8 / 32.4 | 48.9 / 48.3 / 48.6 |
| TSConv | 32.4 | 32.4 | 37.0 |

Half or double strength: no better. On 4-repetition sub-averages: the same (31.4 -> 30.2). What the re-centring removes is
the new subject's OWN offset; nothing of it is common to all subjects and knowable in advance. Strict setting: no shift lever.

## K33 few-shot output rotation (14:10) — `k33_fewshot_rotation.py`, ATM seeds 3300 / 3301 averaged, 10 folds, 3 draws
R = orthogonal Procrustes of `X^T Y + n0 c I` on n LABELLED training-image trials of the new subject (4-repetition
averages), identity prior with pseudo-count n0. Reference: re-centring alone 48.6.

| labelled trials | n0 = 0 | n0 = 200 | **n0 = 1000** | n0 = 5000 | n0 = 1000, centre from the n trials only |
|---|---|---|---|---|---|
| 20 | 9.5 | 46.2 | **50.8** | 49.3 | 40.3 |
| 50 | 22.5 | 48.7 | **53.7** | 50.8 | 43.4 |
| 200 | 46.9 | 56.5 | **59.3** | 54.8 | 50.4 |
| 1,000 | 64.6 | 66.6 | **66.8** | 61.6 | 59.1 |
| 4,000 | 71.7 | 71.9 | 71.2 | 68.0 | 63.9 |
| 16,540 | 74.2 | 74.4 | 74.0 | 72.0 | 66.9 |

- One prior (n0 = 1000) is safe everywhere: never below re-centring, +5 with 50 labelled trials, +10.5 with 200, +18 with
  1,000, 74 with all trials (seen-subject level under re-centring: 81).
- The centre must still come from trials at the query's repetition count (last column: 7-8 points lower when it comes
  from the 4-repetition labelled trials), as in K15.
- `1 - tr(R)/d` is 0.77 for a new subject and 0.54 for a seen one whose accuracy barely moves: this number is dominated
  by directions the fit does not determine; not a usable measure of "how much the cloud is turned".

## K29 key fine-tuning (14:10) — `k29_key_tuning.py`, ATM seed 3300, folds 1 3 5 7; frozen model, only the 63x250 key is trained
References on these folds: average key K 44.3 plain / 46.7 re-centred; own key 35.9 / 50.0.

| calibration trials | labelled, plain | labelled, re-centred | unlabelled (image set known), plain | unlabelled, re-centred |
|---|---|---|---|---|
| 50 | 47.5 / 45.9 | 53.0 / 53.7 | 41.3 / 43.0 | 47.9 / 50.6 |
| 200 | 51.9 / 50.1 | 56.1 / 57.8 | 43.2 / 44.2 | 47.9 / 51.1 |
| 1,000 | 54.4 / 51.5 | 58.1 / 58.1 | 42.4 / 45.6 | 47.9 / 51.2 |
| 5,000 | 52.9 / 50.5 | 58.5 / 58.9 | | |
(init K / init own key)

- Labelled key tuning works from 50 trials and saturates near 58 (the K07 oracle): the key is a small lever.
- Unlabelled (one-to-one assignment to the known image set): pseudo-labels 50 % right at 50 trials, 9 % at 1,000;
  ends at re-centring alone (+1). NOT a lever as built.
- Against K33 at equal labels (re-centred): 50 trials 53.7 vs 53.7; 200: 57.8 vs 59.3; 1,000: 58.1 vs 66.8. The output
  rotation is closed-form, never worse, and keeps growing. (Folds differ: 4 here, 10 in K33.)

## Three distilled heads on a TEST-selected trunk (14:10) — `TRUNK=atm128/seed3300 k21_heads.py`, 10 folds
Single head re-centred 50.1; three heads 53.3 (mean rank 2.56), without re-centring 36.95. The validation-selected trunk
gave 53.65 / 36.0: test selection of the trunk adds nothing once the heads are on.

## K32 on other encoders (14:25) — `k32_centre_vs_geometry.py tsconv_vc eegproject_vc labram_vc`, seed 3300, 10 folds
| encoder | new: plain -> re-centred -> + oracle rotation | seen, own key: re-centred -> + rotation | CKA new / seen | key swap constant share, new / seen |
|---|---|---|---|---|
| ATM | 33.4 -> 48.9 -> 71.9 | 81.0 -> 83.7 | 0.73 / 0.75 | 70 % / 38 % |
| TSConv | 32.4 -> 37.0 -> 62.3 | 71.2 -> 73.7 | 0.70 / 0.71 | 73 % / 45 % |
| EEGProject | 22.1 -> 35.6 -> 50.5 | 67.9 -> 67.6 | 0.71 / 0.75 | 79 % / 55 % |
| LaBraM | 25.7 -> 40.3 -> 51.8 | 77.1 -> 77.1 | 0.69 / 0.74 | 47 % / 22 % |

The picture holds in all four families (convolutional, MLP, pretrained transformer, attention): same shape, wrong
centre, wrong orientation; a labelled rotation adds +11 to +25 for a new subject and 0 to +2.7 for a seen one. After it,
own key and substituted key score the same everywhere. TSConv gains little from the centre (+4.6) and most from the
rotation (+25): its subject effect is almost all orientation.

## K34 few-shot rotation on the three-head readout (14:25) — `k34_heads_rotation.py`, trunk seeds 3300 / 3301, 10 folds
One rotation per head (n0 = 1000), on top of re-centring.

| labelled trials | single head | three heads |
|---|---|---|
| 0 | 48.6 / 48.1 | 53.65 / 51.3 |
| 50 | 54.3 / 52.6 | 58.3 / 57.3 |
| 200 | 60.5 / 59.6 | 63.6 / 63.1 |
| 1,000 | 67.6 / 66.4 | 70.4 / 69.8 |
| 4,000 | 73.0 / 71.0 | 75.4 / 74.2 |
| 16,540 | 75.6 / 75.0 | 77.4 / 76.5 |

The heads and the rotation stack: +2 to +5 from the heads at every label count. Few-shot standing: 200 labelled
trials 63.3, 1,000 trials 70.1, all trials 77 (a seen subject, single head, re-centred: 81).

## User steer (14:35) and K35 re-centring + GALA (15:00) — `k35_recentre_gala.py` — NO GAIN
User: "the rotation stuff is not very interesting, what edge does it have over GALA?" (none as a method: labels needed,
below label-free GALA) -> rotation (K32-K34) and GALA set aside. One question kept: "does re-centring (with shrunk shift)
combine well with GALA to give it an even greater superiority over SCORE?" After that: only fully inductive methods, or
methods needing a small amount of unlabelled data.

Paper's causal stream protocol (CVPR_paper/scripts/inc_causal.py: query t decoded by the calibration fitted on the
queries before it; K = 200, q = 20; ATM seeds 3300-3302; 10 subjects; 5 stream orders; GALA = `gala_sw_freq1`, SCORE with
ATM, both with their stream-selected settings). Re-centring step from the past queries, shrinkage n0 = 5, image anchor
from the training images. Top-1 at 1 / 5 / 20 minutes / end of stream:

| | uniform | skewed | skewed, a quarter never queried |
|---|---|---|---|
| zero-shot | 29.2 / 27.4 / 28.3 / 27.6 | 32.3 / 26.6 / 29.1 / 28.2 | 27.8 / 28.5 / 28.2 / 29.4 |
| re-centring alone | 35.2 / 38.2 / 39.3 / 39.2 | 36.9 / 36.2 / 40.2 / 40.1 | 36.0 / 39.1 / 38.3 / 37.8 |
| GALA | 35.3 / 45.8 / 58.6 / 66.9 | 40.0 / 44.0 / 58.7 / 62.5 | 36.8 / 47.3 / 56.1 / 57.7 |
| re-centring, then GALA | 34.0 / 46.2 / 58.8 / 68.8 | 37.8 / 43.9 / 59.5 / 63.4 | 35.7 / 47.9 / 56.5 / 58.3 |
| SCORE with ATM | 29.0 / 41.8 / 51.3 / 53.1 | 29.8 / 41.3 / 52.1 / 53.1 | 28.0 / 44.0 / 50.0 / 50.4 |
| re-centring, then SCORE | 29.7 / 41.8 / 51.4 / 53.2 | 29.9 / 41.5 / 52.2 / 53.3 | 28.5 / 44.1 / 50.0 / 50.5 |

- GALA reproduced (paper, 10 orders: 37.2 / 46.2 / 59.1 / 67.3 uniform). Normalising the queries first changes nothing.
- The combination is a wash: -1 to -2 in the first minute, +0.2 to +0.8 at 5 and 20 minutes, +0.6 to +1.9 at the end.
  The lead over SCORE does not move notably. SCORE is unaffected (it matches means itself).
- Re-centring alone gives +10 over zero-shot from the first minute and equals GALA's first minute, then GALA leaves it
  behind (39 against 59 at 20 minutes). GALA already centres (whitening) and corrects the gallery-side bias (CSLS):
  the shift brings it nothing new. Closed.

## User steer (16:25) and K36-K39 the in-trial key (17:45) — PURE INDUCTIVE RESULT
User: drop per-head distillation (`hydraft_atm_157`) and the larger ATM (`msL_atm_157`), both removed from the queue;
"prioritize an architecture that uses our insights regarding how encoders naturally use mean signal to encode
subject-specific info ... a built-in induction bias that allows better zero-shot decoding on unseen subjects, in the pure
inductive setting."

**Observation.** The key m_s is the subject's steady-state response to the 5 Hz image stream. Keys: lag-50-sample
(200 ms) autocorrelation .96, and 94 % of |m_s - K|^2 is 200 ms-periodic. Image-specific content c: lag-50
autocorrelation .07. Energies are equal (key .0195, content of an 80-repetition trial .0191, key deviation from K .0083).
So one trial x = m_s + c carries its own key: k(x) = K + P_ac (x - K), P_ac = mean over the five 200 ms periods of the
window, minus its temporal mean (comb filter; no parameter, nothing from the subject but the trial).

**K36** `k36_self_key.py`, frozen ATM-128 validation-selected, 3 seeds x 10 folds, attention map of the raw trial,
readout fed x - k + K:

| key estimate | seed 3300 | 3301 | 3302 | mean | share of the key deviation explained |
|---|---|---|---|---|---|
| none (plain model) | 33.4 | 31.35 | 31.85 | 32.2 | 0 |
| **in-trial, comb without the temporal mean** | **40.9** | **38.25** | **39.95** | **39.7** | .52 |
| in-trial, comb with the temporal mean | 40.05 | 38.25 | 39.35 | 39.2 | -.15 |
| in-trial, half strength | 39.85 | 36.8 | 37.9 | 38.2 | .48 |
| in-trial, Wiener filter over time fitted on the training subjects | 40.1 | 39.2 | 40.0 | 39.8 | .66 |
| subject's mean test ERP (test data required) | 41.9 | 40.85 | 42.35 | 41.7 | 1 |

The comb recovers 7.5 of the 9.5 points of the key substitution with no other data of the subject. A better key
estimate (Wiener) does not decode better: the comb is kept (no parameter).

**K37** `k37_self_key_heads.py`, same trunks + the three distilled heads of K21, no re-centring:

| readout | plain | in-trial key | subject's mean ERP (test data) |
|---|---|---|---|
| model's own head | 32.2 | 39.7 | 41.7 |
| three distilled heads | 35.5 | **43.9** (44.4 / 43.25 / 43.95) | 46.0 |

Per subject (seeds averaged): in-trial key +7.5 on the own head (9 of 10 subjects, p = .004, worst -2.8 on subject 5,
where the oracle key loses too) and +8.4 on three heads (p = .005); three heads over one +4.2 (p = .003).
**Pure inductive: 32.2 -> 43.9 with one frozen ATM; published 35.3.**

**K38** `k38_self_key_encoders.py`, other encoders, seed 3300, input x - lam (k - K):

| encoder | plain | in-trial lam .5 | in-trial lam 1 | oracle lam .5 | oracle lam 1 |
|---|---|---|---|---|---|
| TSConv | 32.35 | 34.05 | 34.15 | 35.2 | 35.6 |
| EEGProject | 22.1 | 26.05 | 29.1 | 27.8 | 31.25 |
| LaBraM | 25.65 | 31.95 | 29.75 | 34.0 | 32.05 |
| Conformer | 23.4 | 28.3 | 28.55 | 28.5 | 29.45 |
| ATM (no attention split) | 33.4 | 39.7 | 38.7 | 42.05 | 40.1 |

Every encoder gains (+1.8 to +7).

**K39** `k39_self_centre.py`: the in-trial key does not give the output centre. e(x) - e(k(x)) + anchor: 34.6 (plain
32.2); even with the true mean ERP, e(x) - e(m_s) + anchor = 37.4 against 48.6 for the re-centring: the mean embedding
is not the embedding of the mean. In-trial key then re-centring = 45.3, below re-centring alone (48.6): as with the
oracle key (K10), the two do not add. The test-data-required best is unchanged.

**Code.** `train.py --subject_key comb` (every trial replaced by x - P_ac x, train and test) and `comb_split` (ATM: the
attention map from the raw trial, the readout sees x - P_ac x). No subject mean is computed in these modes.
Queue, folds 1 5 7, ahead of the demean runs: `atm_comb_157`, `tsconv_comb_157`, `ep_comb_157`, `ms_atm_comb_157`.

Caveats: the period (50 samples) is the paradigm's stimulus onset asynchrony; a paradigm without a periodic stream has
no such handle. Queries are 80-repetition averages (not tested at fewer repetitions yet).

## K40 the in-trial key at fewer repetitions per query (18:05) — `k40_self_key_reps.py`, trunk seed 3300, 10 folds
Pure inductive, no re-centring; one head -> three heads.

| repetitions | plain | in-trial key |
|---|---|---|
| 1 | 6.75 -> 6.95 | 5.35 -> 5.85 |
| 4 | 16.1 -> 16.5 | 15.65 -> 17.05 |
| 16 | 27.15 -> 28.1 | 30.75 -> 34.1 |
| 80 | 33.4 -> 36.0 | 40.95 -> 44.4 |

The in-trial key needs averaged queries: +8.4 at 80 repetitions, +6.0 at 16, +0.5 at 4, -1.1 on single trials (the
periodic part of a noisy trial is mostly noise). Same limit as re-centring and the three heads (K31).

## User (19:25): "keep going in this direction"; bonus: a method that does not rely on the paradigm; bonus: any pure inductive method at 45 %
Queue changes: `ms_atm_rest` stops after fold 8 (placeholders `_skip-sub-09/10`, folds 9 10 requeued as `ms_atm_rest2`);
new arms `tsconv_swap_157`, `atm_swap_157` (`--subject_key swap | swap_split`: in training the readout sees a random
training subject's key in place of the trial's own; raw trial at test; needs no paradigm knowledge). Follow-up GPU job
78371 (`key_gpu`, starts when 77581 ends on 19 Nov 17:21, same node, same queue).

## K42 in-trial key without the period (19:40) — `k42_key_no_paradigm.py`, frozen ATM seed 3300, 10 folds
Estimators fitted on the training subjects only; readout fed x - k + K.

| key estimate | top-1 | share of the key deviation explained |
|---|---|---|
| none (plain) | 33.4 | 0 |
| comb (hard-coded 200 ms period) | 40.9 | .52 |
| Wiener over time, free (finds the period by itself) | 40.1 | .66 |
| Wiener over time, lags below 100 ms only (cannot use the period) | 31.05 | .38 |
| Wiener over the electrodes | 28.15 | .40 |
| electrodes + local time | 34.05 | .20 |
| projection on the 9 training key deviations | 33.5 | .19 |
| subject's mean test ERP (test data required) | 41.9 | 1 |

- The period need not be hard-coded: a filter learned from the training subjects' keys and trials finds it (40.1).
- Without the periodic repetition nothing works: local time or spatial statistics explain 20-40 % of the key deviation
  but remove as much content, and decode below the plain model. In one trial the key and the content separate only
  because the key repeats every 200 ms and the content does not. **A single trial of a one-image-per-trial paradigm
  cannot give its key**; there the only pure inductive route is a readout trained not to depend on the key (swap arms).

## K43 models trained WITHOUT the key, scored without the subject's mean (19:40) — `k43_keyfree_strict.py`
`results/things_eeg/mi/train/{demean, ep_demean}` (10 Nov, `--subject_key demean`, 10 folds, test-selected epoch as their
baselines), input x - k:

| model (baseline, keyed, plain) | k = subject's mean test ERP (as trained) | k = in-trial comb key (pure inductive) | k = K (pure inductive, no paradigm) |
|---|---|---|---|
| TSConv (35.7) | 38.9 | **38.65** | 34.55 |
| EEGProject (27.5) | 35.9 | **34.55** | 18.5 |

A key-free model takes the in-trial key at almost no cost (-0.25 / -1.35): training with the comb built in should give
these numbers in the pure inductive setting. Left with the subject's key deviation in the input (k = K) it breaks.

## K44 the readout key chosen inside the trial (19:55) — `k44_self_mixture.py`, 3 trunk seeds x 10 folds, pure inductive
Readout fed x - k + [K + a (mix - K)], mix = convex combination of the 9 training keys fitted per trial to the trial's
200 ms-periodic part (largest weight .38 on average: a real mixture, not one subject).

| a | own head (3300 / 3301 / 3302) | mean | three heads | mean |
|---|---|---|---|---|
| 0 (K, K37) | 40.9 / 38.25 / 39.95 | 39.7 | 44.4 / 43.25 / 43.95 | 43.9 |
| 0.5 | 41.65 / 39.95 / 40.5 | 40.7 | 45.35 / 44.45 / 44.6 | 44.8 |
| 1 (in-trial mixture key) | 42.2 / 40.6 / 40.95 | 41.25 | 45.75 / 44.85 / 44.5 | **45.0** |

One head +1.55 (7 of 10 subjects, p = .053); three heads +1.15 (5 of 10 subjects, p = .21; subject 10 loses 3.3).
45.0 is reached on the 3-seed mean but the step from 43.9 is not significant on its own.

## User (19:55): CPU partition off, one 3080 node only; no new seeds for now, explore
The `Mee_Commun_CPU` jobs are cancelled (K41 on CPU among them); evaluations run inside `key_gpu` (`run.sh`, or a queue
line for the heavy ones). Exploration on seed 3300.

## K45 the directly trained three-head ATM with the in-trial key (20:05) — `k45_multispace_comb.py ms_atm`, folds 1-8
Pure inductive, no re-centring, validation-selected checkpoint (folds 9 10 not trained yet: `ms_atm_rest2`).

| | plain: main head / all heads | in-trial key: main head / all heads |
|---|---|---|
| ATM trained with three heads (`ms_atm`) | 31.6 / 33.4 | 40.6 / **43.1** |
| frozen ATM + three distilled heads, same folds (K37) | 32.4 / 34.3 | 40.0 / 42.9 (44.2 with the mixture key, K44) |

The directly trained model matches the distilled one under the in-trial key: no teacher is needed.

## K41 heads refitted in the key-swapped condition (20:05) — `k41_heads_refit.py 1,5,7` on the GPU queue — NO GAIN
Three heads, in-trial key at test, folds 1 / 5 / 7: heads fitted on own-key features 52.5 / 39.5 / 42.5 (44.8); fitted
on features under K 53.5 / 37.5 / 39.5 (43.5); fitted under the in-trial key 53.0 / 39.0 / 40.5 (44.2). Closed.

## K46 / K47 the output centre is a function of the key: a simulated, in-trial re-centring (20:15-20:35) — MAIN LEAD
`k46_synthetic_centre.py`, `k47_key_estimate.py`, frozen ATM seed 3300, own head, folds 1 5 7 (inside `key_gpu`, CPU).
Re-centring (K10) subtracts the subject's mean output embedding. K39: the embedding of the key alone is not that mean.
But the mean can be SIMULATED: o = mean_j e(k + c_j), the subject's key k carried by key-free contents c_j of the
training subjects; query scored as e(x) - o + anchor.

| centre | top-1 (folds 1 5 7) | cosine of (o - anchor) with the true centre |
|---|---|---|
| none (plain) | 35.8 | |
| true mean of the 200 queries (K10, test data required) | 49.7 | 1 |
| simulated, true key, contents = single 4-repetition training trials | 45.7 | .84 |
| simulated, true key, contents = 9-subject averages (36 repetitions) | **49.3** | **.98** |

**The offset that re-centring removes is determined by the subject's key** (given contents at the right noise level:
K15's repetition effect shows in the 4-repetition row). So an in-trial key gives an in-trial re-centring.
In-trial keys (`sk_lib.key_estimators`: comb; Wiener over time; Wiener with separable electrode x time covariances),
J = 32 contents per query:

| in-trial key | share of key deviation explained | key replaced in the readout (K36) | simulated centre |
|---|---|---|---|
| comb | .63 | 41.2 | 43.5 |
| Wiener, time | .75 | 41.7 | 44.3 |
| Wiener, electrodes x time | .75 | 41.8 | **45.3** |

Pure inductive, one head: 35.8 -> 45.3 on these folds. Here the quality of the key estimate matters (it did not for the
replacement), and the true key would give 49.3: the estimate is the limit.

## K48 the in-trial simulated re-centring with three heads, 10 folds (21:05) — `k48_centre_heads.py`, trunk seed 3300 — PURE INDUCTIVE 48.5
Frozen ATM + the three distilled heads; per query: in-trial key k, J = 64 simulated trials k + c_j, one centre per head.

| | one head (own) | three heads |
|---|---|---|
| plain | 33.4 | 36.0 |
| in-trial key replaced in the readout (K36 / K37) | 40.9 | 44.4 |
| **in-trial simulated centre, comb key** | **44.9** | **48.55** |
| in-trial simulated centre, Wiener time | 44.4 | 47.6 |
| in-trial simulated centre, Wiener electrodes x time | 45.35 | 48.4 |
| true centre of the 200 queries (K10, test data required) | 48.85 | 53.65 |

Per subject, three heads: simulated centre against plain +12.4, 10 of 10 subjects, smallest gain +3.5, p = .0005;
against the key replacement +4.0, 7 of 10, p = .048. The three estimators are equivalent on 10 folds (the lead of the
separable Wiener on folds 1 5 7 was noise): the parameter-free comb is enough.
Pure inductive, one frozen ATM (seed 3300): 33.4 -> 44.9 (one head) -> 48.5 (three heads); published 35.3.

## K49 variants of the in-trial simulated re-centring (20:57) — `k49_centre_variants.py 1,5,7`, three heads, comb key
Folds 1 5 7 (plain 38.7, K48 setting 50.0, true centre 55.5):
- **Number of simulated trials per query matters**: J = 8: 42.3, 16: 46.0, 32: 46.8, 64: 50.0, 128: 51.0 (still rising);
  the same 64 contents for every query 48.7.
- Strength (centre = o(K) + beta (o(k) - o(K))): beta 0: 39.7 (a fixed shift gives +1: the offset is the subject's),
  0.5: 47.0, 0.75: 49.3, 1: 50.0, 1.25: 49.3, 1.5: 42.0. The simulated offset has the right size as it is.
- Added to the key-swap score (44.8): 49.7. No gain.
- Fewer repetitions: 16: 29.2 -> 35.2; 4: 15.0 -> 13.8. As every lever so far, it needs averaged queries.

## K50 how many simulated trials (21:13) — `k50_centre_mc.py 1,5,7`, three heads, comb key
plain 38.7, true centre 55.5. All 400 contents per query 50.2; fresh J = 16 / 64 / 128: 46.0 / 50.0 / 51.0 (the rise at
128 in K49 was noise: the plateau is ~50); **paired estimator** o = mean_400 e(K + c) + mean_J [e(k + c_j) - e(K + c_j)]:
J = 16: 51.5, 64: 50.3, 128: 50.0. Sixteen simulated trials per query are enough once the content is paired out
(the 400-content term is computed once per model). The remaining gap to the true centre (5 points) is the key estimate.

## K51 the in-trial simulated re-centring on other models (21:25) — `k51_centre_any.py`, paired estimator J = 16, seed 3300
Pure inductive (centre) against plain and against the true centre (cal, test data required).

| model | folds | plain | in-trial simulated centre | true centre |
|---|---|---|---|---|
| ATM-128, own head | 10 | 33.4 | **45.5** | 48.85 |
| ATM trained with three heads (`ms_atm`), main head | 8 | 31.6 | 43.25 | 46.9 |
| ATM trained with three heads, all heads | 8 | 33.4 | **46.75** | 50.6 |
| frozen ATM + three distilled heads, same 8 folds (K48, J = 64) | 8 | 34.3 | 46.6 | 52.25 |
| TSConv | 10 | 32.35 | 34.15 | 37.0 |
| EEGProject | 10 | 22.1 | 31.75 | 35.55 |
| LaBraM | 10 | 25.65 | 36.2 | 40.3 |

- Every encoder gains, and keeps 40-75 % of what the true centre gives.
- The directly trained three-head ATM equals the distilled heads (46.75 against 46.6 on the same folds): the recipe is
  one ATM trained with three heads, no teacher, and a per-query centre computed from the query alone.

## K51 side checks (21:35) — one-off edits of `k51_centre_any.py`, ATM seed 3300, 10 folds, one head (centre 45.5)
- Simulated contents without their own periodic part (the in-trial key already carries the query's): 45.1. No change.
- No paradigm knowledge, the query itself as its key, k = K + b (x - K): b = .25: 30.85, .5: 23.6, 1: 4.6 (plain 33.4).
  The centre then contains the query's own signal and removes it. Confirms K42: without the period, no in-trial key.
Both options removed from the script.

## User (23:45): free the GPU node after the current experiment
`atm_comb_157` finishes its last fold (7) around 00:55 on 19 Nov; the next queue line is `scancel $SLURM_JOB_ID`, so
`key_gpu` (77581) releases the node by itself. The chained follow-up job 78371 is cancelled. The nine remaining queue
lines are parked at the end of `deferred.txt` (in order). The placeholders `train/ms_atm/_skip-sub-09`, `_skip-sub-10`
stay until `ms_atm_rest2` runs (its command moves them away first). Nothing runs after that until the user says so.
To resume: resubmit `key_gpu` (one rtx3080, Brain_GPU, body `worker.sh`) and re-add the parked lines with `q.sh`.

## User (00:50, 19 Nov): the two key-swap arms may run overnight on one 3080 node, then release the node

`key_gpu` resubmitted (78476, sl-tp-br-518). Queue: `tsconv_swap_157`, `atm_swap_157` (`--subject_key swap` /
`swap_split`, folds 1 5 7), `k51_swap` (plain and in-trial simulated re-centring of both arms,
`k51_centre_any_swap.csv`), then `release_gpu` (the job cancels itself). The seven other parked lines stay in
`deferred.txt`; nothing else is to be launched without the user's go-ahead.

### Built-in arm `atm_comb` (`--subject_key comb_split`, folds 1 5 7, finished 00:49)
Pure inductive, in-run metric (readout fed the trial minus its 200 ms-periodic part, attention from the raw trial).
| fold | val-selected epoch (66) | last epoch (86, early stop) | mean of epochs 61-80 |
|---|---|---|---|
| 1 | 47.5 | 45.0 | 45.3 |
| 5 | 36.5 | 42.0 | 39.8 |
| 7 | 30.0 | 36.5 | 35.3 |
| mean | 38.0 | 41.2 | 40.1 |
The epoch curve moves by +-3 points and the validation top-1 is flat (1.5 to 1.7 %), so the selected epoch is close to a
random draw; the 20-epoch mean (40.1) is the usable figure. Same folds, frozen keyed ATM: plain 35.8, comb key swap at
test time 43.5 (single val-selected checkpoints). Reading: training with the decomposition built in does not beat
applying it at test time on a keyed model; no reason to prefer it.

### Key-swap training arms (`tsconv_swap`, `atm_swap`, folds 1 5 7, finished 06:27; node released by the queue at 06:27)
Training: the readout sees the content plus a random training subject's key (`swap`; ATM `swap_split`, attention from
the raw trial). Test: the raw trial, nothing else (pure inductive, paradigm-independent). `k51_centre_any_swap.csv`.
Validation-selected checkpoints, top-1 on folds 1 / 5 / 7 and mean; "centre" = in-trial simulated re-centring (needs the
paradigm), "cal" = true re-centring (test data required).
| model | plain | centre | cal |
|---|---|---|---|
| ATM keyed (seed 3300) | 40.0 / 40.0 / 27.5 = 35.8 | 45.3 | 49.7 |
| ATM key-swap trained | 45.5 / 40.5 / 26.5 = 37.5 | 44.8 | 46.8 |
| TSConv keyed (seed 3300) | 42.5 / 26.0 / 32.5 = 33.7 | 34.2 | 36.5 |
| TSConv key-swap trained | 44.0 / 32.0 / 30.5 = 35.5 | 36.0 | 38.7 |
Mean of the in-run test top-1 over the last 40 % of the epochs: `atm_swap` 35.7, `tsconv_swap` 33.2 (keyed multi-head
ATM `ms_atm`, main head, same folds: 34.6; `atm_comb` 40.3).
Reading: +1.7 (ATM) and +1.8 (TSConv) on three folds, with single-epoch noise of +-3 per fold and mixed signs per fold:
not a gain one can claim. The swap-trained ATM still gains 7.3 from the simulated re-centring, so it has not become
key-independent. The paradigm-independent pure-inductive best stays the plain three-head readout (36.0, seed 3300).
State: no job of mine runs; seven parked lines remain in `deferred.txt`.

## K52 what the key does in the projection head (19 Nov) — `figures/head_data.py`, fold 1, seed 3300, validation-selected, seen subjects
Head = Linear -> h -> h + W2 GELU(h) -> LayerNorm. Content of seen subject a decoded with the key of seen subject b (72 ordered
pairs); "linearised" = GELU replaced by its tangent at the subject's mean head input mu (offset terms mu + W2 GELU(mu), gain
g = GELU'(mu) on the content c = h - mu).

| top-1 | TSConv | ATM-128 |
|---|---|---|
| own key | 68.4 | 75.6 |
| other subject's key (input) | 49.4 | 41.8 |
| other key only as a shift of h (mu_a -> mu_b) | 45.8 | 68.8 |
| linearised head, own operating point | 64.4 | 74.8 |
| linearised, other subject's offset, own gains | 63.5 | 74.7 |
| linearised, own offset, other subject's gains | 36.4 | 67.2 |

- TSConv: a wrong key changes h by a constant vector (r = .98 with mu_b - mu_a) and that shift alone gives the whole loss.
- In both models the damage goes through the gains: the key sets each unit's operating point on the GELU, hence which units
  pass the image signal; the readout of the content is I + W2 diag(g(key)). The offset part does not matter for seen subjects.
- TSConv: 122 of 1,024 units have a mean slope range > .6 across subjects; ATM 46 of 128. Example: TSConv unit 953, subject 3
  slope 1.0, subject 6 slope 0.0.
Figure: `figures/key_gates_head.png`.

## User (19 Nov afternoon): TSConv with a 128-wide head for the gating figure
No TSConv with `--eeg_backbone_dim 128` and kept weights exists (the April runs had their checkpoints deleted). One fold
trained: `tsconv_bb128` (fold 1, validation-concept recipe of `train_arm.sh tsconv`, head 1024 -> 128), single Slurm job
`key_bb128` on one rtx3080 that ends with the run (no worker, nothing queued after it).

### `tsconv_bb128` result (fold 1, finished 18:02, job ended by itself) and K52 on it
Held-out subject 1, validation-selected (epoch 26, early stop at 46): top-1 42.5, the same as the 1024-wide head (42.5).
K52 on seen subjects (72 pairs): own key 52.8 (1024-wide: 68.4); another subject's key 51.3 (49.4); shift at h only 51.9;
linearised 53.1; other offset 52.9; other gains 49.6. Units with gain < .2 for one subject and > .8 for another: 1 of 128
(1024-wide: 76 of 1,024); units whose gain ranges over > .6: 3 of 128 (122 of 1,024). Between-subject sd of unit means /
within-subject sd .21 (.41).
Reading (one fold, one seed): the narrow head builds no per-subject readout. Seen subjects lose 16 points and stop depending
on their key; the held-out subject is unchanged. The subject-specific gating of the wide head buys seen-subject accuracy
only. Figures: `figures/key_gates_head_bc.png` (1024), `figures/key_gates_head_bc_bb128.png` (128).
