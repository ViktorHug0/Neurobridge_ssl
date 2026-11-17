# Mechanistic-interpretability experiments — running log

Proposal: https://claude.ai/artifact/7YmZjVJQXxQLMiNgnZb1Ud. Code `scripts/mi/`, outputs `results/things_eeg/mi/`,
logs `logs/mi/`. Run with `scripts/mi/run.sh eXX.py` (a step inside the `mi_gpu` Slurm job).
Models: TSConv k30 fd512 LOSO twins, seed 3300, `subjectmix_rebuttal/{method_subjectmix,base_no_aug}` = `mix` / `nomix`.
All checkpoints are test-selected (best epoch on the held-out subject): fine for diagnostics, not for claims of a method.

## E00 reproduction + seen-subject ceiling (2026-11-10)
Staged forward (`mi_lib.forward_from`) reproduces `best top1 acc` on all 20 checkpoints exactly.
Seen-subject top-1 on the 200 test images (new images, known subject), same checkpoints:

| run | held-out | seen subjects | gap |
|---|---|---|---|
| mix | 34.8 | 66.8 | 32.0 |
| nomix | 30.6 | 59.0 | 28.4 |

SubjectMix lifts seen subjects on new images by +7.8 and the held-out subject by +4.2: it is mostly a
new-image regulariser; the subject gap does not shrink.

## E01 variance partition with split-half reliability (A1)
Per unit: rel (same subject, two halves of the 80 reps), shared (different subjects), inter = rel - shared.
shared/rel, mean over 10 folds (mix; nomix within .015 everywhere):

| stage | rel/var | shared/rel all | held-out | train |
|---|---|---|---|---|
| input | .08 | .30 | .31 | .30 |
| temporal | .20 | .30 | .31 | .30 |
| spatial | .59 | .42 | .42 | .42 |
| backbone | .61 | .42 | .42 | .42 |
| trunk | .83 | .61 | .59 | .62 |
| align | .81 | .55 | .51 | .56 |

- 70% of the reliable image-locked EEG variance is subject-specific at the input; the network brings it to 39% (trunk).
- The linear projector RAISES the interaction again (.61 -> .55 shared).
- Early layers treat the held-out subject exactly like a seen one; a (small) difference appears only at trunk/align.
- SubjectMix does not change the partition. Variance fractions do not explain the +4.2 / +7.8.
- Seen subjects also keep 44% subject-specific reliable variance at align, yet score 67: variance share is not the
  retrieval-relevant quantity. Next: accuracy-denominated interventions.

## E02 moment patching (A2a, label-free), mix, 10 folds — baseline 34.8
| stage | mean | mean+std | cov (whiten-colour) |
|---|---|---|---|
| input | 34.7 | 34.8 | 33.5 (= Euclidean alignment) |
| temporal | 35.8 | 37.6 | – |
| spatial | 35.6 | 36.2 | 36.4 |
| backbone | 37.0 | 37.4 | 34.6 |
| trunk | 38.7 | 39.5 | 37.6 |
| align | 38.7 | 39.5 | 37.2 |

First and second moments recover at most +4.7 of a 32-point gap. Normalisation statistics are not the origin.
Euclidean alignment of the input does not help (-1.3).

## E04 output geometry, mix (seen / held-out)
top1 66.8 / 34.8; own-mean centring 48.6 / 28.6 (HURTS both); oracle per-gallery-image bias fitted on the test labels
86.4 / 52.3; hub10 17% / 26%; cos to the correct image .588 / .547, to the others .381 / .385.
Even the best possible gallery prior leaves the held-out subject at 52: hubness is a minor part, the deficit is
pair-specific (correct-pair similarity drops, wrong-pair similarity does not move).

## E05 global time shift / gain of the input (frozen model)
- Latency is not the cause: the best global shift is 0 for 9/10 held-out subjects (oracle best shift 35.0 vs 34.8).
- The model is destroyed by small shifts: seen subjects 66.8 -> 49 at +-8 ms, 20 at +-16 ms, 8 at +-24 ms.
- Gain-insensitive: 0.7x-1.5x costs < 5 points.

## E06 mean ERP and frequency bands
x_s(i) = m_s (image-independent mean ERP of the subject) + c_s(i).

| input | held-out | seen |
|---|---|---|
| plain | 34.8 | 66.8 |
| c only (mean removed) | 25.0 | 40.5 |
| c + another training subject's mean | 34.3 | 50.1 |
| shift only m by 8 ms | 31.2 | 58.8 |
| shift only c by 8 ms | 28.3 | 57.7 |

**The image-independent mean ERP acts as a subject key.** A seen subject loses 16.7 points when its mean ERP is
replaced by another training subject's; the held-out subject is indifferent to the swap (34.3 vs 34.8).
Bands: accuracy needs 1-30 Hz (lp<12 Hz: 42 seen / 23 held; hp>4 Hz: 32 / 14); the held-out subject follows the same
profile proportionally, so there is no band in which transfer is spared.

## E03 trained low-rank stitching (ORACLE: held-out subject's labelled train-image trials), mix, folds 1-2 only
(baseline 42.0, seen ceiling 71.0; sweep stopped after 2 folds to free the GPU)

| stage \ rank | bias | 1 | 2 | 4 | 8 | 16 | full | 64 |
|---|---|---|---|---|---|---|---|---|
| input (63x63 electrode mix) | 45.8 | 48.3 | 52.3 | 54.0 | 55.2 | 54.2 | 55.0 | |
| temporal (electrode mix) | 46.2 | 47.5 | 47.3 | 51.7 | 50.5 | 54.8 | 54.5 | |
| spatial (40 filters mix) | 44.8 | 44.7 | 46.0 | 47.2 | 50.5 | 53.2 | 54.3 | |
| backbone (1400) | 50.2 | 58.0 | 57.2 | 61.5 | 68.8 | 78.2 | 81.5 | 82.3 |
| trunk (1024) | 44.2 | 47.7 | 55.2 | 62.7 | 71.3 | 76.0 | 79.3 | 82.7 |
| align (512) | 43.8 | 48.0 | 52.2 | 58.2 | 68.8 | 75.2 | 80.0 | 82.0 |

With labels, a rank-8 correction at any flat stage reaches the seen-subject level and full rank exceeds it (it is
within-subject training). Electrode re-mixing alone caps at ~55. This is supervised adaptation, not a method.

## E07 the subject key (main finding so far)
x_s(i) = m_s + c_s(i); "key" = m_s. Accuracy of data c_s under key m_t (mix / nomix):

| data | own key | another training key | held-out subject's key | mean of training keys | no key |
|---|---|---|---|---|---|
| seen subject | 66.8 / 59.0 | 51.4 / 44.6 | 53.0 | 58.6 | 40.5 |
| held-out subject | 34.8 / 30.6 | 33.4 / 29.7 | – | 36.8 | 25.0 |

- A seen subject needs ITS OWN mean ERP: any other key (a training subject's or the held-out one's) costs ~15 points.
- The held-out subject's own key is worth no more than a random training key (34.8 vs 33.4). There is no decoder
  behind it. The mean of the training keys is slightly better (+2.0, label-free); the best training key per fold,
  chosen with test labels, 39.6.
- **Localisation (C).** Seen subject under a wrong key, then restore the right key's MEAN ACTIVATION at one sub-stage:
  restored fully (66.6-66.8) at every sub-stage up to the first Linear of the head (fc1), not at all after the residual
  GELU block (52.2). The two ELU stages do not use the key: its effect is additive until fc1. The subject-conditional
  computation is the single block `proj_eeg[1]` = h + Linear(GELU(h)); the key is the 1024-d mean of h.
- Same mechanism and same size without SubjectMix (59.0 -> 44.6; restored up to fc1). SubjectMix lifts own-key and
  wrong-key accuracy alike: it does not create or remove the key mechanism.
- Linearisation f(m) + J(m)c reproduces only part (60.5 of 66.8): the model is not just a key-indexed linear decoder.
- Gap accounting (mix): 66.8 seen/own key -> 51.4 seen/wrong key (key lookup, ~15) -> 34.8 held-out (content path
  fitted to the seen subjects, ~17).

## E08 key tuning for the held-out subject (oracle, labels on its train images)
free 63x250 additive input pattern 44.6 (+9.8); softmax combination of the 9 training keys 38.1; own key + free
combination of key differences 35.8. The operating point alone is worth ~10 of the 32 points.

## E09 the subject key across 9 architectures (cvpr/loso, same recipe, 10 folds each) — 2026-11-10 02:20
Top-1; "other" = mean of two other training subjects' keys; "mean key" = average of the 9 training keys.

| arch | seen: own key | seen: other key | held-out: own key | held-out: mean key | gain |
|---|---|---|---|---|---|
| TSConv | 64.7 | 50.0 | 35.7 | 36.7 | +1.0 |
| Squeezeformer (sqf) | 69.7 | 59.4 | 36.5 | 39.2 | +2.7 |
| ATM (128) | 71.6 | 38.7 | 35.2 | 39.8 | +4.6 |
| EEGProject | 65.0 | 37.5 | 27.5 | 34.1 | +6.6 |
| EEGNet | 60.9 | 25.7 | 25.1 | 31.3 | +6.2 |
| EEGConformer | 58.1 | 33.9 | 26.9 | 30.4 | +3.5 |
| TokenREVE (rmix) | 68.6 | 22.7 | 31.0 | 36.6 | +5.6 |
| REVE-6 | 69.1 | 24.5 | 29.0 | 33.5 | +4.5 |
| LaBraM (pretrained) | 79.4 | 24.8 | 31.9 | 38.0 | +6.1 |

- Every architecture decodes a seen subject conditionally on that subject's mean ERP: a wrong key costs 10 (sqf) to
  55 (LaBraM) points. The two conv models are the least key-dependent AND the best on unseen subjects; the models
  with the highest seen-subject accuracy (LaBraM 79, ATM 72) are the most key-dependent.
- **Label-free test-time fix:** replacing the held-out subject's mean ERP by the mean training key helps all 9
  architectures, +1.0 to +6.6 (mean +4.5). Per-fold spread is large (sub-06: +7.5 to +22.5; sub-05: down for some).
- Code drift: the current train.py gives 50.5 on fold 1 for the paper recipe (= cvpr/loso/tsconv), the July run 42.0.
  Training arms are therefore compared to cvpr/loso/tsconv/seed3300 (35.7), not to subjectmix_rebuttal.

## E10 test-time key substitution, variants (held-out subject, label-free) — 02:30
key = m_u + lam (K - m_u). K from the 200 test images of the training subjects (Ktest) or from their TRAINING images
(Ktrain, deployable). 10 folds.

| arch | own | Ktrain .5 | Ktrain 1 | Ktrain 1.5 | key ensemble (9 keys) | best single training key (oracle) | n=5 / 20 / 50 trials for m_u |
|---|---|---|---|---|---|---|---|
| TSConv mix (rebuttal) | 34.8 | 36.9 | 37.1 | 34.4 | 36.9 | 39.5 | 28.8 / 33.9 / 36.3 |
| TSConv nomix | 30.6 | 32.2 | 31.5 | 29.7 | 32.7 | 34.1 | 24.5 / 28.7 / 30.6 |
| ATM-128 | 35.2 | 42.2 | 40.3 | 29.0 | 35.5 | 36.1 | 29.7 / 36.7 / 39.5 |
| EEGProject | 27.5 | 33.9 | 34.7 | 27.9 | 31.6 | 33.1 | 26.3 / 33.8 / 34.2 |

Ktrain = Ktest (no test data of other subjects needed). lam .5 is the safe setting (ATM 9/10 folds up). The key of the
new subject needs >= ~50 (80-rep-averaged) trials. Picking one training key by confidence, or ensembling keys: worse.
Stats over the 9 architectures (E09, lam 1): mean +4.5 per held-out subject, t-test over the 10 subjects p = 0.036;
largest for sub-06 (+15.9) and sub-01 (+9.6), negative for sub-05 (-2.8). Across architectures, key dependence
(seen own - seen other) correlates .71 with the substitution gain. reve6 has only 2 folds on disk: excluded.

## E11 — substitution vs transductive heads, key distance, key window — 03:05
| arch | head | own key | lam .5 | lam 1 |
|---|---|---|---|---|
| ATM-128 | plain / csls / sinkhorn / full_sattc | 35.2 / 48.4 / 63.4 / 78.2 | 42.2 / 51.9 / 61.9 / 75.4 | 40.3 / 47.2 / 55.6 / 69.2 |
| EEGProject | same | 27.5 / 37.7 / 47.1 / 59.7 | 33.9 / 42.9 / 47.5 / 60.0 | 34.7 / 41.9 / 45.9 / 58.0 |
| TSConv | same | 35.7 / 43.0 / 46.7 / 61.6 | 37.3 / 43.7 / 46.6 / 61.8 | 36.7 / 43.0 / 45.4 / 60.4 |
| sqf | same | 36.5 / 44.8 / 47.1 / 61.5 | 39.0 / 45.5 / 47.4 / 61.8 | 39.0 / 44.4 / 47.7 / 61.5 |

- **The gain is an inductive-regime gain.** Under full SAGE-TTA global substitution is neutral (TSConv, sqf,
  EEGProject) or harmful (ATM -2.8 / -9.0). TTA already repairs what substitution repairs.
- Distance of the held-out key to the training keys does not predict the per-fold gain (r -.34 to .30).
- Substituting the key in one time window only is harmful (inconsistent key); it has to be replaced whole.

## E12 — does TTA replace the key lookup? No: the key is what TTA calibrates — 03:15
plain cosine / full_sattc, 10 folds:

| arch | seen own | seen wrong key | held-out own | held-out wrong key | held-out no key |
|---|---|---|---|---|---|
| TSConv | 64.7 / 88.9 | 48.6 / 75.5 | 35.7 / 61.6 | 35.1 / 60.5 | 24.8 / 57.5 |
| sqf | 69.8 / 90.9 | 58.7 / 85.2 | 36.5 / 61.5 | 37.9 / 60.0 | 25.2 / 57.7 |
| ATM-128 | 71.4 / 98.0 | 36.1 / 73.3 | 35.2 / 78.2 | 30.3 / 60.6 | 12.6 / 62.1 |
| EEGProject | 65.7 / 94.2 | 37.5 / 67.8 | 27.5 / 59.7 | 27.2 / 51.6 | 22.2 / 48.9 |
| EEGNet | 61.2 / 82.8 | 25.0 / 48.8 | 25.1 / 51.0 | 23.2 / 43.3 | 9.4 / 44.9 |
| Conformer | 58.2 / 91.5 | 34.2 / 66.2 | 26.9 / 58.9 | 25.6 / 53.9 | 9.8 / 47.8 |
| TokenREVE | 68.4 / 96.1 | 23.2 / 54.3 | 31.0 / 65.8 | 15.8 / 40.5 | 16.5 / 33.9 |
| LaBraM | 79.4 / 98.7 | 24.7 / 53.3 | 31.9 / 69.0 | 21.6 / 45.5 | 9.6 / 43.0 |

- After TTA the held-out subject's OWN (never seen) key is worth +1.1 (TSConv), +1.5 (sqf), +5.0, +7.7, +8.1,
  **+17.6 (ATM), +23.5 (LaBraM), +25.3 (TokenREVE)** over a wrong key. The key-conditional decoder does extrapolate
  to a new subject: its output is informative but mis-calibrated, and TTA supplies the calibration.
- With the key neutralised, ATM + TTA (60.6) = TSConv + TTA (60.5) = sqf + TTA (60.0). **ATM's 78 under TTA is the
  key mechanism.** The encoder ranking under TTA is the ranking of how well the key use extrapolates.
- Plain cosine: for conv models own key = wrong key on a new subject; token models already prefer the own key.

## E13 — ATM: the electrode attention is a key-derived spatial filter — 03:22
- Attention maps are image-invariant: corr .96 between two images of one subject, .98 with the map of the key
  alone, .68 between subjects on the same image.
- One map per subject, computed from the mean ERP alone and used for all images: 72.0 / 97.9 seen, 35.9 / 78.7
  held-out = same as (slightly above) per-image attention. ATM's attention is a hypernetwork key -> 63x63 mixing.
- Attention patching, held-out subject, full_sattc: own 78.2, wrong key 60.6, wrong key + own attention 71.6,
  own key + wrong attention 67.7: ~60% of the key's value on an unseen subject sits in the attention map.
- **Dissociation (E13b, held-out, 10 folds; K = mean training key from training images):**

| readout key \ attention key | own | half | K |
|---|---|---|---|
| own — plain / TTA | 35.9 / 78.7 | 35.1 / 75.2 | 34.1 / 72.2 |
| half | 42.9 / 77.5 | 42.4 / 75.1 | 41.4 / 72.4 |
| K | 44.5 / 74.8 | 41.9 / 71.9 | 40.0 / 69.1 |

  The attention use of the key generalises to a new subject (keep the subject's own key there); the readout use does
  not (give it the canonical key).
- **Replication (E13c), plain cosine, label-free, no training:**

| checkpoints | own | K everywhere | K readout + own attention | half readout + own attention (plain / TTA vs own TTA) |
|---|---|---|---|---|
| atm128 seed 3300 | 35.2 | 40.3 | 44.5 | 42.9 / 77.5 vs 78.2 |
| atm128 seed 3301 | 36.2 | 39.5 | 44.0 | 43.8 / 76.3 vs 77.9 |
| atm128 seed 3302 | 36.0 | 40.5 | 42.8 | 44.6 / 77.7 vs 78.2 |
| atm (fd512) 3300 | 30.0 | 37.2 | 38.6 | 38.9 / 68.3 vs 66.9 |
| atm64 3300 | 34.5 | 40.2 | 42.9 | 43.0 / 75.6 vs 74.9 |

  ATM-128 inductive LOSO 35.8 -> 43.8 (3-seed mean), +8.0, with TTA unchanged (-0.4 with "half").
  Per subject (seed 3300): +17.5 +5.5 +14.0 +10.5 +1.0 +22.0 +12.5 +1.0 -0.5 +10.0.
  Caveats: test-selected checkpoints (same checkpoints in both arms); the new subject's key is the mean of its
  200 unlabelled test trials (80-rep averages).

## E13c on validation-selected checkpoints (atm128_vc: ValCon, 10% of concepts held out for selection) — 03:45
| seed | own (plain / TTA) | K readout + own attention (plain / TTA) | half readout + own attention (plain / TTA) |
|---|---|---|---|
| 3300 | 33.4 / 76.2 | 42.4 / 72.7 | 41.7 / 75.1 |
| 3301 | 31.3 / 73.6 | 40.9 / 70.1 | 39.8 / 73.7 |
| 3302 | 31.9 / 77.0 | 42.0 / 73.3 | 40.4 / 76.0 |

3-seed mean 32.2 -> 41.8 (+9.6) inductive; 28 of 30 subject x seed cells improve (sub-05 and sub-08 ~0).

## E14 few-repetition test trials — 03:44
Query = mean of the first r repetitions; key always from all test trials. Plain cosine, held-out.
ATM-128 ValCon s3300: r=1 6.7 -> 6.2 (targeted) ; r=4 16.1 -> 17.6 ; r=80 33.4 -> 42.4.
TSConv (half): 4.1 -> 4.2 ; 12.0 -> 12.7 ; 35.7 -> 37.3.  EEGProject (half): 4.5 -> 4.5 ; 10.2 -> 10.7 ; 27.5 -> 33.9.
The substitution repairs calibration, which only binds once the trial is clean. Little to gain at r ~ 3 (sealed regime).

## E16 inductive score ensembles with key substitution (seed 3300, test-selected checkpoints) — 04:00
Uniform mean of cosine score matrices. ATM: own-key attention + K readout; others: key at the midpoint to K.

| members | own keys | substituted |
|---|---|---|
| ATM-128 | 35.2 | 44.6 |
| ATM-128 + sqf | 41.4 | 47.4 |
| ATM-128 + TSConv | 41.5 | 46.1 |
| ATM-128 + TSConv + sqf | 43.9 | **49.0** |
| + EEGProject (4) | 42.0 | 47.9 |

## E17 ATM: key use along the layers (seen subjects, wrong key 38.7, own key 71.6) — 04:02
Restore the right key's mean activation at one sub-stage: embed 71.6 | after attention 62.5 | norm1 64.2 |
FFN+norm2 53.7 | conv ELU#1 50.9 | ELU#2 50.0 | fc1 50.0 | residual GELU block 40.2 | LayerNorm 41.2.
The 33-point key effect has three sites of about 10 points each: the attention map, the transformer FFN, and the
same residual GELU head block as in TSConv (plus ~4 in the conv ELUs).

## Training arm: EEGProject with the key removed (`train.py --subject_key demean`) — 04:20
x - m_s at train and test (test subject: mean of its own unlabelled test trials). Paper recipe, seed 3300, 50 ep,
`results/things_eeg/mi/train/ep_demean`. Baseline `cvpr/loso/eegproject/seed3300`. result.csv metric (best-loss epoch).

| | base | key-free training | test-time substitution (lam .5) |
|---|---|---|---|
| held-out, plain (10 folds) | 27.5 | **35.9** | 33.9 |
| held-out, full_sattc (9 folds) | 60.8 | 63.1 | 61.2 |
| seen subjects, plain | 65.0 | 55.6 | (55.4 with the mean key) |

- +8.4 inductive (paired p = 0.024 on the first 9 folds), TTA not hurt (+2.3, n.s.).
- The seen/unseen gap shrinks from 37.5 to 19.7: seen subjects lose exactly the key-lookup advantage (65.0 -> 55.6,
  the level they had under the mean key), the unseen subject gains. Key-free EEGProject (35.9) = TSConv (35.7).
- TSConv demean fold 1 only (43.0 vs 50.5 at the best-loss epoch, curve max 50.5 both): single fold, noise level.
  Not pursued tonight (TSConv is the least key-dependent encoder; a run takes the whole GPU).
- `best top1 acc` in result.csv is the top-1 at the lowest-test-loss epoch (noisy, +-4 on one fold).

## Training arm: ATM-128 split key (`--subject_key split`), started 04:20
New forward in `module/eeg_encoder/atm.py`: input (B, 2, C, T) = [content, key]; the attention map is computed from
the key alone and applied to the key-free content; the readout never sees the key. 50 epochs, folds 1 5 6 9 3 8,
`results/things_eeg/mi/train/atm128_split`. Compare with cvpr/loso/atm128 truncated at 50 epochs (constant LR).

## E18 / E19 key-free EEGProject, more views — 04:35
- Seen subjects in the key-free models: 55.6 (baseline 65.0 own key, 55.4 mean key); held-out 35.9.
- Low test SNR (mean of r reps, plain, held-out; baseline -> key-free): r=1 3.8 -> 4.0; r=4 9.9 -> 10.9;
  r=16 20.0 -> 25.1; r=80 27.5 -> 35.9.

## Across the 8 architectures (from E12) — 04:40
TTA accuracy on an unseen subject = content path (TTA under a wrong key) + value of the key.

| arch | key dependence (seen own - wrong, plain) | held-out TTA, wrong key | value of own key after TTA | held-out TTA |
|---|---|---|---|---|
| TSConv | 16.1 | 60.5 | +1.1 | 61.6 |
| sqf | 11.1 | 60.0 | +1.5 | 61.5 |
| ATM-128 | 35.3 | 60.6 | +17.6 | 78.2 |
| EEGProject | 28.1 | 51.6 | +8.2 | 59.7 |
| EEGNet | 36.1 | 43.3 | +7.7 | 51.0 |
| Conformer | 24.0 | 53.9 | +5.0 | 58.9 |
| TokenREVE | 45.2 | 40.5 | +25.3 | 65.8 |
| LaBraM | 54.7 | 45.5 | +23.5 | 69.0 |

corr(key dependence on seen subjects, value of the unseen subject's key after TTA) = 0.92; corr(key value, TTA gain)
= 0.76. Conv encoders: strong content path, little key use. Token encoders: weak content path, strong key use.
ATM has both (conv readout + key-derived attention), which is why it leads under TTA.

## E13d key of the new subject from disjoint unlabelled data — 04:55
ATM-128 ValCon, targeted substitution with m_u = mean ERP of the held-out subject's TRAINING-image trials (never its
test trials): seed 3300 33.4 -> 42.3 (42.4 with the test-trial key), TTA 73.6; seed 3301 31.3 -> 40.9 (40.9), TTA 70.8.
The method needs unlabelled EEG of the new subject under the same paradigm, not the test trials themselves.

## Training arm: ATM-128 split key — stopped after fold 1 (05:10)
Fold 1, 50 epochs: plain 39.5 at the best-loss epoch (baseline truncated at 50 epochs: 38.5), mean of the last 20
epochs 35.0 vs 31.9, full_sattc 83.5. The 100-epoch baseline checkpoint has 41.5 plain / 92.5 TTA and 59.0 with the
test-time targeted substitution. So on this fold a key-free readout is far below "keyed readout fed the mean key":
the keyed model seems to learn a cleaner shared code by factoring subject effects out through the key, and the
substitution then selects that shared code. One fold, and no 50-epoch baseline checkpoint for a TTA comparison:
inconclusive, stopped to give the GPU to the TSConv key-free arm (10 folds with matched baselines on disk).

## Midpoint key (lam .5, K from training images), all 8 architectures, 10 folds — 05:35
plain -> / full_sattc ->:
TSConv 35.7 -> 37.3 / 61.6 -> 61.8 | sqf 36.5 -> 39.0 / 61.5 -> 61.8 | ATM-128 35.2 -> 42.2 / 78.2 -> 75.4
(targeted 42.9 / 77.5) | EEGProject 27.5 -> 33.9 / 59.7 -> 60.0 | EEGNet 25.1 -> 31.9 / 51.0 -> 52.2 |
Conformer 26.9 -> 31.3 / 58.9 -> 59.7 | TokenREVE 31.0 -> 38.5 / 65.8 -> 65.9 | LaBraM 31.9 -> 40.0 / 69.0 -> 68.9.
Mean +5.4 inductive; TTA unchanged except ATM with the global version. The midpoint is the safe universal setting.

## Midpoint key (lam .5, K from training images), all 8 architectures, 10 folds — 05:35
plain -> / full_sattc ->:
TSConv 35.7 -> 37.3 / 61.6 -> 61.8 | sqf 36.5 -> 39.0 / 61.5 -> 61.8 | ATM-128 35.2 -> 42.2 / 78.2 -> 75.4
(targeted 42.9 / 77.5) | EEGProject 27.5 -> 33.9 / 59.7 -> 60.0 | EEGNet 25.1 -> 31.9 / 51.0 -> 52.2 |
Conformer 26.9 -> 31.3 / 58.9 -> 59.7 | TokenREVE 31.0 -> 38.5 / 65.8 -> 65.9 | LaBraM 31.9 -> 40.0 / 69.0 -> 68.9.
Mean +5.4 inductive; TTA unchanged except ATM with the global version. The midpoint is the safe universal setting.

## E20 mechanistically targeted TTA: self-train ONLY the subject's spatial filter — 05:25-06:15
Label-free, transductive (same regime as GALA R1: 200 unlabelled test trials + the 200-image gallery).
Loop: pseudo-labels = argmax of the transductive head on the current embeddings; 10 Adam steps of cross-entropy
(tau 20, plain cosine logits) on ONE small component, everything else frozen; 12 rounds. Hyper-parameters fixed a
priori (lr 0.05 for attention logits; two lrs shown for the others), no label-based stopping. Reported = head top-1.

| encoder / checkpoints | component adapted | head | before | after |
|---|---|---|---|---|
| ATM-128 s3300 (test-sel.) | attention map (4x64x64 logits, init = map of own key) | full_sattc | 78.7 | **88.3** |
| ATM-128 s3300 | attention map | GALA (sage_plus, cross-fold params) | 82.75 | **87.75** |
| ATM-128 s3300 | 63x63 input electrode re-mixing, lr .003 / .01 | full_sattc | 78.2 | 86.9 / 85.0 |
| ATM-128 s3300 | CONTROL: linear map on the output (128x128), lr .0003 / .001 / .003 | full_sattc | 78.7 | 82.6 / 82.9 / 81.7 |
| ATM-128 ValCon s3300 / 3301 / 3302 | attention map | full_sattc | 76.0 / 73.7 / 77.1 | 85.6 / 84.3 / 86.5 |
| TSConv s3300 | 63x63 input electrode re-mixing, lr .01 | full_sattc | 61.65 | **72.0** |
| TSConv s3300 | CONTROL: linear map on the output (512x512) | full_sattc | 61.65 | 65.5 |

- Adapting the subject-specific SPATIAL filter (attention map, or an input electrode re-mixing with only 4k
  parameters) gains +8.7 to +10.4; the same self-training on an output linear map gains +4. On top of the tuned GALA
  the attention version adds +5.0 (9/10 subjects up; sub-04 59.5 -> 67.5, sub-09 78 -> 90).
- Per subject, ATM s3300 att / full_sattc: 93->96.5, 76->80.5, 90->96, 51.5->75, 82->89, 81.5->88.5, 65.5->81,
  72.5->90, 81->88, 94->98.5. Converged by round ~9 (test-selected) / still rising at 12 (ValCon).
- The adapted encoder's PLAIN cosine accuracy reaches 66-71 (ATM) / 71 (TSConv): the calibration is absorbed into
  the encoder.
- Consistent with the mechanism: what differs between subjects is a spatial re-mixing that the encoder otherwise
  derives from the key; correcting it at the electrode level beats correcting the output.

## E20 continued: more encoders, seeds and the temporal control — 06:25
| encoder | component | before | after (full_sattc) |
|---|---|---|---|
| ATM-128 s3301 / s3302 | attention map | 77.35 / 78.75 | 87.45 / 88.3  (3-seed mean 78.3 -> 88.0) |
| sqf | 63x63 input re-mixing, lr .01 | 61.55 | 74.2 |
| EEGProject | 63x63 input re-mixing, lr .01 | 59.7 | 66.0 |
| TSConv | 63x63 input re-mixing, lr .003 | 61.65 | 73.85 |
| TSConv | CONTROL: per-electrode temporal FIR (63x21 taps), lr .003 | 61.65 | 70.5 (still rising) |
| ATM-128 s3300 | CONTROL: per-electrode temporal FIR | 78.2 | 86.4 |

**Correction to the reading:** a per-electrode temporal filter does almost as well as the electrode re-mixing
(70.5 vs 73.9 TSConv; 86.4 vs 86.9 ATM). The effect is therefore not specific to a SPATIAL operator: what matters is
adapting a small linear operator on the INPUT side (sensor space) rather than a linear map on the output (+4).
LaBraM run OOMed under the 2 GB cap (GPU shared with training), not rerun.

## E20 continued — 06:50
Input electrode re-mixing (lr .003), full_sattc: EEGNet 51.0 -> 60.2; Conformer 58.9 -> 70.85;
TSConv ValCon (tsconv_vc) 55.35 -> 68.35. Seven encoders so far: +6.3 (EEGProject) to +13.0.

## E20 ablations — 07:05
- Joint attention + input re-mixing (ATM s3300): 87.05 (attention alone 88.3). No stacking.
- Pseudo-labels from PLAIN cosine instead of the transductive head: ATM 78.7 -> 80.0, TSConv 61.65 -> 58.0.
  The head's assignment is what makes the self-training work; this is an add-on to GALA/SATTC, not a replacement.
- Fewer unlabelled test trials (first n images as queries, gallery of 200): n=100 ATM 60.6 -> 63.3, TSConv
  49.3 -> 53.5; n=50 ATM 48.0 -> 45.6, TSConv 40.0 -> 39.2. Needs >= ~100 trials; harmful at 50.
- ValCon ATM s3300 with 24 rounds: 76.0 -> 86.75 (plateau from round ~16).

## E20 in the paper's protocol (ValCon checkpoints, GALA head with cross-fold params) + sanity check — 07:40
| checkpoints | GALA | GALA + attention-map self-training (16 rounds) |
|---|---|---|
| atm128_vc seed 3300 | 82.15 | 88.45 |
| atm128_vc seed 3301 | 81.35 | 87.10 |

Shuffled-gallery sanity check (PERM=1: gallery order permuted, targets remapped; folds 4, 7, 2, ATM s3300, att,
full_sattc): 51.5 -> 75.0, 65.5 -> 81.0, 76.0 -> 80.5, identical to the unshuffled run. No step profits from the
query i <-> gallery i ordering.

## E20, paper protocol, more — 08:25
- atm128_vc seed 3302, GALA head: 81.55 -> 88.65. Three ValCon seeds: **81.7 -> 88.1** (paper's GALA: 81.8).
- tsconv_vc seed 3300, GALA head (cross-fold params) + 63x63 input re-mixing (lr .003): **63.9 -> 73.6**, 10/10
  subjects up (75->83, 70.5->76, 61->80, 45.5->47, 61.5->77, 73->85.5, 63.5->69.5, 41.5->56, 64->73, 83.5->89).

## Training arm: TSConv with the key removed (`--subject_key demean`), 10 folds — 09:10
`results/things_eeg/mi/train/demean` vs `cvpr/loso/tsconv/seed3300` (same recipe, seed, code).

| | baseline | key-free |
|---|---|---|
| held-out plain, top-1 at the best-loss epoch (result.csv) | 35.7 | **38.9** (9/10 folds up, paired p = 0.045) |
| held-out plain, mean of the last 20 epochs | 33.0 | 36.6 (p = 0.0075) |
| min test loss | 2.36 | 2.22 (lower in 10/10 folds) |
| held-out full_sattc | 61.65 | 61.9 (n.s.) |

Per fold (result.csv): 43.0 45.0 30.5 32.0 37.5 40.5 38.0 26.0 42.5 54.0 (baseline 50.5 43.0 27.0 29.5 29.5 36.0
32.5 25.0 36.0 48.0). Best single inductive TSConv so far; TTA unchanged, as predicted (TTA already calibrates).

## E20: TSConv test-selected, GALA head + input re-mixing — 09:00
tsconv/seed3300: 67.45 -> 75.7.
sqf/seed3300, GALA head + input re-mixing: 68.75 -> 75.9.

## End of night — 09:17
GPU job `mi_gpu` (71689) cancelled. Remaining battery items (GALA head on atm128 s3301/s3302, input re-mixing on
atm128_vc) not run. Code changes outside scripts/mi: `train.py --subject_key {keep,demean,swap,split}`,
`module/eeg_encoder/atm.py` (split-key forward, optional forced attention). Nothing committed.

### Summary of what holds
1. Mechanism: every encoder decodes a seen subject conditionally on its mean ERP (key); sites localised in TSConv
   (residual GELU head block) and ATM (attention map = key-derived spatial filter, FFN, same head block).
2. Unseen subject: the key use extrapolates but is mis-calibrated; TTA supplies the calibration; encoder ranking
   under TTA = content path + value of the key (r = .92 with key dependence).
3. Inductive, label-free: midpoint key +5.6 (8 encoders, all 10 subjects); ATM targeted 32.2 -> 41.8 (ValCon,
   3 seeds); 3-model ensemble 43.9 -> 49.0; key-free training TSConv 35.7 -> 38.9, EEGProject 27.5 -> 35.9.
4. Transductive, label-free: self-training a small input-side operator on the head's pseudo-labels: ATM ValCon under
   GALA 81.7 -> 88.1 (3 seeds); tsconv_vc under GALA 63.9 -> 73.6; sqf 68.75 -> 75.9; output-map control +4.

## A6 subject level (n = 10, descriptive only) — 09:40
`results/things_eeg/mi/a6_subject_level.csv`. Spearman over the 10 subjects: split-half reliability of the input
(noise ceiling) vs held-out plain accuracy -0.43, vs held-out TTA -0.31. sub-08 has the MOST reliable signal
(.138) and the least shared one (shared/rel .247), is among the hardest as a held-out subject (25.9 plain, 48.9 TTA)
and normal as a seen subject (70.5). The per-subject spread is not a data-quality limit. Nothing reaches
significance at n = 10; the self-training gain is larger where the TTA baseline is lower (rho -.44).

## A5 target side (E22): which images transfer — 10:00
Plain cosine, per test image, TSConv / ATM / sqf / EEGProject, 10 folds.
- Image difficulty is shared: hit rate when held out vs when seen correlates .65-.76 over the 200 images.
- Typical targets (high mean cosine to the other gallery targets) are harder in both settings (r about -.35 / -.40).
- held/seen ratio by THINGS category (n >= 5 test images): lowest for food-like items (dessert .38, fruit .40,
  food .41, vegetable .42, electronic device .41), highest for shaped man-made objects (furniture .59, toy .58,
  tool .57, kitchen tool .57, weapon .54, container .53); animals .51. Two images are never retrieved when held out
  (gorilla, scallop); 7 are retrieved >= 75% of the time (folder, hoodie, face_mask, eagle, blowtorch, tiara, cart).
- The loss is spread over all content (ratio .38-.62); no category or target subspace carries the transfer.
  A target-reweighting mitigation is not supported. Small category sizes (5-53 images, overlapping).

## A4 sparse dictionary with cross-subject consistency (E21), TSConv trunk, 10 folds — 09:50
TopK SAE (4096 latents, k 32) on the 9 training subjects' train-image trunk activations; FVU .19; accuracy through
the reconstruction 30.1 held-out / 56.4 seen (model 35.7 / 66.5), so the dictionary is lossy and only the
interventions with the SAE residual kept are interpretable.
Zero a set of latents (residual kept), top-1 held-out / seen:

| set removed | 10% | 25% | 50% |
|---|---|---|---|
| least consistent across training subjects | 35.4 / 66.8 | 35.1 / 67.1 | 34.4 / 64.6 |
| random | 35.1 / 65.0 | 31.8 / 59.9 | 22.2 / 47.2 |
| most consistent | 19.3 / 41.5 | 13.5 / 28.1 | 7.0 / 13.8 |

- Cross-subject consistency (computed on training subjects only) identifies the latents that carry retrieval: the
  inconsistent half is dispensable, for seen and unseen subjects alike.
- Removing inconsistent latents does NOT help the unseen subject (35.4 vs 35.7). The failure is not wrong
  subject-specific features firing; it is inside the features that are consistent among the training subjects.
  The consistency-gating mitigation of the proposal is negative.

## E20 with cross-fold selection of (learning rate, rounds) — 10:10
For each held-out subject, the config with the best mean over the other nine subjects (`e20_crossfold.py`).
- ATM-128 ValCon s3300, GALA head, attention map, lr in {.01, .02, .1} x rounds 0..16: last-round means 86.3 / 87.8 /
  88.75 (lr .05 from the earlier run: 88.45). **Cross-fold selected: 82.15 -> 89.0**, 8/10 subjects up
  (sub-04 60.5 -> 90.5; sub-05 88 -> 87; sub-06 unchanged). Every lr in the grid gains >= +4.
- TSConv ValCon s3300, full_sattc head, input re-mixing, lr in {.001, .003, .01, .03}: 68.2 / 68.45 / 66.35 / 60.8.
  **Cross-fold selected: 55.35 -> 68.4**, 10/10 subjects up.
- ATM-128 ValCon s3300, full_sattc head, attention map, lr in {.01, .02, .05, .1}: 81.7 / 84.2 / 86.35 / 86.7.
  **Cross-fold selected: 76.0 -> 86.25**, 10/10 subjects up.

## Open gallery (200 test images + 200 distractor training images), ATM-128 ValCon s3300 — 10:25
Attention-map self-training, 12 rounds: CSLS head 36.25 -> 40.7 (plain 26.55 -> 39.15); full_sattc head
52.35 -> 56.2. The gain shrinks to about +4 with distractors (it follows the quality of the head's pseudo-labels).
The 1000-distractor and TSConv runs were not finished.

## GPU released on the user's request — 10:27 (job 72069 cancelled)
Interrupted: key-free Squeezeformer (`sqf_demean`): fold 1 only, 47.0 vs 42.0 baseline (one fold, noise level);
open-gallery runs with 1000 distractors and for TSConv. Not started: key-free ATM / token encoders, streaming regime.
