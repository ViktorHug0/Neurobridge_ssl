# SUBJECT_LOG: how do unseen subjects differ, and how much is correctable? (2026-10-15 -)

User question (2026-10-15 15:30): derive from first principles a loss/architecture that accommodates subject
diversity, e.g. summarise a new subject by a few learned axes and correct for it on the fly. Agreed plan "A": three
diagnostics on frozen cv4 fold models before building anything (a 4th GPU allowed for this only; arch search keeps
the other 3). Framing: domain generalisation over a meta-distribution of subjects (Blanchard et al. 2021, marginal
transfer learning; ARM, Zhang et al. 2021); N = 25 training subjects -> few learnable subject parameters.

Prior evidence (ROUTING_LOG / ALIGN_LOG): second-order EEG statistics do not predict subject affinity (r 0.02), the
grand-average ERP does (0.25); output offset = 2% of the mean direction but worth +3 when removed (E5); subject-level
routing has no headroom (E3); training under the batch transform -3.5..-5.5 (E26); learned subject parameters fail
(FiLM -0.75, MMD -9.9, MixStyle, LEACE); dataset-level BN +4.3.

Script: neurips_challenge/subject_geometry.py (trial caches from trial_cache.py; corrections on unit rows inside the
top-k PC subspace of the DINOv2 training targets, after the deployed label-free centring P - 0.5 m_s).
D0 RSA (relational structure shared across subjects?), D1 oracle vs population corrections (offset / rotation /
affine): subject-specific headroom, D2 dimensionality of the corrections (LOO top-j axes), D3 label-free inference of
the subject coordinates (from all trials / from 6 trials).

## Runs
- 15:50 reve_d6_things + sqf_dbn_things, k = 16, 10 splits (CPU, node 516). No free GPU on the cluster for the
  TSC-bn64 trial cache yet (4th GPU dropped: user, 3 x 3080 max, interleave A with the arch search).

## Results (16:20), reve_d6_things + sqf_dbn_things, 20 held-out subjects, 10 A/B image splits
First run INVALID (offset/affine fits with intercepts injected the target mean: 35.9 -> 12). Rerun with linear-only
maps (both sides centred, targets scale-matched, prediction keeps its mean); identity reproduces the centred baseline.
D0 RSA of per-image mean outputs: subject-subject 0.49 REVE / 0.44 SQF (min 0.16 / 0.10), subject-DINOv2 0.11 / 0.11.
   Subjects share much more relational structure with each other than with the target geometry.
D1 top-5 on B (centred baseline REVE 35.92, SQF 34.39), oracle = fitted on the subject's own 100 A images (labels),
   population = same fit on the other 19 subjects' A images:
| k | map | REVE oracle / pop | SQF oracle / pop |
|---|---|---|---|
| 4 | rotation | 32.26 / 32.10 | 31.15 / 31.29 |
| 4 | linear l=3 | 34.62 / 34.61 | 33.41 / 32.98 |
| 8 | linear l=0.3 | 31.35 / 29.59 | 29.76 / 27.93 |
| 16 | rotation | 24.64 / 23.62 | 24.79 / 22.90 |
| 16 | linear l=0.3 | 30.91 / 28.15 | 29.49 / 26.62 |
| 16 | linear l=3 | 34.21 / 32.82 | 32.39 / 31.02 |
   No output-space linear correction beats doing nothing, even fitted with the subject's labels (100 images x 80
   repeats). Subject-specific part (oracle - pop) +0..+2.9, smaller than the estimation noise of the map.
D2 per-subject corrections are not low-dimensional (k=16: top-1/2/4/8 axes 15/26/43/68% REVE, 17/29/45/68% SQF);
   projecting a subject's correction on the others' top-j axes recovers nothing (flat in j).
D3 moot (nothing to infer).
Reading: at the output of models trained on 25 subjects, the only correctable subject effect is the first-order
offset already removed by label-free centring. SCORE's "shared structure, rotated coordinates" holds for the
relational structure (D0) but the rotation is not estimable/useful here. Open: input-level adaptation (fine-tune
the subject-facing layers on the subject's own TRAINING trials) = the intra-subject ceiling.

## D4 fine-tune oracle (17:25), TSC-bn64 fold 1: the intra-subject ceiling
Fold model (never saw the subject) fine-tuned 6 epochs (lr 1e-4, all parameters) on the held-out subject's OWN
training-image trials, scored on its test images (sealed rule, half the repeats):
| subject | base | ep 1..6 | last |
|---|---|---|---|
| 2 | 36.0 | 46.0 51.5 53.0 56.5 55.0 55.0 | 55.0 |
| 4 | 31.5 | 39.0 39.0 39.0 40.5 40.0 39.0 | 39.0 |
| 5 | 26.0 | 35.5 39.5 42.0 42.0 42.5 42.0 | 42.0 |
| 11 | 26.0 | 28.5 32.0 31.5 31.0 31.5 31.5 | 31.5 |
| 15 | 27.0 | ... | 39.0 |
| mean | 29.3 | | **41.3 (+12.0)** |
=> ~12 points of subject-specific headroom (labels on ~other images), and D1 shows it is NOT a linear output-space
correction: it lives inside the network. Next: which part carries it (ftor_tsc_bn / _spat / _temp / _head, queued).
- (18:40) ftor_tsc_spat (only the spatial conv trained; BN running stats adapt too): 29.3 -> 31.9 (+2.6;
  11: 29.0, 15: 29.5, 2: 40.0, 4: 33.5, 5: 27.5) vs all parameters +12.0. Subject-specific electrode mixing is ~1/5
  of the headroom. Pending: _bn (BN only), _head (head only), _temp (temporal filters only).
- (19:45) ftor_tsc_bn (only BN affine trained + running stats adapt): 29.3 -> 29.9 (+0.6; 11: 28.0, 15: 28.5,
  2: 35.0, 4: 32.5, 5: 25.5). Per-subject normalisation (gain/offset, AdaBN-like) carries ~nothing; spatial mixing
  +2.6; all parameters +12.0 -> the headroom is in the temporal filters and/or the head (ftor_tsc_temp / _head next).

## D4 fine-tune oracle, final parts (2026-10-16; TSC-bn64 fold 1; each subject's own training trials, 6 ep, lr 1e-4;
subject_agg top-5 of the last epoch, so absolute numbers differ slightly from the first table)
| fine-tuned part | mean | s2 / s4 / s5 / s11 / s15 | gain |
|---|---|---|---|
| none (fold model) | 31.2 | 37.0 / 36.0 / 26.5 / 29.5 / 27.0 | |
| BN only | 32.9 | 37.5 / 38.0 / 28.0 / 29.0 / 32.0 | +1.7 |
| spatial conv only | 35.8 | 42.0 / 37.5 / 32.0 / 31.5 / 36.0 | +4.6 |
| **head only** (flatten -> 64 -> 1536 readout) | **43.8** | 52.0 / 43.0 / 43.0 / 38.0 / 43.0 | **+12.6** |
| everything | 45.3 | 56.5 / 44.0 / 42.5 / 37.0 / 46.5 | +14.1 |
~90% of the per-subject headroom lives in the READOUT of the time-locked features, not in the spatial / temporal
filters or the normalisation. With D1 (no output-space map, even label-fitted, beats centring), the subject-specific
part is a person-specific mapping from (filter x time) features to the embedding: timing / latency / which features
carry the image, read differently per person. ftor_tsc_temp (temporal filters only) still queued.
Implication: the transferable lever would be a readout that adapts to the person (e.g. latency-tolerant or
fingerprint-conditioned readouts); subject FiLM (-0.75) and latency aligners (+0.03) were the closest tries so far.
- D4 temporal filters only (ftor_tsc_temp): 32.8 (39.5 / 39.0 / 30.0 / 21.5 / 34.0) = +1.6, and subject 11 drops to
  21.5. Final ranking of the per-person fine-tune headroom: head +12.6 >> spatial +4.6 > BN +1.7 ~ temporal +1.6;
  all +14.1. The temporal filters are shared across people; the readout is what is person-specific.

## D5 / D6 (2026-10-16, user): can the per-person readout correction be learned from unlabeled batches?
- D5 queued: LoRA head oracles (ftor_tsc_lora1 / 4 / 16; train.py --lora_rank, merged into the saved weights): is the
  +12.6 head headroom low-rank?
- D6 queued (feat_tf, feature_transform.py): the stateless batch transform applied INSIDE the TSC-bn64 head, on the
  1440-d time-locked features (L0) or the 64-d bottleneck before the nonlinear residual MLP + LayerNorm (L1), with /
  without the output router; vs the deployed output router. Tests whether low-rank batch structure lives upstream of
  the head (D1: output-space maps cannot beat centring).
- Next if either shows signal: per-person LoRA updates on all 20 subjects -> shared subspace? predictable from
  unlabeled feature statistics? -> amortised adapter (attention over the batch's features -> low-rank head correction).
- D6 result (feat_tf, TSC-bn64 cv4, grader_sim stream, 20 subjects; reference = output router 34.17, plain 30.05):
  every transform INSIDE the head hurts. L0 (1440-d features): centring .5 / 1 -> 23.90 / 14.97 (+ output router
  29.38 / 19.93), k2 25.20 (30.25), router 25.35 (31.40). L1 (64-d before GELU residual + LayerNorm): c.5 24.10
  (29.75), c1 12.95 (16.78), k2 27.75 (32.30), router 28.13 (32.48 = best, -1.70, 4/13).
  Reading: the head's nonlinearity was trained on UNCENTRED features; a batch shift moves every trial off its operating
  point. The output transform works only because it sits after the last linear layer. Stateless batch transforms
  cannot move upstream; a correction inside the head must be learned with the head (amortised adapter, D5 next).
- D5 first point: LoRA rank 4 (12,928 params on the 3 head Linear weights, lr 1e-3, 6 ep), subject 2: 38.0 vs fold
  model 37.0 vs head-only fine-tune 52.0 (train loss 6.465 vs 6.414). Caveat: the head-only oracle also trained every
  head BIAS, the LayerNorm and the 40x40 1x1 filter-mixing conv; LoRA adapts only the 3 weight matrices. Queued two
  separating oracles: ftor_tsc_hbias (head biases + LayerNorm only = per-person SHIFTS, "rank 0") and ftor_tsc_hconv
  (the 1x1 filter-mixing conv only), lr 1e-3.
- D5 LoRA rank 4 final: 37.1 (38.0 / 39.5 / 39.0 / 30.0 / 39.0) = +5.9 vs fold model, i.e. 47% of the head-only +12.6 with 13k vs 198k parameters (subject 2 was the outlier: +1.0). A substantial part of the per-person readout correction is LOW-RANK. hbias / hconv / rank 1 / rank 16 next.
- D5 hconv final: the 40x40 1x1 filter-mixing conv ALONE (1,640 params, lr 1e-3) = 38.5 (48.0 / 41.0 / 35.5 / 31.5 / 36.5) = +7.3 = 58% of the head's +12.6; head biases + LayerNorm alone 34.8 (+3.6, 29%); LoRA r4 on the 3 head Linear weights +5.9 (47%). A per-person LINEAR RE-MIXING OF THE 40 FILTERS (same at every time step, before the time-locked readout) is the most compact carrier of the per-person headroom found so far. Next: structure of the 5 learned maps (rank of dM, shared vs person-specific, covariance-recolouring?).
- D5 structure of the 5 learned 40x40 maps (hconv): |dM| = 20-59% of |M0|; top 1 / 4 / 8 directions hold 19-33% /
  55-80% / 78-93% of each dM's energy (moderately low-rank); cosine between people's dM 0-0.31, shared part 29% of the
  energy (chance with 5 people ~20%): mostly PERSON-SPECIFIC, a small common (train -> test) component.
- D7 queued (feat_coral, feature_coral.py): label-free recolouring of the 40 filter outputs at the 1x1 conv input,
  x' = mu_pop + C_pop^1/2 C^-1/2 (x - mu), population stats from training-image trials of the fold's training people,
  (mu, C) per person (oracle grouping, no labels) or per 64-trial batch (deployable), full / diagonal, alpha 1 / .5,
  with / without the output router. Maps TO the population operating point (D6 lesson), not to zero.
- D5 LoRA rank 1: 31.7 (+0.5; 33.0 / 35.0 / 29.0 / 23.5 / 38.0, s11 -6) vs rank 4 +5.9: one direction is not enough; the low-rank part needs a few (2-4+) directions per person, consistent with dM's top-4 energy 55-80%.
- D7 result (feat_coral, TSC-bn64 cv4, 20 subjects): label-free recolouring of the 40 filter outputs to the training
  population's mean/covariance is NULL. Without the output router: per-person stats (oracle grouping) full / diag
  +0.30 / +0.10 (a .5: +0.23 / +0.23), per-batch -0.35 / -0.32 (a .5: -0.10 / +0.13) vs plain 30.05. With the router
  (34.17): per-person full a .5 +0.45 (7/8), everything else -0.55 .. +0.17. So the per-person 40x40 map that buys
  +7.3 (D5 hconv) is NOT a second-order statistics (covariance) alignment of the filter outputs: it depends on how the
  filters relate to the IMAGE targets, which unlabeled statistics do not reveal.
- D5 LoRA rank 16: 43.1 (49.5 / 40.5 / 45.5 / 36.5 / 43.5) = +11.9 = 94% of the head-only +12.6. Rank curve 1 / 4 / 16 = +0.5 / +5.9 / +11.9: the per-person readout correction is low-rank (~16 of 64), but needs more than a handful of directions.

## D8 (2026-10-17, user): few-shot labelled oracle - how much of the head headroom appears with little data?
Inside the sealed predict() each person has ~6 trials per stateless 64-trial call, with no labels. Upper bound for any
within-batch test-time adaptation: LoRA rank 16 on the TSC-bn64 head (the +11.9 setting of D5) fine-tuned on only K
LABELLED training-image trials of the person, K = 6 / 64 / 512, same ~200 optimizer steps each (batch 6 / 64 / 256,
200 / 200 / 100 epochs), scored at 4 checkpoints. If K = 6 labelled gives ~0, unlabeled TTA within a batch cannot work.
train.py --train_max_trials K (seeded subset); ft_oracle_parts.sh ftor_tsc_fs<K>. Fold 1, 5 subjects, base 31.2.
- D8 first attempt INVALID: with BatchNorm in train mode, the running statistics re-estimate on the K trials and wreck
  the network (subject 2: K = 6 / 64 / 512 -> 19.5 / 7.5 / 20 vs fold model 37.0, already 26 after 50 steps at K = 512).
  Rerun with --freeze_bn_stats (BN keeps the fold model's statistics; only the LoRA adapters learn).
- D8 result (frozen BN, subject 2 only; the other subjects were stopped by the user): full-repetition top-5 after
  adaptation K = 6 / 64 / 512 labelled trials -> 23.0 / 9.5 / 23.5 vs fold model 37.0 (all-trials LoRA16: 49.5).
  Few-shot head adaptation HURTS (-13 to -27); the curves fall within 50 steps (K = 512: 27.0 -> 22.5 by epoch 50)
  while the full-data run gains from the first ~64 steps. The +12 head headroom needs thousands of labelled trials of
  the person; within a stateless 64-trial batch (~6 UNLABELED trials per person) per-person head adaptation is a dead
  end. Line CLOSED by the user (2026-10-17); a gentle low-lr / few-step variant was not run (expected <= +1).

## D9 (2026-10-19, user): do the per-person filter maps share a low-dimensional subspace with more people?
D5 had 5 maps (cv4 fold 1): moderately low-rank, mostly person-specific (shared energy 29% vs chance ~20%). Maps are
only comparable within ONE base model (each model has its own filter basis), so pooling cv4 folds would give 4 x 5,
not 20. Instead: the s11 split models (tsc_dbn40_things, 9 training people + THINGS), 11 unseen people per model,
22 in all. ftor_s11_hconv (ft_oracle_parts.sh): only the 1x1 40x40 `projection` conv, own training trials, 6 ep,
lr 1e-3 (= D5 hconv). Analysis: filter_maps.py (gain, |dM|/|M0|, rank, shared share vs chance 1/11, pairwise
cosines, leave-one-out share of a person's dM inside the top-j PCs of the other 10, null = the other split's
people in an unrelated filter basis). Started after trial_norm (node 516).
- D9 result (filter_maps.py, 2 x 11 people):
| | split 1 | split 2 |
|---|---|---|
| top-5 gain of the per-person map | -0.4 (6/11) | +2.4 (10/11) |
| abs(dM)/abs(M0) | 0.43 | 0.39 |
| energy in top 1 / 4 / 8 directions of dM | 30 / 68 / 86% | 30 / 73 / 91% |
| shared energy abs(mean dM)^2 / mean abs(dM)^2 (chance 1/11 = 0.09) | 0.32 | 0.25 |
| pairwise cosine mean / max | 0.29 / 0.61 | 0.21 / 0.45 |
| LOO: share of a person's dM in the others' top 1 / 2 / 4 / 8 PCs (null: other split) | 22 / 25 / 29 / 35% (2 / 6 / 8 / 12) | 13 / 19 / 22 / 31% (2 / 4 / 8 / 11) |
  Reading: with 11 people per basis there IS a shared component (2.8-3.5x chance), but it is essentially ONE
  direction (the mean dM = a common train -> unseen-person shift): the others' first PC carries 13-22% of a new
  person's map, further PCs add ~2-4% each, barely above the null. ~65-75% of each map stays person-specific. No
  low-dimensional person subspace emerges with more people. Also the per-person map gain is small on this base
  (tsc_dbn40, no bottleneck head: -0.4 / +2.4 vs +7.3 on TSC-bn64 in D5): the filter-mixing headroom is
  model-dependent. Line closed; the shared direction is a static (labelled) correction, not TTA.
