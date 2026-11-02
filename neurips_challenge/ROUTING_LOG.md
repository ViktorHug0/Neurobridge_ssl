# Label-free affinity and sample routing: experiment log

## TL;DR (2026-10-12, sealed cohort = 11 new participants)
- **Deploy (new):** `score_track1.batch_transform_fp(P, eeg_fingerprint(X), lda, p=6, c=1, k=3)` per member (after
  dividing by the batch mean norm), members summed. `lda = fit_fingerprint_lda(<Alljoined dir>, [1..20] + [21..30])`
  is fitted OFFLINE on the raw training-image EEG of all Alljoined + THINGS-EEG2 subjects (656x29 matrix, ships with
  the submission). Inside each batch it regresses the top-6 within-batch PCs of the trials' subject-discriminant
  EEG fingerprints out of the centred outputs (label-free subject routing), then halves the top-3 output
  directions and keeps half the batch mean. Stateless; uses only X.
- **Evidence:** s11 exact testbed (one model, 11 unseen subjects per shuffled batch, E42): plain 31.02 -> 35.52
  (old transform) -> 36.95 (+5.93, 22/0). grader_sim.py --shuffle --fp (official metric, real raw X, cv4, 5 per
  batch, E43): 33.63 -> 38.62 (old) -> **39.80 (+6.17, 20/20; +1.18 vs old, 13/5)**.
- Fallback without the router: `batch_transform(P, k=3)` (+0.3-0.6 over k=2 everywhere, E27-E30).
- THINGS: (i) the transform helps THINGS-trained (+per-dataset BN) conv models 1-2 points more than their no-THINGS
  twins and makes the all-THINGS ensemble the best pick (E29, E31); (ii) THINGS subjects make the router's LDA
  better (E38). Input normalisations (CORAL recolouring, common average reference) training on folds 1-2.
- Dead at N=11: static subject-offset subspace across models (E33), spectral/ridge shrink variants, query
  expansion, member reweighting, row normalisation, image-subspace projection, a 30-60 Hz fingerprint band (E39);
  training under the shuffled transform (E26).

## TL;DR (2026-10-11 evening)
- **Deploy:** `score_track1.batch_transform(P, a=0.5, k=2, b=0.5)` inside `predict()`. Apply it to each member's
  batch output, after dividing by the batch mean norm, then sum the members. It subtracts half the batch mean and
  halves the batch's top-2 centred principal directions. It is stateless and needs no labels, subject IDs or
  gallery, so it is legal under the organisers' rules (batch statistics allowed, no state, shuffled mixed batches).
- **Sealed condition, official metric (E25):** ensemble REVE + SQF + TSC goes 33.63 -> **38.62 (+5.0, 20/20
  subjects better)**. Batches mix 5 participants. The gain decays as more participants mix (E23), to about +2 at
  15 per batch.
- **What did not work:**
  - subject-level routing and expert reweighting: no headroom, even with an oracle (E2, E3)
  - label-free per-trial weighting (E4)
  - SAGE-style static whitening, Procrustes and ridge calibration maps (E17-E19)
  - per-dimension standardisation
  - a learned LDA subject metric (E21)
- **What worked only partly:** in-batch neighbour routing by EEG fingerprint (+1.3 on REVE alone, ~0 on the
  ensemble, 0 at 15 participants per batch). Label-free subject affinity is real but small (E0, E1): only
  grand-ERP similarity predicts it, at r 0.25.
- **Pending:** E26, training under the shuffled-batch transform (TSC).

Goal: route each unseen test subject's trials (or weight training subjects/experts) by something computable from
unlabelled EEG, and beat the plain model/ensemble under the grader's rules.

Grader constraints (neurips-track1-rules): `predict(X)` on batches of 64 trials, shuffle=False, no subject IDs; the
model object persists across calls; no training/downloading/writing during predict; inference < 60 min on one A100.
Scoring averages predictions within (subject, image), then 200-way retrieval -> top-5. A router can therefore be:
per trial (always legal), per batch (legal: uses only X), or stateful across batches (legal in letter, no training;
note the risk if batches mix subjects).

## E0. Is there subject affinity at all? (2026-10-11, affinity_summary.py)
40 REVE-6 scratch models on random 5-subject sets (no THINGS), each scored on its 15 held-out subjects.
- Datamodel a[i,j] (effect of source i in the training set on target j). Interaction (double-centred) split-half
  reliability 0.31; residual-shuffle null 0.00 +- 0.055 (max 0.15 / 300) -> REAL.
- Partly symmetric: corr(I[i,j], I[j,i]) = 0.36.
- Cross-validated: swapping one of a target's bottom-5 sources for one of its top-5 (ranked by interaction on the
  other half) = +0.83 points per swap (36/40 targets); a global source ranking gives +0.59 (30/40).
- Scale caveat: these are 5-subject models (mean 18.7), not full models (~28).

## E1. Which label-free similarities predict it? (2026-10-11, affinity_labelfree.py)
Per-subject features from the test-split EEG (3000 random trials, no labels); Mantel test, subject-label shuffles.
Max observable r ~0.69 (reliability ceiling).

| similarity | r vs interaction | p | r vs raw a | p |
|---|---|---|---|---|
| log-Euclidean spatial covariance | +0.02 | 0.73 | +0.00 | 0.98 |
| channel correlation matrix | -0.00 | 0.99 | +0.06 | 0.46 |
| log spectrum (absolute) | +0.11 | 0.07 | +0.08 | 0.21 |
| log spectrum (shape per channel) | +0.12 | 0.04 | +0.11 | 0.11 |
| **grand-average ERP (all trials, label-free)** | **+0.25** | **<0.001** | **+0.28** | **<0.001** |
| per-channel power | -0.04 | 0.52 | -0.01 | 0.87 |

Reading: subjects whose average evoked response looks alike help each other; second-order statistics
(covariance, power) carry nothing. The ERP needs many trials, so this works only per subject (stateful) or
after per-trial denoising.

## E2. Ceiling of subject-level routing over an expert pool (routing_pool.py)
For each target j, 30 affinity models that never saw it (5-subject REVE-6) are the experts; scored from the saved
per-(subject, image) mean predictions (sealed rule, half the repeats).
| mixing | top-5 | vs uniform |
|---|---|---|
| single expert (mean) | 18.69 | -5.01 |
| uniform over 30 | 23.70 | 0 |
| oracle softmax weights (fit on 100 images, scored on the other 100) | 24.73 | +1.03 (10/20) |
| oracle best-5 experts (same split) | 24.60 | +0.90 (15/20) |
| in-sample oracle best-k subset (cheats) | 27.43 | +3.73 |
Even a label-oracle router gains only ~+1 over uniform mixing; the in-sample number is mostly selection noise.

## E3. Ceiling of per-subject member weighting at full scale (scratchpad oracle_members.py)
Best ensemble REVE-scratch + SQF-dbn + TSC-dbn40 (uniform 33.58). Per subject, 63 weightings (each in {0,.5,1,2}).
- Honest oracle (weights picked on 100 images, scored on the other 100): 33.30 (-0.28, 9/20). **No headroom.**
- In-sample oracle: 36.27 (+2.70), which is pure selection noise.
-> Subject-level routing (pick or reweight models per test subject) is dead at full scale, even with labels.
Remaining levers: per-TRIAL weighting/routing (within-subject heterogeneity), and changing what the experts are.

## Grader order (checked 2026-10-11, github neural-interfaces26/2026-competition @26d5baf)
Test loader = neuralset SegmentDataset of the test split, batch 64, shuffle=False; `predict(X)` only (info with
subject_id is NOT passed to the model). Our cache builder sorts events by (session, run, start), i.e. the timeline
order, so the grader most likely serves trials chronologically, subject after subject; within a subject the RSVP
image order is random. -> "random order" simulations below are the realistic case; one batch per subject boundary
mixes two subjects; a stateful mean cannot see the boundary.

## E4. Per-trial weighting from label-free features (trial_routing.py, tsc_dbn40_things, 20 subjects)
Per-trial caches: trial_cache.py (raw outputs, reproduces run scores exactly); features: trial_features.py
(EEG: log std, log max, clamp, ERP corr, PCA distance, low/high-freq fraction, baseline ratio, worst channel;
model: log norm, margin/max/std of similarity to 2000 TRAINING images, agreement, typicality).
Usefulness target = z-scored cosine margin of the true image; fit leave-fold-out.
- Fit R2 = 0.04: per-trial usefulness is essentially unpredictable from label-free features.
- learned exp(g*pred) weights: -0.10 / -0.15 / -0.85; drop bottom 10/25%: +0.08 / +0.05 (noise).
- Hoke inverse-variance weights: std^-1 -0.80, std^-2 -2.05 (hurt: high-variance trials are not the bad ones).
-> DEAD. Trial-level quality weighting has no label-free handle here.
- Side finding: subtracting 0.5 x the subject's mean output: **+2.93 (17/20)** -> E5.

## E5. Output centring = label-free per-subject adaptation (centring.py; 20 subjects, sealed rule, all repeats)
Return q - a*m. Ranking becomes q.g - a*(m.g): gallery images the subject's average output already resembles are
penalised (hubness correction), AND the subject's own offset in output space is removed.
| m from | a | TSC-dbn40 (26.40) | SQF-dbn (30.55) | SQF+TSC (30.30) |
|---|---|---|---|---|
| subject mean (all trials) | 0.5 | +2.93 (17/3) | +3.50 (18/1) | +3.32 (16/3) |
| running mean, random order (stateful) | 0.5 | +2.95 (17/3) | +3.30 (18/1) | +3.30 (16/4) |
| running mean, image-major order | 0.5 | +2.83 | +3.45 | |
| **mean of the 64-trial batch (stateless)** | **0.5** | **+3.00 (17/3)** | **+3.57 (20/0)** | **+3.25 (16/2)** |
| other test subjects' mean (static constant) | 0.5 | +2.10 | +0.32 | +1.68 |
| training-image mean direction (static) | 0.5 | -19.4 | -20.9 | |
a sweep: 0.25 is ~+2, 0.5 is best, 0.75 is ~+2, 1.0 is ~0 (full centring overshoots).
cos(other-subject mean, subject mean) = 0.98: the mean is mostly shared, but the subject-specific 2% is where
SQF's gain lives (static constant: +0.3 vs batch: +3.6).
Why batch64 is safe even in the worst order: if a batch held repeats of one image, subtracting 0.5 x its mean
only rescales that image's average, and cosine ignores scale -> worst case ~= no change.

## E6. Grader-stream stress test of batch-mean centring (centring_stream.py)
Each fold's 5 subjects streamed back to back (random order within a subject), cut into batches of B ignoring
subject boundaries (boundary batches mix two subjects), q - a * batch mean.
| setting | TSC-dbn40 (26.40) | SQF-dbn (30.55) |
|---|---|---|
| B=16 / 32 / 64 / 128 / 256, a=0.5 | +3.33 / +2.98 / +3.12 / +3.10 / +3.00 | +3.30 / +3.55 / +3.32 / +3.47 / +3.47 |
| B=64, a=0.4 / 0.6 | +2.80 / +3.15 | +3.25 / +3.57 |
| image-major order (worst case), B=64 a=0.5 | +0.45 | +1.12 |
| per-dimension batch standardisation | -7.40 | +0.12 |
SQF: better on 19-20/20 in every streamed setting. Insensitive to B and to boundary mixing; never harmful even
in the adversarial order. Standardisation is harmful: it is the mean shift, not a rescaling, that helps.

## E7. AdaBN: test-time batch statistics in every BatchNorm (trial_cache.py --adabn; adabn_compare.py)
64-trial batches, random order within subject; + stream batch-mean centring (B=64, a=0.5).
| | TSC-dbn40 | SQF-dbn |
|---|---|---|
| plain | 26.40 | 30.55 |
| plain + centring | +3.12 (17/3) | +3.32 (20/0) |
| AdaBN | +0.88 (13/5) | +1.30 (12/6) |
| AdaBN + centring | +3.23 (18/2) | +3.62 (16/3) |
AdaBN helps alone, but centring captures nearly all of it (+0.1 / +0.3 on top, less consistent on SQF).
Not adopted for now: BN-only models, more batch-composition risk, marginal gain.

## E8. Beyond the mean: shrink the batch's top principal directions (centring_pca.py; stream B=64, after a=0.5)
y = q - 0.5*mu - b * proj of (q - mu) on the top-k right singular vectors of the centred batch.
| k, b | TSC-dbn40 vs centring (29.52) | SQF-dbn vs centring (33.88) |
|---|---|---|
| 1, 0.5 | **+2.03 (15/5)** | **+3.00 (17/3)** |
| 2, 0.5 | +1.95 (13/4) | +3.17 (15/5) |
| 4, 0.5 | +1.65 | +1.83 |
| 8, 0.5 | +0.05 | +0.27 |
| 1, 1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 | -3.2 / -5.9 / -11.1 / -18.8 | -1.7 / -4.6 / -10.8 / -21.2 |
The dominant within-batch direction is a nuisance axis: halving it helps a lot, removing it hurts (it still
carries signal). Selected on the same 20 subjects -> E9 re-tests a finer grid per fold pair.

## E9. Per-fold-pair confirmation of PC shrinkage (centring_pca2.py; deltas vs centring alone; folds 1-2 | 3-4)
| config | TSC-dbn40 | SQF-dbn |
|---|---|---|
| pc k=1 b=0.4 | +1.25 (7/2) / +2.15 (7/0) | +2.20 (8/1) / +2.60 (9/1) |
| **pc k=1 b=0.5** | **+1.55 (7/3) / +2.50 (8/2)** | **+2.35 (8/2) / +3.65 (9/1)** |
| pc k=1 b=0.6 | +1.30 / +2.10 | +2.15 / +3.65 |
| pc k=2 b=0.5 | +1.35 / +2.55 | +2.40 / +3.95 |
| white g=0.25 (partial whitening) | +0.25 / +0.75 | +1.25 / +2.05 |
| pc k=1 b=0.5 at B=32 / 128 | +1.65 / +2.30, +1.50 / +2.50 | +2.10 / +3.30, +2.30 / +3.50 |
| plain (no centring at all) | -4.20 / -2.05 | -3.25 / -3.40 |
Screening on folds 1-2 picks k=1 (or 2), b=0.5; it replicates on folds 3-4 (+2.50 TSC, +3.65 SQF). Insensitive to
batch size (32-128); whitening the whole spectrum is worse than shrinking only the top direction.
Total vs plain: TSC 26.40 -> ~31.5, SQF 30.55 -> ~36.9.

## E10. What is the top batch direction? (pc1_diag.py, SQF-dbn)
|cos| between batches' PC1 0.43-0.59 (semi-stable); |cos(PC1, batch mean)| ~0.19; |corr(PC1 score, trial norm)|
~0.2; PC1 carries 19-25% of the centred batch variance. A STATIC direction (leading axis of the other subjects'
batch PC1s) gives 34.98 vs per-batch PC1 36.38 (centring alone ~33.9): the static axis captures ~40% of the
gain; per-batch estimation is worth it. Fallback if a batch is tiny: static direction.

## E11. Ensemble search with the transform (ens_transform.py; centring a=0.5 + PC1 shrink b=0.5, B=64 stream)
Plain reproduces cv4_ensemble exactly (33.15 / 33.55), validating the reduction.
| | plain | transformed |
|---|---|---|
| REVE-scratch+THINGS | 29.40 | **38.92** |
| SQF-dbn+THINGS | 30.55 | 36.88 |
| TSC-dbn40+THINGS | 26.40 | 31.55 |
| TSC-b11 | 27.68 | 30.25 |
| REVE + SQF | 33.15 | 40.75 |
| REVE + SQF + TSC-dbn40 | 33.55 | **40.85** |
| + TSC-b11 (4) | 33.70 | 39.55 |
The transform adds more to a single model (+9.5 REVE) than to the ensemble (+7.3): ensembling already averaged
away part of the nuisance variation. Member choice is unchanged.

## E12. End-to-end grader simulation (grader_sim.py) -- the deployability check
- Order: each subject's REAL test trials in recording order, reconstructed from the neuralset timeline (session,
  run, start; the harness's test loader neither shuffles nor reorders: neuralset sorts events by (timeline,
  start), nb_task builds test loaders with shuffle=False). The fold's 5 subjects back to back, batches of 64 cut
  across subject boundaries. Only real trials (no cycled fills): ~80k per fold.
- predict(X) is stateless: each member's batch output / its batch mean norm, then batch_transform
  (a=0.5, k=1, b=0.5), members summed. No subject IDs, no labels, no gallery, nothing kept across calls.
- Metric: official top5_acc_subject_agg (official_metrics.py = verbatim benchmark_utils/metrics.py @26d5baf).
REVE-scratch + SQF-dbn + TSC-dbn40, best-epoch checkpoints:
| fold | plain | transformed |
|---|---|---|
| 1 | 35.30 | 41.00 |
| 2 | 31.70 | 38.30 |
| 3 | 32.70 | 41.70 |
| 4 | 31.10 | 38.90 |
| **mean (20 subjects)** | **32.70** | **39.98 (+7.28, 20/20 better)** |
(The plain number differs slightly from E11's 33.55: per-batch norm scaling instead of a per-fold constant, and
only real trials.)

## E13. Train under the transform (train.py --batch_transform; recipe tsc_dbn40_bt_things) -- in progress
Per subject group inside each training batch (~41 trials of one subject at batch 1024), apply the same transform
before the loss (SVD direction detached); sealed evals during training use random-order 64-trial batches + transform.
Bug found and fixed on the way (2026-10-11 ~16:00): inserting batch_transform into score_track1.py had moved the
`@torch.no_grad()` decorator from score_modules onto batch_transform -> training got no gradient through the
transform (loss flat at 7.05, chance accuracy, even with the identity a=b=0), and score_modules lost no_grad.
Bisection (identity / centring-only / PC-only all flat; control without the flag learns) located it. First runs
deleted; self-test now asserts the transform is differentiable. Results computed outside train.py (E5-E12: trial
caches, grader_sim under explicit no_grad) are unaffected.

## E14. The OFFICIAL loader's order, observed directly (official_order_probe.py, kit venv, warm-up THINGS-EEG2 test)
The challenge kit's neuralbench test loader (what the harness iterates), batch 64: 162,240 trials, 200 images;
subjects served one after another (order 0,2,3,...,9,1); within a subject images interleaved -- distinct images
per 64-batch median 55 (min 46, max 64), longest same-image run 3, mean run 1.01. This is the condition under
which the batch transform was validated (E6, E12); the sealed set goes through the same neuralset pipeline.
(The loader carries subject_id, but the harness passes only X to predict.)

## !! RULES UPDATE (Discord, organisers, 2026-09-29 / 10-09 / 10-10) -- changes the deployment condition
- "Yes, the evaluation batches will be shuffled. This change is coming in the next neuralbench release. You can
  use the full batch statistics, but be aware that the test batch size may change later on." (hubertjb)
- "No persistent state may be carried across predict calls. Statistics from the current batch are allowed, but
  sealed batches are shuffled, may mix participants, and must be treated independently." (Caillet, confirmed)
- The sealed objective will aggregate repeats within subject (neuralbench pooling order); gallery size and number
  of repeats undisclosed ("models should be agnostic to both").
=> E6/E12/E14 validated the transform on subject-pure, chronological batches (today's warm-up loader). In the
sealed phase a batch is a random mix of participants: the batch mean is then mostly the POPULATION mean, the
subject-specific part (most of the gain, E5) is diluted, and any running/stateful estimate is forbidden.
Re-test: E15 (shuffled mixed-participant batches, static corrections); E16 (can a batch's trials be routed to
their participant without labels?).

## E16. Can a shuffled batch's trials be routed to their participant without labels? (subject_separability.py)
Fold's 5 unseen test subjects, REVE-scratch outputs and raw-EEG fingerprints (per-channel log-variance + channel
correlations).
| features | nearest-centroid id (centroids from labelled half; chance 20%) | k-means purity inside one 64-batch |
|---|---|---|
| raw EEG fingerprint | 55-74% | 0.38-0.46 |
| model output | 38-44% | 0.36-0.39 |
Subjects are identifiable with labelled centroids, but unsupervised grouping inside a single mixed batch is weak.
A learned subject-embedding would be needed for in-batch routing (and the rebuttal-period floor: per-subject
statistics from < 20 trials are harmful).

## E17. Gallery-free whitening from the query side (static_whiten.py): q' = (S_img + l)^(-alpha) q
S_img = covariance of unit-norm TRAINING-image DINOv2 features, so q'.g is a whitened inner product. Static.
Catastrophic everywhere: REVE 29.40 -> 19.9 (alpha .25) ... 5.1 (alpha 1); SQF, ensemble likewise, on plain AND on
transformed outputs. Whitening only the gallery side amplifies directions where the EEG prediction is pure noise.
SAGE pairs it with query whitening Sq^(-1/2) and a rotation R -> E18.
(E13, training under the subject-pure transform, and the warm-up check were stopped: both assume subject-pure
batches, which the sealed grader will not provide.)

## E18. Static SAGE map from the query side (static_sage.py), REVE-scratch
q' = Wg R^T Wq (q - mq): query whitening (population, other folds' subjects), gallery whitening (training
images), Procrustes rotation R. Even the LEAKY version (R and Sq fitted on the same 200 evaluation images) is
strongly negative: best (heaviest shrinkage lq=lg=1) 24.02 vs 29.40 plain; query whitening alone 4.7-16.0.
SAGE's gains came from per-subject, transductive fits against the known gallery; population whitening of noisy
single-subject predictions amplifies noise directions. DEAD as a static, gallery-free, cross-subject map.

## E15. Shuffled, mixed-participant batches (mixed_batch.py) -- the sealed condition
Fold's 5 unseen test subjects pooled, shuffled uniformly, cut into batches of B.
REVE-scratch (plain 29.40):
| variant | delta | better/worse |
|---|---|---|
| per-subject batches (old assumption) | +9.52 | 20/0 |
| **shuffled B=64 centre + PC1 shrink** | **+4.50** | 16/1 |
| shuffled B=256 / 1024 centre + PC1 | +4.52 / +4.30 | 17/2, 17/3 |
| shuffled B=64 centre only | +1.58 | 11/6 |
| shuffled B=64 PC only | +0.78 | 12/8 |
| static mean (other subjects) | +0.35 | 9/10 |
| static mean + static direction | +2.62 | 14/6 |
Half the gain survives mixing. Centring and PC shrink are synergistic in mixed batches: after centring, the top
direction of a mixed batch partly IS the spread of the subjects' offsets, so shrinking it is implicit label-free
routing. Open: more subjects per batch (sealed N unknown) -> more PCs needed?

## E15 cont. (shuffled mixed batches, 5 subjects per fold pooled)
| | plain | per-subject (old) | shuffled B=64 centre+PC1 | shuffled B=256 | static mean+dir |
|---|---|---|---|---|---|
| REVE-scratch | 29.40 | +9.52 | +4.50 (16/1) | +4.52 | +2.62 |
| SQF-dbn | 30.55 | +6.12 | +3.35 (17/3) | +3.68 | +2.22 |
| TSC-dbn40 | 26.40 | +5.12 | +3.80 (18/1) | +3.60 | +3.10 |
TSC's gain is mostly population-level (static correction alone +3.10); REVE's is the most subject-specific.

## E19. Clean static calibration (train_split_cache.py + calib_map.py)
Fit on the OTHER held-out subjects of the same fold model, on their TRAINING-image trials (no test image, no
evaluated subject), evaluated on the left-out subject's 200 test images. REVE-scratch (plain 29.40):
- clean static mean + top direction (a=.5, b=.5): **31.90 (+2.50, 14/6)** -> the static fallback is real and clean.
- clean static mean only: +0.58.
- ridge map to DINOv2 targets (shrunk to identity): -25.6 ... -10.9; centred version: -12.9 ... -1.5 (-> 0 as
  lam -> inf). A learned linear recalibration never beats the network's own readout: DEAD.
- SQF-dbn clean static mean+dir: +2.03 (16/4); TSC-dbn40: +3.12 (19/1); ridge maps negative for both.

## E15 ensemble (REVE + SQF + TSC, plain 33.55), shuffled mixed batches of the fold's 5 subjects
| variant | delta | better/worse |
|---|---|---|
| per-subject batches (old assumption) | +7.23 | 20/0 |
| **shuffled B=64 centre + PC1 (per member)** | **+4.08 -> 37.62** | 18/1 |
| shuffled B=256 / 1024 | +4.12 / +4.20 | 17/2, 18/1 |
| centre only / PC only | +0.58 / +0.58 | |
| static mean + static dir (no batch statistics) | +3.28 | 15/5 |
Under the sealed condition the batch transform keeps ~57% of its subject-pure gain on the ensemble.

## E20. Label-free in-batch routing: neighbour centring (neighbour_centring.py), REVE-scratch, shuffled mixed 64-batches
Each trial is centred on the mean of its K nearest batch-mates in raw-EEG fingerprint space (per-channel log-var +
channel correlations, z-scored within the batch), then top-direction shrink.
| centring on | delta vs plain 29.40 | better/worse |
|---|---|---|
| whole batch (K=0) | +4.38 | 16/3 |
| **K=4 fingerprint neighbours** | **+5.15** | 18/2 |
| K=8 / 16 / 32 | +4.80 / +4.62 / +4.77 | 18/1 |
| oracle: the trial's true same-subject batch-mates | **+8.38** | 20/0 |
Routing trials to "same-looking" batch-mates is worth +0.8 already, and a perfect label-free router would be worth
+4 more -> the subject metric is the lever. E21: LDA subject metric learned on training subjects.
SQF-dbn (plain 30.55): whole batch +3.22; K=4/8 +3.55/+3.57; oracle same-subject +4.95.

## E22. Shuffled-batch sweep (mixed_sweep.py; deltas vs plain; folds 1-2 | folds 3-4)
Ensemble REVE+SQF+TSC (plain 34.20 | 32.90):
| config | folds 1-2 | folds 3-4 |
|---|---|---|
| B=64 a=.5 k=1 b=.5 | +2.95 (8/1) | +5.10 (10/0) |
| **B=64 a=.5 k=2 b=.5** | **+4.35 (10/0)** | **+6.50 (10/0)** |
| B=64 a=.5 k=3 b=.5 | +4.55 (10/0) | +6.40 (10/0) |
| B=64 a=.5 k=4 b=.5 | +5.00 (10/0) | +6.25 (10/0) |
| B=64 a=.5 k=3 b=.7 (screen best) | +5.95 (9/1) | +5.90 (10/0) |
| B=64 a=.5 k=8 b=.7 | +5.00 | +1.15 (breaks) |
| B=256 a=.5 k=2 b=.5 | +4.20 | +6.75 |
| static mean+dir only (no batch stats) | +1.90 | +4.35 |
k=2-4 with b=.5 improves all 20 subjects; k=2 b=.5 -> ensemble ~39.0 vs 33.55. Singles: REVE k=2 b=.5 +2.90/+6.50,
SQF +2.70/+4.85. a=.5 stays best; a=1 (full centring) hurts.

## E23. Scaling with the number N of participants mixed in the stream (nsubject_scaling.py)
Affinity models s1-s4 (REVE-6, 5 training subjects, 15 unseen each; plain 20.85), B=64, a=.5, b=.5:
| N | centre only | + top-1 | + top-2 | + top-4 |
|---|---|---|---|---|
| 1 | +4.85 | +6.18 | +5.88 | +4.67 |
| 3 | +1.57 | +3.07 | +3.50 | +2.97 |
| 5 | +0.84 | +2.29 | +2.34 | +2.31 |
| 15 | +0.50 | +1.81 | +2.17 | +1.67 |
The gain decays with N but plateaus near 1/3 of the single-subject gain; centring alone fades (the batch mean
becomes the population mean), the top-direction shrink carries it. k=2 is best for N >= 3 -> default k=2, b=.5.

## E21. Learned subject metric for routing (neighbour_lda.py), REVE-scratch, shuffled mixed 64-batches
Fingerprint = per-channel log-var + channel correlations + per-channel log band power (1-4, 4-8, 8-13, 13-30 Hz).
LDA fitted on the other folds' 15 subjects (identities of training subjects known offline).
| | delta vs plain 29.40 | better/worse | same-subject neighbour purity (chance 0.19) |
|---|---|---|---|
| whole-batch centring (K=0) | +4.15 | 17/2 | |
| raw z-scored fingerprint K=4 / K=8 | +5.27 / **+5.62** | 19/1, 18/2 | 0.49 / 0.40 |
| LDA K=4 / 8 / 16 | +5.45 / +5.30 / +5.02 | 18/2 | 0.50 / 0.43 / 0.34 |
The richer fingerprint helps (+5.62 vs +5.15 in E20); a learned linear subject metric does not raise purity
(0.50 vs 0.49): the fingerprint content, not the metric, limits routing. Label-free routing is worth +1.1-1.5 over
whole-batch centring in the sealed condition (oracle +4.2 more).

## E24. Final candidates under the sealed condition (route_final.py; shuffled mixed 64-batches)
| | whole k=1 | whole k=2 | route K=8 k=2 | route K=4 k=2 |
|---|---|---|---|---|
| REVE (cv, 5 subj; f1-2 / f3-4) | +2.15 / +6.45 | +2.50 / +6.35 | +3.25 / +7.65 | **+3.60 / +7.85** |
| Ensemble REVE+SQF+TSC (cv) | +3.35 / +5.75 | **+4.35 / +6.25 (20/0)** | +4.45 / +6.50 (20/0) | +3.75 / +5.85 |
| affinity N=5 (plain 20.85) | +2.52 | +2.78 | +3.37 | +3.22 |
| affinity N=15 | +1.82 | +2.14 | +2.12 | +2.14 |
Label-free neighbour routing helps a single model (+1.3 on REVE) and small subject mixes (+0.6 at N=5), but is
~nothing on the ensemble (+0.1-0.25) and nothing at N=15 (too few same-subject batch-mates to find).
Robust choice for the sealed phase: whole-batch transform a=.5, k=2, b=.5 per member (no routing): ensemble
33.55 -> ~38.9 with 5 mixed subjects, never worse on any subject; expected to shrink toward +2 if many
participants are mixed (E23).

## E25. End-to-end under the SEALED condition (grader_sim.py --shuffle): official metric, real trials, all of a
fold's 5 unseen subjects shuffled together (every batch mixes participants), batches of 64, no state, per-member
batch-norm scaling, ensemble REVE + SQF + TSC.
(A first run was invalid: the fold loop variable `k` shadowed the transform's `k` -> fixed, `n_dirs`.)
| fold | plain | k=1 | **k=2** |
|---|---|---|---|
| 1 | 36.20 | 37.60 | 39.80 |
| 2 | 32.60 | 35.60 | 36.80 |
| 3 | 34.20 | 39.10 | 40.80 |
| 4 | 31.50 | 36.60 | 37.10 |
| mean (20 subjects) | 33.63 | 37.23 (+3.60, 18/2) | **38.62 (+5.00, 20/0)** |
=> DEPLOY: score_track1.batch_transform(P, a=0.5, k=2, b=0.5) on each member's predict() output (after dividing
by the batch mean norm), then sum the members. Defaults changed to k=2.

## E26. Train under the shuffled-batch transform? (tsc_dbn40_btm_things vs tsc_dbn40_things, route_final.py cv)
TSConv-dbn40 + THINGS trained with `--batch_transform --batch_transform_mixed 64` (transform on random 64-chunks,
directions detached), evaluated under the same shuffled mixed transform (5 subjects / batch).
| | plain | whole k=1 | whole k=2 |
|---|---|---|---|
| trained under transform (f1-2 / f3-4) | 17.10 / 17.15 | 24.80 / 27.20 | 24.60 / 26.45 |
| normal training + post-hoc (f1-2 / f3-4) | 26.20 / 26.60 | 30.35 / 28.90 | 30.00 / 30.65 |
=> NEGATIVE (-5.5 / -3.5 at the deployed setting). Keep the transform post-hoc only.

---
# Phase 2 (2026-10-12): the sealed cohort is 11 participants
tracks.html (checked 2026-10-12): Track 01 sealed set = "This new 11-participant cohort is reserved entirely for
hidden evaluation", same Emotiv hardware / natural-image protocol as Alljoined-1.6M. With shuffled batches of 64,
~11 participants share every batch (~6 trials each). New goal: optimise the transform for N = 11, and find
transforms/normalisations that draw more from THINGS. Alljoined stim_test stays held out; THINGS test may be used.

Testbeds:
- **n11.py (composite):** each of the 20 cv4 subjects predicted by the fold model that held it out (shared DINOv2
  output space), 11 subjects drawn per draw (4 draws), shuffled into 64-batches. Caveat: up to 4 fold models in
  one batch add a model offset.
- **s11.sh (exact):** 2 new splits of 11 test / 9 train Alljoined subjects (+ THINGS), the 3 ensemble recipes
  (reve_d6_things, sqf_dbn_things, tsc_dbn40_things), jobs 60045/60046. One model per split, so batches are as in
  the sealed phase (weaker models: 9 training subjects).

## E27. Transform search at N = 11 (n11.py composite testbed, ensemble REVE-scratch + SQF-dbn + TSC-dbn40)
4 draws x 11 subjects; columns = subject halves folds 1-2 | folds 3-4 (delta vs plain, better/worse over
subject x draw). Plain 32.44 | 35.15.
| transform | f1-2 | f3-4 |
|---|---|---|
| E25 a.5 k2 b.5 (deployed) | +3.42 (20/4) | +5.05 (20/0) |
| a 0.25 / 0.75 / 1.0 (k2 b.5) | +2.96 / +1.54 / -2.29 | +3.80 / +4.10 / +2.03 |
| k1 b.5 | +2.52 | +4.02 |
| k3 b.5 | +4.00 | +5.35 |
| k4 b.5 | +4.10 (23/1) | +5.47 (20/0) |
| k6 b.5 / k8 b.5 / k12 b.5 | +3.52 / +2.48 / +1.27 | +5.23 / +4.20 / +2.22 |
| b 0.75 (k2 / k3 / k4) | +3.77 / +4.15 / +3.67 | +4.28 / +4.45 / +4.05 |
| b 1.0 (projection removal) | catastrophic beyond k=1 | |
| spectral power: kept (lam/mean)^-g/2, g .5 / .75 / 1 | +3.25 / +4.25 / +3.19 | +5.38 / +5.10 / +4.00 |
| spectral ridge 1/(1+b lam/median) | -22 | -22 |
| directions in the image-covariance metric g.5 k2 / k4 | +3.69 / +4.25 | +5.45 / +5.38 |
| project onto training-image PCs r64 / r256 | -8.7 / -3.2 | -9.4 / -1.3 |
| + row norm power p 0 / .5 / 1.5 | +2.21 / +3.06 / +3.56 | +3.80 / +4.75 / +5.00 |
| static subject-offset subspace r4 c.5 (on top of E25) | **+4.54 (23/1)** | **+6.00 (20/0)** |
| static r8 c.5 / r4 c1 / r8 c1 | +3.33 / +1.29 / -2.42 | +6.10 / +3.30 / -0.48 |
| static population mean blended into batch mean wmu .5 / 1 | +3.52 / +3.65 | +5.05 / +4.52 |
| member weights (1.5x one member) | +3.15..+3.58 | +4.70..+5.32 |
| in-batch query expansion (lam .25-.5, 1-3 nbrs) | +1.75..+2.77 | +2.78..+4.08 |
| robust centre (median / 10% norm-trimmed mean) | +3.75 / +3.77 | +5.20 / +5.20 |
Reading: the deployed transform still gives +3.4/+5.1 with 11 participants mixed (more than the affinity-model
N=15 estimate, +2.2; composite caveat: fold-model offsets inflate between-"subject" spread). Room above it is
~+0.5-1: more directions (k 3-4), soft spectral shrink, and a static subject-offset subspace (fitted offline on
OTHER subjects' outputs; r4, c.5) is the best single addition (+1.1/+1.0). The static gain must be re-checked
where fit and test models differ (s11 exact testbed: fit on the other split's model). Neighbour smoothing,
member reweighting, row normalisation, image-subspace projection: dead.

## E28. Round 3 at N = 11 (n11.py --cfgs r3): combinations
Best (f1-2 | f3-4, vs plain 32.44 | 35.15; E25 = +3.42 | +5.05):
| transform | f1-2 | f3-4 |
|---|---|---|
| static k3 r2 c.5 | +5.04 (21/1) | +6.40 (20/0) |
| static a.6 k4 r4 c.5 | +4.69 (22/1) | +6.83 (20/0) |
| static k3 r4 c.25 | +4.60 (23/0) | +6.18 (20/0) |
| static k2 r4 c.25 | +4.38 (23/1) | +6.32 (20/0) |
| a.6 k4 b.5 (no static) | +3.65 | +5.97 |
| spectral g.75 + static | +3.7..+3.9 | +4.9..+5.4 |
All static variants with c <= .5 land within ~0.5 of each other: +1.3 over E25 on the composite testbed.

## E29. Does the transform draw more from THINGS? (n11.py --cfgs pair, single members, N = 11 composite)
| member | plain | E25 | k4 | static k2 r4 c.5 |
|---|---|---|---|---|
| reve_d6 | 25.73 / 27.55 | +3.40 / +4.70 | +3.69 / +4.15 | +2.94 / +4.55 |
| reve_d6_things | 28.13 / 29.75 | +2.40 / +5.65 | +2.56 / +5.83 | +2.69 / +5.38 |
| sqf (30 ep) | 24.88 / 28.55 | +0.73 / +2.60 | +0.08 / +2.20 | +0.67 / +2.47 |
| sqf_dbn_things | 29.88 / 32.02 | +1.96 / +3.47 | +1.44 / +2.98 | +0.71 / +2.38 |
| tsc_b11 | 27.08 / 29.27 | +0.27 / +1.62 | -0.15 / +0.68 | +0.52 / +1.65 |
| tsc_dbn40_things | 25.63 / 28.60 | +2.06 / +3.60 | +1.81 / +2.70 | +2.77 / +4.48 |
THINGS effect (with - without), plain -> after E25: REVE +2.40/+2.20 -> +1.40/+3.15; SQF (+dbn) +5.00/+3.47 ->
+6.23/+4.35; TSC (+dbn, 40 ep) -1.45/-0.67 -> +0.34/+1.30 (static: +0.81/+2.15). The conv models trained with
THINGS + per-dataset BN gain 1-2 points more from the transform: post-hoc batch centring turns TSConv's THINGS
loss into a gain. (Pairs differ in more than THINGS: dbn and epoch budget too.)

## E30. Deployability and batch-size checks of the round-3 winners (ensemble REVE-scratch + SQF-dbn + TSC-dbn40)
(a) Real cv4 batches (N = 5, one model per fold), static subspace fitted on the other 15 subjects AS PREDICTED BY
THE OTHER FOLD MODELS (= deployment: fit on cv models, apply to another model). n11.py --cv --cfgs best.
| transform | f1-2 (plain 34.15) | f3-4 (plain 32.90) |
|---|---|---|
| E25 k2 | +4.35 | +6.45 |
| k3 | +4.75 | +6.65 |
| k4 | +4.50 | +6.15 |
| static k3 r2 c.5 | +6.05 | +5.90 |
| static k2 r4 c.25 | +5.35 | +6.45 |
| static k3 r4 c.25 | +5.45 | +6.20 |
| static a.6 k4 r4 c.5 | +5.45 | +6.40 |
Cross-model static transfer is inconsistent: +1-1.7 on folds 1-2, -0.5-0 on folds 3-4. k3 is a small consistent
gain (+0.4 / +0.2).
(b) N = 11 composite at batch 32 / 64 / 128 (delta vs plain, f1-2 | f3-4):
| transform | b32 | b64 | b128 |
|---|---|---|---|
| E25 k2 | +3.35 / +4.80 | +3.42 / +5.05 | +3.54 / +4.92 |
| k3 | +3.85 / +5.07 | +4.00 / +5.35 | +4.17 / +5.80 |
| k4 | +3.98 / +5.02 | +4.10 / +5.47 | +4.33 / +6.05 |
| static k3 r2 c.5 | +4.81 / +5.93 | +5.04 / +6.40 | +5.25 / +6.30 |
Ordering is stable across batch sizes; k3 > k2 everywhere (+0.3-0.9).

## E31. Ensemble selection UNDER the transform (ens11.py, N = 11 composite, 11 cached members, pick on one subject
half, report on the other; draws differ from E27 so absolute numbers differ slightly)
| transform | current ens (f1-2 / f3-4) | best size-3 picked on f3-4 | best size-3 picked on f1-2 (-> other half) |
|---|---|---|---|
| plain | 34.37 / 34.29 | sqf_dbn+reve_z_things+tsc_b11 (-> 33.28) | reve_scratch+sqf_dbn+tsc_b11 (-> 33.57) |
| E25 k2 | 37.09 / 38.90 | = current (-> 37.09) | sqf_dbn+tsc_dbn40+reve_d6_things (-> 38.48) |
| k4 | 38.17 / 39.95 | = current | = current |
| static k3 r2 c.5 | 38.70 / 40.57 | = current | sqf_dbn+tsc_dbn40+reve_d6_things (-> 40.40) |
Without the transform the selection reaches for the no-THINGS TSConv (tsc_b11); under any transform it picks the
three THINGS-trained members (the current ensemble, or its REVE-d6 twin). Size 4 never beats size 3 out of sample.
=> keep REVE-scratch + SQF-dbn + TSC-dbn40 (all THINGS); the transform is what makes the THINGS TSConv the right pick.

## E32. Label-free regression routing (n11.py --cfgs fp/fp2): per batch, regress the centred outputs on the top-p
within-batch PCs of the trials' raw-EEG fingerprints (log-var, channel correlations, band powers; z-scored with
the batch's own statistics), remove c x the fit, then the k3 shrink. Control: same with random trial-axis
directions (pure extra shrinkage).
| transform | N=11 composite f1-2 / f3-4 | cv4 N=5 f1-2 / f3-4 |
|---|---|---|
| k3 | +4.00 / +5.35 | +4.75 / +6.65 |
| fpreg p4 c1 | +4.27 / +5.70 | +6.65 / +5.75 |
| fpreg p8 c1 | **+4.73 / +5.55** | **+7.00 / +5.65** |
| fpreg p16 c1 | +4.54 / +4.50 | +7.10 / +3.80 |
| random control p4 / p8 / p16 | +3.71 / +3.67 / +2.77 ; +5.12 / +4.75 / +3.60 | +4.45 / +4.45 / +2.45 ; +5.95 / +5.75 / +2.75 |
The fingerprint carries real subject information (+1 over its random control), but against plain k3 it is
fold-dependent, exactly like the static subspace (E30): +2.3 on cv folds 1-2, -1.0 on folds 3-4. Same split in
both subject-aware additions -> folds 3-4 subjects are less separable / their offsets less EEG-predictable.
Not adopted unless the s11 exact testbed (one model, 11 subjects) favours it.

## E33. s11 EXACT testbed (one model per split, 11 unseen subjects per shuffled 64-batch; static fitted on the other
split's model and subjects = deployment situation). n11.py --s11 --cfgs final.
reve_d6_things alone (9 Alljoined + 10 THINGS training subjects), 22 subject x split entries:
| transform | top-5 | delta (better/worse) |
|---|---|---|
| plain | 27.20 | |
| E25 k2 | 31.61 | +4.41 (20/1) |
| k3 | 31.82 | +4.61 (20/1) |
| k4 | 31.75 | +4.55 (22/0) |
| static k3 r2 c.5 | 30.48 | +3.27 (20/1) |
| static k2 r4 c.25 | 31.43 | +4.23 (21/0) |
| fpreg p4 c1 k3 | 32.05 | +4.84 (21/1) |
| fpreg p8 c1 k3 | **32.68** | **+5.48 (21/1)** |
With a single model and a truly mixed 11-subject stream the deployed transform still gives +4.4. The static
subspace does NOT transfer across models (below E25) -> dropped. Fingerprint regression +0.9 over k3. Ensemble
version pending (sqf/tsc s11 runs).

## E34. Fingerprint-regression refinements on the s11 exact testbed (reve_d6_things, plain 27.20)
| transform | delta |
|---|---|
| k3 | +4.61 |
| fpreg p6 / p8 / p12 (c1 k3) | +5.25 / +5.48 / +5.57 |
| fpreg p8 k2 / k4 | +4.95 / +5.66 |
| fpreg p8 c.75 | +5.32 |
| fpreg p8 b.4 / b.6 | +5.05 / +5.30 |
| fplda p4 c1 k3 | +5.34 (22/0) |
| **fplda p8 c1 k3** | **+6.02 (21/0)** |
fplda: the fingerprint is first projected on an LDA subject-discriminant space learned OFFLINE from the other
split's 9 subjects' raw EEG (model-independent, so it transfers, unlike the static output subspace), then the
within-batch PCs are regressed out. +1.4 over k3 on a single model. Checking on cv4 (LDA from the other 15
subjects) and the N=11 composite (ensemble).

## E35. fp3 on the ensemble: cv4 (N = 5, LDA learned on the other 15 subjects' raw EEG) and N = 11 composite
| transform | cv4 f1-2 / f3-4 (k3 = +4.75 / +6.65) | composite N=11 f1-2 / f3-4 (k3 = +4.00 / +5.35) |
|---|---|---|
| fpreg p8 c1 | +7.00 / +5.65 | +4.73 / +5.55 |
| fpreg p8 c.75 | +7.25 / +5.80 | +4.73 / +5.72 |
| fpreg p12 c1 | +7.45 / +5.10 | +4.73 / +5.08 |
| fpreg p8 k2 / k4 | +6.95 / +5.75 ; +7.15 / +5.20 | +5.21 / +5.17 ; +4.73 / +5.30 |
| fplda p4 c1 | +6.90 / +6.25 | +4.52 / +5.43 |
| fplda p8 c1 | +7.35 / +5.45 | +4.60 / +4.98 |
Over all 20 cv subjects: k3 39.23, fplda p4 40.10 (+0.87), fpreg p8 c.75 40.05 (+0.82); but the gain is
+2.2-2.5 on folds 1-2 and -0.4..-0.85 on folds 3-4. The LDA projection helps only on the single model (E34).
Per-fold / per-subject breakdown queued (n11.py --cv --cfgs fpbest).

## E36. Per-fold / per-subject breakdown on cv4 (delta vs E25 k2; one subject = 200 images, 1 image = 0.5)
| transform | f1 | f2 | f3 | f4 |
|---|---|---|---|---|
| k3 | +0.50 | +0.30 | +0.30 | +0.10 |
| fpreg p8 c.75 k3 | +3.00 | +2.80 | -0.70 | -0.60 |
| fplda p4 c1 k3 | +3.60 | +1.50 | +0.00 | -0.40 |
fplda per subject: 4 +8, 15 +8, 20 +4, 3 +3, 14 +2, 6 +2; worst 8 / 19 / 13 at -2 (1-2 images). The gain is
concentrated in a few subjects whose offsets the fingerprint predicts; elsewhere it is neutral. Low downside,
positive mean (+1.2 over E25, +0.9 over k3 across 20 subjects).

## E37. s11 exact testbed, final candidates (n11.py --s11 --cfgs s11c; LDA fitted on the other split's 9 subjects)
| transform | sqf_dbn_things (plain 26.98) | REVE-d6 + SQF-dbn (plain 30.43) |
|---|---|---|
| E25 k2 | +3.98 (19/3) | +4.48 (19/3) |
| k3 | +3.95 | +4.73 |
| k4 | +4.09 | +4.91 |
| fpreg p8 c.75 / c1 | +4.05 / +3.82 | +5.48 / +5.84 (22/0) |
| fplda p4 c1 | **+4.70** | +5.93 (22/0) |
| fplda p8 c1 | +4.61 | **+6.32 (22/0)** |
(REVE alone, E34: E25 +4.41, k3 +4.61, fplda p8 +6.02.) In the exact sealed-like condition (one model, 11 unseen
subjects mixed per batch), LDA-fingerprint regression beats the deployed transform by +1.6-1.8 on every model
tested, never worse on any subject in the ensemble. TSC split runs still training.

## E38. Deployable LDA fit (score_track1.fit_fingerprint_lda: other subjects' TRAINING-image EEG), s11 exact,
REVE-d6 + SQF-dbn (plain 30.43). Implementation now in score_track1.py: eeg_fingerprint (torch, = neighbour_lda
.fingerprint to 1e-6), fit_fingerprint_lda, batch_transform_fp (self-test: c=0 == batch_transform k3).
| transform | delta (better/worse) |
|---|---|
| E25 k2 | +4.48 (19/3) |
| k3 | +4.73 (20/1) |
| fplda p4 / p8 / p12 (LDA on the other split's 9 Alljoined subjects) | +5.77 (22/0) / +5.36 / +5.36 |
| fpldaT p4 / p8 / p12 (+ the 10 THINGS-EEG2 subjects in the LDA) | +6.16 (22/0) / **+6.45 (22/0)** / +6.30 (22/0) |
Fitting the LDA on training-image EEG instead of other subjects' test trials costs ~0.5-1 (E37 p8 +6.32); adding
the THINGS subjects to the LDA buys it back and more: more subjects -> better subject-discriminant directions.
THINGS data therefore also pays off at test time, through the router. For the submission: LDA on all 20 Alljoined
+ 10 THINGS subjects (29 discriminant dims).

## E39. Fingerprint with a 30-60 Hz (EMG) band (s11, REVE-d6 + SQF-dbn): fpldaG p4 / p8 / p12 = +5.48 / +6.14 /
+6.14 vs 4-band fpldaT +6.16 / +6.45 / +6.30 -> no gain, keep the 4 bands.

## E40. cv4 ensemble with the DEPLOYABLE LDA fit (training-image EEG of the other 15 subjects, +THINGS for T);
per-fold delta vs E25 (k3 = +0.50 / +0.30 / +0.30 / +0.10)
| transform | f1 | f2 | f3 | f4 | 20-subject mean vs E25 |
|---|---|---|---|---|---|
| fplda p4 | +1.80 | +1.20 | -0.90 | -0.50 | +0.40 |
| fpldaT p4 | +2.00 | -0.30 | -1.00 | -0.10 | +0.15 |
| fplda p8 | +1.90 | +2.90 | -1.10 | -0.80 | +0.73 |
| fpldaT p8 | +2.70 | +1.60 | -0.70 | -0.60 | +0.75 |
Weaker than with the test-trial fit (E36). Recurrent losers in every fingerprint variant: subjects 8, 13, 19
(-2..-4), the weak/quiet subjects of [[alljoined-subject-quality]]. On cv (N=5) the fingerprint gain is +0.45
over k3; on s11 (N=11, REVE+SQF) +1.7. Full 3-member s11 check running.
Note: fit_fingerprint_lda now caches per-subject training fingerprints (.cache/fp_train_*): its random reads of
train.npy saturated the NAS and starved the GPU jobs (thnorm at 0-14% GPU, process in D state).

## E41. s11 exact testbed, FULL 3-member ensemble REVE-d6 + SQF-dbn + TSC-dbn40 (plain 31.02; deployable LDA fit)
| transform | delta (better/worse) |
|---|---|
| E25 k2 (deployed) | +4.50 (21/1) |
| k3 | +5.09 (21/1) |
| fplda p4 / p8 / p12 | +5.75 (22/0) / +4.82 / +4.82 |
| fpldaT p4 / p8 / p12 | +5.80 / **+5.89 (22/0)** / +5.41 |
The THINGS-augmented LDA router is the best and the most stable across p; the Alljoined-only LDA (9 subjects)
is noisy. fpldaT p8 vs k3: +0.80 here, +0.45 on cv4 (E40).

## E42. Router strength (fpldaT, k3), s11 exact full ensemble (plain 31.02) and cv4 per fold (vs E25)
| transform | s11 delta | cv4 f1 / f2 / f3 / f4 |
|---|---|---|
| k3 | +5.09 | +0.50 / +0.30 / +0.30 / +0.10 |
| p8 c1 | +5.89 (22/0) | +2.60 / +1.10 / -0.20 / -1.00 |
| p8 c.75 | +5.80 (22/0) | +3.50 / +1.20 / +0.10 / -0.60 |
| p8 c.5 | +5.48 (22/0) | +3.00 / +1.00 / +0.20 / -0.60 |
| **p6 c1** | **+5.93 (22/0)** | **+3.20 / +0.50 / +0.40 / -0.30** |
| p10 c.75 | +5.61 (22/0) | +3.10 / +1.70 / -0.50 / -1.10 |
=> candidate: score_track1.batch_transform_fp(P, F, lda, p=6, c=1, k=3) with the LDA fitted on all Alljoined +
THINGS-EEG2 subjects: +1.43 over the deployed batch_transform on s11, +0.95 on cv4 (20 subjects), no fold below
-0.3. Defaults set. End-to-end grader_sim.py --shuffle --fp (official metric, real raw X -> fingerprint inside
the batch loop) running on the interactive GPU.

## E43. End-to-end grader simulation of the router (grader_sim.py --shuffle --k 2 --fp; official metric; real raw
X -> eeg_fingerprint inside the batch loop; LDA per fold on its 15 training subjects + THINGS; REVE-scratch +
SQF-dbn + TSC-dbn40; 5 unseen participants per shuffled 64-batch)
| fold | plain | old transform (k2) | router (p6 c1 k3) |
|---|---|---|---|
| 1 | 36.20 | 39.80 | 42.70 |
| 2 | 32.60 | 36.80 | 38.20 |
| 3 | 34.20 | 40.80 | 39.80 |
| 4 | 31.50 | 37.10 | 38.50 |
| mean (20) | 33.63 | 38.62 (+5.00, 20/0) | **39.80 (+6.17, 20/0)** |
Router vs old transform: +1.18, 13 better / 5 worse. The old-transform column reproduces E25 exactly.

## E44 (fix). Small-batch guard in batch_transform_fp: p is capped at (B-1)//8. Without it, a batch of B <= p+1
trials would have ALL its trial-to-trial variation regressed out (every output = half the batch mean), and B=1
gave NaN (std of one row). B=64 keeps p=6, so E42/E43 are unchanged; self-test now covers B = 1, 2, 8, 16.

## E45. Batch-level whitening with Ledoit-Wolf shrinkage (user question), n11.py --cfgs lwsweep
Sigma = (1-d) S + d nu I on the centred batch outputs (per member); the centred trials lie in span(S), so
whitening rescales eigen-direction i by r_i = (d nu / ((1-d) lam_i + d nu))^(g/2) (normalised to 1 outside the
span; g = 1 full whitening, .5 / .25 partial). Gram-form d matches sklearn.ledoit_wolf (0.8034 = 0.8034) and the
output matches explicit whitening to 4e-6. Half the batch mean kept (a = .5). 15 shrinkage values x 3 strengths.
s11 exact full ensemble (plain 31.02) | cv4 f1-2 / f3-4 (plain 34.15 / 32.90):
| transform | s11 | cv4 |
|---|---|---|
| E25 k2 | +4.50 | +4.35 / +6.45 |
| k3 | +5.09 | +4.75 / +6.65 |
| router p6 k3 | **+5.93** | **+6.20 / +6.50** |
| LW analytic d (per batch), g .25 / .5 / 1 | +1.77 / -7.73 / -25.64 | +2.60/+1.35 ; -8.40/-10.90 ; -27.6/-27.9 |
| best g .25: d .8 | +2.75 | +2.85 / +4.30 |
| best g .5: d .95 | +3.66 | +3.90 / +4.90 |
| best g 1: d .99 | +4.32 | +4.35 / +6.25 |
| d <= .5 at g .5 or 1 | -1.7 .. -28 | -0.6 .. -31 |
| router + LW (best: d .9 g .5) | +3.23 | +4.60 / +2.15 |
Monotone: the more whitening, the worse. Whitening equalises all ~63 in-batch directions, so it amplifies the
small-eigenvalue ones (per-trial noise) relative to the few large ones that carry shared structure; the best LW
setting (d .99, full) only shrinks the very top directions (lam >> 99 nu), i.e. it degenerates into a soft top-k
shrink and lands on E25. The analytic Ledoit-Wolf d is far too small for retrieval: it optimises covariance
estimation, not ranking. => DEAD; the top-k shrink + router stay.

## E46. THINGS input normalisations, folds 1-2 (single REVE-d6 models; plain = best-epoch trial caches, then the
deployed router; cv4 N=5 shuffled batches)
| recipe | plain f1-2 | router f1-2 | per-epoch sealed best vs reve_d6_things (align_curves) |
|---|---|---|---|
| reve_d6_things | 29.80 | 37.60 | |
| reve_d6_coral_things (THINGS recoloured to the Alljoined covariance) | 28.80 | 35.35 | -0.55 |
| reve_d6_car_things (common average reference, train + test) | 29.60 | 36.60 | +0.35 |
Neither helps, plain or under the router. sqf_dbn_coral_things parked (not run). The THINGS gains come through
per-dataset BN + the transform/router (E29, E31, E38), not input harmonisation.

## E47-E48 (2026-10-15 night, user): better person routing. Group each shuffled batch's trials by participant
User idea: an auxiliary network that splits each 64-batch into its ~11 participants, then each group is processed
with its own statistics. Prior facts: hand-crafted fingerprint + LDA gives in-batch neighbour purity 0.50 at N=5
(E21); oracle same-person centring +8.4 vs +5.6 label-free at N=5 (E20). At the sealed N=11 a person has ~6 trials
per batch, so "own statistics" can only be a mean (a 6-trial PCA is noise); the global top-k shrink stays.
- E47 headroom (route_study.py --cfgs oracle, s11 exact, full ensemble REVE-d6 + SQF-dbn + TSC-dbn40, CPU): group
  transform C = X - mu_b - c (W X - mu_b), shrink k3, + .5 mu_b with W = leave-one-out same-person mean (oracle),
  or with purity q (q of the weight on true batch-mates): how accurate must a router be to beat the deployed one?
- E48 learned participant embedding (subject_embed.py, queue route_emb_both / route_emb_fp, GPU ~15 min): AM-softmax
  over the 19 training participants (9 Alljoined + 10 THINGS), fingerprint MLP +/- learned filter-bank log-power;
  scored by in-batch K=4 purity on the 11 unseen participants vs the LDA fingerprint, then plugged into route_study
  --cfgs emb (soft kNN centring, and as the router's regression space).
- E47 result (s11 exact, full ensemble REVE-d6 + SQF-dbn + TSC-dbn40, plain 31.02; router reproduced exactly):
| transform | top-5 | vs plain | vs router |
|---|---|---|---|
| k3 | 36.11 | +5.09 | -0.84 |
| router fpldaT p6 (deployed) | 36.95 | +5.93 | |
| oracle group centring, leave-one-out, c=1 | 38.93 | +7.91 (22/0) | **+1.98 (14/5)** |
| same, purity q = .9 / .7 / .5 / .3 | 38.70 / 38.68 / 38.18 / 37.25 | | +1.75 / +1.73 / +1.23 / +0.30 |
| oracle c=.5 / incl. self / lam 2 / lam 6 | 38.55 / 38.11 / 38.86 / 38.43 | | +1.59 / +1.16 / +1.91 / +1.48 |
  Headroom of perfect person routing at N = 11 is +2.0 over the router (vs +4 at N = 5: smaller groups, noisier
  means). A router needs ~0.5-0.7 of its neighbour weight on true batch-mates to collect most of it; below 0.3 it is
  worth nothing over the regression router.
- E48 (route_emb_both, AM-softmax, fingerprint + learned filter-bank log-power): split 1 in-batch K=4 purity 0.332 vs
  LDA fingerprint 0.287 (N = 11, chance 0.08). Modest; far from the ~0.5 needed.
- E48b queued: route_emb_con, same network, supervised contrastive loss on deployment-like episodes (64 trials from
  11 random training participants). E49 queued (user): route_whiten, neighbour whitening: per trial, k nearest
  batch-mates (cosine in the embedding), centre on their mean and whiten with their Ledoit-Wolf-shrunk covariance
  (shrinkage d in .95 .97 .98 .99 .995 .999 1; k in 4 8 16 32); neighbours from oracle / LDA / both / con; +k3 variants.
  E48 final: purity split 1 / 2 = 0.332 / 0.324 vs LDA 0.287 / 0.301 (mean 0.328 vs 0.294); training loss -> 0.002 = the 19 training identities are memorised; transfer to new people is the limit.
- E48b (route_emb_con, supervised contrastive on 11-person episodes): purity 0.263 / 0.257 vs LDA 0.287 / 0.301: WORSE than LDA and than AM-softmax (0.33). With 19 training identities neither learned embedding transfers much beyond hand-crafted features.
- E48c queued (route_emb_big): the AM-softmax embedder with 69 training identities: + all 50 THINGS-EEG1 subjects
  (eeg1_identity_cache.py re-extracts 4000 training-image trials each from the raw BrainVision files with the export's
  own chain; the sub-31..42 packs were deleted). Hypothesis: 19 identities are too few to learn a person metric that
  transfers (training loss -> 0). route_emb_fp dropped (fingerprint-only is covered by the LDA baseline).
- E49 result (route_whiten2; s11 exact, full ensemble, plain 31.02, router 36.95; neighbour whitening: per trial,
  centre on its k nearest batch-mates' mean and whiten with their Ledoit-Wolf covariance (1-d) S + d nu I, keep .5 mu):
| neighbours | best WITHOUT global shrink | best WITH the global k3 shrink after |
|---|---|---|
| oracle (all true batch-mates) | d .97: 36.91 (-0.05 vs router) | **d 1.0 (no whitening): 38.89 (+1.93)**; d .999 / .99 / .98 / .95: +1.89 / +1.36 / +0.64 / -0.43 |
| LDA fingerprint | K16 d .98: 36.30 (-0.66); K32 d .98: 36.27 | K8 d .999: 37.41 (+0.45, 11/9); d 1.0 36.77; d .99 37.00 |
| learned embedder (both) | K32 d .98: 36.27 (-0.68) | K8 d .999: 36.91 (-0.05) |
  Reading: local whitening at d .97-.98 works only as a substitute for the global top-k shrink (it shrinks the
  neighbourhood's top directions; it reaches k3's 36.1-36.3, below the router). With the global shrink in place,
  every d < 1 hurts monotonically, even with PERFECT neighbours: a person's ~6 batch-mates give a covariance whose
  directions are their own image signal, not the person's nuisance, so whitening removes signal. What helps is
  the neighbours' MEAN (centring), and it is worth +1.9 only when neighbours are the true batch-mates.
  Best label-free: LDA K8 d .999 +k3 +0.45 (11/9) = within noise (picked from 84 configs). The lever stays grouping
  purity (E47: 0.5-0.7 needed; LDA 0.29, learned 0.33); E48c (69 identities) is the remaining attempt.
  E48c: 42 of 50 EEG1 subjects extracted (8 longest recordings OOM in the 3.9 GB CPU step, skipped) -> 61 training identities.
- E48c split 1: purity 0.361 (61 identities) vs 0.332 (19) vs LDA 0.287. More identities help (+0.03). Queued route_emb_eval (kNN centring + as router space, embeddings big / both).
  E48c final: purity 0.361 / 0.352 (mean 0.357) vs 19-identity 0.328 vs LDA 0.294. ~+0.03 per 3x identities.
- E48d (route_emb_eval, s11 exact, full ensemble, router 36.95): routing with the learned embeddings.
| config | top-5 | vs router |
|---|---|---|
| big (61 identities) as the router's regression space, p6 | **37.45** | **+0.50 (12/6)** |
| big kNN K8 centring + k3 | 37.09 | +0.14 (8/11) |
| both (19 identities) kNN K8 | 37.02 | +0.07 |
| both as router p6 | 36.82 | -0.14 |
| K4 / soft tau .05 / lam 4 variants | 35.8-36.7 | -0.2 to -1.1 |
  Purity 0.29 -> 0.36 buys ~+0.5 (E47 curve: +0.3 at 0.3, +1.2 at 0.5). Weak evidence (12/6, one config of 12);
  deploying it needs the embedder per cv4 fold (fold training subjects + THINGS-EEG2 + EEG1) and grader_sim --fp.
  Overall E47-E49 verdict: person routing headroom at N = 11 is +2.0; whitening adds nothing beyond centring; a
  learned identity embedding gets ~1/4 of the headroom. More identities is the only lever that moved purity.
- E50 (route_ldaE, s11 exact, full ensemble): the deployed router with its LDA refitted on 61 identities (+ 42
  THINGS-EEG1): p4 / p6 / p8 = 36.93 / 36.95 / 37.14 vs deployed 36.95 (-0.02 / +0.00 / +0.18, 9-12 better / 9-11
  worse) = null. ROUTING CLOSED (user, 2026-10-16): keep the deployed router (fpldaT p6 c1 k3).

## E51 (2026-10-19, user): per-trial L2 normalisation before the official subject aggregation (trial_norm.py)
The official scorer averages RAW predictions per (participant, image), so a trial's output norm is its vote weight.
cv4, grader_sim --shuffle stream, deployed router, 20 subjects; ref reproduces ens_search exactly (40.05 / 40.60).
| | ref | pre (l2 before router) | mem (l2 each member after router) | ens (l2 of the summed output) | pre_ens |
|---|---|---|---|---|---|
| REVE-d6 | 36.52 | -1.65 (6/11) | -0.65 (8/12) | = mem | -2.10 (2/17) |
| SQF-dbn | 34.40 | -0.57 | -0.52 | = mem | -0.77 |
| TSC-bn64 | 34.17 | -0.45 | -0.43 | = mem | -0.70 |
| rmix | 36.88 | -0.97 | -0.85 | = mem | -1.12 |
| ensemble REVE-d6 + SQF + TSC | 40.05 | -0.35 (6/14) | -0.30 (8/10) | -0.05 (9/11) | -0.45 |
| rmix swap | 40.60 | -0.35 (7/10) | -0.85 (4/14) | -0.75 (2/11) | -0.88 |
DEAD: equal votes per trial never help. After the router, the output norm IS informative as a vote weight (high-norm
trials are the better ones), unlike the THINGS fusion finding (norm ~ AUC 0.5 for score fusion). Keep raw outputs.
