# TTA_LOG: test-time adaptation toward a 42% ensemble (2026-10-20 -)

Goal (user, 2026-10-20): test-time adaptation methods that lift the cv4 ensemble (grader_sim --shuffle --fp,
official subject-aggregated top-5, 20 held-out people, N = 5 per shuffled 64-batch) from 40.05 (REVE-d6 + SQF-dbn
+ TSC-bn64; 40.60 with rmix in place of REVE-d6) to 42. Sealed-like check where it matters: s11 (N = 11).
Constraints: stateless predict() on 64-trial shuffled batches, no labels, no state across calls; 3 x 3080.

## First principles: where does the aggregated score lose accuracy?
The scorer averages RAW predictions over the ~80 repeats of each (person, image) and ranks the L2-normalised mean
against the gallery. So:
1. Zero-mean per-trial noise is averaged away (80 repeats). TTA that only denoises trials buys little, UNLESS the
   noise is heteroscedastic: the scorer's mean is not robust, and we control each trial's scale, i.e. its VOTE
   WEIGHT (Sum w_i p_i). E51: forcing equal weights costs 0.05-2.1, so weights matter; better weights may gain.
2. Any per-trial correction m_i enters the aggregate only through its average over the person's trials: noisy
   per-trial estimates of the person offset are fine, only their BIAS matters. The batch mean (a mix of N people)
   is a biased estimate of each person's own offset; the fingerprint router reduces the bias.
3. What survives averaging is systematic: (a) the person's offset (first order), (b) person-specific distortions
   of the mapping (D1/D5: not linear-correctable from outputs, needs labels), (c) the model's image-level errors
   shared by all people (only better models / ensembles fix these), (d) gallery geometry (hubness).
Plan: measure the headroom of each axis with ORACLES on a per-trial cache (seconds per experiment), then attack the
axes with headroom. Oracles use labels/identities only to bound, never in a deployable method.

## Testbed
- tta_cache.py: per-trial raw member outputs on the exact grader_sim --shuffle stream (cv4, 4 folds x 80000
  trials), fingerprints, fold LDA. tta_lab.py: replays any batch method, official metric.

## Experiments
### T0. Replay check (tta_lab.py ref): 40.05 = ens_search / grader_sim --fp exactly. plain 33.75, k3 39.78.

### T1. Headroom oracles (cv4, ensemble REVE-d6 + SQF-dbn + TSC-bn64, vs ref 40.05)
| oracle | top-5 | delta |
|---|---|---|
| oc3: subtract each person's exact mean output (all their trials), batch k3 shrink, keep .5 batch mean | 42.52 | +2.48 (17/2) |
| + the person's own top-3 directions shrunk by .5 (ocp:3:.5) | 43.33 | +3.28 (17/3) |
| person x session / x run instead of person (oc3:s / :r) | 42.32 / 42.50 | session is NOT the unit; the offset is personal |
| ocb:.7: perfect grouping WITHIN the batch (leave-one-out same-person batch-mates, c .7) | **42.88** | **+2.83 (18/1)** |
| ocb:.7 by session / run within the batch | 41.58 / 40.35 | fewer mates, worse |
| reliability weights, leak-free oracle (1 / distance to own (person, image) mean)^.5 | 40.27 | +0.23: noise is ~homoscedastic |
| (cos to target)^g weights | 73-79 | INVALID: pulls each mean toward the target |
=> All first-order TTA headroom (~+2.8) is reachable statelessly with PERFECT in-batch person grouping; the
   deployed router collects only +0.27 of it over plain k3. The problem is grouping purity, nothing else.

### T2. Vote weights (deployable): |y|^g: g -1 / -.5 / .5 / 1 / 2 -> -0.05 / +0.30 / -0.07 / -0.15 / -1.30;
member agreement^g (g 1 / 2 / 4) -0.30 / -1.07 / -2.55. DEAD (T1: no reliability headroom).

### T3. kNN centring in the router's LDA fingerprint space (knn:K:c): K4 / 8 / 12 c .7 -> +0.70 / +0.70 / +0.78,
softmax tau .1 +0.90, purity@4 0.53-0.66 (chance 0.2). Adding output cosine to the affinity hurts (outputs alone
purity 0.24 = chance): the outputs carry no usable identity.

### T4. WHY the LDA space is poor, and the fix (group_purity.py, purity@4, 4 folds)
The LDA keeps the 24 directions that separate the 25 TRAINING people; a new person differs from others along
directions the LDA never saw. Speaker diarisation (clustering unseen speakers) solves this with within-class
covariance normalisation (WCCN): keep ALL dimensions, whiten by the pooled WITHIN-person covariance, cosine.
| affinity | purity@4 | purity@8 |
|---|---|---|
| raw standardised fingerprint, batch-centred | 0.524 | |
| LDA 24 + batch z-score (= router space) | 0.588 | 0.492 |
| LDA 12 | 0.531 | |
| WCCN shrink .01 / .1 / .3 / .5 / .7 / .9 | 0.691 / 0.729 / **0.736** / 0.729 / 0.713 / 0.674 | .3: 0.612 |
| WCCN .1 euclidean instead of cosine | 0.419 | |

### T5. kNN centring in WCCN space (wknn:K:c, shrink .3), vs ref 40.05
| K, c | top-5 | delta |
|---|---|---|
| 4, .7 | **42.30** | **+2.25 (15/5)** |
| 4, 1 | 41.83 | +1.78 |
| 8, .7 / 8, 1 | 41.80 / 42.03 | +1.75 / +1.98 |
| softmax tau .1, c .7 / 1 | 41.25 / 41.62 | |
First deployable method at 42 on cv4 (N = 5). Checks pending: selection optimism (fold-pair split), N = 11 (s11),
rmix swap, combination with the regression router.

### T5b. Robustness of wknn (cv4, 3-member ensemble, shrink .3)
- Grid K in 3/4/6/8 x c in .5/.7/.85/1: every one of 16 cells +1.25 .. +2.25 (11-16 people better of 20).
- Nested selection (pick on folds 1-2, test 3-4 and vice versa): 4/.7 -> folds 3-4 41.35 vs 38.30 (+3.05);
  8/.85 -> folds 1-2 42.50 vs 41.80 (+0.70). Honest estimate (41.35 + 42.50) / 2 = **41.93 (+1.88)**.
- Per member (ref -> wknn 4/.7 | 4/1 | perfect grouping ocb .7): REVE-d6 36.52 -> +2.27 | +3.02 | +3.55;
  SQF-dbn 34.40 -> +1.83 | +1.43 | +2.18; TSC-bn64 34.17 -> +0.23 | +0.02 | +0.80; rmix 36.88 -> +2.32 | +2.85 |
  +4.35 (rmix + wknn c1 = 39.73, best single). TSC-bn64's bottleneck head carries almost no person offset.
- rmix swap ensemble: ref 40.63 -> wknn 4/.7 41.63 (+1.00), 4/.85 41.55, 8/1 41.45: the 3-member ensemble with
  REVE-d6 is now the better base (42.30).

### T6. Offline statistics for WCCN: Alljoined people only (15) vs + 10 THINGS-EEG2 (other hardware)
purity@4 wccn .3: 0.742 vs 0.736 (.1: 0.736 vs 0.729); LDA24 needs the identities (0.522 vs 0.588). WCCN only needs
the within-person covariance of the SAME hardware -> deploy with the 20 Alljoined people; extra THINGS-EEG1
identities (other hardware) not worth adding. Harness default switched to Alljoined-only.
Scoring alternatives on the base fingerprint: PLDA (between-person covariance shrunk to isotropic .1 / .5 / .9)
0.708 / 0.674 / 0.611; shared-nearest-neighbour affinity K2 6 / 10: 0.735 / 0.721; euclidean 0.419. WCCN cosine wins.

### T7. Sealed-like N = 11 (s11 testbed: 2 splits x 11 unseen people, models trained on 9 + THINGS; tta_cache.py
--set s11; ensemble REVE-d6 + SQF-dbn + TSC-dbn40; NOTE this replay's ref 34.98 differs from n11.py's s11 36.95,
a different pipeline: compare only within this testbed). THINGS still in the WCCN statistics here.
| method | top-5 | delta |
|---|---|---|
| ref | 34.98 | |
| k3 | 34.63 | -0.35 |
| perfect in-batch grouping ocb .7 | 37.70 | +2.73 (15/4) |
| wknn 3/.7 , 4/.7 , 4/1 , 6/.7 | 35.90 , 36.15 , **36.23** , 35.82 | +0.93 , +1.18 , **+1.25 (13/6)** , +0.85 |
Purity@K3/4/6/8 at N = 11: 0.49 / 0.44 / 0.37 / 0.32 (vs ~0.74 at N = 5 K4). wknn transfers to N = 11 but collects
46% of the headroom (vs 80% at N = 5): purity is THE lever for the sealed condition.

### T8. Combinations on cv4 (3-member ensemble unless noted; WCCN statistics now Alljoined-only)
| method | top-5 | delta vs 40.05 |
|---|---|---|
| wknn 4/.7, Alljoined-only statistics | 42.15 | +2.10 (15/5) (vs 42.30 with THINGS in Sigma_w: same within noise) |
| + regress the top 3 / 6 WCCN fingerprint PCs after kNN centring (x1) / 6 x .5 | 41.75 / 41.67 / 41.85 | worse: kNN already takes the offset |
| per-member c (REVE, SQF, TSC) = 1,.7,0 / 1,.7,.3 / 1,1,0 / .85,.7,.3 / 1,.85,.5; K3 1,.7,.3; K6 1,.85,.3 | 41.97 / 41.93 / 41.88 / 41.90 / 41.88 / 41.95 / 42.30 | no gain over one c: dropped |
| 4 members (REVE-d6 + rmix + SQF + TSC): ref 40.37; wknn 4/.7 / 4/.85 / 4/1 | 42.18 / 42.20 / 42.10 | +1.8: no gain over 3 members |

### T9. Richer identity features (group_features.py; WCCN .3 on the concatenation, Alljoined-only statistics, purity@4 cv4)
| features | purity@4 |
|---|---|
| base (deployed eeg_fingerprint: log var, correlations, 4 band powers; 656) | 0.742 |
| fine (2-Hz log power 1-45 Hz) / calpha / erp alone | 0.639 / 0.476 / 0.343 |
| base + fine / + calpha / + cbeta / + erp | 0.744 / 0.719 / **0.774** / 0.729 |
| base + fine + calpha + cbeta / + erp | 0.779 / 0.781 |
Band-limited (beta) spatial correlations add identity; the evoked waveform does not. Next: delta / theta / gamma
correlations and gamma (EMG) power; score the best set with tta_lab (wknng).
- T9 scored (cv4, wknng = wknn on the feature concatenation): base 4/.7 42.15; base+cbeta 4/.7 41.82;
  base+fine+calpha+cbeta 4/.7 **42.40 (+2.35, 18/2)**, 4/.85 41.93, 6/.85 41.97. Purity +0.04 -> score +-0.3:
  within the paired noise at 20 people (1 image = 0.5 per person); decide on s11 where purity binds.

### T10. Deployable implementation: score_track1.fit_fingerprint_wccn (offline: Alljoined people's training-image
fingerprint caches -> m0, sd, WCCN matrix T, 656 x 656) + batch_transform_wknn(Ps, F, wccn, K, c) (stateless, list of
member outputs, shared neighbours). tta_lab `deploy:4:0.7` = **42.15**, identical to the replay wknn:4:0.7.

### T11. Same- vs different-person WCCN cosine in a 64-batch (fold / split 1): same median .06 (IQR -.01..+.16),
different -.04 (-.08..+.01) at N = 5; the SAME distributions at N = 11, only the prior changes (pairs same 20% ->
9%): P(same | S > .15) = .82 at N = 5 vs .53 at N = 11. A fixed K over-trusts neighbours at N = 11. ->
bayes:beta:theta:kappa: w_ij = exp(beta (S_ij - theta)), person mean = (sum w q + kappa mu_batch) / (sum w + kappa):
evidence-weighted, N-agnostic. Grid running on cv4 + s11.
- T11 result: bayes is WORSE than plain kNN on both testbeds. cv4 (ref 40.05; wknn 4/.7 42.15): bayes beta 20
  +1.25..+1.35, beta 40 +0.63..+0.88. s11 (ref 34.98; wknn 4/.7 **36.50 (+1.52)** with Alljoined-only statistics,
  vs 36.15 with THINGS in Sigma_w): bayes beta 20 +0.87..+1.15, beta 40 +0.17..+0.55. Sharper weights are worse:
  exp weights put the mass on 1-2 top mates; an even average over K ~ 4 is the better estimator. Dropped.

### T12. In-batch kernel ridge (GP) regression of outputs on the fingerprint, exact leave-one-out (gp:beta:lam:c),
cv4: beta 2 lam .1 / 1: +1.23 / +0.58; beta 5: -0.02 / -0.20; beta 10: -0.25; beta 5 lam .1 c 1: +0.23 (wknn 4/.7
+2.10). The GP weights decorrelate redundant neighbours and spread mass by kernel width; the hard K = 4 average still
wins. Dropped (with T11: the simple even K-mean is the best person-mean estimator found).

### T9b. Richer features at N = 11 (s11, Alljoined-only statistics, ref 34.98): wknn base 3/.7 36.02, 4/.7 36.50,
5/.7 36.25; base+cbeta 4/.7 36.12; base+fine+calpha+cbeta 4/.7 **36.58 (+1.60, 14/3)**, 3/.7 36.30.
Purity +0.04 buys ~+0.1-0.25 on both testbeds: the purity -> gain map is ~linear (cv4: 0.74 -> +2.1, 1.0 -> +2.8),
so the last purity points are cheap in score. Features are a secondary lever now.
- T9c more bands (purity@4 cv4): alone cdelta .363, ctheta .463, cgamma .552, pgamma (30-58 Hz power) .568;
  base + cgamma .758 / + pgamma .769 / + ctheta .711 / + cbeta + cgamma .779; base+fine+calpha+cbeta+cgamma+pgamma
  **0.794** (best); + cdelta + ctheta 0.778 (slow bands hurt). Gamma = EMG / electrode contact: identity-bearing.
- T9d best feature set scored (rich = base+fine+calpha+cbeta+cgamma+pgamma): cv4 4/.7 42.12 (+2.08, **19/0**), 5/.7
  41.97, 4/.85 41.88; s11 4/.7 36.52 (+1.55, 13/2), 5/.7 36.42, 6/.7 36.30. Same mean as base (42.15 / 36.50) but
  more consistent per person (19/0 vs 15/5 on cv4; 13/2 vs 14/5 on s11).

### T13. Ensemble selection UNDER the transform (tta_ens.py: per-member (person, image) means cached, subsets scored
exactly; 10 members: REVE-d6, SQF-dbn, TSC-bn64, rmix, REVE-d6-KD, SQF-dbn-KD, REVE-scratch, LaBraM-s10, rmix-reg,
rmix-d8; NESTED = best subset picked on folds 1-2 scored on 3-4 and vice versa, averaged)
| transform | size | best on all 20 | nested |
|---|---|---|---|
| ref (deployed router) | 3 / 4 | 40.73 / 40.90 | 40.38 / 39.73 |
| wknn 4/.7 | 3 | 42.35 (SQF + REVE-d6-KD + rmix-d8) | 41.62 |
| wknn 4/.7 | 4 | 42.58 (REVE-d6 + SQF + SQF-KD + REVE-scratch) | 41.80 |
| wknn 4/.7 | 5 | **42.60** (REVE-d6 + SQF + TSC-bn64 + REVE-d6-KD + REVE-scratch) | **42.30** |
**GOAL MET on cv4 with an honest (nested) estimate: 42.30 for a 5-member ensemble under wknn**, vs 40.05 for the
deployed 3-member ensemble + router (+2.25). Under the router, the old 3-member choice was already ~optimal (nested
40.38); under wknn, more REVE-type members (large person offsets, T5b) pay off.

### T14. Model-side test-time augmentation: average each member over the input shifted by -2..+2 samples
(edge-padded; tta_cache.py --shifts, members '<recipe>@tta'), cv4 3-member: ref 40.05 -> 39.88; wknn 4/.7 42.15
-> 41.50. HURTS: the members are latency-locked (unseen shifts move the evoked response off its learned timing),
and per-trial denoising is averaged away by the scorer's 80 repeats anyway (first principles 1). Dropped.

## Summary (2026-10-20)
- The whole first-order TTA headroom is in-batch PERSON GROUPING (T1). The fix was the affinity space: within-person
  whitening (WCCN) of the full fingerprint instead of an LDA fitted to the training people (T4).
- Deployable `batch_transform_wknn` (T10): cv4 3-member 40.05 -> 42.15 (nested ~41.9); s11 (N = 11) 34.98 -> 36.50.
- With ensemble re-selection under the transform (T13): nested 5-member **42.30**, best on all 20 people 42.60.
- Dead: vote weights, Bayesian / GP / PLDA / SNN variants, per-member strengths, regression after kNN, time-shift
  TTA, THINGS in the statistics. Richer identity features: purity 0.74 -> 0.79, score +-0 on average, but more
  consistent per person (19/0).
- Not yet done (needs approval): switching the sealed submission from batch_transform_fp to batch_transform_wknn and
  the 5-member ensemble; an end-to-end grader_sim run of the new path with the live models.

### T15. Re-tune the global steps under wknn 4/.7 (wknnx:K:c:k:b:a; cv4; default k3 b.5 a.5 = 42.15)
| k (top batch dirs shrunk) | 0 | 1 | 2 | **3** | 4 | 6 |
|---|---|---|---|---|---|---|
| top-5 | 36.37 | 40.27 | 41.48 | **42.15** | 41.60 | 41.60 |
b (shrink) .3 / .5 / .7 / 1: 39.95 / **42.15** / 41.73 / 33.77; a (batch-mean removed) .3 / .5 / .7 / 1: 39.70 /
**42.15** / 40.90 / 36.08. The router-era defaults stay optimal and the optimum is sharp: the top 3 batch directions
carry both nuisance and signal (removing them fully costs -6.3, ignoring them -3.7).

### T16. In-batch person SECOND order, oracle grouping (ocbp:c:kp:bp: after perfect in-batch centring, shrink each
person's top-kp directions estimated from their ~13 batch-mates): kp1 b.5 42.75 / kp1 b1 39.07 / kp2 b.5 42.25 /
kp3 b.5 41.20 vs ocb .7 42.88. Even with PERFECT grouping it never helps: ~13 trials' top directions are image signal,
not the person's nuisance (cf. E49). The full-data +0.8 of T1 (ocp) is unreachable statelessly. Second order closed.

### T17. END-TO-END check with the live models (grader_sim.py --shuffle --fp, new 'wknn' output = score_track1.
batch_transform_wknn, WCCN fitted on the fold's 15 Alljoined training people; real raw EEG, official metric)
| ensemble | plain | k2 | router (fp) | **wknn** | wknn - fp |
|---|---|---|---|---|---|
| REVE-d6 + SQF-dbn + TSC-bn64 | 33.73 | 39.20 | 40.05 | **42.15** | +2.10 (15/5) |
| + REVE-d6-KD + REVE-scratch (5) | 33.70 | 38.75 | 40.65 | **42.60** | +1.95 (14/4) |
Identical to the cache replay (T5, T13): the deployable path is verified with live models. wknn - plain: +8.4 / +8.9,
20/20 people better.
- T13b larger ensembles under wknn 4/.7 (nested): size 6 **42.30**, size 7 **42.40** (best on all 20: 42.77 =
  SQF + TSC-bn64 + rmix + REVE-d6-KD + SQF-KD + rmix-reg + rmix-d8). The honest estimate plateaus at 42.3-42.4 from
  5 members on.
- T13c ensemble selection under wknng 4/.7 with the RICH identity features (base+fine+calpha+cbeta+cgamma+pgamma):
| size | best on all 20 | nested |
|---|---|---|
| 3 | 42.58 | 41.85 |
| 4 | 42.88 | 42.08 |
| **5** | **43.05** (REVE-d6 + SQF + REVE-d6-KD + SQF-KD + rmix-reg) | **42.65** |
| 6 | 42.85 | 42.25 |
Best honest estimate of the whole study: **42.65** (vs 42.30 with the base fingerprint, 40.38 under the router).
Rich features are verified in the replay only; the live end-to-end check (T17) covers the base fingerprint path.

## Final state (2026-10-20)
| | cv4 top-5 | how measured |
|---|---|---|
| deployed: 3 members + router | 40.05 | live grader_sim |
| 3 members + wknn (base fingerprint) | 42.15 | live grader_sim (= replay); nested K/c selection ~41.9 |
| 5 members + wknn (base) | 42.60 | live grader_sim; nested subset selection 42.30 |
| 5 members + wknn (rich features) | 43.05 | replay; nested 42.65 |
| s11 (N = 11) 3 members: router -> wknn base -> rich | 34.98 -> 36.50 -> 36.52 | replay |
Open (needs approval): port the rich features to torch in score_track1 and run the live check; switch the sealed
submission to batch_transform_wknn with the chosen ensemble.

### T18. Rich features ported to torch (score_track1.identity_features + fit_fingerprint_wccn(groups=RICH); exact
match to the numpy caches, max diff 0) and checked LIVE (grader_sim --shuffle --fp, output 'wknnr'):
REVE-d6 + SQF-dbn + REVE-d6-KD + SQF-dbn-KD + rmix-reg: plain 33.75, k2 38.88, router 40.75, wknn (base) 42.47,
**wknnr (rich) 43.05** (= replay T13c), wknnr - router +2.30 (17/3). Subset picked on all 20; nested estimate 42.65.
All numbers in the final-state table are now verified with the live models.

### T19. Batch-size robustness (TTA_BS replay of the cv4 stream, 3-member ensemble; rich = T9 features)
| B | router (ref) | wknn K4 | rich K4 | scaled K = B/16: base / rich |
|---|---|---|---|---|
| 16 | 38.32 | +0.47 (12/4) | +0.52 | K1: -0.35 / -0.80 |
| 32 | 38.82 | +1.60 (16/3) | +1.50 | K2: +1.22 / +1.82 |
| 64 | 40.05 | +2.10 (15/5) | +2.08 | - |
| 128 | 40.52 | +1.68 (13/5) | **+2.35** | K8: +1.93 / +2.00 |
| 256 | 41.05 | +0.73 (12/6) | +1.38 | K16: +1.20 / +1.58 |
wknn beats the router at EVERY batch size; fixed K = 4 is the robust default (scaling K helps only at 256 and hurts
at 16). Rich features are the more robust variant at large batches. Small batches (16) leave ~3 same-person mates:
gain shrinks to +0.5 but stays positive.

### T20 (running). Model-internal identity for grouping (tta_backbone.py): the members' outputs carry no identity
(T3), but intermediate layers might ("identity trap"). Cached per fold for the stream + 1000 training-image trials of
each of the fold's 15 training people: TSC-bn64 input of the 1x1 filter-mixing conv (1440; D5's per-person map lives
there), REVE-d6 and SQF-dbn backbone outputs. Purity alone and late-fused with the fingerprint (S_fp + lam S_bb).
- T20 result (purity@4 cv4, WCCN .3 on each layer with the fold's training people; fusion S_fp(base) + lam S_bb):
| layer | alone | lam .5 | lam 1 | lam 2 |
|---|---|---|---|---|
| TSC-bn64 filter-mixing input (1440) | 0.298 | 0.735 | 0.705 | 0.622 |
| REVE-d6 backbone (1536) | 0.416 | **0.754** | 0.718 | 0.632 |
| SQF-dbn backbone (1536) | 0.336 | 0.707 | 0.625 | 0.514 |
(fingerprint alone 0.742.) The trained members are largely person-INVARIANT inside too (cf. memory: identity trap
inverted in this repo); the raw-EEG fingerprint is the identity carrier. Best fusion +0.012 purity = ~+0.1 score
(T9b map). Closed.

## Status (2026-10-20, end of session)
Verified with live models (grader_sim --shuffle --fp): 3-member wknn 42.15; 5-member wknn 42.60; 5-member rich
wknnr 43.05. Nested (selection-honest) estimates: 3-member K/c ~41.9; 5-member base 42.30; 5-member rich 42.65;
7-member base 42.40. Robust at batch sizes 16-256 (T19) and N = 11 (T7/T9b). Remaining in-batch headroom at N = 5:
~0.7 (perfect grouping 42.88 vs 42.15). Exhausted levers: see T2, T8, T11-T16, T20.

### T21. Last estimator variants for the 3-member ensemble (cv4, B = 64) + clean nested selection
- mutual-kNN weights (mknn:K:c:w_nonmutual): K4 .5 / .25: 42.18 / 42.15; K6 .5 / .25: 42.10 / 42.27; rich K4 .5 42.18
  (19/1), K6 .25 42.05.
- adaptive K (thr:theta:kmin:kmax: even mean over mates with S > theta, kmin..kmax): base theta .15 2-8 / 2-12
  42.35 / 42.37, theta .1 41.95-42.07, .2 41.63-42.02; rich 41.50-42.18.
- rich grid K 3-6 x c .55/.7/.85: 41.57-42.12 (4/.7 42.12 19/0, 4/.55 42.05 17/0).
- BUG FOUND + FIXED: the T19 batch-size sweep wrote its JSON results under the B=64 names (wknn:4/8/16, wknng:4/8/16,
  wknn(g):1/2). tta_lab now suffixes __bs<B>; the overwritten B=64 configs were rerun (wknn 4/.7 42.15, 8/.7 41.55,
  16/.7 41.30; rich 4/.7 42.12, 8/.7 41.77, 16/.7 41.22) and the sweep-only files removed. T5b and T13 predate the
  sweep / use separate caches: unaffected.
- Clean nested selection over ALL 78 deployable 3-member configs: pick on folds 1-2 (thr .1 3-8 rich) -> 3-4 40.75
  (ref 38.30); pick on 3-4 (mknn 6 .7 .25) -> 1-2 42.75 (ref 41.80): **nested 41.75**. Config mean 41.76, median
  41.88, 22 of 78 >= 42. Honest ceiling of the 3-member ensemble ~41.8-42.0; the robust >= 42 comes from re-selecting
  the ensemble under the transform (T13: nested 42.30 base, 42.65 rich, clean caches).

### T22. SELECTION-FREE ensemble: all 10 cached members, transform fixed in advance (wknn K4 c.7 = the first config
tried, T5, before any result), no subset search (tta_ens.py --sizes 10)
| transform | all 10 members |
|---|---|
| ref (deployed router) | 40.30 |
| **wknn 4/.7 (base fingerprint)** | **42.33** |
| wknng 4/.7 (rich features) | 42.25 |
(For completeness, nested subset selection at sizes 8 / 9: base 42.10 / 42.20; rich 42.05 / 42.30.)
=> A >= 42 ensemble that involves NO choice made on the 20 evaluation people: 42.33 (+2.03 over the same members
under the router). Together with the nested 5-member estimates (42.30 base, 42.65 rich) and the live-model checks
(T17, T18: replay == live exactly), the 42% target is established on cv4.
- T22 LIVE (grader_sim --shuffle --fp, all 10 members, real models and raw EEG): plain 33.25, k2 38.12, router 40.30,
  **wknn 42.33** (+2.03 vs router, 15/5), wknnr 42.25 (+1.95, 18/2). Identical to the replay. The selection-free
  >= 42 ensemble is verified end to end.

# Round 2 (user, 2026-10-21): SIMPLIFY (still > 42) or PUSH (-> 45), no member selection
Known: kNN centring alone, without the global shrink, is bad (k = 0: 36.37, T15): the shrink must be replaced, not
dropped. Queued:
- T23 lightweight post-steps (tta_lab wlw): `lw:d` = centred Ledoit-Wolf whitening of the person-centred batch
  (replaces "top-3 x .5"), + .5 mu; `lwu:d` = ONE op: person offsets removed, batch mean KEPT, uncentred LW shrink
  (its second moment contains the mean direction, so it shrinks the mean too: replaces k, b AND a by one d).
  lw_shrink verified against explicit LW whitening. d .9 / .95 / .98 / .99 / .995; WCCN fingerprint shrink .9 / .99.
- T24 tangent-space fingerprint (group_features 'tangent'): logm of the trial's 32 x 32 covariance, 528-d.
- T25 self-calibrated adaptive K (zknn:z:c): mates with S > row median + z MAD, 2..12.
- T26 ARM-lite (tta_arm.py): affinity A = I + U V^T learned end to end through the centring operator on the OTHER
  folds' held-out streams (fingerprints only: cannot memorise the shared gallery).
- T27 temporal-contrastive embedder (tta_ssl.py): positives = same person+session within 60 s, negatives = other
  people; conv encoder on raw EEG; purity on held-out people vs WCCN.

### T23 result: the global shrink can be ONE uncentred Ledoit-Wolf whitening (3-member ensemble, K4 c.7)
| post-step after kNN centring | cv4 | s11 |
|---|---|---|
| current: shrink top-3 x .5 + keep .5 mu (k, b, a) | 42.15 | 36.50 |
| lwu d .9 / .95 / .98 / .99 / .995 (uncentred LW, mean kept) | **42.40** / **42.35** / 41.20 / 40.33 / 39.12 | 35.90 / **36.23** / 36.15 / 35.30 / 34.22 |
| lw d .9 / .95 / .98 / .99 / .995 (centred LW + .5 mu) | 30.73 / 36.95 / 41.17 / 41.62 / 40.88 | 27.12 / 32.63 / 35.70 / 35.48 / 35.35 |
| lwu .98 / .99 with c 1 | 41.02 / 39.85 | 36.02 / 35.20 |
| WCCN fingerprint shrink .9 / .99 (instead of .3), current post-step | 41.90 / 41.05 | 36.08 / 35.65 |
The uncentred version works because one shrinkage whitening of the raw second moment shrinks the dominant shared
directions AND the batch mean at once; the centred one must still be told how much mean to keep. Protocol: 5 knobs
(K, c, k, b, a) -> 3 (K, c, d), cv4 42.15 -> 42.35-42.40 (s11 36.50 -> 36.23). Near-isotropic fingerprint whitening
(shrink .99) loses 1.1 on cv4: the within-person whitening is essential. zknn (self-calibrated adaptive K) on s11:
z 1.5 / 2 / 2.5 / 3: 36.27 / 36.10 / 36.25 / 35.77 vs K4 36.50: no gain (cv4 pending).

### T23b. Finer lwu grid (K4; d x c; 3-member ensemble)
| d \ c | .5 | .7 | .85 | 1 |  (cv4 / s11) |
|---|---|---|---|---|---|
| .85 | 41.42 / 35.38 | 42.12 / 35.70 | 42.30 / 35.60 | 42.30 / 35.93 | |
| .90 | 41.50 / 35.50 | 42.40 / 35.90 | 42.50 / 36.18 | 42.43 / 36.17 | |
| .93 | 41.60 / 35.73 | 42.12 / 36.05 | **42.55 / 36.45** | 42.42 / 36.27 | |
| .95 | 41.60 / 36.10 | 42.35 / 36.23 | 42.10 / 36.52 | 42.15 / 36.20 | |
Plateau: d .9-.95 x c .85-1 all within 42.1-42.55 / 35.9-36.5 (current k/b/a pipeline: 42.15 / 36.50). Pick
**lwu d .93, c .85 (K4)**: 3 knobs, +0.40 cv4, -0.05 s11. Simplification reached; no push toward 45.

### T25 result: self-calibrated adaptive K (zknn:z:c .7) cv4: z 1.5 / 2 / 2.5 / 3 -> 41.45 / 42.27 / 41.65 / 41.60
(K4 42.15); s11 no gain either (T23). Purity rises with z (0.78 -> 0.83 at z 3) but fewer mates: the K-vs-purity
trade-off is already optimal at fixed K4. DEAD.

### T26 result: ARM-lite (learned affinity A = I + U V^T, rank 32, trained on the other folds' streams) cv4:
arm:4:.7 / arm:4:.85 / arms(learned soft W, learned c) / arms c.7 -> 40.33 / 40.00 / 40.15 / 39.75 vs wknn 4/.7 42.15.
Training loss barely moved (5.21 -> 5.17 per trial, all folds); learned c -> .50, tau .05 -> .044. The per-trial
InfoNCE signal through the centring operator is too weak to learn a 656 x 32 map from 3 x 60 k trials, and what it
learns does not transfer across people (it overwrites the WCCN geometry). DEAD.

### T27 result: temporal-contrastive identity embedder (tta_ssl.py, fold 1, 15 training people x 4000 trials, dt 60 s,
4000 steps): train loss 4.78 -> 0.05-0.13 (training people trivially separated), held-out stream purity@4 **0.613**
vs WCCN base fingerprint 0.772 (rich 0.79). A learned identity embedder from 15 people overfits them; the
hand-made second-order fingerprint + within-person whitening generalises better. Not run on the other folds: even
perfect grouping (ocb .7 42.88) leaves only +0.3-0.7 over lwu/wknn, so purity is no longer the lever. DEAD.

### T28. Gallery geometry (stateless, training-image DINOv2 bank only: 16540 targets; cos(mean train, mean test) .997)
Post-steps on the final output (tta_lab 'method|post', TTA_BANK), base = lwu .9 4/.7 (42.40):
- gm:a (remove fraction a of the mean-target direction): a .25 / .5 / .75 / 1 -> 42.30 / 42.30 / 42.23 / 41.95. No gain:
  the lwu shrink already removes the shared mean direction.
- kr:tau:b (pull toward softmax-nearest training targets): 5-16 (adds the bank's mean = massive hubness); centred bank
  krc (base lwu .93 4/.85, 42.55): b .1 / .25 / .5 -> 39.48 / 31.65-32.88 / 19.15. DEAD: the members' outputs are
  not on the target manifold and should not be pulled there. Gallery axis closed.

### T29. THE MEMBER-SCALING RESULT: one-op lwu on the SELECTION-FREE 10-member ensemble (T22 members, K4)
| | 3 members | all 10 |
|---|---|---|
| router (deployed) | 40.05 | 40.30 |
| plain k3 (no person centring) | 39.78 | 38.32 |
| lwu .93 only (c 0, no person centring) | 40.33 | 39.52 |
| wknn 4/.7 (current: k3 b.5 a.5) | 42.15 | 42.33 |
| **lwu .93, 4/.85 (base fingerprint)** | 42.55 | **44.02 (+3.73 vs router, 18/1)** |
| **lwu .93, 4/.85, rich fingerprint** | 42.25 | **44.53 (+4.23, 19/1)** |
d, c were picked on the 3-member grid (T23b), BEFORE the 10-member run: no 10-member tuning, no member choice.
Ablation: at 10 members the whitening alone is +1.2 over k3 (39.52 vs 38.32; +0.55 at 3), and person centring adds
~+4.5 on top under both post-steps. Reading: the fixed top-3 shrink is tuned to a 3-member sum; with 10 members (more
REVE-type outputs with large shared directions) a spectrum-adaptive shrink per member is what lets the ensemble add
up. Deployable: score_track1.batch_transform_wknn(..., c=.85, d=.93) (lw_shrink == lab version exactly); live
grader_sim outputs 'wknnl' / 'wknnrl' (job 63322, pending). Robustness grid on 10 members: job 63320 (pending).

### T29b. Robustness of the 10-member one-op result (replay, cv4)
- 10-member grid (no tuning used for the headline): lwu d .9 / .95 x c .7 / 1 -> 43.20 / 44.33 / 43.45 / 43.85; d .93 c .85
  K3 / K6 -> 44.00 / 43.50; rich fingerprint d .9 / .95 c .85 -> 44.65 / 44.45, d .93 c 1 -> 44.32. Plateau 43.2-44.65.
- Batch size (TTA_BS, all 10, router / wknn 4/.7 / lwu .93 4/.85): B 32 39.17 / 40.82 / 41.77; B 64 40.30 / 42.33 /
  44.02; B 128 40.67 / 43.25 / **44.98 (20/0)**. One-op > current at every B.

### T30. Joint vs per-member shrink (wlw:lwj: ONE uncentred LW shrink of the members' concatenated outputs, then sum)
10 members: d .9 / .93 / .95 / .98 -> 43.50 / 43.53 / 43.77 / 44.12 (per-member lwu .93: 44.02); 3 members d .93 / .95
42.80 / 42.80 (per-member 42.55). Equivalent within noise; per-member stays (no cross-member choice of d scale).

### T24 result: tangent-space fingerprint (logm of the trial's 32 x 32 covariance, 528-d, WCCN .3) under lwu .93 4/.85
| fingerprint | 3 members | 10 members |
|---|---|---|
| base (656: log-var + correlations + band powers) | 42.55 | 44.02 |
| tangent alone | 42.25 | 43.48 |
| base + tangent | 42.25 | 43.60 |
The Riemannian fingerprint is a cleaner single feature but loses 0.3-0.5; it adds nothing to base. Not adopted.

### T31. Equal votes after the shrink (lwun: each member's lwu output re-scaled to unit mean norm before the sum)
10 members d .93 / .9 -> 44.18 / 43.95, rich 44.58; 3 members 42.32. = lwu within noise. Not adopted.

### T29 LIVE (grader_sim --shuffle --fp, all 10 members, real models + raw EEG, job 63322)
plain 33.25, k2 38.12, router 40.30, wknn 42.33, wknnr 42.25, **wknnl (one-op, base fingerprint) 44.02** (+3.73 vs
router, 18/1), **wknnrl (one-op, rich fingerprint) 44.52** (+4.23, 19/1). Replay = live (44.02 / 44.53).

## Round 2 conclusion (2026-10-21)
SIMPLER AND BETTER: deploy candidate = all 10 cached members (no selection) + batch_transform_wknn(c=.85, d=.93):
K-nearest batch-mates in WCCN fingerprint space -> subtract c x their offset -> ONE uncentred Ledoit-Wolf shrink per
member -> sum. Knobs 5 -> 3; live 42.33 -> 44.02 (rich fingerprint 44.52). Robust: 10-member grid 43.2-44.65, wins
at B 32 / 64 / 128 (41.77 / 44.02 / 44.98). 45 at B 64 NOT reached.
Dead this round: centred LW, near-isotropic WCCN (.9/.99), adaptive K (zknn), ARM-lite, temporal-contrastive embedder
(purity .61 vs .77), tangent fingerprint (-0.3/-0.5), gallery post-steps (gm, kr, krc), joint-member shrink, equal votes.
Caveats: d, c picked on the 3-member cv4 grid (same 20 people); no N = 11 check of the 10-member ensemble (s11 has 3
members cached; at 3 members lwu .93/.85 = 36.45 vs current 36.50).

# Round 3 (user, 2026-10-21): <= 5 members (3 > 5 > 10 at equal score), small LEARNED fingerprint, fewer knobs, 44-45
### T32. WHY 10 members scale under the one-op shrink (cv4, K4 c.85 d.93)
| | 3 members | 5 (one per family: REVE-d6, SQF, TSC, rmix, LaBraM) | 10 |
|---|---|---|---|
| wknn 4/.7 (k3 b.5 a.5) | 42.15 | 41.80 | 42.33 |
| lwu per member | 42.55 | 42.82 (rich 43.40) | 44.02 |
| lws: ONE shrink of the SUMMED (person-centred) output | **43.10** | | 44.15 |
| oracle grouping + lwu (ocbl .85 .93) | 43.45 | 44.73 | 45.38 |
Single members (plain / k3 / wknn / lwu): REVE-d6 28.85/34.55/38.73/39.05; SQF 30.60/34.03/35.73/33.98; TSC
30.05/34.18/34.67/34.58; rmix 30.00/34.77/38.98/39.27; **REVE-d6-KD 33.38/38.78/40.73/42.68**; SQF-KD
31.65/35.38/36.78/38.22; REVE-scratch 29.40/34.57/38.70/39.20; LaBraM 28.98/33.60/37.85/38.65; rmix-reg
30.50/34.95/39.83/40.23; rmix-d8 29.38/34.38/39.23/39.48.
Readings: (1) the member-count effect is NOT grouping (oracle grouping at 3 members caps at 43.45): it is ensemble
averaging of image-level errors, which the fixed k3/b/a post-step blocked (10 members under wknn = 3 members).
(2) the shrink belongs on the ENSEMBLE output (lws >= lwu, +0.55 at 3 members): the hub / shared directions add
coherently across members, the image-specific errors do not, so the right shrink depends on the sum's spectrum.
(3) KD members (distilled from the ensemble at training time) carry most of the ensemble gain in ONE model: REVE-d6-KD
alone 42.68 under lwu (vs 40.73 wknn). Under the router KD looked solo-only (memory); under the shrink it is not.

### T33. Fingerprint ablation (lwu .93 4/.85, 3 base members; columns of the 656-d base fingerprint, WCCN .3)
| fingerprint | dims | purity@4 (fold 1) | cv4 |
|---|---|---|---|
| base (log var + correlations + 4 band powers) | 656 | .772 | 42.55 |
| log variance | 32 | .497 | 40.73 |
| log var + correlations (= spatial covariance) | 528 | .673 | 41.88 |
| **band powers (32 ch x 1-4/4-8/8-13/13-30 Hz log power)** | **128** | .735 | **42.70** |
| log var + band powers | 160 | .732 | 42.47 |
Identity lives in the per-channel POWER SPECTRUM, not in the spatial correlations. 5 x smaller fingerprint, same
score. With lws: 3 members 43.10 (base) / 42.65 (bands); 5 family members 43.17 / 43.40.

### T34. KD members as the small ensemble (lws .93 4/.85; KD = distilled from the fold's 3 base members, A18)
| members | ref | lws | rich fp | oracle grouping |
|---|---|---|---|---|
| REVE-d6-KD alone | 33.38 (plain) | 42.68 | | 43.65 |
| REVE-d6-KD + SQF-KD | 39.30 | 42.10 | 42.37 | 43.43 |
| REVE-d6-KD + SQF-KD + TSC-bn64 | 38.12 | 41.22 | 41.60 | 42.40 |
| **REVE-d6-KD + SQF-KD + rmix-reg** | 40.55 | **44.12 (18/2)** | 44.12 | 45.77 |
3 members in the 44-45 band. Caveat: the trio was picked AFTER the single-member table (T32) = strongest member of
the REVE / SQF / rmix families; a mild selection on the same 20 people. Two KD members share a teacher (correlated
errors: the pair is below the single); rmix-reg is the decorrelated third. Robustness grid + live check queued.

### T35. Mechanism check: is it the coherent batch MEAN? (10 members, old pipeline wknnx K4 c.7 k b .5, kept mean 1 - a)
a .5 (default) / .75 / 1 -> 42.33 / 41.62 / 37.28; k5 a1 37.48. NO: removing more of the mean hurts. The mean carries
signal (the shared component of the batch's images) and must be partly kept; what the LW shrink does better is the
SMOOTH, spectrum-adaptive flattening of all strong directions (incl. the mean in proportion to its energy) instead of
a hard top-3 cut. Band-power fingerprint without WCCN (raw z-scored; shrink 1) on the KD trio: 42.40 vs WCCN .3 43.88,
full base raw 42.48 vs 44.12: the within-person whitening (the learned part) is essential.

### T36. Knob-free prior-matched shrink (tms: scale batch direction v by min(1, sqrt(target-prior energy / batch energy)),
prior = second moment of the 16540 TRAINING DINOv2 targets; uncentred tms:0 / centred tms:1)
3 base members: lws 43.10 vs tms:0 38.98, tms:1 40.25 (c 1 40.52); 10 members tms:1 40.90; KD trio 38.82 / 39.60.
DEAD: the members' outputs are shrunk regressions of the targets, their spectrum is not the targets' spectrum, so the
image prior misreads signal as nuisance. The batch's own spectrum (LW, one d) is the right reference.

### T37. Robustness of the KD trio (REVE-d6-KD + SQF-KD + rmix-reg), lws
| variant | cv4 |
|---|---|
| wknn 4/.7 (old post-step) | 41.72 |
| lws d .9 / .93 / .95 (K4 c.85) | 44.10 / 44.12 / 44.05 |
| c .7 / .85 / 1 (d .93 K4) | 43.95 / 44.12 / 44.35 |
| K 3 / 4 / 6 | 43.83 / 44.12 / 43.78 |
| band-power fingerprint (128-d) | 43.88 |
| rich fingerprint (2976-d) | 44.12 |
| swap the rmix member: rmix / rmix-d8 | 43.83 / 43.28 |
| pairs: REVE-KD + rmix-reg / + rmix; SQF-KD + rmix-reg | 43.73 / 43.32 / 43.17 |
| + rmix (4 members) / + rmix + rmix-d8 (5) / + rmix + REVE-d6 (5) | 44.00 / 43.70 / 43.48 |
Plateau 43.8-44.35 for the trio over every knob: 3 members = 10 members (44.15). More members do not help once the
ensemble holds a KD member (KD already contains the base-ensemble averaging).

### T38. Smallest fingerprint (KD trio, lws .93 4/.85, WCCN .3 on the subset; purity@4 folds 1-4)
| fingerprint | dims | purity f1..f4 | cv4 |
|---|---|---|---|
| base | 656 | .772/.724/.784/.690 | 44.12 |
| 4 band powers (delta theta alpha beta x 32 ch) | 128 | .735/.724/.737/.651 | 43.88 |
| ... unsupervised PCA to 8 / 16 / 32 dims first, then WCCN | 8/16/32 | f1 .509 / .551 / .639 | - / - / 42.85 |
| delta / theta / alpha / beta alone | 32 each | f1 .362 / .476 / .505 / .716 | 40.28 / 41.70 / 41.70 / 43.00 |
| **alpha + beta (8-13, 13-30 Hz log power per channel)** | **64** | .726/.660/.743/.630 | **43.93** |
| theta + alpha + beta | 96 | .744/.655/.758/.670 | 43.57 |
Identity = the alpha/beta power TOPOGRAPHY (the classic, heritable EEG trait). PCA does not find it (identity is
spread over many low-variance directions: T4's reason to keep all dims and whiten within-person). A 64-d
fingerprint + the 64 x 64 within-person whitening (fit on the training people) = the base fingerprint's score.

### T39. Can the shrink choose its own strength? (KD trio unless noted)
- Ledoit-Wolf's analytic optimal intensity per batch (lwa): picks d = .13-.17 (sd .01), i.e. near-full whitening ->
  38.30 (c .85) / 39.48 (c 1); alpha+beta fingerprint 37.45; 3 base 38.65. DEAD: covariance-MSE-optimal is not
  retrieval-optimal (the Frobenius loss is dominated by the top eigenvalues, which here carry signal too).
- Knob-free candidate lwn: shrink toward the MEAN BATCH eigenvalue lam_bar = tr / n instead of nu = tr / p, i.e.
  whiten with (S + lam_bar I): d = p / (p + n) = .96 at B 64 (.98 at B 32, .92 at B 128): running.
- lwn result (KD trio): B 64 c .85 / 1 -> 44.18 / **44.33** (fixed d .93 c .85: 44.12); B 128 44.33 / 44.55 (44.45);
  B 32 41.72 / 41.90 (42.45: at B 32 d = .98 is too mild); 3 base members 42.72 / 42.45 (43.10). At the grader's
  B 64, d = p / (p + n) and c = 1 (subtract the mates' full offset) remove both knobs at no cost.

### T34 LIVE (grader_sim --shuffle --fp, REVE-d6-KD + SQF-KD + rmix-reg, real models + raw EEG, job 63492)
plain 33.83, k2 38.82, router 40.55, wknn 41.70, wknnr 42.30, **wknnl (lws d .93 c .85, base fingerprint) 44.12**
(+3.58 vs router, 18/2; replay 44.12 exact), wknnrl (rich) 44.10. Three models in the 44-45 band, verified end to end.

### T40. The lean pipeline, all pieces together (replay): K4 mates in the 64-d alpha+beta WCCN space, subtract their
full offset (c 1), ONE uncentred shrink of the summed output with d = p / (p + n). Only K and the WCCN shrink (.3).
KD trio **44.05 (20/0)**; 3 base members 42.48; REVE-d6-KD alone 42.00. Deploy: batch_transform_wknn(per,
F[:, FP_AB], fit_fingerprint_wccn(..., cols=FP_AB), c=1, d='auto'); grader_sim output 'wknna'.

# Round 4 (user, 2026-10-21): <= 5 members, NO KD members, 45; novel approaches, keep it simple
(kdx training cancelled: KD excluded.) Baselines without KD: 3 base 43.10 (lws) / 42.48 (lean); 5 family 43.17.

### T41. Cross-person same-image pooling (each 64-batch holds ~10 same-image pairs, 8 across people; the scorer
averages within a person only, so pooling ACROSS people adds evidence it never sees). y_i += b x mean(same-image mates)
| | 3 base (lean 42.45) | 5 family (lean 43.43) |
|---|---|---|
| ORACLE (true image labels) b .5 / 1 / 2 | 46.48 / 49.58 / 52.30 | - / 50.75 / 53.78 |
| label-free soft pooling on output cosine (b, tau) .5/.05, 1/.05, .5/.1, 1/.02 | 40.23 / 37.52 / 39.85 / 37.05 | 40.85 / 37.92 |
Why it fails: precision of output-similarity pairs (fold 1, 3 base): most similar pair per batch 0.96% same image,
mutual nearest neighbours 0.88%, random 0.50%. Single-trial outputs cannot tell which trials share an image (only
~0.3 true mates per trial). Huge headroom, unreachable without labels. DEAD.
Non-KD member survey under lws (for the record): REVE-d6+SQF+rmix-reg 42.80; REVE-d6+SQF+TSC+rmix-reg+REVE-scratch
43.67; +rmix instead of TSC 42.48; rmix x3 + REVE-d6 + REVE-scratch 41.73; 5 family 43.17 (lean 43.43).

### T42. Diarisation-style spectral clustering of the batch instead of kNN (normalised Laplacian, eigengap k, k-means;
mates = the whole cluster): cluster-mate purity .47-.49 (eigengap, ~4 clusters) / .57-.59 (true k 5) vs kNN@4 .77.
Clusters are far less pure than the 4 nearest neighbours. DEAD.
### T43. Bias-corrected offset (theory: E[mates - mu] ~ purity x offset, so c = 1 / purity ~ 1.3), lwn, 5 members
| K \ c | 1 | 1.2 | 1.4 | (5 family / 5 survey = REVE-d6 SQF TSC rmix-reg REVE-scratch) |
|---|---|---|---|---|
| 4 | 43.43 / **44.05** | 42.85 / 43.25 | 41.82 / 42.08 | |
| 6 | 43.72 / 43.63 | 43.20 / 43.77 | 42.75 / 43.20 | |
| 8 | 43.50 / 43.38 | 43.80 / 43.83 | 43.73 / 43.65 | |
No gain: the contaminating mates are not "zero offset" (they are the batch's other people, partly cancelling in the
shrink). Shift-TTA members under the new shrink (3 base): lwn 42.28 / lws 42.65 vs 42.45 / 43.10: still DEAD.
Best without KD so far: 44.05 (5 survey members, lean lwn K4 c1).
### T43b. Person-specific latency (3 base, lwn): all inputs shifted by -2 / -1 / 0 / +1 / +2 samples -> 36.72 / 40.67 /
42.45 / 41.72 / 39.10; ORACLE best shift per person 43.50 (+1.05), chosen shifts mostly 0 / +-1, and the max over 5
noisy per-person scores carries its own selection bias. Too little headroom for a label-free choice. DEAD.
### T44 (running). Mirror TTA: left-right mirrored scalp (Fp1<->Fp2, ..., midline fixed) as a second view of the same
model (the EEG analogue of horizontal-flip TTA; DINOv2 targets are nearly flip-invariant). tta_cache.py --mirror.
### T44 result: mirror TTA (lwn K4 c1): 5 survey members mirrored-only 32.55 (models are NOT mirror-invariant: -11),
normal + mirrored mean 41.70 vs 44.05; 3 base 40.70 vs 42.45. DEAD.
### T48. MC-dropout (8 stochastic passes averaged, dropout layers only: REVE-d6 1 layer p .15, SQF 15, TSC 3): 3 base
lwn 41.85 vs 42.45. DEAD (cheap pseudo-members add noise, not diversity).
### T45. Power-law normalisation per output dimension (image-retrieval burstiness fix) after lwn, 5 survey members:
a .5 / .75 / 1.5 -> 41.85 / 43.00 / 42.40 vs 44.05. DEAD.
### T46. New member families (not KD), cached for the first time; lwn K4 c1. Alone: ATM-dbn 33.73, pretrained LaBraM
(plab_gain) 38.88, pretrained REVE (reve_z) 39.28, dualstream 33.10. Swapped into the 5 survey set (44.05): ATM for
TSC 43.10, ATM for REVE-scratch 43.63; plab for TSC 42.93; reve_z for TSC 43.40; plab for REVE-scratch 43.97; REVE-d6 +
ATM + rmix-reg 41.57. No family beats the survey set: every <=5 non-KD ensemble lands in 41.6-44.05.
### T47. Second-order grouping (re-ID style): mates by cosine between the trials' SIMILARITY PROFILES over the batch
(knob-free). Purity@4 f1..f4 .772/.724/.784/.690 -> .813/.754/.808/.688, but 5 survey members lwn K4 / 8 / 12 ->
43.80 / 43.67 / 43.85 vs 44.05. Purity +.03 does not move the score (as T9). DEAD.
**Ceiling**: perfect in-batch grouping + lwn on the 5 survey members (ocbn): c 1 44.93, c .85 45.12. => with these
members, 45 sits AT the first-order ceiling; better grouping cannot reach it. The lever left is the member set.
### T49 (running). Nested greedy forward choice of <= 5 among 12 non-KD cached members (tta_greedy.py).
### T49 (partial). Nested greedy choice of <= 5 non-KD members (lwn K4 c1): chosen on folds 1-2 = REVE-scratch + TSC +
plab_gain -> 44.50 on folds 1-2 but **42.20 on folds 3-4**: member choice on 10 people is worth ~2 points of optimism.
(Reverse direction still running at the goal stop.)
### T50. Checkpoint soup (sealed_best + final epoch, weights averaged): not scored, the cache build failed on a bug I
introduced (soup() took cache()'s @torch.no_grad decorator; fixed, rerun `tta_cache.py <recipe> --soup`).
Round 4 stopped by the user (goal cleared). Best <= 5 non-KD: 44.05 (REVE-d6 + SQF + TSC + rmix-reg + REVE-scratch,
lean lwn K4 c1); perfect-grouping ceiling for that set 44.93 (c 1) / 45.12 (c .85).
