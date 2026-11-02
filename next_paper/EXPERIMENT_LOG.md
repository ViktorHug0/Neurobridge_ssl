# Next paper: experiment log

## 2026-10-25 — Why does GALA collapse with distractors while SCORE degrades gracefully?

Scripts: `next_paper/scripts/open_gallery_diag.py`, `open_gallery_shrink.py`. Data: ATM ValCon (atm128_vc) × 3 seeds ×
10 subjects, the paper's R3 setup (80-rep queries, D distractors = one image per training concept, cross-fold GALA
params). Outputs and embedding cache: `results/things_eeg/cvpr/next_paper/open_gallery/`. SCORE = our reimplementation
(internal use only; the CVPR paper uses SCORE's published numbers).

Top-1 (%), D = 0 / 200 / 1654:
- plain 32.2 / 24.6 / 11.5; SCORE 64.3 / 51.4 / 26.4; GALA (balanced) 81.8 / 44.3 / 11.3
- GALA free plan 71.1 / 42.5 / 13.6; GALA injective plan (cols <= 1) 80.5 / 43.8 / 13.6
- GALA fitted on the true support, scored on the full gallery: 81.8 / 80.8 / 77.5 (retrieval competition is minor)
- SCORE moment matching + CSLS(10), no rotation: 56.1 / 44.5 / 23.4; SCORE rho=0: 62.5 / 48.6 / 24.7
- GALA whitening + CSLS only: 52.0 / 36.7 / 17.5; GALA 1 step: 68.3 / 46.9 / 20.7
- GALA with identity-shrunk Procrustes (SCORE's form): lambda 0.1: 83.5 / 51.6 / 17.2; lambda 1: 74.6 / 49.6 / 21.4

Diagnostics (D = 0 → 1654): true-pair mass in GALA's FIRST plan 0.30 → 0.03 (last plan 0.75 → 0.04); distractor mass
in the last plan 0.89 (balanced), 0.84 (free / injective); GALA rotation distance to the true-support rotation 0 → 0.89
(1 = orthogonal); every GALA rotation, the oracle's included, is far from identity (0.93). SCORE landmarks 123 → 179,
precision 74% → 25%.

Reading:
1. The column marginal is NOT the main cause: injective transport ≈ free ≈ balanced at large D (hypothesis falsified).
2. The initial scores are already dominated by distractors (3% true mass before any rotation); the iterated,
   unregularized 128-d rotation then amplifies them (1 step 20.7 > 16 steps 11.3).
3. SCORE's robustness is mostly correspondence-free: moment matching to the gallery + CSLS(10) gives 23.4 of its 26.4
   at D = 1654; its rotation adds +3, and its identity shrinkage +1.7.
4. Identity shrinkage (lambda 0.1) also improves GALA in R1 (83.5 vs 81.8, lambda = SCORE's value, not tuned).
5. The 55-point gap to the true-support oracle is entirely correspondence / support estimation.

## 2026-10-25 (evening) — making GALA robust to distractors (R3), round 1

Scripts: `next_paper/scripts/open_gallery_methods.py` (variants), `open_gallery_gw.py` (relational support),
`open_gallery_alt.py` (alternating support refinement). Same data as above (ATM ValCon, 3 seeds × 10 subjects).
Top-1 (%), D = 0 / 200 / 400 / 800 / 1654 (SCORE 64.3 / 51.4 / 43.2 / 35.1 / 26.4; GALA 81.8 / 44.3 / 31.6 / 20.1 / 11.3):
- **Robust GALA = blend start + ratio shrinkage** (start = (1-r) SAW + r * moment-matching-to-gallery, every Procrustes
  step shrunk towards identity with lambda = r, r = 1 - N/M; N, M known at deployment; == GALA when M = N):
  81.8 / 54.4 / 43.8 / 34.4 / 25.9. Best overall; beats SCORE up to D = 400, ties it after.
- ratio shrinkage alone (SAW start, lambda = 1 - N/M): 81.8 / 51.8 / 40.0 / 30.6 / 21.3
- moment-matching start + ratio shrinkage: 78.2 / 53.2 / 43.5 / 34.5 / 25.8 (c = 2: 26.8 at D = 1654)
- fixed shrinkage lambda 0.1 / 0.3 / 1: 83.5|80.3|74.6 at D = 0; 17.2|19.6|21.4 at D = 1654
- self-pruning gallery (capacity caps from received mass^b): no gain (b = 1: 12.4 at D = 1654; b = 4 worse)
- support first (Hungarian / plan mass) then GALA: 17.5 / 11.0 at D = 1654
- CSLS k scaled with M/N: slightly worse (24.4)
- **Relational support (partial entropic GW, own solver): FALSIFIED.** Pure GW support precision = chance (12% at
  D = 1654, chance 10.8%); the true matching does not score higher than a wrong Hungarian matching under the GW
  objective, and GW started AT the truth drifts to chance (D = 1654) or ~50% (D = 200). Second-order structure of 200
  noisy queries does not identify the support in a dense gallery.
- **Alternating support refinement: no bootstrapping.** Hungarian support precision 67 / 54 / 42 / 30% (D = 200..1654)
  after round 1 and flat afterwards; top-1 flat (25.8 at D = 1654).
Remaining gap at D = 1654: robust GALA 25.9 vs true-support oracle 77.5 (fit on truth, scored on full gallery).
Hypothesis to test: in R2 the ratio N/M < 1 early in a stream, so the ratio shrinkage may act as automatic damping.

## 2026-10-26 — D-agnostic solution: agreement-calibrated expert fusion (ACEF, `tri_p`)

Constraint (user): no knowledge of the number / share of distractors (N/M not allowed either).
Dead ends on the way (all D-agnostic, ATM 3 seeds): decoy-hit model selection (criterion INVERTS once the fit has seen
the distractors; with decoys in the fit the sign is right but weak, rho ~ -0.35, and decoys cost 11 pts at D = 0);
Storey pi0 of the absent share from decoy p-values (column statistic AUC 0.60-0.72, pi0 biased high: -9 pts at D = 0);
rotation instability as trust signal (flat across D); robust (Cauchy / Geman-McClure IRLS) Procrustes (no gain: even
the ORACLE fit on the true pairs among the Hungarian matches gives 28.3 at D = 1654 — with ~48 correct pairs a 128-d
rotation is underdetermined; 50 / 100 / 200 perfect pairs give 31.8 / 59.2 / 93.8; subspace rotations do not help);
novel (never-target) distractors are as harmful as training ones (R3 collapse is not a training-image artefact).

**Method (`next_paper/scripts/gated_eval.py`, `tri_p`).** Three transductive experts on the same gallery:
G = GALA; C = conservative GALA (queries moment-matched to the gallery, Procrustes shrunk to identity, lambda = 1);
S = SCORE-style landmark recovery (our reimplementation). Label-free accuracy of each expert from pairwise top-1
agreement a_ij ~ p_i p_j (triplet identity, e.g. p_G = sqrt(a_GC a_GS / a_CS)); fused scores = sum_k p_k z(S_k)
(z = per-query z-score). No tuned parameter, no D, no N/M. p_G tracks GALA's true accuracy (r = 0.95, MAE 8.8 pts);
p_C, p_S are inflated (C and S errors are correlated) but their ratio to p_G is what matters.

Evaluation `next_paper/scripts/summarize_agf.py`: 6 encoders (ATM x 3 seeds, TSConv, EEGNet, EEGProject, Conformer,
LaBraM; ValCon) x {train distractors D 0-1654, novel distractors D 200-1650, coverage U 25/50/100 x D 0/400} = 90 cells.
- vs the best single expert of {GALA, SCORE, C} in each cell (a per-cell oracle choice): mean +0.8, worst -1.8.
- worst loss vs GALA in R1 (D = 0): -0.25.
- wins over SCORE / GALA / robust_NM (which needs N/M): 68 / 88 / 71 of 90 cells.
- ATM train D = 0/200/400/800/1654: 81.6 / 55.4 / 45.5 / 36.3 / 27.4 (GALA 81.8 / 44.3 / 31.6 / 20.1 / 11.3;
  SCORE 64.3 / 51.4 / 43.2 / 35.1 / 26.4; robust_NM 81.8 / 54.4 / 43.8 / 34.4 / 25.9).
- Mean per encoder x regime: best in 16 / 18 rows (LaBraM cover / novel: SCORE >= ACEF).
Other fusions tried: agreement gate a^k z(G) + z(C) (k = 2 tuned: mean +0.2, worst -6.4), with conservative = mean of
C and SCORE (`agf2_mix`: mean +1.0, worst -2.4 but -2.4 in R1), gates on p_G only (worse). Held-out test on the 5
image-target ATM encoders (not used in development) launched.
**Held-out test (method frozen; 5 ATM image-target encoders never used in development: InternViT L20 / L44, DINOv2
L40, CLIP-H final, bigG L27; ValCon s3300; 75 cells):** ACEF vs best single expert per cell: mean +0.18, worst -2.4
(6 cells below -1, all at U = 25 / 50 or on the weak DINOv2-L40 target); R1 vs GALA: +0.1 to +3.6 (never below);
wins over SCORE / GALA / robust_NM: 53 / 75 / 64 of 75.

## 2026-10-26 (night) — ACEF: SCORE-free variants, correspondence bound, refinement, label-informed oracle

Script `gated_eval.py` (+ `lsq_acc`: K-expert accuracies from log-agreements by least squares), `acef_refine.py`,
`summarize_acef.py`. 13 runs (dev: ATM x 3 seeds + 5 encoders; held-out: 5 ATM image targets), 165 cells.
- **Final = ACEF with experts {GALA, conservative GALA, landmark expert}** (`tri_p`). The landmark expert is the
  classical CSLS mutual-NN + Procrustes refinement (Conneau et al. 2018, adapted by SCORE); without it (`acef_own3`:
  GALA, conservative, correspondence-free M) mean -0.47 / worst -6.0 vs the oracle below; ATM train 81.5 / 54.3 /
  44.2 / 35.3 / 26.6 (still >= SCORE at every D).
- **vs a label-informed oracle** that picks per cell the best of {GALA, SCORE, C, C10, robust GALA with the TRUE N/M}:
  ACEF mean +0.30, median +0.10, worst -2.4, within 1 point in 149 / 165 cells.
- Correspondence bound (rotation fitted on the true pairs among ACEF's Hungarian matches), ATM train D = 0..1654:
  87.5 / 64.6 / 54.3 / 43.5 / 32.4 with 166 / 112 / 92 / 74 / 55 true pairs; ACEF 81.6 / 55.4 / 45.5 / 36.3 / 27.4.
- Refinement rounds (refit GALA from the fused plan, re-fuse; 3 seeds): D = 0 81.6 -> 82.8 (above GALA 81.8),
  D >= 200 unchanged (true pairs do not grow: the refitted GALA loop re-attaches distractors).
- **Closing the gap to the bound fails, and why (ATM, 3 seeds; `acef_onefit.py`).** One rotation fitted from ACEF's
  fused plan (plan^g) or from its Hungarian pairs weighted by fused margin^g, no further GALA loop, alone or as a 4th
  expert: best 24.8 alone / 26.9 fused at D = 1654 (ACEF 27.4); sharpening (g = 2..8) is monotonically worse.
  Separability of ACEF's correct vs wrong Hungarian pairs (AUC), D = 0 / 200 / 1654: fused margin 0.80 / 0.73 / 0.67,
  mutual-NN 0.67 / 0.57 / 0.55, expert agreement 0.79 / 0.73 / 0.65; precision of the 25 most confident pairs
  98% / 88% / 53%. With 25 perfect pairs a rotation gives 15.5, with 50 31.8: the bound (32.4, fit on the 55 true
  pairs) needs the identity of the correct pairs, which no label-free signal recovers at D = 1654 (AUC <= 0.67).
- **Match stability across repetition sub-averages (`stability.py`, 4 x 20-rep blocks per query, each expert's
  fitted transform applied to the blocks):** AUC for correct vs wrong ACEF pairs 0.83 / 0.73 / 0.69 / 0.68 at
  D = 0 / 200 / 800 / 1654 (margin 0.80 / 0.72 / 0.68 / 0.68): wrong matches are as STABLE as correct ones, i.e. the
  large-D errors are systematic, not noise. Stability-weighted rotation / 4th expert / block-averaged decoding: no gain
  (best 27.5 at D = 1654 vs ACEF 26.65 on s3301; within noise).
- **Identifiability test (ATM, 3 seeds): label-free objectives PREFER the wrong solution at large D.** At D = 1654,
  rotation fitted on ACEF's 200 pairs (top-1 27.0): mean Hungarian CSLS 0.191, mean max CSLS 0.191, mutual-NN rate
  0.998; rotation fitted on the 200 TRUE pairs (top-1 93.5): 0.173 / 0.173 / 0.990; rotation on the 55 true pairs
  among ACEF's (the bound, 32.4): 0.067 / 0.069 / 0.772. At D = 0 the truth wins (0.222 vs 0.218). So at D = 1654 no
  search or selection on first-order matching objectives can reach the bound: the gap is an identifiability limit of
  the single-query-per-image open-gallery problem, which explains why all attempts to close it failed.

## 2026-10-30 — Toward one principle instead of three hand-picked experts (ATM, 3 seeds, R3 train distractors)

Literature anchors: Wasserstein Procrustes (Grave et al. 2019), VecMap robust self-learning (Artetxe et al. 2018),
partial / unbalanced OT for partial domain adaptation with class-weight estimation (MARS, JCPOT, SSOT),
alpha-TIM (unknown query marginals in transductive few-shot), ProMises (vMF prior on the Procrustes rotation),
spectral meta-learner (Parisi et al. 2014: label-free accuracies from a rank-one agreement matrix; needs >= 3
conditionally independent predictors), NPAIRS (Strother et al. 2002: split-half reproducibility for pipeline choice).

Family F(r): start (1-r) SAW + r moment matching, Procrustes shrunk to identity with lambda = r (F(0) = GALA,
F(1) = conservative expert). Top-1 for r = 0..1 at D = 0 / 1654: 81.8 -> 67.8 / 11.3 -> 26.0 (r = 0.1: 84.4 at D = 0).
Per-subject oracle r: 85.2 / 56.0 / 45.6 / 35.6 / 26.6 (D = 0..1654), still below ACEF at large D.

Dead ends (`crossfit.py`, `anchored.py`, `path.py`):
- **Choosing r by fit, in-sample OR cross-fitted over queries (fit on 100, evaluate on the other 100): fails.** Every
  criterion (max cos, vMF log-sum-exp, Hungarian cos, z-scored max / margin, CSLS max) is a monotone function of r
  (Spearman +-0.95 at D = 0, identical in- and out-of-sample). The wrong large-D rotation TRANSFERS to unseen
  queries: it is a distribution-level match (queries are a random sample of the gallery distribution), so held-out
  queries are fitted as well as fitted ones. Distribution-level criteria cannot detect it.
- Anchored coupling (plan from z(S(T)) + w z(S0), S0 = moment matching + CSLS, inside the loop): 82.6 / 51.1 / 39.0 /
  29.3 / 20.1 (w = 0.5) to 74.4 / 51.4 / 41.4 / 32.2 / 24.2 (w = 2); no w is good at both ends.
- Path marginalisation with cross-member agreement weights (rank-one / eigen / lsq): 82.7 / 56.4 / 45.1 / 34.7 /
  24.3 (uniform 82.9 / 56.4 / 44.8 / 34.4 / 24.1). Neighbouring members share errors (r = 0.9 vs 1.0 agree 0.94-0.97
  at 31% accuracy), the rank-one model inflates the middle of the path: weights hardly move with D.

**Split-half reliability works (`splithalf.py`, `splithalf_fuse*.py`).** Align the two 40-repetition halves of the
queries independently with the same member; agreement of their top-1 = the member's reliability. The replicates have
independent EEG noise, so no independence between DIFFERENT members is assumed.
- argmax_r agreement picks r = 0.22 / 0.75 / 0.76 / 0.86 / 0.83 (oracle 0.09 / 0.41 / 0.54 / 0.71 / 0.85): D-agnostic
  and tracks the oracle; F(r*) 82.3 / 53.1 / 43.5 / 34.5 / 25.5.
- The large-D edge of ACEF comes from the landmark estimator S (sparse mutual-NN correspondences): F(1) + S 27.5 at
  D = 1654.
- Members {F(0), F(0.5), F(1), S}, weights = own split-half agreement^k (`lin_path_k`):
  k = 1: 82.1 / 55.8 / 45.5 / 36.3 / 26.4;  k = 2: 82.1 / 56.5 / 45.7 / 36.7 / 27.6;  k = 4: 82.2 / 56.5 / 46.0 / 36.7 / 27.7
  (ACEF 81.6 / 55.4 / 45.5 / 36.3 / 27.4). ACEF's members {G, C, S} with k = 2: 82.5 / 55.7 / 45.3 / 36.8 / 27.7.
- Parameter-free weights fail: calibrated product of experts (temperature so mean confidence = reliability) 77-79 at
  D = 0; signal-detection d' weights (M-AFC inversion) too flat, 24.4-25.2 at D = 1654 (GALA's confident wrong
  scores dominate).
Frozen for the held-out test (`rwa.py`): members {F(0), F(0.5), F(1), S}, k = 2 main (1 and 4 reported).
**Held-out test (10 encoders not used above, s3300, R3 train D = 0..1654, 50 cells; `rwa.py`, `summarize_rwa.py`).**
Mean over encoders, D = 0 / 200 / 400 / 800 / 1654: ACEF 54.6 / 35.2 / 27.5 / 20.6 / 14.8; RWA k = 2 55.0 / 35.4 /
27.7 / 20.7 / 14.7. RWA k = 2 - ACEF per cell: mean +0.16, worst -0.85, wins 26 / 50 (k = 1: -0.28 / -1.85;
k = 4: +0.21 / -1.40). R1 (D = 0) vs GALA: worst -0.10. => parity with ACEF, cleaner principle; the ATM-dev edge
(+0.5..1) did not survive. Max-reproducibility weights (`rwa_opt.py`, simplex argmax of the fused decoder's split-half
agreement, no exponent): ATM dev ~ACEF (80.6 / 54.9 / 45.4 / 36.3 / 27.5) but held-out (9 encoders) mean -0.86,
worst -5.3, D = 0 52.5 vs 54.7: the halves are noisier than the deployed queries, so reproducibility over-rewards
shrinkage. Not adopted.

## 2026-10-30 (evening) — Simplest D-agnostic upgrade of GALA (ATM, 3 seeds, R3 train, D = 0 / 200 / 400 / 800 / 1654)

GALA 81.8 / 44.3 / 31.6 / 20.1 / 11.3. Mechanism found (`simple.py`, inline diagnostics):
- GALA's FIRST Procrustes step is a near-orthogonal jump (|M - I| ~ 0.8-0.9) that helps at every D (D = 1654:
  accuracy peaks after step 1); later steps are small refinements: +13 at D = 0, -11 at D = 1654 with 4-20% top-1
  churn per step. The true rotation is far from identity (0.93), so identity shrinkage only works as a SLOWING of the
  refinement walk, never as a prior: an anchored prior on the total rotation (re-fit from the start, + l I) costs
  7-25 pts at D = 0 (anc0.1 74.6 / 49.3 / 39.6 / 30.0 / 21.1). The moment-matched start alone does nothing (mm_gala
  11.1 at D = 1654); with slowed steps it adds +3-4 at large D.
- Effective candidates per query n_eff (plan entropy) grows 25 -> 240 with D, but it is a proxy of M: random-vector
  distractors inflate it more (384). A rule r = n_eff / N would be N/M in disguise; dropped.
Tried, no better than the fixed F(0.2) (blend start 0.2 + step shrinkage 0.2: 83.0 / 54.0 / 42.3 / 31.2 / 21.2):
iterate averaging of scores 81.9 / 49.2 / 37.0 / 25.9 / 17.4; gallery whitened too 71.6 / .. / 16.4; absolute prior
mass (ProMises MAP) <= 70 at D = 0; jump-then-crawl (step 1 free, later steps shrunk) 80.2 / 52.9 / 42.4 / 32.3 / 22.0
(from F0.2 start, l 0.3); Conneau mutual-NN refinement after GALA's first step 69 at D = 0; per-query plan-sharpness
weights 72.4 / .. / 9.8.
Contrastive barycentre (`contrastive.py`: Procrustes target b_i(tau) - b_i(c tau), background pull removed) from the
F0.2 start: c = 4 86.2 / 53.6 / 40.4 / 29.6 / 20.0, c = 10 84.5 / 54.2 / 42.0 / 30.9 / 20.4: a strong R1 gain
(+4.4 over GALA at D = 0) but no large-D fix (the large-D pull comes from near neighbours, not the background).
**Held-out (10 encoders s3300, 50 cells; `contrastive.py`, `simple2.py`).** Mean top-1, D = 0 / 200 / 400 / 800 / 1654:
GALA 53.1 / 27.2 / 18.5 / 10.7 / 6.1; F0.2 52.8 / 31.8 / 23.4 / 16.4 / 10.7 (vs GALA: never below at D >= 200, worst
+0.7; D = 0 mean -0.3, worst -8.0); F0.1 54.4 / 30.9 / 22.2 / 14.9 / 9.2 (D = 0 +1.4, worst -5.0; one cell -0.1 at
D >= 200); ctrF10 54.3 / 31.9 / 23.7 / 16.2 / 10.1 (D = 0 worst -5.8); step shrinkage alone (shr0.1..0.3) is worse
everywhere (+1.7..3.5 at D >= 200, D = 0 worst -8..-12). The D = 0 losses are on Conformer / EEGNet / TSConv and come
from the step shrinkage (slower refinement), not from the moment-matching blend.
Note: numpy's float64 gesdd sporadically fails to converge on finite 128x128 matrices here (contrastive.py now falls
back to scipy gesvd).
