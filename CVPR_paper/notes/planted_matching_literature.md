# Planted matching / database alignment: when does the bijective MLE beat per-row NN?

Written 2026-10-21. Every entry below was checked against the arXiv text (PDF pulled and read)
and, where one exists, the publisher record. Our setting: S is a 200x200 cosine matrix, the planted
matching is the identity, "NN" means row argmax, "MLE" means Hungarian/LAP on S. S is the MLE
only if S_ij is (an affine function of) the log-likelihood ratio, so treat Hungarian-on-cosine as
"LAP on a proxy LLR".

## Verified sources

**[MMX21] Moharrami, Moore, Xu. "The planted matching problem: Phase transitions and exact results."**
*Ann. Appl. Probab.* 31(6):2663-2720, 2021. DOI 10.1214/20-AAP1660. arXiv:1912.08880.
- Model: K_{n,n} with independent weights. Planted edges ~ Exp(λ) (mean 1/λ), null edges ~ Exp(1/n) (mean n). Here the MLE is the min-weight matching.
- Result: if λ ≥ 4, overlap is 1 − o(1) w.h.p. If λ < 4, the expected overlap is an explicit α(λ) < 1, derived with local weak convergence and Aldous's ζ(2) method.
- For us: this is the only exact finite-overlap (macroscopic-error) result, i.e. the regime we are in. *Our derivation, not in the paper:* in the same model, row-NN is correct with probability λ/(λ+1), because the smallest of n−1 null weights is ≈ Exp(1). At λ = 4, NN gets ≈ 80% and the MLE gets ≈ 100%, so the constraint alone produces the gap.

**[DWXY23] Ding, Wu, Xu, Yang. "The planted matching problem: Sharp threshold and infinite-order phase transition."**
*Probab. Theory Relat. Fields*, 2023. DOI 10.1007/s00440-023-01208-6. arXiv:2103.09383.
- Model: planted edges plus Erdős–Rényi null edges of average degree d. Weights are independent: P on the planted edges, Q on the others.
- Result (Thm 1): if √d·B(P,Q) ≤ 1, where B(P,Q) = ∫√(pq) is the Bhattacharyya coefficient, the MLE (max-weight matching on log P/Q) has vanishing error. If √d·B(P,Q) ≥ 1+ε, *every* estimator has error bounded away from 0. The exponential case reduces to λ = 4, and for λ ≤ 4−ε the optimal error is exp(−Θ(1/√ε)) (an infinite-order transition).
- **Caveat:** the converse for the dense model needs the null density in the scaled form q(x) = ρ(x/d)/d, so nulls are stretched over a range of size d. It does **not** cover the unscaled Gaussian model N(μ,1) vs N(0,1); for that model see [DCK23]. The positive (MLE) direction is general.
- For us: B(P,Q) between diagonal and off-diagonal score laws is the natural governing quantity for the MLE. NN instead depends on the tail of max over j of Q against P.

**[SSZ20] Semerjian, Sicuro, Zdeborová. "Recovery thresholds in the sparse planted matching problem."**
*Phys. Rev. E* 102, 022304, 2020. arXiv:2005.11274.
- Model: the same independent-weight model on sparse or dense graphs, with generic P and Q.
- Result (physics-level; cavity/BP mapped to a branching random walk): full recovery holds when the BRW velocity v = −inf_θ ln[I(θ)I(1−θ)]/θ is positive, where I(θ) = √c ∫ P̂^θ P^{1−θ}. By convexity this reduces to √c·∫√(p̂ p) = 1, which is the same Bhattacharyya criterion. They also conjectured the infinite-order transition that [DWXY23] later proved.

**[KNW22] Kunisky, Niles-Weed. "Strong recovery of geometric planted matchings."**
*SODA 2022*, pp. 834-876. DOI 10.1137/1.9781611977073.36. arXiv:2107.05567.
- Model: x_i ~ N(0, I_d) and y_i = x_i + z_i with z_i ~ N(0, σ² I_d). Only a permuted copy of the y's is observed. Scores are *dependent* (they share x_i and y_j). The MLE is the LAP on ‖x_i − y_j‖², which equals the LAP on ⟨x_i, y_j⟩: the ‖x_i‖² and ‖y_j‖² terms appear once in every permutation, so they cancel (their eq. 4).
- Signal-to-noise ratio: (d / log n)·log(1 + σ⁻²).
  - Thm 1.1 (d = o(log n)): σ² = o(n^{−4/d}) gives zero errors. n^{−4/d} ≪ σ² ≪ n^{−2/d} gives |E| ≈ σ^d n². σ² ≥ a·n^{−2/d} gives |E| ≥ e^{−cd} n.
  - Thm 1.2 (d ~ a log n): perfect recovery if σ² < 1/(e^{4/a} − 1); sublinear error n^{2 − (a/2) log(1 + σ⁻²)} up to σ² < 1/((2e^{1/a} − 1)² − 1).
  - Thm 1.3 (d = ω(log n)): perfect recovery if σ² ≤ (1/4 − ε)·d / log n.
  - Conjecture 1.4: macroscopic error beyond these thresholds.
- **NN vs MLE (App. A, heuristic, "no detailed proofs", improper i.e. non-bijective):**
  - Greedy *distance* NN matches the MLE's error rate when d = o(log n), but makes n − o(n) errors when d = ω(log n).
  - Greedy *inner-product* NN fails completely when d = o(log n / log log n). Only points on the convex hull of the y's can be matched, so this is a norm/hub effect. When d = ω(log n) it can sometimes beat the MLE's conjectured strong-recovery threshold (½ vs ¼ of d / log n).
  - Around d = Θ(log n), "all three algorithms are competitive and none strictly dominates".
- For us: (i) whether NN or the MLE wins depends on **effective dimension relative to log n** (log 200 ≈ 5.3). (ii) Inner-product NN loses exactly through **column effects** (‖y_j‖, centrality). The LAP is invariant to these and NN is not.

**[DCK19] Dai, Cullina, Kiyavash. "Database alignment with Gaussian features."**
*AISTATS 2019*, PMLR 89:3225-3233. arXiv:1903.01422.
- Model: jointly Gaussian feature databases with arbitrary covariance Σ. Matched pairs have cross-covariance Σ_ab and unmatched pairs are independent. After canonicalisation the cross-correlation is diag(ρ).
- Governing quantity: I_XY = −½ Σ_i log(1 − ρ_i²), where the ρ_i are the canonical correlations.
- Thm 1: MAP (Hungarian on the LLR) gets exact alignment if I_XY ≥ 2 log n + ω(1).
- Thm 2 (Σ_ab = ρI, d = ω(1)): no algorithm can do this if I_XY ≤ 2 log n·(1 − Ω(1)).
- Per-pair LLR thresholding is also analysed.
- For us: correlated, non-isotropic features collapse to a spectrum of canonical correlations. Information is additive across canonical directions, so "effective dimension" means how I_XY is spread over the ρ_i.

**[DCK23] Dai, Cullina, Kiyavash. "Gaussian database alignment and Gaussian planted matching."**
arXiv:2307.02459, 2023. Marked "under review"; no venue found.
- Models:
  - Database alignment as in [DCK19].
  - Gaussian planted matching: independent unit-variance entries, mean μ on planted entries and 0 elsewhere. Signal is ζ = μ²/2 (or ζ = I_XY for databases). The two problems coincide when d = ω(log n) and no single feature is strong (their Condition 1).
- **This is the paper that compares max-row (= row NN, the LAP with the column constraint dropped) with the MLE directly.** Balanced case, sufficient conditions for ≤ n^{1−β} errors (Thm 3.2):
  - Max-row: ζ ≥ (1 + √β)² log n.
  - MLE: ζ ≥ (1 + 2√(β(1−β))) log n for β ≤ ½.
  - Almost-exact recovery (o(n) errors) for all methods at ζ ≥ log n.
  - Exact recovery: max-row needs ζ ≥ 4 log n, the MLE needs 2 log n.
  - Thm 3.4: these MLE rates match the converse for any estimator, so the MLE is rate-optimal.
- Example: at ζ = 2 log n, max-row makes ≈ n^{0.83} errors and the MLE makes ≈ n^{0.5}.
- **Unbalanced case:** with |V| − n = n^α and α > 1, the max-row and MLE boundaries *coincide*. The column constraint only helps when the gallery is about the same size as the query set. This is the mechanism: the gain comes entirely from the one-to-one column constraint.

**[WWXY22] Wang, Wu, Xu, Yolou. "Random graph matching in geometric models: the case of complete graphs."**
*COLT 2022*, PMLR 178. arXiv:2202.10662.
- Quadratic graph matching of latent Gaussian geometry with d = o(log n): perfect recovery when σ = o(n^{−2/d}), almost perfect recovery when σ = o(n^{−1/d}), and both are information-theoretically optimal.
- It concerns graph matching, so it is only tangential to our linear problem; it confirms the n^{−1/d} scaling.

**Hubness and its corrections (verified, empirical rather than theory):**
- Radovanović, Nanopoulos, Ivanović, "Hubs in space: popular nearest neighbors in high-dimensional data", *JMLR* 11:2487-2531, 2010. The k-occurrence distribution N_k becomes skewed as intrinsic dimension grows.
- Dinu, Lazaridou, Baroni, "Improving zero-shot learning by mitigating the hubness problem", ICLR 2015 workshop, arXiv:1412.6568.
- Conneau et al., "Word translation without parallel data", ICLR 2018, arXiv:1710.04087. CSLS(x,y) = 2cos(x,y) − r_T(x) − r_S(y) (their eq. 6). This is an additive row-plus-column correction.
- Bogolin et al., "Cross modal retrieval with querybank normalisation", CVPR 2022, arXiv:2112.12777. Shows hubness affects modern cross-modal embeddings.

## Not verified or not found
- I found **no paper with theory for planted matching under correlated or low-rank *score* noise** (non-i.i.d. W), apart from:
  - the geometric models [KNW22] and [WWXY22], whose dependence comes from shared points;
  - feature covariance in [DCK19] and [DCK23], which reduces to canonical correlations.
- I found no rigorous "LAP with noisy costs" result comparing NN and LAP at finite n.
- I did not check the DCK ISIT 2020 paper "Achievability of nearly-exact alignment for correlated Gaussian databases" (ISIT 2020, pp. 1230-1235) beyond its repository record.
- All theorems above are n → ∞ statements about error *exponents*. None gives finite-n constants for n = 200 at about 35% NN accuracy.

## Framing for the paper (our reasoning, built on the sources)
1. **Under i.i.d. planted matching the NN–MLE gap is a function of (P, Q) alone, and row-NN accuracy equals column-NN accuracy in expectation.** Our two encoders have the same row NN (35/35) but different column NN (46/33) and different Hungarian accuracy (66/52). So their score matrices cannot both be draws from one one-parameter i.i.d. family. The difference lies in distribution shape (B(P,Q) against the tail of max Q) and/or in dependence structure.
2. **The LAP is exactly invariant to S_ij → S_ij + a_i + b_j** ([KNW22] eq. 4 is one instance). Row NN is invariant to a_i only, and column NN to b_j only. Column effects b_j (image hubs) hurt row NN for free, and the MLE removes them. CSLS and Sinkhorn potentials are both additive row/column corrections.
3. **The gain comes only from the bijection constraint** ([DCK23]: it vanishes when the gallery is much larger than the query set). An MLE error needs an augmenting cycle, e.g. for a 2-cycle S_ij + S_ji > S_ii + S_jj. That is a *mutual* confusion. A row-NN error needs only a one-sided confusion. Hubs (one-sided) are fixed; symmetric semantic confusions (i ↔ j similar images) are not.
4. **Effective dimension matters** ([KNW22]). In low dimension, distance-NN is about as good as the MLE (in error rate). In high dimension the MLE dominates distance-NN, and inner-product NN is hub-limited.

## PROPOSAL (not a result): statistics of S predicted to drive the MLE-over-NN gain
- **P1. Additive-effect share.** Fit S_ij = m + a_i + b_j + R_ij (two-way ANOVA) and report Var(b) / Var(R).
  - Also report row-NN accuracy on the double-centred R minus row-NN on S. This is the part of the gap the LAP gets for free.
  - Companions: the skewness of the column k-occurrence N_1 (hubness), and |colNN − rowNN|.
  - Prediction: a larger share means a larger gain, so encoder A should be higher.
- **P2. Confusion reciprocity.** Over off-diagonal pairs, report ρ_rec = corr(S_ij − S_ii, S_ji − S_jj), computed on R. Also report the fraction of row-NN errors (i → j) for which the 2-cycle condition S_ij + S_ji > S_ii + S_jj holds.
  - Under the i.i.d. null, ρ_rec ≈ 0.
  - Prediction: higher reciprocity means more errors the LAP cannot fix, so a smaller gain.
- **P3. Planted/null separation versus an i.i.d. surrogate.** These use labels, so they are analysis only.
  - Estimate √n·B(P̂, Q̂) from the diagonal versus off-diagonal entries of R (or the Gaussian ζ = μ²/2 against log n).
  - Monte Carlo an i.i.d. matrix with the same P̂ and Q̂ and compare its NN and Hungarian accuracy to the real ones. The residual is the part of the gap due to dependence or structure.
  - Add an effective-dimension measure (participation ratio of the embeddings, or the canonical-correlation spectrum of the EEG–image pairs, as in [DCK19]) to test point 4.
