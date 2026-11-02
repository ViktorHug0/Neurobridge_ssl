# TTA_RESEARCH: literature scan on label-free test-time adaptation under the Track-1 constraints (2026-10-20)

This is a companion to TTA_LOG.md. It covers literature only: no experiments were run for it. Gains quoted below
are the ones the authors report, on their own tasks, which are mostly motor-imagery (MI) classification or image
classification. None of them is a 200-way zero-shot regression-to-retrieval task scored over aggregated repeats.
Expect the gains to shrink or vanish when transferred to our setting.

## Our constraints, restated as filters

- **C1 stateless:** one 64-trial batch at a time, with no memory across calls. This rules out every method that keeps
  running statistics or a buffer across batches (online EA/RA, T-TIME, UnMix-TNS, NOTE, RoTTA, ARM streaming).
- **C2 mixed persons:** about 11 unknown people per batch, about 6 trials each, with no ids. Any method that needs a
  per-domain reference (EA, RA, SPD-BN, SPDIM, ARM's per-domain context) has to infer that reference from about 6
  trials, or from a mixture.
- **C3 no labels and no gallery:** we never see the 200 candidate images, so every gallery-side hubness fix (CSLS,
  inverted softmax, QB-Norm, NNN, DBNorm, Sinkhorn, SATTC) is unavailable. Only transforms of our own output vectors
  remain.
- **C4 aggregation:** the scorer computes `sum_i y_i` over about 80 repeats per (person, image), then L2-normalises
  and ranks. Zero-mean per-trial noise is averaged away. What remains is bias (the person offset and the mapping
  distortion) and the relative vote weights `||y_i||`. A method that only reduces per-trial variance buys almost
  nothing. A method that improves **weights** or removes **bias** can help.

C4 is the lens used throughout. Most published TTA methods improve single-trial accuracy, which is the wrong target
for us.

---

## 1. EEG/BCI cross-subject TTA

### 1a. Euclidean / Riemannian alignment (EA, RA, re-centering, RPA)
- **What it is:** whiten each subject's trials by `R^{-1/2}`, where `R` is the mean of that subject's trial
  covariances. RA uses the Riemannian mean instead, and RPA adds stretching and rotation.
  [Zanini et al. 2018 / Rodrigues et al. 2019 RPA](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1049985/full);
  systematic deep-learning evaluation in [Junqueira et al. 2024, arXiv 2401.10746](https://arxiv.org/abs/2401.10746).
- **Reported gains:** +4.33 pp for shared multi-subject models on MI, with more than 70% faster convergence
  (Junqueira). Wimpff et al. report cross-subject results on BCIC-IV-2a: source-only 57.4, EA 62.5, RA 63.6
  ([arXiv 2311.18520](https://arxiv.org/abs/2311.18520)).
- **Data needed:** a per-subject reference. Junqueira and later work use the first **24** trials, citing the
  "16 to 32 trials" guidance ([arXiv 2606.16462](https://arxiv.org/pdf/2606.16462)). Wimpff uses a FIFO buffer of
  **32**. The Riemannian transfer study finds re-centering effective "as soon as a few samples" of the target are
  seen, but "a few" is still tens of trials for 22 channels.
- **Fit: poor.** About 6 trials per person per batch gives a rank-deficient 32x32 covariance. A batch-level reference
  is a mixture of 11 people, which is the batch whitening that already failed for us. A fingerprint-routed reference
  is close to the neighbour whitening that also failed. Per-trial EA, using the trial's own covariance, removes the
  spatial structure the decoder relies on. **Not recommended.** The literature is consistent that EA's benefit
  depends on a clean per-subject reference, and that is exactly what we cannot have.

### 1b. Adaptive BN, with or without EA (Wimpff et al. 2024, BCI conference)
- **Gains (BCIC-IV-2a cross-subject):** source 57.4, BN-adapt 64.3, RA(EMA)+BN 65.4, plus entropy minimisation
  67.3 ([arXiv 2311.18520](https://arxiv.org/html/2311.18520v2)).
- **Batch:** a buffer of 32 trials **from one subject**. The paper notes that very small buffers are not enough for
  BN-1 statistics, and that mixing in source statistics (BN-0.5) helps.
- **Fit: poor.** BN statistics computed on mixed-domain batches are known to break. In UnMix-TNS Table 1, test-BN
  reaches 90.0% error under mixed, label-correlated batches, against 57.3% for the unadapted source model
  ([arXiv 2401.08328](https://arxiv.org/html/2401.08328v2)). That matches our AdaBN failure. Nothing to add.

### 1c. T-TIME (Li et al., IEEE TBME 2024) and the dual-stage follow-up (Duan et al. 2025)
- **T-TIME:** EA'd source models, an ensemble of classifiers, and per-trial updates by conditional-entropy
  minimisation plus adaptive marginal-distribution regularisation, run online on one subject's stream
  ([arXiv 2412.07228](https://arxiv.org/abs/2412.07228)). Third-party numbers (from the BFT paper's table, EEGNet
  backbone): Zhou2016 85.35, BNCI2014001 79.22, HighGamma 77.42.
- **Dual-stage alignment** ([arXiv 2509.19403](https://arxiv.org/html/2509.19403v1)): online EA (`omega = 500`
  running average), EMA BN, and one self-supervised step per trial. It reports +3.6 pp on MI and +4.9 pp on SSVEP
  over source-only, and +0.7 to +1.3 over T-TIME. Plain Tent without EA *hurts* (-1.8 pp).
- **Fit: none.** Both are stateful per-subject streams, both are classification-only (entropy is undefined without
  a label space), and both depend on EA. The useful takeaway is that **naive Tent on EEG hurts**, which is also the
  conclusion of the 2026 EEG foundation-model TTA benchmark: gradient TTA "often degrade[s] performance … gradient-based
  approaches particularly prone to heavy degradation; optimization-free methods showing greater stability"
  ([NeuroAdapt-Bench, arXiv 2604.16926 / MLHC 2026](https://arxiv.org/abs/2604.16926)).

### 1d. SPD / manifold methods (SPDDSMBN, SPDIM, ICLR 2025)
- **What it is:** SPDIM learns one SPD-manifold parameter per target domain by information maximisation, on top of
  TSMNet with domain-specific SPD batch norm. The ICLR slides report about +5% on sleep staging
  ([arXiv 2411.07249](https://arxiv.org/abs/2411.07249)).
- **Fit: none.** It needs domain-grouped target data and a classifier.

### 1e. NeuroTTT (2025): test-time training of EEG foundation models
- **What it is:** a self-supervised head (or Tent on the norm layers) updated per test sample, with batch size 1.
  Cross-subject BCIC-IV-2a: CBraMod 0.503 to 0.567 balanced accuracy (Tent variant), LaBraM 0.411 to 0.451
  ([arXiv 2509.26301](https://arxiv.org/html/2509.26301v2)).
- **Fit: weak.** Batch-1 TTT is compatible with C1/C2. But the gains are on linear-probe foundation models with a
  classification head, measured per trial (C4 discounts that). The cost would be several backward passes per trial
  inside the grader's predict loop. **Not recommended** given the benchmark evidence above and our own failure with
  few-shot head fine-tuning.

**Section-1 bottom line:** the EEG TTA literature assumes one subject per stream, a buffer of at least 24 to 32
trials, and a classification head. None of that holds for us. Our fingerprint-PC regression is already a better match
to C2 than anything published here.

---

## 2. Query-side hubness reduction without a gallery

- **CSLS / inverted softmax / QB-Norm / NNN / DBNorm / globally corrected (GC):** all of these rescore each gallery
  item `g` by its similarity to a bank of queries. The bank can come from the **training set**: QB-Norm on MSR-VTT
  reaches R@1 14.9 to 17.3 with a 60k training-set querybank, against 17.5 with the test-set bank
  ([Bogolin et al. CVPR 2022](https://arxiv.org/abs/2112.12777)). NNN reports up to +7.1 image recall for CLIP on
  COCO ([Chowdhury et al. EMNLP 2024](https://arxiv.org/abs/2410.24114)). Dinu et al.'s GC uses unlabeled "pivot"
  queries ([arXiv 1412.6568](https://arxiv.org/abs/1412.6568)). Dual-bank Sinkhorn notes that query-bank-only
  estimates of hubness are biased because the bank differs from the targets
  ([arXiv 2508.02538](https://arxiv.org/abs/2508.02538)). Gains are large, but **every one of these needs the gallery
  similarities**. Under C3 they are out.
- **What survives under C3 is only a linear function of the query.** The score is `<q/|q|, g>`. The hub term in CSLS,
  `r_g`, is a per-gallery offset, and a query can only act on it through a term linear in `g`:
  `r_g ≈ a + <g, h>` turns into `q' = q/|q| - beta*h`. That is **centering**, which we already do with half the batch
  mean, plus a linear reweighting of directions, which we already do by shrinking the top-3 PCs. Quadratic hub
  structure (`g^T A g`) cannot be reached from the query side at all. So the query-side hubness design space is
  essentially exhausted by what we already ship.
- **Two residual ideas, both cheap and both likely small:**
  1. *Static plus batch centre.* Estimate `h` from a large **training-prediction bank** (QB-Norm's finding that a
     training bank matches a test bank is the justification) and mix it with the batch mean:
     `h = a*mean_batch + (1-a)*h_static`. Optionally fit `h` to predict proxy-gallery hubness (`r_g` computed on held-out
     *training* images with validation predictions, as in NNN), which gives the best linear approximation of CSLS.
  2. *Un-shrinking regression outputs.* Ridge/MSE-style mappings shrink predictions toward the mean, and the items
     nearest the mean become hubs ([Shigeto et al. ECML 2015](https://arxiv.org/abs/1507.00825); Lazaridou et al. ACL
     2015 show that max-margin training reduces this). The query-side version rescales low-variance output directions
     up to the image-embedding spectrum. This is close to our failed static whitening and Procrustes maps, so expect
     little.
- **Fit:** stateless and label-free. Expected gain: well under 0.5 pp. Worth one evening on the cache, no more.

---

## 3. Robust aggregation and reliability weighting (most relevant to C4)

Our scorer's raw sum is a weighted mean whose weights are the output norms. Three bodies of literature address
exactly this problem: fuse N noisy embeddings of the same identity, where per-item quality varies.

- **Face template aggregation (set-based recognition).**
  - *Probabilistic Face Embeddings* (Shi and Jain, ICCV 2019) predicts a per-dimension variance, and fusion is the
    precision-weighted mean `sum(mu_i/sigma_i^2)/sum(1/sigma_i^2)`
    ([arXiv 1904.09658](https://arxiv.org/abs/1904.09658)).
  - *MagFace* (CVPR 2021) trains so that the **pre-normalisation magnitude is the quality**: magnitude increases
    monotonically with the cosine to the class centre ([arXiv 2103.06627](https://arxiv.org/abs/2103.06627)).
  - *Spherical Confidence Learning* (CVPR 2021) uses an r-radius vMF with closed-form pooling
    ([CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf)).
  - *vMF loss* (Scott et al. ICCV 2021): with spherical losses, the embedding norm is the best correct-vs-wrong
    separator they tested ([arXiv 2103.15718](https://arxiv.org/abs/2103.15718)).
  - **Key fact for us:** for vMF observations of a common direction, the posterior natural parameter is
    `sum_i kappa_i mu_i`. If each output vector is `y_i = kappa_i * mu_i`, then the **scorer's raw sum is the
    Bayes-optimal fusion**. So the right train-time objective makes `||y_i||` a calibrated concentration.
- **ERP weighted averaging:** classical EEG work shows that SNR-weighted or robust averaging of repeated trials beats
  the plain mean. Davila and Mobin, IEEE TBME 1992 ([PubMed 1592399](https://pubmed.ncbi.nlm.nih.gov/1592399)); the
  trimmed and tanh estimators of Leonowicz et al., J Neurosci Methods 2005; and robust weighted averaging, reported
  at 45% lower RMSE than the original implementation
  ([Soton eprint](https://eprints.soton.ac.uk/475258/1/1_s2.0_S174680942300109X_main.pdf)). The evidence is old, but
  it directly concerns repeated noisy EEG trials.
- **Ensemble and augmentation agreement as a weight:**
  - BFT (backprop-free EEG TTA, 2026) weights K augmented or MC-masked predictions with a learned ranker. It reports
    +1.3 pp (BFT-A) and +0.6 to +1.0 (BFT-D) over EEGNet on MI, matching T-TIME without gradients. The authors note
    that the gains are often within one standard deviation ([arXiv 2601.07556](https://arxiv.org/html/2601.07556v1)).
  - UEC weights ensemble embeddings by estimated uncertainty, but it is an unreviewed ICLR-2026 submission with
    self-reported gains ([arXiv 2507.20718](https://arxiv.org/abs/2507.20718)).
  - Evidence that **agreement-based weights** help **aggregated retrieval** is thin.
- **Caveat specific to us:** a high-agreement or high-norm trial can be one that collapsed onto the population prior,
  meaning it is confident but carries no information. Agreement must be measured **after** centering, otherwise it
  rewards hub-like outputs.
- **Fit: good.** Per-trial weights are stateless and label-free, and C4 makes them one of the few levers that
  survive aggregation. `tta_lab.py weighted()` already has `norm:g`, `agree:g` and an `oracle:g` bound (cosine to the
  target). **Run the oracle first:** if `oracle` gains less than about 1 pp, drop this whole section.

---

## 4. Test-time augmentation for EEG decoders

- **Evidence:**
  - Cropped decoding (Schirrmeister et al. 2017) averages hundreds of overlapping-crop predictions per trial. It is
    part of why the deep and shallow ConvNets match FBCSP (84.0 vs 82.1 on MI), but no clean ablation of crop
    averaging alone is published ([arXiv 1703.05051](https://arxiv.org/abs/1703.05051)).
  - BFT-A (noise, amplitude scaling, frequency shift, sliding window, K=12) gives about +1.3 pp on MI
    ([arXiv 2601.07556](https://arxiv.org/html/2601.07556v1)).
  - MEMO (single-sample marginal-entropy TTA with augmentation) gives +1 to 8% on image-classification robustness
    ([arXiv 2110.09506](https://arxiv.org/abs/2110.09506)). That is vision, with classification, and with gradients.
- **Typical size:** about +0.5 to 1.5 pp on single-trial EEG classification.
- **Fit: as a denoiser, near zero.** Averaging K shifted copies reduces per-trial variance, and C4 already averages
  about 80 repeats. As a **reliability signal**, it is plausible: the per-trial variance across shifts, or across
  members, feeds Section 3. `tta_cache.py` already caches shifted outputs (`shifts=`). A cheap check is whether
  `1/var_shift` correlates with the oracle cosine on the cache.
- **Risk:** the 120 Hz signal at ±1 to 3 samples is 8 to 25 ms. The visual ERP is latency-sensitive, so a shift
  augmentation the model was not trained with may bias outputs rather than just add noise.

---

## 5. Amortised / learned batch-level adaptation

- **ARM** (Zhang et al., NeurIPS 2021, [arXiv 2007.02931](https://arxiv.org/abs/2007.02931)):
  - *ARM-CML:* a context network is averaged over the unlabeled batch and fed to the predictor. *ARM-BN:* BN
    statistics are meta-trained on per-domain batches.
  - Worst-case / average accuracy, ERM vs ARM-CML: MNIST 74.5/93.6 vs 88.0/96.3; FEMNIST 62.4/79.1 vs 70.9/86.4;
    Tiny-ImageNet-C 20.3/41.9 vs 29.1/43.3. On CIFAR-10-C, CML is no better than ERM on average.
  - **The ablation that matters:** training with uniformly mixed batches (no per-domain batches) drops MNIST
    worst-case from 88.0 to 63.5. The benefit comes from batches whose composition matches the test batches.
  - Streaming: near-full accuracy after about 50 points.
- **Successors:** Meta-DMoE ([arXiv 2210.03885](https://arxiv.org/abs/2210.03885)); GeneralizeFormer, a transformer
  that generates BN and classifier parameters from the target batch (WACV 2025); in-context domain adaptation by
  transformers, a theory workshop paper ([arXiv 2405.16819](https://arxiv.org/abs/2405.16819)).
- **Fit: structurally good, empirically unproven.**
  - We can meta-train on **simulated grader batches**: 64 trials, about 11 training persons, about 6 trials each.
    That is ARM's key ingredient (train batches distributed like test batches), applied to our mixture rather than
    to a single domain.
  - **Post-hoc version, which keeps the members frozen:** a small permutation-equivariant set network (DeepSets or
    a 2-layer Set Transformer) over the 64 trials. Its per-trial input is `[member outputs, fingerprint LDA
    projection]`. It outputs a per-trial offset `delta_i` and a log-weight `w_i`, and the final output is
    `exp(w_i) * (y_i - delta_i)`.
  - The loss is the **scorer's own**: sum the outputs per (person, image) over a simulated stream, L2-normalise,
    then apply an InfoNCE loss against the training-image gallery.
  - This strictly generalises our hand pipeline (centering, PC shrinkage, fingerprint-PC regression, vote weights).
    It also learns the weights of Section 3 jointly.
- **Risks:**
  - (a) The number of training persons per fold is small (tens), so the set network can memorise identities. Cross-validate on the cv4
    folds, never on training persons.
  - (b) Our identity-embedding grouping failed because purity was low. A set network that implicitly does soft
    grouping may fail for the same reason. Start with the network constrained to the hand pipeline's degrees of
    freedom (a learned `a` for centering, learned PC shrink factors, learned fingerprint-regression gains, learned
    weights) before giving it free-form offsets.
  - (c) There is no published evidence in EEG.

---

## Batch-size requirements at a glance

| Method | Needs | Our batch offers | Fits? |
|---|---|---|---|
| EA / RA / re-centering | about 24 to 32 trials of **one** subject | about 6 per person, mixed | no |
| AdaBN / Wimpff BN-adapt | about 32 trials of one subject | mixed 64 | no (TBN collapses on mixtures) |
| T-TIME, dual-stage, UnMix-TNS, NOTE | a stateful stream | stateless | no |
| SPDIM / SPD-DSBN | domain-grouped data | no ids | no |
| NeuroTTT / MEMO | batch 1, gradients | yes, but costly | weak |
| QB-Norm / NNN / CSLS / IS | gallery similarities | no gallery | no (only linear query-side parts) |
| Reliability weights (PFE, MagFace, vMF, agreement) | per trial | yes | **yes** |
| TTA shifts / MC masks | per trial | yes | yes (as a weight signal) |
| ARM-CML / set network | a batch distributed like training batches | 64 mixed, simulable | **yes** if meta-trained on simulated mixtures |

---

## Top 5 for us, ranked

1. **Learned vote weights, trained against the scorer's own aggregation** (Section 3; face-template literature plus
   the vMF argument).
   - **Why first:** C4 discounts any per-trial denoising, but not weights. E51 already showed that the weights
     matter (forcing equal weights costs 0.05 to 2.1). The face-set literature shows that a learned, calibrated norm
     is the right aggregation weight.
   - **Implementation:** start post-hoc on `tta_cache` with a tiny head `w(features)`, where the features are the
     per-member norms, member agreement after centering, the fingerprint's distance to the training-person
     manifold (LDA-space Mahalanobis), and `1/var` across shifts. Train it on training-person streams with the
     aggregated InfoNCE loss and evaluate on the cv4 folds.
   - **Gate:** the `oracle:g` bound in `tta_lab.py`. Plausible gain: +0.3 to 1.5 pp. Weak external evidence: no
     EEG retrieval paper does this.
2. **ARM-style set network meta-trained on simulated 64-trial, 11-person batches** (Section 5).
   - **Why:** it is the only family whose training assumption we can make true. It subsumes centering, PC shrink,
     fingerprint regression and weights in one learned, stateless map.
   - **Implementation:** constrained parameterisation first (risk b), with the aggregated loss.
   - **Gain:** unknown. ARM reports +1 to 14 worst-case points on image benchmarks, while the mixed-batch ablation
     shows how much batch composition matters. Highest ceiling and highest overfitting risk.
3. **Agreement / augmentation-variance weights without training** (Sections 3 and 4).
   - **Why:** zero training, and `agree:g` plus shifted-output caching already exist.
   - **The new parts:** compute agreement **after** centering, and add cross-shift variance.
   - **Evidence:** BFT +0.6 to 1.3 on MI classification, often within one standard deviation.
   - **Expected gain:** 0 to 0.5 pp. Run it first because it costs seconds, and it also tells us whether #1 has
     features to learn from.
4. **Linearised CSLS: centring on a static training-prediction bank mixed with the batch mean** (Section 2).
   - **Why:** QB-Norm shows training-set banks work as well as test banks. It is a one-parameter addition to what we
     ship.
   - **Expected gain:** at most about 0.3 pp, because the gallery-free design space is mostly used up.
5. **Per-trial TTA averaging (time shifts ±1 to 2 samples, channel dropout) as a denoiser** (Section 4).
   - **Why last:** the literature gain (+0.5 to 1.5 pp, single-trial classification) should mostly be averaged away
     by the scorer's roughly 80 repeats, and untrained shifts may bias the ERP latency. Include it only if the
     ensemble has spare inference budget, and only if a 4-fold cache check shows a gain.

**Not recommended**, with evidence that it fails or does not apply: EA/RA in any form (needs at least 24 single-subject
trials; batch and neighbour versions already failed), AdaBN-family methods (they collapse on mixed batches,
UnMix-TNS Table 1), entropy-minimisation TTA such as Tent, T-TIME, SAR or MEMO (classification-only and stateful,
and they hurt EEG foundation models in NeuroAdapt-Bench), and gallery-side hubness methods (no gallery).

## Honesty notes
- No paper studies label-free adaptation for **cross-subject EEG-to-image retrieval with mixed-subject stateless
  batches**. The closest is SATTC (CVPR 2026), and it works on the gallery side
  ([CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SATTC_Structure-Aware_Label-Free_Test-Time_Calibration_for_Cross-Subject_EEG-to-Image_Retrieval_CVPR_2026_paper.html)).
- The rankings above come from our own C4 argument, not from external evidence.
- The EEG TTA gains (+1 to 10 pp) are all single-subject-stream MI classification from small cohorts of 9 to 54
  subjects. Several are within seed noise.
- I read abstracts plus the key tables for T-TIME (third-party numbers), Wimpff, the dual-stage method, BFT, ARM,
  QB-Norm, UnMix-TNS and NeuroTTT. For MagFace, PFE, SCF, Shigeto, the ERP weighted averaging work and UEC I read
  abstracts or summaries only.
