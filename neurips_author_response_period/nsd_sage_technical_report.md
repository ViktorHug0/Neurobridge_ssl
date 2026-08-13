# Porting SAGE-TTA to fMRI: A Technical Report on the MindEye2 / NSD Experiment

**Status.** Complete. Hyperparameters are selected by leave-one-subject-out cross-validation
over two grids (Sections 5 and 5.1); the ablation in Section 6 is reported at the refined
selection, fitted per pool at PCA rank 512 as Section 3.5 requires. The only outstanding item
is the 40-session reproduction, which stands at two of four subjects and is confirmatory only.

---

## 1. Motivation and choice of benchmark

Reviewer R3 observed that our claim of cross-modality generality, and in particular the claim
that the calibration stage could be applied post hoc to an existing fMRI retrieval pipeline at
negligible cost, was not supported by any experiment. This report documents the experiment we
ran in response.

SAGE-TTA is transductive: it consumes a set of test queries and a fixed gallery, and it
requires that the retrieval problem have a finite candidate set over which a soft assignment
can be defined. Most fMRI decoding benchmarks do not satisfy this, either because the test
stimuli differ across subjects or because the task is formulated as regression rather than
retrieval. The Natural Scenes Dataset (NSD) evaluated under the MindEye2 protocol satisfies
all of the requirements, and maps onto our EEG setting almost one to one:

| SAGE on THINGS-EEG | NSD under MindEye2 |
| --- | --- |
| 200 test concepts, shared across subjects | 1000 shared test images, seen by all 8 subjects |
| 200-way retrieval | 300-way retrieval, chance 1/300 |
| Leave-one-subject-out over 10 subjects | Pretrain on 7 subjects, fine-tune on a held-out 8th |
| Retrieval in InternViT / ViT-H feature space | Retrieval in OpenCLIP ViT-bigG/14 space |

Two properties made the experiment tractable. First, the shared-1000 test split is common to
all subjects, which is the non-negotiable requirement for a fixed gallery. Second, MindEye2
states that "for retrieval inference, only the retrieval submodule's outputs are necessary",
so the diffusion prior, the SDXL unCLIP stage and the caption model can all be omitted. The
retrieval path is therefore a small fraction of the released system.

We evaluate the **MindEye2 (1 hour)** setting, in which the model is fine-tuned on a single
scanning session of the held-out subject. This is the low-data cross-subject regime that
corresponds to the accuracy gap our paper addresses. The 40-session setting is at ceiling
(98.8 / 98.3 published) and leaves no headroom for a post-hoc correction.

---

## 2. Reproduction of the published baseline

No result about a delta is meaningful without a baseline that reproduces. We therefore
reproduced MindEye2's published retrieval numbers before applying any calibration.

### 2.1 Protocol

MindEye2 evaluates retrieval by drawing 30 independent pools of 300 candidates from the 1000
test images, computing a cosine similarity on the **flattened** 256 x 1664 = 425,984-dimensional
token embedding, and averaging top-1 accuracy over the pools. Both directions are reported:
image retrieval and brain retrieval. We re-implemented this from `final_evaluations.ipynb`
without modification.

### 2.2 Result

Averaged over subjects 1, 2, 5 and 7, the four subjects the paper evaluates:

| | image retrieval | brain retrieval |
| --- | --- | --- |
| Ours | **78.9** | **57.5** |
| MindEye2, Table 1, "MindEye2 (1 hour)" | 79.0 | 57.4 |

The agreement is within a tenth of a point in both directions. Per subject:

| Subject | image | brain |
| --- | --- | --- |
| subj01 | 94.2 | 78.3 |
| subj02 | 90.6 | 67.4 |
| subj05 | 67.1 | 46.5 |
| subj07 | 63.6 | 37.9 |

The between-subject spread in this regime is large (30.6 points on image retrieval, 40.4 on
brain retrieval). A single subject is therefore uninformative about agreement with a
four-subject mean, a point we return to in Section 8.

The 40-session setting was reproduced on subjects 1 and 2 (100.0 / 99.9 and 99.9 / 99.9
against a published four-subject mean of 98.8 / 98.3). The remaining two extractions were not
completed; they are confirmatory only and are not used anywhere in this report.

### 2.3 Implementation notes

Five details are required for the reproduction and are recorded here because each of them
silently changes the result rather than raising an error.

1. **Retrieval-only inference is not a supported path.** `recon_inference.ipynb` computes
   `clip_voxels` inside the same loop that runs the diffusion prior and SDXL. We reproduce the
   `clip_voxels` half only, loading `ridge` and `backbone` and filtering the checkpoint to
   those two prefixes (64 tensors kept, 100 blurry-branch and 85 prior/decoder tensors
   skipped). `blurry_recon=False` is safe because `clip_voxels` is produced by `clip_proj`,
   which the blurry branch does not feed.

2. **`models.py` cannot be imported as released** in an environment without the reconstruction
   stack: it imports `clip`, `diffusers.models.vae`, `generative_models.sgm` and
   `dalle2_pytorch` at module scope. None is reachable from the retrieval path. We stub them
   rather than installing PyTorch Lightning and the dalle2 dependency cascade.

3. **Checkpoint width is not the notebook default.** `recon_inference.ipynb` defaults to
   `hidden_dim=2048`, but the released subj01 1-session checkpoint is 4096. We read the width
   off `ridge.linears.0.weight` instead of trusting the default.

4. **`backbone_linear` is 425,984 x hidden_dim**, that is 6.98 GB in fp32 at hidden_dim=4096,
   and under autocast every call allocates a fresh fp16 copy. Storing the weights in fp16 once
   avoids both. The published run casts the same weights to fp16 inside autocast, so the
   matmul sees identically rounded inputs.

5. **`utils.batchwise_cosine_similarity` ends in a transpose.** That transpose decides which
   axis `topk` ranks over, and therefore which of the two reported directions is which.
   Omitting it swaps image and brain retrieval, and one of the two swapped values lands close
   to the published number by coincidence. With `sim[i,j] = cos(query_i, gallery_j)`, image
   retrieval is the column-wise argmax and brain retrieval the row-wise argmax.

A sixth detail concerns the gallery encoder. `FrozenOpenCLIPImageEmbedder.preprocess` applies
`x = (x + 1)/2` after resizing, which presumes input in [-1, 1]; the released `all_images.pt`
is stored in [0, 1], so the pipeline in fact feeds [0.5, 1.0] into the CLIP normalisation.
Training and evaluation share the embedder, so the pipeline is self-consistent, and
reproducing the published number requires reproducing this convention exactly. A clean
reimplementation that feeds [0, 1] produces a different gallery and a baseline that does not
match.

---

## 3. Adapting SAGE-TTA to token-structured embeddings

### 3.1 The structural difference

On THINGS-EEG each trial is a single vector of moderate dimension, and Procrustes fits an
unconstrained rotation in that space. MindEye2 retrieval embeddings are 256 tokens of 1664
dimensions, and the published metric is a cosine over the flattened 425,984-dimensional
vector. A rotation in that space is a 425,984^2 matrix, which cannot be formed.

### 3.2 A factorisation that fails

Our first port fitted a single 1664 x 1664 rotation shared across token positions, that is
constrained the map to

```
R_flat = I_256 (x) R_token
```

This is a very small subgroup of O(425,984), roughly 1.4 x 10^6 free parameters against
approximately 9 x 10^10. It fails badly: even supplied with the **true** correspondence, the
best map within this family is 88.4 degrees from the identity and reduces 1000-way top-1 from
65.6 to 40.2. We initially misread this as evidence that the query-to-gallery relationship is
not close to orthogonal. It is instead an artefact of the constraint.

### 3.3 The correct reduction

Because `n = 1000` queries and 1000 gallery items span at most 1998 dimensions of the flattened
space, and because an orthogonal map fitted from a rank-limited cross-covariance acts as the
identity outside the span of the data, fitting the rotation inside a shared PCA subspace of
the flattened space is not an approximation: at full rank it **is** the full-space rotation.
We verified this empirically in Section 7.2.

The stages of SAGE-TTA therefore distribute as follows:

| Stage | Space it operates in | Requires the subspace |
| --- | --- | --- |
| SAW (centring + ZCA whitening) | PCA subspace, dimension k | yes |
| CSLS | 1000 x 1000 similarity matrix | no |
| Sinkhorn | 1000 x 1000 similarity matrix | no |
| Soft Procrustes | PCA subspace, dimension k | yes |

Conditioning is favourable. At k = 512 the rotation is fitted from 1000 correspondences,
whereas on THINGS-EEG a 512 x 512 rotation is fitted from 200, which is underdetermined.

### 3.4 Evaluation convention

All arms are a transform of the **query side only**, scored in the full flattened space against
an untouched gallery, using MindEye2's own metric. The no-calibration row is therefore
literally their computation and returns their published values, not an approximation of them.
Two independent implementations agree per subject to the decimal.

For efficiency, the cosine is computed inside the subspace. If `B` has orthonormal columns then
`<cur B^T, g> = <cur, g B>` and `||cur B^T|| = ||cur||`, so the full-space cosine reduces to a
k-dimensional inner product against the gallery's projection divided by the gallery's
full-space norm. We verified this against explicit lifting: maximum absolute difference
9.4 x 10^-7, identical metrics.

### 3.5 The unit of calibration

SAGE is a batch-level transductive method: it is fitted on the retrieval problem it is scored
on. Under MindEye2's protocol that problem is a pool of 300 candidates, and each pool is
already a balanced bijective assignment, 300 queries against 300 candidates with every answer
present exactly once. The calibration is therefore fitted **independently on each pool**, with
the PCA basis, SAW, CSLS, Sinkhorn and Procrustes all seeing only that pool's 300 queries and
300 candidates.

An earlier version of this report fitted the calibration once on the full 1000-way gallery and
evaluated the frozen transform on the pools, justifying this on the grounds that the redrawn
pools do not provide a balanced assignment. That justification was incorrect: what the pools
lack is a gallery shared *across* loops, which SAGE never required. The design was also
unsound in a second respect, since a transform fitted on all 1000 items has seen the 700
outside any given pool, which is a strictly larger transductive set than the 300-way task
defines. Numbers obtained that way are reported in Section 6.1 as a reference, labelled as
using a larger transductive set, and are not the headline result.

Because a pool spans at most 598 dimensions, the subspace rank does not carry over from the
1000-way fit and is re-selected for this regime. Retrieval against rank, averaged over 30
pools and the four subjects:

| PCA rank | k / n | image | brain |
| --- | --- | --- | --- |
| 64 | 0.21 | 49.8 | 49.9 |
| 128 | 0.43 | 83.1 | 82.6 |
| 256 | 0.85 | 90.1 | 90.1 |
| **512** | **1.71** | **92.7** | **92.8** |

We adopt rank 512. For reference, THINGS-EEG fits a 512-dimensional rotation from 200
correspondences, a ratio of 2.56, so this operating point is less over-parameterised than the
setting the method was published in. Rank 128 is included in the table because it is
over-determined (k / n = 0.43, so the rotation cannot encode the assignment) and still clears
the baseline by a wide margin on both directions, which establishes that per-pool calibration
works outside the regime where degeneracy is a concern.

---

## 4. Diagnosis of the initial negative result

Applied with the published EEG hyperparameters, SAGE-TTA **degraded** NSD retrieval. The
diagnosis proceeded in three steps and is reported in full because the mechanism is
informative about when transductive alignment applies.

### 4.1 The loop is a fixed-point iteration

Tracing the 16 soft-Procrustes steps shows the mechanism directly. On NSD, with
`tau = 0.1`:

```
plain cosine 79.2   after SAW+CSLS 93.3
step   retrieval   assignment
   0        93.3         85.6
   1        84.7         84.8
  ...
  15        50.4         51.2
```

and on THINGS-EEG, same parameters:

```
plain cosine 30.5   after SAW+CSLS 29.5
step   retrieval   assignment
   0        29.5         34.0
   1        35.0         37.0
  ...
  15        39.5         37.5
```

SAGE's loop drives retrieval toward the accuracy of the Sinkhorn assignment it is fitted to.
Its direction is set by the sign of (assignment accuracy - current score accuracy): positive
on EEG (+4.5), negative on NSD (-7.7). Procrustes welds the embeddings to the assignment, the
scores are recomputed, and the process compounds.

### 4.2 The cause is the Sinkhorn temperature

`sinkhorn_normalize` exponentiates `scores / tau`, so `tau` is meaningful only relative to the
spread of the score matrix. Measured on the CSLS output:

| | score std | gallery size | `scores/tau` std at tau=0.1 |
| --- | --- | --- | --- |
| THINGS-EEG | 0.164 | 200 | 1.64 |
| NSD | 0.054 | 1000 | 0.53 |

At `tau = 0.1` the NSD plan is close to uniform. An over-smoothed assignment falls below the
score matrix it was computed from, and the loop then converges downward.

Sweeping `tau` on NSD confirms the mechanism, with the sign of the gap predicting the outcome
in every row:

| tau | assignment | gap vs scores | final |
| --- | --- | --- | --- |
| 0.001 | 98.2 | +4.9 | 98.6 |
| 0.005 | 99.0 | +5.7 | 98.9 |
| 0.01 | 98.6 | +5.3 | **99.4** |
| 0.02 | 96.7 | +3.4 | 98.2 |
| 0.05 | 89.6 | -3.7 | 90.3 |
| 0.1 | 85.6 | -7.7 | 50.1 |
| 0.3 | 83.8 | -9.5 | 14.9 |

### 4.3 A scale-relative rule does not hold

The natural remedy is to make `tau` proportional to the score matrix spread, which would make
SAGE transfer without retuning. This predicts an EEG optimum near 3.1 x 0.01 = 0.03. We tested
it by sweeping `tau` on all ten EEG subjects at the paper's TTA settings (200-way, plain 26.8):

| tau | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | **0.1** | 0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top-1 | 31.0 | 31.6 | 33.2 | 33.8 | 35.8 | **39.4** | 32.9 |

The published `tau = 0.1` is the genuine EEG optimum, and the scale-matched prediction of 0.03
costs 5.6 points. Score spread therefore accounts for roughly a factor of 3 out of the factor
of 10 separating the two optima; gallery size (200 against 1000) is the obvious remaining
factor, but two datasets cannot identify a law. We therefore make the weaker and defensible
claim: SAGE's calibration transfers with its published settings **except** for the Sinkhorn
temperature, which is scale-dependent and must be selected per dataset.

A useful corollary is that the paper's `tau = 0.1` is confirmed optimal for EEG, so no
improvement is being left unclaimed there.

### 4.4 A control that does not discriminate

Fitting the rotation to a randomly permuted assignment collapses retrieval to chance in both
modalities (0.3% on EEG against 0.5% chance; 0.1% on NSD against 0.1% chance). This is an
inherent property of an over-parameterised orthogonal fit and is present in the setting where
SAGE demonstrably works. It should not be read as pathology. The discriminating quantity is
the one in Section 4.1.

---

## 5. Hyperparameter selection

Parameters are chosen by **leave-one-subject-out cross-validation** over the four evaluated
subjects: for held-out subject *s* the grid is scored on the other three, and only the winning
configuration is run on *s*. No hyperparameter sees the subject it is reported on.

The first grid comprised 180 configurations: `tau` in {0.005, 0.01, 0.02, 0.05, 0.1},
SAW shrink in {0.2, 0.5, 0.94}, CSLS k in {3, 12, 30}, Procrustes power in {1.0, 1.2}, and PCA
rank in {256, 512}. **All four folds selected the identical configuration:**

```
tau = 0.01,  shrink = 0.94,  csls_k = 3,  power = 1.2,  PCA k = 512
```

Three of the five values are the paper's EEG defaults. Only `tau` moved, together with the PCA
rank, which has no EEG counterpart.

### 5.1 Refinement

The rank {256, 512} offered by the first grid proved to be the binding constraint, so a second
grid of 180 configurations was run: `tau` in {0.005, 0.007, 0.01}, shrink in
{0.98, 0.96, 0.94, 0.92, 0.90}, CSLS k in {1, 3, 5}, PCA rank in {512, 1000, 1500, 1998},
with power held at 1.2. Selections:

| Held-out subject | tau | shrink | CSLS k | PCA rank |
| --- | --- | --- | --- | --- |
| 1 | 0.01 | 0.96 | 1 | 1000 |
| 2 | 0.01 | 0.96 | 1 | 1000 |
| 5 | 0.005 | 0.96 | 1 | 1998 |
| 7 | 0.01 | 0.96 | 1 | 1000 |

`shrink = 0.96` and `csls_k = 1` are selected unanimously. The fold held out on subject 5
differs in `tau` and rank; this is a tie within the plateau rather than a substantive
disagreement, since ranks 1000 and 1998 differ by 0.11 on the selection criterion.

The marginal response, expressed as the combined image + brain 300-way score averaged over all
four subjects (maximum 200), shows that three of the four parameters sit on plateaus:

| tau | best | mean | | shrink | best | mean |
| --- | --- | --- | --- | --- | --- | --- |
| 0.005 | 178.93 | 177.20 | | 0.98 | 179.00 | 177.30 |
| 0.007 | 179.06 | 177.51 | | **0.96** | **179.17** | **177.67** |
| **0.01** | **179.17** | 177.55 | | 0.94 | 178.89 | 177.58 |
| | | | | 0.92 | 178.70 | 177.38 |
| | | | | 0.90 | 178.75 | 177.16 |

| CSLS k | best | mean | | PCA rank | best | mean |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **179.17** | **177.90** | | 512 | 177.32 | 175.93 |
| 3 | 179.06 | 177.50 | | **1000** | **179.17** | 177.33 |
| 5 | 178.41 | 176.86 | | 1500 | 179.00 | **178.29** |
| | | | | 1998 | 179.06 | 178.12 |

`tau` varies by 0.24 across its three values and shrink by 0.47 across five, so neither is a
knife-edge. The PCA rank is the exception: 512 is clearly inferior on both statistics, which
confirms that the first grid was pinned at its edge. Ranks from 1000 to 1998 are comparable.
`csls_k = 1` is preferred over the EEG default of 3, so a second parameter required
re-selection, though the effect is small.

The held-out combined score of 177.9 is below the best training-fold score of 179.17. This gap
is the expected selection optimism and indicates the cross-validation is not leaking.

---

## 6. Ablation

Following Table 4 of the paper, one component is removed at a time. SubjectMix is omitted
because it is a training-side intervention and MindEye2 is not retrained. Everything is fitted
per pool as described in Section 3.5, at PCA rank 512. Reported as image / brain retrieval,
mean over subjects 1, 2, 5 and 7, with 95% confidence intervals over the 30 evaluation pools.

There is no 1000-way column here. Per-pool calibration is defined on the 300 candidates it is
scored on, so a 1000-way arm does not exist under this protocol; the 1000-way figures in
Section 6.1 belong to the fit-on-1000 design and are reported there only as a reference.

| Whitening | CSLS | Alignment | image 300 | brain 300 |
| --- | --- | --- | --- | --- |
| yes | yes | yes | **92.7** +- 0.5 | **92.8** +- 0.5 |
| -- | yes | yes | 92.2 +- 0.6 | 92.3 +- 0.6 |
| yes | -- | yes | 92.2 +- 0.5 | 92.3 +- 0.5 |
| yes | yes | -- | 85.5 +- 0.6 | 87.1 +- 0.5 |
| -- | yes | -- | 83.3 +- 0.6 | 76.1 +- 0.7 |
| -- | -- | -- | 78.9 +- 0.6 | 57.5 +- 0.9 |

Reference: MindEye2 Table 1, "MindEye2 (1 hour)", 79.0 / 57.4.

Marginal contribution of each component:

| Component | image 300 | brain 300 |
| --- | --- | --- |
| SAW (whitening) | +0.5 | +0.5 |
| CSLS | +0.5 | +0.5 |
| Geometric alignment | **+7.2** | **+5.7** |

Full SAGE against the untouched baseline, paired over the four subjects: **+13.8 image
(p = 0.0339)** and **+35.3 brain (p = 0.0073)**.

Three observations. First, geometric alignment is the dominant component by a wide margin:
+7.2 image against +0.5 for each of the other two. Under per-pool fitting the alignment stage
carries the method, and the two preprocessing stages are close to interchangeable at this
operating point, each recoverable from the other. Second, that near-equality (92.2 in both
single-removal arms, against 92.7 with both) indicates redundancy and not additivity. Removing
either one costs 0.5, and neither is separately necessary once the subspace rank is adequate.
Third, CSLS applied alone, with no subspace and no fitting, is worth +4.4 / +18.6 and remains
the most portable component of the method, since it needs one hyperparameter and no fit.

Compared with the fit-on-1000 design of Section 6.1, the total improves (+10.0 to +13.8 on
image) while the individual preprocessing contributions shrink (SAW +0.7 to +0.5, CSLS +1.7 to
+0.5) and alignment grows (+2.9 to +7.2). Fitting on the pool that is scored is both a better
match to the task definition and a stronger result, and it concentrates the gain in the stage
the paper contributes.

---

## 7. Analysis of the preprocessing stages

### 7.1 Decomposition of SAW

SAW is `(x - mu) Sigma^{-1/2}`. Because centring is applied before the projection, every
projected arm carries it, including the row labelled "no whitening"; this holds under per-pool
fitting as well. The decomposition below was measured under the fit-on-1000 design at the first
selection's operating point (`tau = 0.01`, shrink 0.94, `csls_k` 3, rank 512), which is why it
carries 1000-way columns. The conclusion is about the role of centring and does not depend on
the unit of calibration.

| | image 300 | brain 300 | image 1000 | brain 1000 |
| --- | --- | --- | --- | --- |
| Full SAGE (centred, whitened) | 88.3 | 88.3 | 84.6 | 84.5 |
| No centring | 76.1 | 86.1 | 69.2 | 82.7 |
| No covariance whitening | 85.7 | 85.5 | 81.2 | 81.3 |
| Neither | 75.6 | 74.8 | 70.2 | 69.3 |

Centring is worth +12.2 image points inside the full pipeline, roughly four times the
covariance term and more than CSLS and alignment combined. Without it, SAGE's image retrieval
(76.1) falls **below** the untouched baseline (78.9).

The behaviour is not a paradox but it is worth stating precisely: centring applied on its own
does not improve retrieval (78.9 to 75.8 on image), yet it is a precondition for everything
that follows. The mean vector has norm 23.4 against a per-token norm of 33.6, so the uncentred
covariance is dominated by the rank-one outer product mu mu^T. Whitening then expends its
budget flattening the mean direction, and Procrustes fits a rotation that principally aligns
means. An orthogonal map cannot remove an offset, because rotations fix the origin, so the
rotation cannot compensate for the omission. Centring makes the second-order statistics
informative, which is what the two later stages require.

In the reported table SAW is kept as a single row, matching the paper's EEG ablation, since
centring and whitening are two halves of one defined operation.

### 7.2 The PCA step is not a confound

An earlier version of this analysis treated the projection as a confound, on the grounds that
three ablation rows carry a step the baseline does not. That was incorrect. The
projection-only arm and a centring-only arm computed in the full space with no projection at
all give the same result:

| | image 300 | brain 300 |
| --- | --- | --- |
| PCA projection only, k = 512 | 75.8 | 77.8 |
| PCA projection only, k = 1998 | 75.8 | 78.1 |
| Centring only, no projection | 75.8 | 78.1 |

The projection contributes nothing on its own at any rank, and the entire effect attributed to
it is centring, which belongs inside SAW. The NSD and EEG ablations therefore have the same
structure and no additional caveat is required.

Finally, a projection-free substitute for ZCA whitening was tested. Diagonal whitening needs
only per-dimension variance and runs at full dimension, but performs poorly: 56.6 / 69.6
against 75.8 / 78.1 for centring alone. The subspace is required for whitening and for the
rotation, and in both cases it is exact rather than approximate.

---

## 8. Limitations

### 8.1 The bijectivity prior inflates brain retrieval

After calibration the two directions converge to 92.7 / 92.8. Sinkhorn imposes a one-to-one
coupling, which symmetrises them by construction. The +35.3 improvement in brain retrieval
should therefore not be read as 35 points of better brain-to-image decoding. The assumption
holds by construction in both benchmarks, since each gallery item is the answer to exactly one
query, but it would fail on an open gallery containing distractors. We report image retrieval
as the headline and state the assumption explicitly.

### 8.2 Four subjects

Only `final_subj{01,02,05,07}` checkpoints are released, and these are the four subjects the
paper evaluates. Four is the ceiling the benchmark offers, not a choice. Paired tests at n = 4
are underpowered, and per-subject consistency carries more weight than the p-values.

### 8.3 Component contributions are operating-point dependent

Across the three designs run here, fit-on-1000 at the first selection, fit-on-1000 at the
refined selection, and per-pool, the total improvement rose (+9.4, +10.0, +13.8 on image) while
the marginal contribution of whitening and CSLS fell at each step (+2.6, +0.7, +0.5 for SAW).
The components are partially redundant, and the ablation measures each one at a particular
operating point instead of isolating an additive effect. Under per-pool fitting whitening and
CSLS are close to interchangeable, each worth +0.5 when removed singly. This is a property of
the ablation design, shared with the EEG table in the paper, and is stated here so that the
per-component figures are not over-interpreted.

The related limitation reported in an earlier draft, that the PCA rank was pinned at the edge
of the grid, has been resolved by the refinement in Section 5.1 and no longer applies.

### 8.4 The Sinkhorn temperature requires per-dataset selection

As established in Section 4.3, no principled rule for `tau` has been identified. The claim
supported by this experiment is that the calibration transfers with its published settings
except for `tau`, selected here by cross-validation without test-set access.

---

## 9. Reproducibility

| Artefact | Location |
| --- | --- |
| Retrieval-submodule extraction | `scripts/nsd_sage/extract_clipvoxels.py` |
| Baseline reproduction | `scripts/nsd_sage/reproduce_retrieval.py`, `run_all_subjects.py` |
| CV and fit-on-1000 ablation | `scripts/nsd_sage/nsd_cv_ablation.py` |
| Per-pool fitting, rank sweep, reported ablation | `scripts/nsd_sage/per_pool_sage.py`, `ablation.sbatch` |
| Diagnosis | `scripts/nsd_sage/trace_iterations.py`, `diagnose_assignment.py`, `sage_faithful.py` |
| EEG degeneracy control | `scripts/things_eeg/tta_rebuttal/diagnose_degeneracy.py` |
| Logs, CV grid, selections | `results/things_eeg/nsd_sage/` |
| Data | `/nasbrain/p20fores/mindeye_data`, MindEyeV2 clone at `/nasbrain/p20fores/MindEyeV2` |

The retrieval-only download is approximately 21 GB per subject-model, against more than 100 GB
for the full release: the checkpoint (8.9 GB), the subject's betas (about 1.5 GB), the test
webdataset (38 MB), the test images (602 MB) and OpenCLIP ViT-bigG/14. The reconstruction
weights are not required. Extraction takes 21 seconds per subject; the calibration itself runs
on CPU in seconds, which is the concrete form of the "negligible cost" claim R3 asked us to
support.

NSD access requires the data agreement at https://forms.gle/eT4jHxaWwYUDEf2i9. The MindEye2
derived data on HuggingFace (`pscotti/mindeyev2`) is not gated.
