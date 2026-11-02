## Reviewer vxam (follow-up)

**Q1 — Foundation Model Encoders.**

**Answer:**
We are glad to learn that the additional experiment on LaBraM addressed the reviewer's concerns. The LaBraM baseline, the fine-tuning protocol, the per-fold numbers and the discussion will be included in the appendix of the revision.

**Q2 — Additional Augmentations.**

**Answer:**
We did not explore other subject-mixing augmentations in the making of the paper. The omission we mentioned in our previous answer concerned the values of the $\\alpha$ parameter controlling the Beta distribution, which we swept but eventually did not report. As stated in our first answer, we agree that this should have been reported, and we also accept the reviewer's broader point that a wider range of subject-mixing augmentations should have been tested, to establish whether the reported gain is specific to SubjectMix or can be elicited by a wider family of cross-subject mixing schemes. We report below a comparison across such a family, together with non-mixing augmentations (smoothing, noise) that serve as controls. Temporal smoothing is an established EEG augmentation (Zhang et al., 2025). We include it because the gain attributed to SubjectMix could be caused indirectly by signal smoothing from averaging EEG responses. The second control replaces SubjectMix's perturbation $(1-\\lambda)(\\mathbf{x}\_{s',i} - \\mathbf{x}\_{s,i})$ by isotropic Gaussian noise of the same per-sample norm. The magnitude of the perturbation is therefore held fixed and only its direction is randomised, which separates the effect of perturbing a trial by a given amount from the effect of perturbing it towards another subject's response to the same image. For our explanation to hold, SubjectMix should yield a larger gain than smoothing or noise injection alone.

We also add MixCo (Kim et al., 2020) as one additional subject-mixing control. MixCo is the analogue of SubjectMix without the same-stimulus constraint: it mixes two trials with a coefficient drawn from the same Beta distribution, and makes the mixture a partial positive of both source images with weights $\\lambda$ and $1-\\lambda$. When the two trials answer to the same image, these two targets coincide and the MixCo objective reduces exactly to SubjectMix's. Therefore, the two augmentations differ in one respect only: whether the mixed trials share the stimulus. Finally, we report SubjectMix extended beyond pairs of subjects, mixing $k$ subjects with weights drawn from a Dirichlet distribution of the same concentration in place of the Beta distribution used for pairs.

We compare each augmentation on all ten LOSO folds with encoder, schedule and seed held fixed. Best top-1, 200-way, plain cosine, averaged over the ten folds, with the difference against the no-augmentation control:

$$
\\begin{array}{lccc}
\\hline
\\text{Arm} & \\text{Mixing} & \\text{Top-1} & \\Delta \\\\
\\hline
\\text{Norm-matched noise} & \\text{none} & 27.20 & -3.40 \\\\
\\text{Temporal smoothing} & \\text{none} & 29.55 & -1.05 \\\\
\\text{No augmentation} & \\text{none} & 30.60 & \\text{--} \\\\
\\text{MixCo} & \\text{across stimuli} & 30.75 & +0.15 \\\\
\\text{SubjectMix, 2 subjects (proposed)} & \\text{same stimulus} & \\mathbf{35.90} & \\mathbf{+5.30} \\\\
\\text{SubjectMix, 3 subjects} & \\text{same stimulus} & 34.65 & +4.05 \\\\
\\text{SubjectMix, 5 subjects} & \\text{same stimulus} & 35.10 & +4.50 \\\\
\\hline
\\end{array}
$$

Neither norm-matched noise nor temporal smoothing improves on the baseline, showing that the gain from SubjectMix must come from another mechanism. Similarly, MixCo fails to improve the baseline significantly, showing that cross-subject mixing in general is not sufficient: it is cross-subject mixing of the same stimulus that provides the invariance. Finally, extending SubjectMix beyond pairs improves on the baseline at every group size tested, and all of these variants seem to present the same gain.

Due to time constraints, this table is run on a single seed. In the revised paper, we will run all experiments on three seeds to control for initialisation variability.

**Q3 — Generalization Across Modalities and Tasks.**

**Answer:**
We thank the reviewer for expressing this concern and for pointing out that our claim about the applicability of SAGE to other modalities was not supported by any experiment. We answer the three sub-questions in turn:

1 - We consider the retrieval pipeline of MindEye2 (Scotti et al., 2024) on the NSD dataset (Allen et al., 2022) to be the most natural candidate for an fMRI replication of our result. It matches the THINGS-EEG-2 setting closely, with the exception that the evaluation is a 300-way retrieval instead of 200-way, the candidates being drawn from a larger gallery of 1000 images. It is one of the few works (that we know of) presenting a zero-shot retrieval pipeline for fMRI, and has become a reference in the fMRI decoding literature.

2 - The reason is simple: we had no fMRI retrieval pipeline available at the time of the rebuttal, and we assumed that building an end-to-end pipeline aligning raw fMRI to images was out of reach within the submission timeline. On reviewing the literature, however, we found that MindEye2 releases fine-tuned checkpoints, so that their published retrieval numbers can be reproduced without any training.

3 - As the reviewer points out, the discussion period permits and encourages additional experiments. We agree that this one would substantially support the generality claim we made, and since the released MindEye2 checkpoints allow SAGE-TTA to be applied without retraining on fMRI data, we allocated our remaining time before the end of the rebuttal to this endeavor. What remains missing is a replication of SAGE-zero-shot on the fMRI modality, which would require retraining from scratch on the raw fMRI signals and is clearly out of scope in this context. We will emphasize in the revision that, while SAGE-TTA is now demonstrated on two modalities, we make no claim that SubjectMix transfers to fMRI. The revision will state this clearly and explicitly, and will present it as an open question requiring further experiments.

**Protocol.** The MindEye2 test protocol uses a candidate set of 1000 test images shared across subjects, from which 300 candidates are drawn 30 times; both image and brain retrieval accuracies are computed on each draw and averaged. We use MindEye2 (1 hour) as our baseline, since the other reported setting (40 sessions) is at ceiling (98.8) and leaves no room for a transductive correction. We reproduce the published numbers with MindEye2's own evaluation code: across 30 pools of 300 candidates drawn from the 1000 test images, we obtain an averaged top-1 of 78.9 / 57.5 against the published 79.0 / 57.4 for image and brain retrieval respectively. We additionally test SAGE-TTA on the harder 1000-way setting, retrieving against the full test gallery instead of sub-sampled 300-candidate pools. As in the original paper, only subjects 1, 2, 5 and 7 are evaluated. The transductive adaptation protocol is identical to the one used for THINGS-EEG-2, with one exception: since MindEye2 retrieval embeddings are 256 tokens of 1664 dimensions, the retrieval metric is a cosine similarity on the flattened 425,984-dimensional vector, in which an explicit rotation cannot be formed. We therefore project these embeddings into a shared PCA subspace of rank 512 before fitting and applying the Procrustes step. CSLS and Sinkhorn, however, act directly on the $300 \\times 300$ similarity matrix and require no adaptation. As before, all hyperparameters are selected by leave-one-subject-out cross-validation over the four subjects.

**Results.** Mean over subjects 1, 2, 5 and 7. The 300-way columns follow the MindEye2 protocol, with 95% intervals over the 30 pools.

$$
\\begin{array}{ccc|cccc}
\\hline
\\text{Whitening} & \\text{CSLS} & \\text{Alignment} & \\text{Image 300} & \\text{Brain 300} & \\text{Image 1000} & \\text{Brain 1000} \\\\
\\hline
\\text{yes} & \\text{yes} & \\text{yes} & \\mathbf{92.7} \\pm 0.5 & \\mathbf{92.8} \\pm 0.5 & \\mathbf{84.8} & \\mathbf{84.9} \\\\
\\text{--} & \\text{yes} & \\text{yes} & 92.2 \\pm 0.6 & 92.3 \\pm 0.6 & 84.1 & 84.3 \\\\
\\text{yes} & \\text{--} & \\text{yes} & 92.2 \\pm 0.5 & 92.3 \\pm 0.5 & 84.8 & 84.8 \\\\
\\text{yes} & \\text{yes} & \\text{--} & 85.5 \\pm 0.6 & 87.1 \\pm 0.5 & 76.4 & 78.3 \\\\
\\text{--} & \\text{--} & \\text{--} & 78.9 \\pm 0.6 & 57.5 \\pm 0.9 & 68.4 & 44.4 \\\\
\\hline
\\end{array}
$$

SAGE-TTA gives +13.8 on image retrieval and +35.3 on brain retrieval at 300-way, and +16.4 / +40.5 at 1000-way. Geometric alignment dominates at +7.2 on image, against +0.5 each for whitening and CSLS. The two preprocessing stages are close to interchangeable at this operating point.
