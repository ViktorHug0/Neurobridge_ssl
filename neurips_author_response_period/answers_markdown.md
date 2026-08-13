## Reviewer Rekw

### Questions

**1.** The aggregation operator $\\mathcal{A}$ in SubjectMix is not clearly defined. How is
$\\mathcal{A}$ implemented in practice, and why does it preserve stimulus-related information while
suppressing subject-specific variation?

**Answer:** 
The operator $\\mathcal{A}$ is defined abstractly in Section 3 and only instantiated in Section 5.2,
which makes it easy to miss; we will move the instantiation next to the definition. In every reported
experiment $\\mathcal{A}$ is a convex combination of two raw EEG trials recorded from two
*different* subjects viewing the *same* stimulus,

$$\\widetilde{\\mathbf{x}}\_{s,s',i} = \\lambda\\,\\mathbf{x}\_{s,i} + (1-\\lambda)\\,\\mathbf{x}\_{s',i},
\\qquad \\lambda \\sim \\mathrm{Beta}(\\alpha\_{\\mathrm{mix}}, \\alpha\_{\\mathrm{mix}}),$$

with $\\alpha\_{\\mathrm{mix}} = 0.5$. Each batch holds $M$ stimuli with $K = 9$ EEG responses each, one per training
subject; within a stimulus group every trial is paired with a trial from a different subject, one
$\\lambda$ is drawn per pair, and the mixture is formed on the raw $C \\times T$ tensor before the encoder.

The asymmetry is what preserves stimulus content while attenuating subject content. Because only the EEG side is mixed and the image target $\\mathbf{v}\_i$ is fixed, the loss asks the encoder to be *invariant* along the segment joining two same-stimulus trials, not to interpolate along it. Appendix A provides a more comprehensive analysis of the invariance mechanism behind SubjectMix.

**2.** SAGE-TTA assumes access to $N$ EEG queries and ($N$ or 200) candidate images from a
completely unseen subject. The underlying assumptions are unclear. Does adaptation require a
one-to-one correspondence between EEG samples and candidate images (i.e., at least $N$ paired matches
even in the $N$-to-200 setting)? If so, obtaining such paired data from a new subject would be
inconsistent with the definition of a completely unseen subject. If not, it is unclear what assumption
enables EEG–image alignment at test time, and how the alignment problem is well-posed when candidate
images have no corresponding EEG observations or when the matching structure is unknown or noisy.

**Answer:** 
We thank the reviewer for voicing this concern which aligns with similar inquiries from reviewer 6S7i. To address these concerns about the practical applications of our framework to a realistic BCI deployment case, we submit an additional experiment to model a realistic deployment scenario. 

**Protocol.** We take the original EEG queries made of 80 averaged recordings and divide them
into $B$ distinct queries, each made of either 5, 10, 20 or 40 averaged repetitions (respectively corresponding to $B = 16, 8, 4, 2$). Test queries from the unseen subject arrive one at a time from this bank, in a random order and with unknown
corresponding stimulus. The calibration is re-fitted and scored at regular intervals on all the trials available up to that point (50 times per experiment in total). Because the order is random, the query set does not match the candidate set until late in the stream, and queries are progressively duplicated, breaking the one-to-one matching. Moreover, to better asses the usefulness of our method in diverse deployment scenarios, we consider a range of menu size K ($K = 25, 50, 100, 150, 200$), by randomly sub-sampling from the original bank of 200 candidates. We measure both the calibration time necessary to beat the baseline, as well as the final gain obtained once all the available test samples have been used for calibration. We report the averaged results over 3 random seeds.

**Results.**

$$
\\begin{array}{lcccc|cccc}
\\hline
 & \\text{Minutes to beat baseline} &  &  &  & \\text{Top-1 gain once calibrated} &  &  &  \\\\
\\text{Reps per query} & 5 & 10 & 20 & 40 & 5 & 10 & 20 & 40 \\\\
\\hline
K=25 & 1.1 & 1.9 & 2.4 & 4.0 & +23.3\\,(55\\%) & +28.3\\,(55\\%) & +26.7\\,(46\\%) & +19.8\\,(31\\%) \\\\
K=50 & 1.6 & 2.7 & 4.3 & 6.9 & +20.3\\,(68\\%) & +31.7\\,(80\\%) & +28.8\\,(59\\%) & +27.2\\,(49\\%) \\\\
K=100 & 3.7 & 4.8 & 7.5 & 10.1 & +9.6\\,(45\\%) & +22.9\\,(75\\%) & +30.4\\,(78\\%) & +30.1\\,(65\\%) \\\\
K=150 & 4.8 & 6.4 & 8.0 & 9.6 & +4.9\\,(30\\%) & +15.9\\,(67\\%) & +26.0\\,(85\\%) & +27.9\\,(76\\%) \\\\
K=200 & 11.7 & 4.3 & 8.5 & 10.7 & +2.9\\,(21\\%) & +12.8\\,(63\\%) & +24.1\\,(90\\%) & +30.3\\,(94\\%) \\\\
\\hline
\\end{array}
$$

Adaptation pays within one to eleven minutes of recording in every one of the twenty settings. Once the calibration stream is exhausted, the gain reaches +30.3 points on the full 200-item menu, and exceeds +20 points across most of the grid. The regime where the method yields the smallest gains is a weak base against a large menu, at 5 repetitions with $K \\geq 100$, where the encoder leaves too little structure to refine. We acknowledge this as a limit of the method, and we will state it in the paper.

**A progressive scheduler.** The crossover is governed less by when the calibration becomes useful than by when it stops being harmful: while the buffer covers only a fraction of the menu, the rotation is estimated from too few correspondences, and applying it at full strength falls below the unadapted baseline. We therefore repeat the experiment with the rotation damped to strength $\\alpha(t) = 0.8\\,u/(u+1.5)$, where $u = t/K$ is the ratio of arrivals to menu size. Both $t$ and $K$ are known at decoding time, so this requires no labels and no held-out data.

$$
\\begin{array}{lcccc|cccc}
\\hline
 & \\text{Minutes to beat baseline} &  &  &  & \\text{Top-1 gain once calibrated} &  &  &  \\\\
\\text{Reps per query} & 5 & 10 & 20 & 40 & 5 & 10 & 20 & 40 \\\\
\\hline
K=25 & 1.1 & 1.9 & 2.4 & 3.7 & +24.1\\,(57\\%) & +28.0\\,(55\\%) & +26.6\\,(45\\%) & +20.1\\,(31\\%) \\\\
K=50 & 1.6 & 2.7 & 4.3 & 5.3 & +20.9\\,(70\\%) & +31.3\\,(79\\%) & +29.1\\,(59\\%) & +27.2\\,(49\\%) \\\\
K=100 & 2.7 & 4.3 & 6.9 & 8.0 & +10.7\\,(50\\%) & +23.5\\,(77\\%) & +30.7\\,(78\\%) & +29.5\\,(63\\%) \\\\
K=150 & 4.8 & 3.2 & 6.4 & 3.2 & +6.1\\,(37\\%) & +16.9\\,(72\\%) & +26.6\\,(87\\%) & +28.5\\,(78\\%) \\\\
K=200 & 5.3 & 3.2 & 3.2 & 5.3 & +4.2\\,(30\\%) & +14.1\\,(69\\%) & +25.2\\,(94\\%) & +31.1\\,(97\\%) \\\\
\\hline
\\end{array}
$$

The schedule never delays the crossover: it is reached earlier in thirteen of the twenty settings and at the same checkpoint in the other seven, with the final gain preserved or improved. The benefit grows with menu size, the regime where a given number of arrivals covers least of the candidate set: averaged over repetition counts, the time to beat the baseline falls by $3\\%$, $10\\%$, $16\\%$, $39\\%$ and $52\\%$ for $K=25$ to $K=200$. The least favourable setting above, 5 repetitions against 200 candidates, improves from +2.9 points after 11.7 minutes to +4.2 points after 5.3 minutes.

A single schedule serves every subject, repetition count and menu size, and its two constants were selected on the $K=200$ runs alone, so these figures are a floor rather than a tuned result. Adapting the schedule to the operating point, or estimating it online from the observed coverage, should improve it further; we leave this to future work.

**3.** The proposed method achieves very large gains on THINGS-EEG-2 datasets but much smaller
improvements on MEG and AllJoined-1.6M (EEG). Could the authors provide some insight into this
discrepancy?

**Answer:** 
We suggest the difference reflects the overall quality of the EEG recording, in terms of base predictive power, rather than anything specific to a dataset: when the base retrieval is stronger, the transductive step provides a stronger relative improvement. This can be shown on THINGS-EEG-2 alone, by weakening the base encoder capabilities and holding everything else fixed. Rebuilding the 200 test queries from fewer repetitions per image lowers the base accuracy, and we can monitor how the relative gain from TTA drops along with it.

$$
\\begin{array}{lccccc}
\\hline
\\text{Test repetitions} & 10 & 20 & 40 & 60 & 80 \\\\
\\hline
\\text{SAGE-zero-shot (top-1)} & 20.5 & 27.1 & 32.4 & 34.3 & 35.9 \\\\
\\text{SAGE-TTA (top-1)} & 24.1 & 39.4 & 54.9 & 61.9 & 68.3 \\\\
\\text{Relative gain from TTA} & +18\\% & +45\\% & +69\\% & +80\\% & +90\\% \\\\
\\hline
\\end{array}
$$

This is consistent with how the proposed method works. It refines the initial predictions using the geometry of the task at hand, but fails to learn an efficient structure when the base predictions are noisy and not confident.

**4.** Do the authors plan to release the code and pretrained models?

**Answer:** 
Yes. The full training and evaluation code, the configuration files for every reported run, and the pretrained checkpoints for each subject will be released publicly.

**5.** The paper argues that geometric misalignment is the primary cause of cross-subject
performance degradation. Beyond downstream accuracy improvements, could the authors provide more
direct evidence (e.g., embedding visualizations or distribution analyses) to support this claim?

**Answer:** 
We thank the reviewer for pointing out that this claim is not supported by the evidence we present. The
origin of cross-subject variability in EEG remains open and debated. The recent survey of Li et al.,
*Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods* (Progress in
Biomedical Engineering, 2026; arXiv:2604.27033), covers the range of mechanisms and mitigation
strategies that have been proposed for this gap without settling on one, and Lin et al., *The Identity Trap in EEG Foundation Models: A Diagnostic Audit* (arXiv:2606.06647, 2026), audit how strongly subject identity persists in representations trained to be transferable. We take no definite position about the origin of cross-subject degradation, and simply propose a method based on geometric alignment that mitigates the accuracy gap by leveraging both the unlabeled subject's signal and the structure of the decoding task. We will amend the paper as follows to reflect this new, less ambitious claim: we will remove the causal framing in the "Related Work" paragraph on geometric misalignment, where we currently present hubness as a local consequence of "a more fundamental geometric mismatch". We will also amend the accompanying claim that neuro-physiological variability systematically displaces EEG embeddings, and the corresponding sentence in the Introduction, so that the paper describes geometric alignment as an effective correction rather than as evidence about the origin of the gap.

## Reviewer 6S7i

### Questions

**1.** Please distinguish more clearly between the claims supported by SAGE-zero-shot and those
supported by SAGE-TTA. SAGE-zero-shot is closer to genuine new-subject generalization, whereas
SAGE-TTA relies on access to the full unlabeled test query set and the candidate image set. Which
claims apply only to transductive closed-set retrieval, and which claims should be understood as
general cross-subject decoding claims?

**Answer:** 
We apologize if the paper blurs this in places.
Every result reported from the SAGE-zero-shot framework, including the reported improvement over SOTA from $21.8\\%$ to $35.9\\%$ on THINGS-EEG-2, should be read as a claim about general cross-subject decoding capability. Every result reported from the SAGE-TTA framework, including the $77\\%$ figure reported in the abstract, applies only to the transductive closed-set setting. We have identified three passages where the distinction is blurred and will amend them: the closing sentence of the abstract, which presents the $77\\%$ figure as closing the intra-subject gap without restating the transductive setting; the first sentence of the conclusion, which attributes the gap closure to the pipeline as a whole; and panel (c) of Figure 1, which reports the doubling without recalling that it requires a known candidate set. If the reviewer has further passages in mind, we will gladly amend those too.

**2.** Please evaluate SAGE-TTA under more realistic test-time scenarios. For example, what
happens if EEG trials arrive sequentially, or if the candidate image set is incomplete or not in a
strict one-to-one correspondence with the query set? Such experiments would clarify whether the method
mainly benefits from the closed-set matching structure.

**Answer:** 
We thank the reviewer for this request, and we agree that the submission lacked realistic test-time scenarios. To address it, we submit an additional experiment that places the method in all three of the suggested settings at once, so as to reflect the difficulty of real-life BCI deployment.

**Protocol.** We take the original EEG queries made of 80 averaged recordings and divide them
into $B$ queries made of either 5, 10, 20 or 40 averaged repetitions ($B = 16, 8, 4, 2$). Test queries from the unseen subject arrive one at a time from this bank, in a random order and with unknown corresponding stimulus. The calibration is re-fitted and scored at regular intervals on all trials available up to that point (50 times per experiment). Because the order is random, the query set does not match the candidate set until late in the stream, and queries are progressively duplicated, breaking the one-to-one matching. We also vary the menu size ($K = 25, 50, 100, 150, 200$) by sub-sampling the original bank of 200 candidates. We measure the calibration time needed to beat the baseline and the final gain once all test samples have been used, averaged over 3 random seeds.

**Results.**

$$
\\begin{array}{lcccc|cccc}
\\hline
 & \\text{Minutes to beat baseline} &  &  &  & \\text{Top-1 gain once calibrated} &  &  &  \\\\
\\text{Reps per query} & 5 & 10 & 20 & 40 & 5 & 10 & 20 & 40 \\\\
\\hline
K=25 & 1.1 & 1.9 & 2.4 & 4.0 & +23.3\\,(55\\%) & +28.3\\,(55\\%) & +26.7\\,(46\\%) & +19.8\\,(31\\%) \\\\
K=50 & 1.6 & 2.7 & 4.3 & 6.9 & +20.3\\,(68\\%) & +31.7\\,(80\\%) & +28.8\\,(59\\%) & +27.2\\,(49\\%) \\\\
K=100 & 3.7 & 4.8 & 7.5 & 10.1 & +9.6\\,(45\\%) & +22.9\\,(75\\%) & +30.4\\,(78\\%) & +30.1\\,(65\\%) \\\\
K=150 & 4.8 & 6.4 & 8.0 & 9.6 & +4.9\\,(30\\%) & +15.9\\,(67\\%) & +26.0\\,(85\\%) & +27.9\\,(76\\%) \\\\
K=200 & 11.7 & 4.3 & 8.5 & 10.7 & +2.9\\,(21\\%) & +12.8\\,(63\\%) & +24.1\\,(90\\%) & +30.3\\,(94\\%) \\\\
\\hline
\\end{array}
$$

Adaptation pays within one to eleven minutes of recording in all twenty settings. Once the calibration material is exhausted, the gain reaches +30.3 points on the full 200-item menu and exceeds +20 points across most of the grid. The smallest gains come from a weak base against a large menu, at 5 repetitions with $K \\geq 100$, where the encoder leaves too little structure to refine; we will state this limit in the paper.

**A progressive scheduler.** The crossover is governed less by when the calibration becomes useful than by when it stops being harmful: while the buffer covers only a fraction of the menu, the rotation is estimated from too few correspondences, and applying it at full strength falls below the unadapted baseline. We therefore repeat the experiment with the rotation damped to $\\alpha(t) = 0.8\\,u/(u+1.5)$, $u = t/K$ being the ratio of arrivals to menu size. Both $t$ and $K$ are known at decoding time, so this needs no labels.

$$
\\begin{array}{lcccc|cccc}
\\hline
 & \\text{Minutes to beat baseline} &  &  &  & \\text{Top-1 gain once calibrated} &  &  &  \\\\
\\text{Reps per query} & 5 & 10 & 20 & 40 & 5 & 10 & 20 & 40 \\\\
\\hline
K=25 & 1.1 & 1.9 & 2.4 & 3.7 & +24.1\\,(57\\%) & +28.0\\,(55\\%) & +26.6\\,(45\\%) & +20.1\\,(31\\%) \\\\
K=50 & 1.6 & 2.7 & 4.3 & 5.3 & +20.9\\,(70\\%) & +31.3\\,(79\\%) & +29.1\\,(59\\%) & +27.2\\,(49\\%) \\\\
K=100 & 2.7 & 4.3 & 6.9 & 8.0 & +10.7\\,(50\\%) & +23.5\\,(77\\%) & +30.7\\,(78\\%) & +29.5\\,(63\\%) \\\\
K=150 & 4.8 & 3.2 & 6.4 & 3.2 & +6.1\\,(37\\%) & +16.9\\,(72\\%) & +26.6\\,(87\\%) & +28.5\\,(78\\%) \\\\
K=200 & 5.3 & 3.2 & 3.2 & 5.3 & +4.2\\,(30\\%) & +14.1\\,(69\\%) & +25.2\\,(94\\%) & +31.1\\,(97\\%) \\\\
\\hline
\\end{array}
$$

The schedule never delays the crossover: it is reached earlier in thirteen of the twenty settings and at the same checkpoint in the other seven, with the final gain preserved or improved. The benefit grows with menu size, the regime where a given number of arrivals covers least of the candidate set: averaged over repetition counts, the time to beat the baseline falls by $3\\%$ at $K=25$ and by $52\\%$ at $K=200$. The least favourable setting above, 5 repetitions against 200 candidates, improves from +2.9 points after 11.7 minutes to +4.2 points after 5.3 minutes.

A single schedule serves every subject, repetition count and menu size, with constants selected on the $K=200$ runs alone, so these figures are a floor rather than a tuned result. Adapting it to the operating point, or estimating it online from observed coverage, should improve it further.

**3.** Please provide a more detailed failure-mode analysis for transductive alignment. The gains
are large on THINGS-EEG-2 but much more limited on AllJoined-1.6M and THINGS-MEG. Can the authors
analyze when the method works or fails, for example as a function of base encoder quality, number of
test samples, candidate-set size, subject variability, or modality differences?

**Answer:** 

**Base encoder quality.** We agree that the paper would benefit from a more thorough analysis of the limits of transductive alignment, to which the experiment above is a partial answer. The Encoder Grids appendix already reports gains across a wide range of image and EEG encoder architectures of varying strength. For a fixed encoder, reducing the number of averaged test repetitions weakens its predictive power and so simulates a weaker encoder, which means the next point addresses this failure mode as well.

**Number of test samples.** Rebuilding the 200 test queries from fewer repetitions lowers both
the base predictive power of the encoder and the relative gain of the TTA calibration, from $+90\\%$ at
80 repetitions to $+18\\%$ at 10. This is consistent with the smaller gains we report on
AllJoined-1.6M, whose base encoder is weaker.

$$
\\begin{array}{lccccc}
\\hline
\\text{Test repetitions} & 10 & 20 & 40 & 60 & 80 \\\\
\\hline
\\text{SAGE-zero-shot (top-1)} & 20.5 & 27.1 & 32.4 & 34.3 & 35.9 \\\\
\\text{SAGE-TTA (top-1)} & 24.1 & 39.4 & 54.9 & 61.9 & 68.3 \\\\
\\text{Relative gain from TTA} & +18\\% & +45\\% & +69\\% & +80\\% & +90\\% \\\\
\\hline
\\end{array}
$$

**Candidate-set size.** Fitting $N$ queries against $N$ candidates, bijective and $N$-way:

$$
\\begin{array}{lccccc}
\\hline
\\text{Candidate set N} & 25 & 50 & 100 & 150 & 200 \\\\
\\hline
\\text{SAGE-zero-shot (top-1)} & 67.5 & 56.8 & 45.7 & 40.2 & 35.9 \\\\
\\text{SAGE-TTA (top-1)} & 88.0 & 79.9 & 70.3 & 69.4 & 68.3 \\\\
\\text{Relative gain from TTA} & +30\\% & +41\\% & +54\\% & +73\\% & +90\\% \\\\
\\hline
\\end{array}
$$

A larger menu makes retrieval harder but gives the alignment a richer and denser geometry to work with.

**Subject variability.**

$$
\\begin{array}{lcccccccccc}
\\hline
\\text{Subject} & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\\\
\\hline
\\text{SAGE-zero-shot} & 50.5 & 43.0 & 27.0 & 29.5 & 29.5 & 36.0 & 33.5 & 25.0 & 37.0 & 48.0 \\\\
\\text{SAGE-TTA} & 83.0 & 74.0 & 65.0 & 54.0 & 69.5 & 75.0 & 64.0 & 41.0 & 76.0 & 81.5 \\\\
\\text{Relative gain (\\%)} & +64 & +72 & +141 & +83 & +136 & +108 & +91 & +64 & +105 & +70 \\\\
\\hline
\\end{array}
$$

The trend observed above seems to reverse here: weaker subjects benefit more from SAGE-TTA than stronger ones. We suggest this is because SAGE-TTA leverages both the unlabeled signal of the unseen subject and the structure of the task at hand, the second of which compensates for a lack of zero-shot decoding power.

**4.** Please clarify how the SAGE-TTA hyperparameters were selected and whether any tuning used
test-set information. In particular, whitening shrinkage, CSLS neighborhood size, Sinkhorn
temperature, and the number of Procrustes refinement steps may affect the results. If these parameters
are sensitive across datasets, the robustness and reproducibility of the reported gains should be
interpreted more cautiously.

**Answer:** 
We select the transductive hyperparameters by leave-one-subject-out cross-validation, with no access to
the test subject. 

**5.** My score would increase if the authors more strictly delimit the claims supported by the
transductive setting and provide additional evidence in sequential, partial-candidate, or
non-one-to-one retrieval scenarios showing that SAGE-TTA is not merely exploiting the closed-set test
structure. Conversely, my score would decrease if the rebuttal does not clarify these setting
limitations or continues to equate transductive calibration with general new-subject generalization.

**Answer:** 
We thank the reviewer for stating precisely what would raise their assessment, and we agree on both
counts. On delimitation, our answer to Q1 sets out the scope of each stage and lists the passages we
will amend: the 35.9 zero-shot figure is a general cross-subject result, while the transductive
figures hold for a session with a known candidate set. On the additional evidence, the experiment
reported in Q2 places the method in all three of the requested scenarios at once, since testing them
separately would understate the difficulty of real-life BCI deployment. It shows that the method does
not depend on the closed-set structure: the calibrated predictor overtakes the unadapted baseline
within minutes of recording and reaches a relative improvement comparable to the main experiment,
while also quantifying how much recording is needed before calibration becomes useful.

## Reviewer vxam

### Questions

**Q1 — Foundation Models Encoders.** Is there a reason to limit the framework from using EEG
foundation models? If not, could you test a foundation model encoder in the retrieval task without and
with the 2 introduced techniques? If tested, could you be careful to pick a model that does not use any
of your evaluation data in their pretrain corpus to prevent any leakage (e.g., REVE [1] might have used
THINGS-EEG-2). The benefit of this experience would be to see if a large set of pretrain subjects helps
to generalize across subjects on this retrieval task and to what extent the two methods introduced help
those models in this evaluation setting.

**Answer:** 
There is no reason of principle: SubjectMix operates on the raw
EEG recording before the encoder, and the calibration stage operates on final embeddings, so a foundation model can be substituted for the EEG backbone without modifying either. The reason for not including one was that, to our knowledge, all the leading results on this benchmark come from compact encoders trained from scratch, which is coherent with what Liu et al., *EEG Foundation Models: Progresses, Benchmarking, and Open Problems* (ICLR 2026) report. The same work finds linear probing frequently insufficient to expose transferable representations, which is why we adopt full-parameter fine-tuning in the following adaptation protocol. To make sure that SAGE transfers to Foundation model encoders, we ran the experiment the reviewer asks for, using the closest match in existing literature as our baseline.

Following the exact protocol of AVDE (Dai et al., 2026), we fully fine-tune a pretrained LaBraM (5.8M
parameters) in place of our TSConv encoder: the backbone is initialised from its public checkpoint and
trained end-to-end under a contrastive-plus-regression objective ($\\lambda = 0.8$), AdamW, and a cosine
schedule from $2\\times10^{-3}$ to $10^{-5}$. 

We find that both techniques transfer to the foundation model. Averaged over the ten folds, zero-shot top-1 (plain
cosine) is 27.1 without SubjectMix and 31.9 with it; adding SAGE-TTA raises this number to 63.1. Both contributions therefore transfer to a foundation-model backbone.

**Q2 — Additional Augmentations.** Did you try other subject mixing augmentations within the
framework proposed or try to modify the value of the Beta distribution? Providing an additional
empirical analysis of the subject mixing augmentation (e.g., tuning $\\alpha\_{\\mathrm{mix}}$ or using
another distribution) would benefit understanding the sensitivity to its parametrization. Testing new
augmentations would inform about the generalization of the SubjectMix technique or its entanglement to
the specific augmentation introduced.

**Answer:** 
We did, and we agree the submission should have reported it. The mixing coefficient is
$\\lambda \\sim \\mathrm{Beta}(\\alpha\_{\\mathrm{mix}}, \\alpha\_{\\mathrm{mix}})$ with
$\\alpha\_{\\mathrm{mix}} = 0.5$; this also answers the reviewer's minor question about the value of the
parameter.

Sweeping $\\alpha\_{\\mathrm{mix}}$ over two orders of magnitude, on ten LOSO folds with the remaining
configuration held fixed, gives an average top-1 of 33.4, 33.9, 34.5, 35.0 and 35.1 for
$\\alpha\_{\\mathrm{mix}} = 0.01, 0.1, 0.2, 0.5$ and 1.0 respectively, against 32.4 without any
mixing, which demonstrates that the method is not sensitive to this parameter.

**Q3 — Generalization Across Modalities or Tasks.** The framework provided does not seem to be
specific to the EEG brain modality. Would it be possible to generalize to fMRI? Would the subject
mixing be useful to improve cross subject generalization to other EEG-to-image tasks? If so, could you
test it? Being able to provide an additional generalization across modality or EEG to image tasks would
greatly improve the significance of the method proposed.

**Answer:** 
Neither component contains anything EEG-specific. SubjectMix requires only that several subjects be recorded on the same stimuli, and the calibration stage acts on embeddings and is indifferent to their origin. So we can reasonably expect the framework to transfer, and the calibration stage in particular could be applied post hoc to an existing fMRI retrieval pipeline at negligible cost. While we do not have the time to run additional experiments within the initial response window, we agree that measuring the efficiency of the proposed methods on other datasets and benchmarks is an important avenue for future work, and we will mention in the next version of the manuscript.

**Minor comments.**
- The representation learning section introduces notations and concepts that are not used after
like $E\_\\theta = P\_{\\theta\_p} \\circ S\_{\\theta\_s} \\circ T\_{\\theta\_t}$ decomposition.
- In eq (4): $\\mathbf{x}\_{s,s',i}$ notation might be intended to be $\\mathbf{x}\_{(s,s'),i}$ as the
ensemble is denoted as $\\widetilde{\\mathcal{S}} = \\{(s,s')\\}$.
- Line 155: $s \\in \\mathcal{S} \\cup \\widetilde{\\mathcal{S}}$ could you clarify this notation as
$\\mathcal{S}$ is an ensemble of subjects and $\\widetilde{\\mathcal{S}}$ an ensemble of pairs of subjects.
- What value is used for $\\alpha\_{\\mathrm{mix}}$?
- Eq (8) would benefit from a more precise definition of the sum in the denominator
$\\sum\_{(s',j)}$. Is it all the $s'$ subject and $j$ images pairs?
- The result presented on C.2 would benefit from more details than the caption of Figure 4 to
understand how the subject identity is decoded.
- Figure 3 is not explicitly referenced.

**Answer:** 
We thank the reviewer for these additional comments, we will act on all of them.
- *Unused notation.* The encoder decomposition $E\_\\theta = P\_{\\theta\_p} \\circ S\_{\\theta\_s} \\circ
T\_{\\theta\_t}$ is indeed never used again. We will remove it.
- *Eq. (4) and line 155.* The reviewer is right: $\\mathcal{S}$ is a set of subjects and
$\\widetilde{\\mathcal{S}}$ a set of pairs of subjects, so the union $\\mathcal{S} \\cup
\\widetilde{\\mathcal{S}}$ is ill-typed as written. We will use a single index $u \\in \\mathcal{S} \\cup
\\widetilde{\\mathcal{S}}$ and write $\\mathbf{x}\_{u,i}$, with $\\mathbf{x}\_{u,i} =
\\mathcal{A}(\\mathbf{x}\_{s,i}, \\mathbf{x}\_{s',i})$ when $u = (s,s')$.
- *Parameter value.* $\\alpha\_{\\mathrm{mix}} = 0.5$ in every reported experiment, as detailed in
our answer to Q2.
- *Eq. (8) denominator.* The sum runs over all EEG embeddings in the batch, indexed by
(subject or subject pair, stimulus). We will write the index set explicitly.
- *Appendix C.2.* We will describe the subject-identity probe in the text, and not only in the
caption of Figure 4.
- *Figure 3.* The figure will be referenced explicitly in the next version of the manuscript.
