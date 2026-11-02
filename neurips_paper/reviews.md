# Metareview of Submission 28770

The paper presents a method to tackle the performance drop in cross-subject EEG-to-image retrieval. The method handles new subjects using two modes: a training-time strategy (SAGE-zero-shot) that employs a cross-subject mixing protocol to promote subject invariance, and a test-time adaptation strategy (SAGE-TTA) that frames cross-subject retrieval as a calibration problem to align unlabeled test samples with candidate images.

The reviewers identified the following key weaknesses. The method's innovation/novelty is seen to be moderate, as it primarily integrates existing components like Mixup, contrastive learning, whitening, CSLS, Sinkhorn matching, and Procrustes alignment (Reviewer Rekw, Reviewer 6S7i). Moreover, Reviewer 6S7i points out that the strongest performance gains heavily depend on a strong, potentially unrealistic closed-set protocol (requiring batch access to the full query and candidate sets with a one-to-one matching structure). Additionally, Reviewer Rekw states that the claim of geometric misalignment being the primary culprit for cross-subject degradation lacks enough evidence. Reviewer Rekw points out that the aggregation operator  used in the SubjectMix framework is not clearly defined in the text, and it is not clear how it successfully preserves stimulus-related features while suppressing subject-specific variations. Reviewer 6S7i states that the distinction between zero-shot subject generalization and transductive test-set calibration is blurred in some places, and the text needs to acknowledge that the transductive calibration relies heavily on prior access to the full test query set and candidate image set, which introduces an implicit information risk (and also limits its practicality) if not carefully handled.

During the rebuttal, the authors need to evaluate SAGE-TTA under more realistic test scenarios, such as sequential streaming data or non-one-to-one/partial-candidate sets, to prove it isn't simply exploiting the closed-set structure. They should also provide direct structural evidence (e.g., embedding visualizations or distribution analyses) to support the geometric misalignment claim. They should clarify hyperparameter selection protocols and analyze dataset-specific performance discrepancies (e.g., lower gains on MEG and AllJoined) to verify reproducibility and robustness. The authors also need to provide a precise mathematical definition and implementation details for the aggregation operator  in SubjectMix to ensure reproducibility. Moreover, they should conduct a detailed failure-mode analysis as a function of test sample size, candidate-set size, base encoder quality, or dataset differences to show exactly when the alignment pipeline breaks down. Finally, the authors should state whether they intend to open-source their code and release pretrained models to support community verification of their empirical findings.

# Reviews — Submission 28770

*SAGE: Subject-Agnostic EEG-to-Image Retrieval via Subject-Invariant Representation Learning and Transductive Geometric Alignment*

**Score summary**

| Reviewer | Quality | Clarity | Significance | Originality | Rating | Confidence |
|----------|:-------:|:-------:|:------------:|:-----------:|--------|:----------:|
| Rekw | 3 (good) | 3 (good) | 3 (good) | 2 (not good) | 3 — Borderline reject | 3 |
| 6S7i | 3 (good) | 3 (good) | 3 (good) | 3 (good) | 3 — Borderline reject | 3 |
| vxam | 3 (good) | 3 (good) | 3 (good) | 3 (good) | 4 — Borderline accept | 3 |

---

## Review 1 — Reviewer Rekw

- **Submitted:** 26 Jun 2026 at 05:51 (modified 23 Jul 2026 at 18:36)
- **Contribution Type:** General (most submissions fall into this type)
- **Readers:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Rekw

### Summary

This paper addresses the challenging problem of cross-subject EEG-to-image retrieval, where performance drops substantially on unseen subjects. The authors propose SAGE, which combines SubjectMix-based subject-invariant representation learning with a transductive test-time geometric alignment strategy (SAGE-TTA). Experiments show significant improvements over prior methods. Overall, the paper highlights geometric misalignment as a key factor limiting cross-subject generalization in EEG decoding and presents an effective framework to address it.

### Strengths and Weaknesses

This work addresses the important problem of cross-subject EEG-to-image retrieval, where performance drops substantially on unseen subjects. The paper is clearly written and proposes a clean framework that combines subject-invariant representation learning (SAGE-zero-shot) with transductive geometric alignment (SAGE-TTA). The experimental evaluation is comprehensive, and the reported improvements over prior methods are substantial. The ablation studies further suggest that both SubjectMix and the test-time alignment pipeline contribute to the final performance.

However, I have several concerns:

- **Limited methodological novelty.** Many components are adapted from existing techniques such as Mixup, contrastive learning, whitening, CSLS, Sinkhorn matching, and Procrustes alignment.
- **Strong transductive assumptions.** The setting assumed by SAGE-TTA is relatively strong and may not reflect realistic online EEG decoding scenarios, which raises questions about the practical significance of the reported gains.
- **Insufficient direct evidence for the central claim.** While the paper identifies geometric misalignment as the primary cause of cross-subject degradation, this claim is mainly supported by downstream performance improvements and would benefit from more direct evidence.

Overall, the paper is technically solid and empirically strong, but the novelty and the assumptions behind the transductive evaluation warrant further discussion.

### Questions

1. The aggregation operator in SubjectMix is not clearly defined. How is it implemented in practice, and why does it preserve stimulus-related information while suppressing subject-specific variation?
2. SAGE-TTA assumes access to *N* EEG queries and (*N* or 200) candidate images from a completely unseen subject. The underlying assumptions are unclear. Does adaptation require a one-to-one correspondence between EEG samples and candidate images (i.e., at least *N* paired matches even in the *N*-to-200 setting)? If so, obtaining such paired data from a new subject would be inconsistent with the definition of a completely unseen subject. If not, it is unclear what assumption enables EEG–image alignment at test time, and how the alignment problem is well-posed when candidate images have no corresponding EEG observations or when the matching structure is unknown or noisy.
3. The proposed method achieves very large gains on THINGS-EEG-2 but much smaller improvements on MEG and AllJoined-1.6M (EEG). Could the authors provide some insight into this discrepancy?
4. Do the authors plan to release the code and pretrained models?
5. The paper argues that geometric misalignment is the primary cause of cross-subject performance degradation. Beyond downstream accuracy improvements, could the authors provide more direct evidence (e.g., embedding visualizations or distribution analyses) to support this claim?

### Scores

| Criterion | Score |
|-----------|-------|
| Quality | 3: good |
| Clarity | 3: good |
| Significance | 3: good |
| Originality | 2: not good |
| **Rating** | **3: Borderline reject** — Technically solid paper where reasons to reject (e.g., limited evaluation) outweigh reasons to accept (e.g., good evaluation). Please use sparingly. |
| **Confidence** | **3** — Fairly confident. Possible that some parts were not fully understood or some related work is unfamiliar; math/other details not carefully checked. |

- **Limitations:** Yes.
- **Ethical Concerns:** No or very minor ethics concerns only.
- **Paper Formatting Concerns:** No major formatting, anonymity, or policy violations identified.
- **Code of Conduct Acknowledgement:** Yes
- **Responsible Reviewing Acknowledgement:** Yes

---

## Review 2 — Reviewer 6S7i

- **Submitted:** 24 Jun 2026 at 18:06 (modified 23 Jul 2026 at 18:36)
- **Contribution Type:** General (most submissions fall into this type)
- **Readers:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 6S7i

### Summary

This paper studies the cross-subject generalization problem in EEG-to-image retrieval, where models that perform well within a subject often degrade substantially when evaluated on unseen subjects. The authors propose SAGE, a two-stage framework for subject-agnostic EEG-image alignment. During training, SAGE uses SubjectMix, which mixes EEG responses from different subjects viewing the same stimulus, together with stimulus-grouped multi-positive contrastive learning to encourage subject-invariant EEG representations. At test time, SAGE-TTA treats the unlabeled EEG queries from a new subject and the candidate image embeddings as two point sets, and performs transductive geometric calibration using whitening, CSLS, Sinkhorn-based soft assignment, and Procrustes alignment. The method is evaluated on EEG-to-image retrieval benchmarks such as THINGS-EEG-2, with additional analyses including ablations, encoder-grid experiments, sample-size studies, and comparisons between zero-shot cross-subject retrieval and transductive test-time adaptation.

### Strengths and Weaknesses

This paper studies cross-subject generalization in EEG-to-image retrieval. The central observation is that existing methods can achieve strong intra-subject performance, but their accuracy drops substantially under leave-one-subject-out (LOSO) cross-subject evaluation. The authors propose SAGE, which has two components: a training-time SubjectMix strategy combined with multi-positive contrastive learning to reduce subject-specific information, and a test-time transductive geometric alignment procedure that treats the unlabeled EEG query set and candidate image embeddings as two point sets and aligns them using whitening, CSLS, Sinkhorn assignment, and Procrustes alignment.

**Quality.** The experimental evaluation is reasonably complete. The authors report main results on THINGS-EEG-2, include a stricter LOSO-val setting, and provide component ablations, encoder-grid experiments, and test-time sample-size analysis. The SAGE-zero-shot results suggest that SubjectMix is useful for improving cross-subject representations, and the SAGE-TTA results indicate that part of the cross-subject error can be corrected through geometric calibration. However, the strongest results come from a transductive closed-set setting that requires access to the full unlabeled test query set and the full candidate image set, with a strong implicit one-to-one matching structure. These results should not be interpreted as ordinary new-subject generalization or real-time EEG decoding.

**Clarity.** The paper is generally understandable. The method pipeline and experimental setup are mostly clear, and the authors explicitly discuss the information-leakage issue in the commonly used test-set checkpoint selection protocol and additionally report LOSO-val results. However, the distinction between SAGE-zero-shot and SAGE-TTA should be emphasized more carefully. In particular, the paper should avoid giving the impression that transductive set-level calibration is equivalent to a deployable per-trial decoding setting.

**Significance.** The significance of the problem is clear: cross-subject generalization is a central challenge for EEG decoding and BCI applications. The paper demonstrates an interesting finding that unlabeled test-set geometry can substantially improve closed-set retrieval performance. At the same time, the practical scope of this finding is limited. Many realistic BCI scenarios will not provide the complete candidate set and the full batch of unlabeled queries in advance, so the deployment relevance of SAGE-TTA should be stated more cautiously.

**Originality.** Moderate. SubjectMix is a reasonable same-stimulus cross-subject mixup strategy, while whitening, CSLS, Sinkhorn matching, and Procrustes alignment are established tools from embedding alignment, domain adaptation, and point-set registration. The novelty lies mainly in transferring and integrating these ideas into cross-subject EEG-to-image retrieval and showing that they work well in a closed-set retrieval setting. This is a useful contribution, but not a particularly strong fundamental algorithmic innovation.

**Overall.** The paper has a clear motivation, reasonably complete experiments, and strong empirical results under its chosen evaluation setting. My main concern is that the strongest claims depend on a special transductive closed-set protocol. I encourage the authors to weaken claims about broadly resolving the cross-subject gap or enabling deployment, more clearly separate zero-shot subject generalization from transductive test-set calibration, and further evaluate whether the method remains effective in more realistic sequential or partial-candidate scenarios.

### Questions

1. Please distinguish more clearly between the claims supported by SAGE-zero-shot and those supported by SAGE-TTA. SAGE-zero-shot is closer to genuine new-subject generalization, whereas SAGE-TTA relies on access to the full unlabeled test query set and the candidate image set. Which claims apply only to transductive closed-set retrieval, and which should be understood as general cross-subject decoding claims?
2. Please evaluate SAGE-TTA under more realistic test-time scenarios. For example, what happens if EEG trials arrive sequentially, or if the candidate image set is incomplete or not in a strict one-to-one correspondence with the query set? Such experiments would clarify whether the method mainly benefits from the closed-set matching structure.
3. Please provide a more detailed failure-mode analysis for transductive alignment. The gains are large on THINGS-EEG-2 but much more limited on AllJoined-1.6M and THINGS-MEG. Can the authors analyze when the method works or fails — e.g., as a function of base encoder quality, number of test samples, candidate-set size, subject variability, or modality differences?
4. Please clarify how the SAGE-TTA hyperparameters were selected and whether any tuning used test-set information. In particular, whitening shrinkage, CSLS neighborhood size, Sinkhorn temperature, and the number of Procrustes refinement steps may affect the results. If these parameters are sensitive across datasets, the robustness and reproducibility of the reported gains should be interpreted more cautiously.
5. My score would increase if the authors more strictly delimit the claims supported by the transductive setting and provide additional evidence in sequential, partial-candidate, or non-one-to-one retrieval scenarios showing that SAGE-TTA is not merely exploiting the closed-set test structure. Conversely, my score would decrease if the rebuttal does not clarify these setting limitations or continues to equate transductive calibration with general new-subject generalization.

### Scores

| Criterion | Score |
|-----------|-------|
| Quality | 3: good |
| Clarity | 3: good |
| Significance | 3: good |
| Originality | 3: good |
| **Rating** | **3: Borderline reject** — Technically solid paper where reasons to reject (e.g., limited evaluation) outweigh reasons to accept (e.g., good evaluation). Please use sparingly. |
| **Confidence** | **3** — Fairly confident. Possible that some parts were not fully understood or some related work is unfamiliar; math/other details not carefully checked. |

- **Limitations:** Not fully. The authors acknowledge that SAGE-TTA requires batch access to test pairs and a one-to-one matching structure, which is an important limitation.
- **Ethical Concerns:** No or very minor ethics concerns only.
- **Paper Formatting Concerns:** No.
- **Code of Conduct Acknowledgement:** Yes
- **Responsible Reviewing Acknowledgement:** Yes

---

## Review 3 — Reviewer vxam

- **Submitted:** 24 Jun 2026 at 07:50 (modified 23 Jul 2026 at 18:36)
- **Contribution Type:** General (most submissions fall into this type)
- **Readers:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer vxam

### Summary

The paper identifies a gap in image retrieval from EEG signal: the models' performance significantly drops between within- and across-subject retrieval tasks on benchmarks like THINGS-EEG-2. To close this gap, the paper introduces two EEG/image-encoder-agnostic techniques to increase cross-subject generalization:

1. **Training time.** A subject-mixing strategy that mixes EEG signals across pairs of subjects and contrasts their embeddings to the target image embedding. The mixing strategy is combined with multi-positive contrastive learning (SupCon loss).
2. **Test time (TTA).** A method that iteratively aligns the EEG query embedding to the frozen target image embedding. It starts by applying ZCA whitening on the EEG embeddings, then iteratively runs a soft Procrustes alignment using the similarity matrix between the embedding spaces (with CSLS to correct hubness and Sinkhorn to normalize).

### Strengths and Weaknesses

**Strengths**

1. Cross-subject generalization is well identified as a gap and well motivated, with a proposal of two methods to solve it at different stages of the pipeline (training and test time). Both methods significantly improve top-1 and top-5 compared to previous approaches.
2. The methods are agnostic to the EEG or image encoder and are tested across several combinations of them with consistent improvement. An ablation of most pipeline components is provided, including trying alternatives to Sinkhorn.
3. While providing results on three datasets (THINGS-EEG-2, THINGS-MEG, AllJoined-1.6M), it identifies a bias in previous models' evaluation and proposes LOSO-val (adding a validation subject so as not to optimize performance on test results) to improve it, with potential positive impact for the community.
4. It compiles good additional materials, and the discussion section is qualitative — especially the acknowledgment of the TTA one-to-one matching structure's limitation while providing use cases in BCI.

**Weaknesses**

1. EEG encoders based on foundation models (trained across many subjects) are not tested, while they could provide good cross-subject representation.
2. The subject-mixing framework would benefit from a more detailed empirical analysis or comparison with other augmentations.
3. The method does not propose generalization across EEG-to-image tasks or brain-signal modalities.
4. The highest gain observed is the TTA technique and is still limited to an *N*-query-to-*N*-target setting (the one-to-one matching structure). For both methods, especially TTA, performance degrades in a few-query versus *N*-targets regime and would probably not scale to higher *N* than 200. These limit its significance for downstream BCI deployment.

### Questions

Ordered by priority — major points that, if addressed, could increase the score.

**Q1 — Foundation model encoders.**
Is there a reason to limit the framework from using EEG foundation models? If not, could you test a foundation-model encoder in the retrieval task, without and with the two introduced techniques? If tested, please be careful to pick a model that does not use any of your evaluation data in its pretrain corpus to prevent leakage (e.g., REVE [1] might have used THINGS-EEG-2). The benefit would be to see whether a large set of pretrain subjects helps generalize across subjects on this retrieval task, and to what extent the two introduced methods help those models in this evaluation setting.

**Q2 — Additional augmentations.**
Did you try other subject-mixing augmentations within the proposed framework, or try modifying the value of the Beta distribution? An additional empirical analysis of the subject-mixing augmentation (e.g., tuning it or using another distribution) would help understand sensitivity to its parametrization. Testing new augmentations would inform whether SubjectMix generalizes or is entangled with the specific augmentation introduced.

**Q3 — Generalization across modalities or tasks.**

- The framework does not seem specific to the EEG brain modality. Would it be possible to generalize to fMRI?
- Would subject mixing be useful to improve cross-subject generalization to other EEG-to-image tasks? If so, could you test it?

Providing additional generalization across modality or EEG-to-image tasks would greatly improve the significance of the proposed method.

**Minor comments**

- The representation-learning section introduces notations and concepts that are not used afterward (e.g., the decomposition).
- In Eq. (4): the notation might be intended differently, given how the ensemble is denoted.
- Line 155: please clarify the notation, since one symbol is an ensemble of subjects and another an ensemble of pairs of subjects.
- What value is used for the relevant parameter?
- Eq. (8) would benefit from a more precise definition of the sum in the denominator. Is it over all subject-and-image pairs?
- The result presented in C.2 would benefit from more detail than the caption of Figure 4 to understand how the subject identity is decoded.
- Figure 3 is not explicitly referenced.

*[1] El Ouahidi, Yassine, et al. "REVE: A foundation model for EEG — adapting to any setup with large-scale pretraining on 25,000 subjects." Advances in Neural Information Processing Systems 38 (2026): 22541–22577.*

### Scores

| Criterion | Score |
|-----------|-------|
| Quality | 3: good |
| Clarity | 3: good |
| Significance | 3: good |
| Originality | 3: good |
| **Rating** | **4: Borderline accept** — Technically solid paper where reasons to accept outweigh reasons to reject (e.g., limited evaluation). Please use sparingly. |
| **Confidence** | **3** — Fairly confident. Possible that some parts were not fully understood or some related work is unfamiliar; math/other details not carefully checked. |

- **Limitations:** Yes.
- **Ethical Concerns:** No or very minor ethics concerns only.
- **Paper Formatting Concerns:** No.
- **Code of Conduct Acknowledgement:** Yes
- **Responsible Reviewing Acknowledgement:** Yes