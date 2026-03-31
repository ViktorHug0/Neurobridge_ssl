# SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval

Qunjie Huang
Yunnan University, China
huangqunjie@stu.ynu.edu.cn

Weina Zhu*
Yunnan University, China
zhuweina@ynu.edu.cn

# Abstract

Cross-subject EEG-to-image retrieval for visual decoding is challenged by subject shift and hubness in the embedding space, which distort similarity geometry and destabilize top- $k$  rankings, making small- $k$  shortlists unreliable. We introduce SATTC (Structure-Aware Test-Time Calibration), a label-free calibration head that operates directly on the similarity matrix of frozen EEG and image encoders. SATTC combines a geometric expert—subject-adaptive whitening of EEG embeddings with an adaptive variant of Cross-domain Similarity Local Scaling (CSLS)—and a structural expert built from mutual nearest neighbors, bidirectional top- $k$  ranks, and class popularity, fused via a simple Product-of-Experts rule. On THINGS-EEG under a strict leave-one-subject-out protocol, standardized inference with cosine similarities,  $\ell_2$ -normalized embeddings, and candidate whitening already yields a strong cross-subject baseline over the original ATM retrieval setup. Building on this baseline, SATTC further improves Top-1 and Top-5 accuracy, reduces hubness and per-class imbalance, and produces more reliable small- $k$  shortlists. These gains transfer across multiple EEG encoders, supporting SATTC as an encoder-agnostic, label-free test-time calibration layer for cross-subject neural decoding. Code is available at https://github.com/QunjieHuang/SATTC-CVPR2026

# 1. Introduction

Decoding visual perception from brain activity is a longstanding goal in neuroscience and machine learning. Non-invasive electroencephalography (EEG) is particularly attractive because it is portable, relatively low-cost, and offers millisecond temporal resolution. Building on these advantages, recent work has demonstrated EEG-based visual classification, zero-shot EEG-to-image retrieval, and image reconstruction with diffusion models [13, 22, 24].

![[SATTC-_Structure-Aware_Label-Free_Test-Time_Calibration_for_Cross-Subject_EEG-to-Image_Retrieval_p1_img1.jpeg]]
Figure 1. Overview of cross-subject EEG-to-image retrieval under leave-one-subject-out (LOSO) evaluation and the proposed SATTC (Structure-Aware Test-Time Calibration) head. Frozen EEG and image encoders produce a similarity matrix whose baseline top- $k$  shortlist is hub-dominated; SATTC performs subject-adaptive whitening, adaptive CSLS, and a structural expert with product-of-experts (PoE) fusion to yield calibrated top- $k$  candidates in a label-free test-time setting.

In EEG-to-image retrieval, an EEG signal is mapped into a joint embedding space and used to retrieve its nearest image neighbors. With the growing availability of large-scale EEG-image benchmarks and powerful cross-modal encoders [17], EEG-to-image retrieval pipelines can now be evaluated at scale, yet a central question emerges: how can we reliably retrieve rich visual semantics from non-invasive EEG signals in realistic cross-subject settings?

Early work on EEG-based visual retrieval largely relied on supervised learning, training subject-specific classifiers or regressors on labeled EEG-image pairs to predict stimulus labels or features. However, such pipelines require costly trial-level annotations, are constrained by limited data, and typically generalize poorly to unseen subjects. As real cross-subject deployments seldom provide labels for new users, unsupervised and zero-shot paradigms have emerged as natural alternatives. Several recent studies adopt encoder-centric unsupervised or zero-shot EEG-to-image paradigms that align EEG and image embeddings via contrastive learning [3, 4, 13, 22, 23, 27]. These works introduce increasingly expressive EEG encoders and report

---

strong zero-shot recognition and retrieval performance on THINGS-EEG benchmarks *[7]*, but their test-time evaluation typically reduces to non-standardized nearest-neighbor search over learned similarity scores in the shared space. In parallel, a separate line of EEG decoding research improves cross-subject robustness by learning subject-invariant or domain-adapted representations through adversarial inference, multi-source marginal distribution adaptation, and plug-and-play domain adaptation modules *[5, 16, 19, 29]*. While effective for representation learning, these methods operate at training time and have not been instantiated as generic, label-free calibration heads *[25, 28]* that can standardize and improve retrieval shortlists in zero-shot cross-subject EEG-to-image settings.

Despite these advances, current unsupervised and zero-shot EEG-to-image retrieval pipelines still suffer from three test-time limitations: (1) Missing structure-aware, label-free test-time calibration. Existing work largely focuses on encoder design while reducing inference to bare nearest-neighbor search, so the joint effect of cross-subject shift and hubness on top-$k$ rankings has not been systematically characterized, nor corrected, at test time. (2) Lack of subject-adaptive, density-aware hubness mitigation. Most pipelines operate in a single globally normalized similarity space with simple $\ell_{2}$ or global whitening, or omit hubness correction altogether. When CSLS variants are used, they typically rely on a globally fixed neighborhood size tuned on labeled validation data, which fails to capture query- and class-specific local densities in strict zero-shot, cross-subject scenarios. (3) Underused structural cues and absent shortlist calibration. Current pipelines treat each query in isolation and mainly report global Top-1/Top-5 or mAP, without leveraging mutual nearest neighbors, bidirectional top-$k$ ranks, or class popularity patterns to diagnose and correct small-$k$ shortlist quality and per-class imbalance. They also lack a reusable test-time head that can be attached to arbitrary cross-subject EEG-to-image encoders to improve Top-1 while preserving Top-5 and mitigating hubness.

To address these limitations, we propose SATTC (Structure-Aware Test-Time Calibration), a label-free calibration head that operates purely on the test-time similarity matrix of frozen EEG and image encoders. We first standardize the inference pipeline with cosine similarities, $\ell_{2}$-normalized embeddings, and candidate whitening, then combine a geometric expert—subject-adaptive whitening of EEG embeddings followed by an adaptive CSLS scheme based on row- and column-wise local densities—with a structural expert that leverages mutual nearest neighbors, bidirectional top-$k$ ranks, and class popularity patterns. The two experts are fused via a simple Product-of-Experts rule to recalibrate EEG–image similarity scores, explicitly mitigating hubness and producing more stable small-$k$ shortlists for downstream zero-shot decoding. Our main contributions are summarized as follows:

- [leftmargin=*]
- We formulate cross-subject EEG-to-image retrieval as a structure-aware, label-free test-time calibration problem, and empirically characterize how subject shift and hubness jointly destabilize top-$k$ rankings in this setting.
- We introduce a geometric expert that combines subject-adaptive whitening of EEG embeddings with an adaptive CSLS scheme, estimating row- and column-wise local densities to derive query- and class-dependent neighborhoods and reduce hubness without global-$k$ tuning.
- We propose a structural expert that operates on the pre-CSLS similarity matrix using mutual nearest neighbors, bidirectional top-$k$ ranks, and class popularity, and fuse it with the geometric expert via a Product-of-Experts model. Experiments on the THINGS-EEG cross-subject benchmark show that our standardized inference baseline already outperforms the original ATM pipeline, and that SATTC further improves Top-1 accuracy over strong fixed-$k$ and adaptive CSLS baselines without harming Top-5 performance, while reducing hubness and per-class imbalance and yielding more reliable small-$k$ shortlists.

## 2 Related Work

### 2.1 EEG-based Visual Decoding and Cross-Subject Generalization

EEG-based visual decoding has progressed from early coarse category recognition to fine-grained visual classification, image retrieval, and reconstruction *[13, 23, 24]*. A common recipe is to learn an encoder that maps short EEG segments into a visual or multimodal embedding space, followed by nearest-neighbor retrieval or regression to image features on datasets such as THINGS-EEG. Despite strong results, most systems are still evaluated in subject-specific or lightly adapted settings, assuming a separate model or labeled data for each target user, which underestimates the difficulty of deployment to unseen users without per-subject labels.

To improve cross-subject robustness, recent work explores subject-independent training, domain-invariant EEG features, and shared latent spaces using adversarial inference, multi-source marginal distribution adaptation, or plug-and-play domain adaptation modules *[5, 16, 19, 29]*. However, these approaches primarily focus on encoder architectures and training losses. They rarely analyze how similarity scores behave at test time under subject shift and hubness, nor how to recalibrate the retrieval geometry once the encoder is fixed. In contrast, we adopt a cross-subject EEG-to-image retrieval setup with frozen encoders and no target labels, and focus on label-free test-time calibration of the EEG-image similarity structure, complementary to representation-learning methods.

---

### 2.2 Label-Free Test-Time Adaptation and Calibration

Test-time adaptation and post-hoc calibration seek to reduce train-test mismatch by updating models or predictions using unlabeled test data *[15, 26]*. Many test-time adaptation (TTA) methods adapt batch-normalization statistics or minimize prediction entropy on streaming test samples, while post-hoc calibration approaches such as temperature scaling and its variants correct miscalibrated classifier confidences after training *[8]*. However, most of these techniques either require labeled validation data for tuning or implicitly rely on pseudo-labels and class-conditional statistics, which can be fragile for noisy, highly variable EEG data in cross-subject scenarios. Moreover, they are typically designed for closed-set classifier logits rather than dense similarity matrices arising in retrieval.

We instead consider a strict label-free test-time regime where the EEG and image encoders are frozen and only the test-time similarity matrix is observable. Our goal is to design a calibration operator that acts directly on EEG–image similarities to stabilize top-$k$ rankings for downstream neural decoding.

### 2.3 Hubness, CSLS, and Structure-Aware Retrieval Priors

High-dimensional nearest-neighbor retrieval is strongly affected by hubness, where a few points appear in the top-$k$ lists of many queries and dominate rankings *[6, 20]*. Cross-domain Similarity Local Scaling (CSLS) mitigates hubness by using local neighborhoods to rescale similarity scores and down-weight globally “popular” candidates *[11]*. Follow-up work refines CSLS or exploits mutual nearest neighbors and bidirectional ranks to discard inconsistent matches and sharpen alignments *[2, 21, 30]*, but typically assumes a fixed neighborhood size and treats hubness as a purely geometric effect. Rare classes in sparse regions can therefore be over-penalized, while dense hubs are not always fully suppressed when $k$ is small.

Structure-aware re-ranking techniques based on graph diffusion *[10]*, mutual $k$NN graphs, or label propagation further exploit relationships among queries and candidates in image and cross-modal retrieval *[2, 30]*. Yet these methods usually rely on labeled data, transductive supervision, or iterative pseudo-labeling, and are not tailored to noisy cross-subject EEG where only small-$k$ shortlists and unlabeled test data are available. Our approach instead keeps the encoders fixed and operates purely on the pre-CSLS similarity matrix, combining an adaptive CSLS variant with query- and class-dependent neighborhoods and a structural prior built from mutual nearest neighbors, bidirectional top-$k$ ranks, and class popularity, fused via a Product-of-Experts formulation to obtain calibrated EEG–image retrieval scores.

## 3 Method

### 3.1 Label-Free Test-Time Calibration: Problem Setup and Overview

We consider cross-subject EEG-to-image retrieval on the THINGS-EEG dataset *[7, 9]*, which provides multichannel EEG recordings while subjects view images from a fixed vocabulary of object categories. Let $Q$ denote EEG queries from unseen test subjects and $C$ the set of candidate image classes. A pretrained EEG encoder $f_{\text{ee g}}$ maps each query $q\in Q$ to an embedding $z_{q}\in\mathbb{R}^{d}$, and an image encoder $f_{\text{img}}$ maps each class $c\in C$ to an embedding $v_{c}\in\mathbb{R}^{d}$. After subject-adaptive whitening and normalization (Section 3.2), we obtain a pre-CSLS similarity matrix

$S_{\text{new}}(q,c)=\psi(\tilde{z}_{q},\tilde{v}_{c})$ (1)

where $\tilde{z}_{q}$ and $\tilde{v}_{c}$ denote the normalized EEG and image embeddings, and $\psi$ denotes the pre-CSLS scoring function specified in Section 3.2. Retrieval is performed by ranking candidates $c\in C$ in descending order of $S_{\text{new}}(q,c)$, and our main evaluation metric is top-$k$ retrieval accuracy (typically $k\in\{1,5\}$) over queries from unseen subjects.

We adopt a strict label-free test-time regime. The encoders $f_{\text{ee g}}$ and $f_{\text{img}}$ are frozen; no labels or pseudo-labels from test subjects are available, and no adaptation of network weights is allowed. The only object we are allowed to modify at test time is the similarity structure itself. Formally, we seek a calibration operator

$F:S_{\text{new}}\mapsto S_{\text{final}}$ (2)

that uses only unlabeled test-time statistics to produce calibrated scores $S_{\text{final}}(q,c)$ whose induced rankings better reflect true EEG–image correspondences and reduce hubness, yielding more reliable small-$k$ shortlists for downstream decoding.

We instantiate $F$ as a Product-of-Experts (PoE) fusion between two complementary experts. A geometric expert $S_{\text{geom}}$ applies an adaptive CSLS scheme to $S_{\text{new}}$, using query- and class-dependent neighborhood sizes derived from local densities to mitigate hubness without tuning a global $k$ (Section 3.3). In parallel, a structural expert $S_{\text{struct}}$ is built directly on the pre-CSLS similarities, exploiting mutual nearest neighbors, bidirectional top-$k$ ranks, and class popularity patterns as structure-aware priors (Section 3.4). Section 3.5 describes how these experts are fused via a simple PoE to obtain the final calibrated scores $S_{\text{final}}$.

Fig. 1 illustrates this label-free test-time setting and the SATTC calibration head acting on the similarity matrix.

### 3.2 Geometric Normalization via Subject-Adaptive Whitening

Raw EEG embeddings exhibit strong subject-dependent shifts: even when different subjects perform the same task,

---

![[SATTC-_Structure-Aware_Label-Free_Test-Time_Calibration_for_Cross-Subject_EEG-to-Image_Retrieval_p4_img2.jpeg]]

![[SATTC-_Structure-Aware_Label-Free_Test-Time_Calibration_for_Cross-Subject_EEG-to-Image_Retrieval_p4_img3.jpeg]]

![[SATTC-_Structure-Aware_Label-Free_Test-Time_Calibration_for_Cross-Subject_EEG-to-Image_Retrieval_p4_img4.jpeg]]
Figure 2. Effect of SAW and SATTC on subject shift, hubness, and shortlist quality. (a) Per-subject Top-5 accuracy under LOSO. (b) Class popularity  $N_K(c)$ . (c)  $\Delta$  Recall@K over the Std.+SAW baseline. (d) Distribution of per-class Recall@5 for Std.+SAW and SATTC. SAW improves the standardized baseline, while SATTC further reduces hubness and yields more balanced and reliable small-K shortlists.

![[SATTC-_Structure-Aware_Label-Free_Test-Time_Calibration_for_Cross-Subject_EEG-to-Image_Retrieval_p4_img5.jpeg]]

their feature distributions can differ markedly in mean, variance, and covariance structure, leading to pronounced statistical and covariance shifts across subjects and sessions [1, 14].

We apply subject-adaptive whitening (SAW) to EEG features before computing similarities. Let  $s(q)$  denote the subject index of query  $q$ , and

$$
z _ {q} = f _ {\mathrm {e e g}} (q) \in \mathbb {R} ^ {d} \tag {3}
$$

be the encoder output. For each subject  $s$ , we estimate a mean  $\mu_{s}$  and covariance  $\Sigma_{s}$  from the unlabeled embeddings in the current split: training embeddings for training subjects, and unlabeled test-time embeddings (or a calibration window) for the held-out subject. Construct a regularized whitening transform

$$
W _ {s} = \left(\Sigma_ {s} + \lambda I\right) ^ {- \frac {1}{2}} \tag {4}
$$

with  $\lambda &gt; 0$  for numerical stability. Each query is first whitened as

$$
\hat {z} _ {q} = W _ {s (q)} \left(z _ {q} - \mu_ {s (q)}\right) \tag {5}
$$

and then  $\ell_2$ -normalized to obtain

$$
\tilde {z} _ {q} = \frac {\hat {z} _ {q}}{\| \hat {z} _ {q} \| _ {2}} \tag {6}
$$

This yields approximately zero-mean, identity-covariance, unit-norm EEG embeddings per subject and maps different

subjects onto a shared sphere while preserving relative directions. We apply the same SAW transform during training and evaluation to maintain a consistent feature space.

On the image side, we optionally apply global whitening to candidate embeddings. Let

$$
v _ {c} = f _ {\text {i m g}} (c) \tag {7}
$$

denote the candidate-side visual representation for class  $c \in C$ . We estimate a global mean  $\mu_{\mathrm{img}}$  and covariance  $\Sigma_{\mathrm{img}}$ , form

$$
\hat {v} _ {c} = \left(\Sigma_ {\mathrm {i m g}} + \lambda I\right) ^ {- \frac {1}{2}} \left(v _ {c} - \mu_ {\mathrm {i m g}}\right) \tag {8}
$$

and normalize

$$
\tilde {v} _ {c} = \frac {\hat {v} _ {c}}{\| \hat {v} _ {c} \| _ {2}} \tag {9}
$$

using  $\tilde{v}_c$  as the normalized prototype; when whitening is disabled, we simply apply  $\ell_2$ -normalization to  $v_c$ .

Given normalized embeddings  $\tilde{z}_q$  and  $\tilde{v}_c$ , we define a temperature-scaled cosine similarity

$$
S _ {\text {b u s e}} (q, c) = \frac {\alpha}{\tau} \langle \tilde {z} _ {q}, \tilde {v} _ {c} \rangle \tag {10}
$$

where  $\alpha$  is a learned logit scale from contrastive pretraining and  $\tau$  is a global temperature. Optionally, we apply global  $z$ -score normalization to  $S_{\mathrm{base}}$  using precomputed mean and variance over a held-out unlabeled set, yielding the pre-CSLS similarity matrix  $S_{\mathrm{new}}(q,c)$  used in subsequent stages.

---

### 3.3 Adaptive CSLS Geometric Expert for Hubness Mitigation

Even after SAW-based normalization, the pre-CSLS similarity matrix $S_{\text{new}}(q,c)$ still exhibits hubness: a few image classes $c$ appear in the top-$k$ neighbors of many queries $q$, while rare classes are under-retrieved *[6, 12, 18]*. Cross-domain Similarity Local Scaling (CSLS) is a standard remedy for hubness in cross-domain retrieval, using local-neighborhood averages to rescale similarity scores *[11]*. Given similarities $s(q,c)$, CSLS defines

$\operatorname{csls}(q,c)=2\,s(q,c)-r_{q}(q)-r_{c}(c)$ (11)

where $r_{q}(q)$ and $r_{c}(c)$ are average similarities to the $k$ nearest neighbors of query $q$ and class $c$, respectively. By subtracting these local averages, CSLS down-weights candidates that are globally “popular” with many queries and partially compensates for local density.

However, standard CSLS uses a fixed neighborhood size $k$ for all queries and classes, implicitly assuming a roughly uniform density in the embedding space. Cross-subject EEG embeddings violate this assumption: some queries reside in sparse regions with only a few meaningful prototypes, whereas some classes form dense hubs that attract many unrelated queries. A global $k$ can thus under-penalize true hubs and over-penalize rare but correct neighbors, suppressing genuine matches from the top-$k$ list.

To address this, we construct an adaptive CSLS geometric expert that retains the CSLS form but derives query- and class-dependent neighborhoods from local densities on $S_{\text{new}}$. Let

$s(q,c)=S_{\text{new}}(q,c)$ (12)

denote the normalized similarity between query $q$ and class $c$. For each query $q$, we estimate a row-wise local density

$\rho_{\text{row}}(q)=\frac{1}{m}\sum_{c\in N_{\text{row}}^{(m)}(q)}s(q,c)$ (13)

where $N_{\text{row}}^{(m)}(q)$ is the set of $m$ most similar classes to $q$, with $m\gg K_{\max}$, the largest top-$K$ cutoff we evaluate (e.g. $K_{\max}=20$).

$k_{\text{row}}(q)\in[k_{\min},k_{\max}]$ (14)

via a monotonic mapping from densities to integers in $[k_{\min},k_{\max}]$, so that denser queries use larger neighborhoods and sparser queries use smaller ones.

Similarly, for each class $c$ we estimate a column-wise density $\rho_{\text{col}}(c)$, for example from how frequently $c$ appears in row-wise top-$K$ lists, and derive a class-specific neighborhood size

$k_{\text{col}}(c)\in[k_{\min},k_{\max}]$ (15)

Given these adaptive neighborhood sizes, we compute

$r_{q}(q)=\frac{1}{k_{\text{row}}(q)}\sum_{c^{\prime}\in N_{\text{row}}^{(k_{\text{row}}(q))}(q)}s(q,c^{\prime}),$ (16)
$r_{c}(c)=\frac{1}{k_{\text{col}}(c)}\sum_{q^{\prime}\in N_{\text{col}}^{(k_{\text{col}}(c))}(c)}s(q^{\prime},c)$ (17)

where $N_{\text{row}}^{(k_{\text{row}}(q))}(q)$ and $N_{\text{col}}^{(k_{\text{col}}(c))}(c)$ denote the adaptive $k$-nearest neighbors along rows and columns.

Our adaptive CSLS score is then defined as

$S_{\text{geom}}(q,c)=2\,s(q,c)-r_{q}(q)-r_{c}(c)$ (18)

This construction preserves the familiar CSLS form while relaxing the fixed-density assumption: each query and class is normalized by a neighborhood size matched to its local density. All quantities are computed from $S_{\text{new}}$ alone, without access to any labels or pseudo-labels, and when the density mappings are constant $k_{\text{row}}(q)=k_{\text{col}}(c)=k_{0}$, our formulation reduces exactly to standard fixed-$k$ CSLS. We use $S_{\text{geom}}$ as the geometric expert in our Product-of-Experts fusion.

### 3.4 Structural Expert from Pre-CSLS Similarity Evidence

While the adaptive CSLS geometric expert mitigates hubness from a local-density perspective, the pre-CSLS similarity matrix $S_{\text{new}}$ already contains structural patterns that correlate with reliable matches and hubs. We convert these patterns into a structural expert $S_{\text{struct}}$ that complements $S_{\text{geom}}$ without changing the encoder or recomputing similarities.

Given $S_{\text{new}}(q,c)$, we compute row- and column-wise ranks

\[ \begin{split}r_{\text{row}}(q,c)&=1+\#\{\,c^{\prime}\mid S_{\text{new}}(q,c^{\prime})>S_{\text{new}}(q,c)\,\},\\
r_{\text{col}}(c,q)&=1+\#\{\,q^{\prime}\mid S_{\text{new}}(q^{\prime},c)>S_{\text{new}}(q,c)\,\}\end{split} \] (19)

where exact score ties in $S_{\text{new}}$ are broken deterministically. A pair $(q,c)$ is a strict mutual nearest neighbor (MNN@1) if

$r_{\text{row}}(q,c)=r_{\text{col}}(c,q)=1$ (20)

and we also consider bidirectional top-$L$ pairs

$r_{\text{row}}(q,c)\leq L,\quad r_{\text{col}}(c,q)\leq L$ (21)

which extend strict MNN@1 to slightly lower-confidence but still consistent matches. We define a class-popularity statistic

$\rho(c)=\#\{q\mid r_{\text{row}}(q,c)\leq K\}$ (22)

the number of queries for which class $c$ appears in the row-wise top-$K$. Large $\rho(c)$ indicates hub-like behavior, whereas small $\rho(c)$ indicates rare or under-retrieved classes. Normalizing $\rho(c)$ across classes by min-max scaling yields a hubness score $h(c)\in[0,1]$ used to scale penalties on suspected hubs.

Using these quantities, we distinguish three structural cases over $(q,c)$: (i) *anchors*, the strict MNN@1 pairs in (20); (ii) *bidirectional top-$L$ pairs*, satisfying (21); (iii)

---

hub-like candidates, where $r_{\text{row}}(q,c)>K$, $r_{\text{col}}(c,q)$ is small, and $h(c)$ is large, i.e., $c$ acts as a hub across many queries but is not locally supported for this query.

We construct a structural logit matrix $S_{\text{struct}}\in\mathbb{R}^{|Q|\times|C|}$, initialized to zero, and update

\[ S_{\text{struct}}(q,c)=\begin{cases}+\lambda_{\text{anchor}},&(q,c)\in\mathcal{A}\\
-\lambda_{\text{pen}}\,h(c),&(q,c)\in\mathcal{H}\\
0,&\text{otherwise}\end{cases} \] (23)

where $A\subset Q\times C$ collects anchors and strong bidirectional top-$L$ pairs, $H\subset Q\times C$ collects hub-like candidates, and $\lambda_{\text{anchor}},\lambda_{\text{pen}}>0$ control the strength of bonus and penalty. Penalties are masked on pairs in $A$, so high-confidence structural matches are only boosted, never suppressed. All decisions in (23) depend solely on $S_{\text{new}}$; $S_{\text{geom}}$ and the encoder are not used, and $S_{\text{new}}$ itself is never modified.

In summary, $S_{\text{struct}}$ provides a structure-aware prior over query–class pairs: it locks mutual top-1 and bidirectional top-$L$ matches via positive bias and down-weights ubiquitous hub classes, and, being computed once from pre-CSLS evidence and then held fixed, regularizes the final rankings without iterative self-reinforcement.

### 3.5 Product-of-Experts Fusion and Final Scoring

We combine the geometric expert $S_{\text{geom}}$ (adaptive CSLS on $S_{\text{new}}$) and the structural expert $S_{\text{struct}}$ (pre-CSLS evidence; Section 3.4) into a single calibrated scoring function. For each query $q$, the geometric expert defines an unnormalized distribution

$p_{\text{geom}}(c\mid q)\propto\exp\big{(}S_{\text{geom}}(q,c)\big{)}$ (24)

and the structural expert defines

$p_{\text{struct}}(c\mid q)\propto\exp\big{(}S_{\text{struct}}(q,c)\big{)}$ (25)

A simple Product-of-Experts (PoE) fusion then gives

$p_{\text{final}}(c\mid q)\propto p_{\text{geom}}(c\mid q)^{\alpha}\,p_{\text{struct}}(c\mid q)^{\beta}$ (26)

where $\alpha,\beta\geq 0$ control the relative influence of the two experts. In logit space this reduces to a weighted sum

$S_{\text{final}}(q,c)=\alpha\,S_{\text{geom}}(q,c)+\beta\,S_{\text{struct}}(q,c)$ (27)

In practice we fix $\alpha=1$ and tune a single scalar hyperparameter $\beta$, making the fusion lightweight and interpretable. Setting $\beta=0$ recovers pure adaptive CSLS, while $\beta>0$ progressively enforces the structure-aware prior encoded by $S_{\text{struct}}$: mutual top-1 and bidirectional top-$L$ matches receive a constant positive bias and hub-like candidates receive a negative bias proportional to their popularity level (Section 3.4). Because $S_{\text{struct}}$ is computed once from the pre-CSLS matrix $S_{\text{new}}$ and never updated based on $S_{\text{final}}$, the PoE acts as a single-shot regularizer rather than an iterative self-reinforcement scheme.

At test time, the label-free calibration operator from Section 3.1 can thus be written as

$\mathcal{T}_{\beta}(S_{\text{new}}):=\mathrm{PoE}_{\beta}(S_{\text{geom}},S_{\text{struct}})=S_{\text{final}}$ (28)

where $S_{\text{geom}}$ and $S_{\text{struct}}$ are the geometric and structural experts applied to $S_{\text{new}}$. Top-$K$ retrieval is obtained by ranking classes $c\in C$ according to $S_{\text{final}}(q,c)$ for each query $q$.

## 4 Experiments

### 4.1 Experimental Setup

#### Dataset and cross-subject protocol.

We evaluate on the public THINGS-EEG dataset *[7, 9]*, which provides EEG recordings from 10 subjects viewing images from 200 object categories. We adopt a leave-one-subject-out (LOSO) protocol: in each of 10 folds, one subject is held out for testing and the remaining nine are used for training and tuning.

To separate tuning from final evaluation, we use a nested LOSO procedure. From the nine training subjects, we select three representative dev subjects (easy/medium/hard) via a subject-level distance analysis and use only them for hyperparameter and epoch selection. We then fix the selected configuration, train a single model on all nine training subjects, and perform one-shot evaluation on the held-out subject. All hyperparameters were tuned once on a fixed dev pack from the training folds and then frozen across all LOSO folds and across encoders; at test time we perform label-free calibration using only unlabeled EEG from the held-out subject, and test labels are used solely for evaluation.

#### Evaluation metrics and baselines.

We evaluate 200-way EEG-to-image retrieval: for each test EEG query, the model ranks all 200 candidate image classes. We report Top-5 accuracy as the primary metric and Top-1 accuracy as a secondary identification metric, and additionally show Recall@$K$ curves and hubness statistics in later sections.

To isolate the contribution of each component in our calibration pipeline, we compare a sequence of seven models, labeled ATM, Standardized baseline, +SAW, +SAW+CW, +CSLS, +Ada-CSLS, and SATTC (ours) in Table 1. Starting from the original ATM cross-subject baseline and our standardized baseline with cosine similarity, $\ell_{2}$-normalized features, and candidate-side whitening but no SAW or test-time calibration (“Standardized baseline” in Table 1), we incrementally add subject-adaptive whitening on EEG (+SAW), fixed-$k$ CSLS (+CSLS), adaptive CSLS as the geometric expert (+Ada-CSLS), and finally the structural PoE module on top of adaptive CSLS (SATTC). All variants

---

share the same training setup: AdamW optimizer, batch size 1024, learning rate  $5 \times 10^{-4}$ , and a fixed temperature  $\tau = 1.0$ , trained and evaluated on a single NVIDIA RTX 4090 GPU.

# 4.2. Cross-Subject EEG-to-Image Retrieval Results

Overall comparison with baselines. Table 1 reports 200-way cross-subject retrieval averaged over all LOSO folds and three seeds. Cosine similarity with  $\ell_2$ -normalized features and candidate-side whitening (same encoder) already improves substantially over the original ATM baseline with unnormalized dot products, yielding the "Standardized baseline" row in Table 1.

Introducing subject-adaptive whitening on the EEG side (SAW) on top of the cosine + candidate-side whitening (CW) baseline brings a clear gain and serves as our cross-subject reference, showing that subject-specific normalization is crucial in the LOSO setting. On this SAW configuration (with candidate whitening), both fixed-  $k$  CSLS and adaptive CSLS further increase Top-5 accuracy while keeping Top-1 comparable; the adaptive variant matches the best fixed-  $k$  configuration without a single global  $k$ , validating the geometric expert. Finally, adding the structural PoE module on top of adaptive CSLS (SATTC, ours) gives the strongest end-to-end result: relative to the SAW baseline, both Top-1 and Top-5 improve, while relative to adaptive CSLS, SATTC preserves Top-5 and improves Top-1 through structure-aware refinement.

Subject- and seed-wise stability. Across three random seeds per LOSO fold, standard deviations over subjects and seeds are small and the relative ordering between the baseline, SAW, CSLS/Ada-CSLS, and SATTC is consistent, indicating that our label-free test-time calibration is robust to initialization and subject splits.

# 4.3. Ablations for Label-Free Test-Time Calibration

Effect of SAW and candidate whitening. Table 1 summarizes the whitening ablations. Compared to our standardized baseline (cosine +  $\ell_2 + \mathrm{CW}$ ), adding subject-adaptive whitening on the EEG side (+SAW) is the dominant source of improvement: Top-5 rises from  $30.5\%$  to  $36.4\%$  and Top-1 from  $9.2\%$  to  $13.7\%$ . This shows that normalizing subject-specific EEG statistics is the key ingredient for cross-subject generalization under LOSO. Further adjusting candidate whitening in the presence of SAW (+SAW+CW) yields only a small additional Top-5 gain  $(36.8\%)$  with comparable Top-1, so the direct contribution of candidate-side whitening to retrieval remains modest once EEG features have been subject-normalized.

However, we retain candidate whitening (CW) primarily to support the structural expert rather than chase additional

Table 1. Cross-subject 200-way EEG-to-image retrieval results on THINGS-EEG under LOSO. Mean Top-5 and Top-1 accuracy  $(\%)$  over all test folds and three random seeds for the ATM baseline, our standardized baseline (cosine  $+\ell_{2}$  normalization  $^+$  candidate whitening), and successive additions of SAW, candidate whitening, CSLS variants, and SATTC.

|  Method | Top-5 (%) ↑ | Top-1 (%) ↑  |
| --- | --- | --- |
|  ATM (original) | 20.0 | 5.5  |
|  Standardized baseline | 30.5 | 9.2  |
|  + SAW | 36.4 | 13.7  |
|  + SAW + CW | 36.8 | 13.5  |
|  + SAW + CW + CSLS | 38.1 | 14.1  |
|  + SAW + CW + Ada-CSLS | 38.8 | 13.9  |
|  SATTC (ours) | 38.4 | 14.8  |

accuracy. With SAW alone, similarities are normalized row-wise, but column scales can still vary across classes. Under SAW+CW, the pre-CSLS similarity matrix  $S_{\mathrm{new}}$  used in SATTC has both query-wise rows and class-wise columns on a comparable scale, making class-popularity  $NK(c)$ , mutual nearest neighbors, and other rank-based statistics derived from  $S_{\mathrm{new}}$  (Section 3.4) more stable. We therefore adopt SAW+CW as the default backbone for all subsequent CSLS variants and SATTC experiments despite its modest standalone gain over +SAW.

Fixed- $k$  vs. adaptive CSLS. Table 1 reports results for fixed- $k$  CSLS (with  $k = 12$  tuned on dev subjects) and our adaptive CSLS applied on the SAW+CW configuration. The two variants obtain very similar Top-1 and Top-5 accuracy, with adaptive CSLS trading a small decrease in Top-1 for a slightly higher Top-5 while avoiding a globally chosen neighborhood size. Their hubness profiles, however, differ: as shown in Fig. 2, adaptive CSLS further flattens the class-occurrence curve and reduces the dominance of a few hub classes compared to fixed- $k$  CSLS, leading to a more balanced use of prototypes across ranks. This supports our choice of adaptive CSLS as the geometric expert: it preserves retrieval accuracy while providing stronger and more flexible hubness mitigation than a single global  $k$ .

Impact of structural expert and PoE. Finally, Table 1 isolates the contribution of the structural expert and PoE by comparing  $+\mathrm{SAW} + \mathrm{CW} + \mathrm{Ada}$ -CSLS with our full SATTC model  $(+\mathrm{Ada}$ -CSLS+PoE). SATTC keeps Top-5 accuracy essentially unchanged (38.4 vs.  $38.8\%$ ) while providing a clear Top-1 gain (14.8 vs.  $13.9\%$ ), highlighting that the structural expert sharpens strict identification while preserving overall retrieval coverage. Consistent with this, Fig. 2 shows that the structural PoE head in SATTC attains the largest  $\Delta$  Recall@K over the Std.+SAW baseline for small  $K$  (1 and 5); relative to adaptive CSLS alone, it recovers more classes whose true label appears within small-K short

---

lists, reflecting better class-wise calibration even when the global Top-5 score remains comparable.

Practically, the calibration pipeline admits two deployment modes. For single-trial or online inference, the SAW+CSLS branch can be calibrated once from a small unlabeled window and then applied to subsequent trials with frozen parameters, whereas the structural expert and PoE serve as a batch refinement stage when multiple unlabeled test queries are available. In strict disjoint  $N$ -shot experiments, a one-time unlabeled window of  $N = 50$  already reaches  $94.8\%$  of the  $N = 200$  SAW+CSLS upper bound. The PoE hyperparameter is also robust: a broad region of  $\beta$  values yields near-identical Top-5 performance, and the default  $\beta = 1.9$  is within 0.1 percentage points of the best setting. Importantly, the full pipeline shows no end-to-end Top-1/Top-5 trade-off: from the SAW baseline to SATTC, Top-1 improves from  $13.7\%$  to  $14.8\%$ , while Top-5 improves from  $36.4\%$  to  $38.4\%$ .

# 4.4. Hubness Reduction and Shortlist Quality

Hubness and class popularity distribution. We use the class-popularity statistic  $N_K(c)$  from Section 3.4 as a hubness indicator (Fig. 2). Under the SAW+CW configuration, fixed-  $k$  CSLS shortens the heavy tail of  $N_K(c)$  relative to the cosine baseline but still leaves some hubs; adaptive CSLS further flattens  $N_K(c)$ , while SATTC yields the most uniform profile on this class-popularity statistic by suppressing spurious hubs and boosting underused classes.

Per-class fairness and Recall@K. Per-class performance is evaluated with Recall@5 boxplots and Recall@K curves for  $K \in \{1,2,5,10,20\}$  (Fig. 2). Compared to the SAW+CW baseline, both fixed-  $k$  CSLS and adaptive CSLS increase mean Recall@5; adaptive CSLS also achieves a higher median and lower variance across classes. SATTC further raises median Recall@5, narrows the spread, and shows the largest gains for small  $K \leq 5$ , the regime most relevant for downstream decoding.

# 4.5. Encoder-Agnostic Plug-and-Play Calibration

We next test whether SATTC is tied to a specific EEG encoder or can act as a generic calibration layer. Table 2 reports 200-way cross-subject retrieval on THINGS-EEG for four heterogeneous encoders (ATM, EEGNetV4, EEGConformer, and ShallowFBCSPNet), spanning CSP-style, compact CNN, and transformer architectures. For each encoder we compare our standardized inference baseline to the same model equipped with SATTC as a label-free test-time calibration head, without changing the backbone or using labels from test subjects.

SATTC consistently improves all encoders. Top-5 accuracy increases by roughly 8-16 percentage points and Top-1 by about 4-8 points across backbones (e.g., from 30.5

Table 2. Plug-and-play generalization of SATTC across EEG encoders on 200-way THINGS-EEG cross-subject retrieval. For each encoder, we report Top-5 and Top-1 accuracy  $(\%)$  averaged over all LOSO folds and three random seeds, comparing our standardized inference baseline to the same encoder with SATTC applied as a label-free test-time calibration head.

|  Encoder | Top-5 ↑ |   | Top-1 ↑  |   |
| --- | --- | --- | --- | --- |
|   |  Baseline | +SATTC | Baseline | +SATTC  |
|  ATM | 30.5 | 38.4 | 9.2 | 14.8  |
|  EEGNetV4 | 20.5 | 34.8 | 5.4 | 10.8  |
|  EEGConformer | 11.6 | 23.2 | 2.5 | 6.9  |
|  ShallowFBCSPNet | 14.6 | 30.8 | 3.5 | 11.1  |

to 38.4 Top-5 and 9.2 to 14.8 Top-1 for ATM, and from 20.5 to 34.8 and 5.4 to 10.8 for EEGNetV4), with similar gains for EEGConformer and ShallowFBCSPNet. These results indicate that SATTC operates on the geometry of the EEG-image similarity space rather than on encoder details, and can be plugged into existing cross-subject EEG pipelines as an encoder-agnostic, label-free test-time calibration module.

# 5. Discussion and Conclusion

Limitations. SATTC is currently evaluated only on THINGS-EEG under a LOSO cross-subject retrieval protocol. Its structural expert is hand-crafted from ranks, mutual nearest neighbors, and class popularity, and the current implementation operates on precomputed similarity matrices. In practice, the SAW+CSLS branch supports calibration once then frozen inference, whereas the structural expert and PoE are most naturally applied in a batch refinement stage when multiple unlabeled test queries are available. Extending the method to additional datasets, encoder families, and decoding regimes such as retrieval-then-generation or EEG/fMRI reconstruction [3, 13] remains an important direction for future work.

Conclusion. We cast cross-subject EEG-to-image retrieval as a label-free test-time calibration problem on similarity matrices. SATTC operates on frozen EEG and image encoders through a modular calibration head that combines subject-adaptive whitening, adaptive CSLS, and a lightweight structural prior from pre-CSLS similarities. On THINGS-EEG under strict LOSO evaluation, SATTC improves Top-1 and Top-5 over strong standardized baselines, reduces hubness and per-class imbalance, and yields more reliable small- $k$  shortlists. Together with its plug-and-play gains across multiple EEG encoders, these results suggest that similarity-space calibration is a practical route toward more robust cross-subject neural decoding.

---
