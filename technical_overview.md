# Technical Overview of the THINGS-EEG-2 Inter-Subject Pipeline

This document summarizes the main ideas implemented in this repository in a paper-facing way. The goal is to explain the technical logic of the experiments, how the different pieces fit together, and how the system moves from a strong inter-subject baseline in the mid-30% range to roughly 71% top-1 accuracy with transductive test-time adaptation on the 200-way THINGS-EEG-2 benchmark.

The main story is simple:

1. Learn a shared representation in which EEG responses and image embeddings can be directly compared.
2. Make the inter-subject training problem easier by using a compact alignment space, a strong but simple EEG encoder, and training batches that explicitly compare different subjects seeing the same image.
3. At test time, exploit the structure of the entire unlabeled 200-way test set to adapt the query geometry of the held-out subject.
4. Show that the adaptation learned from one subset of the test fold can transfer useful information to unseen samples from that same subject.

## Problem setting

The repository targets image decoding on THINGS-EEG-2 in the inter-subject regime. Training uses EEG from several subjects, while evaluation is performed on a held-out subject. The main reported metric is closed-set 200-way matching: each EEG test trial must identify its corresponding image among 200 candidate images. Top-1 accuracy is the main number of interest, with top-5 also reported.

This is a difficult regime because the model must simultaneously solve two problems:

1. it must learn a useful cross-modal correspondence between EEG and visual representations;
2. it must cope with the fact that EEG geometry changes substantially from subject to subject.

The repository attacks the first problem mainly during training, and the second problem both during training and, much more aggressively, at test time.

## Result tracks that matter most

For paper drafting, it helps to distinguish three result families rather than treating the repository as a single monolithic system.

### Reduced-dimension inter-subject base models

One important family of runs trains a standard inter-subject decoder using a compact shared representation. A representative result file is:

- `results/things_eeg/inter-subjects/20260413-143447_session_seed2099/featdim_128/inter_subject_summary.csv`

This run reports:

- final-epoch average top-1 accuracy of `25.8`
- best-epoch average top-1 accuracy of `32.4`

This is already a strong inter-subject base model and serves as the source model for the later 71% test-time adaptation sweeps.

### Historical mid-30% runs

A separate family of experiments explores cross-subject mixing strategies during training. A representative result folder is:

- `results/things_eeg/inter-subjects/mixup_20260329-225009/mixup_summary.csv`

That run contains seed-level averages whose best top-1 values include:

- `35.5`
- `35.9`

These are the clearest repository results supporting the claim that the training-side inter-subject decoder can be pushed into the mid-30% range.

### Full transductive test-time adaptation around 71%

The strongest 200-way adaptation results appear in:

- `results/things_eeg/inter-subjects/sattc_final_ablation_20260414-155005/sattc_sweep_summary.csv`

Several parameter settings reach about `71.0` to `71.2` top-1 on average across subjects. These numbers are not produced by additional supervised learning. They come from taking a trained inter-subject decoder and adapting its test-time query geometry using the unlabeled structure of the full 200 paired test items for the held-out subject. This is therefore a transductive closed-set adaptation result, not a standard inductive test-time evaluation.

### Transfer to unseen test samples

The repository also contains a cleaner transfer experiment in which the test fold is split into two disjoint parts. Adaptation is estimated on one part and then applied to the other. A representative file is:

- `results/things_eeg/inter-subjects/source_run_split_transfer_20260415-171631/run_average_results.csv`

In the default 100/100 split:

- whitening alone on the evaluation half gives `49.4` top-1
- blending in the learned orthogonal transformation with `alpha = 0.5` raises this to `59.3`

This is close to a 10-point absolute gain on samples that were not used to fit the transformation, which is strong evidence that the learned correction is not purely sample-specific.

## Data representation and preprocessing

The experiments use THINGS-EEG-2 with the standard train/test organization. The training side contains many objects, each with multiple image exemplars and repeated EEG measurements. The test side is a closed 200-item identification problem.

In practical terms, the preprocessing does three important things:

1. it standardizes subject data into a common tensor layout;
2. it crops the EEG signals to a fixed temporal window of 250 samples;
3. it usually averages repetitions before training and evaluation.

That last point matters. Most of the strong inter-subject runs operate on averaged EEG responses rather than single-trial responses. This reduces variance and makes subject-to-subject alignment more tractable.

Channel selection is also supported, but the strongest mainline runs typically use all channels unless an explicit regional ablation is being performed.

## Visual target space

The image side of the problem is intentionally frozen. Rather than learning an image encoder jointly with the EEG model, the repository relies on precomputed image embeddings extracted from a large pretrained vision transformer. The most important design choice is not just the backbone itself, but the layer from which features are taken.

The strongest runs use an intermediate-layer representation from InternViT rather than a very early or very late layer. This is a sensible compromise:

- too early would overemphasize local visual detail and underemphasize semantics;
- too late would emphasize semantics but may discard visual structure that remains decodable from EEG;
- an intermediate layer preserves both object-level meaning and enough perceptual organization to be recoverable from brain signals.

This choice is one of the core reasons the baseline is already strong before any test-time adaptation is applied.

## EEG encoder and shared space

On the EEG side, the central model is a temporal-spatial convolutional encoder (TSConv). Its architecture reflects a deliberate inductive bias for EEG signals. Given an input tensor of shape \((B, C, T)\) — batch size, channels, time points — the encoder proceeds as follows:

1. A first 2-D convolution with kernel \((1, 25)\) and 40 output filters captures temporal patterns at a scale of roughly 25 samples, covering short-range dynamics. An average pooling with kernel \((1, 51)\) and stride 5 then compresses the temporal dimension while preserving its structure.
2. Batch normalization and ELU activation follow.
3. A second convolution with kernel \((C, 1)\) collapses the channel dimension entirely, projecting the multi-channel representation into a single spatial summary per time step. A second batch normalization, ELU, and dropout follow.
4. A \(1\times1\) convolution projects to a fixed number of feature maps, after which the output is flattened and passed through a two-layer MLP with a residual connection, GELU activation, and layer normalization. This produces the final embedding of dimension \(d\).

The final embedding dimension \(d\) is a key hyperparameter. The strongest runs use \(d \in \{64, 128\}\). This compactness is not merely a computational convenience: a small embedding acts as a bottleneck that forces the encoder to discard subject-specific nuisance variation and retain only the features that are consistently predictive of image identity across subjects.

## Core training idea: compare subjects through exact stimulus identity

The most important training design choice is a carefully structured batching strategy. Instead of drawing arbitrary EEG-image pairs independently, batches are organized so that multiple subjects who viewed the exact same image are present together. The sampler controls two key quantities: the number of distinct images per batch, \(M\), and the number of subject-level EEG responses per image, \(K\), giving a total batch size of \(N = MK\).

This batch structure changes the learning problem in an important way. Once a batch contains \(K\) EEG observations all elicited by the same stimulus across \(K\) different subjects, the model can simultaneously see:

- these responses should not compete against one another;
- they are all legitimate evidence for the same visual target;
- they reveal the cross-subject variability around a common underlying image representation.

That structure is the foundation on which the rest of the inter-subject training logic is built.

## Cross-modal objective

The training loss is a symmetric contrastive objective between EEG embeddings and image embeddings. Let \(\mathbf{e}_i \in \mathbb{R}^d\) denote the \(\ell_2\)-normalized EEG embedding for sample \(i\), and \(\mathbf{v}_j \in \mathbb{R}^d\) the \(\ell_2\)-normalized image embedding for image \(j\). The similarity matrix is

\[
S_{ij} = \frac{\mathbf{e}_i \cdot \mathbf{v}_j}{\tau}
\]

where \(\tau > 0\) is a learnable temperature initialized to a fixed value (typically 0.07). The standard (one-positive) cross-entropy direction from EEG to images is

\[
\mathcal{L}_{\text{EEG}\to\text{img}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}
\]

and symmetrically \(\mathcal{L}_{\text{img}\to\text{EEG}}\). The total contrastive loss is their average.

When the grouped batch sampler places multiple subjects' responses to the same image in the same batch, those responses become multiple valid positives for the same image target. The loss is then adapted to a **multi-positive** formulation. Let \(\mathcal{P}_i\) be the set of indices \(j\) such that sample \(j\) corresponds to the same image as sample \(i\). The multi-positive cross-entropy for a single row \(i\) of the logit matrix is

\[
\ell_i = -\frac{1}{|\mathcal{P}_i|} \sum_{j \in \mathcal{P}_i} \log\text{softmax}(S_i)_j
\]

where \(\text{softmax}(S_i)_j = \exp(S_{ij}) / \sum_{k} \exp(S_{ik})\). The full multi-positive loss averages \(\ell_i\) over all rows that have at least one positive, and the symmetric direction is applied analogously with image embeddings as queries.

This multi-positive formulation is a major conceptual point for the paper draft: the training objective is not merely contrastive learning on EEG and images. It is contrastive learning that explicitly acknowledges that the same image can generate multiple valid EEG embeddings across subjects, preventing same-image, different-subject pairs from being treated as false negatives.

## Training-side regularization ideas

Beyond the main contrastive objective, the repository explores several auxiliary ideas. Two of them are central to the paper story.

### Cross-subject same-stimulus mixing

One family of experiments mixes EEG information across subjects, but only when the signals correspond to the same exact image. The operation is applied either in raw EEG signal space or after an initial feature extraction step, and it produces synthetic training samples from convex combinations of real responses.

Within each batch, the same-stimulus groups are first identified: group \(g\) collects all indices \(\{i : (o_i, m_i) = g\}\) where \(o_i\) is the object index and \(m_i\) is the image exemplar index. Mixing is then applied within each group.

**Pairwise mixing.** For each sample \(i\) in group \(g\), a partner \(j \neq i\) from a different subject is selected. A mixing coefficient is drawn as

\[
\lambda_i \sim \mathrm{Beta}(\alpha_\mathrm{mix},\, \alpha_\mathrm{mix})
\]

and the mixed feature is

\[
\tilde{\mathbf{x}}_i = \lambda_i\, \mathbf{x}_i + (1 - \lambda_i)\, \mathbf{x}_j.
\]

Here \(\alpha_\mathrm{mix}\) is a concentration parameter controlling how close to the uniform mixture the coefficient tends to be. When \(\alpha_\mathrm{mix} = 1\), the Beta distribution is uniform on \([0,1]\), producing the widest variety of interpolations. As \(\alpha_\mathrm{mix} \to \infty\), the coefficient concentrates around \(0.5\), always producing near-equal blends.

**Group mixing.** Alternatively, all \(K\) members of a same-stimulus group are mixed simultaneously using a Dirichlet draw. For group \(g\) with members \(\{\mathbf{x}_1, \ldots, \mathbf{x}_K\}\), the mixed version of each member \(i\) is

\[
\tilde{\mathbf{x}}_i = \sum_{k=1}^{K} w_{ik}\, \mathbf{x}_k, \quad \mathbf{w}_i \sim \mathrm{Dirichlet}(\alpha_\mathrm{mix}, \ldots, \alpha_\mathrm{mix}).
\]

In both cases, the mixed representations retain the original image label: the mixing preserves stimulus identity while averaging out subject-specific geometry. These experiments are closely tied to the mid-30% inter-subject results.

### Cross-subject self-supervision

The repository also explores a second regularization principle: EEG responses from different subjects to the same image should resemble one another even before comparing them to the visual target space. Concretely, for each anchor EEG embedding \(\mathbf{e}_i\), a partner embedding \(\mathbf{e}_j\) is selected from a different subject who saw the same image. A symmetric pair-wise InfoNCE loss is computed between all such anchor-partner pairs within the batch:

\[
\mathcal{L}_{\text{SSL}} = -\frac{1}{|\mathcal{A}|} \sum_{(i,j) \in \mathcal{A}} \frac{1}{2}\left[\log \frac{\exp(\mathbf{e}_i \cdot \mathbf{e}_j / \tau)}{\sum_k \exp(\mathbf{e}_i \cdot \mathbf{e}_k / \tau)} + \log \frac{\exp(\mathbf{e}_j \cdot \mathbf{e}_i / \tau)}{\sum_k \exp(\mathbf{e}_j \cdot \mathbf{e}_k / \tau)}\right]
\]

where \(\mathcal{A}\) is the set of valid cross-subject same-stimulus pairs in the batch. The diagonal (self-similarity) terms are masked to infinity before the softmax. This term is added to the main cross-modal loss with a weighting coefficient \(\lambda_{\text{SSL}}\). It directly encourages the model to form a representation in which image identity is more stable across subjects.

### Train-time whitening

Another explored idea is to whiten subject-specific EEG representations during training, so that each subject's feature cloud becomes more isotropic before the cross-modal loss is applied. The operation is the same ZCA whitening used at test time (described below), but applied on the fly to training batches. Its role is conceptually distinct from the test-time version:

- train-time whitening changes the representation the model actually learns during gradient updates;
- test-time whitening changes the geometry of the held-out subject's queries after the model has already been fully trained.

## Evaluation protocol

Evaluation always remains a retrieval problem. For a held-out subject, the model produces one EEG embedding per test item and compares it against the 200 candidate image embeddings. The correct answer is the image that occupies the matching position in the candidate set.

Two forms of evaluation appear in the repository:

1. plain retrieval using the trained model as is;
2. refined retrieval after transductive test-time adaptation.

This distinction is crucial because the jump from the low-30s to the low-70s does not come from replacing the base model with a larger or more heavily supervised one. It comes from adapting the geometry of the held-out subject's test queries using the full unlabeled 200-way test fold.

## A methodological caveat for the paper draft

One detail should be stated clearly in any paper draft derived from this repository: some training runs choose the best checkpoint using the held-out test subject, while others use a separate validation subject. The repository contains both workflows.

That means the paper should be careful not to present every reported number as if it came from the exact same model-selection protocol. The later writing model will need to distinguish:

- exploratory or engineering runs that selected checkpoints on the test fold;
- cleaner validation-based runs that held out a separate subject for selection.

This does not change how the adaptation mechanisms work, but it does matter for how the experiments should be framed.

## Transductive test-time adaptation: why the 71% result is so much higher

The central insight behind the 71% numbers is that the 200-way test problem contains a large amount of unlabeled structure. At test time, the model is not facing isolated EEG samples. It is facing a full set of 200 EEG queries and 200 candidate images that are known to form a one-to-one matching problem.

The repository exploits this structure in four stages applied iteratively.

### 1. Subject-adaptive whitening (ZCA)

The first stage whitens the held-out subject's EEG query distribution. Let \(\{\mathbf{e}_i\}_{i=1}^{N}\) be the \(N = 200\) test EEG embeddings for the held-out subject. The empirical covariance is estimated with Ledoit-Wolf-style shrinkage:

\[
\hat{\boldsymbol{\mu}} = \frac{1}{N}\sum_{i=1}^N \mathbf{e}_i, \qquad \hat{\Sigma} = \frac{1}{N-1}\sum_{i=1}^N (\mathbf{e}_i - \hat{\boldsymbol{\mu}})(\mathbf{e}_i - \hat{\boldsymbol{\mu}})^\top
\]

\[
\tilde{\Sigma} = (1 - \rho)\,\hat{\Sigma} + \rho\,\frac{\mathrm{tr}(\hat{\Sigma})}{d}\,I_d
\]

where \(\rho \in [0,1]\) is the shrinkage coefficient (best results use \(\rho \approx 0.85\)) and \(d\) is the embedding dimension. The ZCA (symmetric) whitening matrix is

\[
W = \tilde{\Sigma}^{-1/2} = V \,\mathrm{diag}(\lambda_1^{-1/2}, \ldots, \lambda_d^{-1/2})\, V^\top
\]

where \(\tilde{\Sigma} = V \Lambda V^\top\) is the eigendecomposition. Each centered EEG embedding is then mapped as

\[
\mathbf{e}_i' = W\,(\mathbf{e}_i - \hat{\boldsymbol{\mu}})
\]

and subsequently \(\ell_2\)-normalized. After this step, the query cloud is isotropic: directions of high variance no longer dominate the cosine similarity computation.

The effect is geometric rather than semantic. Whitening does not tell the model which image is correct; it removes subject-specific anisotropy so that cosine similarity becomes a more uniform measure of alignment.

### 2. Hubness correction with CSLS

After whitening, raw cosine similarities are corrected with Cross-domain Similarity Local Scaling (CSLS). For a query embedding \(\mathbf{e}_i'\) and a candidate image embedding \(\mathbf{v}_j\), the CSLS score is

\[
\mathrm{CSLS}(\mathbf{e}_i', \mathbf{v}_j) = 2\,\cos(\mathbf{e}_i', \mathbf{v}_j) - r_X(\mathbf{e}_i') - r_Y(\mathbf{v}_j)
\]

where

\[
r_X(\mathbf{e}_i') = \frac{1}{k}\sum_{\mathbf{v} \in \mathcal{N}_k(\mathbf{e}_i')} \cos(\mathbf{e}_i', \mathbf{v}), \qquad r_Y(\mathbf{v}_j) = \frac{1}{k}\sum_{\mathbf{e}' \in \mathcal{N}_k(\mathbf{v}_j)} \cos(\mathbf{e}', \mathbf{v}_j)
\]

are the average cosine similarity of each point to its \(k\) nearest neighbors in the opposite set. Subtracting these neighborhood means penalizes "hub" candidates that happen to be close to many queries and makes the matching landscape more locally discriminative. In this repository, the strongest settings use \(k = 1\), meaning only the single nearest neighbor is used for the correction, which suggests that only a light hubness correction is needed once whitening has already improved the geometry.

### 3. Soft permutation structure with Sinkhorn normalization

The 200-way test task is not an arbitrary retrieval problem. Each EEG query should match exactly one image, and each image should correspond to exactly one EEG query. This one-to-one structure is injected into the similarity matrix by converting it into a soft doubly-stochastic assignment matrix via the Sinkhorn-Knopp algorithm.

Given the \(N \times N\) similarity matrix \(S\) (after CSLS), define the initial matrix

\[
P^{(0)} = \exp\!\left(\frac{S}{\tau_\mathrm{sk}}\right)
\]

where \(\tau_\mathrm{sk} > 0\) is a temperature controlling the sharpness of the assignments (best results use \(\tau_\mathrm{sk} \approx 0.1\)). The Sinkhorn iterations then alternate row and column normalization:

\[
P^{(t+\frac{1}{2})} = \mathrm{diag}\!\left(P^{(t)}\mathbf{1}\right)^{-1} P^{(t)}, \qquad P^{(t+1)} = P^{(t+\frac{1}{2})}\,\mathrm{diag}\!\left((P^{(t+\frac{1}{2})})^\top\mathbf{1}\right)^{-1}
\]

After convergence (typically 15–20 iterations), \(P^{(\infty)}\) is a doubly stochastic matrix: all rows and columns sum to 1. It behaves like a softened version of a permutation matrix, where \(P_{ij}\) encodes the soft probability that query \(i\) is matched to image \(j\). This makes the assignment problem much more structured than ordinary cosine retrieval.

### 4. Orthogonal alignment via weighted Procrustes

Once a soft assignment \(P\) is available, an orthogonal transformation \(R \in \mathbb{R}^{d \times d}\) is fit to rotate the held-out subject's EEG feature cloud toward the image feature cloud. This is the most distinctive part of the pipeline.

The objective is to find the orthogonal matrix that best explains the soft assignments:

\[
R^* = \arg\min_{R^\top R = I} \sum_{i,j} P_{ij}\, \|\mathbf{e}_i' R - \mathbf{v}_j\|^2
\]

Expanding and dropping terms independent of \(R\), this is equivalent to maximizing

\[
\mathrm{tr}(R^\top\, \underbrace{E^{\top} P\, V}_{=:\,M})
\]

where \(E \in \mathbb{R}^{N \times d}\) is the matrix of whitened EEG embeddings and \(V \in \mathbb{R}^{N \times d}\) is the matrix of image embeddings. The cross-covariance \(M = E^\top P V\) is a weighted sum of outer products, with weights given by the soft assignment. This problem has a closed-form solution via the SVD:

\[
M = U\,\Sigma_{\mathrm{svd}}\, V_{\mathrm{svd}}^\top \qquad \Rightarrow \qquad R^* = U\, V_{\mathrm{svd}}^\top.
\]

The transformation is orthogonal by construction, which is a strong and useful constraint. It allows the model to correct a rotation-like mismatch between the two spaces without arbitrarily stretching, collapsing, or distorting the representation. In other words, it assumes that much of the subject-specific discrepancy is not a change in information content, but a change in coordinate system — a very natural hypothesis for inter-subject transfer.

### Iterative refinement

The four stages above are not applied once and discarded. They define one iteration of a refinement loop:

1. Compute the cosine (or CSLS) similarity matrix \(S\) between the current EEG queries and the image candidates.
2. Apply Sinkhorn normalization to obtain soft assignment \(P\).
3. Solve the weighted Procrustes problem to obtain orthogonal rotation \(R^*\).
4. Update the EEG queries: \(\mathbf{e}_i' \leftarrow \ell_2\text{-normalize}(\mathbf{e}_i' R^*)\).
5. Return to step 1.

Each iteration refines the soft assignment and the learned rotation jointly. The strongest sweeps use roughly 8 to 18 refinement steps, after which the rotation has converged to a stable correction of the held-out subject's coordinate system.

The strongest sweeps use:

- whitening shrinkage around `0.85`
- CSLS neighborhood size `1`
- Sinkhorn temperature around `0.1`
- roughly `8` to `18` refinement steps
- roughly `15` to `20` Sinkhorn normalization iterations

These settings correspond to the configurations that cluster around 71% in `sattc_sweep_summary.csv`.

## Why this works conceptually

The training stage learns a representation that is already reasonably transferable across subjects. But it does not completely solve subject mismatch. At test time, the adaptation stage uses the full unlabeled matching structure of the held-out subject's test fold to estimate how that subject's query geometry differs from the image space.

This explains the dramatic gap between:

- a strong but imperfect inter-subject baseline in the low-to-mid 30s;
- a fully adapted 200-way retrieval system around 71%.

The improvement is so large because the two stages solve different parts of the problem:

- training learns a broadly transferable stimulus representation;
- test-time adaptation estimates the held-out subject's geometric misalignment.

## Progressive sample-count experiments

The repository also asks a deeper question: how much unlabeled test data is needed before this adaptation strategy becomes effective?

To answer that, it evaluates the same adaptation logic while varying the number of available test queries. The result is exactly what one would expect if the method is genuinely exploiting set-level structure:

- with very few test samples, whitening and alignment statistics are noisy;
- as the number of test samples grows, the covariance estimate improves, the soft matching becomes more reliable, and the orthogonal fit becomes more stable;
- performance improves progressively as more unlabeled test items are available.

This is a useful experiment for the paper because it shows that the adaptation mechanism is not an arbitrary post-processing trick. It behaves in a systematic way that matches its intended statistical role.

## Transfer across disjoint halves of the test set

The most conceptually interesting analysis in the repository is the split-test transfer experiment.

The test fold is partitioned into two disjoint subsets. A whitening transform \(W\) and orthogonal alignment \(R^*\) are fitted on the first subset only, then applied to the second subset, which was not used to estimate them. This creates a cleaner question:

"Does the learned transformation capture subject-level structure, or is it merely overfitting the specific fitted items?"

The answer from the repository is that it does capture reusable subject-level structure, but not in an all-or-nothing way.

## Why blending with the identity helps

Applying the full learned orthogonal transformation can sometimes be too aggressive. If the fit is partly specialized to the calibration subset, then a full transfer may overshoot when applied to unseen items. The repository therefore introduces a blending coefficient \(\alpha \in [0, 1]\) that interpolates between the identity and the fully learned rotation, then re-projects the result back onto the orthogonal group. Concretely, form the convex mixture

\[
\tilde{R}_\alpha = (1 - \alpha)\, I_d + \alpha\, R^*
\]

and then find its nearest orthogonal matrix by computing the SVD \(\tilde{R}_\alpha = U \Sigma V^\top\) and returning

\[
R_\alpha = U V^\top.
\]

This two-step procedure is more principled than using \(\tilde{R}_\alpha\) directly. A pure convex combination of two orthogonal matrices is in general not orthogonal — its singular values are not all equal to one — and applying it would introduce anisotropic rescaling on top of the rotation correction. By re-projecting via SVD, the method finds the rotation closest (in Frobenius norm) to the blend, so the resulting transformation is always a pure rotation. When \(\alpha = 0\) the SVD of \(I_d\) gives \(I_d\) itself; when \(\alpha = 1\) it gives back \(R^*\); for intermediate \(\alpha\) the result is a rotation that lies geometrically between the two. The blending allows the method to keep only the portion of the learned correction that generalizes best.

Empirically, intermediate values of \(\alpha\) are often best. In the default 100/100 split:

- whitening alone on the unseen half gives `49.4`
- applying a blended transformation with `alpha = 0.5` gives `59.3`

This roughly 10-point improvement is one of the strongest results in the repository from a scientific point of view, because it shows that the learned transformation is not merely matching the fitted items themselves. It carries subject-specific information that remains useful on unseen samples.

## Subject identity probes as a diagnostic

The repository includes an optional diagnostic mechanism for assessing how much residual subject-identity information survives the encoding pipeline. The idea is to train a shallow linear classifier each epoch to predict which of the \(S\) training subjects produced a given EEG embedding, and to do this at two different stages of the network:

1. **Backbone representation** — the raw output of the EEG encoder before the alignment projector.
2. **Alignment-space representation** — the final embedding used in the contrastive loss, after the projector has reduced it to dimension \(d\).

A small held-out fraction of the training data (typically 10%) is reserved for probe evaluation. The rest is used to train the probe head via one epoch of Adam per main training epoch. At the end of each epoch, probe accuracy is logged for both representations.

The diagnostic value is in the comparison between the two levels. If the compact alignment space is working as intended — forcing the encoder to discard subject-specific nuisance variation — then the linear probe should be measurably less accurate on the alignment-space representation than on the backbone. A large drop signals that the bottleneck and the multi-positive training objective are successfully suppressing subject identity, while a small drop suggests that subject-specific geometry has leaked through into the alignment space.

This is directly relevant to the paper narrative: it provides a more granular view of what the compact alignment dimension is actually doing beyond the top-1 retrieval number. A model that achieves strong cross-subject retrieval but also achieves low subject-probe accuracy in the alignment space has cleaner evidence of subject-invariant learning than one where probe accuracy remains high throughout.

## The cleanest high-level narrative for a paper draft

A concise and faithful way to describe the repository is the following.

First, the system builds a shared EEG-image representation using a strong frozen visual target space, a temporal-spatial EEG encoder, and a deliberately compact alignment dimension. Training is organized so that several subjects viewing the same exact image appear together, which makes multi-positive cross-subject learning possible and improves inter-subject robustness. Additional training regularization, including same-stimulus cross-subject mixing, can push the inter-subject baseline into the mid-30% range.

Second, the system treats the held-out subject's 200-way test fold not as 200 isolated queries, but as a structured unlabeled set. It whitens the held-out subject's EEG features via ZCA with shrinkage covariance, corrects hubness via CSLS, imposes soft one-to-one matching structure via Sinkhorn normalization, and repeatedly fits an orthogonal rotation from the soft-assignment-weighted Procrustes problem between the EEG and image spaces. This raises average top-1 accuracy to about 71%.

Third, the split-test transfer experiments show that the learned orthogonal correction is not purely tied to the particular samples used to estimate it. When transferred cautiously through blending with the identity, it improves performance on unseen samples from the same held-out subject by about 10 absolute points.

## Main takeaways

The main technical lessons supported by this repository are:

1. The choice of representation matters enormously. Intermediate-layer visual embeddings and a compact shared space make the inter-subject problem much easier.
2. Training batches should reflect the structure of the task. Putting different subjects who saw the same exact image in the same batch is a major enabler of good inter-subject learning.
3. The inter-subject problem is not solved by training alone. A large part of the remaining gap is geometric subject mismatch.
4. Unlabeled test-time adaptation can exploit the one-to-one structure of the 200-way test fold very effectively, through a combination of ZCA whitening, CSLS hubness correction, Sinkhorn-based soft permutation assignment, and weighted Procrustes rotation.
5. The learned orthogonal correction carries subject-level information that partially transfers to unseen test items, especially when regularized through blending.

The most important caveat for later paper writing is simply to keep track of which exact result family supports which claim: the mid-30s results come from one set of inter-subject training runs, the ~71% results come from a later adaptation pipeline applied to a strong source model, and the split-test transfer results answer a different scientific question about how reusable the learned transformation really is.
