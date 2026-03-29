# Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style

Julius von Kügelgen^{∗1,2} &Yash Sharma^{∗3,4} &Luigi Gresele^{∗1} &Wieland Brendel ^{3} &Bernhard Schölkopf^{†1} &Michel Besserve^{†1} &Francesco Locatello^{†5} ^{1} Max Planck Institute for Intelligent Systems Tübingen ^{2} University of Cambridge ^{3} Tübingen AI Center, University of Tübingen ^{4} IMPRS for Intelligent Systems ^{5} Amazon

###### Abstract

Self-supervised representation learning has shown remarkable success in a number of domains. A common practice is to perform data augmentation via hand-crafted transformations intended to leave the semantics of the data invariant. We seek to understand the empirical success of this approach from a theoretical perspective. We formulate the augmentation process as a latent variable model by postulating a partition of the latent representation into a content component, which is assumed invariant to augmentation, and a style component, which is allowed to change. Unlike prior work on disentanglement and independent component analysis, we allow for both nontrivial statistical and causal dependencies in the latent space. We study the identifiability of the latent representation based on pairs of views of the observations and prove sufficient conditions that allow us to identify the invariant content partition up to an invertible mapping in both generative and discriminative settings. We find numerical simulations with dependent latent variables are consistent with our theory. Lastly, we introduce Causal3DIdent, a dataset of high-dimensional, visually complex images with rich causal dependencies, which we use to study the effect of data augmentations performed in practice.

## 1 Introduction

Learning good representations of high-dimensional observations from large amounts of unlabelled data is widely recognised as an important step for more capable and data-efficient learning systems *[10, 72]*. Over the last decade, self-supervised learning (SSL) has emerged as the dominant paradigm for such unsupervised representation learning *[1, 20, 21, 34, 41, 47, 48, 90, 91, 115, 122, 125, 126]*. The main idea behind SSL is to extract a supervisory signal from unlabelled observations by leveraging known structure of the data, which allows for the application of supervised learning techniques. A common approach is to directly predict some part of the observation from another part (e.g., future from past, or original from corruption), thus forcing the model to learn a meaningful representation in the process. While this technique has shown remarkable success in natural language processing *[13, 23, 30, 81, 84, 86, 95, 99]* and speech recognition *[5, 6, 100, 104]*, where a finite dictionary allows one to output a distribution over the missing part, such predictive SSL methods are not easily applied to continuous or high-dimensional domains such as vision. Here, a common approach is to learn a joint embedding of similar observations or views such that their representation is close *[7, 12, 22, 44]*. Different views can come, for example, from different modalities (text & speech; video & audio) or time points. As still images lack such multi-modality or temporal structure, recent advances in representation learning have relied on generating similar views by means of data augmentation.

---

In order to be useful, data augmentation is thought to require the transformations applied to generate additional views to be generally chosen to preserve the semantic characteristics of an observation, while changing other "nuisance" aspects. While this intuitively makes sense and has shown remarkable empirical results, the success of data augmentation techniques in practice is still not very well understood from a theoretical perspective—despite some efforts [17, 19, 28]. In the present work, we seek to better understand the empirical success of SSL with data augmentation by formulating the generative process as a latent variable model (LVM) and studying identifiability of the representation, i.e., under which conditions the ground truth latent factors can provably be inferred from the data [77].

Related work and its relation to the current. Prior work on unsupervised representation learning from an LVM perspective often postulates mutually independent latent factors: this independence assumption is, for example, at the heart of independent component analysis (ICA) [24, 56] and disentanglement [10, 14, 18, 49, 65, 71]. Since it is impossible to identify the true latent factors without any supervisory signal in the general nonlinear case [57, 82], recent work has turned to weakly- or self-supervised approaches which leverage additional information in the form of multiple views [39, 83, 108, 129], auxiliary variables [58, 63], or temporal structure [45, 54, 55, 69]. To identify or disentangle the individual independent latent factors, it is typically assumed that there is a chance that each factor changes across views, environments, or time points.

Our work—being directly motivated by common practices in SSL with data augmentation—differs from these works in the following two key aspects (see Fig. 1 for an overview). First, we do not assume independence and instead allow for both nontrivial statistical and causal relations between latent variables. This is in line with a recently proposed [105] shift towards causal representation learning [40, 76, 85, 87, 106, 107, 112, 123, 127], motivated by the fact that many underlying variables of interest may not be independent but causally related to each other. Second, instead of a scenario wherein all latent factors may change as a result of augmentation, we assume a partition of the latent space into two blocks: a content block which is shared or invariant across different augmented views, and a style block that may change. This is aligned with the notion that augmentations leave certain semantic aspects (i.e., content) intact and only affect style, and is thus a more appropriate assumption for studying SSL. In line with earlier work [39, 54, 57, 58, 63, 69, 82, 83, 129], we focus on the setting of continuous ground-truth latents, though we believe our results to hold more broadly.

![[SSL_provably_isolates_p2_img1.jpeg]]
Figure 1: Overview of our problem formulation. We partition the latent variable  $\mathbf{z}$  into content  $\mathbf{c}$  and style  $\mathbf{s}$ , and allow for statistical and causal dependence of style on content. We assume that only style changes between the original view  $\mathbf{x}$  and the augmented view  $\tilde{\mathbf{x}}$ , i.e., they are obtained by applying the same deterministic function  $\mathbf{f}$  to  $\mathbf{z} = (\mathbf{c}, \mathbf{s})$  and  $\tilde{\mathbf{z}} = (\mathbf{c}, \tilde{\mathbf{s}})$ .

Structure and contributions. Following a review of SSL with data augmentation and identifiability theory (§ 2), we formalise the process of data generation and augmentation as an LVM with content and style variables (§ 3). We then establish identifiability results of the invariant content partition (§ 4), validate our theoretical insights experimentally (§ 5), and discuss our findings and their limitations in the broader context of SSL with data augmentation (§ 6). We highlight the following contributions:

- we prove that SSL with data augmentations identifies the invariant content partition of the representation in generative (Thm. 4.2) and discriminative learning with invertible (Thm. 4.3) and non-invertible encoders with entropy regularisation (Thm. 4.4); in particular, Thm. 4.4 provides a theoretical justification for the empirically observed effectiveness of contrastive SSL methods that use data augmentation and InfoNCE [91] as an objective, such as SimCLR [20];
- we show that our theory is consistent with results in simulating statistical dependencies within blocks of content and style variables, as well as with style causally dependent on content (§ 5.1);
- we introduce Causal3DIdent, a dataset of 3D objects which allows for the study of identifiability in a causal representation learning setting, and use it to perform a systematic study of data augmentations used in practice, yielding novel insights on what particular data augmentations are truly isolating as invariant content and discarding as varying style when applied (§ 5.2).

---

2 Preliminaries and background

#### Self-supervised representation learning with data augmentation.

Given an unlabelled dataset of observations (e.g., images) $\mathbf{x}$, data augmentation techniques proceed as follows. First, a set of observation-level transformations $\mathbf{t}\in\mathcal{T}$ are specified together with a distribution $p_{\mathbf{t}}$ over $\mathcal{T}$. Both $\mathcal{T}$ and $p_{\mathbf{t}}$ are typically designed using human intelligence and domain knowledge with the intention of *not changing the semantic characteristics* of the data (which arguably constitutes a form of weak supervision). For images, for example, a common choice for $\mathcal{T}$ are combinations of random crops *[113]*, horizontal or vertical flips, blurring, colour distortion *[52, 113]*, or cutouts *[31]*; and $p_{\mathbf{t}}$ is a distribution over the parameterisation of these transformations, e.g., the centre and size of a crop *[20, 31]*. For each observation $\mathbf{x}$, a pair of transformations $\mathbf{t},\mathbf{t}^{\prime}\sim p_{\mathbf{t}}$ is sampled and applied separately to $\mathbf{x}$ to generate a pair of augmented views $(\tilde{\mathbf{x}},\tilde{\mathbf{x}}^{\prime})=(\mathbf{t}(\mathbf{x}),\mathbf{t}^{\prime}(\mathbf{x}))$.

The joint-embedding approach to SSL then uses a pair of encoder functions $(\mathbf{g},\mathbf{g}^{\prime})$, i.e. deep nets, to map the pair $(\tilde{\mathbf{x}},\tilde{\mathbf{x}}^{\prime})$ to a typically lower-dimensional representation $(\tilde{\mathbf{z}},\tilde{\mathbf{z}}^{\prime})=(\mathbf{g}(\tilde{\mathbf{x}}),\mathbf{g}^{\prime}(\tilde{\mathbf{x}}^{\prime}))$. Often, the two encoders are either identical, $\mathbf{g}=\mathbf{g}^{\prime}$, or directly related (e.g., via shared parameters or asynchronous updates). Then, the encoder(s) $(\mathbf{g},\mathbf{g}^{\prime})$ are trained such that the representations $(\tilde{\mathbf{z}},\tilde{\mathbf{z}}^{\prime})$ are “close”, i.e., such that $\text{sim}(\tilde{\mathbf{z}},\tilde{\mathbf{z}}^{\prime})$ is large for some similarity metric $\text{sim}(\cdot)$, e.g., the cosine similarity *[20, 129]*, or negative L2 norm *[129]*. The advantage of directly optimising for similarity in representation space over generative alternatives is that reconstruction can be very challenging for high-dimensional data. The disadvantage is the problem of *collapsed representations*. To avoid collapsed representations and force the encoder(s) to learn a meaningful representation, two main families of approaches have been used: (i) *contrastive learning* (CL) *[20, 47, 48, 91, 115, 126]*; and (ii) *regularisation-based* SSL *[21, 41, 128]*.

The idea behind CL is to not only learn similar representations for augmented views $(\tilde{\mathbf{x}}_{i},\tilde{\mathbf{x}}_{i}^{\prime})$ of the same $\mathbf{x}_{i}$, or *positive pairs*, but to also use other observations $\mathbf{x}_{j}$ $(j\neq i)$ to contrast with, i.e., to enforce a dissimilar representation across *negative pairs* $(\tilde{\mathbf{x}}_{i},\tilde{\mathbf{x}}_{j}^{\prime})$. In other words, CL pulls representations of positive pairs together, and pushes those of negative pairs apart. Since both aims cannot be achieved simultaneously with a constant representation, collapse is avoided. A popular CL objective function (used, e.g., in SimCLR *[20]*) is InfoNCE *[91]* (based on noise-contrastive estimation *[42, 43]*):

$\mathcal{L}_{\text{InfoNCE}}(\mathbf{g};\tau,K)=\mathbb{E}_{\{\mathbf{x}_{i}\}_{i=1}^{K}\sim p_{\mathbf{x}}}\Big{[}-\sum_{i=1}^{K}\log\frac{\exp\{\text{sim}(\tilde{\mathbf{z}}_{i},\tilde{\mathbf{z}}_{i}^{\prime})/\tau\}}{\sum_{j=1}^{K}\exp\{\text{sim}(\tilde{\mathbf{z}}_{i},\tilde{\mathbf{z}}_{j}^{\prime})/\tau\}}\Big{]}$ (1)

where $\tilde{\mathbf{z}}=\mathbb{E}_{\mathbf{t}\sim p_{\mathbf{t}}}[\mathbf{g}(\mathbf{t}(\mathbf{x}))]$, $\tau$ is a temperature, and $K-1$ is the number of negative pairs. InfoNCE (1) has an interpretation as multi-class logistic regression, and lower bounds the mutual information across similar views $(\tilde{\mathbf{z}},\tilde{\mathbf{z}}^{\prime})$—a common representation learning objective *[4, 9, 15, 50, 75, 79, 80, 97, 120]*. Moreover, (1) can be interpreted as *alignment* (numerator) and *uniformity* (denominator) terms, the latter constituting a nonparametric entropy estimator of the representation as $K\to\infty$ *[124]*. CL with InfoNCE can thus be seen as alignment of positive pairs with (approximate) entropy regularisation.

Instead of using negative pairs, as in CL, a set of recent SSL methods only optimise for alignment and avoid collapsed representations through different forms of regularisation. For example, BYOL *[41]* and SimSiam *[21]* rely on “architectural regularisation” in the form of moving-average updates for a separate “target” net $\mathbf{g}^{\prime}$ (BYOL only) or a stop-gradient operation (both). BarlowTwins *[128]*, on the other hand, optimises the cross correlation between $(\tilde{\mathbf{z}},\tilde{\mathbf{z}}^{\prime})$ to be close to the identity matrix, thus enforcing redundancy reduction (zero off-diagonals) in addition to alignment (ones on the diagonal).

#### Identifiability of learned representations.

In this work, we address the question of whether SSL with data augmentation can reveal or uncover properties of the underlying data generating process. Whether a representation learned from observations can be expected to match the true underlying latent factors—up to acceptable ambiguities and subject to suitable assumptions on the generative process and inference model—is captured by the notion of identifiability *[77]*.

Within representation learning, identifiability has mainly been studied in the framework of (nonlinear) ICA which assumes a model of the form $\mathbf{x}=\mathbf{f}(\mathbf{z})$ and aims to recover the independent latents, or *sources*, $\mathbf{z}$, typically up to permutation or element-wise transformation. A crucial negative result states that, with i.i.d. data and without further assumptions, this is fundamentally impossible *[57]*. However, recent breakthroughs have shown that identifiability can be achieved if an auxiliary variable (e.g.,

---

a time stamp or environment index) renders the sources *conditionally* independent *[45, 54, 55, 58]*. These methods rely on constructing positive and negative pairs using the auxiliary variable and learning a representation with CL. This development has sparked a renewed interest in identifiability in the context of deep representation learning *[63, 64, 69, 83, 102, 108, 109, 129]*.

Most closely related to SSL with data augmentation are works which study identifiability when given a second view $\tilde{\mathbf{x}}$ of an observation $\mathbf{x}$, resulting from a modified version $\tilde{\mathbf{z}}$ of the underlying latents or sources $\mathbf{z}$ *[39, 69, 83, 101, 108, 129]*. Here, $\tilde{\mathbf{z}}$ is either an element-wise corruption of $\mathbf{z}$ *[39, 69, 101, 129]* or may share a random subset of its components *[83, 108]*. Crucially, all previously mentioned works assume that *any* of the independent latents (are allowed to) change, and aim to identify the individual factors. However, in the context of SSL with data augmentation, where the semantic (content) part of the representation is intended to be shared between views, this assumption does not hold.

## 3 Problem formulation

We specify our problem setting by formalising the processes of data generation and augmentation. We take a latent-variable model perspective and assume that observations $\mathbf{x}$ (e.g., images) are generated by a *mixing* function $\mathbf{f}$ which takes a latent code $\mathbf{z}$ as input. Importantly, we describe the augmentation process through changes in this latent space as captured by a conditional distribution $p_{\tilde{\mathbf{z}}|\mathbf{z}}$, as opposed to traditionally describing the transformations $\mathbf{t}$ as acting directly at the observation level.

Formally, let $\mathbf{z}$ be a continuous r.v. taking values in an open, simply-connected $n$-dim. *representation space* $\mathcal{Z}\subseteq\mathbb{R}^{n}$ with associated probability density $p_{\mathbf{z}}$. Moreover, let $\mathbf{f}:\mathcal{Z}\rightarrow\mathcal{X}$ be a *smooth and invertible* mapping to an *observation space* $\mathcal{X}\subseteq\mathbb{R}^{d}$ and let $\mathbf{x}$ be the continuous r.v. defined as $\mathbf{x}=\mathbf{f}(\mathbf{z})$. The generative process for the dataset of original observations of $\mathbf{x}$ is thus given by:

$\mathbf{z}\sim p_{\mathbf{z}},\qquad\qquad\mathbf{x}=\mathbf{f}(\mathbf{z}).$ (2)

Next, we formalise the data augmentation process. As stated above, we take a representation-centric view, i.e., we assume that an augmentation $\tilde{\mathbf{x}}$ of the original $\mathbf{x}$ is obtained by applying the same mixing or rendering function $\mathbf{f}$ to a modified representation $\tilde{\mathbf{z}}$ which is (stochastically) related to the original representation $\mathbf{z}$ of $\mathbf{x}$. Specifying the effect of data augmentation thus corresponds to specifying a conditional distribution $p_{\tilde{\mathbf{z}}|\mathbf{z}}$ which captures the relation between $\mathbf{z}$ and $\tilde{\mathbf{z}}$.

In terms of the transformation-centric view presented in § 2, we can view the modified representation $\tilde{\mathbf{z}}\in\mathcal{Z}$ as obtained by applying $\mathbf{f}^{-1}$ to a transformed observation $\tilde{\mathbf{x}}=\mathbf{t}(\mathbf{x})\in\mathcal{X}$ where $\mathbf{t}\sim p_{\mathbf{t}}$, i.e., $\tilde{\mathbf{z}}=\mathbf{f}^{-1}(\tilde{\mathbf{x}})$. The conditional distribution $p_{\tilde{\mathbf{z}}|\mathbf{z}}$ in the representation space can thus be viewed as being induced by the distribution $p_{\mathbf{t}}$ over transformations applied at the observation level.

We now encode the notion that the set of transformations $\mathcal{T}$ used for augmentation is typically chosen such that any transformation $\mathbf{t}\in\mathcal{T}$ leaves certain aspects of the data invariant. To this end, we assume that *the representation $\mathbf{z}$ can be uniquely partitioned into two disjoint parts*:

1. an *invariant* part $\mathbf{c}$ which will *always be shared* across $(\mathbf{z},\tilde{\mathbf{z}})$, and which we refer to as *content*;
2. a *varying* part $\mathbf{s}$ which *may change* across $(\mathbf{z},\tilde{\mathbf{z}})$, and which we refer to as *style*.

We assume that $\mathbf{c}$ and $\mathbf{s}$ take values in content and style subspaces $\mathcal{C}\subseteq\mathbb{R}^{n_{c}}$ and $\mathcal{S}\subseteq\mathbb{R}^{n_{s}}$, respectively, i.e., $n=n_{c}+n_{s}$ and $\mathcal{Z}=\mathcal{C}\times\mathcal{S}$. W.l.o.g., we let $\mathbf{c}$ corresponds to the first $n_{c}$ dimensions of $\mathbf{z}$:

$\mathbf{z}=(\mathbf{c},\mathbf{s}),\qquad\qquad$ $\mathbf{c}:=\mathbf{z}_{1:n_{c}},\qquad\qquad$ $\mathbf{s}:=\mathbf{z}_{(n_{c}+1):n},$

We formalise the process of data augmentation with content-preserving transformations by defining the conditional $p_{\tilde{\mathbf{z}}|\mathbf{z}}$ such that only a (random) subset of the style variables change at a time.

###### Assumption 3.1 (Content-invariance).

The conditional density $p_{\tilde{\mathbf{z}}|\mathbf{z}}$ over $\mathcal{Z}\times\mathcal{Z}$ takes the form

$p_{\tilde{\mathbf{z}}|\mathbf{z}}(\tilde{\mathbf{z}}|\mathbf{z})=\delta(\tilde{\mathbf{c}}-\mathbf{c})p_{\tilde{\mathbf{s}}|\mathbf{s}}(\tilde{\mathbf{s}}|\mathbf{s})$

for some continuous density $p_{\tilde{\mathbf{s}}|\mathbf{s}}$ on $\mathcal{S}\times\mathcal{S}$, where $\delta(\cdot)$ is the Dirac delta function, i.e., $\tilde{\mathbf{c}}=\mathbf{c}$ a.e.

###### Assumption 3.2 (Style changes).

Let $\mathcal{A}$ be the set of subsets of style variables $A\subseteq\{1,...,n_{s}\}$ and let $p_{A}$ be a distribution on $\mathcal{A}$. Then, the style conditional $p_{\tilde{\mathbf{s}}|\mathbf{s}}$ is obtained via

$A\sim p_{A},\qquad\qquad p_{\tilde{\mathbf{s}}|\mathbf{s},A}(\tilde{\mathbf{s}}|\mathbf{s},A)=\delta(\tilde{\mathbf{s}}_{A^{c}}-\mathbf{s}_{A^{c}})p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}(\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A})\,,$

where $p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}$ is a continuous density on $\mathcal{S}_{A}\times\mathcal{S}_{A}$, $\mathcal{S}_{A}\subseteq\mathcal{S}$ denotes the subspace of changing style variables specified by $A$, and $A^{c}=\{1,...,n_{s}\}\setminus A$ denotes the complement of $A$.

##

---

Note that Assumption 3.2 is less restrictive than assuming that all style variables need to change, since it also allows for only a (possibly different) subset of style variables to change for any given observation. This is in line with the intuition that not all transformations affect all changeable (i.e., style) properties of the data: e.g., a colour distortion should not affect positional information, and, in the same vein, a (horizontal or vertical) flip should not affect the colour spectrum.

The generative process of an augmentation or transformed observation $\tilde{\mathbf{x}}$ is thus given by

$A\sim p_{A},\qquad\quad\tilde{\mathbf{z}}|\mathbf{z},A\sim p_{\tilde{\mathbf{z}}|\mathbf{z},A},\qquad\quad\tilde{\mathbf{x}}=\mathbf{f}(\tilde{\mathbf{z}}).$ (3)

Our setting for modelling data augmentation differs from that commonly assumed in (multi-view) disentanglement and ICA in that *we do not assume that the latent factors $\mathbf{z}=(\mathbf{c},\mathbf{s})$ are mutually (or conditionally) independent*, i.e., we allow for *arbitrary* (non-factorised) marginals $p_{\mathbf{z}}$ in (2).

Causal interpretation: data augmentation as counterfactuals under soft style intervention. We now provide a causal account of the above data generating process by describing the (allowed) causal dependencies among latent variables using a structural causal model (SCM) *[94]*. As we will see, this leads to an interpretation of data augmentations as counterfactuals in the underlying latent SCM. The assumption that $\mathbf{c}$ stays invariant as $\mathbf{s}$ changes is consistent with the view that content may causally influence style, $\mathbf{c}\rightarrow\mathbf{s}$, but not vice versa, see Fig. 1. We therefore formalise their relation as:

$\mathbf{c}:=\mathbf{f}_{\mathbf{c}}(\mathbf{u}_{\mathbf{c}}),\qquad\qquad\mathbf{s}:=\mathbf{f}_{\mathbf{s}}(\mathbf{c},\mathbf{u}_{\mathbf{s}}),\qquad\qquad(\mathbf{u}_{\mathbf{c}},\mathbf{u}_{\mathbf{s}})\sim p_{\mathbf{u}_{\mathbf{c}}}\times p_{\mathbf{u}_{\mathbf{s}}}$

where $\mathbf{u}_{\mathbf{c}},\mathbf{u}_{\mathbf{s}}$ are independent exogenous variables, and $\mathbf{f}_{\mathbf{c}},\mathbf{f}_{\mathbf{s}}$ are deterministic functions. The latent causal variables $(\mathbf{c},\mathbf{s})$ are subsequently decoded into observations $\mathbf{x}=\mathbf{f}(\mathbf{c},\mathbf{s})$. Given a factual observation $\mathbf{x}^{\text{F}}=\mathbf{f}(\mathbf{c}^{\text{F}},\mathbf{s}^{\text{F}})$ which resulted from $(\mathbf{u}^{\text{F}}_{\mathbf{c}},\mathbf{u}^{\text{F}}_{\mathbf{s}})$, we may ask the counterfactual question: “*what would have happened if the style variables had been (randomly) perturbed, all else being equal?*”. Consider, e.g., a *soft intervention* *[35]* on $\mathbf{s}$, i.e., an intervention that changes the mechanism $\mathbf{f}_{\mathbf{s}}$ to

$do(\mathbf{s}:=\tilde{\mathbf{f}}_{\mathbf{s}}(\mathbf{c},\mathbf{u}_{\mathbf{s}},\mathbf{u}_{A})),$

where $\mathbf{u}_{A}$ is an additional source of stochasticity accounting for the randomness of the augmentation process ($p_{A}\times p_{\tilde{\mathbf{s}}|\mathbf{s},A}$). The resulting distribution over counterfactual observations $\mathbf{x}^{\text{CF}}=\mathbf{f}(\mathbf{c}^{\text{F}},\mathbf{s}^{\text{CF}})$ can be computed from the modified SCM by fixing the exogenous variables to their factual values and performing the soft intervention:

$\mathbf{c}^{\text{CF}}:=\mathbf{c}^{\text{F}},\qquad\qquad\mathbf{s}^{\text{CF}}:=\tilde{\mathbf{f}}_{\mathbf{s}}(\mathbf{c}^{\text{F}},\mathbf{u}^{\text{F}}_{\mathbf{s}},\mathbf{u}_{A}),\qquad\qquad\mathbf{u}_{A}\sim p_{\mathbf{u}_{A}}.$

This aligns with our intuition and assumed problem setting of data augmentations as style corruptions. We note that the notion of augmentation as (hard) style interventions is also at the heart of ReLIC *[87]*, a recently proposed, causally-inspired SSL regularisation term for instance-discrimination *[44, 126]*. However, ReLIC assumes independence between content and style and does not address identifiability. For another causal perspective on data augmentation in the context of domain generalisation, c.f. *[59]*.

## 4 Theory: block-identifiability of the invariant content partition

Our goal is to prove that we can identify the invariant content partition $\mathbf{c}$ under a distinct, weaker set of assumptions, compared to existing results in disentanglement and nonlinear ICA *[39, 69, 83, 108, 129]*. We stress again that our primary interest is not to identify or disentangle individual (and independent) latent factors $z_{j}$, but instead to separate content from style, such that the content variables can be subsequently used for downstream tasks. We first define this distinct notion of *block-identifiability*.

###### Definition 4.1 (Block-identifiability).

We say that the true content partition $\mathbf{c}=\mathbf{f}^{-1}(\mathbf{x})_{1:n_{c}}$ is *block-identified* by a function $\mathbf{g}:\mathcal{X}\rightarrow\mathcal{Z}$ if the inferred content partition $\hat{\mathbf{c}}=\mathbf{g}(\mathbf{x})_{1:n_{c}}$ contains *all* and *only* information about $\mathbf{c}$, i.e., if there exists an *invertible* function $\mathbf{h}:\mathbb{R}^{n_{c}}\rightarrow\mathbb{R}^{n_{c}}$ s.t. $\hat{\mathbf{c}}=\mathbf{h}(\mathbf{c})$.

Defn. 4.1 is related to independent subspace analysis *[16, 53, 73, 114]*, which also aims to identify blocks of random variables as opposed to individual factors, though under an *independence assumption across blocks*, and typically not within a multi-view setting as studied in the present work.

### 4.1 Generative self-supervised representation learning

First, we consider *generative* SSL, i.e., fitting a generative model to pairs $(\mathbf{x},\tilde{\mathbf{x}})$ of original and augmented views. We show that under our specified data generation and augmentation process (§ 3),

---

as well as suitable additional assumptions (stated and discussed in more detail below), it is possible to isolate (i.e., block-identify) the invariant content partition. Full proofs are included in Appendix A.

###### Theorem 4.2 (Identifying content with a generative model).

Consider the data generating process described in § 3, i.e., the pairs $(\mathbf{x},\tilde{\mathbf{x}})$ of original and augmented views are generated according to (2) and (3) with $p_{\tilde{\mathbf{z}}|\mathbf{z}}$ as defined in Assumptions 3.1 and 3.2. Assume further that

1. $\mathbf{f}:\mathcal{Z}\rightarrow\mathcal{X}$ is smooth and invertible with smooth inverse (i.e., a diffeomorphism);
2. $p_{\mathbf{z}}$ is a smooth, continuous density on $\mathcal{Z}$ with $p_{\mathbf{z}}(\mathbf{z})>0$ almost everywhere;
3. for any $l\in\{1,...,n_{s}\}$, $\exists A\subseteq\{1,...,n_{s}\}$ s.t. $l\in A$; $p_{A}(A)>0$; $p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}$ is smooth w.r.t. both $\mathbf{s}_{A}$ and $\tilde{\mathbf{s}}_{A}$; and for any $\mathbf{s}_{A}$, $p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}(\cdot|\mathbf{s}_{A})>0$ in some open, non-empty subset containing $\mathbf{s}_{A}$.

If, for a given $n_{s}$ ($1\leq n_{s}<n$), a generative model $(\hat{p}_{\mathbf{z}},\hat{p}_{A},\hat{p}_{\tilde{\mathbf{s}}|\mathbf{s},A},\hat{\mathbf{f}})$ assumes the same generative process (§ 3), satisfies the above assumptions (i)-(iii), and matches the data likelihood,

$p_{\mathbf{x},\tilde{\mathbf{x}}}(\mathbf{x},\tilde{\mathbf{x}})=\hat{p}_{\mathbf{x},\tilde{\mathbf{x}}}(\mathbf{x},\tilde{\mathbf{x}})\hskip 28.45274pt\forall(\mathbf{x},\tilde{\mathbf{x}})\in\mathcal{X}\times\mathcal{X},$

then it block-identifies the true content variables via $\mathbf{g}=\hat{\mathbf{f}}^{-1}$ in the sense of Defn. 4.1.

Proof sketch. First, show (using (i) and the matching likelihoods) that the representation $\hat{\mathbf{z}}=\mathbf{g}(\mathbf{x})$ extracted by $\mathbf{g}$ is related to the true $\mathbf{z}$ by a smooth invertible mapping $\mathbf{h}=\mathbf{g}\circ\mathbf{f}$ such that $\hat{\mathbf{c}}=\mathbf{h}(\mathbf{z})_{1:n_{c}}$ is invariant across $(\mathbf{z},\tilde{\mathbf{z}})$ almost surely w.r.t. $p_{\mathbf{z},\tilde{\mathbf{z}}}$. Second, show by contradiction (using (ii), (iii)) that $\mathbf{h}(\cdot)_{1:n_{c}}$ can, in fact, only depend on the true content $\mathbf{c}$ and not on style $\mathbf{s}$, for otherwise the invariance from step 1 would be violated in a region of the style (sub)space of measure greater than zero.

Intuition. Thm. 4.2 assumes that the number of content ($n_{c}$) and style ($n_{s}$) variables is known, and that there is a positive probability that each *style* variable may change, though not necessarily on its own, according to (iii). In this case, training a generative model of the form specified in § 3 (i.e., with an invariant content partition and subsets of changing style variables) by maximum likelihood on pairs $(\mathbf{x},\tilde{\mathbf{x}})$ will asymptotically (in the limit of infinite data) recover the true invariant content partition up to an invertible function, i.e., it isolates, or unmixes, content from style.

Discussion. The identifiability result of Thm. 4.2 for generative SSL is of potential relevance for existing variational autoencoder (VAE) *[68]* variants such as the GroupVAE *[51]*, or its adaptive version AdaGVAE *[83]*. Since, contrary to existing results, Thm. 4.2 does not assume independent latents, it may also provide a principled basis for generative causal representation learning algorithms *[76, 107, 127]*. However, an important limitation to its practical applicability is that generative modelling does not tend to scale very well to complex high-dimensional observations, such as images.

### 4.2 Discriminative self-supervised representation learning

We therefore next turn to a discriminative approach, i.e., directly learning an encoder function $\mathbf{g}$ which leads to a similar embedding across $(\mathbf{x},\tilde{\mathbf{x}})$. As discussed in § 2, this is much more common for SSL with data augmentations. First, we show that if an invertible encoder $\mathbf{g}$ is used, then learning a representation which is aligned in the first $n_{c}$ dimensions is sufficient to block-identify content.

###### Theorem 4.3 (Identifying content with an invertible encoder).

Assume the same data generating process (§ 3) and conditions (i)-(iv) as in Thm. 4.2. Let $\mathbf{g}:\mathcal{X}\rightarrow\mathcal{Z}$ be any smooth and *invertible* function which minimises the following functional:

$\mathcal{L}_{\mathrm{Align}}(\mathbf{g}):=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{\mathbf{x},\tilde{\mathbf{x}}}}\left[\left|\left|\mathbf{g}(\mathbf{x})_{1:n_{c}}-\mathbf{g}(\tilde{\mathbf{x}})_{1:n_{c}}\right|\right|_{2}^{2}\right]$ (4)

Then $\mathbf{g}$ block-identifies the true content variables in the sense of Definition 4.1.

Proof sketch. First, we show that the global minimum of (4) is reached by the smooth invertible function $\mathbf{f}^{-1}$. Thus, any other minimiser $\mathbf{g}$ must satisfy the same invariance across $(\mathbf{x},\tilde{\mathbf{x}})$ used in step 1 of the proof of Thm. 4.2. The second step uses the same argument by contradiction as in Thm. 4.2.

Intuition. Thm. 4.3 states that if—under the same assumptions on the generative process as in Thm. 4.2—we directly learn a representation with an *invertible* encoder, then enforcing alignment between the first $n_{c}$ latents is sufficient to isolate the invariant content partition. Intuitively, invertibility guarantees that all information is preserved, thus avoiding a collapsed representation.

---

Discussion. According to Thm. 4.3, content can be isolated if, e.g., a flow-based architecture *[32, 33, 67, 92, 93]* is used, or invertibility is enforced otherwise during training *[8, 60]*. However, the applicability of this approach is limited as it places strong constraints on the encoder architecture which makes it hard to scale these methods up to high-dimensional settings. As discussed in § 2, state-of-the-art SSL methods such as SimCLR *[20]*, BYOL *[41]*, SimSiam *[21]*, or BarlowTwins *[128]* do not use invertible encoders, but instead avoid collapsed representations—which would result from naively optimising (4) for arbitrary, non-invertible $\mathbf{g}$—using different forms of regularisation.

To close this gap between theory and practice, finally, we investigate how to block-identify content without assuming an invertible encoder. We show that, if we add a regularisation term to (4) that encourages maximum entropy of the learnt representation, the invertibility assumption can be dropped.

###### Theorem 4.4 (Identifying content with discriminative learning and a non-invertible encoder).

Assume the same data generating process (§ 3) and conditions (i)-(iv) as in Thm. 4.2. Let $\mathbf{g}:\mathcal{X}\rightarrow(0,1)^{n_{c}}$ be any smooth function which minimises the following functional:

$\mathcal{L}_{\mathrm{AlignMaxEnt}}(\mathbf{g}):=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{\mathbf{x},\tilde{\mathbf{x}}}}\left[\left|\left|\mathbf{g}(\mathbf{x})-\mathbf{g}(\tilde{\mathbf{x}})\right|\right|_{2}^{2}\right]-H\left(\mathbf{g}(\mathbf{x})\right)$ (5)

where $H(\cdot)$ denotes the differential entropy of the random variable $\mathbf{g}(\mathbf{x})$ taking values in $(0,1)^{n_{c}}$. Then $\mathbf{g}$ block-identifies the true content variables in the sense of Defn. 4.1.

Proof sketch. First, use the Darmois construction *[29, 57]* to build a function $\mathbf{d}:\mathcal{C}\rightarrow(0,1)^{n_{c}}$ mapping $\mathbf{c}=\mathbf{f}^{-1}(\mathbf{x})_{1:n_{c}}$ to a uniform random variable. Then $\mathbf{g}^{\star}=\mathbf{d}\circ\mathbf{f}_{1:n_{c}}^{-1}$ attains the global minimum of (5) because $\mathbf{c}$ is invariant across $(\mathbf{x},\tilde{\mathbf{x}})$ and the uniform distribution is the maximum entropy distribution on $(0,1)^{n_{c}}$. Thus, any other minimiser $\mathbf{g}$ of (5) must satisfy invariance across $(\mathbf{x},\tilde{\mathbf{x}})$ and map to a uniform r.v. Then, use the same step 2 as in Thms. 4.2 and 4.3 to show that $\mathbf{h}=\mathbf{g}\circ\mathbf{f}:\mathcal{Z}\rightarrow(0,1)^{n_{c}}$ cannot depend on style, i.e., it is a function from $\mathcal{C}$ to $(0,1)^{n_{c}}$. Finally, we show that $\mathbf{h}$ must be invertible since it maps $p_{\mathbf{c}}$ to a uniform distribution, using a result from *[129]*.

Intuition. Thm. 4.4 states that if we do not explicitly enforce invertibility of $\mathbf{g}$ as in Thm. 4.3, additionally maximising the entropy of the learnt representation (i.e., optimising alignment and uniformity *[124]*) avoids a collapsed representation and recovers the invariant content block. Intuitively, this is because any function that only depends on $\mathbf{c}$ will be invariant across $(\mathbf{x},\tilde{\mathbf{x}})$, so it is beneficial to preserve all content information to maximise entropy.

Discussion. Of our theoretical results, Thm. 4.4 requires the weakest set of assumptions, and is most closely aligned with common SSL practice. As discussed in § 2, contrastive SSL with negative samples using InfoNCE (1) as an objective can asymptotically be understood as alignment with entropy regularisation *[124]*, i.e., objective (5). Thm. 4.4 thus provides a theoretical justification for the empirically observed effectiveness of CL with InfoNCE: subject to our assumptions, CL with InfoNCE asymptotically isolates content, i.e., the part of the representation that is always left invariant by augmentation. For example, the strong image classification performance based on representations learned by SimCLR *[20]*, which uses color distortion and random crops as augmentations, can be explained in that object class is a content variable in this case. We extensively evaluate the effect of various augmentation techniques on different ground-truth latent factors in our experiments in § 5. There is also an interesting connection between Thm. 4.4 and BarlowTwins *[128]*, which only uses positive pairs and combines alignment with a redundancy reduction regulariser that enforces decorrelation between the inferred latents. Intuitively, redundancy reduction is related to increased entropy: $\mathbf{g}^{\star}$ constructed in the proof of Thm. 4.4—and thus also any other minimiser of (5)—attains the global optimum of the BarlowTwins objective, though the reverse implication may not hold.

## 5 Experiments

We perform two main experiments. First, we numerically test our main result, Thm. 4.4, in a fully-controlled, finite sample setting (§ 5.1), using CL to estimate the entropy term in (5). Second, we seek to better understand the effect of data augmentations used in practice (§ 5.2). To this end, we introduce a new dataset of 3D objects with dependencies between a number of known ground-truth factors, and use it to evaluate the effect of different augmentation techniques on what is identified as content. Additional experiments are summarised in § 5.3 and described in more detail in Appendix C.

### 5.1 Numerical data

Experimental setup. We generate synthetic data as described in § 3. We consider $n_{c}=n_{s}=5$, with content and style latents distributed as $\mathbf{c}\sim\mathcal{N}(0,\Sigma_{c})$ and $\mathbf{s}|\mathbf{c}\sim\mathcal{N}(\mathbf{a}+B\mathbf{c},\Sigma_{s})$, thus allowing

---

![[SSL_provably_isolates_p8_img2.jpeg]]
Figure 2: (Left) Causal graph for the Causal3DIdent dataset. (Right) Two samples from each object class.

![[SSL_provably_isolates_p8_img3.jpeg]]

for statistical dependence within the two blocks (via  $\Sigma_{c}$  and  $\Sigma_{s}$ ) and causal dependence between content and style (via  $B$ ). For  $\mathbf{f}$ , we use a 3-layer MLP with LeakyReLU activation functions. The distribution  $p_A$  over subsets of changing style variables is obtained by independently flipping the same biased coin for each  $s_i$ . The conditional style distribution is taken as  $p_{\hat{\mathbf{s}}_A|\mathbf{s}_A} = \mathcal{N}(\mathbf{s}_A, \Sigma_A)$ . We train an encoder  $\mathbf{g}$  on pairs  $(\mathbf{x}, \hat{\mathbf{x}})$  with InfoNCE using the negative L2 loss as the similarity measure, i.e., we approximate (5) using empirical averages and negative samples. For evaluation, we use kernel ridge regression [88] to predict the ground truth  $\mathbf{c}$  and  $\mathbf{s}$  from the learnt representation  $\hat{\mathbf{c}} = \mathbf{g}(\mathbf{x})$  and report the  $R^2$  coefficient of determination. For a more detailed account, we refer to Appendix D.

Results. In the inset table, we report mean  $\pm$  std. dev. over 3 random seeds across four generative processes of increasing complexity covered by Thm. 4.4: "p(chg.)", "Stat.", and "Cau." denote respectively the change probability for each  $s_i$ , statistical dependence within blocks ( $\Sigma_c \neq I \neq \Sigma_s$ ), and causal dependence of style on content ( $B \neq 0$ ). An  $R^2$  close to

|  Generative process |   |   | R2 (nonlinear)  |   |
| --- | --- | --- | --- | --- |
|  p(chg.) | Stat. | Cau. | Content c | Style s  |
|  1.0 | X | X | 1.00 ± 0.00 | 0.07 ± 0.00  |
|  0.75 | X | X | 1.00 ± 0.00 | 0.06 ± 0.05  |
|  0.75 | ✓ | X | 0.98 ± 0.03 | 0.37 ± 0.05  |
|  0.75 | ✓ | ✓ | 0.99 ± 0.01 | 0.80 ± 0.08  |

one indicates that almost all variation is explained by  $\hat{\mathbf{c}}$ , i.e., that there is a 1-1 mapping, as required by Defn. 4.1. As can be seen, across all settings, content is block-identified. Regarding style, we observe an increased score with the introduction of dependencies, which we explain in an extended discussion in Appendix C.1. Finally, we show in Appendix C.1 that a high  $R^2$  score can be obtained even if we use linear regression to predict  $\mathbf{c}$  from  $\hat{\mathbf{c}}$  ( $R^2 = 0.98 \pm 0.01$ , for the last row).

# 5.2 High-dimensional images: Causal3DIdent

Causal3DIdent dataset. 3DIdent [129] is a benchmark for evaluating identifiability with rendered  $224 \times 224$  images which contains hallmarks of natural environments (e.g. shadows, different lighting conditions, a 3D object). For influence of the latent factors on the renderings, see Fig. 2 of [129]. In 3DIdent, there is a single object class (Teapot [89]), and all 10 latents are sampled independently. For Causal3DIdent, we introduce six additional classes: Hare [121], Dragon [110], Cow [62], Armadillo [70], Horse [98], and Head [111]; and impose a causal graph over the latent variables, see Fig. 2. While object class and all environment variables (spotlight position &amp; hue, background hue) are sampled independently, all object latents are dependent, $^{11}$  see Appendix B for details. $^{12}$

Experimental setup. For  $\mathbf{g}$ , we train a convolutional encoder composed of a ResNet18 [46] and an additional fully-connected layer, with LeakyReLU activation. As in SimCLR [20], we use InfoNCE with cosine similarity, and train on pairs of augmented examples  $(\hat{\mathbf{x}},\hat{\mathbf{x}}^{\prime})$ . As  $n_c$  is unknown and variable depending on the augmentation, we fix  $\dim (\hat{\mathbf{c}}) = 8$  throughout. Note that we find the results to be, for the most part, robust to the choice of  $\dim (\hat{\mathbf{c}})$ , see inset figure. We consider the following data augmentations (DA): crop, resize &amp; flip; colour distortion (jitter &amp; drop);

![[SSL_provably_isolates_p8_img4.jpeg]]

and rotation  $\in \{90^{\circ}, 180^{\circ}, 270^{\circ}\}$ . For comparison, we also consider directly imposing a content-style

---

Table 1: Causal3DIdent results:  $R^2$  mean ± std. dev. over 3 random seeds. DA: data augmentation, LT: latent transformation, bold:  $R^2 \geq 0.5$ , red:  $R^2 &lt; 0.25$ . Results for individual axes of object position &amp; rotation are aggregated, see Appendix C for the full table.

|  Views generated by | Class | Positions |   | Hues |   |   | Rotations  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  object | spotlight | object | spotlight | background  |   |
|  DA: colour distortion | 0.42 ± 0.01 | 0.61 ± 0.10 | 0.17 ± 0.00 | 0.10 ± 0.01 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.33 ± 0.02  |
|  LT: change hues | 1.00 ± 0.00 | 0.59 ± 0.33 | 0.91 ± 0.00 | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.30 ± 0.01  |
|  DA: crop (large) | 0.28 ± 0.04 | 0.09 ± 0.08 | 0.21 ± 0.13 | 0.87 ± 0.00 | 0.09 ± 0.02 | 1.00 ± 0.00 | 0.02 ± 0.02  |
|  DA: crop (small) | 0.14 ± 0.00 | 0.00 ± 0.01 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00  |
|  LT: change positions | 1.00 ± 0.00 | 0.16 ± 0.23 | 0.00 ± 0.01 | 0.46 ± 0.02 | 0.00 ± 0.00 | 0.97 ± 0.00 | 0.29 ± 0.01  |
|  DA: crop (large) + colour distortion | 0.97 ± 0.00 | 0.59 ± 0.07 | 0.59 ± 0.05 | 0.28 ± 0.00 | 0.01 ± 0.01 | 0.01 ± 0.00 | 0.74 ± 0.03  |
|  DA: crop (small) + colour distortion | 1.00 ± 0.00 | 0.69 ± 0.04 | 0.93 ± 0.00 | 0.30 ± 0.01 | 0.00 ± 0.00 | 0.02 ± 0.03 | 0.56 ± 0.03  |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.22 ± 0.22 | 0.07 ± 0.08 | 0.32 ± 0.02 | 0.00 ± 0.01 | 0.02 ± 0.03 | 0.34 ± 0.06  |
|  DA: rotation | 0.33 ± 0.06 | 0.17 ± 0.09 | 0.23 ± 0.12 | 0.83 ± 0.01 | 0.30 ± 0.12 | 0.99 ± 0.00 | 0.05 ± 0.03  |
|  LT: change rotations | 1.00 ± 0.00 | 0.53 ± 0.33 | 0.90 ± 0.00 | 0.41 ± 0.00 | 0.00 ± 0.00 | 0.97 ± 0.00 | 0.28 ± 0.00  |
|  DA: rotation + colour distortion | 0.59 ± 0.01 | 0.58 ± 0.06 | 0.21 ± 0.01 | 0.12 ± 0.02 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.33 ± 0.04  |
|  LT: change rotations + hues | 1.00 ± 0.00 | 0.57 ± 0.34 | 0.91 ± 0.00 | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.28 ± 0.00  |

partition by performing a latent transformation (LT) to generate views. For evaluation, we use linear logistic regression to predict object class, and kernel ridge to predict the other latents from  $\hat{\mathbf{c}}$ .

Results. The results are presented in Tab. 1. Overall, our main findings can be summarised as:

(i) it can be difficult to design image-level augmentations that leave specific latent factors invariant;
(ii) augmentations &amp; latent transformations generally have a similar effect on groups of latents;
(iii) augmentations that yield good classification performance induce variation in all other latents.

We observe that, similar to directly varying the hue latents, colour distortion leads to a discarding of hue information as style, and a preservation of (object) position as content. Crops, similar to varying the position latents, lead to a discarding of position as style, and a preservation of background and object hue as content, the latter assuming crops are sufficiently large. In contrast, image-level rotation affects both the object rotation and position, and thus deviates from only varying the rotation latents.

Whereas class is always preserved as content when generating views with latent transformations, when using data augmentations, we can only reliably decode class when crops &amp; colour distortion are used in conjunction—a result which mirrors evaluation on ImageNet [20]. As can be seen by our evaluation of crops &amp; colour distortion in isolation, while colour distortion leads to a discarding of hues as style, crops lead to a discarding of position &amp; rotation as style. Thus, when used in conjunction, class is isolated as the sole content variable. See Appendix C.2 for additional analysis.

# 5.3 Additional experiments and ablations

We also perform an ablation on  $\dim(\hat{\mathbf{c}})$  for the synthetic setting from § 5.1, see Appendix C.1 for details. Generally, we find that if  $\dim(\hat{\mathbf{c}}) &lt; n_c$ , there is insufficient capacity to encode all content, so a lower-dimensional mixture of content is learnt. Conversely, if  $\dim(\hat{\mathbf{c}}) &gt; n_c$ , the excess capacity is used to encode some style information (as that increases entropy). Further, we repeat our analysis from § 5.2 using BarlowTwins [128] (instead of SimCLR) which, as discussed at the end of § 4.2, is also loosely related to Thm. 4.4. The results mostly mirror those obtained for SimCLR and presented in Tab. 1, see Appendix C.2 for details. Finally, we ran the same experimental setup as in § 5.2 also on the MPI3D-real dataset [38] containing  $&gt;1$  million real images with ground-truth annotations of 3D objects being moved by a robotic arm. Subject to some caveats, the results show a similar trend as those on Causal3DIdent, see Appendix C.3 for details.

# 6 Discussion

Theory vs practice. We have made an effort to tailor our problem formulation (§ 3) to the setting of data augmentation with content-preserving transformations. However, some of our more technical assumptions, which are necessary to prove block-identifiability of the invariant content partition, may not hold exactly in practice. This is apparent, e.g., from our second experiment (§ 5.2), where we observe that—while class should, in principle, always be invariant across views (i.e., content)—when

---

using *only* crops, colour distortion, or rotation, $\mathbf{g}$ appears to encode *shortcuts* *[37, 96]*. Data augmentation, unlike latent transformations, generates views $\tilde{\mathbf{x}}$ which are not restricted to the 11-dim. image manifold $\mathcal{X}$ corresponding to the generative process of *Causal3DIdent*, but may introduce additional variation: e.g., colour distortion leads to a rich combination of colours, typically a 3-dim. feature, whereas *Causal3DIdent* only contains one degree of freedom (hue). With additional factors, any introduced invariances may be encoded as content in place of class. Image-level augmentations also tend to change multiple latent factors in a correlated way, which may violate assumption *(iii)* of our theorems, i.e., that $p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}$ is fully-supported locally. We also assume that $\mathbf{z}$ is continuous, even though *Causal3DIdent* and most disentanglement datasets also contain discrete latents. This is a very common assumption in the related literature *[39, 54, 57, 58, 63, 69, 82, 83, 129]* that may be relaxed in future work. Moreover, our theory holds asymptotically and at the global optimum, whereas in practice we solve a non-convex optimisation problem with a finite sample and need to approximate the entropy term in (5), e.g., using a finite number of negative pairs. The resulting challenges for optimisation may be further accentuated by the higher dimensionality of $\mathcal{X}$ induced by image-level augmentations. Finally, we remark that while, for simplicity, we have presented our theory for pairs $(\mathbf{x},\tilde{\mathbf{x}})$ of original and augmented examples, in practice, using pairs $(\tilde{\mathbf{x}},\tilde{\mathbf{x}}^{\prime})$ of two augmented views typically yields better performance. All of our assumptions (content invariance, changing style, etc) and theoretical results still apply to the latter case. We believe that using two augmented views helps because it leads to *increased variability* across the pair: for if $\tilde{\mathbf{x}}$ and $\tilde{\mathbf{x}}^{\prime}$ differ from $\mathbf{x}$ in style subsets $A$ and $A^{\prime}$, respectively, then $(\tilde{\mathbf{x}},\tilde{\mathbf{x}}^{\prime})$ differ from each other (a.s.) in the union $A\cup A^{\prime}$.

Beyond entropy regularisation. We have shown a clear link between an identifiable maximum entropy approach to SSL (Thm. 4.4) and SimCLR *[20]* based on the analysis of *[124]*, and have discussed an intuitive connection to the notion of redundancy reduction used in BarlowTwins *[128]*. Whether other types of regularisation such as the architectural approach pursued in BYOL *[41]* and SimSiam *[21]* can also be linked to entropy maximisation, remains an open question. Deriving similar results to Thm. 4.4 with other regularisers is a promising direction for future research, c.f. *[116]*.

The choice of augmentation technique implicitly defines content and style. As we have defined content as the part of the representation which is always left invariant across views, the choice of augmentation implicitly determines the content-style partition. This is particularly important to keep in mind when applying SSL with data augmentation to safety-critical domains, such as medical imaging. We also advise caution when using data augmentation to identify specific latent properties, since, as observed in § 5.2, image-level transformations may affect the underlying ground-truth factors in unanticipated ways. Also note that, *for a given downstream task*, we may not want to discard all style information since style variables may still be correlated with the task of interest and may thus help improve predictive performance. *For arbitrary downstream tasks*, however, where style may change in an adversarial way, it can be shown that only using content is optimal *[103]*.

What vs *how* information is encoded. We focus on *what* information is learnt by SSL with data augmentations by specifying a generative process and studying identifiability of the latent representation. Orthogonal to this, a different line of work instead studies *how* information is encoded by analysing the sample complexity needed to solve a *given downstream task* using a *linear* predictor *[3, 74, 116, 117, 118, 119]*. Provided that downstream tasks only involve content, we can draw some comparisons. Whereas our results recover content only up to arbitrary invertible nonlinear functions (see Defn. 4.1), our problem setting is more general: *[3, 74]* assume (approximate) independence of views $(\mathbf{x},\tilde{\mathbf{x}})$ given the task (content), while *[118, 119]* assume (approximate) independence between one view and the task (content) given the other view, neither of which hold in our setting.

Conclusion. Existing representation learning approaches typically assume mutually independent latents, though dependencies clearly exist in nature *[106]*. We demonstrate that in a *non-i.i.d.* scenario, e.g., by constructing multiple views of the same example with data augmentation, we can learn useful representations in the presence of this neglected phenomenon. More specifically, the present work contributes, to the best of our knowledge, the first: (i) identifiability result under *arbitrary dependence* between latents; and (ii) empirical study that evaluates the effect of data augmentations not only on classification, but also on other *continuous* ground-truth latents. Unlike existing identifiability results which rely on *change* as a learning signal, our approach aims to identify what is always shared across views, i.e., also using *invariance* as a learning signal. We hope that this change in perspective will be helpful for applications such as optimal style transfer or disentangling shape from pose in vision, and inspire other types of *counterfactual training* to recover a more fine-grained causal representation.


---

## Appendix A Proofs

### Overview:

- Appendix A contains the full proofs for all theoretical results from the main paper.
- Appendix B contains additional details and plots for the Causal3DIdent dataset.
- Appendix C contains additional experimental results and analysis.
- Appendix D contains additional implementation details for our experiments.

## Appendix A Proofs

We now present the full detailed proofs of our three theorems which were briefly sketched in the main paper. We remark that these proofs build on each other, in the sense that the (main) step 2 of the proof of Thm. 4.2 is also used in the proofs of Thms. 4.3 and 4.4.

### A.1 Proof of Thm. 4.2

###### Theorem 4.2 (Identifying content with a generative model).

Consider the data generating process described in § 3, i.e., the pairs $(\mathbf{x},\tilde{\mathbf{x}})$ of original and augmented views are generated according to (2) and (3) with $p_{\tilde{\mathbf{z}}|\mathbf{z}}$ as defined in Assumptions 3.1 and 3.2. Assume further that

1. $\mathbf{f}:\mathcal{Z}\rightarrow\mathcal{X}$ is smooth and invertible with smooth inverse (i.e., a diffeomorphism);
2. $p_{\mathbf{z}}$ is a smooth, continuous density on $\mathcal{Z}$ with $p_{\mathbf{z}}(\mathbf{z})>0$ almost everywhere;
3. for any $l\in\{1,...,n_{s}\}$, $\exists A\subseteq\{1,...,n_{s}\}$ s.t. $l\in A$; $p_{A}(A)>0$; $p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}$ is smooth w.r.t. both $\mathbf{s}_{A}$ and $\tilde{\mathbf{s}}_{A}$; and for any $\mathbf{s}_{A}$, $p_{\tilde{\mathbf{s}}_{A}|\mathbf{s}_{A}}(\cdot|\mathbf{s}_{A})>0$ in some open, non-empty subset containing $\mathbf{s}_{A}$.

If, for a given $n_{s}$ ($1\leq n_{s}<n$), a generative model $(\hat{p}_{\mathbf{z}},\hat{p}_{A},\hat{p}_{\tilde{\mathbf{s}}|\mathbf{x},A},\hat{\mathbf{f}})$ assumes the same generative process (§ 3), satisfies the above assumptions (i)-(iii), and matches the data likelihood,

$p_{\mathbf{x},\tilde{\mathbf{x}}}(\mathbf{x},\tilde{\mathbf{x}})=\hat{p}_{\mathbf{x},\tilde{\mathbf{x}}}(\mathbf{x},\tilde{\mathbf{x}})\hskip 28.45274pt\forall(\mathbf{x},\tilde{\mathbf{x}})\in\mathcal{X}\times\mathcal{X},$

then it block-identifies the true content variables via $\mathbf{g}=\hat{\mathbf{f}}^{-1}$ in the sense of Defn. 4.1.

###### Proof.

The proof consists of two main steps.

In the first step, we use assumption (i) and the matching likelihoods to show that the representation $\hat{\mathbf{z}}=\mathbf{g}(\mathbf{x})$ extracted by $\mathbf{g}=\hat{\mathbf{f}}^{-1}$ is related to the true latent $\mathbf{z}$ by a smooth invertible mapping $\mathbf{h}$, and that $\hat{\mathbf{z}}$ must satisfy invariance across $(\mathbf{x},\tilde{\mathbf{x}})$ in the first $n_{c}$ (content) components almost surely (a.s.) with respect to (w.r.t.) the true generative process.

In the second step, we then use assumptions (ii) and (iii) to prove (by contradiction) that $\hat{\mathbf{c}}:=\hat{\mathbf{z}}_{1:n_{c}}=\mathbf{h}(\mathbf{z})_{1:n_{c}}$ can, in fact, only depend on the true content $\mathbf{c}$ and not on the true style $\mathbf{s}$, for otherwise the invariance established in the first step would have be violated with probability greater than zero.

To provide some further intuition for the second step, the assumed generative process implies that $(\mathbf{c},\mathbf{s},\tilde{\mathbf{s}})|A$ is constrained to take values (a.s.) in a subspace $\mathcal{R}$ of $\mathcal{C}\times\mathcal{S}\times\mathcal{S}$ of dimension $n_{c}+n_{s}+|A|$ (as opposed to dimension $n_{c}+2n_{s}$ for $\mathcal{C}\times\mathcal{S}\times\mathcal{S}$). In this context, assumption (iii) implies that $(\mathbf{c},\mathbf{s},\tilde{\mathbf{s}})|A$ has a density with respect to a measure on this subspace equivalent to the Lebesgue measure on $\mathbb{R}^{n_{c}+n_{s}+|A|}$. This equivalence implies, in particular, that this ”subspace measure” is strictly positive: it takes strictly positive values on open sets of $\mathcal{R}$ seen as a topological subspace of $\mathcal{C}\times\mathcal{S}\times\mathcal{S}$. These open sets are defined by the induced topology: they are the intersection of the open sets of $\mathcal{C}\times\mathcal{S}\times\mathcal{S}$ with $\mathcal{R}$. An open set $B$ of $V$ on which $p(\mathbf{c},\mathbf{s},\tilde{\mathbf{s}}|A)>0$ then satisfies $P(B|A)>0$. We look for such an open set to prove our result.

#### Step 1.

From the assumed data generating process described in § 3—in particular, from the form of the model conditional $\hat{p}_{\tilde{\mathbf{z}}|\mathbf{z}}$ described in Assumptions 3.1 and 3.2—it follows that

$\mathbf{g}(\mathbf{x})_{1:n_{c}}=\mathbf{g}(\tilde{\mathbf{x}})_{1:n_{c}}$ (6)

a.s., i.e., with probability one, w.r.t. the model distribution $\hat{p}_{\mathbf{x},\tilde{\mathbf{x}}}$.

###

---

Due to the assumption of matching likelihoods, the invariance in (6) must also hold (a.s.) w.r.t. the true data distribution $p_{\mathbf{x},\tilde{\mathbf{x}}}$.

Next, since $\mathbf{f},\hat{\mathbf{f}}:\mathcal{Z}\to\mathcal{X}$ are smooth and invertible functions by assumption (i), there exists a smooth and invertible function $\mathbf{h}=\mathbf{g}\circ\mathbf{f}:\mathcal{Z}\to\mathcal{Z}$ such that

$\mathbf{g}=\mathbf{h}\circ\mathbf{f}^{-1}.$ (7)

Substituting (7) into (6), we obtain (a.s. w.r.t. $p$):

$\hat{\mathbf{c}}:=\hat{\mathbf{z}}_{1:n_{c}}=\mathbf{g}(\mathbf{x})_{1:n_{c}}=\mathbf{h}(\mathbf{f}^{-1}(\mathbf{x}))_{1:n_{c}}=\mathbf{h}(\mathbf{f}^{-1}(\tilde{\mathbf{x}}))_{1:n_{c}}$ (8)

Substituting $\mathbf{z}=\mathbf{f}^{-1}(\mathbf{x})$ and $\tilde{\mathbf{z}}=\mathbf{f}^{-1}(\tilde{\mathbf{x}})$ into (8), we obtain (a.s. w.r.t. $p$)

$\hat{\mathbf{c}}=\mathbf{h}(\mathbf{z})_{1:n_{c}}=\mathbf{h}(\tilde{\mathbf{z}})_{1:n_{c}}.$ (9)

It remains to show that $\mathbf{h}(\cdot)_{1:n_{c}}$ can only be a function of $\mathbf{c}$, i.e., does not depend on any other (style) dimension of $\mathbf{z}=(\mathbf{c},\mathbf{s})$.

#### Step 2.

Suppose *for a contradiction* that $\mathbf{h}_{c}(\mathbf{c},\mathbf{s}):=\mathbf{h}(\mathbf{c},\mathbf{s})_{1:n_{c}}=\mathbf{h}(\mathbf{z})_{1:n_{c}}$ depends on some component of the style variable $\mathbf{s}$:

$\exists l\in\{1,...,n_{s}\},(\mathbf{c}^{*},\mathbf{s}^{*})\in\mathcal{C}\times\mathcal{S},\qquad\text{s.t.}\qquad\frac{\partial\mathbf{h}_{c}}{\partial s_{l}}(\mathbf{c}^{*},\mathbf{s}^{*})\neq 0,$ (10)

that is, we assume that the partial derivative of $\mathbf{h}_{c}$ w.r.t. some style variable $s_{l}$ is non-zero at some point $\mathbf{z}^{*}=(\mathbf{c}^{*},\mathbf{s}^{*})\in\mathcal{Z}=\mathcal{C}\times\mathcal{S}$.

Since $\mathbf{h}$ is smooth, so is $\mathbf{h}_{c}$. Therefore, $\mathbf{h}_{c}$ has continuous (first) partial derivatives.

By continuity of the partial derivative, $\frac{\partial\mathbf{h}_{c}}{\partial s_{l}}$ must be non-zero in a neighbourhood of $(\mathbf{c}^{*},\mathbf{s}^{*})$, i.e.,

$\exists\eta>0\qquad\text{s.t.}\qquad s_{l}\mapsto\mathbf{h}_{c}\big{(}\mathbf{c}^{*},(\mathbf{s}_{-l}^{*},s_{l})\big{)}\quad\text{is strictly monotonic on}\quad(s_{l}^{*}-\eta,s_{l}^{*}+\eta),$ (11)

where $\mathbf{s}_{-l}\in\mathcal{S}_{-l}$ denotes the vector of remaining style variables except $s_{l}$.

Next, define the auxiliary function $\psi:\mathcal{C}\times\mathcal{S}\times\mathcal{S}\to\mathbb{R}_{\geq 0}$ as follows:

$\psi(\mathbf{c},\mathbf{s},\tilde{\mathbf{s}}):=|\mathbf{h}_{c}(\mathbf{c},\mathbf{s})-\mathbf{h}_{c}(\mathbf{c},\tilde{\mathbf{s}})|\geq 0\,.$ (12)

To obtain a contradiction to the invariance condition (9) from Step 1 under assumption (10), it remains to show that $\psi$ from (12) is *strictly positive* with probability greater than zero (w.r.t. $p$).

First, the strict monotonicity from (11) implies that

$\psi\big{(}\mathbf{c}^{*},(\mathbf{s}_{-l}^{*},s_{l}),(\mathbf{s}_{-l}^{*},\tilde{s}_{l})\big{)}>0\,,\quad\forall(s_{l},\tilde{s}_{l})\in(s_{l}^{*},s_{l}^{*}+\eta)\times(s_{l}^{*}-\eta,s_{l}^{*})\,.$ (13)

Note that in order to obtain the strict inequality in (13), it is important that $s_{l}$ and $\tilde{s}_{l}$ take values in *disjoint* open subsets of the interval $(s_{l}^{*}-\eta,s_{l}^{*}+\eta)$ from (11).

Since $\psi$ is a composition of continuous functions (absolute value of the difference of two continuous functions), $\psi$ is continuous.

Consider the open set $\mathbb{R}_{>0}$, and recall that, under a continuous function, pre-images (or inverse images) of open sets are always *open*.

Applied to the continuous function $\psi$, this pre-image corresponds to an *open* set

$\mathcal{U}\subseteq\mathcal{C}\times\mathcal{S}\times\mathcal{S}$ (14)

in the domain of $\psi$ on which $\psi$ is strictly positive.

Moreover, due to (13):

$\{\mathbf{c}^{*}\}\times\big{(}\{\mathbf{s}_{-l}^{*}\}\times(s_{l}^{*},s_{l}^{*}+\eta)\big{)}\times\big{(}\{\mathbf{s}_{-l}^{*}\}\times(s_{l}^{*}-\eta,s_{l}^{*})\big{)}\subset\mathcal{U},$ (15)

so $\mathcal{U}$ is *non-empty*.

Next, by assumption (iii), there exists at least one subset $A\subseteq\{1,...,n_{s}\}$ of changing style variables such that $l\in A$ and $p_{A}(A)>0$; pick one such subset and call it $A$.

###

---

Then, also by assumption (iii), for any $\mathbf{s}_A \in \mathcal{S}_A$, there is an open subset $\mathcal{O}(\mathbf{s}_A) \subseteq \mathcal{S}_A$ containing $\mathbf{s}_A$, such that $p_{\tilde{\mathbf{s}}_A|\mathbf{s}_A}(\cdot|\mathbf{s}_A) &gt; 0$ within $\mathcal{O}(\mathbf{s}_A)$.

Define the following space

$$
\mathcal {R} _ {A} := \left\{\left(\mathbf {s} _ {A}, \tilde {\mathbf {s}} _ {A}\right): \mathbf {s} _ {A} \in \mathcal {S} _ {A}, \tilde {\mathbf {s}} _ {A} \in \mathcal {O} \left(\mathbf {s} _ {A}\right) \right\} \tag {16}
$$

and, recalling that $A^c = \{1,\dots,n_s\} \setminus A$ denotes the complement of $A$, define

$$
\mathcal {R} := \mathcal {C} \times \mathcal {S} _ {A ^ {c}} \times \mathcal {R} _ {A} \tag {17}
$$

which is a topological subspace of $\mathcal{C} \times \mathcal{S} \times \mathcal{S}$.

By assumptions (ii) and (iii), $p_{\mathbf{z}}$ is smooth and fully supported, and $p_{\tilde{\mathbf{s}}_A|\mathbf{s}_A}(\cdot |\mathbf{s}_A)$ is smooth and fully supported on $\mathcal{O}(\mathbf{s}_A)$ for any $\mathbf{s}_A\in \mathcal{S}_A$. Therefore, the measure $\mu_{(\mathbf{c},\mathbf{s}_{A^c},\mathbf{s}_A,\tilde{\mathbf{s}}_A)|A}$ has fully supported, strictly-positive density on $\mathcal{R}$ w.r.t. a strictly positive measure on $\mathcal{R}$. In other words, $p_{\mathbf{z}}\times p_{\tilde{\mathbf{s}}_A|\mathbf{s}_A}$ is fully supported (i.e., strictly positive) on $\mathcal{R}$.

Now consider the intersection $\mathcal{U} \cap \mathcal{R}$ of the open set $\mathcal{U}$ with the topological subspace $\mathcal{R}$.

Since $\mathcal{U}$ is open, by the definition of topological subspaces, the intersection $\mathcal{U} \cap \mathcal{R} \subseteq \mathcal{R}$ is open in $\mathcal{R}$, (and thus has the same dimension as $\mathcal{R}$ if non-empty).

Moreover, since $\mathcal{O}(\mathbf{s}_A^*)$ is open containing $\mathbf{s}_A^*$, there exists $\eta' &gt; 0$ such that $\{\mathbf{s}_{-l}^*\} \times (s_l^* - \eta', s_l^*) \subset \mathcal{O}(\mathbf{s}_A^*)$. Thus, for $\eta'' = \min(\eta, \eta') &gt; 0$,

$$
\left\{\mathbf {c} ^ {*} \right\} \times \left\{\mathbf {s} _ {A ^ {c}} ^ {*} \right\} \times \left(\left\{\mathbf {s} _ {A \backslash \{l \}} ^ {*} \right\} \times \left(s _ {l} ^ {*}, s _ {l} ^ {*} + \eta\right)\right) \times \left(\left\{\mathbf {s} _ {A \backslash \{l \}} ^ {*} \right\} \times \left(s _ {l} ^ {*} - \eta^ {\prime \prime}, s _ {l} ^ {*}\right)\right) \subset \mathcal {R}. \tag {18}
$$

In particular, this implies that

$$
\left\{\mathbf {c} ^ {*} \right\} \times \left(\left\{\mathbf {s} _ {- l} ^ {*} \right\} \times \left(s _ {l} ^ {*}, s _ {l} ^ {*} + \eta\right)\right) \times \left(\left\{\mathbf {s} _ {- l} ^ {*} \right\} \times \left(s _ {l} ^ {*} - \eta^ {\prime \prime}, s _ {l} ^ {*}\right)\right) \subset \mathcal {R}, \tag {19}
$$

Now, since $\eta'' \leq \eta$, the LHS of (19) is also in $\mathcal{U}$ according to (15), so the intersection $\mathcal{U} \cap \mathcal{R}$ is non-empty.

In summary, the intersection $\mathcal{U} \cap \mathcal{R} \subseteq \mathcal{R}$:

- is non-empty (since both $\mathcal{U}$ and $\mathcal{R}$ contain the LHS of (15));
- is an open subset of the topological subspace $\mathcal{R}$ of $\mathcal{C} \times \mathcal{S} \times \mathcal{S}$ (since it is the intersection of an open set, $\mathcal{U}$, with $\mathcal{R}$);
- satisfies $\psi &gt; 0$ (since this holds for all of $\mathcal{U}$);
- is fully supported w.r.t. the generative process (since this holds for all of $\mathcal{R}$).

As a consequence,

$$
\mathbb {P} (\psi (\mathbf {c}, \mathbf {s}, \tilde {\mathbf {s}}) &gt; 0 | A) \geq \mathbb {P} (\mathcal {U} \cap \mathcal {R}) &gt; 0, \tag {20}
$$

where $\mathbb{P}$ denotes probability w.r.t. the true generative process $p$.

Since $p_A(A) &gt; 0$, this is a contradiction to the invariance (9) from Step 1.

Hence, assumption (10) cannot hold, i.e., $\mathbf{h}_c(\mathbf{c},\mathbf{s})$ does not depend on any style variable $s_l$. It is thus only a function of $\mathbf{c}$, i.e., $\hat{\mathbf{c}} = \mathbf{h}_c(\mathbf{c})$.

Finally, smoothness and invertibility of $\mathbf{h}_c: \mathcal{C} \to \mathcal{C}$ follow from smoothness and invertibility of $\mathbf{h}$, as established in Step 1.

This concludes the proof that $\hat{\mathbf{c}}$ is related to the true content $\mathbf{c}$ via a smooth invertible mapping.

## A.2 Proof of Thm. 4.3

Theorem 4.3 (Identifying content with an invertible encoder). Assume the same data generating process (§ 3) and conditions (i)-(iv) as in Thm. 4.2. Let $\mathbf{g}:\mathcal{X}\to \mathcal{Z}$ be any smooth and invertible function which minimises the following functional:

$$
\mathcal {L} _ {\text {A l i g n}} (\mathbf {g}) := \mathbb {E} _ {(\mathbf {x}, \tilde {\mathbf {x}}) \sim p _ {\mathbf {x}, \tilde {\mathbf {x}}}} \left[ \left| \left| \mathbf {g} (\mathbf {x}) _ {1: n _ {c}} - \mathbf {g} (\tilde {\mathbf {x}}) _ {1: n _ {c}} \right| \right| _ {2} ^ {2} \right] \tag {4}
$$

Then $\mathbf{g}$ block-identifies the true content variables in the sense of Definition 4.1.

---

Proof.

As in the proof of Thm. 4.2, the proof again consists of two main steps.

In the first step, we show that the representation $\hat{\mathbf{z}}=\mathbf{g}(\mathbf{x})$ extracted by any $\mathbf{g}$ that minimises $\mathcal{L}_{\mathrm{Align}}$ is related to the true latent $\mathbf{z}$ through a smooth invertible mapping $\mathbf{h}$, and that $\hat{\mathbf{z}}$ must satisfy invariance across $(\mathbf{x},\tilde{\mathbf{x}})$ in the first $n_{c}$ (content) components almost surely (a.s.) with respect to (w.r.t.) the true generative process.

In the second step, we use the same argument by contradiction as in Step 2 of the proof of Thm. 4.2, to show that $\hat{\mathbf{c}}=\mathbf{h}(\mathbf{z})_{1:n_{c}}$ can only depend on the true content $\mathbf{c}$ and not on style $\mathbf{s}$.

#### Step 1.

From the form of the objective (4), it is clear that $\mathcal{L}_{\mathrm{Align}}\geq 0$ with equality if and only if $\mathbf{g}(\tilde{\mathbf{x}})_{1:n_{c}}=\mathbf{g}(\mathbf{x})_{1:n_{c}}$ for all $(\mathbf{x},\tilde{\mathbf{x}})$ s.t. $p_{\mathbf{x},\tilde{\mathbf{x}}}(\mathbf{x},\tilde{\mathbf{x}})>0$.

Moreover, it follows from the assumed generative process that the global minimum of zero is attained by the true unmixing $\mathbf{f}^{-1}$ since

$\mathbf{f}^{-1}(\mathbf{x})_{1:n_{c}}=\mathbf{c}=\hat{\mathbf{c}}=\mathbf{f}^{-1}(\tilde{\mathbf{x}})_{1:n_{c}}$ (21)

holds a.s. (i.e., with probability one) w.r.t. the true generative process $p$.

Hence, there exists at least one smooth invertible function ($\mathbf{f}^{-1}$) which attains the global minimum.

Let $\mathbf{g}$ be *any* function attaining the global minimum of $\mathcal{L}_{\mathrm{Align}}$ of zero.

As argued above, this implies that (a.s. w.r.t. $p$):

$\mathbf{g}(\tilde{\mathbf{x}})_{1:n_{c}}=\mathbf{g}(\mathbf{x})_{1:n_{c}}.$ (22)

Writing $\mathbf{g}=\mathbf{h}\circ\mathbf{f}^{-1}$, where $\mathbf{h}$ is the smooth, invertible function $\mathbf{h}=\mathbf{g}\circ\mathbf{f}$ we obtain (a.s. w.r.t. $p$):

$\hat{\mathbf{c}}=\mathbf{h}(\tilde{\mathbf{z}})_{1:n_{c}}=\mathbf{h}(\mathbf{z})_{1:n_{c}}.$ (23)

Note that this is the same invariance condition as (9) derived in Step 1 of the proof of Thm. 4.2.

#### Step 2.

It remains to show that $\mathbf{h}(\mathbf{z})_{1:n_{c}}$ can only depend on the true content $\mathbf{c}$ and not on any of the style variables $\mathbf{s}$. To show this, we use the same Step 2 as in the proof of Thm. 4.2. ∎

### A.3 Proof of Thm. 4.4

###### Theorem 4.4 (Identifying content with discriminative learning and a non-invertible encoder).

Assume the same data generating process (§ 3) and conditions (i)-(iv) as in Thm. 4.2. Let $\mathbf{g}:\mathcal{X}\to(0,1)^{n_{c}}$ be any smooth function which minimises the following functional:

$\mathcal{L}_{\mathrm{AlignMaxEnt}}(\mathbf{g}):=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{\mathbf{x},\tilde{\mathbf{x}}}}\left[\left|\left|\mathbf{g}(\mathbf{x})-\mathbf{g}(\tilde{\mathbf{x}})\right|\right|_{2}^{2}\right]-H\left(\mathbf{g}(\mathbf{x})\right)$ (5)

where $H(\cdot)$ denotes the differential entropy of the random variable $\mathbf{g}(\mathbf{x})$ taking values in $(0,1)^{n_{c}}$. Then $\mathbf{g}$ block-identifies the true content variables in the sense of Defn. 4.1.

###### Proof.

The proof consists of three main steps.

In the first step, we show that the representation $\hat{\mathbf{c}}=\mathbf{g}(\mathbf{x})$ extracted by any smooth function $\mathbf{g}$ that minimises (5) is related to the true latent $\mathbf{z}$ through a smooth mapping $\mathbf{h}$; that $\hat{\mathbf{c}}$ must satisfy invariance across $(\mathbf{x},\tilde{\mathbf{x}})$ almost surely (a.s.) with respect to (w.r.t.) the true generative process $p$; and that $\hat{\mathbf{c}}$ must follow a uniform distribution on $(0,1)^{n_{c}}$.

In the second step, we use the same argument by contradiction as in Step 2 of the proof of Thm. 4.2, to show that $\hat{\mathbf{c}}=\mathbf{h}(\mathbf{z})$ can only depend on the true content $\mathbf{c}$ and not on style $\mathbf{s}$.

Finally, in the third step, we show that $\mathbf{h}$ must be a bijection, i.e., invertible, using a result from *[129]*.

#### Step 1.

The global minimum of $\mathcal{L}_{\mathrm{AlignMaxEnt}}$ is reached when the first term (alignment) is minimised (i.e., equal to zero) and the second term (entropy) is maximised.

Without additional moment constraints, the *unique* maximum entropy distribution on $(0,1)^{n_{c}}$ is the uniform distribution *[25, 61]*.

---

First, we show that there exists a smooth function $\mathbf{g}^{*}:\mathcal{X}\to(0,1)^{n_{c}}$ which attains the global minimum of $\mathcal{L}_{\mathrm{AlignMaxEnt}}$.

To see this, consider the function $\mathbf{f}_{1:n_{c}}^{-1}:\mathcal{X}\to\mathcal{C}$, i.e., the inverse of the true mixing $\mathbf{f}$, restricted to its first $n_{c}$ dimensions. This exists and is smooth since $\mathbf{f}$ is smooth and invertible by assumption (i). Further, we have $\mathbf{f}^{-1}(\mathbf{x})_{1:n_{c}}=\mathbf{c}$ by definition.

We now build a function $\mathbf{d}:\mathcal{C}\to(0,1)^{n_{c}}$ which maps $\mathbf{c}$ to a uniform random variable on $(0,1)^{n_{c}}$ using a recursive construction known as the *Darmois construction* *[29, 57]*.

Specifically, we define

$d_{i}(\mathbf{c}):=F_{i}(c_{i}|\mathbf{c}_{1:i-1})=\mathbb{P}(C_{i}\leq c_{i}|\mathbf{c}_{1:i-1}),\hskip 28.45274pti=1,...,n_{c},$ (24)

where $F_{i}$ denotes the conditional cumulative distribution function (CDF) of $c_{i}$ given $\mathbf{c}_{1:i-1}$.

By construction, $\mathbf{d}(\mathbf{c})$ is uniformly distributed on $(0,1)^{n_{c}}$ *[29, 57]*.

Further, $\mathbf{d}$ is smooth by the assumption that $p_{\mathbf{z}}$ (and thus $p_{\mathbf{c}}$) is a smooth density.

Finally, we define

$\mathbf{g}^{*}:=\mathbf{d}\circ\mathbf{f}_{1:n_{c}}^{-1}:\mathcal{X}\to(0,1)^{n_{c}},$ (25)

which is a smooth function since it is a composition of two smooth functions.

###### Claim A.1.

$\mathbf{g}^{*}$ as defined in (25) attains the global minimum of $\mathcal{L}_{\mathrm{AlignMaxEnt}}$.

###### Proof of Claim A.1.

Using $\mathbf{f}^{-1}(\mathbf{x})_{1:n_{c}}=\mathbf{c}$ and $\mathbf{f}^{-1}(\tilde{\mathbf{x}})_{1:n_{c}}=\tilde{\mathbf{c}}$, we have

$\mathcal{L}_{\mathrm{AlignMaxEnt}}(\mathbf{g}^{*})$ $=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{(\mathbf{x},\tilde{\mathbf{x}})}}\left[\left|\left|\mathbf{g}^{*}(\mathbf{x})-\mathbf{g}^{*}(\tilde{\mathbf{x}})\right|\right|^{2}\right]-H\left(\mathbf{g}^{*}(\mathbf{x})\right)$ (26)
$=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{(\mathbf{x},\tilde{\mathbf{x}})}}\left[\left|\left|\mathbf{d}(\mathbf{c})-\mathbf{d}(\tilde{\mathbf{c}})\right|\right|^{2}\right]-H\left(\mathbf{d}(\mathbf{c})\right)$ (27)
$=0$ (28)

where in the last step we have used the fact that $\mathbf{c}=\tilde{\mathbf{c}}$ almost surely w.r.t. to the ground truth generative process $p$ described in § 3, so the first term is zero; and the fact that $\mathbf{d}(\mathbf{c})$ is uniformly distributed on $(0,1)^{n_{c}}$ and the uniform distribution on the unit hypercube has zero entropy, so the second term is also zero.

Next, let $\mathbf{g}:\mathcal{X}\to(0,1)^{n_{c}}$ be *any* smooth function which attains the global minimum of (5), i.e.,

$\mathcal{L}_{\mathrm{AlignMaxEnt}}(\mathbf{g})=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{(\mathbf{x},\tilde{\mathbf{x}})}}\left[\left|\left|\mathbf{g}(\mathbf{x})-\mathbf{g}(\tilde{\mathbf{x}})\right|\right|^{2}\right]-H\left(\mathbf{g}(\mathbf{x})\right)=0.$ (29)

Define $\mathbf{h}:=\mathbf{g}\circ\mathbf{f}:\mathcal{Z}\to(0,1)^{n_{c}}$ which is smooth because both $\mathbf{g}$ and $\mathbf{f}$ are smooth.

Writing $\mathbf{x}=\mathbf{f}(\mathbf{z}),$ (29) then implies in terms of $\mathbf{h}$:

$\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{(\mathbf{x},\tilde{\mathbf{x}})}}\left[\left|\left|\mathbf{h}(\mathbf{z})-\mathbf{h}(\tilde{\mathbf{z}})\right|\right|^{2}\right]$ $=0\,,$ (30)
$H\left(\mathbf{h}(\mathbf{z})\right)$ $=0\,.$ (31)

Equation (30) implies that the same invariance condition (9) used in the proofs of Thms. 4.2 and 4.3 must hold (a.s. w.r.t. $p$), and (31) implies that $\tilde{\mathbf{c}}=\mathbf{h}(\mathbf{z})$ must be uniformly distributed on $(0,1)^{n_{c}}$.

#### Step 2.

Next, we show that $\mathbf{h}(\mathbf{z})=\mathbf{h}(\mathbf{c},\mathbf{s})$ can only depend on the true content $\mathbf{c}$ and not on any of the style variables $\mathbf{s}$. For this we use the same Step 2 as in the proofs of Thms. 4.2 and 4.3.

---

Step 3.

Finally, we show that the mapping $\hat{\mathbf{c}}=\mathbf{h}(\mathbf{c})$ is invertible.

To this end, we make use of the following result from *[129]*.

###### Proposition A.2 (Proposition 5 of *[129]*).

Let $\mathcal{M},\mathcal{N}$ be simply connected and oriented $\mathcal{C}^{1}$ manifolds without boundaries and $h:\mathcal{M}\to\mathcal{N}$ be a differentiable map. Further, let the random variable $\mathbf{z}\in\mathcal{M}$ be distributed according to $\mathbf{z}\sim p(\mathbf{z})$ for a regular density function $p$, i.e., $0<p<\infty$. If the pushforward $p_{\#h}(\mathbf{z})$ of $p$ through $h$ is also a regular density, i.e., $0<p_{\#h}<\infty$, then $h$ is a bijection.

We apply this result to the simply connected and oriented $\mathcal{C}^{1}$ manifolds without boundaries $\mathcal{M}=\mathcal{C}$ and $\mathcal{N}=(0,1)^{n_{c}}$, and the smooth (hence, differentiable) map $\mathbf{h}:\mathcal{C}\to(0,1)^{n_{c}}$ which maps the random variable $\mathbf{c}$ to a uniform random variable $\hat{\mathbf{c}}$ (as established in Step 1).

Since both $p_{\mathbf{c}}$ (by assumption) and the uniform distribution (the pushforward of $p_{\mathbf{c}}$ through $\mathbf{h}$) are regular densities in the sense of Prop. A.2, we conclude that $\mathbf{h}$ is a bijection, i.e., invertible.

We have shown that for any smooth $\mathbf{g}:\mathcal{X}\to(0,1)^{n_{c}}$ which minimises $\mathcal{L}_{\mathrm{AlignMaxEnt}}$, we have that $\hat{\mathbf{c}}=\mathbf{g}(\mathbf{x})=\mathbf{h}(\mathbf{c})$ for a smooth and invertible $\mathbf{h}:\mathcal{C}\to(0,1)^{n_{c}}$, i.e., $\mathbf{c}$ is block-identified by $\mathbf{g}$.

## Appendix B Additional details on the Causal3DIdent data set

Using the Blender rendering engine *[11]*, 3DIdent *[129]* is a recently proposed benchmark which contains hallmarks of natural environments (e.g. shadows, different lighting conditions, a 3D object), but allows for identifiability evaluation by exposing the underlying generative factors.

Each $224\times 224\times 3$ image in the dataset shows a coloured 3D object which is located and rotated above a coloured ground in a 3D space. Furthermore, each scene contains a coloured spotlight which is focused on the object and located on a half-circle around the scene. The images are rendered based on a $10$-dimensional latent, where: (i) three dimensions describe the XYZ position of the object, (ii) three dimensions describe the rotation of the object in Euler angles, (iii) two dimensions describe the colour (hue) of the object and the ground of the scene, respectively, and (iv) two dimensions describe the position and colour (hue) of the spotlight. For influence of the latent factors on the renderings, see Fig. 2 of *[129]*.

### B.1 Details on introduced object classes

3DIdent contained a single object class, Teapot *[89]*. We add six additional object classes: Hare *[121]*, Dragon *[110]*, Cow *[62]*, Armadillo *[70]*, Horse *[98]*, Head *[111]*.

### B.2 Details on latent causal graph

In 3DIdent, the latents are uniformly sampled independently. We instead impose a causal graph over the variables (see Fig. 2). While object class and all environment variables (spotlight position, spotlight hue, background hue) are sampled independently, all object variables are dependent. Specifically, for spotlight position, spotlight hue, and background hue, we sample from $U(-1,1)$. We impose the dependence by varying the mean ($\mu$) of a truncated normal distribution with standard deviation $\sigma=0.5$, truncated to the range $[-1,1]$.

Object rotation is dependent solely on object class, see Tab. 2 for details. Object position is dependent on both object class & spotlight position, see Tab. 3. Object hue is dependent on object class, background hue, & object hue, see Tab. 4. Hares blending into their environment as a form of active camouflage has been observed in Alaskan *[78]*, Arctic *[2]*, & Snowshoe hares.

### B.3 Dataset Visuals

We show $40$ random samples from the marginal of each object class in Causal3DIdent in Figs. 3 to 9.

---

Table 2: Given a certain object class, the center of the truncated normal distribution from which we sample rotation latents varies.

|  object class | μ(φ) | μ(θ) | μ(ψ)  |
| --- | --- | --- | --- |
|  Teapot | -0.35 | 0.35 | 0.35  |
|  Hare | 0.35 | -0.35 | 0.35  |
|  Dragon | 0.35 | 0.35 | -0.35  |
|  Cow | 0.35 | -0.35 | -0.35  |
|  Armadillo | -0.35 | 0.35 | -0.35  |
|  Horse | -0.35 | -0.35 | 0.35  |
|  Head | -0.35 | -0.35 | -0.35  |

Table 3: Given a certain object class &amp; spotlight position, the center of the truncated normal distribution from which we sample  $xy$ -position latents varies. Note the spotlight position  $\mathrm{pos}_{\mathrm{spl}}$  is rescaled from  $[-1,1]$  to  $[-\pi /2,\pi /2]$ .

|  object class | μ(x) | μ(y) | μ(z)  |
| --- | --- | --- | --- |
|  Teapot | 0 | 0 | 0  |
|  Hare | - sin(posspl) | - cos(posspl) | 0  |
|  Dragon | - sin(posspl) | - cos(posspl) | 0  |
|  Cow | sin(posspl) | cos(posspl) | 0  |
|  Armadillo | sin(posspl) | cos(posspl) | 0  |
|  Horse | - sin(posspl) | - cos(posspl) | 0  |
|  Head | sin(posspl) | cos(posspl) | 0  |

Table 4: Given a certain object class, background hue, and spotlight hue, the center of the truncated normal distribution from which we sample the object hue latent varies. Note that for the Hare and Dragon classes, in particular, the object either blends in or stands out from the environment.

|  object class | μ(hue)  |
| --- | --- |
|  Teapot | 0  |
|  Hare | huehe+huespl/2  |
|  Dragon | -huehe+huespl/2  |
|  Cow | -0.35  |
|  Armadillo | 0.7  |
|  Horse | -0.7  |
|  Head | 0.35  |

![[SSL_provably_isolates_p23_img5.jpeg]]
Figure 3: 40 random samples from the marginal distribution of the Teapot object class.

---

![[SSL_provably_isolates_p24_img6.jpeg]]
Figure 4: 40 random samples from the marginal distribution of the Hare object class.

![[SSL_provably_isolates_p24_img7.jpeg]]
Figure 5: 40 random samples from the marginal distribution of the Dragon object class.

![[SSL_provably_isolates_p24_img8.jpeg]]
Figure 6: 40 random samples from the marginal distribution of the Cow object class.

---

![[SSL_provably_isolates_p25_img9.jpeg]]
Figure 7: 40 random samples from the marginal distribution of the Armadillo object class.

![[SSL_provably_isolates_p25_img10.jpeg]]
Figure 8: 40 random samples from the marginal distribution of the Horse object class.

![[SSL_provably_isolates_p25_img11.jpeg]]
Figure 9: 40 random samples from the marginal distribution of the Head object class.

---

# C Additional results

- Appendix C.1 contains numerical experiments, namely linear evaluation &amp; an ablation on  $\dim(\hat{\mathbf{c}})$ .
- Appendix C.2 contains experiments on Causal3DIdent, namely (i) nonlinear &amp; linear evaluation results of the output &amp; intermediate feature representation of SimCLR with results for the individual axes of object position &amp; rotation, and (ii) evaluation of BarlowTwins.
- Appendix C.3 contains experiments on the MPI3D-real dataset [38], namely SimCLR &amp; a supervised sanity check.

# C.1 Numerical Data

In Tab. 5, we report mean  $\pm$  std. dev.  $R^2$  over 3 random seeds across four generative processes of increasing complexity using linear (instead of nonlinear) regression to predict  $\mathbf{c}$  from  $\hat{\mathbf{c}}$ . The block-identification of content can clearly still be seen even if we consider a linear fit.

In Fig. 10, we perform an ablation on  $\dim(\hat{\mathbf{c}})$ , visualising how varying the dimensionality of the learnt representation affects identifiability of the ground-truth content &amp; style partition. Generally, if  $\dim(\hat{\mathbf{c}}) &lt; n_c$ , there is insufficient capacity to encode all content, so a lower-dimensional mixture of content is learnt. Conversely, if  $\dim(\hat{\mathbf{c}}) &gt; n_c$ , the excess capacity is used to encode some style information, as that increases entropy.

Table 5: Results using linear regression for the experiment on numerical data presented in § 5.1

|  Generative process |   |   | R2 (linear)  |   |
| --- | --- | --- | --- | --- |
|  p(chg.) | Stat. | Cau. | Content c | Style s  |
|  1.0 | X | X | 1.00 ± 0.00 | 0.00 ± 0.00  |
|  0.75 | X | X | 0.99 ± 0.00 | 0.00 ± 0.00  |
|  0.75 | ✓ | X | 0.97 ± 0.03 | 0.37 ± 0.05  |
|  0.75 | ✓ | ✓ | 0.98 ± 0.01 | 0.78 ± 0.07  |

![[SSL_provably_isolates_p26_img12.jpeg]]
Figure 10: Identifiability of the content &amp; style partition in the numerical experiment as a function of the model latent dimensionality

![[SSL_provably_isolates_p26_img13.jpeg]]

![[SSL_provably_isolates_p26_img14.jpeg]]

![[SSL_provably_isolates_p26_img15.jpeg]]

On Dependence. As can be seen from Tab. 5, the corresponding inset table in § 5.1, and Fig. 10, scores for identifying style increase substantially when statistical dependence within blocks and causal dependence between blocks are included. This finding can be explained as follows.

If we compare the performance for small latent dimensionalities  $(\dim(\hat{\mathbf{c}}) &lt; n_c)$  between the first two (without) and the third plot (with statistical dependence) of Fig. 10, we observe a significantly higher score in identifying content for the latter (e.g.,  $R^2$  of ca. 0.4 vs 0.2 at  $\dim(\hat{\mathbf{c}}) = 1$ ). This suggests that the introduction of statistical dependence between content variables (as well as between style variables, and in how style variables change) in the third plot/row, reduces the effective dimensionality of the ground-truth latents and thus leads to higher content identifiability for the same  $\dim(\hat{\mathbf{c}}) &lt; n_c$ . Since the  $R^2$  for content is already close to 1 for  $\dim(\hat{\mathbf{c}}) = 3$  in the third plot of Fig. 10 (due to the smaller effective dimensionality induced by statistical dependence between  $\mathbf{c}$ ), when  $\dim(\hat{\mathbf{c}}) = n_c = 5$  is used (as reported in Tab. 5), excess capacity is used to encode style, leading to a positive  $R^2$ .

Regarding causal dependence (i.e., the fourth plot in Fig. 10 and fourth row in Tab. 5), we note that the ground truth dependence between  $\mathbf{c}$  and  $\mathbf{s}$  is linear, i.e.,  $p(\mathbf{s}|\mathbf{c})$  is centred at a linear transformation  $\mathbf{a} + B\mathbf{c}$  of  $\mathbf{c}$ , see the data generating process in Appendix D for details. Given that our evaluation

---

consists of predicting the ground truth  $\mathbf{c}$  and  $\mathbf{s}$  from the learnt representation  $\hat{\mathbf{c}} = \mathbf{g}(\mathbf{x})$ , if we were to block-identify  $\mathbf{c}$  according to Defn. 4.1, we should be able to also predict some aspects of  $\mathbf{s}$  from  $\hat{\mathbf{c}}$ , due to the linear dependence between  $\mathbf{c}$  and  $\mathbf{s}$ . This manifests in a relatively large  $R^2$  for  $\mathbf{s}$  in the last row of Tab. 5 and the corresponding table in § 5.1.

To summarise, we highlight two main takeaways: (i) when latent dependence is present, this may reduce the effective dimensionality, so that some style is encoded in addition to content unless a smaller representation size is chosen; (ii) even though the learnt representation isolates content in the sense of Defn. 4.1, it may still be predictive of style when content and style are (causally) dependent.

# C.2 Causal3DIdent

Full version of Tab. 1: In Tab. 6, we a) provide the results for the individual axes of object position &amp; rotation and b) present additional rows omitted from Tab. 1 for space considerations.

Interestingly, we find that the variance across the individual axes is significantly higher for object position than object rotation. If we compare the causal dependence imposed for object position (see Tab. 3) to the causal dependence imposed for object rotation (see Tab. 2), we can observe that the dependence imposed over individual axes is also significantly more variable for position than rotation, i.e., for  $x$  the sine nonlinearity is used, for  $y$  the cosine nonlinearity is used, while for  $z$ , no dependence is imposed.

Regarding the additional rows, we can observe that the composition of image-level rotation &amp; crops yields results quite similar to solely using crops, a relationship which mirrors how transforming the rotation &amp; position latents yields results quite similar to solely transforming the position latents. This suggests that the rotation variables are difficult to disentangle from the position variables in Causal3DIdent, regardless of whether data augmentation or latent transforms are used.

Finally, we can observe that applying image-level rotation in conjunction with small crops &amp; colour distortion does lead to a difference in the encoding, background hue is preserved, while the scores for object position &amp; rotation appear to slightly decrease. When using three augmentations as opposed to two, the effects of the individual augmentations are lessened. While colour distortion discourages the encoding of background hue, both small crops &amp; image-level rotation encourages it, and thus it is preserved when all three augmentations are used. While colour distortion encourages the encoding of object position &amp; rotation, both small crops &amp; image-level rotation discourage it, but as a causal relationship exists between the class variable and said latents, the scores merely decrease, the latents are still for the most part preserved. In reality, where complex interactions between latent variables abound, the effect of data augmentations may be uninterpretable, however with Causal3DIdent, we are able to interpret their effects in the presence of rich visual complexity and causal dependencies, even when applying three distinct augmentations in tandem.

Table 6: Full version of Tab. 1.

|  Views generated by | Class | Positions |   |   |   | Hues |   |   |   | Rotations  |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  object(x) | object(y) | object(z) | spotlight | object | spotlight | background | object(x) | object(y) | object(z) |   |
|  DA: colour distortion | 0.42 ± 0.01 | 0.58 ± 0.01 | 0.75 ± 0.00 | 0.52 ± 0.01 | 0.17 ± 0.00 | 0.10 ± 0.01 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.36 ± 0.01 | 0.33 ± 0.01 | 0.32 ± 0.00 |   |
|  LT: change hues | 1.00 ± 0.00 | 0.81 ± 0.02 | 0.81 ± 0.02 | 0.15 ± 0.02 | 0.91 ± 0.00 | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.30 ± 0.02 | 0.30 ± 0.01 | 0.30 ± 0.01 |   |
|  DA: crop (large) | 0.29 ± 0.04 | 0.01 ± 0.02 | 0.03 ± 0.01 | 0.19 ± 0.02 | 0.21 ± 0.13 | 0.07 ± 0.00 | 0.09 ± 0.02 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.02 ± 0.00 |   |
|  DA: crop (small) | 0.14 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |   |
|  LT: change positions | 1.00 ± 0.00 | 0.65 ± 0.00 | 0.67 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.01 | 0.46 ± 0.02 | 0.00 ± 0.00 | 0.97 ± 0.00 | 0.30 ± 0.00 | 0.29 ± 0.00 | 0.29 ± 0.00 |   |
|  DA: crop (large) + colour distortion | 0.97 ± 0.00 | 0.59 ± 0.03 | 0.52 ± 0.01 | 0.68 ± 0.01 | 0.59 ± 0.05 | 0.28 ± 0.00 | 0.01 ± 0.01 | 0.01 ± 0.00 | 0.74 ± 0.01 | 0.78 ± 0.00 | 0.72 ± 0.00 |   |
|  DA: crop (small) + colour distortion | 1.00 ± 0.00 | 0.82 ± 0.02 | 0.65 ± 0.02 | 0.70 ± 0.00 | 0.93 ± 0.00 | 0.30 ± 0.01 | 0.00 ± 0.00 | 0.02 ± 0.03 | 0.53 ± 0.00 | 0.57 ± 0.01 | 0.58 ± 0.01 |   |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.10 ± 0.10 | 0.49 ± 0.02 | 0.00 ± 0.05 | 0.07 ± 0.09 | 0.32 ± 0.02 | 0.00 ± 0.01 | 0.02 ± 0.03 | 0.34 ± 0.09 | 0.34 ± 0.04 | 0.34 ± 0.09 |   |
|  DA: rotation | 0.33 ± 0.00 | 0.29 ± 0.03 | 0.11 ± 0.01 | 0.12 ± 0.04 | 0.33 ± 0.12 | 0.03 ± 0.01 | 0.30 ± 0.12 | 0.99 ± 0.00 | 0.02 ± 0.01 | 0.06 ± 0.03 | 0.07 ± 0.01 |   |
|  LT: change rotations | 1.00 ± 0.00 | 0.78 ± 0.01 | 0.72 ± 0.03 | 0.09 ± 0.03 | 0.90 ± 0.00 | 0.41 ± 0.00 | 0.00 ± 0.00 | 0.97 ± 0.00 | 0.28 ± 0.00 | 0.28 ± 0.00 | 0.28 ± 0.00 |   |
|  DA: rotation + colour distortion | 0.59 ± 0.01 | 0.63 ± 0.01 | 0.57 ± 0.08 | 0.54 ± 0.02 | 0.21 ± 0.01 | 0.12 ± 0.02 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.36 ± 0.03 | 0.34 ± 0.04 | 0.30 ± 0.03 |   |
|  LT: change rotations + hues | 1.00 ± 0.00 | 0.80 ± 0.02 | 0.77 ± 0.01 | 0.13 ± 0.02 | 0.91 ± 0.00 | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.28 ± 0.00 | 0.28 ± 0.01 | 0.28 ± 0.00 |   |
|  DA: rot. + crop (lg) | 0.26 ± 0.01 | 0.03 ± 0.02 | 0.03 ± 0.01 | 0.15 ± 0.04 | 0.04 ± 0.03 | 0.04 ± 0.06 | 0.10 ± 0.01 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.02 | 0.02 ± 0.00 |   |
|  DA: rot. + crop (sin) | 0.15 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |   |
|  LT: change rot. + pos. | 1.00 ± 0.00 | 0.02 ± 0.03 | 0.48 ± 0.02 | 0.01 ± 0.01 | 0.02 ± 0.03 | 0.49 ± 0.03 | 0.03 ± 0.02 | 0.98 ± 0.00 | 0.29 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 |   |
|  DA: rot. + crop (lg) + col. dist. | 0.99 ± 0.00 | 0.69 ± 0.03 | 0.60 ± 0.01 | 0.70 ± 0.02 | 0.86 ± 0.03 | 0.28 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.68 ± 0.01 | 0.64 ± 0.02 | 0.61 ± 0.01 |   |
|  DA: rot. + crop (sin) + col. dist. | 1.00 ± 0.00 | 0.61 ± 0.02 | 0.59 ± 0.01 | 0.64 ± 0.01 | 0.82 ± 0.01 | 0.38 ± 0.00 | 0.01 ± 0.01 | 0.78 ± 0.03 | 0.44 ± 0.00 | 0.48 ± 0.02 | 0.45 ± 0.01 |   |
|  LT: change vol. + pos. + hues | 1.00 ± 0.00 | 0.30 ± 0.12 | 0.50 ± 0.04 | 0.14 ± 0.11 | 0.15 ± 0.12 | 0.32 ± 0.01 | 0.00 ± 0.00 | 0.02 ± 0.01 | 0.33 ± 0.04 | 0.33 ± 0.02 | 0.32 ± 0.03 |   |

Linear identifiability: In Tab. 7, we present results evaluating all continuous variables with linear regression. While, as expected,  $R^2$  scores are reduced across the board, we can observe that even with a linear fit, the patterns observed in Tab. 6 persist.

Intermediate feature evaluation: In Tab. 8 and Tab. 9, we present evaluation based on the representation from an intermediate layer (i.e., prior to applying a projection layer [20]) with nonlinear and linear regression for the continuous variables, respectively. Note the intermediate layer has an

---

Table 7: Evaluation results using a linear fit for not only class, but all continuous variables.

|  Views generated by | Class | Positions |   |   |   | Hues |   |   |   | Rotation  |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  abjec(φ) | abjec(μ) | abjec(γ) | speflight | abjec | speflight | background | abjec(φ) | abjec(θ) | abjec(γ)  |
|  DA: colour distortion | 0.42 ± 0.01 | 0.37 ± 0.03 | 0.20 ± 0.10 | 0.22 ± 0.02 | 0.01 ± 0.01 | 0.03 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.13 ± 0.01 | 0.04 ± 0.01 | 0.09 ± 0.02  |
|  LT: change hues | 1.00 ± 0.00 | 0.72 ± 0.07 | 0.56 ± 0.04 | -0.00 ± 0.00 | 0.65 ± 0.07 | 0.29 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.27 ± 0.01 | 0.26 ± 0.03 | 0.26 ± 0.01  |
|  DA: crop (large) | 0.28 ± 0.04 | 0.00 ± 0.00 | 0.02 ± 0.00 | 0.04 ± 0.07 | 0.00 ± 0.13 | 0.51 ± 0.05 | 0.03 ± 0.02 | 0.26 ± 0.04 | 0.03 ± 0.00 | 0.02 ± 0.00 | 0.01 ± 0.00  |
|  DA: crop (small) | 0.14 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.17 ± 0.05 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00  |
|  LT: change positions | 1.00 ± 0.00 | -0.00 ± 0.00 | 0.44 ± 0.02 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.29 ± 0.04 | 0.00 ± 0.00 | 0.72 ± 0.16 | 0.26 ± 0.01 | 0.25 ± 0.04 | 0.25 ± 0.04  |
|  DA: crop (large) + colour distortion | 0.97 ± 0.00 | 0.12 ± 0.02 | 0.24 ± 0.03 | 0.21 ± 0.00 | 0.08 ± 0.01 | 0.13 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.14 ± 0.04 | 0.18 ± 0.05 | 0.22 ± 0.02  |
|  DA: crop (small) + colour distortion | 0.39 ± 0.00 | 0.35 ± 0.02 | 0.50 ± 0.01 | 0.33 ± 0.01 | 0.00 ± 0.01 | 0.28 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.29 ± 0.00 | 0.26 ± 0.00 | 0.29 ± 0.01  |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.42 ± 0.06 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.27 ± 0.02 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.23 ± 0.07 | 0.26 ± 0.03 | 0.23 ± 0.04  |
|  DA: rotation | 0.33 ± 0.06 | 0.04 ± 0.04 | 0.04 ± 0.00 | 0.02 ± 0.01 | 0.12 ± 0.08 | 0.46 ± 0.06 | 0.06 ± 0.04 | 0.30 ± 0.13 | 0.03 ± 0.00 | 0.03 ± 0.02 | 0.02 ± 0.00  |
|  LT: change rotations | 1.00 ± 0.00 | 0.34 ± 0.21 | 0.48 ± 0.01 | -0.00 ± 0.00 | 0.60 ± 0.15 | 0.28 ± 0.00 | 0.00 ± 0.00 | 0.59 ± 0.26 | 0.27 ± 0.01 | 0.27 ± 0.00 | 0.27 ± 0.01  |
|  DA: rotation + colour distortion | 0.59 ± 0.01 | 0.31 ± 0.02 | 0.26 ± 0.06 | 0.25 ± 0.07 | 0.02 ± 0.00 | 0.03 ± 0.02 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.07 ± 0.01 | 0.06 ± 0.01 | 0.10 ± 0.01  |
|  LT: change rotations + hues | 1.00 ± 0.00 | 0.68 ± 0.02 | 0.57 ± 0.01 | -0.00 ± 0.00 | 0.72 ± 0.10 | 0.29 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.28 ± 0.00 | 0.26 ± 0.00 | 0.28 ± 0.00  |
|  DA: rel. + crop (lg) | 0.26 ± 0.01 | -0.00 ± 0.00 | 0.02 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.58 ± 0.05 | 0.02 ± 0.01 | 0.26 ± 0.04 | 0.03 ± 0.00 | 0.01 ± 0.00 | 0.01 ± 0.00  |
|  DA: rel. + crop (sm) | 0.15 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.29 ± 0.21 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00  |
|  LT: change rel. + pos. | 1.00 ± 0.00 | -0.00 ± 0.00 | 0.45 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.32 ± 0.02 | 0.00 ± 0.00 | 0.60 ± 0.09 | 0.27 ± 0.00 | 0.27 ± 0.01 | 0.27 ± 0.01  |
|  DA: rel. + crop (lg) + col. dist. | 0.99 ± 0.00 | 0.33 ± 0.02 | 0.26 ± 0.07 | 0.26 ± 0.01 | 0.51 ± 0.14 | 0.21 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.21 ± 0.04 | 0.28 ± 0.02 | 0.22 ± 0.02  |
|  DA: rel. + crop (sm) + col. dist. | 1.00 ± 0.00 | 0.26 ± 0.02 | 0.48 ± 0.01 | 0.21 ± 0.02 | 0.61 ± 0.05 | 0.31 ± 0.00 | -0.00 ± 0.00 | 0.34 ± 0.02 | 0.26 ± 0.00 | 0.30 ± 0.01 | 0.29 ± 0.01  |
|  LT: change rel. + pos. + hues | 1.00 ± 0.00 | 0.03 ± 0.05 | 0.46 ± 0.01 | 0.01 ± 0.01 | 0.01 ± 0.02 | 0.29 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.27 ± 0.00 | 0.28 ± 0.01 | 0.28 ± 0.01  |

output dimensionality of 100. While it is clear that all  $R^2$  scores are increased across the board, we can notice that certain latents which were discarded in the final layer, were not in an intermediate layer. For example, with "LT: change hues", in the final layer the  $z$ -position was discarded ( $R^2 = 0.15$  in Tab. 6), inexplicably we may add, as position is content regardless of axis with this latent transformation. But in the intermediate layer,  $z$ -position was not discarded ( $R^2 = 0.88$  in Tab. 8).

Table 8: Evaluation of an intermediate layer. Logistic regression used for class, kernel ridge regression used for all continuous variables.

|  Views generated by | Class | Positions |   |   |   | Hues |   |   |   | Rotation  |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  abjec(φ) | abjec(μ) | abjec(γ) | speflight | abjec | speflight | background | abjec(φ) | abjec(θ) | abjec(γ)  |
|  DA: colour distortion | 0.71 ± 0.02 | 0.68 ± 0.02 | 0.80 ± 0.01 | 0.63 ± 0.01 | 0.25 ± 0.01 | 0.13 ± 0.00 | 0.02 ± 0.01 | 0.01 ± 0.01 | 0.44 ± 0.01 | 0.48 ± 0.01 | 0.29 ± 0.00  |
|  LT: change hues | 1.00 ± 0.00 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.99 ± 0.01 | 0.98 ± 0.00 | 0.34 ± 0.01 | -0.00 ± 0.00 | 0.20 ± 0.10 | 0.71 ± 0.02 | 0.68 ± 0.04 | 0.68 ± 0.02  |
|  DA: crop (large) | 0.43 ± 0.03 | 0.41 ± 0.05 | 0.35 ± 0.05 | 0.32 ± 0.04 | 0.41 ± 0.13 | 0.88 ± 0.00 | 0.14 ± 0.01 | 1.00 ± 0.00 | 0.03 ± 0.02 | 0.06 ± 0.01 | 0.08 ± 0.00  |
|  DA: crop (small) | 0.20 ± 0.01 | 0.04 ± 0.05 | 0.20 ± 0.02 | 0.01 ± 0.02 | 0.20 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 1.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00  |
|  LT: change positions | 1.00 ± 0.00 | 0.78 ± 0.02 | 0.90 ± 0.01 | 0.75 ± 0.01 | 0.58 ± 0.02 | 0.82 ± 0.01 | 0.18 ± 0.02 | 0.99 ± 0.00 | 0.64 ± 0.02 | 0.53 ± 0.02 | 0.56 ± 0.02  |
|  DA: crop (large) + colour distortion | 1.00 ± 0.00 | 0.82 ± 0.00 | 0.83 ± 0.00 | 0.92 ± 0.00 | 0.90 ± 0.01 | 0.29 ± 0.00 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.87 ± 0.00 | 0.90 ± 0.00 | 0.85 ± 0.00  |
|  DA: crop (small) + colour distortion | 1.00 ± 0.00 | 0.92 ± 0.00 | 0.87 ± 0.01 | 0.90 ± 0.00 | 0.97 ± 0.00 | 0.46 ± 0.04 | 0.02 ± 0.02 | 0.58 ± 0.12 | 0.79 ± 0.01 | 0.83 ± 0.00 | 0.79 ± 0.00  |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.83 ± 0.04 | 0.90 ± 0.01 | 0.81 ± 0.04 | 0.75 ± 0.08 | 0.42 ± 0.00 | 0.04 ± 0.02 | 0.52 ± 0.26 | 0.72 ± 0.05 | 0.69 ± 0.07 | 0.67 ± 0.06  |
|  DA: rotation | 0.46 ± 0.04 | 0.35 ± 0.04 | 0.19 ± 0.02 | 0.28 ± 0.04 | 0.34 ± 0.08 | 0.85 ± 0.01 | 0.35 ± 0.12 | 1.00 ± 0.00 | 0.03 ± 0.01 | 0.08 ± 0.02 | 0.10 ± 0.01  |
|  LT: change rotations | 1.00 ± 0.00 | 0.97 ± 0.00 | 0.96 ± 0.01 | 0.84 ± 0.01 | 0.98 ± 0.00 | 0.82 ± 0.01 | 0.17 ± 0.02 | 0.99 ± 0.00 | 0.64 ± 0.02 | 0.59 ± 0.01 | 0.60 ± 0.01  |
|  DA: rotation + colour distortion | 0.87 ± 0.02 | 0.76 ± 0.01 | 0.81 ± 0.01 | 0.71 ± 0.01 | 0.39 ± 0.08 | 0.19 ± 0.02 | -0.00 ± 0.00 | 0.02 ± 0.02 | 0.55 ± 0.03 | 0.55 ± 0.03 | 0.48 ± 0.02  |
|  LT: change rotations + hues | 1.00 ± 0.00 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.87 ± 0.00 | 0.99 ± 0.00 | 0.39 ± 0.05 | 0.04 ± 0.02 | 0.37 ± 0.21 | 0.69 ± 0.01 | 0.68 ± 0.01 | 0.68 ± 0.00  |
|  DA: rel. + crop (lg) | 0.43 ± 0.03 | 0.38 ± 0.04 | 0.34 ± 0.02 | 0.28 ± 0.03 | 0.30 ± 0.05 | 0.86 ± 0.04 | 0.17 ± 0.02 | 1.00 ± 0.00 | 0.02 ± 0.00 | 0.05 ± 0.01 | 0.10 ± 0.01  |
|  DA: rel. + crop (sm) | 0.20 ± 0.01 | 0.07 ± 0.03 | 0.09 ± 0.10 | 0.01 ± 0.01 | 0.20 ± 0.01 | -0.00 ± 0.00 | -0.00 ± 0.00 | 1.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00  |
|  LT: change rel. + pos. | 1.00 ± 0.00 | 0.81 ± 0.01 | 0.90 ± 0.01 | 0.76 ± 0.01 | 0.67 ± 0.04 | 0.84 ± 0.01 | 0.28 ± 0.04 | 0.99 ± 0.00 | 0.62 ± 0.02 | 0.57 ± 0.01 | 0.55 ± 0.01  |
|  DA: rel. + crop (lg) + col. dist. | 1.00 ± 0.00 | 0.92 ± 0.01 | 0.89 ± 0.00 | 0.92 ± 0.00 | 0.93 ± 0.01 | 0.30 ± 0.00 | 0.02 ± 0.02 | 0.18 ± 0.16 | 0.81 ± 0.00 | 0.84 ± 0.00 | 0.79 ± 0.00  |
|  DA: rel. + crop (sm) + col. dist. | 1.00 ± 0.00 | 0.87 ± 0.00 | 0.85 ± 0.00 | 0.87 ± 0.00 | 0.93 ± 0.00 | 0.71 ± 0.02 | 0.33 ± 0.05 | 0.96 ± 0.00 | 0.72 ± 0.00 | 0.75 ± 0.00 | 0.71 ± 0.00  |
|  LT: change rel. + pos. + hues | 1.00 ± 0.00 | 0.84 ± 0.02 | 0.91 ± 0.01 | 0.82 ± 0.02 | 0.78 ± 0.06 | 0.40 ± 0.01 | 0.06 ± 0.01 | 0.50 ± 0.05 | 0.72 ± 0.04 | 0.70 ± 0.05 | 0.67 ± 0.04  |

Table 9: Evaluation of an intermediate layer. Logistic regression used for class, linear regression used for all continuous variables.

|  Views generated by | Class | Positions |   |   |   | Hues |   |   |   | Rotation  |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  abjec(φ) | abjec(μ) | abjec(γ) | speflight | abjec | speflight | background | abjec(φ) | abjec(θ) | abjec(γ)  |
|  DA: colour distortion | 0.71 ± 0.02 | 0.53 ± 0.01 | 0.70 ± 0.01 | 0.46 ± 0.01 | 0.13 ± 0.01 | 0.11 ± 0.01 | -0.01 ± 0.00 | 0.00 ± 0.00 | 0.28 ± 0.01 | 0.19 ± 0.01 | 0.25 ± 0.01  |
|  LT: change hues | 1.00 ± 0.00 | 0.93 ± 0.00 | 0.93 ± 0.00 | 0.60 ± 0.04 | 0.95 ± 0.00 | 0.31 ± 0.00 | 0.01 ± 0.01 | 0.06 ± 0.04 | 0.44 ± 0.02 | 0.41 ± 0.02 | 0.42 ± 0.00  |
|  DA: crop (large) | 0.43 ± 0.03 | 0.18 ± 0.00 | 0.06 ± 0.01 | 0.17 ± 0.02 | 0.19 ± 0.14 | 0.82 ± 0.02 | 0.09 ± 0.04 | 0.98 ± 0.00 | 0.01 ± 0.00 | 0.05 ± 0.01 | 0.05 ± 0.01  |
|  DA: crop (small) | 0.20 ± 0.01 | 0.01 ± 0.01 | 0.03 ± 0.02 | 0.08 ± 0.01 | 0.02 ± 0.01 | -0.00 ± 0.00 | -0.01 ± 0.00 | 0.99 ± 0.06 | -0.01 ± 0.01 | -0.01 ± 0.00 | -0.00 ± 0.01  |
|  LT: change positions | 1.00 ± 0.00 | 0.49 ± 0.04 | 0.72 ± 0.05 | 0.43 ± 0.03 | 0.19 ± 0.03 | 0.71 ± 0.02 | 0.09 ± 0.02 | 0.98 ± 0.00 | 0.39 ± 0.01 | 0.36 ± 0.01 | 0.35 ± 0.00  |
|  DA: crop (large) + colour distortion | 1.00 ± 0.00 | 0.87 ± 0.01 | 0.56 ± 0.01 | 0.66 ± 0.02 | 0.67 ± 0.02 | 0.28 ± 0.00 | -0.01 ± 0.00 | 0.01 ± 0.01 | 0.58 ± 0.02 | 0.61 ± 0.02 | 0.56 ± 0.01  |
|  DA: crop (small) + colour distortion | 1.00 ± 0.00 | 0.76 ± 0.01 | 0.70 ± 0.02 | 0.66 ± 0.01 | 0.90 ± 0.00 | 0.30 ± 0.01 | 0.00 ± 0.01 | 0.39 ± 0.12 | 0.50 ± 0.02 | 0.50 ± 0.01 | 0.49 ± 0.01  |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.61 ± 0.09 | 0.74 ± 0.02 | 0.51 ± 0.08 | 0.40 ± 0.15 | 0.34 ± 0.04 | 0.02 ± 0.01 | 0.25 ± 0.22 | 0.47 ± 0.04 | 0.40 ± 0.02 | 0.41 ± 0.03  |
|  DA: rotation | 0.46 ± 0.04 | 0.21 ± 0.02 | 0.10 ± 0.01 | 0.10 ± 0.02 | 0.21 ± 0.09 | 0.77 ± 0.01 | 0.25 ± 0.11 | 0.97 ± 0.01 | 0.02 ± 0.01 | 0.06 ± 0.02 | 0.08 ± 0.01  |
|  LT: change rotations | 1.00 ± 0.00 | 0.92 ± 0.00 | 0.88 ± 0.01 | 0.51 ± 0.02 | 0.95 ± 0.00 | 0.70 ± 0.06 | 0.07 ± 0.02 | 0.98 ± 0.00 | 0.36 ± 0.01 | 0.34 ± 0.00 | 0.34 ± 0.01  |
|  DA: rotation + colour distortion | 0.87 ± 0.02 | 0.60 ± 0.01 | 0.62 ± 0.03 | 0.52 ± 0.02 | 0.23 ± 0.02 | 0.18 ± 0.02 | -0.01 ± 0.00 | 0.02 ± 0.01 | 0.33 ± 0.04 | 0.29 ± 0.01 | 0.28 ± 0.01  |
|  LT: change rotations + hues | 1.00 ± 0.00 | 0.94 ± 0.00 | 0.92 ± 0.01 | 0.58 ± 0.01 | 0.96 ± 0.00 | 0.33 ± 0.02 | 0.02 ± 0.01 | 0.15 ± 0.10 | 0.40 ± 0.02 | 0.38 ± 0.01 | 0.41 ± 0.03  |
|  DA: rel. + crop (lg) | 0.43 ± 0.03 | 0.24 ± 0.01 | 0.08 ± 0.02 | 0.16 ± 0.03 | 0.07 ± 0.01 | 0.80 ± 0.04 | 0.10 ± 0.01 | 0.98 ± 0.00 | 0.01 ± 0.00 | -0.05 ± 0.01 | 0.06 ± 0.01  |
|  DA: rel. + crop (sm) | 0.20 ± 0.01 | 0.01 ± 0.01 | 0.03 ± 0.01 | -0.00 ± 0.01 | 0.04 ± 0.01 | -0.01 ± 0.00 | -0.01 ± 0.00 | 0.99 ± 0.00 | -0.01 ± 0.00 | -0.03 ± 0.00 | -0.00 ± 0.01  |
|  LT: change rel. + pos. | 1.00 ± 0.00 | 0.55 ± 0.05 | 0.72 ± 0.02 | 0.44 ± 0.04 | 0.31 ± 0.08 | 0.76 ± 0.01 | 0.14 ± 0.01 | 0.99 ± 0.00 | 0.28 ± 0.01 | 0.35 ± 0.01 | 0.36 ± 0.02  |
|  DA: rel. + crop (cm) + col. dist. | 1.00 ± 0.00 | 0.71 ± 0.01 | 0.69 ± 0.01 | 0.69 ± 0.00 | 0.84 ± 0.03 | 0.28 ± 0.00 | -0.00 ± 0.00 | 0.07 ± 0.07 | 0.51 ± 0.01 | 0.50 ± 0.02 | 0.51 ± 0.01  |
|  DA: rel. + crop (cm) + col. dist. | 1.00 ± 0.00 | 0.66 ± 0.00 | 0.68 ± 0.02 | 0.65 ± 0.02 | 0.83 ± 0.00 | 0.17 ± 0.03 | 0.18 ± 0.02 | 0.89 ± 0.01 | 0.46 ± 0.01 | 0.45 ± 0.02 | 0.45 ± 0.01  |
|  LT: change rel. + pos. + hues | 1.00 ± 0.00 | 0.65 ± 0.04 | 0.75 ± 0.05 | 0.57 ± 0.03 | 0.49 ± 0.12 | 0.35 ± 0.01 | 0.02 ± 0.01 | 0.23 ± 0.04 | 0.48 ± 0.04 | 0.43 ± 0.01 | 0.43 ± 0.01  |

In [20], the value in evaluating an intermediate layer as opposed to a final layer is discussed, where the authors demonstrated that predicting the data augmentations applied during training is significantly more accurate from an intermediate layer as opposed to the final layer, implying that the intermediate layer contains much more information about the transformation applied. Our results suggest a distinct hypothesis, the value in using an intermediate layer as a representation for downstream tasks is not due to preservation of style information, as can be seen,  $R^2$  scores on style variables are not significantly higher in Tab. 8 relative to Tab. 6. The value is in preservation of all content variables, as we can observe certain content variables are discarded in the final layer, but are preserved in an

---

Table 10: BarlowTwins  $\lambda = 0.0051$  results:  $R^2$  mean  $\pm$  std. dev. over 3 random seeds. DA: data augmentation, LT: latent transformation, bold:  $R^2 \geq 0.5$ , red:  $R^2 &lt; 0.25$ . Results for individual axes of object position &amp; rotation are aggregated.

|  Views generated by | Class | Positions |   | Hues |   |   | Rotations  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  object | spotlight | object | spotlight | background  |   |
|  DA: colour distortion | 0.48 ± 0.02 | 0.51 ± 0.14 | 0.07 ± 0.01 | 0.08 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.04  |
|  LT: change hues | 1.00 ± 0.00 | 0.56 ± 0.20 | 0.76 ± 0.07 | 0.30 ± 0.01 | 0.00 ± 0.00 | 0.01 ± 0.00 | 0.35 ± 0.01  |
|  DA: crop (large) | 0.17 ± 0.02 | 0.10 ± 0.03 | 0.06 ± 0.02 | 0.29 ± 0.13 | 0.11 ± 0.05 | 0.99 ± 0.00 | 0.02 ± 0.01  |
|  DA: crop (small) | 0.15 ± 0.00 | 0.04 ± 0.02 | 0.05 ± 0.02 | 0.02 ± 0.01 | 0.00 ± 0.01 | 1.00 ± 0.00 | 0.00 ± 0.01  |
|  LT: change positions | 0.88 ± 0.00 | 0.19 ± 0.20 | 0.05 ± 0.00 | 0.50 ± 0.02 | 0.04 ± 0.01 | 0.98 ± 0.00 | 0.27 ± 0.03  |
|  DA: crop (large) + colour distortion | 0.87 ± 0.02 | 0.49 ± 0.06 | 0.32 ± 0.03 | 0.25 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.02  |
|  DA: crop (small) + colour distortion | 0.81 ± 0.01 | 0.39 ± 0.07 | 0.42 ± 0.06 | 0.47 ± 0.04 | 0.03 ± 0.01 | 0.85 ± 0.02 | 0.30 ± 0.02  |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.28 ± 0.20 | 0.12 ± 0.05 | 0.31 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.37 ± 0.06  |

Table 11: BarlowTwins  $\lambda = 0.051$  results:  $R^2$  mean  $\pm$  std. dev. over 3 random seeds. DA: data augmentation, LT: latent transformation, bold:  $R^2 \geq 0.5$ , red:  $R^2 &lt; 0.25$ . Results for individual axes of object position &amp; rotation are aggregated.

|  Views generated by | Class | Positions |   | Hues |   |   | Rotations  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  object | spotlight | object | spotlight | background  |   |
|  DA: colour distortion | 0.52 ± 0.07 | 0.43 ± 0.18 | 0.07 ± 0.02 | 0.10 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.21 ± 0.05  |
|  LT: change hues | 1.00 ± 0.00 | 0.55 ± 0.24 | 0.74 ± 0.02 | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.33 ± 0.02  |
|  DA: crop (large) | 0.19 ± 0.05 | 0.08 ± 0.02 | 0.05 ± 0.01 | 0.39 ± 0.36 | 0.08 ± 0.05 | 0.96 ± 0.05 | 0.01 ± 0.02  |
|  DA: crop (small) | 0.15 ± 0.00 | 0.05 ± 0.02 | 0.07 ± 0.02 | 0.00 ± 0.01 | 0.01 ± 0.01 | 1.00 ± 0.00 | 0.00 ± 0.00  |
|  LT: change positions | 0.89 ± 0.01 | 0.19 ± 0.20 | 0.05 ± 0.01 | 0.48 ± 0.04 | 0.05 ± 0.02 | 0.98 ± 0.00 | 0.25 ± 0.03  |
|  DA: crop (large) + colour distortion | 0.86 ± 0.03 | 0.40 ± 0.07 | 0.23 ± 0.02 | 0.24 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.47 ± 0.04  |
|  DA: crop (small) + colour distortion | 0.99 ± 0.01 | 0.63 ± 0.03 | 0.88 ± 0.01 | 0.32 ± 0.02 | 0.00 ± 0.00 | 0.16 ± 0.13 | 0.52 ± 0.03  |
|  LT: change positions + hues | 1.00 ± 0.00 | 0.21 ± 0.22 | 0.07 ± 0.01 | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.01 | 0.46 ± 0.06  |

intermediate layer. With that being said, our theoretical result applies to the final layer, which is why said results were highlighted in the main paper. The discarding of certain content variables is an empirical phenomenon, likely a consequence of a limited number of negative samples in practice, leading to certain content variables being redundant, or unnecessary, for solving the contrastive objective.

The fact that we can recover certain content variables which appeared discarded in the output from the intermediate layer may suggest that we should be able to decode class. While scores are certainly increased, we do not see such drastic differences in  $R^2$  scores, as was seen above. The drastic difference highlighted above was with regards to latent transformation, for which we always observed class encoded as a content variable. So, unfortunately, using an intermediate layer does not rectify the discrepancy between data augmentations and latent transformations. While latent transformations allow us to better interpret the effect of certain empirical techniques [20], as discussed in the main paper, we cannot make a one-to-one correspondence between data augmentations used in practice and latent transformations.

BarlowTwins: We repeat our analysis from § 5.2 using BarlowTwins [128] (instead of SimCLR) which, as discussed at the end of § 4.2, is also loosely related to Thm. 4.4. The BarlowTwins objective consists of an invariance term, which equates the diagonal elements of the cross-correlation matrix to 1, thereby making the embedding invariant to the distortions applied and a redundancy reduction term, which equates the off-diagonal elements of the cross-correlation matrix to 0, thereby decorrelating the different vector components of the embedding, reducing the redundancy between output units.

In Tab. 10 we train BarlowTwins with  $\lambda = 0.0051$ , the default value for the hyperparameter which weights the redundancy reduction term relative to the invariance term. To confirm the insights are robust to the value of  $\lambda$ , in Tab. 11, we report results with  $\lambda$  increased by an order of magnitude,  $\lambda = 0.051$ . We find that the results mirror Tab. 1, e.g. colour distortion yields a discarding of hue, crops isolate background hue where the larger the crop, the higher the identifiability of object hue, and crops &amp; colour distortion yield high accuracy in inferring the object class variable.

# C.3 MPI3D-real

We ran the same experimental setup as in § 5.2 also on the MPI3D-real dataset [38] containing  $&gt;1$  million real images with ground-truth annotations of 3D objects being moved by a robotic arm.

---

Table 12: MPI3D-real results:  $R^2$  mean ± std. dev. over 3 random seeds for  $\dim(\hat{\mathbf{c}}) = 5$ . DA: data augmentation, bold:  $R^2 \geq 0.5$ , red:  $R^2 &lt; 0.25$ .

|  Views generated by | object color | object shape | object size | camera height | background color | horizontal axis | vertical axis  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  DA: colour distortion | 0.39 ± 0.01 | 0.00 ± 0.00 | 0.16 ± 0.01 | 1.00 ± 0.00 | 0.09 ± 0.15 | 0.60 ± 0.06 | 0.42 ± 0.08  |
|  DA: crop (large) | 0.65 ± 0.17 | 0.01 ± 0.02 | 0.31 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.37 ± 0.06 | 0.08 ± 0.03  |
|  DA: crop (small) | 0.00 ± 0.02 | 0.03 ± 0.00 | 0.19 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.21 ± 0.02 | 0.07 ± 0.00  |
|  DA: crop (large) + colour distortion | 0.34 ± 0.00 | 0.00 ± 0.00 | 0.22 ± 0.03 | 1.00 ± 0.00 | 0.39 ± 0.02 | 0.54 ± 0.01 | 0.29 ± 0.01  |
|  DA: crop (small) + colour distortion | 0.25 ± 0.02 | 0.00 ± 0.00 | 0.10 ± 0.01 | 1.00 ± 0.00 | 0.75 ± 0.16 | 0.54 ± 0.01 | 0.29 ± 0.03  |

Table 13: Supervised MPI3D-real results:  $R^2$  mean ± std. dev. over 3 random seeds. DA: data augmentation. bold:  $R^2 \geq 0.5$ , red:  $R^2 &lt; 0.25$ .

|  Views generated by | object color | object shape | object size | camera height | background color | horizontal axis | vertical axis  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  Original | 0.90 ± 0.01 | 0.25 ± 0.02 | 0.61 ± 0.02 | 0.99 ± 0.00 | 0.97 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00  |
|  DA: colour distortion | 0.61 ± 0.01 | 0.11 ± 0.00 | 0.47 ± 0.01 | 0.98 ± 0.00 | 0.93 ± 0.00 | 0.99 ± 0.00 | 1.00 ± 0.00  |
|  DA: crop (large) | 0.82 ± 0.01 | 0.05 ± 0.01 | 0.42 ± 0.02 | 0.97 ± 0.01 | 0.91 ± 0.00 | 0.96 ± 0.00 | 0.97 ± 0.01  |
|  DA: crop (small) | 0.71 ± 0.04 | 0.01 ± 0.00 | 0.32 ± 0.02 | 0.95 ± 0.00 | 0.85 ± 0.01 | 0.79 ± 0.02 | 0.90 ± 0.01  |
|  DA: crop (large) + colour distortion | 0.45 ± 0.02 | 0.02 ± 0.00 | 0.22 ± 0.00 | 0.95 ± 0.01 | 0.67 ± 0.01 | 0.91 ± 0.00 | 0.94 ± 0.00  |
|  DA: crop (small) + colour distortion | 0.45 ± 0.02 | 0.00 ± 0.00 | 0.17 ± 0.02 | 0.91 ± 0.02 | 0.55 ± 0.03 | 0.69 ± 0.01 | 0.79 ± 0.08  |

As MPI3D-real contains much lower resolution images  $(64 \times 64)$  compared to ImageNet &amp; Causal3DIdent  $(224 \times 224)$ , we used the standard convolutional encoder from the disentangle-ment literature [82], and ran a sanity check experiment to verify that by training the same backbone as in our unsupervised experiment with supervised learning, we can recover the ground-truth factors from the augmented views. In Tab. 13, we observe that only five out of seven factors can be consistently inferred, object shape and size are somewhat ambiguous even when observing the original image. Note that while in the self-supervised case, we evaluate by training a nonlinear regression for each ground truth factor separately, in the supervised case, we train a network for all ground truth factors simultaneously from scratch for as many gradient steps as used for learning the self-supervised model.

In Tab. 12, we report the evaluation results in the self-supervised scenario. Subject to the aforementioned caveats, the results show a similar trend as those on Causal3DIdent, i.e. with colour distortion, color factors of variation are decoded significantly worse than positional/rotational information.

# D Experimental details

Ground-truth generative model. The generative process used in our numerical simulations (§ 5.1) is summarised by the following:

$\mathbf{c}\sim p(\mathbf{c}) = \mathcal{N}(0,\Sigma_{\mathbf{c}}),\quad \mathrm{with}\quad \Sigma_{\mathbf{c}}\sim \mathrm{Wishart}_{n_c}(\mathbf{I},n_c),$

$\mathbf{s}|\mathbf{c}\sim p(\mathbf{s}|\mathbf{c}) = \mathcal{N}(\mathbf{a} + B\mathbf{c},\Sigma_{\mathbf{s}}),\quad \mathrm{with}\quad \Sigma_{\mathbf{s}}\sim \mathrm{Wishart}_{n_s}(\mathbf{I},n_s),\quad a_i,b_{ij}\stackrel {\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,1),$

$\tilde{\mathbf{s}}_A|\mathbf{s}_A,A\sim p(\tilde{\mathbf{s}}_A|\mathbf{s}_A) = N(\mathbf{s}_A,\Sigma (A))$  with  $\Sigma \sim \mathrm{Wishart}_{n_s}(\mathbf{I},n_s)$

$(\tilde{\mathbf{x}},\mathbf{x}) = (\mathbf{f}_{\mathrm{MLP}}(\tilde{\mathbf{z}}),\mathbf{f}_{\mathrm{MLP}}(\mathbf{z}))$

where the set of changing style vectors  $A$  is obtained by flipping a (biased) coin with  $\mathrm{p(chg.)} = 0.75$  for each style dimension independently, and where  $\Sigma(A)$  denotes the submatrix of  $\Sigma$  defined by selecting the rows and columns corresponding to subset  $A$ .

When we do not allow for statistical dependence (Stat.) within blocks of content and style variables, we set the covariance matrices  $\Sigma_{\mathbf{c}}$ ,  $\Sigma_{\mathbf{s}}$ , and  $\Sigma$  to the identity. When we do not allow for causal dependence (Cau.) of style on content, we set  $a_i, b_{ij} = 0, \forall i, j$ .

For  $\mathbf{f}_{\mathrm{MLP}}$ , we use a 3-layer MLP with LeakyReLU ( $\alpha = 0.2$ ) activation functions, specified using the same process as used in previous work [54, 55, 129]. For the square weight matrices, we draw  $(n_c + n_s) \times (n_c + n_s)$  samples from  $U(-1,1)$ , and perform  $l_2$  column normalisation. In addition, to control for invertibility, we re-sample the weight matrices until their condition number is less than or equal to a threshold value. The threshold is pre-computed by sampling 24,975 weight matrices, and recording the minimum condition number.

---

#### Training encoder.

Recall that the result of Thm. 4.4 corresponds to minimizing the following functional (5):

$\mathcal{L}_{\mathrm{AlignMaxEnt}}(\mathbf{g}):=\mathbb{E}_{(\mathbf{x},\tilde{\mathbf{x}})\sim p_{\mathbf{x},\tilde{\mathbf{x}}}}\big{[}\big{(}\mathbf{g}(\mathbf{x})-\mathbf{g}(\tilde{\mathbf{x}})\big{)}^{2}\big{]}-H\left(\mathbf{g}(\mathbf{x})\right).$

Note that InfoNCE *[20, 91]* (1) can be rewritten as:

$\small\mathcal{L}_{\text{InfoNCE}}(\mathbf{g};\tau,K)=\mathbb{E}_{\{\mathbf{x}_{i},\tilde{\mathbf{x}}_{i}\}_{i=1}^{K}\sim p_{\mathbf{x},\tilde{\mathbf{x}}}}\big{[}-\textstyle\sum_{i=1}^{K}\text{sim}(\mathbf{g}(\mathbf{x})_{i},\mathbf{g}(\tilde{\mathbf{x}})_{i})/\tau+\log\sum_{j=1}^{K}\exp\{\text{sim}(\mathbf{g}(\mathbf{x})_{i},\mathbf{g}(\tilde{\mathbf{x}})_{j})/\tau\}\big{]}.$ (32)

Thus, if we consider $\tau=1$, and $\text{sim}(u,v)=-(u-v)^{2}$,

$\small\mathcal{L}_{\text{InfoNCE}}(\mathbf{g};K)=\mathbb{E}_{\{\mathbf{x}_{i},\tilde{\mathbf{x}}_{i}\}_{i=1}^{K}\sim p_{\mathbf{x},\tilde{\mathbf{x}}}}\big{[}\sum_{i=1}^{K}\big{(}\mathbf{g}(\mathbf{x})_{i}-\mathbf{g}(\tilde{\mathbf{x}})_{i}\big{)}^{2}+\log\sum_{j=1}^{K}\exp\{-(\mathbf{g}(\mathbf{x})_{i}-\mathbf{g}(\tilde{\mathbf{x}})_{j})^{2}\}\big{]}$ (33)

we can approximately match the form of (5). In practice, we use $K=6,144$.

For $\mathbf{g}$, as in *[129]*, we use a $7$-layer MLP with (default) LeakyReLU ($\alpha=0.01$) activation functions. As the input dimensionality is $(n_{c}+n_{s})$, we consider the following multipliers $[10,50,50,50,50,50,10]$ for the number of hidden units per layer. In correspondence with Thm. 4.4, we set the output dimensionality to $n_{c}$.

We train our feature encoder for $300,000$ iterations, using Adam *[66]* with a learning rate of $10^{-4}$.

#### Causal3DIdent.

We here elaborate on details specific to the experiments in § 5.2. We train the feature encoder for $200,000$ iterations using Adam with a learning rate of $10^{-4}$. For the encoder we use a ResNet18 *[46]* architecture followed by a single hidden layer with dimensionality $100$ and LeakyReLU activation function using the default ($0.01$) negative slope. The scores are evaluated on a test set consisting of $25,000$ samples not included in the training set.

#### Data augmentations.

We here specify the parameters for the data augmentations we considered:

- colour distortion: see the paragraph labelled “Color distortion” in Appendix A of *[20]* for details. We use $s=1.0$, the default value.
- crop: see the paragraph labelled “Random crop and resize to $224\times 224$” in Appendix A of *[20]* for details. For small crops, a crop of random size (uniform from $0.08$ to $1.0$ in area) of the original size is made, which corresponds to what was used in the experiments reported in *[20]*. For large crops, a crop of random size (uniform from $0.8$ to $1.0$ in area) of the original size is made.
- rotation: as specified in the captions for Figure $4$ & Table $3$ in *[20]*, we sample one of $\{0^{\circ},90^{\circ},180^{\circ},270^{\circ}\}$ uniformly. Note that for the pair, we sample two values without replacement.

A visual overview of the effect of these image-level data augmentations is shown in Fig. 11.

#### Latent transformations.

To generate views via latent transformations (LT) in our experiments on Causal3DIdent (§ 5.2), we proceed as follows.

Let $\mathbf{z}$ refer to the latent corresponding to the original image. For all latents specified to change, we sample $\hat{\mathbf{z}}^{\prime}$ from a truncated normal distribution constrained to $[-1,1]$, centered at $\mathbf{z}$, with $\sigma=1..$ Then, we use nearest-neighbor matching to find the latent $\hat{\mathbf{z}}$ closest to $\hat{\mathbf{z}}^{\prime}$ (in $L^{2}$ distance) for which there exists an image rendering.

#### Evaluation.

Recall that Thm. 4.4 states that $\mathbf{g}$ block-identifies the true content variables in the sense of Defn. 4.1, i.e., there exists an *invertible* function $\mathbf{h}:\mathbb{R}^{n_{c}}\rightarrow\mathbb{R}^{n_{c}}$ s.t. $\hat{\mathbf{c}}=\mathbf{h}(\mathbf{c})$.

Since this is different from typical evaluation in disentanglement or ICA in that we do not assume independence and do not aim to find a one-to-one correspondence between inferred and ground truth latents, existing metrics, such as MCC *[54, 55]* or MIG *[18]*, do not apply.

We therefore treat identifying $\mathbf{h}$ as a regression task, which we solve using kernel ridge regression with a Gaussian kernel *[88]*. Since the Gaussian kernel is universal, this constitutes a nonparametric

---

![[SSL_provably_isolates_p32_img16.jpeg]]
Figure 11: Visual overview of the effect of different data augmentations (DA), applied to 10 representative samples. Rows correspond to (top to bottom): original images, small random crop (+ random flip), large random crop (+ random flip), colour distortion (jitter &amp; drop), and random rotation.

regression technique with universal approximation capabilities, i.e., any nonlinear function can be approximated arbitrarily well given sufficient data.

We sample  $4096 \times 10$  datapoints from the marginal for evaluation. For kernel ridge regression, we standardize the inputs and targets, and fit the regression model on  $4096 \times 5$  (distinct) datapoints. We tune the regularization strength  $\alpha$  and kernel variance  $\gamma$  by 3-fold cross-validated grid search over the following parameter grids:  $\alpha \in [1,0.1,0.001,0.0001]$ ,  $\gamma \in [0.01,0.22,4.64,100]$ .

Compute. The experiments in § 5.1 took on the order of 5-10 hours on a single GeForce RTX 2080 Ti GPU. The experiments in § 5.2 on 3DIdent took 28 hours on four GeForce RTX 2080 Ti GPUs. The creation of the Causal3DIdent dataset additionally required approximately 150 hours of compute time on a GeForce RTX 2080 Ti.