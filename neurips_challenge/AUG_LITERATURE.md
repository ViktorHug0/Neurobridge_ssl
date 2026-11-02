# AUG_LITERATURE: training-time augmentation and regularisation for cross-subject EEG->image retrieval (2026-10-26)

Literature only; no experiments were run for this note. It is a companion to HISTORY.md. The gains quoted are the
ones the authors report on their own tasks (image DG benchmarks, sleep staging, motor imagery, fMRI). None of those is
32-channel single-trial EEG -> 1536-d DINOv2 InfoNCE scored over ~80 averaged repeats, so expect the gains to shrink.

**Filter used for the ranking (from TTA_RESEARCH.md, C4).** The scorer sums ~80 repeats per (person, image) before
ranking. Zero-mean per-trial noise averages away, and what survives is the person-level bias (the person's offset and
mapping distortion). So a method ranks higher when it simulates or penalises *systematic* between-person differences
(spatial mixing, spectral profile, per-subject loss landscapes), and lower when it only adds per-trial noise.
HISTORY.md says the person fingerprint sits in the background alpha/beta topography, which favours the spectral entries.

**Excluded (already dead here, see HISTORY.md):** mixup, raw SubjectMix, SCORE multi-positive, MixStyle, latency and
montage jitter, channel dropout, gain, stochastic depth, wd, EMA, repetition averaging, blurred / soft / multi-layer
targets, MMD / CORAL, FiLM + subject dropout, gradient reversal, EA / CAR, extra EEG1 / Alljoined-1 data, global
negatives, loss truncation, TTA. Being tested now: noise transplant, covariance operator transfer, SubjectMix + THINGS.

braindecode in `.venv` already ships `FTSurrogate`, `FrequencyShift`, `SmoothTimeMask`, `BandstopFilter`
(`braindecode.augmentation`); `torch.optim.swa_utils` has `AveragedModel` / `update_bn`.

## Ranked candidates

### 1. Cross-subject response conversion (hyperalignment / neural code conversion) as augmentation
**Mechanism.** Fit, per ordered pair of training subjects (s, t), a linear map W_st that takes s's image-evoked
responses to t's responses to the *same* images (ridge or orthogonal Procrustes, the hyperalignment/SRM family).
At train time, re-render a trial of s as W_st x and keep its image label, which creates "virtual subjects" whose
spatial mixing comes from real people. The map is fitted on stimulus-locked correspondences, so it follows the
label-carrying signal rather than the background covariance.
**Sources.** Haxby et al., *Neuron* 2011 (hyperalignment); Chen et al., NeurIPS 2015 (shared response model);
Ho, Horikawa, Majima, Cheng, Kamitani, *NeuroImage* 271:120007, 2023 (pairwise "neural code converters" trained on
identical image sets; DNN features stay decodable after conversion and inter-individual reconstruction works);
Wang et al., *Nat. Comput. Sci.* 2025 (converters without shared stimuli, including across sites); Thual et al.,
arXiv 2312.06467 (2023): functional alignment improves out-of-subject decoding "up to 75%" over anatomical alignment;
Ravishankar, Toneva, Wehbe, *Front. Comput. Neurosci.* 2021 (MEG pairwise subject-to-subject mapping, used for
denoising). MindAligner (arXiv 2502.05034, 2025) learns a "brain transfer matrix" to known subjects for fMRI.
**Effect sizes.** All of these use alignment as a test-time mapping, not as augmentation; I found no EEG/MEG paper that
uses cross-subject trial conversion as a training augmentation.
**Why it could transfer.** It differs from the covariance-operator transfer now being tested: R_t^1/2 R_s^-1/2 is
fitted on second-order statistics, which the alpha/beta background dominates. Here W_st is fitted on repeat-averaged
evoked responses to shared images. Alljoined shares images across subjects (SubjectMix already samples a "same-image
trial of another subject"), as do THINGS-EEG2 training images.
**Sketch.** Offline, per dataset: E_s[i] = mean over repeats (C x T, 32 x 120) for each shared training image i.
Concatenate over i and t, then ridge-fit W_st (32x32) with lambda chosen on held-out images, shrunk toward I. Optionally
use a 3-5 tap spatio-temporal FIR (C x C x L) to absorb latency. That gives 20x19 + 10x9 small matrices. Per batch, with
p = 0.3-0.5: pick t != s from the same dataset, set W = (1-a) I + a W_st with a ~ U(0, 1), x' = W x, then rescale to
t's single-trial RMS. Fit only on training-fold subjects.

### 2. Sharpness-aware minimisation (SAM; SAGM for DG)
**Mechanism.** Minimise the worst-case loss in an L2 ball of radius rho around the weights (two forward/backward passes
per step). SAGM also minimises the gap between the perturbed and the clean loss, which aligns their gradients.
**Sources.** Foret et al., ICLR 2021 (arXiv 2010.01412); Chen, Hsieh, Gong, ICLR 2022 (arXiv 2106.01548); Wang et al.,
CVPR 2023 (SAGM, arXiv 2303.10353).
**Effect sizes.** On ViT-B/16 trained from scratch on ImageNet, SAM gives +5.3 top-1 (79.9), and ImageNet-C robustness
improves (Chen et al.). On DomainBed (5 datasets, ResNet-50), the SAGM paper's Table 1 reports ERM 63.9, SAM 64.5,
GSAM 65.1 and SAGM 66.1, with SAGM ahead of ERM on all five datasets.
**Why it could transfer.** REVE-scratch and token-REVE are transformers trained from scratch on limited data, which is
the regime where Chen et al. see the largest SAM gains. Flat minima are the standard explanation for robustness to
distribution shift, and an unseen person is such a shift. The method changes the optimiser only.
**Sketch.** Wrap AdamW (or `module/stable_adamw.py`). rho in {0.02, 0.05, 0.1}; SAGM alpha in {5e-4, 1e-3}. For BN
encoders (TSC-bn64, SQF-dbn), update running stats only in the first pass. Training costs about 2x compute, so screen
on cv4 fold 1 at matched epochs.

### 3. SWAD (dense, overfit-aware weight averaging)
**Mechanism.** Average the weights uniformly at *every* iteration inside a window [t_s, t_e]. t_s is where the
validation loss stops improving for N_s = 3 evaluations, and t_e is where it exceeds r = 1.2-1.3 times its value at t_s
for N_e = 6 evaluations.
**Source.** Cha et al., NeurIPS 2021 (arXiv 2102.08604).
**Effect sizes.** DomainBed average: ERM 63.3 vs SWAD 66.9 (+3.6), and CORAL+SWAD 67.3. On PACS, EMA gives only +0.2
(85.3 -> 85.5) and SWA +0.6, while SWAD gives +1.8 (87.1). SAM+SWAD is only +1.0 over SAM, so the two partly overlap.
**Why it could transfer.** EMA was neutral here, but the paper itself shows that EMA and SWAD differ by about 1.6pp on
PACS. Dense uniform averaging inside a validation-chosen window is a different estimator from an exponential tail.
**Sketch.** `AveragedModel` updated each step once t_s fires, then `update_bn` for BN encoders. Take the window from
the training-subject validation split, never from the sealed people; that also keeps away from the sealed_best bias.
The T50 "checkpoint soup" in HISTORY.md was never scored (the run hit a bug), so this is untested here.

### 4. Fishr (match per-subject gradient variances)
**Mechanism.** Treat each subject as a domain. Penalise the spread across domains of the per-domain variance of
per-sample loss gradients with respect to the last layer. This aligns the per-subject loss curvature (Fisher/Hessian)
rather than the feature statistics.
**Source.** Rame, Dancette, Cord, ICML 2022 (arXiv 2109.02934).
**Effect sizes.** DomainBed with training-domain model selection (paper Table 9): ERM 66.6 vs Fishr 67.1 average over 7
datasets, with OfficeHome +1.3 and TerraInc +1.3. Test-domain selection: 68.7 -> 70.8, mostly from Colored MNIST.
Overhead is 3-7% in time.
**Why it could transfer.** MMD and CORAL (feature-moment matching) failed here. Fishr is the gradient-side
alternative, and on DomainBed it beats CORAL on TerraIncognita, the dataset whose shift (camera location) looks most
like person identity.
**Sketch.** Sample batches as K subjects x M trials (e.g. 8 x 32). Use only the EEG->image InfoNCE direction, so that
loss term i depends only on y_i. Let g_i = dl_i/dy_i (autograd of the summed loss w.r.t. the output) and h_i = the
input of the final linear layer. Diagonal per-subject variance: v_s = mean_i (g_i^2)(h_i^2)^T - (mean_i g_i h_i^T)^2,
smoothed with EMA gamma = 0.95. Penalty: lambda * sum_s ||v_s - mean_s v_s||^2, with lambda in 10^[1, 4] (default 1e3),
switched on after a 20-30% ERM warmup.

### 5. Fourier amplitude mix across subjects (FACT-style), band-limited
**Mechanism.** Keep a trial's Fourier phase and interpolate its amplitude spectrum toward that of another subject's
trial: A' = (1 - l) A + l A_p with l ~ U(0, eta). This changes each channel's spectral profile while keeping the timing.
**Source.** Xu et al., CVPR 2021 oral (arXiv 2105.11120).
**Effect sizes.** PACS ResNet-18: 79.94 -> 84.51 for full FACT, and 83.44 (+3.5) for amplitude mix alone, without the
co-teacher. ResNet-50: 84.08 -> 88.15. eta = 1.0 on PACS and 0.2 on OfficeHome.
**Why it could transfer.** It transfers the per-channel alpha/beta power profile, which is where HISTORY.md locates the
person fingerprint, without needing repeat means. Noise transplant instead swaps the whole residual time series.
**Sketch.** X = rfft(x, dim=-1) per channel (1 Hz bins at 120 Hz / 1 s). The partner is a random trial of a different
subject in the same dataset. Mix only inside a band mask (start with 7-30 Hz; widen or narrow according to the pending
`band_probe.py` result on where the image signal lives). eta in {0.3, 0.5, 1.0}, p = 0.5. Skip the co-teacher at
first, since EMA-style teachers were neutral here.

### 6. FrequencyShift (individual alpha frequency variability)
**Mechanism.** Shift the whole PSD by Df through the analytic signal: Re(hilbert(x) * exp(2i pi Df t)).
**Source.** Rommel et al., *J. Neural Eng.* 2022 (arXiv 2206.14483); the augmentation is from Rommel et al. (CADDA,
arXiv 2106.13695).
**Effect sizes.** It was the runner-up to FTSurrogate. Small shifts are best for cross-subject sleep staging
(Df_max 0.3 Hz, explicitly motivated as mimicking between-subject peak-frequency variability) and 2.7 Hz for BCI IV-2a.
Bomatter & Gouk (arXiv 2503.13497, 2025) found it helped only LaBraM on PhysioNet, with no consistent gains otherwise.
**Why it could transfer.** Individual alpha peak frequency is a trait-like difference between people (Klimesch,
*Brain Res. Rev.* 1999), and none of the failed augmentations touched frequency.
**Sketch.** `braindecode.augmentation.FrequencyShift`, Df ~ U(-D, D) with D in {0.5, 1, 2} Hz, p = 0.5. Prefer applying
it to the residual (x - leave-one-out repeat mean) so that the evoked waveform is not shifted.

### 7. V-REx / Group DRO on per-subject InfoNCE
**Mechanism.** V-REx adds beta * Var_s(L_s) over the per-subject mean losses in the batch. Group DRO instead upweights
the worst subject online.
**Sources.** Krueger et al., ICML 2021 (arXiv 2003.00688); Sagawa et al., ICLR 2020 (arXiv 1911.08731).
**Effect sizes.** In the DomainBed table of the SAGM paper, both are below ERM: VREx 61.9, GroupDRO 60.7, ERM 63.9.
**Why it might still help.** It costs one line, and the per-subject batch layout from #4 makes it free. HISTORY.md and
memory record that dropping weak subjects did not help, which argues against expecting much.
**Sketch.** beta in {1, 10}, ramped up after warmup. Run it only as a rider on the #4 run.

### 8. FTSurrogate on the residual (phase-randomised noise)
**Mechanism.** Randomise the Fourier phases while keeping the amplitudes. Applied to the residual r = x - mu_img, this
yields new noise realisations with the subject's own spectrum.
**Sources.** Schwabedal et al., arXiv 1806.08675 (2018); Rommel et al. 2022 (best overall, up to +45% relative on small
BCI subsets).
**Effect sizes.** Bomatter & Gouk 2025 found it hurt TCN and mAtt on PhysioNet when channels get independent phases
(it destroys cross-channel structure).
**Why it ranks low.** The variation it adds is per-trial and zero-mean, which the 80-repeat average removes (C4).
**Sketch.** `FTSurrogate(channel_indep=False)` on r only, with phase_noise_magnitude = 1 (full 2 pi), p = 0.5; x' = mu + r'.

### 9. SmoothTimeMask
**Mechanism.** Zero a random contiguous span using sigmoid edges.
**Source.** Mohsenvand et al. 2020; benchmarked in Rommel et al. 2022.
**Effect sizes.** It helped BCI and was mixed for sleep. Rommel et al. also note that it erases transients.
**Why it ranks low.** The ERP at 100-300 ms is a short transient, and latency jitter already failed.
**Sketch.** Span of 0.05-0.15 s at p = 0.3.

### 10. RSC (representation self-challenging)
**Mechanism.** Mask the top-p% penultimate features ranked by gradient, then retrain on the rest.
**Source.** Huang et al., ECCV 2020 (arXiv 2007.02454).
**Effect sizes.** DomainBed: 62.7 vs ERM 63.9 (SAGM table) and 66.1 vs 66.6 (Fishr Table 9).
**Why it ranks low.** Channel dropout was already neutral here, and the method is close to targeted dropout.
**Sketch.** Drop 33% of the features with the largest dl/dh at p = 0.33.

**Considered and dropped.** DiWA / model soups (Rame et al., arXiv 2205.09739): the target metric is already a
function ensemble, and a weight soup only matches an ensemble at lower inference cost. RandConv (arXiv 2007.13003) is
close to the smoothing and spectral entries, and smoothing never beat no-aug here. DSU and other feature-statistics
perturbations belong to the MixStyle family.

## 2025-2026 EEG/MEG/fMRI cross-subject visual decoding: flags

| Paper | What it does for cross-subject | Status for us |
|---|---|---|
| SCORE, arXiv 2608.19134 (2026) | label-free subject coordinate recovery; 53.23 top-1 LOSO THINGS-EEG2 | multi-positive part tried, dead |
| SVTL, arXiv 2609.36971 (2026) | learned visual target views + MMD across source subjects; LOSO 35.3/65.6 (48.1 with training-free refinement) | MMD dead; excluded by the CVPR v2 rules |
| SAMGA, arXiv 2604.17782 (2026) | multi-layer visual targets + subject-dependent residual calibration (training only); +12.0 top-1 LOSO over the best rival | multi-layer targets dead |
| ProCA, arXiv 2609.05094 (Sep 2026) | progressive contrastive anchors + "structure-consistent interpolation" (feature mixing weighted by channel/time importance); +28.1% relative top-1 in strict cross-subject; evaluates on Alljoined-1.6M | only augmentation-type 2026 item not yet tried; mixup family, so low prior; read the method before spending GPU |
| ZEBRA, NeurIPS 2025 (arXiv 2510.27128) | fMRI; adversarial split into subject vs semantic parts, zero-shot new subjects | adversarial dead |
| ViEEG, arXiv 2505.12408 (2025) | hierarchical encoder; 22.9 top-1 cross-subject THINGS-EEG | architecture, out of scope |
| SATTC, CVPR 2026 | test-time whitening + mutual-NN calibration | TTA, out of scope |
| Bomatter & Gouk, arXiv 2503.13497 (2025) | participant-disjoint splits: AmplitudeScaling / FrequencyShift / PhaseRandomisation effects "small and inconsistent" | negative prior for #6 and #8 |

No 2025-2026 paper I found reports a training-time *input augmentation* that improves zero-shot cross-subject
EEG/MEG->image retrieval. The cross-subject gains in this literature come from targets, alignment or test-time
calibration.

## Suggested screen order (cv4 fold 1, REVE-d6 + THINGS, matched epochs; kill early per memory)
1. #1 conversion aug (cheap; compare directly with the running op-transfer). 2. #5 amplitude mix and #6
FrequencyShift as one cheap spectral pair. 3. #2 SAM (2x cost) on REVE-scratch / token-REVE first. 4. #3 SWAD stacked
on whichever run wins. 5. #4 Fishr (+ #7 as a rider). Judge every candidate by the 5-member ensemble replay, not by
the solo score.
