# EEG-to-image and EEG-to-caption experiments

THINGS-EEG2, subject 01 unless stated otherwise. The test split is 200 concepts, one image each,
80 repetitions averaged; the training split is 16540 images over 1654 concepts, 4 repetitions
averaged. Training and test concepts are disjoint (verified: 0 of 1654 overlap), a property several
of the controls below depend on.

## 1. Reconstruction

### 1.1 Within-subject baseline

An EEG encoder is trained to regress raw ViT-H/14 image embeddings, which are then inverted to
pixels by SDXL-Turbo with IP-Adapter (`ip-adapter_sdxl_vit-h`). Targets are not L2-normalised: the
predicted norm carries the model's confidence, and IP-Adapter is scale-sensitive.

Forcing the predicted embedding to the empirical ViT-H norm (28.61 +- 0.10 predicted vs 22.27 +-
1.15 real) was tested and rejected: it improved CLIP similarity but degraded Inception, a wash
overall. Predicted magnitudes are near-constant, indicating the confidence head does not modulate.

### 1.2 ENIGMA reproduction

To establish that the generative stage is not the limiting factor, the ENIGMA reference
implementation was run unmodified on our data (`enigma_adapter.py` supplies the metadata parquet,
ViT-H feature tensors and repetition-averaged EEG in the format their loader expects). Four defects
prevented the published code from executing; all are mechanical and none alter the method:

1. `source/dataset.py` calls `get_eeg_data()` before `get_metadata()`, but the former indexes with
   `stim_indices`, which the latter assigns.
2. `source/dataset.py` hard-codes an Alljoined stimulus directory.
3. `train.py` calls `dist.init_process_group()` unconditionally, so it runs only under a torchrun
   launcher; the repository's own `retrieval.sh` invokes it directly and therefore fails.
4. `requirements.txt` omits the `clip` package and leaves `diffusers` and `transformers`
   unpinned, so resolution proceeds past the pinned `torch==2.4.1`.

Results for subject 01 at the final (epoch 150) checkpoint, against the published
ENIGMA single-subject row:

| | reproduction | published |
|---|---|---|
| PixCorr | 0.1882 | 0.1718 |
| SSIM | 0.4268 | 0.4233 |
| AlexNet(2) | 82.31 | 83.64 |
| AlexNet(5) | 88.60 | 89.49 |
| InceptionV3 | 76.75 | 77.65 |
| CLIP | 79.16 | 81.48 |
| Top-1 | 22.50 | 27.60 |
| Top-5 | 50.50 | 59.35 |
| Top-10 | 64.50 | 71.15 |

All six reconstruction metrics agree within approximately two points. The three retrieval metrics
are lower by 5-9 points. Two explanations remain untested: the published row is reported under a
ten-subject heading and plausibly averages ten independently trained models, whereas the
reproduction is a single subject; and training overfits, with validation loss reaching 3.29 near
epoch 30 and rising to 3.60 by epoch 150, the checkpoint from which all metrics are computed.
ENIGMA applies no early stopping, and none was introduced.

## 2. Captioning

Reference captions are Qwen2-VL descriptions of the test images, one per concept.

### 2.1 Metric, and a defect in its floor

The reported metric is cwBLEU: unigram BLEU computed after removing words occurring in more than
15% of the caption pool, with colour terms exempt. Stripping the shared scaffolding is necessary,
since an uninformative caption otherwise scores approximately 35 BLEU-1.

Two null models were used, and they are not equivalent.

The *floor* null emits one fixed generic caption for every trial. **This null is unusable and every
comparison against it should be disregarded.** Its value depends entirely on the sentence chosen,
ranging from 0.00 for "a photo" to 5.38 for the sentence originally used to 16.94 for the
greedy-optimised "white black green brown blue red front image". The last of these exceeds every
EEG arm reported below. Two properties combine to produce this: colour terms are exempt from
template stripping, and they are the most frequent guessable attribute words; and BLEU's brevity
penalty punishes only hypotheses shorter than the reference, so padding with likely words is free.
A tuned constant is therefore an adversarial baseline, not a floor.

The *permutation* null shuffles an arm's own emitted captions across concepts, holding caption
distribution, style and length fixed and destroying only the pairing. Hedging cannot help it, and
it is the sound test. All significance claims below use it, or a paired bootstrap between two arms.
Both are in `caption_bootstrap.py` (`--mode permute` / `--mode floor`).

### 2.2 Routes evaluated

Four ways of turning an EEG-derived embedding into a sentence were compared. Three share ENIGMA's
predicted ViT-H embedding as input, so they differ only in what follows it.

1. **Via image.** The embedding drives SDXL-Turbo, and BLIP-2 captions the rendered image.
2. **Retrieval.** Emit the caption of the nearest *training* image in the same embedding space.
   Training and test concepts are disjoint (verified: 0 of 1654 overlap), so a retrieved caption
   can never be the reference.
3. **Generative bridge.** A linear map from the embedding to BLIP-2 Q-Former tokens (32x768),
   decoded by the frozen OPT-2.7b. A text decoder replacing SDXL.
4. **Generative bridge, fit on predictions.** As (3), but the bridge is trained on ENIGMA's own
   predicted embeddings for the 16540 training images rather than on real image embeddings with
   Gaussian noise standing in for the corruption. Supersedes (3).
5. **Earlier learned maps** from our own 512-d InternViT space, superseded by (3).

The **ceiling arm** referenced throughout is route (3) with one substitution: the true ViT-H
embedding of the real stimulus image replaces ENIGMA's predicted one. Bridge weights, decoder and
decoding configuration are identical. It therefore isolates the cost of the EEG: everything
downstream of the embedding is held fixed. It is not BLIP-2 captioning an image in the normal way,
which runs a ViT-g tower and Q-Former over image patches; the bridge instead predicts that
Q-Former output from a single pooled 1024-d vector.

### 2.3 Results

| arm | cwBLEU | permutation null | p |
|---|---|---|---|
| ceiling: real image, sigma 0 bridge | 29.38 | 4.68 | <0.001 |
| ceiling: real image, sigma 2.59 bridge | 19.56 | 2.36 | <0.001 |
| **generative bridge fit on predictions, sigma 1.08** | **10.06** | 4.36 | <0.001 |
| generative bridge fit on predictions, sigma 0 | 8.84 | 4.49 | <0.001 |
| retrieval, ENIGMA embedding | 8.61 | 4.29 | <0.001 |
| generative bridge fit on images, sigma 2.59 | 7.09 | 3.16 | <0.001 |
| generative bridge, sigma 0 | 4.59 | 2.49 | <0.001 |
| via image (SDXL then BLIP-2) | 5.40 | 2.12 | <0.001 |

Paired bootstraps over the 200 concepts, 2000 resamples:

| comparison | delta | 95% CI | p |
|---|---|---|---|
| fit on predictions (sigma 1.08) vs fit on images (sigma 2.59) | +2.98 | [+1.39, +4.57] | <0.0001 |
| fit on predictions, sigma 1.08 vs sigma 0 | +1.22 | [+0.00, +2.53] | 0.050 |
| fit on predictions (sigma 1.08) vs retrieval | +1.45 | [-0.01, +2.98] | 0.052 |
| generative, noise 2.59 vs 0 | +2.52 | [+1.07, +4.02] | <0.0001 |
| ceiling, noise 0 vs 2.59 | +9.91 | [+6.44, +13.68] | <0.0001 |
| generative vs via image | +1.69 | [+0.02, +3.45] | 0.050 |
| retrieval vs generative | +1.50 | [-0.42, +3.32] | 0.120 |

### 2.4 Findings

**Decoding text beats rendering an image.** From the same ENIGMA embedding, the generative bridge
scores 7.09 against 5.40 for rendering pixels first and captioning them. The effect is modest and
borderline (p = 0.050), but it is not attributable to retrieval, which this route does not use.

**Fitting the bridge on predictions helps only in combination with the decoding loss.** The 2x2,
all four cells decoding ENIGMA's test predictions with the same frozen OPT and decoding config:

| cwBLEU | closed-form ridge (MSE only) | trained (MSE + 0.1 CE, 30 epochs) |
|---|---|---|
| fit on real image embeddings | 7.50 | 7.09 |
| fit on ENIGMA's predictions | 7.48 | **10.06** |

Paired bootstraps over the 200 concepts: closed-form, predictions against images, -0.02
[-1.37, +1.38], p = 0.948; trained against closed-form on images, -0.50 [-2.12, +1.31], p = 0.572;
trained against closed-form on predictions, **+2.58 [+1.13, +4.13], p = 0.0013**.

Neither factor acts alone. Changing what the bridge is fit on, with a solver that cannot overfit,
does nothing. Adding the cross-entropy term, when fitting on clean image embeddings, does nothing.
Only together do they move the arm, by about 2.6 to 3.0.

An earlier reading of this section attributed the gain to Gaussian noise reproducing the error's
magnitude but not its geometry. That is falsified: the geometry alone is worth nothing. The
mechanism is that cross-entropy optimises **decodability** -- whether the frozen decoder can emit
the reference from the predicted prefix -- which is not the same objective as L2 proximity, and it
transfers only when the training inputs are drawn from the distribution that will be decoded.
MSE-optimal prefixes are not decodable prefixes, and decodability can only be learned on the
distribution you will decode.

A practical consequence: the closed-form solver reaches 7.5 in about two seconds against the
trained bridge's two hours, and matches it in the image-fit cell, so it is a usable screen. It
cannot reach the winning cell, so the trained run is still required for a final number.

**Retrieval no longer leads, but has not been beaten.** 10.06 against 8.61 is +1.45
[-0.01, +2.98], p = 0.052. The earlier position, generative trailing and statistically tied at
p = 0.12, has become generative ahead and statistically at the boundary. Neither supports a claim.

**The bridge is not the bottleneck.** Trained without noise it reaches 29.38 on real images, against
29.82 for BLIP-2's own pipeline, with 84.75 colour accuracy and no degenerate decodes. Mapping an
embedding to fluent text is solved at this scale.

**The failure is distributional.** ENIGMA's predicted embedding has cosine 0.36 with the true ViT-H
embedding of the same image, and a norm of 33.83 against 22.27 (coefficient of variation 1.7%,
so magnitude is uninformative and both sides are normalised). Training the bridge against matched
Gaussian noise, sigma 2.59 from cos ~ 1/sqrt(1+sigma^2), buys 2.52 cwBLEU on the EEG arm and costs
9.91 on the clean ceiling. A robustness/accuracy trade-off of that shape is direct evidence that
the constraint is the input distribution, not model capacity.

**Decoding configuration dominated every modelling choice tested.** Beam search without a
repetition penalty or n-gram blocking collapsed into degenerate loops on both the EEG arm and the
clean ceiling; correcting it moved those arms from 5.19 to 7.09 and from 14.08 to 19.56. Generated
text must be inspected directly before any caption metric is trusted.

**Nucleus sampling does not help.** The 16% degenerate decodes were expected to be recoverable
value. They are not. Sampling (top_p 0.9) removes them entirely, 33/200 to 0/200, and lowers cwBLEU
on both arms, monotonically in temperature: 7.09 (beam) to 5.47 (t=0.7) to 3.91 (t=1.0), with the
clean ceiling falling 19.56 to 16.98 to 13.17. Since the ceiling had little degeneration to remove,
the loss is the sampling variance itself, not an EEG-specific effect. Beam search with a repetition
penalty remains the configuration. Residual failures persist under sampling in a form the
repetition detector does not catch ("a a an animal animals"), indicating a prefix too diffuse to
commit to a noun rather than a decoding defect with headroom.

**Attribute detail survives once the bridge is fit on predictions.** Colour accuracy on the EEG arm
moves 29.94 to 51.98, against 84.75 at the clean ceiling. Under the earlier training it sat at or
below control level and the conclusion recorded here was that attributes do not reach language.
That conclusion was a property of the training distribution, not of EEG.

**The degenerate decodes were a training artefact, not a decoding one.** 30 of 200 under the
noise-fit bridge, **0 of 200** under both prediction-fit arms, with mean length rising from 16.0 to
23.9 words. Nucleus sampling had removed the degeneration only by lowering the score on every arm
(section 2.4), which was read here as evidence that the 16% was not recoverable headroom. It was
recoverable; the isotropic-noise prefix was simply too diffuse for the decoder to commit to a noun.
Compare, for antelope: "a a a a an a a" against "A ground squirrel with a light brown coat is
standing on a sandy surface."

## 3. Limitations

All caption results are subject 01, and the reported intervals are over test items rather than
subjects, so no claim of cross-subject generality is supported.

A closed-form sweep over lambda, which is sigma^2 by the Bishop (1995) equivalence, is flat:
cwBLEU 7.07 to 7.58 across sigma 0.32 to 5.48, six points in 5 min 08 s. That result is confined to
the image-fit, MSE-only cell and does not license any claim about sigma in the trained
prediction-fit cell, which is where the best arm lives and where sigma has not been swept.

All prediction-fit results use ENIGMA's *in-sample* training-split predictions with sigma correcting
the resulting cosine gap in aggregate. That correction is isotropic and therefore matches the
cosine without matching the in-sample error's geometry, which is the same objection this route
raises against the noise baseline, at smaller scale. Cross-fitting removes it: ENIGMA trains in
5 min 54 s, so five folds cost about 30 minutes. Untested.

The retrieval route emits a caption describing a different object near which the EEG landed, and is
not generation. cwBLEU is defined here rather than taken from the literature, and section 2.1
documents a way to game it that any future use must account for.

## Files

`enigma_adapter.py` writes ENIGMA-format inputs. `enigma_repro.sbatch` runs their three stages;
`enigma_eval.sbatch` and `enigma_retr.sbatch` re-run evaluation and retrieval alone.
`reconstruct_eval.py` and `run_recon_subject.sh` are the within-subject pipeline.
`caption_bridge.py` is the generative bridge (`--input vith --enigma_dir` for the ENIGMA arm,
`--noise` for the augmentation, `--do_sample` for nucleus decoding); `enigma_bridge.sbatch`,
`enigma_regen.sbatch` and `enigma_nucleus.sbatch` run it. `caption_nn.py` is the retrieval route
(`--space enigma`), `clip_space_probe.py` the CLIP-space probe, `make_captions.py` the captioner,
`caption_eval.py` the scorer and `caption_bootstrap.py` the null models. Scripts are invoked from
the repository root, e.g. `.venv/bin/python reconstruction_experiments/caption_eval.py`.
