# Next experiments, in priority order

Companion to `experiments.md`, which records what has been run. Goal: the best possible
**intra-subject** caption on THINGS-EEG2 subject 01. Cross-subject generalisation is out of scope
and no longer drives the ordering.

## What constrains the list

- **The bridge is not the bottleneck.** A linear map reaches cwBLEU 29.38 against BLIP-2's own
  29.82, so at most 0.44 is available to any increase in bridge capacity. Every remaining gap is
  upstream, in the embedding.
- **Retrieval accuracy does not predict caption quality.** Job 47607: an 81.00 top-1 intra-subject
  encoder gave cwBLEU 6.51, against a fixed-caption floor of 5.38 and a shuffle control of 5.15.
  ENIGMA, at 22.50 top-1, gives 7.09. That run was handicapped twice (no noise augmentation, and it
  predates the decoding fix), so it is suggestive rather than settled, but the direction matches the
  fMRI literature.
- **The cost asymmetry decides the strategy.** An intra-subject encoder trains in **2.5 minutes**
  (`train.log`, 50 epochs, 17 occipital channels, batch 1024). A bridge takes **2 hours**, almost
  all of it the cross-entropy term backpropagating through the frozen OPT-2.7b. Generation is 2 min
  and scoring 1 min.

The strategy follows directly: **screen encoders with a cheap diagnostic, build bridges only for
the winner.** Do not tune sigma for an embedding space that may be abandoned.

---

## E0. Embedding diagnostic, and use it to screen encoders

**Question.** Which embedding space is actually suited to text decoding, before spending 2 h finding
out?

**Method.** Three numbers, computed on the test split in whatever space the bridge would consume:

1. **cos(EEG prediction, true image embedding).** ENIGMA is at 0.36. This is the quantity that sets
   sigma analytically, via cos ~ 1/sqrt(1+sigma^2).
2. **Effective rank** of the predicted embeddings against the real image embeddings. Previously
   measured at 34/17 for EEG against 97/65 for images. A prediction living on a 17-dimensional
   manifold cannot support regression onto 24576 Q-Former values, however good its retrieval.
3. **Coefficient of variation of the predicted norm.** ENIGMA's was 1.7 percent, i.e. effectively
   constant, which is why both sides are normalised. Contrastive training with `img_l2norm` discards
   magnitude by construction.

**Then screen encoder variants at 2.5 min each**, scoring them on (1) and (2) rather than on top-1:

- the current 81.00 top-1 checkpoint, `results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01`
  (EEGProject, InternViT-6B layer 28, 512-d, verified pure InfoNCE: `alpha=1.0` skips the MSE branch
  entirely);
- the same with a reconstruction term, `--alpha 0.9 --mse_on_raw`, and a small alpha grid. **Both
  flags already exist** in `module/loss.py`; `mse_on_raw` puts the MSE on the *unnormalised* target,
  which is exactly what ENIGMA does and what `img_l2norm=true` currently throws away;
- a multi-granularity image target from `make_multigranularity_features.py`, which concatenates
  L2-normalised InternViT depths and needs no change to `train.py`.

**Cost.** ~10 min for the diagnostic code, then ~1 h for a dozen encoder variants.

**Success.** At least one variant clears ENIGMA's 0.36 cosine without collapsing in effective rank.
**Kill.** Every variant sits near rank 17, which would establish that the 512-d contrastive space is
capped and redirect effort to the target side (E4).

---

## E0 RESULTS (2026-08-25, done)

**The diagnostic itself needed fixing twice.** First, effective rank has two conventions that differ
~3x on identical data: exp(H) over the covariance spectrum (s^2) against over singular values (s).
The project's older 34/17 vs 97/65 figures are the s^2 convention; `caption_diag.py` now follows it.
Second and more serious, **cos(EEG prediction, image embedding) is confounded**: `train.py` learns
the image projector, so the target moves. At `--alpha 0` the projector simply collapses the image
side onto the EEG -- cos 0.998 with **top-1 2.00%**, against 0.5% chance. The target's own effective
rank falls monotonically long before that (95.3 -> 89.1 -> 84.0 -> 72.5 as alpha drops 1.0 -> 0.1),
so the confound is present throughout. Encoders are therefore ranked by `caption_ridge.py` against
the **fixed** cached Q-Former targets, which depend on no encoder.

Rank on `cos_qf`, not R2. The two disagree, and cos_qf reproduces the one ordering with ground
truth: ENIGMA scored cwBLEU 7.09 against base's 6.51, and cos_qf puts ENIGMA ahead (0.271 vs 0.238)
while R2 inverts it (0.0077 vs 0.0560). R2 is scale-sensitive and the spaces differ in width.

| encoder | cos_qf | cos(proj) | top-1 | note |
|---|---|---|---|---|
| ENIGMA, 1024-d ViT-H | **0.271** | 0.360 | 22.50 | cwBLEU 7.09, the only clean anchor |
| mse03, `--alpha 0.3 --mse_on_raw` | 0.251 | 0.426 | 86.50 | best of 12 variants |
| mse05 / mse05norm / mse07 | 0.247 | 0.399 | 85.50 | |
| mse02 / mse02norm / cat3mse | 0.246 | 0.447 | 87.50 | |
| mse01 | 0.244 | 0.497 | 87.50 | highest raw cosine, 8th here |
| cat3 / cat7 (multi-granularity) | 0.242 | 0.342 | 79.50 | |
| base, `--alpha 1.0` (pure InfoNCE) | 0.238 | 0.346 | 81.00 | reproduces the 81.00 checkpoint exactly |
| mse00, `--alpha 0.0` (pure MSE) | 0.155 | 0.998 | 2.00 | degenerate |

**Four findings.**

1. **A reconstruction term helps, and `--mse_on_raw` is not why.** Every variant with one beats
   base; the best is alpha 0.3 at 0.251 against 0.238. But raw and normalised MSE targets are
   indistinguishable at both ends of the range -- mse05 0.247 vs mse05norm 0.247, mse02 0.246 vs
   mse02norm 0.246. The ENIGMA magnitude argument does not survive; the mechanism is just "add a
   reconstruction term".
2. **Retrieval and caption-relevant quality dissociate, measured directly.** ENIGMA at 22.50 top-1
   and the baseline at 81.00 sit at cos 0.360 and 0.346, and at cos_qf 0.271 and 0.238 -- with
   ENIGMA *ahead* on both. 3.6x apart on retrieval, and the gap runs backwards on everything a
   bridge consumes.
3. **Multi-granularity targets are a wash here.** cat3 0.242 and cat7 0.242 against base 0.238, and
   both are *below* base on retrieval (79.50, 80.00 vs 81.00). What is tested is the concat-plus-
   linear-projector simplification of SAMGA, not a learned router, so this does not settle the
   original abstraction-level finding -- confirm which regime that came from before concluding.
4. **ENIGMA's space still leads.** No variant reaches 0.271. The reconstruction term is a real gain
   over our own baseline and not yet a win over ENIGMA.

**Cost, for planning.** Encoder 2.5 min, diagnostic seconds, ridge screen 3 min for all 12 on CPU,
bridge 2 h. The screen is ~40x cheaper than one bridge and ranks everything at once.

---

## E1. Train the bridge on out-of-fold EEG predictions instead of noise

**Question.** The bridge is fit on real image embeddings corrupted by Gaussian noise, then tested on
ENIGMA's predictions. Why simulate the corruption rather than train on it?

**Why this is the strongest remaining lever.** sigma = 2.59 matches the *magnitude* of ENIGMA's
error, since cos ~ 1/sqrt(1+sigma^2) and the measured cosine is 0.36. It matches nothing about its
*geometry*. ENIGMA's predictions occupy a low-dimensional manifold, effective rank 34/17 against
97/65 for real images, and its errors are structured rather than isotropic. Gaussian noise added to
a real embedding produces a full-rank perturbation that does not lie near that manifold, so the
bridge currently trains on inputs with the correct cosine and the wrong support. Training on real
predictions corrects both at once.

**The defect to avoid.** ENIGMA was trained on the same 16540 images, so its predictions there are
in-sample and better than at test. Using them naively reproduces the distribution mismatch in
miniature.

**Method.**

1. **DONE.** Training-split cosine is **0.531** against 0.360 on test, so in-sample optimism does
   not cancel the 4-rep vs 80-rep gap. Running (job 49540) as two arms: `raw` (sigma 0, deliberately
   mismatched, isolates what the error geometry alone buys) and `n108` (sigma 1.08 on top, since
   0.531/sqrt(1+1.08^2) = 0.360, giving real geometry *and* the test cosine without cross-fitting).
2. **Cross-fit if they do not cancel.** Five folds over the training concepts, each ENIGMA model
   predicting its held-out fold. Training is 5 min 54 s per model (`enigma_48496.out`, 150 epochs),
   so five folds cost ~30 min.
3. **Retain a small sigma.** Noise draws fresh every batch, giving 15510 distinct corrupted views
   across training; real predictions give one per image. Use out-of-fold predictions for the error
   geometry plus a small residual sigma for coverage.

**Cost.** ~40 min of code, ~30 min of folds, ~2 h for the bridge. The regression *target* is
unchanged, still the cached Q-Former output of the real image, so nothing needs re-caching. Only
the input side moves. `recon_inference.py` is hard-coded to `split="test"`, but `enigma_adapter.py`
already writes the training EEG in ENIGMA's format, so loading `last.pth` and forwarding it is
~20 lines, plus a branch in `caption_bridge.embeddings()`.

**It generalises.** The InternViT encoder's training-split embeddings are equally in-sample, so the
general form is: fit the bridge on out-of-fold EEG-predicted embeddings rather than on real image
embeddings plus noise, whichever encoder wins E0. Adopt it for every subsequent bridge if it holds.

**Success.** Beats 7.09 with sigma at or near zero, which would show the noise was standing in for
the real error distribution and doing it imperfectly. **Kill.** Matches or trails 7.09, which would
mean isotropic noise already captures what matters and the manifold argument is wrong.

---

## E2. Build one bridge, on the winner

**Question.** Does the best-diagnosing intra-subject encoder beat ENIGMA's 7.09?

**Method.** `caption_bridge.py --input proj --checkpoint_dir <ckpt>`, sigma set analytically from
E0's cosine, current decoding configuration (beam 3, `repetition_penalty` 1.5,
`no_repeat_ngram_size` 3). If E1 has already reported, train on out-of-fold predictions in this
space instead, which is the same substitution applied to a different encoder. Then cwBLEU against the Qwen references and the permutation null. This is
the corrected rerun of job 47607: both defects that handicapped it are now fixed.

**Cost.** ~2 h. Run the ENIGMA arm's numbers alongside for a paired bootstrap.

**Note the bottleneck moves.** The projected space is 512-d against ViT-H's 1024-d, so the bridge
predicts 24576 values from half as many inputs. If E0 shows a low effective rank, that is the
binding constraint and not the width.

**Status: running, job 49558**, queued behind E1. Two arms under identical modern settings, base
(sigma 2.71) and mse03 (sigma 2.12), because base's old 6.51 predates both the noise augmentation
and the decoding fix and so cannot calibrate the screen. The ridge proxy has exactly one clean
anchor to cwBLEU today; these two add a second.

**Success.** Beats 7.09, or clears its own permutation null by a margin comparable to ENIGMA's.
**Kill.** Lands at 6.51 again, which would establish the earlier result was the encoder rather than
the two defects, and make E4 the only remaining direction.

---

## E3. Ridge screen for sigma, in the winning space only

**Question.** Is sigma near its optimum? It was set analytically and never tuned, and the measured
effect is large: sigma 2.59 against 0 is +2.52 cwBLEU.

**Why it is now fourth rather than first.** Sigma is a knob on a chosen space, so tuning it before
E0 and E2 have chosen that space spends 10 h on something that may be discarded. E1 weakens the case
further: if out-of-fold predictions supply the real error distribution, sigma is reduced to a small
residual and there is much less left to sweep.

**Method.** For a linear map under MSE, additive isotropic input noise of variance sigma^2 is
equivalent to ridge regression with lambda proportional to sigma^2 (Bishop, 1995). The bridge is
linear and MSE dominates its loss, so the sweep is closed-form rather than gradient descent.

1. Solve ridge over a lambda grid, seconds per point.
2. Decode and score each. Generation at 2 min per point is the real cost.
3. Run the full `--ce_weight 0.1` training at the argmax and at the analytic sigma.

**Cost.** Screen ~30 min, confirmation 2 x 2 h.

**Two approximations, not to be ignored.** Inputs are re-normalised after the noise is added, so the
perturbation is a projection onto the sphere rather than pure additive noise; and the cross-entropy
term is nonlinear, so the equivalence does not hold for it. The screen locates a neighbourhood, it
does not select the value.

**Success.** The two full runs rank as the screen predicted, and its argmax beats the analytic
sigma. **Kill.** The ranking disagrees, in which case the screen does not transfer and the choice is
a direct 5-point sweep at ~10 h or leaving sigma where it is.

---

## E4. Multi-abstraction targets

**Question.** Does aligning EEG to several levels of visual abstraction, rather than one pooled
embedding, produce a representation suited to text decoding?

**Why it belongs here.** Two independent findings point the same way. On the alignment side,
retrieval improves when the image target mixes intermediate InternViT depths rather than one final
embedding. On the decoding side, the fMRI literature converged on giving the encoder more than one
objective: MindEye and MindEye2 run separate contrastive and diffusion-prior heads, and ENIGMA mixes
MSE on unnormalised targets with InfoNCE. Both are the same prescription, that a single
contrastively-trained vector is the wrong object. The Q-Former target is itself a 32-slot
multi-aspect representation, so a single pooled bridge input is the narrowest point in the chain.

**Method.** The cheap variant is already inside E0's encoder screen: concatenate L2-normalised
aligned embeddings from several InternViT depths and widen the bridge input. If a concatenated
target diagnoses well and the 512-d single-layer target does not, that is the result, and it costs
one bridge to confirm.

The expensive variant, only if E0 shows the collapse is on the target side, is a per-depth head
architecture rather than a concatenated target. That is a real change to `train.py` and should not
be started before E0 reports.

**Cost.** E0's screen already covers the cheap variant; ~2 h for the confirming bridge.

---

## Parked

**Multi-subject replication.** Every caption number is subject 01, and the reported intervals are
over the 200 test items rather than over subjects. This does not make captions better and is
therefore not in the ordering above, but no write-up should claim a result from n=1. The pipeline is
per-subject and needs no new code, so it can run whenever a slot is free: 2.5 min per encoder plus
2 h per bridge.

## Not worth running

Recorded so they are not re-proposed.

- **A larger or nonlinear bridge.** The ceiling bounds the available gain at 0.44, and the noise
  result (+2.52 on EEG, -9.91 on the clean ceiling) places the problem in the regularisation regime,
  where added capacity is the wrong direction.
- **Predicting ViT-H's 257 x 1280 patch tokens.** ENIGMA's entire representation of a trial is 184
  numbers (`mlp_proj.0.weight` is 1024 x 184), so there is nothing to extract; predicting them would
  require a new model; and the ceiling shows patch tokens are worth at most 0.44 for this task.
- **Nucleus sampling.** Tested. It removes all degeneration and lowers cwBLEU monotonically in
  temperature on both arms. See `experiments.md` section 2.4.
- **Self-attention across the 32 bridge output slots.** All 32 are functions of the same input
  vector, so there is no independent information to exchange, and the cross-entropy term already
  supplies a joint gradient over the slots.
