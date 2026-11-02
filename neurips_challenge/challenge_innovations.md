# Challenge innovations

NeurIPS 2026 Track 1, sealed phase: EEG -> DINOv2 image embedding, scored by subject-aggregated top-5 over a fixed
200-image gallery, on 11 unseen participants. The grader calls `predict(X)` on batches of 64 trials **shuffled across
participants**, with no subject ID, no gallery and no memory between calls. Numbers below are on our frozen 4-fold
cross-validation (20 held-out Alljoined subjects) with the official metric under the same shuffled condition
(`grader_sim.py --shuffle`).

**Current best: 40.05**, from REVE-d6 + Squeezeformer + TSConv, 33.7 before post-processing.

---

## 1. Two conv encoders that share a front end

**TSConv** (~0.25M parameters):
1. Learns 40 temporal filters.
2. Smooths them over 0.2 s.
3. Learns one spatial filter per map across the 32 electrodes.
4. **Flattens** the 40 x 36 map, so every map and time step has its own weight. The readout is locked to the
   stimulus timing.
5. Squeezes through a 64-dimensional code, then lifts to the 1536-dimensional DINOv2 space.

**Squeezeformer** (~2.2M parameters) uses the same front end (steps 1-3), then departs:
- Each of the 36 time steps becomes a token.
- Three blocks mix the tokens: attention over time, a local temporal convolution, and a feed-forward layer.
- Instead of flattening, it pools with an attention-weighted mean and standard deviation over time. That makes it
  tolerant to small latency shifts, where TSConv assumes a fixed waveform at a fixed latency.

The two make different mistakes, so they are worth more together than alone. Squeezeformer was weaker than TSConv
alone at first but always helped the ensemble.

Today's architecture search (ARCH_LOG.md) found no design that beats TSConv's own recipe:
- Adding attention after the spatial filters loses 1-2.5 points.
- Coarser pooling loses ~7.
- Extra fast paths lose ~2.
- 20 to 160 filters are flat.

## 2. Per-dataset BatchNorm

We add the 10 THINGS-EEG2 subjects to the 15 Alljoined training subjects. They were recorded on different hardware
(research amplifier vs consumer Emotiv headset), so their features have different means and scales.

Per-dataset BatchNorm keeps two sets of normalisation statistics (and gain/offset), one per dataset, around otherwise
shared weights. Each training trial is normalised with its own dataset's statistics. At test time, unseen participants
use the Alljoined statistics, which is the same headset.
- Without it, THINGS hurt TSConv (-3.4).
- With it, THINGS became neutral for TSConv and gave Squeezeformer +4.3, its largest gain.
- REVE has no BatchNorm and needs no routing.

Why it works where per-subject tricks failed: two domains, each with hundreds of thousands of trials, are easy to
estimate. Subject-level parameters have to generalise from 25 examples of a subject.

## 3. Ensemble assembly

Per batch, each member's outputs are divided by the member's **batch-average** norm. Individual trials are not
normalised, so each trial keeps its relative norm, which the training loss turns into a per-trial confidence. Members
are post-processed separately (sections 4-5), then summed.

---

## 4. Batch post-processing: half-mean centring + top-direction shrink (the detailed part)

```
mu  = mean of the 64 outputs in the batch
C   = outputs - mu
V   = top-k principal directions of C        (k = 2)
C   = C - 0.5 * (projection of C on V)       (shrink them by half)
out = C + 0.5 * mu                            (keep half of the mean)
```

### 4.1 Finding 1: each subject's outputs carry a common offset, and removing half of it helps
How we got there (ROUTING_LOG.md):
- **E4.** We first tried to *weight* trials by label-free quality features (signal quality, norm, agreement). They
  predict nothing (R^2 = 0.04). The side finding was that subtracting half of each subject's **average output** from
  all its trials gives +2.9 (17/20 subjects better).
- **E5.** We then asked what that average contains.
  - *It is mostly shared.* The cosine between one subject's mean output and the other subjects' mean is 0.98.
  - *The useful part is the subject-specific remainder.* Subtracting a **fixed** mean taken from other subjects
    gives only +0.3 for SQF, while subtracting the subject's **own** mean gives +3.6.
  - So each participant's outputs carry an extra component of their own, which the model adds to every trial
    whatever the image.
  - The mean of a 64-trial batch from that subject works as well as the mean of all its trials (+3.0 / +3.6), so it
    can be estimated statelessly.
- **Why half.** Subtracting a times the mean changes each score to q.g - a (m.g). Two things happen at once:
  - the subject offset is removed;
  - gallery images that *every* output resembles are penalised, a hubness correction.
  Sweeping a, 0.5 is best; 0.25 and 0.75 give ~+2; full removal (a = 1) gives ~0, because the mean also holds image
  information shared by the test images.
- **Today's check (SUBJECT_LOG D1)** confirms this is the only output-space correction available. Even rotations or
  linear maps fitted per subject *with labels* do not beat simple centring.

### 4.2 Finding 2: the largest direction of variation in a batch is mostly nuisance
- **E8.** After centring, shrinking the batch's top principal direction by half adds +2.0 (TSC) / +3.0 (SQF). Removing
  it completely costs -3 to -6, and shrinking 8 directions breaks everything.
- **E10.** What that direction is:
  - it holds 19-25% of the batch variance;
  - it points the same way from batch to batch (|cos| 0.43-0.59), so it is a stable property of the model or
    recording, not of the images in the batch;
  - it is not the batch mean (|cos| 0.19) and is only weakly linked to the trial norm (0.2);
  - a **fixed** direction learned on other subjects recovers ~40% of the gain, and the per-batch estimate the rest.
  So it is a recurring nuisance axis (subject/recording variation dominates the spread of single-trial outputs), but
  it still carries some signal: hence shrink by half, never remove.
- **E15/E23, mixed batches.** In the sealed condition a batch mixes ~11 participants. Centring alone then loses most
  of its effect (the batch mean becomes the population mean): +0.5 at 15 participants per batch. The top-direction
  shrink keeps working (+1.8 to +2.2). After centring, the top directions of a mixed batch are largely *the spread of
  the participants' offsets*, so shrinking them is an implicit, label-free per-subject correction. With more
  participants per batch, 2 directions beat 1, so the default is k = 2.
- **E22/E25, ensemble under the sealed condition.**
  - 33.63 -> 38.62 (+5.0, all 20 subjects better).
  - Insensitive to batch size (16-256).
  - Worst case (a batch of one image) is harmless, because cosine ignores rescaling.

### 4.3 Refinement: route by EEG fingerprint (+1.2)
Centring with the *whole* mixed batch removes the average offset of ~11 people. Removing each trial's *own* person's
offset is worth much more:
- **E20, oracle.** Centring each trial on its true same-person batch-mates gives +8.4, vs +4.4 for whole-batch
  centring.
- **E20/E21, neighbours.** Without labels, centring on the trial's nearest batch-mates in a raw-EEG "fingerprint"
  space (channel variances, channel correlations, band powers: properties of the recording, not of the image) gives
  +5.2 to +5.6. Learning the fingerprint metric with LDA did not raise neighbour purity (0.50 vs 0.49).
- **E32-E43, deployed form.**
  - Regress out of the centred outputs the batch's top 6 directions of variation in subject-discriminant fingerprint
    space. The LDA is learned offline on training subjects.
  - Then shrink the top 3 output directions.
  - Result: 38.62 -> 39.80 (13 subjects better, 5 worse). A random-direction control gets about 1 point less, so the
    gain is subject information.

### 4.4 Tried and failed
| idea | result | why it fails |
|---|---|---|
| training the networks *under* the transform (E13, E26) | -3.5 to -5.5 | the model learns to rely on batch composition |
| full or partial whitening of the batch outputs (E45) | -25 (full) to about +4.3 (heaviest shrinkage) | equalises all ~63 directions, amplifying single-trial noise (section 5) |
| whitening by the DINOv2 gallery statistics (E17) | 29.4 -> 5-20 | up-weights image directions where the EEG prediction is pure noise |
| static population whitening + rotation, SAGE's gallery-free analogue (E18) | 24.0 vs 29.4 | SAGE's gains need the gallery and per-subject fits (section 6) |
| ridge recalibration maps to the targets (E19) | -25.6 to -1.5 | the network's own readout is already optimal; any map adds estimation noise |
| per-trial quality weighting (E4); norm-capped or unit statistics (today, A28) | -2 to +0.08; +0.03 / -0.47 | trial usefulness is unpredictable without labels |
| subject-level member reweighting, with oracle (E3) | no honest headroom (-0.28) | |
| AdaBN on top of centring (E7) | +0.1 to +0.3 | centring already captures it |
| static output subspaces across models (E30) | do not transfer | output spaces differ between models; EEG-side fingerprints do transfer |

### 4.5 Still open
- **Better person routing.** The oracle gap is +4.2 points beyond our label-free routing (E20/E21), and the fingerprint
  content is the bottleneck, not the metric. Richer per-trial recording descriptors (learned on training subjects to
  predict the subject's output offset directly, rather than the subject's identity) are untested.
- **Combining members jointly instead of separately**: see the discussion in the session notes (consensus or
  product-of-experts over a training-image proxy gallery); untested.

## 5. Why whitening does not work here

A 64-trial batch spans at most 63 output directions. A few carry structure (the nuisance axes above and the image
signal); most are single-trial noise, because one EEG trial has very low signal-to-noise. Whitening rescales every
direction to the same variance:
- it amplifies the many noise directions to the weight of the informative ones;
- it erases the per-trial confidence carried by the norm.

The post-processing that works does the opposite: it touches only the top 2-3 directions, and only halfway.

**Is our transform just Ledoit-Wolf partial whitening at shrinkage 0.99?** Close, but not the same, and the router is
better.
- At shrinkage d = 0.99, whitening multiplies direction i by about sqrt(1 / (1 + 0.01 * lambda_i / nu)), where
  lambda_i is that direction's variance and nu the average.
  - A dominant direction with 100x the average variance is shrunk to ~0.7.
  - Typical directions stay at ~0.99.
  - That is a smooth version of "shrink only the top few directions".
- It scores like our k = 2 shrink (s11 testbed: +4.32 vs +4.50; cv4: +4.35 / +6.25 vs +4.35 / +6.45).
- It is below the k = 3 shrink (+5.09) and the fingerprint router (+5.93).
- Any lighter shrinkage (more whitening) loses monotonically.

## 6. Why SAGE-TTA (our NeurIPS submission) does not transfer

SAGE adapts one subject at a time:
1. It takes all of that subject's test queries **and the test gallery**.
2. It corrects hubness and matches queries to gallery images with optimal transport.
3. It fits a rotation on the confident matches, and iterates.

On THINGS-EEG2 it lifts top-1 from 27 to 50. The sealed grader removes all three things it needs:
- **No gallery inside `predict()`.** The grader does the ranking itself.
- **No subject-pure query set.** Batches mix ~11 people, and nothing may be remembered across calls.
- **The gallery-free, population-level substitutes are strongly negative** (E17-E19, section 4.4).

Today's D1 shows why: even per-subject maps fitted with labels do not beat centring.

What survives the rules is the stateless part of the idea: correct each batch's offsets and nuisance directions.
