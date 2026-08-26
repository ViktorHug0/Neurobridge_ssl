# Decorrelated twin-model training experiments

## Objective

Test whether useful ensemble diversity can be induced directly during training rather than
obtained indirectly from architecture search. The first causal experiment uses two matched
TSConv models: if a small diversity leash lowers their prediction correlation while preserving
their solo strength and increasing their row-z ensemble accuracy, then at least part of the
ATM/TSConv complementarity mechanism is trainable.

This is a diagnostic test-selected wave. Any promising recipe should subsequently be repeated
with ValCon checkpoint and hyperparameter selection before it supports a generalization claim.

## Matched twin-TSConv setup

Both members use the established TSConv reference recipe:

- `TSConv_parameterizable`: temporal kernel 30, pooling kernel 51/stride 5, 40 temporal,
  spatial, and projection filters, ELU, dropout 0.5;
- InternViT-6B layer 28 mean-pooled 8-bit image features;
- linear projector, EEG backbone dimension 1024, alignment dimension 512;
- pairwise raw-EEG SubjectMix with alpha 0.5;
- grouped multi-positive batches, 9 samples per image, batch size 1024;
- learning rate `3e-4`, 50 epochs, plain-cosine evaluation;
- test-loss checkpoint selection on the held-out LOSO subject.

Member A is initialized with seed 3300 and member B with seed 3301. Initialization seeds must be
set explicitly per model. After both models are initialized, reset to one fixed training RNG so
all six arms see the same batches and SubjectMix draws. Otherwise lambda would be confounded with
data-order randomness.

The primary pair uses each member's individually test-selected checkpoint. Save a separate
contemporaneous pair checkpoint selected by ensemble test loss only as a clearly labelled
diagnostic; do not substitute it silently for the individual-selection protocol.

Assumption: “train 10 epochs normally” means retaining the 50-epoch reference schedule, with the
diversity coefficient equal to zero for epochs 1–10 and switched to its assigned constant value
at epoch 11.

## Training objective

For the two individual contrastive losses and the score-averaged ensemble loss:

\[
L = L_A + L_B + \beta L_{\mathrm{ensemble}} + \lambda L_{\mathrm{div}}.
\]

For each EEG query, row-standardize both score vectors, collapse repeated columns to unique image
identities, and remove every positive image. The initial diversity loss is the squared Pearson
correlation between the remaining negative scores:

\[
L_{\mathrm{div}}
= \frac{1}{B}\sum_i
\operatorname{corr}\left(z^A_{i,\mathrm{neg}},z^B_{i,\mathrm{neg}}\right)^2.
\]

Squaring targets zero correlation rather than encouraging pathological anticorrelation. The
positive entries are excluded so the models are not rewarded for disagreeing about the correct
image. JSD/KL diversity is deliberately deferred to a later wave.

`L_ensemble` applies the ordinary multi-positive contrastive objective to the uniform mean of
the two row-standardized score matrices. The individual losses remain present in every arm so
one member cannot become a sacrificial weak expert.

## First six intervention arms

| Arm | Lambda | Beta | Epochs 1–10 | Epochs 11–50 |
|---|---:|---:|---|---|
| `corr_l001_b0` | 0.01 | 0.0 | Individual losses only | Individual + correlation leash |
| `corr_l005_b0` | 0.05 | 0.0 | Individual losses only | Individual + correlation leash |
| `corr_l010_b0` | 0.10 | 0.0 | Individual losses only | Individual + correlation leash |
| `corr_l001_b05` | 0.01 | 0.5 | Individual + ensemble losses | Same + correlation leash |
| `corr_l005_b05` | 0.05 | 0.5 | Individual + ensemble losses | Same + correlation leash |
| `corr_l010_b05` | 0.10 | 0.5 | Individual + ensemble losses | Same + correlation leash |

Each arm is evaluated on all ten LOSO subjects, so the wave contains 60 joint-training folds.
The first three arms isolate the diversity leash. The second three test whether directly training
the deployed uniform ensemble converts the induced disagreement into useful accuracy.

## Controls

Use the existing matched seed-3300/3301 TSConv runs as the external zero-leash baseline. Before
launching the full wave, run the new joint trainer with `lambda=0, beta=0` on one fold and verify
that both members reproduce the expected solo range and that disabling the new terms produces
zero diversity/ensemble gradients. If the new trainer materially changes either member, a full
ten-fold `lambda=0, beta=0` control becomes necessary; the historical independent runs would no
longer be a clean control.

A later mechanism-control wave should include shuffled-partner decorrelation and post-hoc noise
matched to the achieved score correlation. Those controls distinguish structured complementary
errors from correlation reduced by noise.

## Measurements and decision rule

For every arm, export both members' embeddings and report:

- each member's ten-subject solo top-1;
- uniform row-z pair top-1;
- gain over the stronger member and gain over the pair mean;
- row-z score, true-margin, and correctness correlations;
- prediction disagreement and wrong-winner agreement;
- oracle accuracy and oracle headroom over the stronger member;
- fraction of oracle headroom realized by uniform averaging;
- the same measurements by subject, not only their ten-fold means.

Do not select the arm with the lowest correlation. Select from the strength–diversity Pareto
frontier: both members should remain close to the matched TSConv solo reference, and the primary
objective is the highest pair accuracy/gain over the stronger member. A lower correlation without
greater oracle headroom and ensemble accuracy is junk diversity.

## Rescue wave

The follow-up ATM + TSConv experiment replaces direct correlation pressure with soft
multiple-choice rescue. For each EEG query, detached responsibilities favor whichever member has
the lower multi-positive loss, and the corresponding weighted loss is added with coefficient
`gamma_rescue`. The ordinary member losses remain active. Rescue begins at epoch 1; `lambda=0`,
`beta=0`, assignment temperature is 0.5, and batch size is 512.

The scheduled dose response contains a full-data test-selected `gamma=0.50` arm and concept-ValCon
arms at `gamma=0, 0.10, 0.30, 0.50`. ValCon holds out a fixed 10% of source concepts with seed
20260822. Its primary pair combines independently validation-loss-selected members; matched
test-selected checkpoints from the same reduced-data trajectory are exported only as controls.

The proposed lambda range is a sensible conservative first sweep. If it barely changes negative
score correlation, extend the next wave upward to 0.3 and 1.0. If 0.1 already damages solo
accuracy, refine between 0.01 and 0.05 instead.

## Prepared implementation

- `train_twins.py` jointly trains the two independent TSConv branches, applies the scheduled
  decorrelation leash, saves member-selected and pair-selected checkpoints separately, and
  exports ensemble-ready embeddings.
- `losses.py` contains the row-z ensemble loss and negative-only squared-correlation loss.
- `run_first_wave.sh` maps indices 0–5 to the pre-registered arms and runs the ten LOSO folds.
  It also exposes a one-fold `control` mode, which is not part of the six-arm array.
- `first_wave.sbatch` is the unsubmitted six-job Slurm array, capped at three concurrent GPUs.
- `smoke.py` checks the loss geometry, gradient flow into both branches, and TSConv wiring.

No job is submitted by these files. Before the full wave, run the local smoke test and the
one-fold zero-leash control:

```bash
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
python -m ensemble_experiments.decorrelated_models.smoke
SUBJECTS=1 bash ensemble_experiments/decorrelated_models/run_first_wave.sh control
```

After the control is accepted, the prepared full-wave submission command is:

```bash
sbatch ensemble_experiments/decorrelated_models/first_wave.sbatch
```
