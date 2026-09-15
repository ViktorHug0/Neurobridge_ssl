# F1/reference gap audit

Read-only analysis of existing runs; no new training or checkpoint inference.
Outer subjects 1–5, seed 3300, 200-way cosine retrieval with equal row-z fusion.

## Reproduced score comparisons

Candidate (object, image) identities were checked exactly across both references
and F1 before fusion. No fusion weight was selected on these test scores.

| Subject | Reference TS + reference ATM | F1 TS + reference ATM | Reference TS + F1 ATM | F1 pair |
|---|---:|---:|---:|---:|
| 1 | 51.5 | 52.0 | 46.0 | 44.0 |
| 2 | 50.5 | 49.0 | 50.0 | 48.0 |
| 3 | 34.0 | 32.5 | 28.0 | 27.5 |
| 4 | 36.5 | 30.5 | 33.5 | 32.5 |
| 5 | 43.0 | 42.0 | 39.5 | 39.0 |
| Mean | **43.1** | **41.2** | **39.4** | **38.2** |

Replacing only the reference TS costs 1.9pp; replacing only reference ATM costs
3.7pp. These substitutions are not additive causal attributions: the combined
drop is 4.9pp, not 5.6pp. They localize the larger saved-model degradation to
the ATM replacement, without distinguishing architecture, training or selection.
Solo means: reference TS 33.1 vs F1 TS 31.2; reference ATM 35.3 vs F1 ATM 30.4.

## Checkpoint-selection evidence

| Subject | Reference TS selected | Reference ATM selected | F1 TS loss minimum | F1 ATM loss minimum | F1 common selected | F1 stopped |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 94 | 15 | 70 | 50 | 70 |
| 2 | 26 | 88 | 26 | 59 | 42 | 62 |
| 3 | 17 | 68 | 13 | 53 | 38 | 58 |
| 4 | 26 | 96 | 26 | 62 | 46 | 66 |
| 5 | 26 | 96 | 26 | 74 | 56 | 76 |

F1 individual minima above are over its recorded trajectory, not hypothetical
independently trained trajectories. Simulating patience20 on those losses stops
the TS branch at epochs 35,46,33,46,46 with the same selected TS minima above.
ATM has not exhausted independent patience20 at the end of any recorded fold.
Thus common selection chooses TS well after its validation optimum and ATM
before its observed optimum; common stopping also censors ATM's trajectory.

At the common checkpoint, TS loss exceeds its observed individual minimum by
0.037417,0.022085,0.032939,0.041000,0.060485. ATM penalties are
0.034417,0.028717,0.021716,0.014011,0.017802. These are validation-loss penalties,
not measured test-accuracy penalties and cannot be converted into percentage points.

F1 saved only best.pth (common) and last.pth, not each branch's best weights or
per-epoch test scores. Independent-checkpoint test performance cannot generally
be recovered from these files. Loss curves do not establish that independent
selection would restore 43.1%.

## Implementation/configuration audit

Matched: pairwise alpha0.5, grouped batch1024/effective1017, nine source subjects,
concept holdout ratio0.1/seed20260822, raw IV33/IV28 features, 128D alignment,
AdamW3e-4/wd1e-4, fixed softplus temperature initialized0.07, image normalization
enabled and EEG normalization disabled inside the loss. No SATTC adaptation.
The TS reference and F1 manifests have identical hashes for all shared source
files; current F1 sources still match its manifest.

Not matched:

- F1 ATM temporal kernel30 versus archived kernel25. Confirmed directly from
  archived checkpoint tensor enc_eeg.0.tsconv.0.weight, shape(40,1,1,25).
  Kernel30 changes pooled length36 to35 and readout input width1440 to1400.
- F1 overwrites ATM's temporal stem initialization with a copy of TS's stem.
  This allows initial-function matching to weight-tied F2 but changes the native
  independent reference. A separate native-initialization control is missing.
- F1 separately seeds its EEG heads3301/3302 and image heads3303/3304. The TS
  reference constructs backbone and heads sequentially after seed3300. CPU
  construction verifies the TS EEG-head weight and bias differ at initialization.
- F1 combines two dropout-bearing branches in one forward pass; its subsequent
  random-number consumption differs from a solo run even with the same seed.
  F1/compact TS isolate epoch-specific mixup RNG; archived ATM config has no
  post-initialization train_rng_seed reset. Equal seed labels do not mean matched
  stochastic trajectories across these trainers.
- F1 minimizes the mean of two losses; the reference models optimize separately.
  With no shared parameters this creates no cross-branch gradient interaction,
  but scales each gradient by0.5. AdamW mostly cancels constant gradient scaling;
  this must not be described as simply halving the learning rate.
- F1 uses activation recomputation and a common validation-selection/stopping
  rule. Recomputation is restricted to the pure temporal conv/pool (not BN) in F1.

## Interpretation and smallest next experiment

The selection/stopping mismatch is real and systematic, but the 4.9pp loss has
not been causally assigned. F1 is a compatibility baseline, not a faithful
replication of the independently trained reference pair. Preserve F1–F6 as
completed exploratory evidence; do not label the full reference gap a sharing loss.

Before another grid, run only an instrumented F1 replay (initially one subject
for pipeline checking, then all five for a mean). Preserve its architecture,
initialization, batch/mixup and optimizer trajectory; save each branch's own
validation-best checkpoint plus the common-best checkpoint. Record virtual
common and independent patience20 stopping points while collecting the trajectory
through the chosen maximum budget. Evaluate the predeclared selection policies
only after validation selection; do not select a policy by test accuracy.

This first comparison isolates checkpoint/early-stopping policy within F1. It
does not isolate native architecture or RNG differences. A subsequent native
reference replication with matched RNG and independent selection is needed if
the selection-only comparison leaves a material gap. No such run was launched
as part of this audit.

## Inputs

- results/things_eeg/full_sharing_20260914/runs/F1/seed3300/sub-*/
- results/things_eeg/tsconv_bb128_valcon_20260911/runs/single/seed3300/sub-*/
- results/things_eeg/synthetic_subjects/ensemble_screen/dumps/atm_iv_valcon-sub*.npz
- results/things_eeg/honest_ensemble/atm_iv_valcon/seed3300/*-sub-*/
- ensemble_experiments/full_sharing_v2/{models,train}.py
- ensemble_experiments/tsconv_bb128_valcon.py and compact_valcon/train.py
