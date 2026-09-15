# Two-task Nash-direction experiment

## Fixed protocol

One shared TSConv dual-head model, identical to compact_valcon dual_head:
3,213,770 parameters, backbone896, two alignment128 heads (IV33 then IV28).
Pairwise raw EEG mixup alpha .5, seed3300, AdamW lr3e-4/wd1e-4, FP32,
effective batch1017, 131 steps/epoch, max100 epochs and patience20. The same
165 source-validation concepts (seed20260822) select minimum mean-head loss.
Outer test is evaluated once after checkpoint selection. No teacher or distillation.
Reuse the completed 35.25% dual_head LOSO baseline; no duplicate baseline training.

For shared backbone task gradients g1,g2, use direction
g1/||g1|| + g2/||g2||, rescaled to ||(g1+g2)/2||. This is the two-task
Nash direction, not the paper's full step-size prescription. AdamW remains in
place; no claim that its preconditioned/momentum update inherits common descent.
Private EEG/image projector gradients retain ordinary half-loss scaling. The
baseline's frozen temperature parameters remain frozen. Zero gradients (<1e-12)
or near-antiparallel normalized sums (<1e-6)
fall back to the ordinary mean; fallback frequency is logged. No frequency sweep.

Log per-step gradient cosine, norm ratio, conflict frequency, first-order task
improvements for ordinary mean and balanced directions, plus per-block cosine.
Every ten steps measure actual AdamW parameter displacement (including decay)
and its dot products with both task gradients. Epoch logs store means.
These are training-local diagnostics, not held-out improvement guarantees.

## Execution and verification

`python -m ensemble_experiments.nash_experiment` runs preflight, bitwise resume
audit, then all ten LOSO folds sequentially. Root:
`results/things_eeg/nash_direction_20260909`; persistent tmux: `nash-direction`.
Queue/process locks, completion guards, epoch checkpoints and RNG restoration
permit restart without rerunning completed folds. Code hashes guard queue resumes.
Partial epochs replay from the last complete epoch.

Preflight tests positive first-order improvements, invariance of direction to
positive task rescaling, degenerate fallbacks, norm preservation and agreement
of separately differentiated mean gradients with ordinary backward. Private
gradient equality is tested for the balanced rule too. Real-data same-GPU step
benchmarks compare the ordinary and balanced paths; an isolated two-epoch,
two-batch/epoch test checks interrupted/resumed final and selected weights exactly.

Training and validation reuse the existing compact trainer; its default mean-loss
path and existing configuration compatibility are preserved. Gradient diagnostics
and the new rule activate only with `--gradient-rule nash`, requiring a separate
output root. Existing run artifacts are not modified.

## Scientific interpretation

This isolates shared-gradient combination, not a new architecture or new targets.
An accuracy gain would support an optimization contribution to the shared-model
gap; improved heads without improved fusion would leave complementarity as a
candidate limitation. Norm matching controls raw gradient magnitude, not actual
AdamW displacement or the entire effective optimization trajectory. One seed
cannot establish robustness. Benchmark overhead and observed early stopping must
both inform the accuracy/compute comparison.
