# IV28-only matched control

Search: 6,134 saved config/train_config JSON files and 170 model_size files under
results/things_eeg yielded no matching IV28-only backbone896 run. Existing 896
families are joint, hybrid, electrode, distillation, Nash, or their smoke tests.

Train ten LOSO folds, seed3300. Instantiate the exact compact dual_head model,
then remove its IV33 EEG/image head and loss. Preserve initialization of the
backbone896 and surviving IV28 alignment128 head exactly. The discarded IV33
input block is loaded for compatibility but provably cannot affect the output.
No full independently trained model or teacher is involved.

Use full single-task IV28 loss (not half loss), as in the prior single-target
baseline; this removes competition but changes the surviving task's loss scale
relative to its half-weight joint contribution. Retain AdamW3e-4/wd1e-4,
pairwise raw EEG mixup alpha.5, FP32, batch1017, 131 steps/epoch, max100,
patience20. Select minimum IV28 concept-validation loss using the same held165
concepts/seed20260822; evaluate outer test only once after selection. Joint models
select on mean-head loss, so comparisons involve both training and checkpoint
selection effects, not a pure fixed-checkpoint gradient intervention.

The wrapper uses the unchanged compact trainer in-process. Folder/config arm
label dual_head is a compatibility detail; manifest identifies this as IV28-only,
and model_size alignment_dims is [128]. Existing active Nash source hashes remain
unchanged. The wrapper cannot write to the historical result root.

Preflight: exact surviving initialization/output match, IV33-input invariance,
correct single-task loss and gradients, real-data two-epoch bitwise resume audit.
Queue: one additional RTX3080, six CPUs, 28GB host RAM, Slurm persistence,
epoch-resumable and fold-idempotent. Source hash guard protects resumed runs.

Run: sbatch ensemble_experiments/iv28_control.sbatch
Outputs: results/things_eeg/iv28_control_20260909.

Compare individual IV28 head accuracy against ordinary joint and Nash joint
models. A solo advantage is consistent with task interference but does not alone
identify its cause; an absence of a gap weakens interference as the explanation.
