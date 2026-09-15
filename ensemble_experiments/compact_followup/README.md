# Compact follow-up: electrode attention and ranking distillation

## Recovery — 2026-09-09

After the cluster interruption, the user authorized two RTX3080 GPUs. Electrode
remains on513; distillation resumes on a second3080 via `distill_3080.sbatch`,
requesting28GB host RAM and8CPUs (these nodes have31GB, so the old40GB request
would not fit). The A100 script below documents the original launch only.
Both arms completed folds1–2. Electrode fold3 resumes after epoch50; distillation
fold3 after epoch38. Folds4–10 had not started. All model/training source hashes
matched the frozen manifest before recovery. No completed results are overwritten.

The pre-recovery manifest, statuses, runtime metadata, and checkpoint/completion
hashes are preserved under `recovery_20260909/` in the results directory. The active
manifest is refreshed only for this launch script and documentation. Distillation
fold3 mixes A100 and3080 training; its first38 epochs remain the A100 trajectory.
Resume restores optimizer/RNG/early-stopping state, but cross-GPU numerical identity
is not promised. Preserve this distinction in compute and reproducibility reports.
Local tmux still depends on the current interactive allocation (now52739), not
the failed52517 allocation mentioned in the original launch notes below.

Current relaunch commands (after checking no old worker holds the locks):

```bash
bash ensemble_experiments/compact_followup/launch_local.sh
sbatch ensemble_experiments/compact_followup/distill_3080.sbatch
```

## Authorized scope

Two experiments, ten LOSO folds each, seed3300. Latest user instruction:
**electrode folds on the RTX3080 on sl-tp-br-513; distillation folds on one A100**.
Each arm has its own sequential ten-fold queue; they run in parallel on the two
GPUs. The A100 Slurm job requests40GB host RAM and8CPUs (the earlier data-loading
workers used about14GB RAM, leaving room for prefetch, teacher loading, and cache).
Do not change the preceding compact study. The original single-GPU manifest and
status are preserved as `manifest_single_gpu.json` and `status_before_split.json`.

Fixed constraints: pairwise cross-subject same-image raw-EEG mixup alpha0.5,
alignment128 for ALL heads, FP32, original ValCon165-concept split seed20260822,
nine source subjects, batch1024 (effective1017), maximum100 epochs, patience20.
AdamW lr3e-4, weight decay1e-4, no scheduler. Source validation alone selects the
minimum mean-head supervised contrastive loss. Test data are loaded only after
selection, except a prelaunch fixed-teacher archival reproduction check which
does not choose architecture or hyperparameters.

The unchanged source data, indexing, mixed-EEG RNG streams, loss normalization,
validation batches and retrieval routines come from `compact_valcon/train.py`.
The existing `dual_head` mean35.25% is the fixed compact baseline. The existing
ATM-IV28 + NEW single TSConv-IV33 pair averages40.20% with row-z fusion: BOTH
teachers have pairwise mixup and alignment128. This is the appropriate engineering
target, rather than the historical41.40% pair with group/align512 TSConv.

## Experiment 1: `electrode` — from scratch, no teacher

The prior hybrid applied temporal attention after electrode collapse. This model
instead shares the temporal convolution/pooling/BN/ELU (40 filters, kernel30,
pool51/stride5), retaining a tensor of shape B x40 x63 electrodes x35 time bins.
It produces two spatial views:

1. Ordinary view -> shared spatial convolution -> private BN -> IV33 head.
2. Electrode-attended residual view -> SAME spatial convolution -> private BN -> IV28 head.

The attention matrix is computed from four temporally pooled bins per electrode:
LayerNorm160 -> Linear160-to32 -> GELU, learned electrode-position embeddings,
and separate32-wide queries/keys. Softmax acts over electrodes. VALUES retain
all40x35 features of each electrode, so the descriptive pooling does not discard
the temporal signal sent to the spatial projection. Residual coefficient is
tanh of a learned scalar initialized0.1. No montage coordinates are needed.

Both paths share projection and dense-tail weights (backbone896), but evaluate
those inexpensive stages separately. Their spatial BN states and alignment128
heads are separate. The expensive temporal convolution executes once. Parameters:
3,223,387 total, versus3,213,770 for the prior dual-head and3,131,081 single model.
Train from scratch, mean of the two supervised losses, no extra fusion objective.
This is a test of inexpensive input-dependent electrode mixing, not a claim of
exactly implementing/compressing ATM or preserving all of its mechanisms.

## Experiment 2: `distill` — unchanged student, frozen existing teacher pair

Student is EXACTLY `CompactDecoder('dual_head')`, including initialization,
backbone896, two128 heads, parameters3,213,770. Train from scratch, not from the
completed student's checkpoint. The ONLY intended objective change is:

`loss = mean(supervised_head_losses) + KL(teacher_distribution || student_distribution)`

Fixed KL weight1.0, temperature1.0, no hyperparameter search. For both student and
teacher: cosine scores per head -> per-query population-standardization -> mean
across heads -> softmax. Distill query-to-image distributions over UNIQUE images
in the current source batch (normally113), not a duplicated1017-column gallery.
Do not average incompatible teacher embedding coordinates. Teacher scores are
detached; teacher modules are eval/frozen and are NOT in the student's optimizer
or checkpoints. Validation selection excludes KL and teacher predictions.

Teacher provenance, separately for each outer fold:

- `honest_ensemble/atm_iv_valcon/seed3300/*-sub-XX/checkpoint_test_best.pth`
- `compact_valcon_20260908/runs/single/seed3300/sub-XX/best.pth`

Check the source subjects, concept split, pairwise recipe, alignment128 and saved
checkpoint hashes. Frozen teacher forward passes run ONLINE on the EXACT SAME
mixed EEG tensor and image gallery given to the student. No interpolated or
stale soft targets, no extra unmixed student pass, no finite mixed-view catalog.
Teachers run in no-grad chunks128 to fit10GB, preserving full student batches,
student BN statistics, and negatives. ATM has a legacy batch-level unknown-subject
token rule; reproduce that rule globally before chunking to avoid changed teacher
predictions when a chunk does not contain subject10.

Teacher learning used source training concepts, so their training predictions may
be overconfident; failure of this KD configuration does not establish that the
student lacks capacity. Conversely success would show compression is possible,
not that training without teachers is equally easy.

Cost accounting: log synchronized teacher-forward time separately within total
epoch training time. It is a SUBSET, not an additional cost to add a second time.
Student parameter counts exclude teachers; report teacher parameters separately.
Existing teacher training is a sunk cost for this run, but must be added when
claiming end-to-end from-scratch efficiency. There is no teacher at inference.
No test/validation examples provide distillation targets, no teacher refits.

## Measurements and prelaunch checks

Primary: fixed mean200-way test top1 over ten folds; paired differences against
the existing dual-head baseline and fixed40.20% pair. Also top5, head solos,
reference correct-query retention, per-epoch supervised/KD/validation losses,
selected/stopping epochs, parameter counts, inference latency, peak memory,
teacher time, training/validation time and total task wall time including I/O.
One seed, overlapping LOSO training sets: exploratory, not proof of equivalence.

Same513 FP32 benchmark, eight timed steps after three warmups:

| Arm | Step incl. teacher | Teacher subset | Peak allocated | EEG inference, batch200 |
|---|---:|---:|---:|---:|
| electrode |206.3ms|0|7.59GiB|7.13ms|
| distill |229.7ms|64.5ms|6.43GiB|5.02ms|

These both fit the fixed gate: <=1.05x single-model parameters and <=1.20x the
previously measured max(TSConv, ATM) step time (ATM281.8ms). Inference timing is
EEG-side with cached projected gallery embeddings, not raw-image feature extraction.
Teacher benchmark allocation is included conservatively even for the electrode
preflight; the production electrode run never instantiates teachers.

Unit tests cover gradients, shapes, shared temporal computation, private BN,
parameter budget, exact student initialization, checkpoint roundtrip, teacher
gradient isolation in KL, and zero self-distillation loss. `preflight.py` verifies
teacher retrieval against saved outputs and chunking: max score errors below2e-5.
`verify_resume.py` runs real EEG for both arms, interrupts after an epoch checkpoint,
and checks bitwise-identical final and selected weights after resume.

## Launch and monitoring

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m ensemble_experiments.compact_followup.test_followup
python -m ensemble_experiments.compact_followup.preflight
python -m ensemble_experiments.compact_followup.verify_resume
python -m ensemble_experiments.compact_followup.run_queue --prepare
bash ensemble_experiments/compact_followup/launch_local.sh
sbatch ensemble_experiments/compact_followup/distill_a100.sbatch
```

Preparation freezes code/README SHA256s and all20 teacher checkpoint hashes. Do not
edit the study or shared imported modules while the queue runs: it stops before
the next task if code changes. Run and queue locks prevent simultaneous writers.
Epoch checkpoints retain optimizer, all RNG states, patience, authoritative history,
and selected weights. Completed tasks are skipped; interrupted epochs replay.
Smoke outputs are excluded from summaries. Existing outputs are never deleted.

```bash
tmux attach -t compact-followup
cat results/things_eeg/compact_followup_20260908/status_electrode.json
cat results/things_eeg/compact_followup_20260908/status_distill.json
tail -f results/things_eeg/compact_followup_20260908/logs/queue_electrode.log
```

There are20 tasks: electrode subjects1..10 on513, distill subjects1..10 on A100.
Logs and epoch JSON give live progress; `summary.csv` and `folds.json` refresh
under a shared lock after each task. `runtime.json` records each run's actual GPU
and allocation. Same-GPU preflight timings remain the fair compute comparison;
do NOT directly attribute the A100-vs-3080 wall-time difference to the method.
Partial means need matched subjects before comparison. Outputs are isolated under
`results/things_eeg/compact_followup_20260908/`.

Tmux survives SSH disconnection, but not termination of its enclosing interactive
Slurm allocation (currently52517), node failure, or reboot. Keep that allocation
alive. The A100 worker is a persistent Slurm batch job and does not depend on
that interactive allocation. After an interruption, restart only after the
previous worker exits; reuse the frozen manifest and appropriate arm argument.
