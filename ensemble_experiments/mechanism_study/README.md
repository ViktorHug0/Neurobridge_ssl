# Ensemble mechanism study — 2026-09-07

Persistent local experiment suite authorized after the ENSEMBLE_RECAP discussion.
Implementation is isolated here; historical checkpoints and training code are reused
without editing them. Results live in `results/things_eeg/ensemble_mechanism_20260907/`.

## Launch and monitor

```bash
bash ensemble_experiments/mechanism_study/launch_tmux.sh
# Submit the extra two cards once; do not resubmit while their workers are active:
sbatch ensemble_experiments/mechanism_study/workers_3080.sbatch
tmux attach -t eeg-ensemble-mechanisms
# Detach without stopping: Ctrl-b, then d.
cat results/things_eeg/ensemble_mechanism_20260907/status_worker*.json
tail -f results/things_eeg/ensemble_mechanism_20260907/logs/queue.log
```

GPU: RTX 3090 (24 GB), selected by UUID `GPU-bf704308-654f-501f-3a82-1f6b38a4fb18`
and remapped to `cuda:0`. CUDA and nvidia-smi numeric ordering differ on this host;
the launcher avoids numeric selection. The persistent queue leaves the T1000 unused.
Two additional RTX 3080 workers are allocated with `workers_3080.sbatch` (Slurm
array 1-2, maximum two cards). Slurm workers persist independently of SSH/tmux.
Worker 0 (3090) runs subjects 1/4/7/10, worker 1 runs 2/5/8, worker 2 runs 3/6/9.
One GPU task runs at a time on each card. The CPU score analysis runs on worker 0.
The tmux shell remains open after completion/failure. Interrupt the queue with Ctrl-C
in the attached session; detach with Ctrl-b d to leave it running. tmux survives SSH
disconnects, not machine reboots. Relaunch after an interruption to resume: detach/
close the old idle session before calling the launcher again. The queue lock prevents
two runners in this study. Never delete an active lock file to bypass it.

Each task has an append-only log; `status_workerN.json` identifies the active child PID.
`events_workerN.jsonl` records exit codes and durations. Three consecutive failures stop the
queue. Failed isolated jobs are recorded and other jobs proceed. Completed jobs are
idempotent. New controlled training checkpoints optimizer and RNG state every epoch;
an interrupted epoch is replayed. Existing result directories are not deleted.

## Predeclared experiments

1. **Distractor alignment.** Four fixed committees (three seeds, historical diverse
   triple, quartet, concept-validation quartet), ten subjects. Preserve the true score
   and every member's wrong-score multiset; align wrong ranks, independently permute
   them, or permute within ten strata of mean visual similarity. 100 randomizations
   per stochastic condition. Report true margins, exact distractor bonus, accuracy,
   all-wrong rescues, and a label-informed convex-weight oracle. These are explanatory
   interventions, not inference algorithms. Average randomizations within subjects
   before averaging subjects. Historical kernels are computed after image projection.
2. **Repetition count.** The historical diverse triple and three TSConv seeds on all
   ten subjects; counts 1, 2, 4, 8, 16, 32, 80. Ten deterministic draws (one for 80),
   nested counts and identical recording indices across members. Each prediction
   uses ONLY that count, not all 80 split into views. Frozen eval-mode normalization.
   Full-average scores must reproduce the audited dumps within 0.002 maximum absolute
   score error. Historical checkpoints are test-selected; member/rule choices are fixed.
3. **Architecture × target.** ATM and TSConv, InternViT layers 28 and 33, seed 3300
   only (user narrowed the plan before launch). `geometry28`/`geometry33` train the two architectures independently
   inside one process. Export all six fixed pairs of the four resulting models.
   Geometry diagnostics fit a fixed regularized logistic model on the 200-concept
   source probe gallery, then assess rescue prediction on the outer subject's test
   gallery. Compare baseline margin alone with margin plus own-minus-other kernel
   percentile. The predictor uses truth for explanatory features and is not routing.
4. **Temporal specialization.** TSConv with input samples [0,75) versus [75,150),
   trained with the same masks used at inference; all 250 tensor positions retained
   so architecture and parameter count remain equal. At 250 Hz these are 300-ms
   windows. Data have 250 samples; the original metadata retains a longer baseline
   time axis, so sample indices are the authoritative definition. Main preprocessing
   discards pre-stimulus samples. Full-window TSConv controls come from `geometry28`.
   Report full+early and full+late pairs. New same-window seed controls are deferred
   under the user's one-seed constraint; the historical seed controls remain available.
5. **Frozen/joint fusion.** `geometry28` is the independent beta=0 control;
   `joint_b030` adds deployed unique-gallery fusion loss. `frozen_b000` and
   `frozen_b030` load only the selected ATM branch from the matching geometry28 run,
   then train a freshly initialized TSConv with beta 0 or 0.3. All use the same source
   split, RNG seed, batch size, and epoch budget. This is a new matched comparison;
   it does not merge with the older, differently configured frozen-fusion sweep.
6. **Sharing.** TSConv targeting layers 28 and 33: independent encoders, shared
   temporal conv/pool/BN/activation stem, or fully shared encoder with separate
   alignment heads. A fourth arm widens the shared encoder to match the independent
   pair's total parameter count (within rounding). Shared features are computed once
   per batch, including BN; optimizer parameters are unique. Parameters and elapsed
   epoch time are logged. Parameter matching does not imply FLOP/latency matching.

## New training protocol

For outer subject h, validation subject is h % 10 + 1; the other eight subjects train
the models. A fixed permutation (seed 20260907) reserves 200 training concepts for
checkpoint selection and a disjoint 200 for mechanism probes. BOTH sets are excluded
from all eight training subjects. Each validation/probe gallery uses image index zero,
one image for each of 200 concepts, on the unseen validation subject. Only minimum
fused validation cross-entropy selects a contemporaneous pair checkpoint; no test
accuracy tie-break. Probe and outer test galleries are evaluated after selection.
Both members therefore use the same selected epoch, including independent controls.

All new arms use normalized input visual blocks, linear alignment heads of width 128,
backbone width 128 (except parameter-matched sharing), AdamW lr=3e-4/wd=1e-4,
50 epochs, batch 512, eight same-image source samples, pairwise SubjectMix alpha=0.5,
individual multi-positive contrastive loss and row-z uniform inference. Temporal
kernel=30/pool=51 for TSConv; ATM uses the maintained defaults. Encoder/projector
training uses BF16 autocast on all three GPUs so batch 512 fits the 3080. Parameters,
contrastive losses, cosine training geometry, and all evaluation remain FP32.
Member initialization seed is identical across target/window conditions. Training RNG
7330 is restored after initialization. Initialization seed 3300 and batch order
are held constant across the controlled arms.

The outer subject is not used to fit preprocessing, select epochs, choose arms,
learn weights, or stop training. There is no refit to nine source subjects. Thus
absolute accuracies are NOT directly comparable to the historical nine-source,
test-selected leaderboard. The 400 reserved concepts reduce training data further.
This is a mechanism replication protocol, not a claim of a new best base model.
Prior reuse of these ten subjects still makes this programme exploratory.

The fixed queue contains 100 paired training jobs (10 arms × 1 seed × 10 subjects)
and 10 repetition-evaluation jobs, plus CPU analysis, distributed across three GPUs.
It is a multi-day programme,
not a short sweep. Tasks are ordered by subject, then seed, then arm; no arm is
promoted or rejected based on interim scores. Parameter count and actual timing
should be used to assess cost. Automatic summaries always include fold counts;
partial means must not be compared as completed ten-fold results.

## Outputs and checks

`queue_manifest_workerN.json`: commands, GPU mapping, source hashes. `controlled/ARM/seedS/sub-H/`:
configuration, exact subject/concept split, parameter count, epochs.csv, atomic best/last
checkpoints, validation/probe/test embedding and score dumps, result.csv, completion marker.
`summaries/`: fold-level training results, predeclared cross-arm fusions, geometry probe
transfer AUCs, repetition curves. `score_analysis/`: historical interventions and geometry.

```bash
source .venv/bin/activate
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest ensemble_experiments.mechanism_study.test_mechanisms
python -m ensemble_experiments.mechanism_study.summarize
```

Tests cover label-rank preservation, an oracle exceeding member selection, temporal
information exclusion, shared-stem gradient/BN handling, frozen parameters/BN, and
the parameter-matched shared encoder. Separate real-data smoke directories are excluded
from the primary summaries.

Initial calibration on the RTX 3090: the heterogeneous pair took 64.29 seconds per
full epoch in FP32 versus 39.13 seconds with BF16 training (same batch 512 and data).
BF16 peak allocated memory was 2.47 GiB, peak reserved 4.98 GiB. These are one-epoch
timings, not guarantees for every arm. A 50-epoch heterogeneous run is approximately
33 minutes on the 3090 before startup/export overhead. The overall estimate is
roughly 24–36 hours across the three GPUs, subject to measured 3080 speed and I/O.
