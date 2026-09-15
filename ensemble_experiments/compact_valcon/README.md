# Compact ValCon study — 2026-09-08

## Question and fixed design

Can one shared EEG network retain heterogeneous ensemble gains at approximately
single-model training cost? User authorized three designs, one seed (3300), ten
LOSO folds, and three RTX 3080 GPUs. Latest user constraint: **pairwise mixup only;
128-dimensional alignment for every head**. No teachers or additional training arms.

| Design | EEG backbone output width | Outputs | Total parameters |
|---|---:|---|---:|
| `single` | 1024 | IV33, align128 | 3,131,081 |
| `dual_head` | 896 | IV33 + IV28, align128 each | 3,213,770 |
| `hybrid` | 896 | local-conv IV33 + attention IV28, align128 each | 3,230,290 |

All use the maintained TSConv parameterizable front end: 40 temporal/spatial/
projection filters, temporal kernel30, pool51/stride5, dropout0.5, 63 channels,
250 samples. The two-head models use a narrower dense tail to pay for their extra
image/EEG heads. Total parameters are within 3.2% of the single model; these are
budget-matched, not identical-width, comparisons. Dual versus hybrid DOES hold
the trunk width fixed. Count includes visual projectors and fixed loss scales;
also report trainable and EEG-only parameters. Unused historical text projectors
are not instantiated or counted.

Hybrid: common temporal/spatial convolutions -> 35 temporal tokens of width40 ->
two small residual branches -> shared dense-tail weights -> separate align128 heads.
Local branch: LayerNorm, depthwise temporal conv3, GELU, pointwise conv.
Global branch: learned temporal position, LayerNorm, four-head self-attention,
and width80 feedforward residual. Attention here is over TEMPORAL TOKENS, not the
electrode attention in ATM. This is an inexpensive local/global hypothesis, not
an exact algebraic compression of ATM. The costly convolutional front end runs
once; the shared dense tail is evaluated separately for each hybrid branch.
Dual-head computes its entire backbone only once.

## Training and selection

- Train on all nine source subjects. The outer subject is loaded only after
  checkpoint selection finishes; no transductive adaptation, test-loss selection,
  test tie-breaks, test-based early stopping, or teacher supervision.
- Hold out 10% of the 1654 training concepts: 165 concepts chosen by NumPy
  default_rng(20260822), exactly as the original ValCon runner. Every image and
  every source subject of these concepts is excluded from training.
- Historical validation ordering and batches are reproduced: flatten image
  groups in insertion order, batch200, all source recordings, same-image
  multi-positive bidirectional CE, equally averaged over validation batches.
- Select the minimum validation loss, averaged across heads in multi-output
  models. This is a single contemporaneous checkpoint, not separately selected
  epochs per head. Early stop after20 epochs without improvement, maximum100.
- Batch1024 requested, effective1017 =113 images x9 source subjects. Drop final
  incomplete training batch. Sampler seed3300, same batches across designs.
- Pairwise same-image, cross-subject raw EEG mixup, alpha0.5, probability1.
  Mixup has an architecture-independent RNG stream; same mixed EEG across arms.
- AdamW lr3e-4, weight decay1e-4, no scheduler. Mean of individual-head losses;
  no explicit decorrelation or fused loss. Fixed softplus temperature from0.07.
- Preserve historical asymmetric loss normalization: EEG unnormalized for
  contrastive loss, projected images normalized; cosine evaluation normalizes both.
  BOTH raw visual blocks are kept raw before their projectors, matching solo runs.
  The generic dataset auxiliary-block normalization is deliberately bypassed.
- FP32 throughout. No microbatching or gradient accumulation; preserve negatives
  and BatchNorm batch size. All heads align to128 dimensions.
- Single-seed exploratory comparison; LOSO training sets overlap. Do not treat
  folds or queries as independent training replications.

The historical 41.40% reference is `atm_iv_valcon + iv33g_valcon`, with fixed row-z
uniform fusion. Its ATM used pairwise/align128/backbone128, but its TSConv used
GROUP mixup/align512/backbone1024. Therefore 41.40% is an engineering target, **not
an exactly recipe-matched comparator** under the user's latest restrictions.
The historical pair was chosen across the ten folds. No claim of fresh untouched
confirmatory evaluation is appropriate. Its checkpoints are reused for analysis,
not retrained or used as teachers.

## Outcomes and compute gate

Primary: mean200-way outer top1 of each fixed design. Fixed score rule: cosine
per head, per-query standardization (population SD), uniform mean; ordinary cosine
ranking for the single head. No learned weights or post-hoc member search.
Report paired fold changes, top5, head solos, score/margin diagnostics, reference
correct-query retention and all-members-wrong rescues retained. A mean at least
41.40% meets the numerical target; one seed cannot establish statistical equivalence.

Also report same-GPU synchronized training step time, batch200 EEG inference
latency, peak CUDA memory, epoch training/validation times, completed epochs, and
queue wall time including startup/checkpoint/I/O. Do not substitute parameter count
or FLOPs for actual speed. Frozen microbenchmark gate: each new design <=1.05x
single parameters and <=1.20x max(single TSConv, reference ATM) training step time.
Reference ATM is a timing-only model, never another training task. Cached visual
gallery features need not be reprojected for per-query inference.

## Execution and persistence

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m ensemble_experiments.compact_valcon.test_designs
python -m ensemble_experiments.compact_valcon.benchmark
python -m ensemble_experiments.compact_valcon.verify_resume
python -m ensemble_experiments.compact_valcon.run_queue --prepare
bash ensemble_experiments/compact_valcon/launch_local.sh
sbatch ensemble_experiments/compact_valcon/workers.sbatch
```

Worker0 runs inside the existing allocation on sl-tp-br-513, in tmux
`compact-valcon`. Workers1/2 are persistent Slurm array tasks, requesting exactly
two further3080s and excluding513. Thirty training tasks are distributed ten per
worker, rotating arm/subject assignments to avoid assigning each design to a
different GPU. One active training process per GPU. The two Slurm workers survive
SSH disconnects; local tmux survives disconnection but NOT termination of its
enclosing interactive Slurm allocation or a node reboot. Do not cancel job52517
while worker0 is running.

Checkpoints include optimizer, RNG, patience, and authoritative epoch history.
An interrupted epoch is replayed, not appended as a duplicate. Per-run and
per-worker OS locks prevent simultaneous writers. Complete tasks are skipped.
The launch manifest freezes source SHA256s; queues stop before a new task if
source files change. Do not edit running study code. Relaunch the same worker
after its previous process exits to resume; do not submit duplicate live workers.

Monitor:

```bash
tmux attach -t compact-valcon
cat results/things_eeg/compact_valcon_20260908/status_worker*.json
tail -f results/things_eeg/compact_valcon_20260908/logs/worker0.log
squeue -u p20fores
```

Results: `results/things_eeg/compact_valcon_20260908/`. Each run has config,
split, model size, best/last checkpoints, epoch JSON, test embeddings/scores,
and completion JSON. `summary.csv`, `folds.json`, and `paired.json` refresh after
each task under a summary lock. Incomplete fold means are explicitly labeled
by fold count; do not compare unpaired partial means. `smoke_v*` is excluded.

Prelaunch validation: all five unit tests passed, all three designs completed
real-data smoke training/validation/test exports, all three concept splits matched
(134010 training /14850 validation items), and an actual interrupted hybrid run
resumed with bitwise-identical final AND selected weights to an uninterrupted run.
FP32 benchmark on513 (12 timed steps after3 warmup): single161.2ms,
dual165.9ms, hybrid172.2ms, ATM281.8ms; peak allocations6.40/6.40/6.41/8.44GiB.
