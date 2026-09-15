# Tiny TSConv + ATM controlled sharing study — implementation

Implements `ensemble_experiments/tiny_sharing_plan.md`. Self-contained: nothing
here is imported by `train.py`, the Nash runs, or the IV28 control.

| File | Role |
|---|---|
| `models.py` | the ten configurations C0–C9 as one `PairedModel` |
| `data.py` | inner-fold splits, grouped pairwise-mixed batches, fixed galleries |
| `train_pair.py` | one condition × one screening fold: train, select, measure |
| `gate.py` | Phase A implementation gate (CPU, no EEG) |
| `analyze.py` | Phase B deliverables: table, contrast graph, decompositions |
| `smoke.sbatch` / `screen.sbatch` | Phase A on-GPU check / Phase B, one RTX 3080 |

```bash
source .venv/bin/activate
python -m ensemble_experiments.tiny_sharing.gate      # 62 checks, ~40 s, CPU
sbatch ensemble_experiments/tiny_sharing/screen.sbatch
python -m ensemble_experiments.tiny_sharing.analyze
```

Results land in `results/things_eeg/tiny_sharing/screen/slot{1,2,3}/C{0..9}/`
(`manifest.json`, `summary.json`, `metrics_best_mean*.json`, `scores_*.npz`,
`best_mean.pth`, resumable `last.pth`).

## Decisions the plan left to the implementation

### September 11 repair/relaunch

**Two-GPU update:** job 53110 was intentionally stopped at an epoch checkpoint
to repartition the queue. Worker A is now Slurm job **53116** on 515 (all slot 1,
then slot 2 C0–C4); worker B is tmux session **tiny-sharing-513** on 513 (all slot
3, then slot 2 C5–C9). B waits for A's baseline gate to pass. Each owns 15 runs;
there are no overlapping assignments. Both use the same versioned result root,
frozen experiment source hash `5383ed884ef6806c`, and deterministic settings.
Logs: `worker-A-53116.log` and `worker-B-513.log` under the tiny_sharing result root's
parent directory. The original single-worker launcher below is historical;
do not submit it while these two workers run.

Use `sbatch ensemble_experiments/tiny_sharing/relaunch.sbatch` for the repaired
screen (current job **53110**, node **sl-tp-br-515**). It runs independently of
the terminal, on one additional RTX 3080, with 29 GB host RAM and a 48-hour limit.
Results are isolated in `results/things_eeg/tiny_sharing/screen_20260911_v2/`;
the previous failed screen is preserved. Do not submit a duplicate while it runs.

The job first runs the structural gate, real-data full-batch training/evaluation
for C0–C9, and exact interrupted-versus-continuous resume tests for C0/C7/C9.
Then it trains slot 1 C0/C1 for 40 epochs. The remaining sequential screen is
blocked if C1 loses more than 2 percentage points of fused validation accuracy
or more than half C0's positive gain over its best branch. See `bridge_gate.json`.
Otherwise all ten configurations run on each of three inner folds, one seed,
pairwise mixup and 128-dimensional alignment. These are inner-validation
screening results, not 10-fold outer-test accuracy estimates.

Repairs include equal-scale fused margins, averaging all sampled gradient
diagnostics, atomic checkpoints/artifacts, CPU and CUDA RNG restoration,
checkpoint-authoritative best artifacts, and refusal to resume incompatible
source/manifests. Real-data preflight uses only two training batches per epoch;
the actual screen uses every full batch. `preflight/passed.json` certifies the
tests only after they all finish successfully. Initial repair job 53108 passed
the structural gate and C0 real-data test, then stopped because CUDA adaptive
pooling backward cannot run under strict deterministic mode. The bridge now
uses the same adaptive averaging bins expressed with slice/mean operations;
this preserves the pooling definition and enables strict resume verification.

```bash
squeue -j 53110
tail -f results/things_eeg/tiny_sharing/relaunch-53110.log
python -m ensemble_experiments.tiny_sharing.analyze --help
```

**Bridge parameter count.** The plan's analytic bridge figure of 430,388 counts
`ATMS.subject_wise_linear`, 56,500 parameters the ATM forward never touches. The
bridge here builds its attention front from `iTransformer` directly and does not
carry them, so the instantiated count is **373,888**; `373,888 + 56,500 =
430,388` reconciles it. C0 instantiates at **416,756**, matching the plan's
native figure exactly. Both image projectors are 409,728 each and are always
reported separately. `gate.py` asserts both numbers.

**Subject token.** Native `SubjectEmbedding` switches the *whole batch* to its
shared token as soon as one id is out of range, so training composition and
evaluation chunking silently take different paths — the hazard the plan flags.
All three screening folds train with subject 10 present, so native training is
*already* on the shared token; only validation would have flipped to a
never-updated per-subject token. Every forward in every condition and phase is
therefore pinned to the unknown/shared token. `subject_embedding.weight` and
`mask_embedding` are then dead and are counted as unreachable, not trainable.
`gate.py` reproduces the native hazard rather than repairing it.

**Batching.** Batches are built in-process, not through a `DataLoader`, so the
exact grouped, pairwise-mixed EEG tensor is reproducible from `(epoch, step)`
alone and is byte-identical across all ten conditions. `_pairwise_mix` is a
vectorized restatement of `train.py`'s `cross_subject_stimulus_mix` pairwise
branch for groups of distinct subjects.

**Dropout streams.** Branch dropout is drawn from explicit per-branch
generators. Two exceptions, both documented rather than hidden: C0 uses the
native modules' own global-RNG dropout, and the C8/C9 electrode-attention block
uses `nn.TransformerEncoderLayer`'s internal dropout. Neither crosses a branch
boundary, and mixup is seeded per step independently of both.

**C7 normalization.** `PairedBatchNorm2d.pooled` evaluates both pre-normalization
activations, takes equally weighted pooled moments
(`(va+vb)/2 + (ma-mb)^2/4`), normalizes both paths with them and updates the
running state once, Bessel-corrected over the combined element count.
`eps=1e-5`, `momentum=0.1`. LayerNorm shares affine weights only; its statistics
stay per-sample.

**Validation loss chunks.** Retrieval uses the padded 200-candidate panels so
every query is scored exactly once; the checkpoint-selection loss uses the same
permutation without padding, so no validation row is counted twice.

## What is not built here

Phase C (confirmation LOSO) is deliberately absent. The plan freezes its 3–5
architectures **from inner validation only**, after the screen — building the
runner before that decision would be guessing at its config list.
