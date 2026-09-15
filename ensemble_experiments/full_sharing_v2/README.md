# Repaired full-size sharing experiment — September 14, 2026

This package supersedes the failed `../full_sharing.py` queue. Historical outputs
remain untouched. Results: `results/things_eeg/full_sharing_20260914/`.

## Grid

Thirty runs: F1–F6 × held-out subjects 1–5, seed 3300. Each fold still trains on
the other **nine** subjects from 1–10. Both alignment and backbone readouts are
128D; temporal stems retain 40 filters and ATM attention retains width 250.

| Configuration | Mechanism | Unique total parameters |
|---|---|---:|
| F1 | Independent stems; native attention before ATM convolution | 1,898,434 |
| F2 | F1, temporal convolution weights tied, but computed twice | 1,897,194 |
| F3 | Independent raw-EEG stems; electrode attention after ATM stem | 2,543,415 |
| F4 | F3, temporal stem shared and computed once | 2,542,175 |
| F5 | F4, also tie readout dense weights; private adapters and normalization | 2,346,335 |
| F6 | One TS backbone computed once; private residual/attention branches | 1,586,775 |

Private EEG alignment and image heads remain in every configuration. F3–F5
include identity-initialized private adapters in the controls as well as the
shared variants. Their post-stem descriptor projections increase parameter
count: they are mechanism controls, not assumed parameter-efficiency wins.
F6 attention operates after spatial pooling and therefore tests a more
aggressive change than merely tying weights.

F1/F2 start with identical functions, as do F3/F4 and F4/F5. Sharing aliases
survive checkpoint loading, and parameter counts count each parameter once.
F1 is a compatibility control, not an exact reproduction of archived ATM:
its temporal kernel is 30 instead of 25, and the two independent stems start
with identical weights. Native ATM readout columns are permuted to preserve
its function under the common channel-major flatten layout.

## Protocol and safety

- Pairwise cross-subject mixup, alpha 0.5; grouped effective batch 1017.
- FP32; AdamW 3e-4, weight decay 1e-4; maximum 100 epochs, patience 20.
- Source-concept split is checked against each completed TS128 reference.
- Minimum mean branch validation loss selects the checkpoint. Reported outer
  test accuracy is **not** maximum test accuracy across epochs.
- Raw IV33/IV28 image inputs; actual subject IDs and native subject-token policy.
- Primary retrieval: equal-weight row-z score ensemble. A secondary weight is
  selected on source-concept validation and saved before outer test loading.
- Non-reentrant activation checkpointing excludes BatchNorm. It permits the
  full contrastive batch on a 3080 without changing the loss batch size.
- Both launchers enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Cold-process launches with the default allocator exhausted contiguous free
  memory despite substantial unused reservations; short warm-process smoke
  checks alone did not catch this. Initial failed launch logs are retained.
- Atomic epoch checkpoints include optimizer and RNG state; a partial epoch
  replays on restart. No automatic relaunch after a worker error is implied.
- Source hashes and immutable manifests prevent silently mixing code versions.
- Completed baseline checks, per-worker/device locks, and disjoint task lists
  prevent the earlier dependency race and duplicate runs within this queue.

Preflight passed 24 checks, including all six full-batch forward/backward tests,
native-layout checks, shared-computation counts, and bitwise interrupted/resumed
training comparisons for F1, F4, F5, and F6. Peak allocated memory in these short
checks was approximately 5.6–7.7 GiB; these are not full-training benchmarks.

## Running and monitoring

Activate the project `.venv` before Python. Entry points:

```bash
python -m ensemble_experiments.full_sharing_v2.queue --prepare
bash ensemble_experiments/full_sharing_v2/local.sh
sbatch ensemble_experiments/full_sharing_v2/worker.sbatch
```

The local worker is launched in tmux `full-sharing-v2-513`. The Slurm worker
requests one additional 3080, excludes node 513, and requests 29 GiB RAM.
Each worker has 15 tasks. Inspect `status_worker{0,1}.json`, per-fold logs,
`runs/F*/seed3300/sub-*/epochs.json`, and the incrementally updated
`summary.json` under the result root. Restart the same worker to resume its
interrupted fold and skip completed folds. Do not launch the legacy scripts.

### Third GPU added September 14

Slurm job 53506 on node 515 runs `ensemble_experiments/full_sharing_third.py`
via `full_sharing_third.sbatch` (one 3080, 29 GiB RAM). This external supervisor
does not change frozen model sources or interrupt the two existing workers.
It processes the last five tasks of each original queue in reverse order.
Original workers skip the resulting completed folds. A guard stops the helper
and leaves an epoch-resumable checkpoint if either original worker reaches
index 9, before the assisted tail begins at index 10. Therefore the original
manifest remains authoritative and no unfinished task loses its original owner.
Monitor `status_worker2.json`, `third_worker_plan.json`, and `worker2-53506.log`.
