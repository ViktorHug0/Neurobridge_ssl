# F1 checkpoint-policy replay

Five subjects (1–5), seed3300; original F1 architecture, initialization, AdamW,
pairwise mixup alpha0.5, raw IV33/IV28 inputs, FP32, 128D alignment and grouped
effective batch1017. The frozen F1 sources are hash-checked. No original outputs
are edited. New output: `results/things_eeg/f1_checkpoint_replay_20260915`.

## Predeclared policies

All policies use **fixed equal row-z fusion**, with checkpoint selection based
only on source-concept validation loss. No calibration or test-selected weights.

| Policy | Selection | Available trajectory |
|---|---|---|
| common_original | Minimum mean of both branch losses, one common checkpoint | Original common patience20 stopping point, capped100 |
| independent_same_horizon | Minimum individual loss for each branch | Same common stopping point as above |
| independent_early | Minimum individual loss, frozen at each branch's own patience20 stop | Individual stopping points, capped100 |
| common_100 | Minimum mean branch loss | Full100 epochs |
| independent_100 | Minimum individual loss for each branch | Full100 epochs |

Primary contrast: independent_same_horizon minus common_original isolates
selection under a matched training horizon. Comparing independent_early or
independent_100 with independent_same_horizon measures effects of the available
trajectory/stopping policy. common_100 versus independent_100 controls compute
budget while comparing selection. None of these selects a policy on test scores.

The physical run continues both branches through100 epochs. Early stopping is
**virtual**: it freezes the relevant selector, not the live optimizer/model.
This preserves both branches' RNG consumption and the original F1 trajectory.
It is not a claim to reproduce two separately launched solo-training trajectories.
TS and ATM in F1 have no shared weights. Full model snapshots at selected epochs
are retained; evaluation uses branch0 from the selected TS epoch and branch1
from the selected ATM epoch, including each branch's corresponding image head.

## Checkpoints, verification and recovery

`last.pth` atomically stores model/optimizer/RNG/history and all current/frozen
policy-best model snapshots every epoch. Resume restores these snapshots and
replays any interrupted partial epoch. `epochs.json` records losses and virtual
stop/selection epochs. At the end, `selection.json` is written **before** loading
outer-test data; named policy `.pth` checkpoints and `*_scores.npz` files follow.
`complete.json` reports all predeclared policies, not just the most accurate one.

Preflight compares the original and instrumented trainers on real full batches
for bitwise model/optimizer/RNG/loss equality, checks interrupted/resumed policy
state, tests synthetic early-stopping behavior, and exercises all score exports.
Short preflight test accuracies are not experimental results.

Launchers use expandable CUDA allocation and one process per3080. The local
tmux worker handles subjects1,3,5; Slurm handles2,4 with29GiB RAM. Restart the
same launcher after interruption; complete folds are skipped. `summary.json`
contains only complete production folds. With roughly69seconds/epoch, the
three-fold worker is expected to take about6hours (allow additional I/O/startup).

Entry points: `f1_checkpoint_replay.py`, `f1_checkpoint_replay_preflight.py`,
`f1_checkpoint_replay_local.sh`, `f1_checkpoint_replay.sbatch` in this directory.

Launch: tmux `f1-checkpoint-replay-513` on513; Slurm job53560 on514.
Monitor `status_worker0.json`, `status_worker1.json`, and per-subject logs under
the new output root. Both workers depend on the passed preflight source hashes.
