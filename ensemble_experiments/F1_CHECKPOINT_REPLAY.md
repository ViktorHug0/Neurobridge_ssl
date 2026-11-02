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

## Three-GPU recovery

The original queues completed subjects1 and2, then halted on source hashing:
commit2ebb195 added unrelated foundation-encoder support to live `train.py`.
The original source is available in commit34292e1 and matches the preflight
SHA256 `22e3eb1d166c58c99a747d63f9b6a7e427c47317c57ca6391fb89515874872fb`.
No completed results or live repository code were reverted.

`f1_replay_snapshot/launch.py` validates every original source and explicitly
loads `f1_replay_snapshot/train_original.py` as the imported `train` module.
The snapshot matches that original hash exactly. Logical experiment provenance
continues to describe the executed original sources; separate physical paths,
live-source hashes and launcher hash are saved in `relaunch_provenance.json`.
The launcher rejects any further physical-source change after that approval.

The recovery assigns one fold per3080: subject3 in tmux `f1-replay-sub3-513`,
subjects4 and5 as separate Slurm array tasks. Monitor
`status_relaunch_sub{3,4,5}.json` and `relaunch-sub-*.log`; original worker status
files refer to the stopped old queues. The same production fold directories,
resume protocol and five predeclared selection policies are retained.
