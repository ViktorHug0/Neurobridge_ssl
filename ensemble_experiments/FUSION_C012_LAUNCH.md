# C0/C1/C2 matched follow-up — 2026-09-15

User requested all three overnight, cancellation of G3, and at most three
concurrently allocated GPUs. G3 task 53581_3 was canceled; no outputs deleted.

| Arm | Architecture | Execution |
|---|---|---|
| C0 | F4 with private dense readouts, current G2 selector | tmux `fusion-c0`, local 513 / allocation 52739 |
| C1 | G2 rank16 first dense corrections; private residual dense | Slurm 53655, node 515 |
| C2 | G2 rank32 corrections on both dense layers | Slurm 53656, node 514, afterok:53581_2 |

C2 additionally checks that all five G2 completion files exist before training.
If G2 fails, C2 intentionally stays blocked instead of bypassing the dependency.

All arms: held-out subjects 1–5, seed 3300, pairwise mixup alpha 0.5,
128D alignment, 100 epochs maximum, patience 20. Reuse the unchanged
`fusion_three.train.run` and recipe, with process-local model/output substitutions.
Selection is strict maximum fused source-validation top1; ties retain the earlier
checkpoint and consume patience. Report outer-test accuracy at that checkpoint,
not the best test epoch. Equal row-z fusion, private IV33/IV28 image heads.

Outputs: `results/things_eeg/fusion_c012_20260915/`.
Preflight records source hashes and checks exact initial evaluation equivalence to
G2/F4, parameter-sharing identities, 128D outputs, finite GPU backward gradients,
and shared-gradient diagnostic compatibility. Training uses the existing atomic
checkpoint/RNG resume mechanism; a separate bitwise-resume test was not rerun.

The existing G2 source files were not changed. Workers load code once and run
five folds sequentially. Restart with
`bash ensemble_experiments/run_fusion_c012.sh --arm C0` (or C1/C2), only when
the assigned worker has stopped and GPU budget permits; per-device and per-arm
locks prevent duplicate workers within this experiment.
