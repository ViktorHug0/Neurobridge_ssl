# G2 private temporal stems — 2026-09-16

Controlled change: construct exactly G2, deepcopy temporal stem A into stem B,
then evaluate each stem on the raw EEG independently. Initial stem weights are
identical; parameters and subsequent gradients are independent. Shared rank16
corrected readouts, private normalization and image heads, post-stem attention,
and all training/selection settings remain unchanged.

Implementation uses the existing FullPair F3 forward dispatch after constructing
G2/F4. It does not invoke the F3 constructor: F3/F4 forward dispatch differs only
in evaluating stem B rather than reusing stem A's output.

Five LOSO subjects, seed3300; pairwise mixup alpha0.5, alignment128; max100 epochs,
patience20, common strict fused source-validation top1 checkpoint, equal row-z
fusion. Existing atomic checkpoint/optimizer/RNG resume logic is reused.

Workers: local 513 subjects1/4; remote 514 subjects2/5; remote 515 subject3.
Each worker holds one GPU and processes its folds sequentially. At most three
GPUs total, including the existing local interactive allocation.

Primary comparison: G2 mean40.5%. Diagnostic prediction: private stems increase
both-wrong rescues and distractor-disagreement margin bonus without degrading
branch accuracy. Track coverage, harmful fusion, identical-wrong predictions,
and branch accuracy as well. No tuning on the outer-test scores. All five subjects
have already informed research choices; new-subject confirmation remains needed.

Driver: `ensemble_experiments/run_fusion_private_stems.sh --worker 0` (or1/2).
Original interrupted outputs preserved: `results/things_eeg/fusion_private_stems_20260916/`.
Relaunch outputs: `results/things_eeg/fusion_private_stems_testmon_20260916/`.
Restart all folds at epoch1 to obtain complete diagnostic test curves. Every
epoch now records fused test top1/top5 and branch top1 in `epochs.json` under
`test`. Evaluation preserves model state, train/eval mode, and all training RNG
states. Test metrics never affect checkpoint selection, stopping or optimization.
The trainer remains `fusion_three/train.py`; no replay snapshot is imported.
Root `train.py` independently already logs per-epoch test metrics to TensorBoard;
it now also writes atomic `epoch_metrics.csv` including test and validation
loss/accuracy. This ablation does not switch to its different single-encoder loop.
Preflight checks identical initial outputs, independent storage, branch-specific
perturbations and gradients. Full production epochs verify full-batch memory.
