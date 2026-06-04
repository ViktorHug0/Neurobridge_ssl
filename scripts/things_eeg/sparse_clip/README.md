# Sparse CLIP scripts (THINGS-EEG inter-subject)

Training drivers and post-hoc sparsity / plotting utilities.

## Train sparsity at best checkpoint (full train set)

For each LOSO run with `checkpoint_test_best.pth`, measure post-ReLU active-feature fraction on **all training subjects** (same definition as `train.py` `collect_split_sparsity`).

```bash
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

SESSION=results/things_eeg/inter-subject-sparse/sparse_clip_confidence_bb64_seed3300_20260527-172056

# ~105 runs; writes incremental cache, then merges into sweep_summary.csv
PYTHONUNBUFFERED=1 python scripts/things_eeg/sparse_clip/measure_session_train_sparsity.py \
  --session_dir "$SESSION" \
  --device cuda:0 \
  --skip_existing
```

Outputs:

- `$SESSION/train_sparsity_at_best.csv` — per-subject train sparsity
- `$SESSION/sweep_summary.csv` — updated with `eeg_active_feat_frac_train` (and image train columns)

Resume after interruption: re-run with `--skip_existing`.

## Regenerate confidence sweep figure (test + train active panels)

Requires `eeg_active_feat_frac_train` in `sweep_summary.csv` (from step above).

```bash
python scripts/things_eeg/sparse_clip/plot_confidence_sweep.py \
  --csv "$SESSION/sweep_summary.csv" \
  --output "$SESSION/confidence_sweep_summary.png"
```

## Other utilities

| Script | Purpose |
|--------|---------|
| `plot_detached_sparsity_accuracy.py` | Test sparsity vs accuracy for detached relu/gelu session |
| `analyze_sparse_alignment.py` | Per-run L0 + object-coherence on test embeddings |
| `inter-subjects-sparse-clip.sh` | LOSO training (bb64, ReLU projector) |
| `sparse_clip_logit_fd_sweep.sh` | feature_dim × logit cap × eeg_l2norm sweep |
| `sparse_clip_confidence_sweep.sh` | Detached confidence × fd × relu / relu_gelu |
