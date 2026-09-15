# Maintained ensemble architectures

This directory contains the experimental EEG encoders that remain useful for
the ensemble program. Rejected architecture implementations and their dedicated
launch, gate, reporting, and maintenance scripts have been removed.

## Available encoders

| Encoder | Purpose |
|---|---|
| `FastTSConvSqueezeformer` | Full-size convolution/attention hybrid |
| `TinySqueezeformer` | Parameter-matched proxy for rapid ensemble studies |
| `MultiScaleTSMixer` | Full-size multi-resolution temporal-spatial MLP mixer |
| `TinyMultiScaleTSMixer` | Parameter-matched proxy for rapid ensemble studies |
| `TinyGatedChannelTransformer` | Attention-native channel-token control (arm A) |
| `TinyDifferentialChannelTransformer` | Tiny Channel A with differential attention |
| `TinyGraphDiffusionNet` | Sparse montage-graph diffusion with a virtual global node |

The public encoder names no longer use the historical `Ortho` prefix. The
loader still accepts it for historical maintained names so existing checkpoints
and experiment configurations remain reproducible.

## Files

- `ortho_encoders.py` contains the maintained encoders and their
  shared implementation blocks.
- `smoke.py` checks output shape, parameter count, and backward propagation.
- `run_fast_tsconv_squeezeformer.sh` runs the full-size Squeezeformer family.
- `architecture_target_matrix.sbatch` and `run_arch_target_testsel.sh` run the
  maintained full-size architecture/visual-target experiments.

Historical result directories and ensemble manifests are preserved as
provenance; they are not executable model definitions.
