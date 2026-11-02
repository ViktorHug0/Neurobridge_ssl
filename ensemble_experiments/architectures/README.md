# Maintained registry encoders

The `--eeg_encoder_type` names that `train.py` resolves through
`ortho_encoders.build_architecture_encoder`. Everything else that once lived
here was removed in the 2026-11-02 cleanup.

## Available encoders

| Encoder | Purpose |
|---|---|
| `FastTSConvSqueezeformer` | Convolution/attention hybrid (`sqf_dbn*` cv4 recipes, CVPR screens, ValCon checkpoints) |
| `MultiScaleTSMixer` | Multi-resolution temporal-spatial MLP mixer (`honest_ensemble/multi_*_valcon` checkpoints) |
| `TokenREVE` | Scratch REVE with a switchable tokenizer (`rmix*` cv4 recipes, CVPR `rmix` screens) |

The loader still accepts the historical `Ortho` prefix, so existing checkpoints
and configurations remain loadable.

## Files

- `ortho_encoders.py`: `FastTSConvSqueezeformer`, `MultiScaleTSMixer`, their
  shared blocks, and the registry.
- `hybrid_encoders.py`: `TokenREVE` (self-check: `PYTHONPATH=. .venv/bin/python ensemble_experiments/architectures/hybrid_encoders.py`).
- `alljoined_encoders.py`: `_SwitchBlock`, the transformer block `TokenREVE` uses.
