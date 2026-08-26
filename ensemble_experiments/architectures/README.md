# Orthogonal architectures for ensemble diversity

## Why

Nested-LOFO ensemble selection plateaus at **45.35** (k=4: `atm_iv + ge100 + tsconv_eva + tsconv_vith`).
Four different ways of restructuring *selection* — MoE routing, stacking, top-15-by-accuracy
pruning, diversity pruning — all failed, and the correlation diagnostic says why: over 48
candidates the **minimum** pairwise score-matrix correlation is **0.628** (median 0.820).
Every member is the same alignment recipe over the same EEG, so there is no decorrelated
member to select. New diversity has to be *built*, not selected.

## What

Eight encoders that each break an assumption TSConv and ATM share. Everything else in the
recipe is held identical to `tsconv_iv` (InternViT-6B layer 28, pairwise SubjectMix
a=0.5 raw_eeg, 50 epochs, fd512/bb1024, seed 3300, LOSO over 10 subjects), so any
difference is attributable to the architecture alone.

| arm | encoder | what it breaks | inspiration |
|-----|---------|----------------|-------------|
| `riemann` | `OrthoRiemann` | **first-order statistics** — sees only per-window channel covariance, never the waveform | SPDNet / Riemannian tangent space |
| `sincpow` | `OrthoSincPow` | **time domain & linear pooling** — fixed band-pass bank + log-variance (phase-invariant) | FBCSP |
| `spec` | `OrthoSpec` | **the input object** — complex STFT (real+imag) as a 2-D image over (freq, time) | time-frequency CNNs |
| `mixer` | `OrthoMixer` | **convolution and attention** — dense learned token mixing, no locality prior | MLP-Mixer |
| `tcn` | `OrthoTCN` | **single temporal scale** — dilations 1..16 at full resolution, no early 5x downsample | WaveNet / TCN |
| `convgru` | `OrthoConvGRU` | **parallel readout** — a bidirectional recurrent state machine over time | ConvRNN |
| `covpool` | `OrthoCovPool` | **linear pooling** — covariance readout over *learned* feature maps (reserve second-order arm) | bilinear/second-order pooling |
| `perceiver` | `OrthoPerceiver` | **attention topology** — spatio-temporal tokens squeezed through 16 latent queries | Perceiver |

`covpool` and `perceiver` are held in reserve: they are only launched if one of the first
six fails a gate, so the GPU budget stays at 3 concurrent.

Two design points carried over from what *works* in TSConv/ATM:
- the same `Linear -> ResidualAdd(GELU/Linear/Dropout) -> LayerNorm` projection head, so only bodies differ;
- `OrthoRiemann`'s covariance is left **uncentred** on purpose, so `(1/T)Σ xxᵀ` retains the
  evoked mean's outer product — a centred covariance throws the ERP away and the task is ERP-driven.

## Target

≥5 arms above **30%** on the 10-fold LOSO average (baseline 35.90, best single 37.00).

## Kill-fast gates

Baseline is not flat across folds (`{1:50.5, 2:43.0, 3:27.0, 4:29.5, 5:29.5, ...}`), so the
floor is per-gate: **31.0 after fold 3** (baseline 40.17 there), **28.0 after fold 5**
(baseline 35.90). An arm below the floor exits 3 and frees its GPU.

## Files

- `ortho_encoders.py` — the six encoders + `build_ortho_encoder`. `train.py` routes any
  `--eeg_encoder_type` starting with `Ortho` here (3-line hook in `build_eeg_encoder`).
- `smoke.py` — shape/param/grad self-check, `--cuda` also times a step.
- `run_ortho.sh <arm>` — the 10-fold LOSO driver with the abort gates. `FOLDS="6 7 8"` lets a
  second worker backfill a disjoint subset on another GPU.
- `abort_check.py <run_dir> <floor>` — the gate.
- `ortho.sbatch` — array over all six arms, 3 concurrent.
- `results.md` — outcomes (written as arms land).

Results land in `results/things_eeg/ortho_arch/<arm>/seed3300/`.
