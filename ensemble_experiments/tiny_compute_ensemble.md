# Compute-matched TinyTSConv + TinyATM ensemble

## Question

At a fixed encoder-forward FLOP budget, does allocating compute to ten lightweight,
independently trained functions outperform one full TSConv plus one full ATM?

## Locked primary protocol

- 5 TinyTSConv members, seeds 3300--3304.
- 5 TinyATM members, seeds 4300--4304.
- All members use InternViT-28, pairwise raw-EEG SubjectMix, batch 1024, 50 epochs,
  and test-loss checkpoint selection.
- Primary prediction is the uniform mean of all ten per-query row-z score matrices.
- No model or subset selection is allowed for the primary Tiny-10 result.
- Reference is the existing full TSConv-28 + full ATM-28 row-z pair (40.65% top-1).

## Locked architectures

| Encoder | Key dimensions | Parameters | Forward FLOPs/sample |
|---|---|---:|---:|
| Full TSConv | conv width 40; backbone 1024; alignment 512 | 2,630,912 | 40.895M |
| TinyTSConv | conv width 10; backbone 256; alignment 128 | 165,440 | 7.895M |
| Full ATM | d_model 250; conv width 40; backbone/alignment 128 | 876,396 | 96.316M |
| TinyATM | d_model/d_ff 112; conv width 12; backbone/alignment 128 | 201,908 | 18.528M |

The ten tiny encoders total 132.115M forward FLOPs/sample, versus 137.211M for the
two full encoders (96.3% of the reference budget). Parameter totals are 1.837M versus
3.507M. Including the EEG and 3200-dimensional InternViT image projectors gives
140.80M versus 142.39M paired-encoding FLOPs/sample (98.9%). At batch 1024, adding
the contrastive score-matrix products gives approximately 143.42M versus 143.70M
FLOPs/sample, a 99.8% theoretical training-forward match. The same-hardware calibration
separately records realized end-to-end time, including fixed overhead, retrieval
evaluation, and checkpoint writes.

Ten-epoch RTX-3080 calibration (job 50302):

| Encoder | End-to-end time | Seconds/epoch |
|---|---:|---:|
| Full TSConv | 293 s | 29.3 |
| TinyTSConv | 199 s | 19.9 |
| Full ATM | 462 s | 46.2 |
| TinyATM | 195 s | 19.5 |

Thus Tiny-10 is FLOP-matched but not GPU-hour-matched: ten small independent runs take
about 2.61 times the realized GPU time of the two full runs. Small kernels, ten separate
data/evaluation loops, and fixed per-model overhead make the 3080 substantially less
efficient at executing the theoretical compute budget. Both quantities must therefore
be reported rather than treating FLOPs as a proxy for elapsed time.

## Secondary diagnostics

- TinyTSConv and TinyATM mean solo accuracy.
- Full Tiny-10 top-5 and oracle top-1.
- Within-family and cross-family score correlations.
- A predeclared balanced prefix curve ordered TS0, ATM0, TS1, ATM1, ..., TS4, ATM4.
- Per-subject delta against the fixed full TSConv-28 + ATM-28 reference.

Implementation:

- `tiny_compute_ensemble.sbatch`
- `run_tiny_compute_ensemble_fold.sh`
- `analyze_tiny_compute_ensemble.py`
- `calibrate_tiny_compute.sbatch`
