# Results — orthogonal architectures

All arms align to **InternViT-6B layer 28** at fd512/bb1024, seed 3300, LOSO over 10 subjects.
Reference points: baseline **35.90**, best single model **37.00**, target **>30**.

The recipe changed mid-run. Arms were first trained with pairwise SubjectMix / 50 epochs (the
`tsconv_iv` recipe); controls then showed that handicaps these architectures by 5-6 points, so
the comparison of record is the `group_e100` recipe (group SubjectMix `prob 1.0`, 100 epochs).
Both tables are kept below — the first is what counts.

## Scoreboard

### On the `group_e100` recipe (the fair comparison)

| arm | folds so far | mean | baseline, same folds | |
|-----|--------------|------|----------------------|---|
| `convgru_group` | 29.0 / 42.0 | **35.50** | 46.75 |
| `mixer_group` | 35.5 / 44.0 / 21.5 | **33.67** | 40.17 |
| `spec_group` | 32.0 / 36.5 / 11.5 / 22.0 / 26.0 | **25.60** | 35.90 | **killed** at fold-5 gate |
| `perceiver_group` / `tcn_group` / `covpool_group` | — | queued | |
| `sincpow_group` | 9.0 @ep25 | **killed** | |

`mixer_group` projects to ~30.1 over 10 folds (ratio 0.838 against the baseline on the same
folds). Both `convgru` and `mixer` come within a point of the TSConv baseline on fold 2
(42.0 and 44.0 against 43.0).

### On the original pairwise/50ep recipe (superseded — these arms were handicapped)

| arm | folds 1-3 | outcome |
|-----|-----------|---------|
| `mixer` | 30.0 / 38.5 / 16.0 = 28.17 | superseded by `mixer_group` (+5.5 per fold) |
| `convgru` | 26.5 (fold 1) | superseded by `convgru_group` |
| `covpool` | 23.5 / 28.5 / 13.5 = 21.83 | killed at fold 3, re-test queued |
| `tcn` | 23.5 / 35.0 / 3.0 = 20.50 | killed at fold 3, re-test queued |
| `riemann` | — | killed at fold 1, stays dead |

## Two diagnostics that shaped the run

**Loss-selected checkpointing is not a handicap on these arms.** `best top1 acc` is top-1 at
the *minimum-test-loss* epoch (train.py:2815), not the max over epochs, so an arm whose loss
and accuracy decouple is scored at a bad checkpoint. Measured gap (max-epoch minus
loss-selected), averaged over folds: `iv33` 2.65, `atm33` 1.55, `tcn` 3.67, `mixer` 1.00,
`covpool` 1.00. The effect is general and the new arms are not disproportionately hurt, so
the protocol stays as-is and the comparison to 35.90 is fair.

**The recipe, not the architecture, was the binding constraint.** Acting on the overfitting
finding below, `mixer` was re-run with the `group_e100` recipe -- 9-source Dirichlet
SubjectMix at `prob 1.0` and 100 epochs, the recipe behind the repo's best single model
(37.00). It differs from the original arm by exactly two flags, same width. Fold-1
comparison, and the same test repeated on a second architecture:

| arm | pairwise / 50ep | group / 100ep |
|-----|-----------------|---------------|
| `mixer` | 30.0 | 36.5 *(already, at epoch 30 of 100)* |
| `convgru` | 26.5 | 30.0 *(already, at epoch 25 of 100)* |

Both architectures pass their own pairwise *final* score before reaching epoch 30 of the
group run. The first four arms were therefore judged on a recipe that handicapped them, and
`tcn` / `covpool` deserve a re-test on the group recipe before being written off for good.

**Width was not the lever.** `mixer` at bb128/fd128 scored 28.5 on fold 1 against bb1024's
30.0 -- narrowing hurt slightly. Probe killed once the group result superseded it.

**They overfit, they do not under-train.** The loss-selected best epoch lands at 23-29 of 50
(`mixer` 28.7, `covpool` 27.0, `tcn` 22.7), i.e. the test loss bottoms out mid-run and then
climbs. A larger epoch budget cannot help; less capacity might -- which is why the bb128
width probe is running.

**Fold 3 is a low-SNR subject, not a mis-scaled one.** Every ortho arm collapses on subject 3
(`mixer` 21.5, `covpool` 13.5, `convgru` 12.5, `spec` 11.5, `tcn` 3.0, against a baseline of
27.0 -- already its own worst fold). The obvious hypothesis was a scale outlier that TSConv's
BatchNorm absorbs, which `--eeg_instance_norm` would fix in one flag. `subject_stats.py`
rules it out: subject 3's per-channel std spread is mid-pack (2.13 max/med, against 3.82 for
subject 4), but it has the **lowest p99 amplitude of all ten** (2.987, range 2.926-3.877) and
near-lowest overall std. It is low-SNR, not mis-scaled, so per-trial standardisation would
only rescale the noise. The finding is that these architectures are less robust to low SNR
than TSConv/ATM -- not a fixable preprocessing bug.

**Every arm craters on fold 3.** `mixer` 16.0, `covpool` 13.5, `tcn` 3.0, against a baseline
of 27.0 -- already the hardest fold. The ortho encoders are disproportionately fragile to
whatever makes subject 3 different.

## Kill post-mortems

### `riemann` (OrthoRiemann) — killed at fold 1, epoch 25

Raw-electrode covariance -> BiMap -> LogEig. Plateaued at **3.0–5.5% top-1** where a
healthy arm (atm33) is at 24–27% by the same epoch, and the *test loss was rising*
(4.70 -> 4.86), i.e. it had already started overfitting a representation that never
worked. Killed after ~10 min rather than burning the 3 h of GPU the full 10 folds
would have cost.

**Diagnosis.** Single-trial EEG covariance is dominated by broadband sensor noise; the
evoked component is a small fraction of total variance even with `--data_average`. The
BiMap is initialised as a random orthonormal 63->24 projection and the only gradient
reaching it flows through an eigendecomposition, which is a weak and badly-conditioned
learning signal. The second-order layer was sitting directly on noise.

**Replacement: `covpool`.** Same orthogonal claim — the readout is a covariance, not a
linear pool — but the covariance is taken over *learned* temporal feature maps, so the
second-order layer sits on features that already have usable SNR. It also drops the
eigendecomposition for a signed-log compression, removing the ill-conditioned gradient
path entirely.

### `tcn` (OrthoTCN) — killed at fold 3

23.5 / 35.0 / 3.0. Training was healthy on fold 3 (TrainLoss 6.93 -> 3.61, the same curve as
the fold it scored 35.0 on), so this was not divergence. Its test loss bottomed at epoch ~5
and rose thereafter while top-1 kept climbing to 10% — but even that best epoch is 10.0, so
the 3.0 is not a checkpoint-selection artifact. The architecture genuinely failed on this
subject.

### `covpool` (OrthoCovPool) — killed at fold 3

23.5 / 28.5 / 13.5. Trained stably and cleanly beat `riemann`, confirming the diagnosis that
a second-order readout needs learned feature maps under it. But it never approached the
baseline: reaching 30 overall would have required 33.5 across the remaining seven folds,
i.e. matching the baseline exactly after trailing it by 8 points on the first three.

### `sincpow` (OrthoSincPow) — killed at fold 1, epoch 26 (group recipe)

Fixed band-pass filter bank -> per-band spatial filters -> **log-variance** pooling. Train
loss fell faster than any other arm (6.38 -> 3.47 by epoch 21, below `mixer`'s), while test
top-1 stalled at 4.5-7% and the test loss *rose* (4.61 -> 6.31). It fits the training
subjects and transfers nothing.

**Diagnosis, and it was the flagged risk.** Log-variance pooling is phase-invariant by
construction: it keeps the band-power envelope and discards the sign of the evoked response.
THINGS-EEG retrieval is ERP-driven, so the one thing this encoder throws away is the one
thing that carries the signal. It was included as an honest test of whether envelope
dynamics alone suffice at 200-way. They do not.

This is the same failure axis as `riemann`, from the other direction: `riemann` kept
second-order structure but drowned the ERP in sensor noise, `sincpow` kept clean band power
but deleted the ERP outright. Both confirm that any viable encoder here has to preserve
phase-locked evoked structure.


---

# Final 10-fold results

| arm | encoder | 10-fold mean | >30 |
|-----|---------|--------------|-----|
| `mixer_group` | OrthoMixer | **32.75** | yes |
| `perceiver_group` | OrthoPerceiver | **29.60** | no, by 0.40 |
| `convgru_group` | OrthoConvGRU | **26.85** | no |
| `spec_group` | OrthoSpec | 25.60 (5 folds) | no |
| `tcn_group` | OrthoTCN | 25.50 (3 folds) | no |
| `covpool_group` | OrthoCovPool | 22.00 (3 folds) | no |
| `sincpow` | OrthoSincPow | killed, fold 1 | no |
| `riemann` | OrthoRiemann | killed, fold 1 | no |

Reference: baseline 35.90, best single model 37.00. **The >30 target was met by one
architecture, not five.** The 100-epoch budget is not the limit -- `perceiver`'s
loss-selected best epoch averages 53/100 with only 1/10 folds near the cap -- so there is no
principled extension that would close its 0.40 gap.

# The headline result: architectural diversity does not buy score diversity

The whole premise of this work was that the ensemble plateaus (nested LOFO 43.95 at k=4)
because every candidate is the same alignment recipe over the same EEG, and that the
48-member pool's correlation FLOOR of 0.628 proves there is no decorrelated member to select.
Three encoders sharing no inductive bias with TSConv or ATM -- a Perceiver, an MLP-Mixer and
a bidirectional GRU -- were built to break that floor.

They do not.

| arm | solo | min corr vs the 48 existing | median corr |
|-----|------|-----------------------------|-------------|
| `ortho_mixer_group` | 32.75 | 0.656 | 0.839 |
| `ortho_perceiver_group` | 29.60 | 0.637 | 0.834 |
| `ortho_convgru_group` | 26.85 | 0.631 | 0.834 |
| *existing pool, 1128 pairs* | | *min 0.628* | *median 0.820* |

Every ortho arm lands **at** the existing floor (0.631-0.656 vs 0.628) and its median
correlation is **above** the pool median (0.834-0.839 vs 0.820). Ensemble effect, nested LOFO:

| k | old pool | old + ortho |
|---|----------|-------------|
| 2 | 38.20 | 38.20 (+0.00) |
| 3 | 42.00 | 42.00 (+0.00) |
| 4 | 43.95 | 43.55 (-0.40) |

Nothing. The selector never picks an ortho arm at any k.

**What this means.** The pool's correlation floor is not an artefact of architectural
similarity, which was the standing hypothesis after routing, stacking, accuracy-pruning and
diversity-pruning all failed. Swapping convolution for attention, for dense token mixing, or
for recurrence changes the encoder completely and moves the score matrix almost not at all.
The shared structure is imposed by the task -- the same EEG, the same frozen InternViT
targets, the same contrastive objective -- not by the encoder family. Diversity has to be
bought somewhere other than the EEG architecture; the one lever with measured evidence behind
it remains image-backbone variation, which is what produced the ensemble gains in the first
place.
