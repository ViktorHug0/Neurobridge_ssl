# Mech-interp for the NeurIPS Track 1 sealed regime (from 2026-11-10 afternoon)

Goal (user): apply the subject-key findings of `MI_LOG.md` to cross-subject decoding in the challenge's current
evaluation setting. Code: `scripts/mi/nc_*.py`, helper `nc_lib.py`, launcher `nc_train.sh`; outputs
`results/things_eeg/mi/nc/`; training runs `results/alljoined/track1/cv4/mi_*`.

**Setting.** New people, new images, single 32x120 trials at 120 Hz, ~3 repeats per (person, image), gallery ~40k,
64-trial batches mixing 11 people, no state across predict() calls. Judge = the clean proxy of NIGHT_LOG N71/N73:
the fold's 5 held-out people x the 5,000 images no `--holdout_images 5000` run trained on x 4 repeats, top-5 % at
r=1/3/4 (chance 0.10 %), BIG = + 9.4k distractors. ~300 hits per cell, so one cell has SE ~6 % relative.

**GPU.** One extra rtx3080 granted by the user on 11-10 (job `mi_gpu`); the other instance keeps its three.

## Plan
- Phase A, frozen members: (1) person x image 2x2; (2) key dependence on single trials and the ceiling of key
  normalisation; (3) deployable key estimates; (4) site of the person gap via oracle person-specific corrections;
  (5) where the late-epoch memorisation of the REVE members sits (epoch stitching).
- Phase B, training levers on the clean proxy: key-swap / key-free training; part-wise schedule if (5) localises.
- Phase C, test-time levers inside the rules, only if (3) is positive.
- Not transferable: the transductive self-training of E20 (predict() never sees the gallery, no state).

## N01 person x image 2x2 (fold 1, 5k gallery, top-5 % r=3; `nc01_cells_key.py`)
Seen people = fold 2's five (training people of the fold-1 models, matched difficulty). "trained" = 5,000 of the
11,540 trained images (for seen people these are the exact training trials).

| member | new person, new image | new person, trained image | seen person, new image | seen person, trained image |
|---|---|---|---|---|
| REVE-d6 ep10 | 1.20 | 1.56 | 2.00 | 5.75 |
| rmix ep10 | 1.26 | 1.55 | 1.78 | 4.50 |
| sqf_dbnX | 1.08 | 1.35 | 1.33 | 2.82 |

On new images the person factor is x1.67 (d6), x1.41 (rmix), x1.23 (SQF); for new people the image factor is
x1.2-1.3. Both gaps matter; the REVE family has the larger person gap.

## N02 key dependence on single trials (new images, r=3)
Key = the person's mean ERP over all its training-set trials (norm 3.7, between-person deviation 3.2, single-trial
norm 56). K = mean key of the 15 Alljoined training people.

| member | seen: own | seen: wrong key | seen: no key | seen: K | new: own | new: no key | new: K | new: midpoint | new: nearest training key |
|---|---|---|---|---|---|---|---|---|---|
| REVE-d6 | 2.00 | 1.36 | 1.21 | 1.32 | 1.20 | 0.81 | 1.05 | 1.15 | 1.04 |
| rmix | 1.78 | 1.20 | 1.11 | 1.20 | 1.26 | 0.83 | 1.10 | 1.22 | 1.24 |
| sqf_dbnX | 1.33 | 1.22 | 1.19 | 1.22 | 1.08 | 1.03 | 1.10 | 1.09 | 1.11 |

- The key mechanism holds on single trials: a 3.7-norm offset inside a 56-norm trial decides whether the REVE members
  decode a seen person at 2.0 or at 1.3.
- With a wrong key a seen person scores like a new person (d6 1.36 vs 1.20, rmix 1.20 vs 1.26): the whole seen-person
  advantage on new images is key-gated person-specific decoding. SQF barely uses the key (as on THINGS).
- New person: the own key is the best input (REVE members). Removing it, or replacing it by K or by the nearest
  training key, loses 0-30 %. The key-conditional decoder interpolates to a new key; it is not a nuisance to remove.
  -> test-time key normalisation has no headroom even with the oracle key: Phase C and A(3) are closed.

## N03 what is learned late (REVE-d6 ho5k fold 1, top-5 % r=3, 5k gallery; `nc02_epochs.py`)

| epoch | new person, new image | new person, trained image | seen person, new image | same, wrong key | seen person, trained image | same, wrong key |
|---|---|---|---|---|---|---|
| 4 | 0.98 | 1.26 | 0.98 | 0.81 | 1.57 | 1.24 |
| 6 | 1.15 | 1.38 | 1.34 | 0.97 | 2.40 | 1.73 |
| 8 | 1.16 | 1.39 | 1.65 | 1.18 | 3.73 | 2.53 |
| 10 | 1.20 | 1.56 | 2.00 | 1.36 | 5.75 | 3.82 |
| 12 | 1.12 | 1.46 | 2.03 | 1.51 | 10.81 | 6.81 |
| 14 | 1.08 | 1.36 | 2.06 | 1.36 | 24.08 | 15.03 |
| 16 | 0.81 | 0.97 | 1.75 | 1.16 | 51.40 | 31.40 |

- At epoch 4 a seen person on new images scores like a new person (0.98 = 0.98). From epoch 6 on, everything the model
  gains on new images goes to the seen people (0.98 -> 2.06) while new people stay at 1.15-1.20.
- The decline after epoch 10 that NIGHT_LOG N72 read as image memorisation hits NEW PEOPLE first (1.20 -> 1.08 at ep 14)
  while seen people keep their level on never-trained images (2.00 -> 2.06), and new people decline on trained
  images too (1.56 -> 1.36). Up to epoch 14 it is person specialisation; trial memorisation (in-sample 5.7 -> 51) takes
  everyone down only at epoch 16.
- The person-specific gain is key-gated at every epoch (wrong key removes about 2/3 of it); the in-sample memorised
  trials are partly key-gated as well (51 -> 31).

## N06 output-side person offset has no headroom in the sealed regime (`nc04_output_offset.py`, fold 1, cached outputs)
Removing b x the person's mean output, top-5 % r=3 at 5k / 14k:

| member | none | oracle person mean b=.5 | b=1 | in-batch same-person mean b=.5 | b=1 | batch mean b=.5 | b=1 |
|---|---|---|---|---|---|---|---|
| REVE-d6 | 1.22 / 0.52 | 1.18 / 0.50 | 1.02 / 0.41 | 1.19 / 0.51 | 0.99 / 0.45 | 1.17 / 0.54 | 1.01 / 0.46 |
| rmix | 1.38 / 0.58 | 1.28 / 0.52 | 1.12 / 0.47 | 1.29 / 0.54 | 1.12 / 0.48 | 1.25 / 0.52 | 1.07 / 0.50 |
| sqf_dbnX | 1.03 / 0.38 | 1.04 / 0.39 | 0.92 / 0.38 | 1.02 / 0.39 | 0.85 / 0.35 | 1.00 / 0.39 | 0.90 / 0.35 |

Even the oracle person mean (all ~20k trials of the person) hurts. The lean TTA's failure at r <= 4 is therefore not
an estimation problem: at this SNR the mean output direction is useful prior, and centring is closed for good.

## N04 epoch stitching (REVE-d6 fold 1, ep10 vs ep16; `nc03_stitch_amp.py stitch`), top-5 % r=3, new / seen people
Parts: 0 = patch embedding, 1-6 = blocks, 7 = readout. Pure ep10 1.20 / 2.00, pure ep16 0.81 / 1.75.

| first j parts from | j=1 | j=2 | j=3 | j=4 | j=5 | j=6 | j=7 |
|---|---|---|---|---|---|---|---|
| ep16, rest ep10 | 1.04 / 1.80 | 0.96 / 1.77 | 0.94 / 1.79 | 1.00 / 1.82 | 0.93 / 1.76 | 0.87 / 1.65 | 0.78 / 1.85 |
| ep10, rest ep16 | 0.94 / 1.63 | 1.03 / 1.71 | 1.08 / 1.89 | 1.09 / 1.81 | 1.15 / 1.88 | 1.16 / 1.88 | 1.17 / 1.90 |

- Hybrids stay functional (seen people 1.6-1.9 everywhere), so the comparison is meaningful.
- No hybrid beats ep10 for new people: a part-wise epoch budget is not a lever.
- The late loss for new people is input-side: swapping in only the ep16 embedding costs 1.20 -> 1.04, embedding +
  block 0 -> 0.96 (60 % of the full drop); the ep16 readout alone costs almost nothing (1.17).

## N05 key amplification (ep10; x + a (m - K), own key m)
New people: a=-0.5 1.15, 0 1.20, +0.25 1.14, +0.5 1.11, +1 0.94, +2 0.64. Seen people: 1.68, 2.00, 1.92, 1.79, 1.54,
1.02. The natural key amplitude is the optimum for both; a stronger person cue does not help. Closed.

## N01/N02 replication on fold 3 (`K=3 nc01_cells_key.py d6m,rmixm`, ep10 of the `*_ho5k_mid_optx_things` runs), r=3

| member | new, new image | new, trained image | seen, new image | seen, trained | seen: wrong key | seen: no key | new: K | new: midpoint | new: no key |
|---|---|---|---|---|---|---|---|---|---|
| REVE-d6 | 1.20 | 1.66 | 1.96 | 5.28 | 1.31 | 1.11 | 1.23 | 1.26 | 1.10 |
| rmix | 1.18 | 1.64 | 1.97 | 4.96 | 1.23 | 1.06 | 1.12 | 1.18 | 1.03 |

Same picture as fold 1: person factor x1.6-1.7 on new images, entirely key-gated; no key substitution moves a new
person by more than the noise (+-6 %).

## N04b earlier input layers inside the ep10 model (`PAIRS=6:10,8:10,4:10 JS=1,2,3 nc03_stitch_amp.py stitch`)
new / seen, r=3 (pure ep10 1.20 / 2.00): ep8 embedding 1.22 / 1.89, + block 0 1.18 / 1.89, + block 1 1.22 / 1.84;
ep6 embedding 1.17 / 1.66, + block 0 1.15 / 1.62, + block 1 1.18 / 1.59; ep4 parts break the model (0.80-0.95).
The input layers of epochs 6-10 differ only in what they give seen people (1.59 -> 2.00); new people are indifferent.
Seen-person specialisation is input-side (as on THINGS), and freezing or slowing the early layers cannot help new people.

## N07 where the image content is read, seen vs new people (`nc07_occlusion.py d6`, fold 1, r=3)
Inside one window or electrode group the trial is replaced by the person's key (person cue intact, content removed).
Score relative to no ablation, new / seen: prestimulus 1.03 / 0.96, 0-100 ms 0.98 / 0.96, 100-200 ms 0.47 / 0.41,
200-300 ms 0.50 / 0.36, 300-500 ms 0.50 / 0.56, 500-800 ms 0.86 / 0.85, O + PO electrodes 0.10 / 0.05, P 0.24 / 0.16,
CP 0.74 / 0.75, frontal + central 0.90 / 0.82.
Same map for both groups (100-500 ms, occipital and parietal); the person-specific gain has no separate site in the
input, and no ablation helps new people. Input masking is not a lever.

## N08 key-swap training, first attempt (killed at epoch 5: design flaw)
`mi_d6_ho5k_kswap` (= reve_d6_ho5k_long_optx_things + `--subject_key swap`, fold 1) drew the partner key among all 25
training people. THINGS-EEG2 keys (sub-21..30) are about twice as large (norm 5-12 vs 2-8) and uncorrelated with the
Alljoined keys (mean r 0.09; within Alljoined 0.26, within THINGS 0.67; mean distance 8.7 across vs 4.7 within
Alljoined), so the swap also erased the dataset cue. In-run sealed curve ep1-4: 15.9 / 18.4 / 19.5 / 21.4 vs baseline
18.6 / 21.3 / 22.7 / 25.6. Fixed in train.py: the partner comes from the same dataset; rerun as `mi_d6_ho5k_kswapd`.

# Dataset integration (user, 11-10 afternoon: is there something better than per-dataset BatchNorm?)

## D01/D02 how the two datasets differ at the input (`nd01_dataset_align.py`, `nd02_lag.py`; fold-1 training people)
Both datasets show the same 16,540 training images, so per-image mean responses are paired (4,000 images used).
- Amplitude: THINGS-EEG2 single trials have half the RMS of Alljoined trials on every electrode (ratio 0.31-0.59),
  but a mean ERP 3.3x larger (key norm 6.8 vs 2.1): the evoked response is about 6x larger relative to the noise.
- Both are 5 Hz RSVP (keys periodic at 200 ms).
- Latency: image-specific content (per-image mean minus the dataset key, posterior electrodes) correlates across
  datasets at +0.030 unshifted and peaks at +0.051 when THINGS is shifted LATER by 4-6 samples (33-50 ms); two halves
  of Alljoined peak at lag 0 (0.023), two halves of THINGS at lag 0 (0.132). Alljoined responses lag THINGS-EEG2 by
  about 45 ms.
- Per-image identification from sensor means is at chance even within a dataset (<= 4 repeats x 5-8 people), and a
  fitted 32x32 electrode re-mix (1-7 taps) adds nothing measurable on held-out images: the data resolve the lag, not a
  spatial map.

## D03 a 5-sample lag makes THINGS-EEG2 people decodable by an Alljoined-only model (`nd03_cross_dataset.py`)
Alljoined-ONLY models (fold 3, `*_ho5k_mid_optx_aj`, ep10) on five THINGS-EEG2 people (sub-21, 23, 25, 27, 29: a
dataset they never saw), never-trained images, 5k gallery, top-5 % r=1 / r=3 / r=4. THINGS trials shifted later by n
samples (x(t) -> x(t - n)).

| condition | REVE-d6 | rmix |
|---|---|---|
| own held-out Alljoined people (reference) | 0.59 / 1.06 / 1.31 | 0.52 / 1.01 / 1.25 |
| THINGS raw | 0.33 / 0.56 / 0.58 | 0.30 / 0.56 / 0.58 |
| THINGS lag +2 | 0.61 / 1.13 / 1.25 | 0.60 / 1.10 / 1.29 |
| THINGS lag +3 | 0.74 / 1.36 / 1.53 | 0.71 / 1.32 / 1.48 |
| THINGS lag +4 | 0.79 / 1.39 / 1.58 | 0.76 / 1.32 / 1.53 |
| THINGS lag +5 | 0.83 / 1.48 / 1.61 | 0.77 / 1.38 / 1.52 |
| THINGS lag +6 | 0.85 / 1.44 / 1.62 | 0.74 / 1.28 / 1.51 |
| THINGS lag +7 | 0.78 / 1.32 / 1.52 | 0.69 / 1.26 / 1.47 |
| THINGS lag +8 | 0.73 / 1.14 / 1.34 | 0.62 / 1.01 / 1.31 |
| THINGS lag +10 | 0.58 / 0.87 / 1.02 | 0.50 / 0.79 / 0.88 |
| lag +5 and gain x0.33 / x0.5 / x2 | 1.48 / 1.46 / 1.50 (r=3) | 1.38 / 1.37 / 1.38 |
| lag +5 and dataset key -> Alljoined key | 0.80 / 1.52 / 1.71 | 0.73 / 1.32 / 1.69 |
| raw, dataset key -> Alljoined key | 0.24 / 0.44 / 0.42 | 0.23 / 0.36 / 0.45 |

- The dataset gap is a latency offset of 5 samples (42 ms): after the shift, people of the other dataset are decoded
  BETTER than the model's own held-out Alljoined people (cleaner recordings), x2.5 over raw.
- Gain is irrelevant for the REVE family; the dataset key matters little once the lag is fixed.
- Consequence: in joint training the two datasets cannot share a time-locked decoder (the members lose half their
  accuracy at +-2 samples). Per-dataset BatchNorm fixes amplitude statistics, not time.
- Lever: `train.py --things_lag 5` (shifts ids >= 21 later by 5 samples on the fly). Runs: `mi_rmix_ho5k_lag5`,
  `mi_sqfx_ho5k_lag5` (fold 1) vs `rmix_reg_ho5k_long_optx_things`, `sqf_dbn_ho5k_long_optx_things`.

## D04 per-person latency against the Alljoined group (`nd04_person_lag.py`, paired per-image content, fold-1 training people)
- THINGS-EEG2 people: +5.66 samples on average (47 ms), SD 1.0, range +3.7..+7.0, sharp peaks (correlation doubles
  from lag 0 to the peak). The offset is a property of the dataset, not of a few people.
- Alljoined people against the other Alljoined people: mean -0.55, SD 2.2 samples, but the correlation curves are flat
  (peak vs lag 0 differ by < 10 %) and the weakest people (13, 17) do not correlate at all: per-person latency inside
  Alljoined is not resolved by this measure and is not pursued.

## D05 held-out Alljoined people are not latency outliers (`nd05_person_shift.py 1 d6`)
Per held-out person, top-5 % r=3 as its trials are shifted -4..+4 samples: every person peaks at 0 or within noise of it
(sub-2 2.28 at 0 / 2.32 at +1; sub-5 1.94 at 0; sub-15 0.76 at 0; sub-4 and sub-11 flat at 0.4-0.7). Pooled 1.22 with no
shift, 1.26 with the oracle best shift per person (selection-biased), 1.09 with a label-free key-lag estimate.
The hard people stay hard at every shift. Per-person latency is closed; only the dataset-level lag matters.

## D06 finer lag (`nd06_lag_fine.py`; Alljoined-only REVE-d6 on THINGS sub-22, 24, 26, 28, 30), top-5 % r=1 / r=3 / r=4
lag 5: 1.06 / 1.95 / 2.26; lag 5.5 (interpolated): 1.07 / 1.96 / 2.25; lag 6: 1.11 / 1.95 / 2.28; per-person lag of
D04 rounded 1.09 / 1.91 / 2.23, interpolated 1.08 / 1.89 / 2.21. One global lag of 5-6 samples is enough.
(These five THINGS people are decoded at about twice the level of the model's own held-out Alljoined people.)

## Runs in progress (fold 1, clean proxy, sharing the one GPU)
- `mi_rmix_ho5k_lag5` = rmix_reg_ho5k_long_optx_things + `--things_lag 5`, stop 12, checkpoints 8 / 10 / 12.
- `mi_sqfx_ho5k_lag5` = sqf_dbn_ho5k_long_optx_things + `--things_lag 5`, 40 epochs, checkpoints 24 / 32 / 36 / 40.
- `mi_d6_ho5k_kfree` = reve_d6_ho5k_long_optx_things + `--subject_key demean`, stop 12, checkpoints 8 / 10 / 12.
In-run sealed curve, epoch 1: rmix lag5 18.1 vs baseline 13.5; SQF-X lag5 13.7 vs 13.4.

## N09 key-free training does not help new people in the sealed regime (`mi_d6_ho5k_kfree`, fold 1, epoch 8)
`nc05_eval_arm.py 1 mi_d6_ho5k_kfree reve_d6_ho5k_long_optx_things 8 raw,K,batch,own`: new people x never-trained images,
top-5 %, baseline -> arm (paired bootstrap), by what is subtracted from the test trial:

| test-time input | 5k r=3 | 5k r=4 | 14k r=3 | 14k r=4 | seen people 5k r=3 (baseline 1.65) |
|---|---|---|---|---|---|
| raw trial | 1.16 -> 0.84 (-27 +- 5 %) | 1.39 -> 1.00 (-28 +- 4 %) | 0.47 -> 0.31 (-33 +- 8 %) | 0.57 -> 0.40 (-29 +- 7 %) | 0.99 |
| minus mean training key (static) | 1.00 (-13 +- 5 %) | 1.19 (-14 +- 4 %) | 0.42 (-11 +- 9 %) | 0.45 (-21 +- 7 %) | 0.98 |
| minus the 64-batch mean ERP | 0.94 (-19 +- 5 %) | 1.18 (-16 +- 5 %) | 0.37 (-21 +- 8 %) | 0.41 (-28 +- 7 %) | 1.04 |
| minus the person's own key (oracle) | 1.04 (-10 +- 6 %) | 1.33 (-4 +- 5 %) | 0.38 (-18 +- 9 %) | 0.49 (-14 +- 8 %) | 1.02 |

- Without the key, seen people fall to exactly the new-person level (1.65 -> 1.0): third confirmation that the
  seen-person advantage is carried by the key.
- New people lose too, even with their oracle key removed: the baseline's key-conditional decoding is worth about +10 %
  to a NEW person (it interpolates between training people). On THINGS (80-repeat averages, 200-way) key-free training
  gained; on single trials at a 5k-14k gallery it costs.
- Verdict: the key is a feature in this regime. Key-free and key-normalising levers are closed (training and test time).
  Run stopped after epoch 8 (17:25) to give the GPU to the lag runs; its epoch-8 checkpoint is kept.

## D07 the joint model holds two separate time-locked decoders (`nd07_joint_lag.py rmix`; fold-1 rmix ep10, trained on both)
Never-trained images, 5k gallery, top-5 % r=1 / r=3 / r=4.

| input | r=1 | r=3 | r=4 |
|---|---|---|---|
| new Alljoined people, raw | 0.67 | 1.25 | 1.47 |
| shifted earlier by 2 / 4 / 5 / 6 / 8 samples (towards THINGS timing) | 0.58 / 0.46 / 0.39 / 0.30 / 0.15 | 1.03 / 0.82 / 0.66 / 0.53 / 0.22 | 1.30 / 0.95 / 0.73 / 0.68 / 0.29 |
| shifted earlier by 5, key -> THINGS mean key | 0.44 | 0.77 | 0.94 |
| THINGS training people (seen people), raw | 6.31 | 15.59 | 18.89 |
| shifted later by 2 / 5 / 8 (towards Alljoined timing) | 3.10 / 1.10 / 0.72 | 7.19 / 2.06 / 1.28 | 8.53 / 2.37 / 1.36 |
| shifted later by 5, key -> Alljoined mean key | 1.00 | 1.93 | 2.19 |

- THINGS-EEG2 people are decoded 8x better than Alljoined seen people on new images (15.6 vs 1.8-2.0 at r=3): the
  cleaner dataset carries by far the best image decoder the model has learned.
- That decoder is locked to THINGS timing: moved to Alljoined timing the same trials fall to 2.06, the level the
  Alljoined route gives them (D03). Alljoined people moved to THINGS timing do not reach it either (0.66; 0.77 with the
  THINGS key). The two datasets are served by two time-locked routes that share little.
- This is what `--things_lag 5` is meant to merge. Whether the merged decoder transfers to new Alljoined people is the
  open question the lag runs answer.

## D08 lag-aligned joint training, first read: rmix fold 1, epoch 8 (`nc05_eval_arm.py 1 mi_rmix_ho5k_lag5 rmix_reg_ho5k_long_optx_things 8`)
New people x never-trained images, baseline -> lag 5 (paired bootstrap): 5k r=3 1.320 -> 1.296 (-1.8 +- 5.2 %), r=4
1.448 -> 1.476 (+1.9 +- 5.0 %); 14k r=3 0.544 -> 0.592 (+8.8 +- 9.2 %), r=4 0.644 -> 0.572 (-11.2 +- 6.6 %).
Seen Alljoined people (5k r=3): 1.53 -> 1.75. No effect on new people at epoch 8 on this fold; epochs 10 / 12, fold 3
and SQF-X pending.

## D09 the lag does merge the two routes, without helping new Alljoined people (`nd08_route.py`; rmix fold 1, epoch 8)
Each model at its training timing; never-trained images, 5k gallery, top-5 % r=3.

| people, dataset key at the input | baseline | lag 5 |
|---|---|---|
| THINGS training people, own key | 13.90 | 13.75 |
| THINGS training people, key -> Alljoined mean key | 5.55 | 10.72 |
| THINGS training people, no dataset key | 7.33 | 7.48 |
| new Alljoined people, own key | 1.32 | 1.30 |
| new Alljoined people, key -> THINGS mean key | 0.73 | 1.20 |
| new Alljoined people, no dataset key | 0.98 | 0.97 |

In the baseline the dataset key selects the route (the other dataset's key costs 45-60 %); after lag alignment it barely
matters (-7 % for Alljoined people, -22 % for THINGS people): the model now serves both datasets with one time-locked
decoder. New Alljoined people score the same either way (1.30 vs 1.32), so at epoch 8 their limit is not which decoder
they reach.

## D10 lag-aligned training, fold 1 at the budget epochs (18:30)
New people x never-trained images, top-5 %, paired bootstrap vs the baseline.

| read | 5k r=3 | 5k r=4 | 14k r=3 | 14k r=4 |
|---|---|---|---|---|
| rmix lag, ep10, alone (`nc05`, my stream) | 1.252 -> 1.180 (-5.8 +- 5.2 %) | 1.468 -> 1.572 (+7.1 +- 5.2 %) | 0.524 -> 0.548 (+4.6 +- 8.7 %) | 0.632 -> 0.632 (0.0 +- 7.8 %) |
| rmix lag, ep10, alone (`nc06`, logs/holdout/f1 stream) | 1.376 -> 1.312 (-4.7 +- 4.9 %) | 1.492 -> 1.520 (+1.9 +- 4.7 %) | 0.576 -> 0.636 (+10.4 +- 8.6 %) | 0.684 -> 0.632 (-7.6 +- 6.8 %) |
| ens5 (sub-6 budgets), arm swapped for rmix | 1.412 -> 1.492 (+5.7 +- 3.1 %) | 1.640 -> 1.668 (+1.7 +- 2.8 %) | 0.564 -> 0.564 (0.0 +- 5.5 %) | 0.668 -> 0.672 (+0.6 +- 4.6 %) |
| pair rmix + 0.5 SQF-X, arm swapped for rmix | 1.344 -> 1.368 (+1.8 +- 5.0 %) | 1.532 -> 1.664 (+8.6 +- 4.9 %) | 0.580 -> 0.584 (+0.7 +- 6.9 %) | 0.624 -> 0.708 (+13.5 +- 8.1 %) |
| SQF-X lag, ep24, alone (`nc05`) | 1.060 -> 1.044 (-1.5 +- 5.9 %) | 1.176 -> 1.164 (-1.0 +- 5.8 %) | 0.400 -> 0.352 (-12.0 +- 9.6 %) | 0.448 -> 0.436 (-2.7 +- 10.0 %) |

Seen Alljoined people (5k r=3): rmix 1.77 -> 1.95, SQF-X 1.16 -> 1.13.
Reading: alone, the lag changes nothing for new people (rmix and SQF-X). Inside the ensemble and the pair every cell is
>= 0 (mean +4 %), none beyond 2 SE. Not a result on one fold; fold 3 (rmix) and the SQF-X end of training pending.
(The two streams give 1.25 vs 1.38 for the same baseline checkpoint: different repeat triplets and batch composition.)
- 18:55 rmix lag, fold 1, epoch 12 (final), alone: 5k r=3 1.324 -> 1.200 (-9.4 +- 5.2 %), r=4 1.380 -> 1.436 (+4.1 +- 5.6 %);
  14k r=3 0.572 -> 0.488 (-14.7 +- 7.7 %), r=4 0.620 -> 0.616 (-0.6 +- 8.1 %). Epochs 8 / 10 / 12 alone average ~-1 %:
  no effect on fold 1.

## D11 SQF-X with the lag, and both lag members together (fold 1, 19:35)
SQF-X lag alone vs baseline (`nc05`): ep32 5k -1.1 +- 6.0 / +7.8 +- 6.1 %, 14k -5.0 +- 10 / 0.0 +- 10 %; ep40 5k
-3.3 +- 5.6 / -0.7 +- 5.7 %, 14k -8.1 +- 9.5 / -6.6 +- 9.0 % (r=3 / r=4). Flat alone, like rmix.
Ensemble level (logs/holdout/f1 stream; `nc06_ens.py`, `nd09_both_lag.py`), top-5 % r=3 / r=4, paired gain:

| swap | 5k | 14k |
|---|---|---|
| ens5, SQF-X -> lag | 1.412 -> 1.472 (+4.2 +- 2.5 %) / 1.640 -> 1.668 (+1.7 +- 2.3 %) | +0.7 +- 3.7 % / +2.4 +- 3.3 % |
| ens5, rmix -> lag | +5.7 +- 3.1 % / +1.7 +- 2.8 % | 0.0 +- 5.5 % / +0.6 +- 4.6 % |
| ens5, both -> lag | 1.412 -> 1.476 (+4.5 +- 3.2 %) / 1.640 -> 1.716 (+4.6 +- 3.0 %) | 0.564 -> 0.584 (+3.5 +- 5.7 %) / 0.668 -> 0.640 (-4.2 +- 5.0 %) |
| pair rmix + .5 SQF-X, both -> lag | 1.344 -> 1.396 (+3.9 +- 5.2 %) / 1.532 -> 1.636 (+6.8 +- 4.8 %) | 0.580 -> 0.648 (+11.7 +- 8.2 %) / 0.624 -> 0.660 (+5.8 +- 7.8 %) |

Lag members are flat alone but 13 of 14 ensemble-level cells are >= 0, around +4 %, each within 1.5 SE. Reading: the
lag changes the members' errors more than their accuracy (one shared decoder instead of two), which the sum can use.
One fold; fold-3 rmix lag running, fold-3 SQF-X lag launched 19:37 for the same read there.

## D12 fold 3 replication of the rmix lag, and folds 1 + 3 pooled (19:45)
Fold 3 alone (`nc05`, vs rmix_reg_ho5k_mid_optx_things, same epoch), 5k r=3 / r=4 | 14k r=3 / r=4:
ep8 +7.7 +- 5.3 / +7.5 +- 4.5 | +14.9 +- 10.2 / +9.5 +- 7.9 %; ep10 +13.9 +- 5.9 / +4.3 +- 4.7 | +26.3 +- 10.7 / +9.2 +- 7.8 %.
Fold 3 ensemble level (`K=3 nc06_ens.py`, ep10): ens5 rmix -> lag +0.9 / -0.5 | +2.5 / +4.0 %; pair +5.0 / +7.7 | -2.6 / -0.6 %.

Folds 1 + 3 pooled (`nd10_pool.py 1,3`; only rmix swapped for its lag version, ep10; paired bootstrap; per-fold in brackets):

| level | 5k r=3 | 5k r=4 | 14k r=3 | 14k r=4 | all cells |
|---|---|---|---|---|---|
| rmix alone | +0.8 +- 3.6 % (-5, +7) | +2.0 +- 3.3 % (+2, +2) | +10.4 +- 6.2 % (+10, +10) | +3.1 +- 5.3 % (-8, +15) | +2.9 % |
| ens5 | +3.3 +- 2.0 % (+6, +1) | +0.6 +- 1.7 % (+2, 0) | +1.3 +- 3.6 % (0, +3) | +2.3 +- 3.0 % (+1, +4) | +1.8 % |
| pair rmix + .5 SQF-X | +3.5 +- 3.3 % (+2, +5) | +8.2 +- 3.1 % (+9, +8) | -1.0 +- 4.7 % (+1, -3) | +6.2 +- 5.1 % (+13, -1) | +5.0 % |
| REVE trio | +4.6 +- 2.5 % (+3, +7) | +3.1 +- 2.3 % (+4, +2) | +2.4 +- 3.4 % (+9, -4) | +0.3 +- 3.2 % (-8, +9) | +3.1 % |

15 of 16 pooled cells >= 0. Swapping ONE member for its lag version is worth about +2 to +5 %; no single cell beyond
2.6 SE. Fold 1 also has SQF-X lag: ens5 with both swapped +4.5 / +4.6 | +3.5 / -4.2 %. SQF-X lag fold 3 running
(ends ~21:30). Not yet measured: d6 and REVE-scratch with the lag (the full five-member lag ensemble).

# Proxy calibration against the sealed board (user, 11-10 evening: 3 submissions left, sub 2 is 8th)

## C00 rank bug in the clean-proxy scoring (`nc09_rankcheck.py`)
`logs/holdout_eval.py`, `logs/clean_*.py` (and `nc_lib.rank` until 20:50) compare the gallery scores `q @ gal.T` with the
target score recomputed as a separate dot product `(q * gal[t]).sum(1)`. The two differ in the last float bits, so the
target is counted above itself in 47.4 % of the queries (rank one too high). rmix ep10, fold 1, r=1: top-5 0.740 exact vs
0.672 legacy at 5k (0.297 vs 0.275 at 14k); top-1 0.182 vs 0.107 (0.070 vs 0.042). Every proxy number so far was a mix
of top-5 and top-4, top-1 was 40 % too low, and paired comparisons carried extra coin-flip noise.
Fix: read the target score from the same product (`S.gather(1, t[:, None])`); done in `nc_lib.rank`. The other session's
`logs/*.py` are NOT edited (in use there). All numbers below this line use exact ranks.

## C01 what the board says (`results/things_eeg/mi/nc/board_2026-11-10.csv`, top 50)
- Our ranks: aggregated top-5 8th, aggregated top-1 6th, per-epoch top-5 10th, per-epoch top-1 3rd.
- Sampling noise (N ~1.5M epochs, ~0.6M groups): per-epoch top-1 ~213 hits, SE 6.9 %; per-epoch top-5 SE 3.5 %;
  aggregated top-1 SE 7.9 %; aggregated top-5 SE 4.0 %. Eight teams are within one SE of our per-epoch top-1.
- Across teams, aggregated top-5 follows per-epoch top-5 (r 0.89; residual SD 6 %); top-1 adds nothing to the fit.
  Our aggregation gain (1.90) equals the top-10 mean (1.91): nothing is lost at aggregation. We are 8th-10th on top-5
  at both levels; the 3rd place on top-1 is inside its noise.
- Board shape (top 10 / top 50 mean): aggregation gain top-5 1.91 / 1.83, top-1 2.02 / 1.96; top-5 / top-1 per-epoch
  4.29 / 4.47, aggregated 4.09 / 4.21. Ours: 1.90, 1.88, 3.90 (z -2, mostly the high top-1 draw), 3.95.

## C02 which proxy setting reproduces the board shape (`nc08_calibrate.py 1,2,3`; sub-2 ensemble on the proxy, folds pooled)
Exact target ranks in the 14,367-image clean gallery; smaller galleries by hypergeometric thinning.

| setting | agg gain top-5 | agg gain top-1 | top-5/top-1 per-epoch | top-5/top-1 aggregated |
|---|---|---|---|---|
| sealed board (top-10 mean; ours) | 1.91; 1.90 | 2.02; 1.88 | 4.29; 3.90 | 4.09; 3.95 |
| proxy 5k, r=2 | 1.54 | 1.73 | 4.47 | 3.97 |
| proxy 5k, r=3 | 1.89 | 2.11 | 4.47 | 4.01 |
| proxy 5k, r=4 | 2.21 | 2.43 | 4.47 | 4.06 |
| proxy 14k, r=2 | 1.62 | 1.96 | 4.97 | 4.10 |
| proxy 14k, r=3 | 2.04 | 2.37 | 4.97 | 4.28 |
| proxy 14k, r=4 | 2.35 | 2.79 | 4.97 | 4.18 |

- 5k at r=3 reproduces all four statistics; the 14k gallery does not (per-epoch rank distribution too flat, top-1 gain
  too high). The sealed set has a 40k gallery but is harder per trial (our per-epoch top-5 is 4.4x chance there, 7x on
  the 5k proxy, 8x at 14k), so its rank statistics look like a smaller proxy gallery.
- Absolute levels are not matched by any setting (proxy extrapolated to 40k: per-epoch top-5 0.11 % vs sealed 0.055 %).
- At 5k r=3 (exact ranks, 3 folds): sub-2 ensemble 1.354, REVE trio 1.270, rmix alone 1.218, SQF-X 1.036.
- Recommended judge: 5k, r=3 top-5 (plus r=1 top-5), folds pooled, exact ranks, paired bootstrap. Top-1 is not a target.
- Not done: replaying the sub 1 -> sub 2 step (sealed +14.4 % aggregated top-5) on this judge, the only direct check.

## D12b lag results re-scored with exact ranks (same cached outputs)
Folds 1 + 3, only rmix swapped (ep10), 5k r=3 / r=4 | 14k r=3 / r=4: rmix alone +1.8 +- 3.5 / +5.8 +- 3.3 | +6.5 +- 5.6 /
+2.5 +- 5.0 %; ens5 +1.7 +- 1.8 / +2.9 +- 1.6 | +3.8 +- 3.3 / +1.9 +- 2.8 %; pair +5.8 +- 3.2 / +8.4 +- 3.0 | +3.7 +- 4.6 /
+4.5 +- 4.7 %; REVE trio +2.3 +- 2.3 / +1.8 +- 2.0 | +1.0 +- 3.4 / +2.6 +- 3.1 %. All 16 cells >= 0.
Fold 1, rmix and SQF-X both swapped: ens5 +4.9 +- 2.9 / +7.6 +- 2.9 | +6.1 +- 5.5 / +2.2 +- 4.8 %; pair +4.4 +- 5.0 /
+8.7 +- 4.6 | +7.5 +- 7.6 / +6.9 +- 7.4 %.

## D13 lag rmix by epoch, folds 1 + 3 pooled, exact ranks (`nd11_lag_epochs.py rmix mi_rmix_ho5k_lag5 1,3 8,10,12`)
Arm vs baseline at the same epoch; in ens5 the other four members stay at sub-6 budgets. Paired gain (per fold).

| epoch | alone 5k r=3 | alone 5k r=4 | in ens5 5k r=3 | in ens5 5k r=4 | in ens5 14k r=3 | in ens5 14k r=4 |
|---|---|---|---|---|---|---|
| 8 | +3.6 +- 3.4 % (+4, +4) | +2.2 +- 3.0 % | 1.522 -> 1.556 (+2.2 +- 1.8 %) | +2.4 +- 1.5 % | +2.8 +- 3.2 % | +3.2 +- 2.6 % |
| 10 | +1.8 +- 3.5 % (-5, +10) | +5.8 +- 3.3 % | 1.526 -> 1.552 (+1.7 +- 1.8 %) | +2.9 +- 1.6 % | +3.8 +- 3.3 % | +2.1 +- 2.8 % |
| 12 | -2.9 +- 3.7 % (-10, +6) | +5.4 +- 3.7 % | 1.544 -> 1.556 (+0.8 +- 1.9 %) | +0.9 +- 1.8 % | +0.3 +- 3.4 % | -0.8 +- 2.8 % |

In the ensemble the lag rmix scores the same at epochs 8, 10, 12 (1.556 / 1.552 / 1.556 at 5k r=3) while the baseline
creeps up with epochs, so the gain is about +2 % at 8-10 and about +1 % at 12. Small and never negative at 5k.

## D14 REVE-d6 with the lag, and all lag members together (fold 1, 22:35)
`nd11_lag_epochs.py d6 mi_d6_ho5k_lag5 1 8,10` (exact ranks, lag arm vs its baseline at the same epoch):
alone 5k r=3 ep8 1.252 -> 1.372 (+9.6 +- 5.5 %), ep10 1.348 -> 1.428 (+5.9 +- 5.5 %); in ens5 5k r=3 ep8 +4.7 +- 2.7 %,
ep10 1.548 -> 1.620 (+4.7 +- 2.8 %), r=4 +2.4 / +9.7 %; 14k r=3 +3.9 / +7.4 %. All 16 cells >= 0. Largest lag effect of
the three members tested.

`nd12_all_lag.py 1`: ens5 (fold 1, ep10 / ep10 / ep40) with members swapped for their lag versions, 5k r=3 / r=4:

| lag members | 5k r=3 | 5k r=4 | 14k r=3 |
|---|---|---|---|
| rmix | +1.6 +- 2.6 % | +4.8 +- 2.6 % | +7.4 +- 5.5 % |
| SQF-X | +3.6 +- 2.4 % | +5.5 +- 2.3 % | +3.4 +- 3.3 % |
| d6 | +4.7 +- 2.8 % | +9.7 +- 2.8 % | +7.4 +- 4.3 % |
| d6 + rmix + SQF-X | 1.548 -> 1.700, +9.8 +- 3.5 % | +11.3 +- 3.7 % | +11.5 +- 6.4 % |

The three effects add up (1.6 + 3.6 + 4.7 = 9.9). One fold; rmix replicated on fold 3 (D12b), SQF-X fold 3 pending.
Decision: build an all-lag ens5. Finals: rmix lag @10 (other session, job 72829), d6 lag @12 with an ep10 checkpoint
(job 72857, 22:35); SQF-X, REVE-scratch and plain SQF lag finals follow as GPUs free.

# Round 3 (22:10, new goal): challenge-first search, nothing imported from the THINGS-EEG2 round

## E00 the sealed set is about twice as hard as the clean proxy predicts
Proxy sub-2 ensemble extrapolated to a 40k gallery (C02 table): per-trial top-5 0.113 % (0.10 % with slope -1), r=3
0.246 %. Sealed board, sub 2: per-trial 0.0554 %, aggregated 0.1054 %. Reaching the sealed level from the 14k proxy
needs a log-log slope of -1.58, impossible by gallery growth alone (slope >= -1). Not the concept novelty of the sealed
images: the concept-holdout REVE-d6 (500 new concepts, 10 images each in the gallery) is only ~11 % below the
image-holdout one (1.06 vs 1.20, legacy ranks). The preprocessing is the organisers' own (cache built from the
neuralbench extractor). Open causes: image domain outside THINGS (at most 9.4k THINGS images are unused, the gallery is
~40k), or a cohort difference. Not measurable offline. Hit rate is about linear in amplitude SNR here (r=3 -> 4:
x1.15 SNR, x1.17 hits), so the +18 % to the top of the board is about +16 % single-trial SNR.

## E01 what the hits are made of (`ne01_hits.py`, ens5 at sub-6 budgets, folds 1-3, 5k, r=3, exact ranks)
- Repeat weighting. The grader averages raw outputs within (person, image), so the trial norm is its weight. Equal
  weights -7.6 +- 2.0 %, |y|^0.5 -1.0 %, |y|^1.5 -1.7 %, |y|^2 -1.3 %, |y|^3 -4.9 %; members unit-normalised per trial
  -5.5 %. The norm of the summed output is a real confidence (it grows when members agree) and its natural exponent is
  the optimum. Closed.
- Which images are hit. 87 % of the images are never hit by 15 people (81 % if hits fell at random); the top 10 % of
  images give 87 % of the hits. An image's ease (-log(1 + rank)) correlates .47-.54 across folds (disjoint people,
  different models): decodability is a stable property of the image. Atypical images are the hits: top-5 by quintile of
  cosine to the gallery mean 2.57 / 1.51 / 1.23 / 1.08 / 0.55 %; lowest density quintile 2.18 % vs ~1.2 %.
- Which target directions are decoded. Correlation prediction-target per principal direction of the gallery: 0.36,
  0.31, 0.27 (3-4), 0.17 (5-8), 0.13 (9-16), 0.09 (17-32), 0.05 (33-64), 0.037, 0.024, 0.014, 0.006 (513-1536); the
  profile is identical across folds (r .98). But the weak directions carry the retrieval in aggregate: keeping only the
  first 512 directions -21 %, 256 -42 %, 64 -63 %; removing the first one -30 %.
- A linear readout for new people (fitted on folds 1-2 x half the images, tested on fold 3 x the other half, 11.9k
  gallery): ridge -89 %, weights r^0.5 -32 %, r^1 -66 %, r^-0.25 -39 %, gallery-sd^+-0.25 about -20 % either way,
  minus .25 mean prediction -2 %. Every re-weighting loses. With the measured numbers the LDA weight r / (sd_pred
  sd_gallery) is the same for the first direction (29) and the last 1,000 (26): the contrastive output is already the
  optimal metric across the spectrum. Static output maps are closed, with the reason.

## E02 can a batch find its cohort latency without labels? (`ne02_selfalign.py`, Alljoined-only d6 + rmix, fold 3)
Motivation: E00 + D03 (a 5-sample offset halves accuracy and a shift restores it).
- THINGS people (true offset about +5): per-batch argmax of member agreement +4.4 (SD 0.7), of the batch-mean ERP
  match to the Alljoined key +5.2 (SD 1.7), of the output norm +3.4. Accuracy is flat over +3..+6.
- Held-out Alljoined people (true offset 0): agreement -0.8 (SD 1.2), key match -0.5 (SD 1.25), norm -1.5; 6-7 % of
  batches land at |shift| >= 3. Accuracy r=1: 0.72 at 0, 0.66 / 0.67 at -1 / +1, 0.64 / 0.61 at -2 / +2.
Label-free statistics recover a large cohort offset but are biased and noisy near zero: usable only as a safety net
with a dead zone (act at |shift| >= 3), cost about -1.5 % when there is no offset. Not built into a submission.

## E03 re-rendering a new person through the training people's channel covariance (`ne03_recolor.py 1`, REVE-d6, fold 1)
The optx members are trained with operator transfer, so a covariance-swapped trial is in-distribution. New people's
covariances are not outliers (distance to the mean training covariance .29-.41 vs .35 for training people).
Top-5 r=3 vs raw 1.320: person -> mean training covariance -7.9 +- 3.3 %; 64-batch covariance -> mean -8.2 %; single
trial (shrink .5 / .9) -> mean -4.2 / -0.9 %; person -> one training person -7.3 / -3.0 / -8.5 %; outputs averaged
over 4 training targets -2.1 % (person), -5.2 % (batch). The person's own covariance is the best input, like its own
key (N02). Stopped after REVE-d6. Closed.

## E05 / E06 who the new people are (`ne05_people.py`, `ne06_seen_new.py`; 15 held-out people, 5k, r=3, exact ranks)
- ens5 per new person: 0.22 % (sub-17) to 2.86 % (sub-02), a 13x range; mean 1.38, median 1.14; the best 5 people give
  56 % of the hits. Not explained by trial RMS or clamping. The person's mean output norm tracks it (r .83).
- Same person, new vs seen (decoded by a fold that trained on it; d6 + rmix + SQF-X): 1.37 -> 2.37 on average (x1.73),
  x1.2-2.3 for 14 of 15 people. r(new, seen) across people .96: weak people stay weak when seen (0.56 -> 0.94 for the
  weakest five, 2.35 -> 4.22 for the strongest five). The person gap is a roughly constant factor on top of a
  person-intrinsic signal quality; it comes from ~46k labelled trials of that person, which the sealed regime never
  gives. A cohort of 11 people has a +-17 % sampling spread from this alone.

## D15 lag verdict so far (23:10)
- SQF-X lag, fold 3 (`nd11_lag_epochs.py sqfx mi_sqfx_ho5k_lag5 1,3 40`): alone 5k r=3 1.110 -> 1.232, +11.0 +- 4.3 %
  (folds +12 / +10); in ens5 5k r=3 +2.6 +- 1.6 %, r=4 +3.9 +- 1.5 %, 14k +2.5 / +0.5 %. Replicated.
- rmix + SQF-X both lag, folds 1 + 3 (`nd12_all_lag.py 1,3`): ens5 5k r=3 1.526 -> 1.612, +5.6 +- 2.1 %; r=4 +5.1 +- 1.8 %;
  14k +4.7 / +3.2 %.
- REVE-d6 lag, fold 1, epoch 12: in ens5 5k r=3 1.532 -> 1.652 (+7.8 +- 3.1 %), r=4 +5.1 %; alone +0.6 / +6.1 %.
  In absolute terms the lag d6 in ens5 is 1.600 (ep8), 1.620 (ep10), 1.652 (ep12): budget 12 kept for the final.
- Finals running: d6 lag @12 (+ep10 checkpoint; job 72857), rmix lag @10 (72829), SQF-X lag @39 and plain SQF lag @33
  (both inside mi_gpu since 22:58). REVE-scratch lag follows when a GPU frees.

## E07 a linear mismatch remains between THINGS-EEG2 and Alljoined beyond the lag (`ne07_things_adapter.py`)
Frozen Alljoined-ONLY REVE-d6 (fold 3, ep10) as the instrument. A linear adapter on THINGS trials, initialised at the
5-sample lag, fitted through the model on five THINGS people x 5,000 trained images, tested on five OTHER THINGS people
x never-trained images (5k, exact ranks; lag only = 0.897 / 1.620 / 1.772 at r=1 / 3 / 4):

| adapter | parameters | r=3 | r=4 |
|---|---|---|---|
| per-electrode gain | 32 | 0 | 0 |
| temporal filter shared by all electrodes | 11 | +1.7 +- 3.4 % | +5.0 +- 3.5 % |
| temporal filter per electrode | 352 | 1.808, +11.6 +- 4.2 % | +13.1 +- 4.0 % |
| spatial re-mix at the lag | 1,024 | -0.2 +- 5.1 % | +7.4 +- 5.2 % |
| full spatio-temporal | 11,264 | +0.2 +- 5.2 % | -2.5 +- 4.8 % (lowest fit loss: overfits) |

The per-electrode filters keep their centre on the lag (no per-electrode latency) and have negative flanks: a
sharpening / high-pass, strong on fronto-central electrodes (DC gain ~0 at FCz, F1, Pz, CP2) and nearly absent on
occipito-parietal ones (0.9-1.0 at Oz, PO3, PO8, P2). Saved: `results/things_eeg/mi/nc/things_filter_e07.npy` (11 x 32).

## E08 control: no such filter helps new Alljoined people (`ne08_newperson_filter.py d6 1 3`)
Same fit through the JOINT REVE-d6 on its own held-out Alljoined people (fold 1), tested on never-trained images.
Shared filter: stays the identity, -1.5 +- 2.8 % (same people), -1.8 % (fold 3, other people and model). Per-electrode:
-3.3 +- 3.8 % (same people), -7.2 +- 3.6 % (fold 3). So E07 is a dataset mismatch, not a generic gain, and there is no
static input filter to deploy for new people.
New training flag `--things_filter <npy>` (train.py `lag_things`; a one-hot tap 5 reproduces `--things_lag 5`, checked).
Fold test queued for the next free GPU.

## D16 swap, do not add (`nd13_union.py 1,3`; ens5 5k, exact ranks)
Lag versions swapped in (d6 + rmix + SQF-X on fold 1, rmix + SQF-X on fold 3), folds pooled: r=3 1.526 -> 1.650,
+8.1 +- 2.3 %; r=4 +6.9 +- 2.1 %. Keeping the non-lag members as well (8 members on fold 1): +3.1 +- 1.8 % / +6.9 %
whether every member weighs 1 or each lag/non-lag pair shares one weight. The submission swaps.

## E04 a committee of person routes (`ne04_routes.py 1`, REVE-d6, fold 1)
A new trial sent through training person t's route by adding (k_t - K): single routes 0.73-1.17 % vs 1.35 % raw
(top-5, r=3); the mean of 8 routes -1.2 %; raw + mean of 8 routes +0.3 +- 2.8 %; raw + .5 x routes +2.7 +- 2.4 %
(r=4 +0.0 %). Foreign routes are worse and not diverse enough to pay for 9 forward passes. Closed.

## E09 a new-concept proxy from existing fold models (`ne09_newconcept.py 1`)
Queries = the 200 official test images (concepts absent from training) x fold-1 held-out people x 80 repeats, cut into
r=3 / r=4 groups; gallery = 200 test targets + 5k held-out (5.2k) or + 14.4k never-trained images (14.6k); cluster
bootstrap over (person, image). Noisier than the clean proxy (1,000 clusters; SE 4-8 % at 5.2k, 6-11 % at 14.6k).
- Level: ens5 REVE@10 0.908 % at 5.2k r=3, against 1.548 % for the same people on never-trained images of trained
  concepts (5k): -41 %. Part of it may be the test runs themselves (200 images repeated 80 times), so not all of it is
  concept novelty; the concept-holdout REVE-d6 was only -11 %. Either way the sealed images are of this kind.
- The lag holds on new concepts: d6 + rmix + SQF-X lag vs ens5, 5.2k r=3 +7.2 +- 4.8 %, r=4 +9.6 +- 4.4 %; 14.6k r=3
  +25.9 +- 11.3 %, r=4 +2.2 +- 8.0 %. Single swaps at 5.2k r=3 / r=4: rmix +3.4 / +8.3, SQF-X +3.0 / +1.3, d6 -1.3 / +3.1 %.
- REVE budget on new concepts: ens5 @8 -1.3 %, @12 +9.7 +- 4.8 % (5.2k r=3; r=4 +1.3 / +7.4 %); all-lag ens5 @8 -0.4 %,
  @12 +6.7 +- 4.5 % (r=4 0.0 / +2.4 %). Budget 12 is not penalised on new concepts here: the lag finals keep 12 for
  REVE-d6 and REVE-scratch.

## E10 the levers by strength of the new person (`ne10_weak.py`, folds 1-3, 5k, exact ranks)
Weak half (8 people, ens5 0.76 % at r=3) vs strong half (7 people, 2.10 %). Nothing reverses between the two:
- ens5 beats every member and subset in both halves (REVE trio -2.6 / -3.8 %, SQF pair -22 / -20 %).
- REVE budget: weak @8 +1.0 +- 3.6 %, @12 -0.9 +- 4.4 %; strong @8 -0.7 %, @12 +2.7 +- 2.6 %. Within noise.
- Lag: rmix + SQF-X (folds 1, 3) weak +3.8 +- 4.7 %, strong +6.0 +- 2.2 %; d6 + rmix + SQF-X (fold 1) weak
  +13.8 +- 7.4 % (r=1 +17.6 +- 5.4 %), strong +8.9 +- 4.0 %.
A harder sealed cohort does not call for a different ensemble; the lag helps weak people at least as much.

## E11 / E12 concept novelty is cheap; the test RUNS are what is hard (`ne11_test_reps.py 1`, `ne12_concept_vs_test.py`)
REVE-d6 ep10, fold 1 held-out people, top-5 % r=1 / r=3, exact ranks:
| queries | model | gallery | r=1 / r=3 |
|---|---|---|---|
| new images of trained concepts | image-holdout | its 5k | 0.690 / 1.348 |
| new concepts, training-split runs | concept-holdout | its 5k (10 per concept) | 0.652 / 1.152 |
| same, 500 concepts x 3 images + 9.4k others | image-holdout (trained concepts) | 10.9k | 0.378 / 0.617 |
| same | concept-holdout (new concepts) | 10.9k | 0.350 / 0.720 |
| 200 test images (test runs) | image-holdout | 200 + 5k | 0.329 / 0.842 |
| 200 test images (test runs) | concept-holdout | 200 + 5k | 0.396 / 1.015 |
A new concept costs 5-15 % (nothing with a matched gallery). Trials of the test runs are ~45 % harder at r=1 for both
models, flat over the repetition index (ens5 r=1 by tens of repeats: .43 .41 .18 .44 .41 .37 .33 .34): a property of
those runs (200 images cycling 80 times), not of concept novelty. So E09's level says nothing about the sealed set,
its paired comparisons stay valid, and E00 is not explained by new concepts within THINGS.

## E13 the E07 filter is a property of the datasets (`ne13_filter_transfer.py`)
The per-electrode filter fitted through the Alljoined-only REVE-d6, applied to THINGS trials decoded by the
Alljoined-only rmix (other architecture; five test THINGS people, never-trained images): lag only 0.839 / 1.480 / 1.648
-> filter 0.886 / 1.600 / 1.776 (r=1 / 3 / 4), r=3 +8.1 +- 4.7 %, r=4 +7.8 +- 4.2 %.
- Fold 3 (`ne09_newconcept.py 3`, lag rmix + SQF-X only): ens5 @12 vs @10 +6.8 +- 2.9 % (5.2k r=3), +8.5 +- 3.3 % (r=4); @8 -2.1 %; rmix + SQF-X lag +3.0 +- 3.1 % / +2.0 %; all-lag @12 vs @10 +2.6 / +4.2 %. Budget 12 confirmed on new concepts on both folds.

## E14 the test runs are not harder because of their key (`ne14_testrun_key.py 1`)
Test-run keys differ from training-run keys in amplitude (norm ratio 0.8-1.9 by person) but not in shape (corr .88-.96)
or latency. Through d6 + rmix + SQF-X: test trials as recorded 0.361 / 0.873 (r=1 / r=3); test key replaced by the
person's training-run key 0.312 / 0.808 (-13.5 +- 3.6 %); key removed 0.193 / 0.369; training-run trials given the
test key 0.764 / 1.420 vs 0.791 / 1.456. The run's own key is again the best input; the cause of the harder test runs
stays open (and is not needed: the sealed protocol has ~2.4 repeats, like the training split).

## E15 the two BatchNorm paths of SQF-X, without and with the lag (`ne15_bn_paths.py`, fold 1, new Alljoined people)
| model | Alljoined statistics (deployed) | THINGS statistics | both paths averaged | cos between the two outputs |
|---|---|---|---|---|
| SQF-X | 0.566 / 1.100 / 1.336 | 0.200 / 0.204 / 0.228 (-81 %) | 0.489 / 0.884 / 1.104 | 0.21 |
| SQF-X + lag | 0.614 / 1.232 / 1.316 | 0.411 / 0.696 / 0.788 (-43 %) | 0.645 / 1.256 / 1.416 (+1.9 +- 5.5 %) | 0.56 |
The lag brings the two paths much closer, but the running statistics still differ (mean shift .58, variances x2.5:
THINGS trials have half the RMS), so the per-dataset BatchNorm stays necessary. Unknown subjects use the Alljoined
path (`_auxiliary_mask`), checked.

## E16 several checkpoints of one run as members (`ne16_multi_epoch.py`, fold 1)
REVE members = mean output of epochs 10 and 12: +0.0 +- 2.2 % vs ens5 @12 (r=4 +0.7 %); epochs 8, 10, 12: -2.8 %. Closed.

## E07b richer dataset adapters do not beat the per-electrode temporal filter (`ne07b_things_adapter.py`)
Temporal per electrode + spatial re-mix fitted jointly: r=1 0.988 (+10 %) but r=3 -1.0 +- 5.3 %, r=4 +5.2 %: the
spatial part fits the five fitting people. The E07 filter (352 parameters) is the one tested in training:
`mi_d6_ho5k_tfilt` fold 1 (job 72886) and fold 3 (mi_gpu, after the SQF finals), `mi_rmix_ho5k_tfilt` fold 1 (72909).

## E17 an oracle input operator per new person recovers a third of the person gap (`ne17_person_adapter.py 1 d6`)
Per held-out person of fold 1, fitted through the frozen joint REVE-d6 on that person's trials of 5,000 trained
images (labels used), tested on the same person x never-trained images (none: 0.690 / 1.348 / 1.520 at r=1 / 3 / 4):
- per-electrode temporal filter (352 parameters): 0.757 / 1.416 / 1.672, r=3 +5.0 +- 4.6 %, r=4 +10.0 +- 4.3 %;
- + 32x32 spatial re-mix (1,376 parameters): 0.776 / 1.668 / 1.976, r=3 +23.7 +- 6.2 %, r=4 +30.0 +- 5.9 %; every
  person gains (r=3: sub-02 2.50 -> 3.10, sub-04 0.94 -> 1.06, sub-05 1.98 -> 2.58, sub-11 0.52 -> 0.60, sub-15 0.80 -> 1.00).
For reference new -> seen is x1.73 (E06). A third of the person gap is a linear, mostly spatial misalignment of the input.

## E18 the operators are person-specific (`ne18_operator_structure.py`)
|M - I| 1.6-2.1 (Frobenius), cosine between people's (M - I) about 0.1, five comparable singular values. Own spatial
operator alone +12.8 +- 5.3 % (r=3); half strength +4.7 %; symmetric part only -7.1 %; the mean of the other four
people's operators -12.8 +- 3.5 %; another person's operator -20.5 +- 4.4 %. No shared operator to deploy; a foreign
operator moves a person about as far as the person gap itself.
Consequence: nothing for test time in the sealed regime (no labels, no person grouping), but M_t^-1 M_s is a
decoder-relevant re-rendering of person s as person t. New augmentation `--func_transfer_prob p --func_transfer_ops
<files>` (module/augment.py `functional_operator_transfer`, takes precedence over the covariance operator transfer on
its batches; unit-checked). Operators for training: `ne19_fit_operators.py <fold>` (spatial only, each person through
the fold that holds it out). Fold-1 test `mi_d6_ho5k_lag5_func` = d6 + lag + p .3 with the operators of the fold-2 and
fold-3 people (job 72943), against `mi_d6_ho5k_lag5`.

## Queue at 00:25
517: d6 lag final -> `mi_d6_ho5k_lag5_func` f1 (72943) -> `mi_rmix_ho5k_tfilt` f1 (72945). 513: rmix lag final ->
`mi_d6_ho5k_lag5` f3 (72885) -> `mi_d6_ho5k_lag5_tr10` f1 (72944: lag + 10 test reps of the training people, to check
that the finals' test-rep trials do not hurt new images). mi_gpu: SQF-X / SQF lag finals -> `mi_d6_ho5k_tfilt` f1.
515: REVE-scratch lag final. Packaging: sub12 (4 lag + scratch@12) and sub13 (5 lag) by `scripts/mi/finish_lag_sub.sh`.

## E19 spatial operators for all 15 held-out people (`ne19_fit_operators.py 1|2|3`, spatial only, own operator vs none)
fold 1: 0.690 / 1.348 / 1.520 -> 0.776 / 1.548 / 1.868 (r=3 +14.8 +- 5.8 %, r=4 +22.9 +- 5.3 %);
fold 2: 0.549 / 0.972 / 1.196 -> 0.646 / 1.152 / 1.372 (+18.5 +- 7.3 %, +14.7 +- 6.3 %); sub-17 0.24 -> 0.58;
fold 3: 0.716 / 1.340 / 1.680 -> 0.805 / 1.532 / 1.820 (+14.3 +- 5.6 %, +8.3 +- 4.5 %).
Replicated on three folds. Operators are well conditioned (cond 2.6-5.3) except sub-17 (47), which is excluded as a
re-rendering target. Files `results/things_eeg/mi/nc/e19_ops_f<k>.pt`.

## E20 the operators are not bad-electrode suppressors (`ne20_badchan.py`)
Residual variance of each electrode given the other 31 (population regression): several people have electrodes at
5-14x the population residual (sub-04 CP4, sub-10 PO3, sub-13 CP1, sub-14 P1, sub-03 P8) but they are not the weak
decoders (sub-10 is strong; sub-17, the weakest, peaks at 3.1) and the fitted operators do not down-weight those
electrodes (their least-read electrodes have ordinary residuals). No bad-channel repair lever.

## 00:50 after the interactive job moved to node 513
Lost: the local analyses and watchers (restarted). Jobs 72885 / 72944 failed at launch (`env bash` under sbatch: "Key
has expired"; launcher fixed). Now: `mi_d6_ho5k_lag5` f3 on Slurm (72971), `mi_d6_ho5k_lag5_func` f1 on the local GPU,
`mi_d6_ho5k_lag5_tr10` f1 after the d6 lag final (72972), `mi_rmix_ho5k_tfilt` f1 after 72971 (72973),
`mi_d6_ho5k_tfilt` f1 in mi_gpu after the SQF finals. sub10 and sub11 (other session) are packaged.

## D17 REVE-d6 lag replicated on fold 3; all three lag members on two folds (02:30)
`nd11_lag_epochs.py d6 mi_d6_ho5k_lag5 1,3 8,10,12` (lag arm vs its baseline at the same epoch, exact ranks, pooled):
alone 5k r=3 +7.9 +- 3.9 % (ep8), +7.7 +- 3.8 % (ep10, folds +6 / +10), +4.9 +- 4.1 % (ep12); in ens5 5k r=3
+3.1 +- 1.8 % (ep8), +3.5 +- 1.8 % (ep10, +5 / +2), +5.9 +- 2.0 % (ep12, +8 / +4); r=4 +4.1 / +5.7 / +3.0 %;
14k r=3 +5.4 / +5.7 / +0.6 %. 23 of 24 pooled cells positive.
`nd12_all_lag.py 1,3` (ens5 REVE@10 -> d6 + rmix + SQF-X lag, folds 1 + 3 pooled):
| d6 lag epoch | 5k r=3 | 5k r=4 | 14k r=3 | 14k r=4 |
|---|---|---|---|---|
| 10 | 1.526 -> 1.666, +9.2 +- 2.4 % | +7.0 +- 2.2 % | +7.5 +- 4.0 % | +1.9 +- 3.5 % |
| 12 | 1.526 -> 1.670, +9.4 +- 2.5 % | +8.2 +- 2.3 % | +7.2 +- 4.0 % | +5.1 +- 3.7 % |
The all-lag ensemble is confirmed on two folds at ~3.8 SE on the judge metric.

## F01 do the finals' test-repeat trials hurt? (`mi_d6_ho5k_lag5_tr10` fold 1 = d6 lag + `--train_test_reps 10`, 03:25)
The finals add 10 of the 80 test repeats of every training person (user call of 2026-11-05, before the clean proxy
existed); the fold models never did. Against the same run without them (`BASE=mi_d6_ho5k_lag5 nd11_lag_epochs.py`):
alone 5k r=3 -1.5 / -5.3 / -4.5 % (ep 8 / 10 / 12, SE ~5), r=4 +3.2 / -2.5 / -1.7 %; in ens5 5k r=3 0.0 / -2.2 / -1.5 %
(SE 2.5), r=4 +0.6 / -5.0 / -3.2 %; 14k r=3 -5.0 / -7.5 / -4.5 % (SE 4), r=4 -3.8 / -2.2 / -12.4 %. 20 of 24 cells
negative, none significantly positive. One fold, one member: a 2-5 % cost is plausible (the 200 test images get 250
trials each from 25 people in runs that E12 showed to be of a different kind). Replication: fold 3 submitted (d6).

## E22 functional operator transfer as an augmentation: no gain (`mi_d6_ho5k_lag5_func` fold 1 vs `mi_d6_ho5k_lag5`)
d6 + lag + `--func_transfer_prob 0.3` with the operators of the 10 fold-2/3 people (sub-17 excluded as a target).
alone 5k r=3 +9.0 +- 4.7 % (ep8), -6.2 +- 4.4 % (ep10), -1.8 +- 5.4 % (ep12); in ens5 5k r=3 +3.5 / -1.2 / -2.4 %
(SE 2.4), r=4 +1.1 / -2.3 / -2.8 %; 14k mixed. Faster at epoch 8, nothing left at 10-12: the best cell (ep8, 1.656 in
ens5) equals the lag-only run at ep12 (1.652). Re-rendering 10 people into each other does not widen the generic
route. The oracle operators (E17-E19) stay a measurement of the person gap, not a lever.

## E23 the dataset filter adds nothing to the lag in training (`mi_d6_ho5k_tfilt`, `mi_rmix_ho5k_tfilt`, fold 1, 03:50)
`--things_filter` (lag + per-electrode FIR of E07) against `--things_lag 5`, same recipe and seed:
- REVE-d6: alone 5k r=3 +5.5 / +3.1 / -3.6 % (ep 8 / 10 / 12), in ens5 5k r=3 -0.3 / -1.2 / -1.7 % (SE 2.4), r=4
  +2.4 / -2.7 / -5.7 %, 14k r=3 -5.6 / -1.3 / -3.2 %.
- rmix: alone 5k r=3 -6.4 / -5.3 / +8.4 %, in ens5 5k r=3 +2.0 / -1.0 / +2.6 %, r=4 -3.2 / -3.7 / -3.5 %, 14k r=3
  -7.5 / -9.4 / -4.4 %.
No gain, slightly negative. The filter makes THINGS people more decodable by an Alljoined-only model (E07, E13) but
the joint model does not need it once the timing is aligned. The lag alone is the dataset-integration lever. Closed.
sub12 (`sub12_ens5_lag4_scr12_sum.zip`: d6 lag @12, rmix lag @10, SQF-X lag @39, scratch @12, SQF lag @33) packaged
03:34, fold-1 replay 33.05 / 33.07 (batch 64 / 37), ingestion check clean.

## F01b the test-repeat cost does not replicate: neutral (06:15)
lag + `--train_test_reps 10` against lag only, in ens5, 5k r=3 / r=4 (other cells in the watcher output):
- REVE-d6, folds 1 + 3 pooled: ep8 +0.4 / -1.7 %, ep10 +0.4 / -3.4 %, ep12 -2.6 +- 1.9 / -2.5 +- 1.7 %; fold 3 alone
  +1 / +3 / -4 % at r=3.
- rmix, fold 1: ep8 +1.8 / -3.0 %, ep10 +3.6 +- 2.6 / -1.1 %, ep12 +5.7 +- 2.8 / -1.5 %; 14k r=3 -4 to -6 %.
- SQF-X, fold 1 (ep40): -2.2 +- 2.1 / -3.5 +- 2.1 %; 14k 0.0 / -0.6 %.
At the budgets used (d6 @12, rmix @10, SQF-X): -2.6, +3.6, -2.2 % at r=3: mean about -0.4 %. The first-fold signal
was mostly noise; the test repeats are neutral within +-2 %. The existing finals stand. The two finals without test
repeats (`reve_d6_optx_things_lag5nt`, `rmix_reg_optx_things_lag5nt`) are equivalent spare members.

## Submissions packaged (06:50)
- sub12 `sub12_ens5_lag4_scr12_sum.zip`: d6 lag @12, rmix lag @10, SQF-X lag @39, REVE-scratch @12 (no lag), SQF lag @33.
- sub13 `sub13_ens5_lag5_sum.zip`: the same with REVE-scratch lag @12 (all five members lag-trained).
Proxy evidence: d6 + rmix + SQF-X lag vs the non-lag ens5, folds 1 + 3, 5k r=3 +9.4 +- 2.5 %, r=4 +8.2 +- 2.3 %; holds
on new concepts (E09) and for weak people (E10). REVE-scratch lag and plain SQF lag have no fold test yet
(`mi_scr_ho5k_lag5`, `mi_sqf_ho5k_lag5`, fold 1, running).

## G: other leads (from 06:40, GPUs idle after the finals)
- G01 `mi_d6_ho5k_lag5_jit3` f1 (job 73153): d6 lag + `--time_jitter 3` (new flag: each training trial shifted by a
  random -3..+3 samples). Purpose: insurance against a latency offset of the sealed cohort (E00, E02): to be read at
  shift 0 (cost) and at shifts of +-2..4 samples on held-out people (robustness).
- G02 `mi_d6_ho5k_lag5_long` f1 (local): d6 lag to epoch 16 with checkpoints 12 / 14: does the lag move the REVE budget?
- `mi_sqf_ho5k_lag5` f1 (mi_gpu) and `mi_scr_ho5k_lag5` f1 (73148): fold tests of the two lag members without one.
- G03 `mi_d6_ho5k_lag5_atyp` f1 (mi_gpu): d6 lag + `--atypical_weight 0.5` (new flag: per-trial loss weight exp(-0.5 z), z = standardised cosine of the image target to the batch-mean target): the hits at large galleries are atypical images (E01); does emphasising them in the loss raise the hit rate?

## G01 time jitter: neutral in the ensemble at zero shift, large robustness to a latency offset (09:00)
`mi_d6_ho5k_lag5_jit3` (d6 lag + `--time_jitter 3`) vs `mi_d6_ho5k_lag5`, fold 1.
- Zero shift: alone 5k r=3 -3.5 / +3.6 / +10.4 +- 6.1 % (ep 8 / 10 / 12); in ens5 +1.2 / 0.0 / -2.7 +- 2.8 % (r=4
  +0.2 / -2.3 / -0.6 %).
- Held-out people shifted by n samples (`ng01_shift_robust.py 1 12`, d6 alone, top-5 % r=1 / r=3):
| shift | -4 | -3 | -2 | -1 | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| lag only | .38 / .68 | .49 / .85 | .58 / 1.14 | .66 / 1.23 | .64 / 1.34 | .66 / 1.32 | .61 / 1.16 | .58 / 1.00 | .46 / .85 |
| lag + jitter 3 | .63 / 1.22 | .73 / 1.32 | .76 / 1.40 | .77 / 1.50 | .75 / 1.48 | .75 / 1.40 | .71 / 1.41 | .67 / 1.34 | .65 / 1.16 |
The jitter model keeps ~90 % of its accuracy at +-3 samples and ~80 % at +-4, where the lag-only model keeps 63-75 %
and 51-63 %. A +-4-sample (33 ms) offset would cost the current models about 2x: the size of E00's unexplained gap
(not evidence that this is the cause). Insurance candidate: an all-lag + jitter ensemble.
Launched 09:05: finals `*_lag5jit` (d6 @12 job 73280, REVE-scratch @12 job 73279, rmix @10 job 73281 after the scratch
fold test, SQF-X @39 and SQF @33 in mi_gpu) and fold-1 tests `mi_rmix_ho5k_lag5_jit3`, `mi_sqfx_ho5k_lag5_jit3` (local).

## G02 the lag does not move the REVE budget (`mi_d6_ho5k_lag5_long` fold 1)
d6 lag in ens5, 5k r=3 / r=4: epoch 12 1.652 / 1.880, epoch 14 1.572 / 1.888, epoch 16 1.624 / 1.884. Budget 12 stays.

## D18 plain SQF with the lag: neutral (`mi_sqf_ho5k_lag5` fold 1, ep32)
alone 5k r=3 -3.0 +- 6.3 %, r=4 -13.3 +- 5.0 %; in ens5 5k r=3 +0.3 +- 2.4 %, r=4 -1.4 +- 2.2 %, 14k +6.8 / -0.6 %.
No gain and no loss in the ensemble: sub12 / sub13 are unaffected by this member's lag.

## G03 atypical-image loss weighting: strongly negative, by construction (`mi_d6_ho5k_lag5_atyp` fold 1, 09:20)
alone 5k r=3 -50 / -48 / -43 % (ep 8 / 10 / 12); in ens5 -10 / -10 / -11 % (r=4 -16 / -16 / -14 %). In-run 200-way
top-5 7 % against 21 % from epoch 1. The per-anchor loss path is numerically identical to the default one (checked),
so the weights are the cause: weighting an anchor by a function of its LABEL (w in 0.3-3.5) is a label-prior shift,
the learned posterior becomes p(i | x) w(i), and at this SNR the posterior is mostly prior, so every prediction leans
to atypical images. Not a lever; any "emphasise the retrievable images" loss needs a logit correction that undoes it.

## D19 REVE-scratch lag is positive; the exact sub12 / sub13 compositions on fold 1 (10:10)
`mi_scr_ho5k_lag5` fold 1 vs non-lag at the same epoch: alone 5k r=3 +6.7 / +5.9 / +7.0 % (ep 8 / 10 / 12; r=4 +10.5 /
+4.9 / +6.2 %); in ens5 5k r=3 +2.6 / +3.1 / +1.8 % (SE 2.7), r=4 +1.8 / +2.8 / +1.3 %; 14k r=3 +3.4 / +7.4 / +3.9 %.
23 of 24 cells >= 0.
`nd15_five_lag.py` (fold 1, top-5 % r=3 / r=4):
| ensemble | 5k | vs no lag, same budgets | 14k | vs no lag, same budgets |
|---|---|---|---|---|
| no lag, sub-10 budgets (d6 12, rmix 10, scr 12) | 1.568 / 1.764 | | 0.628 / 0.728 | |
| sub12 composition (scratch without the lag) | 1.652 / 1.916 | +5.4 +- 3.8 / +8.6 +- 3.8 % | 0.676 / 0.756 | +7.6 / +3.8 % |
| sub13 composition (all five lag) | 1.672 / 1.940 | +6.6 +- 4.3 / +10.0 +- 4.1 % | 0.696 / 0.804 | +10.8 +- 7.1 / +10.4 +- 6.8 % |
sub13 >= sub12 in every cell. Every lag member now has a fold test (d6, rmix, SQF-X on two folds; scratch and SQF on
one; SQF neutral, the other four positive). sub13 is the upload.

# Round 4 (11-11, 11:10): inside the challenge members
User, 11:05: the E round was measurement from outside (only E04, E15, E17-E20 opened a model). This round asks where,
inside the lag members, a new person's trial stops being decoded like a trained person's. Fold models, never-trained
images (HELD, 5,000), new = the fold's five held-out people, seen = five training people (the next fold's), exact ranks.

## M01 a new person gets the same shared code; the trained-person advantage is private (`nm01_layers.py`, rmix lag ep10)
After every block the member's own readout (token mean -> head) gives a 1536-d pattern per trial. It is ranked against
(a) the image embeddings and (b) "neural galleries": other people's patterns for the same image (person mean removed),
no image embedding involved. Top-5 % among 5,000 images, r=3 (r=1 in brackets); own = the person's other repeats.

| fold | after block | image emb, seen / new | 10 other training people, seen / new | own repeats r=1, seen / new |
|---|---|---|---|---|
| 1 | 3 | 0.38 / 0.38 | 0.58 / 0.83 | 0.50 / 0.49 |
| 1 | 4 | 0.85 / 0.64 | 1.10 / 1.09 | 0.62 / 0.54 |
| 1 | 5 (output) | 2.17 / 1.34 (0.89 / 0.66) | 1.58 / 1.53 (0.71 / 0.64) | 0.75 / 0.56 |
| 3 | 3 | 0.48 / 0.43 | 0.94 / 1.13 | 1.05 / 0.62 |
| 3 | 4 | 0.97 / 0.70 | 1.35 / 1.26 | 1.10 / 0.60 |
| 3 | 5 (output) | 2.14 / 1.40 (1.14 / 0.65) | 1.66 / 1.48 (0.79 / 0.64) | 0.99 / 0.62 |

- Against other people's patterns a new person scores like a trained one at every depth (1.53 vs 1.58, 1.48 vs 1.66):
  the part of the code that is common to people is computed equally well for a person the model never saw.
- The x1.5-1.6 person gap exists only against the image embeddings and appears in blocks 4-5. Trained people also have
  a more repeatable code of their own (own-repeat gallery .75-.99 vs .56-.62). The trained-person advantage is a
  private code that the model has learned to read, not a better shared code.
- Matched 4-person galleries (seen4 / new4) show no "new people are less aligned" effect (noise-level differences).
- For a new person the neural gallery beats the image embeddings (1.53 vs 1.34, 1.48 vs 1.40): see M04.
(First version with raw flattened activations and cosine was uninformative: 0.1-0.5 % at every layer.)

## M02 the private code is read by the feed-forward layers, mostly late (`nm02_components.py`)
Knock-out = a component's output replaced by its mean over the batch (1,000 trials mixing people). r=3 top-5 against the
image embeddings, change vs intact, new / seen (SE 3-5 / 2-4 points):

| member, fold | attention, per block | feed-forward, blocks before the last | last feed-forward | last attention |
|---|---|---|---|---|
| rmix f1 (1.30 / 2.11) | -7 -6 -11 -6 -3 / -18 -10 -9 -11 -11 | +4 +3 +2 +2 -2 / -10 -5 -8 -10 -28 | -15 / -39 | -1 / -5 |
| rmix f3 (1.46 / 2.28) | -19 -5 -18 -6 -4 / -21 -17 -27 -8 -3 | -10 -4 -6 -5 -9 / -11 -10 -10 -18 -20 | -19 / -37 | +3 / -4 |
| REVE-d6 f1 (1.37 / 2.53) | -24 -13 -18 -13 -7 / -22 -14 -11 -21 -7 | -11 -8 -13 -9 -8 / -4 -13 -9 -16 -18 | -17 / -49 | 0 / -8 |
| SQF-X f1 (1.15 / 1.52) | -10 -2 -22 / -13 -24 -28; conv -38 -33 -32 / -46 -47 -49 | +7 +4 -1 / -11 -10 -9 (all three) | | |

- Attention (and SQF's conv) is shared machinery: about the same cost for new and trained people.
- Feed-forward layers carry about twice as much for trained people in rmix and SQF-X; in REVE-d6 only the late ones
  do. In every member the last feed-forward layer is where the gap is largest (-37..-49 vs -15..-19), and the last
  attention layer does nothing for new people.
- Combined (rmix f1): feed-forward of blocks 0-3 out: new 1.12 (-14 %), seen 1.22 (-42 %): the person gap falls from
  x1.62 to x1.09. SQF-X, all feed-forward out: f1 new 1.19 (+4 +- 5) / seen 1.15 (-24); f3 new 1.18 (-6 +- 4) / seen
  1.25 (-16): the gap closes at the new-person level.
- No knock-out helps new people beyond noise (mixed sign across folds): the private machinery is useless for them, and
  harmless. Removing it at inference is not a lever; and since nothing is broken on the new-person path, "suppress the
  person-specific part" training levers have no mechanism to gain (consistent with key-free N09, SubjectMix, E22).

## M03 trial confidence: the output norm already holds all of it (`nm03_confidence.py`, ens5 cached, folds 1-3)
Quality q = cosine(ensemble output, target). Oracle repeat weights exp(b z(q)): +29 / +51 / +69 % (b .25 / .5 / 1) at
r=3: large headroom. A boosted regressor on 51 label-free features (member norms, agreement of each member with the
rest, input RMS per electrode, peak, prestimulus RMS, band powers), fitted on two folds' people and applied to the
third: r(predicted, true q) .44 / .40 / .45, against .43 / .39 / .44 for the norm of the sum alone; input features
alone .08-.17. Re-weighting by it: -2.0 .. +0.1 % (all), -6.8 .. -1.0 % (input only). Closed: no better confidence is
available from the trial or from member agreement (same verdict as the October one, now in the sealed regime).

## M04 other people's outputs are a better gallery than the image embeddings, and it cannot be used (`nm04_forward_gallery.py`, rmix f1)
New people, raw queries, r=1 / r=3: image embeddings 0.656 / 1.340; mean output of the 15 training people for the
gallery image 0.760 / 1.576 (+16 / +18 %). The member's expected output for an image differs from its embedding in an
image-specific, person-independent way (cosine(d_j, mean output) .07), and matching against that expectation is the
better test. The grader's gallery is fixed, so this needs a forward model d_j -> expected output moved to the output
side. Linear forward model (ridge, cross-fitted over images, R2 .59-.65): gallery G d_j 0.93-1.00 (-25 %), and the
deployable form G^T y against d_j 0.35-0.81 (-40 .. -74 %). The systematic part is not linear in the embedding at
4,000 fitting images. Not deployable.

## M05 a full output matrix re-fitted on new-person outputs: negative (`nm05_stack.py`, ens5 cached, folds rotated)
W (1536 x 1536, identity at start) trained on the retrieval loss on out-of-fold outputs (two folds' new people x 2,500
never-trained images), tested on the third fold x the other 2,500 (+ 9.4k distractors): r=3 -8.1 +- 4.3 % after one
epoch, -13 .. -25 % later; r=1 -6 .. -15 %. It learns the fitting images. With E01, output maps are closed in every form.

## Round 4 verdict (12:20)
Mechanism: a new person is decoded through the code that is common to people, and that code is as good for them as for
trained people. Trained people get x1.5-1.85 more from a private code read by the feed-forward layers; reading a new
person's private code needs labels (E17: a labelled 32 x 32 input operator recovers a third of it). No component is
broken or harmful for new people, so there is no inference-time or "remove the person-specific part" lever. What can
raise the sealed score is a stronger shared code (more people, alignment between datasets = the lag) and robustness of
that code to cohort shifts (jitter).

## G04 rmix with time jitter, fold 1 (`mi_rmix_ho5k_lag5_jit3` vs `mi_rmix_ho5k_lag5`, 11:30)
alone 5k r=3 +5.3 / +6.7 / +12.9 % (ep 8 / 10 / 12, SE 5-6), r=4 +4.7 / +5.8 / +11.4 %; in ens5 5k r=3 +3.6 / +2.3 /
+5.7 % (SE 2.7), r=4 -3.5 / -2.4 / -3.0 % (SE 2.4); 14k r=3 -6.8 / -2.5 / -4.4 %, r=4 -3.9 / -1.1 / +2.9 % (SE 4).
Neutral in the ensemble at zero shift, as for d6 (G01).

## G05 jitter twins as swaps or as extra members: neutral (`nd16_jitter_union.py`, fold 1, vs the sub13 composition 1.672 / 1.940)
5k r=3 / r=4: swap d6 +2.4 / -0.8 %, swap rmix @10 +1.4 / -1.2 %, swap both -0.2 / 0.0 %, add d6 twin +1.0 / +0.6 %,
add rmix twin +1.7 / +4.1 %, add both +0.7 / +4.5 % (SE 2-3); 14k: swaps -6 .. +1 %, additions -5 .. +1.5 % (SE 4).
Jitter costs nothing at zero shift and adds nothing: sub14 (all five lag + jitter) is pure insurance against a latency
offset of the sealed cohort (G01). Launched 12:25: `mi_d6_ho5k_lag5_jit3_long` fold 1 (stop 16, saves 12 / 14): does
jitter move the REVE budget?

## G06 SQF-X with time jitter, fold 1 (`mi_sqfx_ho5k_lag5_jit3` vs `mi_sqfx_ho5k_lag5`, ep 40, 12:50)
alone 5k r=3 -4.6 +- 5.4 %, r=4 +5.8 +- 5.8 %, 14k +7.1 / +11.1 % (SE 10); in ens5 5k r=3 -4.5 +- 2.1 %, r=4
-3.5 +- 2.2 %, 14k -2.0 / 0.0 % (SE 4). Unlike d6 and rmix (neutral), the jitter SQF-X costs about 4 % in the ensemble
at zero shift on this fold (2 SE at 5k, nothing at 14k). The ensemble-under-shift run (`ng02_ens_shift.py`) decides
what sub14 keeps under an offset; a mixed sub14 (jitter REVE members, lag-only SQF members) is the alternative.
12:35 user: back to 4 GPUs in total; `mi_d6jit_f3` (73503) held until a GPU frees, `mi_rmixjit_f3` (73504) pending.
Research note for the user written 12:40-12:55: https://claude.ai/artifact/Uq8CHro9WJUHuZxHHuddHe

## G07 the ensemble under a cohort latency offset (`ng02_ens_shift.py`, fold 1, 5k, 13:10)
ens5 with d6 / rmix / SQF-X lag-only vs the same three lag + jitter 3 (REVE-scratch and SQF = non-jitter fold members in
both). Top-5 % r=1 / r=3 / r=4, paired gain of the jitter ensemble:
| shift (samples) | lag ens5 | jitter ens5 | gain r=3, r=4 |
|---|---|---|---|
| -4 | 0.461 / 0.860 / 1.016 | 0.682 / 1.212 / 1.492 | +40.9 +- 7.4 %, +46.9 +- 7.4 % |
| -2 | 0.740 / 1.416 / 1.604 | 0.817 / 1.572 / 1.824 | +11.0 +- 4.0 %, +13.7 +- 4.1 % |
| 0 | 0.822 / 1.684 / 1.916 | 0.825 / 1.676 / 1.984 | -0.5 +- 3.3 %, +3.5 +- 3.3 % |
| +2 | 0.720 / 1.440 / 1.768 | 0.791 / 1.536 / 1.912 | +6.7 +- 3.9 %, +8.1 +- 3.6 % |
| +4 | 0.550 / 1.100 / 1.352 | 0.669 / 1.296 / 1.600 | +17.8 +- 5.3 %, +18.3 +- 4.9 % |
Members alone at shift 0 (r=3): d6 1.34 -> 1.48, rmix 1.42 -> 1.52, SQF-X 1.23 -> 1.17.
Reading (user, 13:00: "jitter is overall negative no?"): at zero shift jitter is not a gain. The three-member swap is
neutral (-0.5 / +3.5 %); single swaps scatter around zero with a slightly negative mean (16 cells of G05 / G06: mean
-1.5 %, SQF-X -4 % at 5k). Its value exists only under an offset (+7..14 % at 2 samples, +18..47 % at 4), and there is
no positive evidence of one: every held-out Alljoined person peaks at zero shift (D05). sub13 stays the upload; sub14
is a hedge whose only use is to test the offset hypothesis on the board.
Housekeeping 13:05: the three watchers of the previous session had survived as orphans and ran in duplicate with the
relaunched ones (two `ng02` at once); the orphans were stopped, one packaging watcher for sub14 remains.

## G08 which members should carry the jitter: all of them (`ng03_mixed_shift.py`, fold 1, cached shifted outputs, 13:20)
Gain over the all-lag ensemble at the same shift, r=3 / r=4 (SE 2-7):
| composition | -4 | -2 | 0 | +2 | +4 |
|---|---|---|---|---|---|
| jitter d6 + rmix + SQF-X | +40.9 / +46.9 | +11.0 / +13.7 | -0.5 / +3.5 | +6.7 / +8.1 | +17.8 / +18.3 |
| jitter d6 + rmix, SQF-X lag-only | +32.1 / +38.6 | +6.2 / +9.5 | 0.0 / +1.3 | +5.0 / +6.8 | +14.5 / +17.5 |
| jitter d6 only | +14.0 / +20.5 | +4.2 / +5.0 | +0.5 / +1.9 | +1.4 / +2.7 | +6.5 / +8.6 |
| jitter rmix only | +21.9 / +21.3 | +3.1 / +4.2 | -0.2 / +5.0 | +4.7 / +4.5 | +8.0 / +8.0 |
| jitter SQF-X only | +10.2 / +10.6 | -2.3 / +2.7 | +0.5 / +1.9 | +1.7 / +3.4 | +7.6 / +4.4 |
At zero shift every composition is within noise of the all-lag ensemble (the -4 % of G06 for SQF-X was against other
members at the old budgets and does not show here); under a shift each jitter member adds protection. No reason for a
mixed zip: sub14 stays all five lag + jitter. User 13:15: will upload sub14 as the third submission.

# Round 5 (11-11, 13:25): noise alignment between the datasets
User, 13:22: once sub14 is ready, pivot away from jitter. New subject, continuing the dataset-integration line (the lag):
after the lag, timing matches but THINGS-EEG2 trials are far cleaner; the joint model decodes THINGS people ~8x better.
Hypothesis: as easy examples they teach the shared code little about decoding at Alljoined noise; noise-matched, they are
ten more Alljoined-like people.

## H01 the noise level is what separates the two datasets after the lag (`nh01_snr_match.py`)
THINGS trials (lag 5) scaled by g + real Alljoined single-trial noise = (a - b) / sqrt 2 of two repeats of one
(person, image) (no evoked response, no key; rms 1.2-1.6 against a THINGS trial rms of 0.64). Never-trained images, 5k,
exact ranks, top-5 % r=1 / r=3 / r=4.
| condition | (a) Alljoined-only d6, fold 3: THINGS people it never saw | (b) joint lag d6, fold 1: THINGS training people |
|---|---|---|
| Alljoined reference people | 0.635 / 1.180 / 1.400 (held-out) | 1.058 / 2.532 / 3.208 (training) |
| THINGS, lag only | 0.897 / 1.620 / 1.772 | 7.643 / 19.556 / 23.380 |
| THINGS x1 + Alljoined noise | 0.395 / 0.676 / 0.856 | 0.795 / 1.928 / 2.316 |
| THINGS x0.5 + noise | 0.225 / 0.316 / 0.424 | 0.281 / 0.396 / 0.632 |
| THINGS x0.31 + noise | 0.169 / 0.232 / 0.260 | 0.180 / 0.224 / 0.284 |
- At their NATIVE amplitude plus Alljoined noise, THINGS people land at Alljoined level (1.93 vs 2.53 for training
  people; 0.68 vs 1.18 for new ones): the x8-10 decodability gap between the datasets is the noise level. The image
  response of THINGS people is about 0.6-0.75 of Alljoined's in these units; the key-norm ratio (6.8 vs 2.1) does not
  give the evoked-amplitude ratio, so g = 0.31 was wrong (chance level).
- Decodability falls x10 for a x2.1 loss of amplitude SNR on training people: far steeper than the x1.15 -> x1.17
  relation at the Alljoined operating point (E00); the clean route is not the noisy route.
- Lever: `train.py --things_noise p` (new, default 0): with probability p a THINGS trial gets (a - b) / sqrt 2 of two
  Alljoined batch trials of one person (self-check on synthetic data: Alljoined rows untouched, added rms = Alljoined
  rms). Runs 13:40 in mi_gpu (two steps sharing its GPU): `mi_d6_ho5k_lag5_tn`, `mi_rmix_ho5k_lag5_tn` fold 1
  (= lag members + `--things_noise 1.0`), judged against the lag-only members alone and in ens5.

## Submissions packaged: sub14 (15:42)
`neurips_challenge/submissions/sub14_ens5_lag5jit_sum.zip` = all five members with `--things_lag 5 --time_jitter 3`
(d6 @12, rmix @10, SQF-X @39, REVE-scratch @12, SQF @33), scaled sum, no TTA.
sha256 3846a9b192477757b3eb9d43d08790ac11c523cc7b7efe893a93e57aef2ac8f3, 399 MB. Batch-size consistency of the package
on fold 1: 32.03 (batch 64) / 31.98 (batch 37); contract check clean (85 s). That replay is in-sample (the finals train
on fold 1's people): sub13 gave 33.69 there, sub12 33.05; the jitter members fit their training people a little less.
It says nothing about new people (fold tests: G01, G04-G08). User will upload it (decided 13:15).
Versus sub 2 (configs diffed): + lag, + jitter, REVE budgets 9 / 9 / 8 -> 12 / 10 / 12; members, data, SQF budgets and
predict() unchanged.
Pivot (user 13:22): jitter line closed. The two fold-3 jitter jobs that never started (73503, 73504) cancelled and
their watcher stopped; `mi_d6_ho5k_lag5_jit3_long` fold 1 ran to epoch 16 (checkpoints 12 / 14 on disk, not scored).

## H02 training THINGS people at Alljoined noise: no gain (`--things_noise 1.0`, fold 1, 16:05)
vs the lag-only twin at the same epoch (ep 8 / 10 / 12):
- rmix (`mi_rmix_ho5k_lag5_tn`): alone 5k r=3 -2.5 / -4.5 / +2.4 % (SE 5), 14k r=3 -2.6 / -11.3 / -11.8 %; in ens5 5k
  r=3 +1.8 / -1.0 / +1.8 % (SE 2.8), r=4 -2.8 / +3.3 / +1.1 %, 14k r=3 -6.8 / -2.5 / -8.1 %, r=4 +1.7 / +0.5 / +1.1 %.
- d6 (`mi_d6_ho5k_lag5_tn`): alone 5k r=3 -6.1 / -5.0 / +6.0 %, r=4 -8.0 / -11.8 / +0.5 %; in ens5 5k r=3 -1.5 / -1.0 /
  -3.4 % (SE 2.8), r=4 -0.2 / -5.0 / -1.1 %, 14k r=3 -8.8 / -6.9 / -1.3 %, r=4 +1.6 / +1.6 / -6.2 %.
In-run 200-image curves start 25-35 % behind and reach the twin at epochs 9-10. Verdict: neutral (rmix) to slightly
negative (d6, 10 of 12 ensemble cells below zero). The ten THINGS people give new Alljoined people about the same help
clean or noisy: their contribution does not depend on their noise level, so neither "noisier THINGS" nor (by the same
logic) "cleaner supervision" is a lever. Closed; half-dose run and the two queued fold-3 jobs stopped. Slurm queue: new
rtx3080 jobs are estimated to start on 14 Nov, so the working budget is the interactive node + mi_gpu.

# Round 6 (16:10): ensemble composition on the clean proxy
The five-member set was picked on 11-05 by a search under r=3 on the 200-image test proxy (14.53 vs 14.20 / 14.33 for
the sets with TSConv), before the clean proxy, the exact ranks and the lag. No TSConv member has ever been scored on the
clean proxy. Runs 16:10: `mi_tsc_ho5k_lag5` folds 1 (mi_gpu) and 3 (local) = tsc_dbn40_bn64_optx_things + lag, 40
epochs, checkpoints 24 / 29 (cv4 budget) / 32 / 36; scored by `ni01_new_member.py` as a sixth member and as a swap.

## I02 subsets and weights of the current five: nothing to gain inside the set (`ni02_composition.py`, folds 1 + 3, 16:30)
- Subsets (31): the five-member sum is the best; every four-member subset is below it (mean of 5k / 14k, r=3 / r=4:
  -1.5 % without d6, -2.2 % without rmix, -2.3 % without scratch or without SQF, -3.6 % without SQF-X); the best
  three-member set (d6 + rmix + SQF-X) -0.6 %. The best subset of fold 1 is the full set; fold 3's best (the trio)
  costs 2-3.5 % on fold 1. No member is dispensable.
- Weights: one member at x0.5 / x1.5 / x2, the REVE trio or the SQF pair at x1.5 / x2 / x3: every setting within
  +-2 % of uniform (best: SQF-X x1.5 +0.5 %, rmix x2 +0.2 %); x3 on either family -3 to -7 %. Uniform weights stay.
-> the ensemble still grows with members and is flat in weights: only a new, different member can add.
16:35 second candidate family started: `mi_plab_ho5k_lag5` fold 1 (mi_gpu, beside TSConv f1) = pretrained LaBraM
(recipe recovered from the pre-cleanup tag: gen --eeg_encoder_type LaBraM, lr 5e-4, batch 128, EMA, stop 8) + operator
transfer + lag, checkpoints 4 / 6. Loads 220 / 221 pretrained tensors on the current code, 6.1 M parameters.

## I03 a recurrent member (user 16:45: "something JEPA or mamba inspired?")
- Mamba itself is not practical: without compiled kernels (not allowed in the submission) transformers' pure-PyTorch
  MambaMixer costs 0.44 s per 64-trial step for 1.1 M parameters (about 27 h per run) and 0.6 GB per 64 trials.
- `RecurrentEEG` (module/eeg_encoder/hybrid_encoders.py, registered in HYBRIDS): TSConv stem -> 36 time tokens ->
  3 residual bidirectional GRU layers (d 160) -> flatten -> 64-d bottleneck head; 0.89 M parameters, 0.145 s per
  1024-trial step, self-check passed (shapes, gradients, eval batch-size invariance 2e-5). Same state-scan bias at 36 steps.
- Runs: `mi_rec_ho5k_lag5` fold 1 (local, 16:48; scripts/mi/job_rec.sh = gen RecurrentEEG, per-dataset BN, 40 epochs,
  operator transfer, lag, checkpoints 24 / 32 / 36); fold 3 queued on Slurm (73689) and to be run in mi_gpu when
  TSConv fold 1 ends, whichever comes first. LaBraM fold 3 queued (73690). Scored with `ni01_new_member.py`.
- JEPA: the loss is already latent prediction of a frozen image encoder; SSL pretraining has not mattered (scratch REVE
  = pretrained); the cross-person latent variant is second priority (xembed failed on THINGS-EEG2, SubjectMix dead here).

## I04 TSConv with the lag adds nothing (`mi_tsc_ho5k_lag5`, folds 1 + 3, `ni01_new_member.py`, 18:00)
Against the five-member lag ensemble (5k r=3 1.646, r=4 1.946; 14k 0.708 / 0.810), epochs 24-40:
- alone 0.96-0.99 at 5k r=3 (-40 %): the weakest single of all families on the clean proxy;
- as a sixth member: 5k r=3 +0.2 .. +1.6 % (SE 1.7), r=4 -1.2 .. -2.0 % (SE 1.5); 14k r=3 +0.3 .. +2.0 %, r=4
  -0.7 .. -1.7 %: zero net;
- swapped for any member: d6 -0.5 .. -6 %, rmix -1.4 .. -6 %, SQF-X -0.2 .. -8 %, REVE-scratch -1.4 .. -7 %,
  SQF +1.0 .. -5.6 %.
Dropping TSConv on 11-05 was right, and it holds with the lag on the clean proxy. Closed.
18:12 LaBraM fold 3 moved from the queue to the local GPU (73690 cancelled); recurrent fold 3 runs in mi_gpu (73689
cancelled by the chain). In-run 200-image curves, fold 1: LaBraM 20 26 28 29 29 29 (epochs 1-6; d6 lag 29-30),
recurrent 24-25 around epoch 30 (TSConv 27-28, SQF-X 28). GPU catchers queued on Slurm (start estimate 12 Nov):
`mi_d6_ho5k_lag5` and `mi_rmix_ho5k_lag5` fold 2 = a third fold of lag members for the proxy (lower SE on later calls).

## I05 pretrained LaBraM with the lag, fold 1 (`mi_plab_ho5k_lag5`, 18:35): the best single, neutral in the ensemble
alone 5k r=3 1.42 / 1.56 / 1.56 (ep 4 / 6 / 8; d6 lag 1.34, rmix 1.42, SQF-X 1.23; ensemble 1.672), r=4 1.68 / 1.76 /
1.75. As a sixth member: 5k r=3 +1.9 / +1.0 / +2.9 % (SE 2.5), r=4 -1.0 / +1.0 / +0.4 %; 14k r=3 -4.0 / -2.9 / -2.9 %,
r=4 +2.5 / +1.5 / -0.5 %. Swapped for rmix at ep 8: 5k +2.2 / +2.7 %, 14k +0.6 / +3.5 %; for d6: 5k -1.2 / +1.9 %, 14k
-2.3 / +5.0 %; for SQF-X, scratch or SQF: -7.5 .. +1 %. One fold: neutral. Fold 3 running (local).

## I06 why TSConv is weak here: neither its 64-d bottleneck nor a stronger time-locking (user question, 18:40)
- Output rank (`ni03_output_rank.py`, fold 1): TSConv 64 (the bottleneck), d6 508, rmix 257, SQF-X 276, LaBraM 199;
  participation ratios 9.5-13.8 for all. Cutting the others to their top 64 directions costs 3-9 % only (d6 1.340 ->
  1.288, rmix 1.424 -> 1.372, SQF-X 1.228 -> 1.184, LaBraM 1.560 -> 1.540 at 5k r=3): rank 64 does not explain 0.98.
- Time shifts (`ng01_shift_robust.py 1 36 mi_tsc_ho5k_lag5`, r=1 / r=3): -4 .24 / .36, -2 .42 / .77, 0 .52 / .98, +2
  .49 / .92, +4 .37 / .66, i.e. -21 / -6 % at +-2 samples and -63 / -33 % at +-4: the same profile as d6 (-15 / -13,
  -49 / -37 %) and SQF-X (-20 / -17, -57 / -35 %). Every member is time-locked to the same degree.
- What is measured: TSConv is the weakest single-trial decoder (r=1 0.52 vs 0.64 d6); SQF-X = the same stem + three
  attention / conv blocks is +25 %. Whether the lag itself costs TSConv was never measured: `mi_tsc_ho5k_nolag` fold 1
  started 18:45 (mi_gpu) to answer it.

## I07 the LaBraM "fine-tuning" is mostly a retrain from a pretrained start (`ni04_finetune_distance.py`, user question 18:50)
`mi_plab_ho5k_lag5` = lag data (things_lag 5), every weight trainable, one stage, lr 5e-4 one-cycle, batch 128, 8
epochs (~72k steps), EMA, new head; 220 / 221 pretrained tensors loaded (temporal embedding random: 120-sample window).
Fold-1 final weights against the public checkpoint, all weight matrices of the 12 blocks: cosine 0.275 (a random
initialisation: 0.001), relative change |w - w0| / |w0| = 1.68, norm x1.65. Block 5: qkv .29, proj .33, mlp .30 / .22;
only scale parameters stay (gamma, norm weights .92-.99). Early screen (30 Sep): pretrained 32.0 vs scratch 30.25.
So the pretraining is an initialisation that leaves a trace, not a preserved model. Test of a real fine-tune started
18:55 (local): `mi_plab_ho5k_lag5_lr1` fold 1 = same with lr 1e-4; read: alone, in the ensemble, and its distance.

## I08 recurrent member, folds 1 + 3 (ni01_new_member.py mi_rec_ho5k_lag5 1,3 24,32,36,40)
- Alone (5k r=3): 0.970 / 0.970 / 1.002 / 0.968 at epochs 24 / 32 / 36 / 40, the level of TSConv (0.98); 14k r=3 0.39-0.41.
- As a sixth member: 5k r=3 -0.7 / -1.8 / -2.1 / -1.3 % (+- 1.8), 14k r=3 +1.7 / -1.1 / -0.3 / +0.0 % (+- 2.8); r=4 the same.
- Swapped for any of the five: negative everywhere (SQF-X -0.2..-1.7 %, SQF -1.5..-2 %, d6 -3..-5 %, rmix -4.5..-6 %).
- Verdict: a second time-locked low-capacity family (GRU over the TSConv stem) behaves like TSConv: half the ensemble's
  accuracy alone and nothing to add. With I04 (TSConv) and I05 (LaBraM, fold 1) no new family has moved the ensemble. Closed.

## I09 JEPA-style cross-person latent prediction (user 16:45), rmix, fold 1 (launched 20:00)
- Rationale: M01/M02 place the trained-person advantage in blocks 4-5; nothing but the image loss at the output shapes the
  middle of the network. Auxiliary loss `--xperson_lambda 1` (train.py) with TokenREVE `jepa=3`: the token mean after block 3
  -> 2-layer predictor -> 1 - cosine to the stop-gradient FINAL token mean of ANOTHER person's trial of the same image.
  At the readout itself the loss would be redundant (the head is an injective linear map onto the image target).
- Pairs need `--grouped_batch_sampler --samples_per_image 2` (64 images x 2 trials per batch; holdout path now keeps the
  image groups) and `--multi_positive_loss` (the batch holds every image twice). Epoch length unchanged (9015 steps).
- Runs: `mi_rmix_ho5k_lag5_grp` (control: sampler + multi-positive only) and `mi_rmix_ho5k_lag5_jepa`, both lag 5, 12 epochs,
  checkpoints 8 / 10 / 12; read as "swapped for rmix" in the lag ensemble (ni01), against each other and against rmix lag.

## M06 trial quality read from the output vector (`nm06_conf_probe.py`, 20:30; follows M03)
MLP probe (1536 -> 512 -> 1) on the direction of the ensemble output (or of the five member outputs) + log norms, fitted
on the new people of two folds x half the never-trained images, applied to the third fold's people x the other half.
Repeat weight exp(b z(prediction)), r=3, 5k gallery, pooled, paired against the plain scaled sum (1.397).
- Target q = cosine(output, target): the probe reads it far better than the norm, r .66-.68 vs .39-.44, and also within
  a (person, image) group, r .48 vs .24 (folds 1, 2). As a repeat weight: -0.4 / +0.2 / -1.7 % (b .1 / .2 / .4); fitted
  on the within-group part only: +1.3 +- 2.0 (ensemble vector), +1.9 +- 2.0 (member vectors) at b .2. Replacing the norm
  by it: -1..-5 %.
- A simulated predictor with the same within-group correlation and independent errors would be worth +15.5 % (r .5),
  +5.3 % (r .25), +32 % (r 1). So the part of q the probe reads is a part that cannot move ranks: how image-like the
  output is in general (it raises the cosine to every gallery image alike).
- Target = the z-score of the target's cosine among 9.4k other images (what is specific to the target): the probe reads
  no more than the norm does (within-group r .17-.24 vs .16-.23), weights -1..-3 %. Its oracle: +41 %; r .25 -> +8 %.
- Verdict: the M03 "oracle headroom" is real only for information about the target itself. The output vector holds no
  target-specific quality signal beyond its norm (which the grader already uses). Input statistics (M03) do not either.
  Closed; a probe on mid-layer activations is the only untested reader and M03's input features argue against it.

## Ops 20:25-20:35 node move
Interactive job 72966 (513) cancelled 20:23; the session now runs inside sbatch job 73733 (518). The two LaBraM runs
started on 513 and the srun clients of the three mi_gpu steps kept running on 513 after the cancellation (logs still
advancing at 20:30; 513 is not reachable from here). Watchers relaunched on 518 (file-based: `wait_run.sh`). Keep-alive
`sleep 2d` appended to job_d6.sh (73733) and job_rmix.sh (73734): both jobs stay as GPUs after their fold-2 runs;
scancel both at the end.

## I10 pretrained LaBraM, lag 5, folds 1 + 3 pooled (ni01, logs/mi/ni01_plab_f13.out; fold deltas f1 / f3)
| epoch | alone 5k r3 | 6th 5k r3 | 6th 5k r4 | 6th 14k r3 | 6th 14k r4 | swap rmix 5k r3 | swap rmix 14k r4 |
|---|---|---|---|---|---|---|---|
| 4 | 1.422 | +0.5 +- 1.6 | +0.2 | -0.3 +- 2.5 | +1.2 | -0.9 | -0.7 |
| 6 | 1.560 | +2.1 +- 1.7 | +1.5 | -0.3 +- 2.4 | +0.0 | +1.3 | +3.5 |
| 8 | 1.554 | +3.0 +- 1.7 (+3 / +3) | +2.3 +- 1.5 (+0 / +4) | +0.3 +- 2.4 (-3 / +3) | +1.2 +- 2.3 | +2.3 +- 1.9 (+2 / +2) | +4.2 +- 2.8 (+3 / +5) |
- Base 1.646 / 1.946 / 0.708 / 0.810. Swaps for d6, SQF-X, SQF: within +-1 % (SQF-X 14k -3 %).
- Reading: the first new family with a positive sign on both folds, about +2-3 % at 5k, flat at 14k r=3; below 2 sigma
  everywhere. The contribution grows with the epoch (4 < 6 < 8), so the 8-epoch budget may be short:
  `mi_plab_ho5k_lag5_long` (stop 12, checkpoints 8 / 10) launched 20:50, fold 1 on 518, fold 3 in job 73734 (515).

## I11 does the lag cost TSConv? (user question 17:30; `mi_tsc_ho5k_nolag` vs `mi_tsc_ho5k_lag5`, fold 1, alone, 5k r=3)
| epoch | 24 | 29 | 32 | 36 | 40 |
|---|---|---|---|---|---|
| no lag | 0.948 | 0.920 | 0.936 | 0.924 | 0.924 |
| lag 5 | 0.948 | 0.952 | 0.972 | 0.980 | 1.008 |
As a sixth member both are +0..2 % (+- 2.5). The lag does not hurt TSConv: it is equal or slightly better with it (+9 %
at the last epoch, about 1.5 sigma of the single-fold noise). TSConv is simply the weakest single-trial decoder (I06).

## Ops 21:35 a stale copy of this session kept acting on 513 from 20:23 to 21:35 (written by that copy)
The Claude process of the cancelled interactive job (513) survived the cancellation and kept working on the same
conversation, in parallel with the live session on 518 and without seeing the user's 20:28 message. It found out at
21:30 from the run collision below. What it did after 20:23:
- 20:45 launched `mi_plab_ho5k_lag5_long` fold 1 on the 513 GPU, same name as the live session's run (launched 20:47 on
  518). The two shared one output folder and one log file; the 513 copy died at 21:27 (its run folder was replaced).
  The live run on 518 (`20261111-204849-gen`) is the valid one; `logs/mi/train_mi_plab_ho5k_lag5_long_f1.log` holds a
  traceback from the dead copy near line 200, to be ignored.
- The 513 GPU was given to another user's job (73780_1) at 20:23:54. The low-learning-rate LaBraM run
  (`mi_plab_ho5k_lag5_lr1` fold 1, started 18:48) and the duplicate above therefore ran on someone else's GPU. Killed
  21:33 at epoch 7.8 of 8: `checkpoint_ep04.pth` and `checkpoint_ep06.pth` exist, no final checkpoint, no sealed score
  line in its log (`wait_run.sh mi_plab_ho5k_lag5_lr1_f1` will report STALLED). Score epochs 4 / 6 or rerun it on a held GPU.
- Wrote duplicate I10 / I11 entries here (removed) and scored TSConv no-lag / LaBraM on the 513 GPU (same cache files).
- Its watchers are stopped. Left alive on 513 on purpose: the two `srun` clients of the mi_gpu steps 72520.90 / .91
  (rmix JEPA-style and its control); they use no GPU there and killing them would kill both runs.
- A second old `claude` process (pid 3439493, started 12:45) is still on 513; it is probably the stale copy of the CVPR
  session and was not touched.
The stale copy (pid 3439147 on 513) then killed itself. Nothing else was launched, cancelled or deleted by it.

## I12 lag on fold 2, first two members (the catcher jobs 73733 / 73734; d6 lag @12, rmix lag @10 swapped into the cached non-lag ens5 of fold 2)
Fold 2 is the hard fold: base 1.100 (5k r=3) / 1.380 (r=4) / 0.464 (14k r=3) / 0.580 (14k r=4).
| | 5k r3 | 5k r4 | 14k r3 | 14k r4 |
|---|---|---|---|---|
| d6 alone, no lag -> lag | 0.972 -> 1.000 | 1.196 -> 1.208 | 0.408 -> 0.440 | 0.500 -> 0.540 |
| rmix alone, no lag -> lag | 0.972 -> 0.932 | 1.100 -> 1.184 | 0.424 -> 0.408 | 0.512 -> 0.496 |
| d6 swapped | +0.7 +- 3.7 | -2.0 +- 3.3 | +7.8 +- 6.0 | +2.8 +- 5.5 |
| rmix swapped | -4.0 +- 3.2 | -6.4 +- 2.9 | +0.0 +- 4.7 | -6.2 +- 4.3 |
| both swapped | +2.2 +- 4.3 | -2.6 +- 3.8 | +5.2 +- 6.5 | +1.4 +- 6.0 |
- Reading: on fold 2 the two REVE-family lag members do NOT reproduce the gain of folds 1 / 3 (d6 about +2, rmix about -4;
  both together about +1.5 +- 5). The fold is noisy (2.5 x fewer hits) and the comparison mixes budgets (cached non-lag
  members at the sub-6 budgets, lag at 12 / 10), but it is the first fold where the lag is not clearly positive.
- The three other lag members on fold 2 were launched 21:45 (SQF-X in mi_gpu, SQF in 73734, REVE-scratch on 518) to
  score the all-lag five against the non-lag five on a third fold.

## I13 LaBraM with a real fine-tune (lr 1e-4), fold 1 (run died at epoch 7.8 with the 513 clean-up; checkpoints 4 and 6)
Alone 1.352 / 1.412 at epochs 4 / 6 (lr 5e-4: 1.42 / 1.56 pooled). As a sixth member, fold 1: +0.7 / +3.3 +- 2.6 % (5k r=3),
-4.6 / -1.1 +- 3.9 % (14k r=3); lr 5e-4 on fold 1: +2 / +1 / +3 at epochs 4 / 6 / 8. The gentler fine-tune is a weaker
member with the same contribution: staying close to the pretrained weights does not buy diversity. Closed.

## I09 result (22:45): JEPA-style cross-person latent prediction, rmix, fold 1
Alone, 5k r=3, epochs 8 / 10 / 12:
| | 8 | 10 | 12 |
|---|---|---|---|
| rmix lag, standard sampler | 1.428 | 1.424 | 1.336 |
| control: 2 trials per image + multi-positive | 1.304 | 1.284 | 1.144 |
| control + cross-person prediction (lambda 1, block 3) | 1.276 | 1.376 | 1.256 |
Swapped for rmix in the lag ensemble (base 1.672 / 0.696, fold 1): control -1.0 / -1.7 / -2.9 % (5k), -5.7 / -3.4 / -5.7 %
(14k); JEPA -0.2 / -0.7 / -0.5 % (5k), -5.7 / -9.2 / -4.6 % (14k); errors +- 2.7 (5k), +- 4.3 (14k).
- The pairing itself costs 9-14 % alone (64 distinct images per batch instead of 128: half the negatives). The
  cross-person loss wins some of it back at epochs 10-12 (+7, +10 % over its control, about 2 sigma) but never passes the
  standard recipe, alone or in the ensemble. No lever. Closed (a version without the sampler cost would need 256-trial
  batches: twice the memory and time for an effect of this size).

## I14 LaBraM as a deployable sixth member: packaging check and final (22:45)
- `package_submission.py` could not trace LaBraM (braindecode `chs_info` property raises when unset): one line added
  beside the `_sfreq` one (`_chs_info` placeholder). Exported fold-1 member (25 MB) reproduces the cached proxy outputs
  exactly (cosine 1.000, norm ratio 1.000) at 17 ms per 64-trial batch on a 3080: no inference-time issue.
- Final started in mi_gpu (516): `results/alljoined/track1/final/labram_optx_things_lag5` = the proxy recipe (lag 5, no
  jitter) on 30 people + 10 test reps, stop 12, checkpoints 8 / 10 (15.9k steps per epoch: done around 08:00-09:30).
  Budget (8 / 10 / 12) to be picked from `mi_plab_ho5k_lag5_long` (folds 1, 3 and now 2, launched 22:45 on 516).
  Speculative: only used if the 12-epoch runs and fold 2 keep the sign.

# Round 7 (11 Nov, 23:00) research period: new directions (user: "think outside the box ... maybe more mechinterp")
Constraints re-read first: no state across predict() calls (organisers, Discord), batch statistics allowed, ~5-6 trials
per person per batch. The steepest lever ever measured here: distinct training images (7.5k -> 11.5k: +14..28 %), and all
30 training people saw the same 16.5k images.

## O01 linear encoding model image -> EEG (`no01_encoding.py 1`)
Ridge from the DINOv2-g embedding to each person's repeat-averaged response (channels whitened), fitted on the 11,540
trained images, read on the 5,000 never-trained ones. All 3,840 cells, no selection.
| | split-half r (2 vs 2 repeats) | r(prediction, 4-repeat mean) | matched-filter top-5, own model, r=3 |
|---|---|---|---|
| Alljoined people (8) | 0.000-0.013 | 0.002-0.016 | 0.04-0.20 % |
| THINGS-EEG2 people (4) | 0.006-0.008 | 0.024-0.036 | 0.24-0.40 % |
- New people with the mean model of the Alljoined training people: r 0.001-0.010, top-5 0.10-0.22 % (chance 0.10).
- For scale: the deep members reach about 2.2 % on the same cell for trained people. A linear embedding -> EEG model
  with a Gaussian noise model holds about a tenth of what the decoders extract; per cell it reaches 6-20 % (Alljoined)
  and 20-30 % (THINGS-EEG2) of the attainable correlation. THINGS-EEG2 responses are several times more predictable from
  the image than Alljoined ones at equal repeat reliability.
- Consequence for the "re-target real trials to unrecorded images" idea: the linear form injects little image signal.
  Not a verdict on the idea (a decoder trained on such trials keeps the real noise), but the cheap version is weak.

## O02 the RSVP neighbours inside the epoch (`no02_neighbours.py 1`)
Alljoined: 20 images in a row, SOA ~0.215 s, then 1.4 s; repeats of an image are in 2 sessions with different neighbours
(checked in the BIDS events). The epoch of image j holds the responses to j-1 and j+1..j+3. Lag ensemble, fold 1, new
people, single trials (n 100,000; target cosine 0.0211, random image 0.0093, target top-5 r=1 0.799 %).
- Leak toward the neighbours, as a share of the target excess: j-2 -1.2 %, j-1 -10.4 %, j+1 -9.5 %, j+2 -1.3 %, j+3 0.0 %.
  The adjacent images are retrieved BELOW chance from this epoch (0.049 % and 0.073 % against 0.10 %).
  The members push the output away from the two adjacent images: their responses are read with the opposite sign.
- Natural knock-out by position in the sequence (r=1 top-5): first image (no predecessor) 0.888 +- 0.133, middle
  0.809 +- 0.034, last (no successor) 0.776 +- 0.124, second-from-last 0.705 +- 0.117. No position effect.
- Verdict: neighbour responses are a structured but small interference (10 % of the signal each, far below the EEG
  noise) and their absence buys nothing. No "explaining away" lever. Closed.

## O01b ridge strength of the encoding model
r(prediction, 4-repeat mean) on never-trained images is flat from ridge 30 to 3,000 (person 1: .0096 -> .0099; person 23:
.0369 -> .0336) although the median eigenvalue of F^T F is 0.4: only the few largest embedding axes carry the prediction.
In the most predictable tenth of the cells: THINGS-EEG2 people r .16-.19, Alljoined people r .03-.07. A linear image ->
EEG model is a handful of image dimensions, which the decoders certainly hold already: the cheap "re-target trials to
unrecorded images" run was NOT launched (it could only teach what is already known). See O03 for how many dimensions exist.

## M07 does the member route by WHO the trial is from? (`nm07_identity.py mi_rmix_ho5k_lag5 1 checkpoint_ep10.pth`)
rmix lag, fold 1; identity subspace = span of the 15 Alljoined training people's centroids of the token mean (14 of 256
dimensions) after each block; never-trained images (2,500), 5k gallery, r=3 (intact: new 1.456, seen 2.288).
- Identity is there: 21-24 % of the token-mean variance, 48 -> 65 % nearest-centroid identification of the 15 people from
  one trial (blocks 0 -> 3), 50 % after the last block.
- New people are not outsiders in it: their trials lie as close to the nearest known person as known people's own trials
  do (1.3-1.7 against 1.4-1.7), spread over all known people (most-chosen one takes 12-19 %).
- Distance to the nearest known person rises with decoding quality in the late blocks (r +.29 new, +.46 seen at block 5):
  it measures response strength, not unfamiliarity.
- Patches (every token shifted so the mean's identity coordinates move to a chosen point), 36 conditions: removing the
  identity, halving it, snapping to the nearest known person or imposing a fixed person changes NEITHER group beyond
  noise (new -6.6..+2.2 %, seen -8..+3.5 %, errors 2-6 %; worst case person 14 at the last block -13.7 +- 6.7).
- Verdict: the member represents who the trial is from but does not use that variable: erasing or swapping it leaves the
  trained-person advantage intact. The private code is therefore not "identify the person, then apply their map"; it
  sits in the image features themselves (how each person's response is laid out), which is why no id-level steering can
  hand it to a new person. (Limit: only the token-mean component was patched.) No lever. Closed.

## O03 image dimensions: in the population EEG (linear, no model) against what the ensemble decodes (`no03_shared_dims.py 1`)
(a) EEG side: 15 Alljoined training people in two groups (8 / 7), group-mean response per image (about 30 trials each),
    SVD of the cross-covariance fitted on the trained images, read on the 5,000 never-trained images: the paired
    directions correlate across groups at r <= 0.05; 5 of 300 pass 3 sigma, none passes 0.1. On those 5, a ridge from the
    embedding reaches r .15 against a ceiling of .30. A linear read of raw cells barely sees image content even in a
    30-trial average.
(b) Decoder side: the lag ensemble's outputs for the 5 NEW people, 20 trials per never-trained image, on the principal
    axes of the gallery: r(decoded, true) .58 .44 .57 .37 .38 on the first five; 60 axes above .1, 17 above .2, 3 above .4;
    the axes above .1 hold 55 % of the gallery variance.
- Reading: the members extract an order of magnitude more image dimensions than any linear model of the raw signal
  (about 60 against about 5). Whatever they compute to get there is non-linear in the input. Two consequences:
  (i) manufacturing trials for unrecorded images through an image -> raw EEG model is hopeless at this noise level (the
  generator would have to be as good as the decoders are); idea closed in raw space. (ii) the question worth asking is
  WHICH non-linear operation buys the factor 10 (per-trial normalisation? latency tolerance?): O04.

## O04 per-trial adaptive spatial noise filter (`no04_mvdr_probe.py 1`), linear probe then two member runs
Idea: a trial is almost all noise, so its own 32 x 32 channel covariance C is its noise covariance; the optimal read-out
of a fixed topography under that noise is a fixed decoder applied to C^-1 x. A network cannot invert a covariance, so the
filter is an input transform. Linear probe: ridge (3,840 cells -> 64 embedding axes), 8 Alljoined training people x 3,000
trained images, read on the NEW people x 2,500 never-trained images, mean single-trial cosine to the target (+- .0009):
| input | target cosine | vs raw |
|---|---|---|
| raw x | .0537 | |
| per-trial per-channel z-score (rmix's first operation) | .0623 | +16 % |
| C_pop C_s^-1 x, C_s = .4 C + .6 C_pop (population covariance) | .0661 | +23 % |
| D C_s^-1 x, C_s = .4 C + .6 D (the trial's own diagonal; no stored constant) | .0611 | +14 % |
| the same, then per-channel z-score | .0697 | +30 % (+12 % over the z-score alone) |
| static C_pop^-1 x (control) / C_s^-1 x without the C_pop factor | .040 / .043-.050 | the plain inverse hurts a ridge |
- The filter helps a linear decoder beyond what per-channel normalisation gives (+12 %, about 5 sigma). Top-5 counts
  agree in sign but are ~100 hits (r=1 .212 -> .278).
- Implemented as `--adaptive_filter a` in `run_eeg_backbone` (train, test and packaging path; float32; identity at a = 1,
  self-check passed). Member tests launched 23:40, fold 1, lag 5, a = .6: `mi_d6_ho5k_lag5_af6` (518) and
  `mi_rmix_ho5k_lag5_af6` (job 73734), against the same recipes without it, alone and swapped into the lag ensemble.
- Extensions in the same probe (all followed by the per-channel z-score; spatial .6 + z = .0697):
  temporal version (each channel's spectrum divided by its own smoothed power, shrunk to the population spectrum) alone
  .0638-.0646, on top of the spatial filter .0707 (+1.4 %); clipping at 3 robust sigmas .0621 (nothing), with the filter
  .0694-.0697 (nothing); covariance from the trial + its 5 nearest trials of the 64-batch .0709 (+1.7 %); a = .8 / .4:
  .0682 / .0686. The gain is the trial's OWN spatial covariance at a = .6; the rest is within a sigma or two. Kept simple.
- Queued behind the two running jobs on 515: TSConv and SQF-X with the filter (fold 1). They have no per-trial
  normalisation at all, so the raw -> filter step (+14..30 % for the linear decoder) applies to them in full.
- Fold 3 replication of the probe (other new people, other training people): raw .0557, per-channel z-score .0649 (+17 %),
  filter (a .6, after 3-sigma clipping, which adds nothing) + z-score .0697 (+25 % over raw, +7 % over the z-score alone;
  fold 1: +30 % / +12 %). Same sign and order on both folds.

## O05 do people differ in response latency? (inline analyses, 00:00)
- Image-independent mean ERP of each person against the mean of the others, best cross-correlation lag: THINGS-EEG2 cohort
  against Alljoined -5.3 samples (the known lag 5: the method works at cohort level). Within Alljoined the lags spread
  with std 3.7 samples (31 ms): persons 4, 9, 17 at the search bound (+-8), 7 +4.8, 16 -4.7, 20 +4.1, 19 +2.8, 6 +2.2.
  The outliers are the known weak people and fold 2 holds four of them.
- Test with frozen members: decoding of each held-out person as a function of an input shift (fold 1: three lag members,
  cached shifts 0, +-2, +-4; folds 2 and 3: rmix lag @10, shifts -6..+6). Every person peaks at 0 (or at +2 within noise:
  14: 2.22 vs 2.10, 20: 1.30 vs 1.16); person 4 (estimated -8): 0.64 / 0.94 / 1.06 / 0.86 / 0.58 across -4..+4.
- Verdict: the image-specific response is stimulus-locked for everyone; the mean-ERP lags reflect differences in the shape
  of the steady-state response, not a timing offset. No per-person latency alignment lever. Closed.
- By-product: held-out people differ sixfold or more (r=3 top-5: person 2 3.0 %, 3 2.6 %, 5 2.6 %, 14 2.1 % ... 11 0.54 %,
  19 0.56 %, 17 0.24 %). Next: are the weak ones weak even when trained on (signal quality) or only as new people?

## O06 each person as a NEW and as a TRAINED person (rmix lag @10, folds 1-3, 2,500 never-trained images, 5k gallery, r=3)
| person | 2 | 3 | 5 | 14 | 10 | 12 | 8 | 20 | 4 | 16 | 7 | 15 | 11 | 19 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new | 2.68 | 2.68 | 2.28 | 2.08 | 1.88 | 1.68 | 1.20 | 1.04 | 0.96 | 0.80 | 0.76 | 0.76 | 0.44 | 0.40 | 0.12 |
| trained | 5.84 | 4.62 | 4.20 | 5.32 | 2.78 | 2.54 | 1.60 | 2.86 | 1.62 | 0.94 | 1.30 | 1.24 | 0.52 | 0.98 | 0.88 |
Fold-4 people (trained only): 6 4.20, 18 3.60, 13 1.41, 1 1.25, 9 0.68.
- correlation(new, trained) = .93: the weak people are weak even when the member was trained on them. Their weakness is
  signal quality, not unfamiliarity. The trained / new ratio is about 1.2-2.7 for everyone (17 apart).
- The score is carried by the few people with a good signal, and for those the trained-person advantage is a factor 2
  (person 2: 2.68 -> 5.84; 14: 2.08 -> 5.32): absolute gap 2.2 points in the 5 strongest against 0.5 in the 5 weakest.
- With O05: nothing about timing separates weak from strong; with M07: nothing about identity routing either.

## I10b LaBraM 12-epoch runs, folds 1 + 3 (01:50)
As a sixth member (5k r=3 / 5k r=4 / 14k r=3): epoch 8 +3.0 / +2.3 / +0.3, epoch 10 +3.4 / +2.4 / +1.1, epoch 12 +3.2 / +1.0 /
+0.3 (errors 1.8 / 1.6 / 2.6); both folds positive at 5k for every epoch. Alone it degrades after 8 (1.554 -> 1.500 -> 1.394).
Budget 8-10; nothing gained by 12. Fold 2 (`mi_plab_ho5k_lag5_long` f2) running; final checkpoints 8 / 10 will serve.

## O04b adaptive filter in real members, first checkpoints (fold 1, 01:55)
- rmix + filter @8: alone 1.248 against 1.428 without (-13 %; 14k .524 against .620); swapped for rmix in the lag ensemble
  -0.5 +- 2.7 (5k r=3), -3.5 +- 2.4 (5k r=4), -6.9 +- 3.9 (14k r=3).
- TSConv + filter @24: alone .980 against .948 (5k r=3), .372 against .336 (14k): inside the noise; as a sixth member +0.7.
- The linear-probe gain (+12 % / +7 % over the z-score) does not carry to the deep members so far: with a million trials
  they learn their spatial filters, and the transform removes shared-across-channel activity they were using. Later
  checkpoints, REVE-d6 and SQF-X still to come before closing.

## I12b lag on fold 2, all five members (04:20; lag d6 @12, rmix @10, SQF-X @40, REVE-scratch @12, SQF @32 against the cached non-lag five)
| | 5k r3 | 5k r4 | 14k r3 | 14k r4 |
|---|---|---|---|---|
| base, no lag | 1.100 | 1.380 | 0.464 | 0.580 |
| all five lag | 1.108 (+0.7 +- 5.2) | 1.344 (-2.6 +- 4.6) | 0.460 (-0.9 +- 8.0) | 0.632 (+9.0 +- 7.9) |
Single swaps (5k r3 / r4): d6 +0.7 / -2.0, rmix -4.0 / -6.4, SQF-X +1.8 / -2.3, REVE-scratch +1.5 / -8.4, SQF +1.1 / +0.3.
- On fold 2 the lag is neutral (0 +- 5 %), against +8-10 % on folds 1 and 3. Not a contradiction at this noise (difference
  about 1.4 sigma), but the three-fold estimate of the lag drops to about +6-7 % (hit-weighted), never negative.
  Budgets differ slightly between the two sides (non-lag cache at the sub-6 budgets). Fold 2 = the low-signal people (O06).

## I10c LaBraM as a sixth member, three folds pooled (ni01 now uses the all-lag base on folds 1 and 2; base 1.467 / 1.745 / 0.625)
| epoch | 5k r3 (folds 1 / 2 / 3) | 5k r4 | 14k r3 | alone 5k r3 |
|---|---|---|---|---|
| 8 | +3.7 +- 1.5 (+3 / +6 / +3) | +1.9 +- 1.3 | +1.9 +- 2.2 | 1.392 |
| 10 | +3.9 +- 1.5 (+3 / +5 / +4) | +3.2 +- 1.4 (+1 / +6 / +3) | +3.6 +- 2.3 (-2 / +11 / +4) | 1.360 |
Swapped for rmix: +3.6 / +3.2 (5k r3). ADOPT as sixth member at epoch 10: the first new member that is positive on all
three folds (2.6 sigma at 5k r=3). Final `final/labram_optx_things_lag5` (lag, no jitter) reaches epoch 10 around 06:00.
04:32: jitter-consistent variant started: final `final/labram_optx_things_lag5jit` (job 73734, stop 10) and its proxy
check `mi_plab_ho5k_lag5_jit3` on folds 1 (518) and 3 (516).

## O04c adaptive filter in members: closed (04:20)
Alone, fold 1, 5k r=3 (no filter -> filter): rmix @8 / 10 / 12 1.428 / 1.424 / 1.336 -> 1.248 / 1.220 / 1.152 (-13 %);
SQF-X @40 1.228 -> 1.144 (-7 %); TSConv @32 / 36 / 40 .972 / .980 / 1.008 -> 1.040 / .988 / .996 (0).
Swapped in the lag ensemble: rmix -0.5 / -1.4 / +0.5 (5k), -7 / -6 / -1 (14k); SQF-X +1.4 / +0.2 / +0.5 (5k), -5..-7 (14k).
A linear decoder's gain from a per-trial noise filter does not transfer: the trained members already filter better than
the closed form, and lose what the transform removes. (REVE-d6 with the filter: scored by the watcher, for the record.)
Lesson: a linear probe on raw cells is not a valid predictor of what helps the deep members.

## I15 pretrained REVE (fine-tuned, lr 2e-4, lag 5) as a member, fold 1 (05:20; `mi_prev_ho5k_lag5`, `ni05_stack_members.py`)
- Alone 5k r=3: 1.436 / 1.424 / 1.300 / 1.276 / 1.312 at epochs 2-6 (peaks at 2-3, as in October).
- As a sixth member: 5k r=3 +1.0 / +3.6 / +4.3 / +2.9 / +1.2; 14k r=3 -1.7 / -9.8 / -8.0 / -2.3 / -4.0 (+- 2.6 / 3.9).
- On top of LaBraM (seventh member), fold 1: 5k r=3 -2.1 (@3) / +0.9 (@4) +- 2.2; 14k -0.6 / -2.4.
- Reading: at 5k it does what LaBraM does (+3-4 %) and the two do not add; at 14k it is worse than LaBraM. One pretrained
  member is enough and LaBraM is the better one. Fold 3 (running) for the record. CBraMod fold 1 running.
- sub15 = the sub13 five + LaBraM lag @10: packaged by a watcher as soon as the final writes checkpoint_ep10.
- I15, folds 1 + 3 (05:40): pretrained REVE as a sixth member 5k r=3 +1.5 / +4.3 / +4.1 at epochs 2 / 3 / 4 (+- 1.7), 14k r=3
  -3.1 / -2.8 / -0.6; as a SEVENTH member on top of LaBraM @10: 5k r=3 +0.4 (@3) / +1.3 (@4) +- 1.5, 5k r=4 +1.0 / +1.3,
  14k r=3 -0.6 / -2.0. The two pretrained models are substitutes, not complements. Closed: LaBraM stays the sixth member.

## Package: sub15 (05:55)
`neurips_challenge/submissions/sub15_ens6_lag5_labram.zip` (424 MB, sha256 8c1b71d17815e56224c78401aab85be876330648af27e05741711c66f1960740)
= the five lag members of sub13 (d6 @12, rmix @10, SQF-X @39, REVE-scratch @12, SQF @33; `--things_lag 5`, no jitter) +
pretrained LaBraM, lag 5, epoch 10 of `final/labram_optx_things_lag5` (linked as `final/labram_optx_things_lag5_ep10`).
predict() unchanged (scaled sum). Fold-1 in-sample replay 34.75 (batch 64) / 34.67 (batch 37) against 33.69 / 33.65 for
sub13; contract check clean (78 s for 600 windows). Proxy expectation over sub13: +3.9 +- 1.5 % (5k r=3, three folds).
Packager change: one line (`_chs_info` placeholder so that LaBraM traces). The user uploads.
Pending: jitter-consistent twin (sub14 five + LaBraM lag + jitter): final `labram_optx_things_lag5jit` (~11:00) and proxy
check `mi_plab_ho5k_lag5_jit3` folds 1 / 3.

## I16 LaBraM trained with lag + jitter (`mi_plab_ho5k_lag5_jit3`, folds 1 + 3, 08:20), as a sixth member of the lag five
| | 5k r3 @8 / @10 | 5k r4 | 14k r3 | alone 5k r3 |
|---|---|---|---|---|
| lag + jitter | +3.9 / +4.4 (+- 1.7; folds +3 / +6 at 10) | +2.6 / +2.0 | -1.1 / -0.6 | 1.586 / 1.586 |
| lag only (I10c, same folds) | +3.0 / +3.4 | +2.3 / +2.4 | +0.3 / +1.1 | 1.554 / 1.500 |
LaBraM keeps its contribution with jitter (and is a little better alone: 1.586 against 1.500 at epoch 10). The jitter twin
of sub15 is therefore valid: sub16 = the sub14 five + `final/labram_optx_things_lag5jit` @10, packaged by a watcher.

# Round 8 (12 Nov, 08:35-): literature review + new analyses (user: "detailed bibliography review of cross-subject eeg/fmri decoding, and more mechinterp experiments, and write a report")
Report (Claude Doc): https://claude.ai/artifact/Eg1At6sPVYAxZkcYBdVZ37 (literature tables with links, P01-P05, ranked next steps).

## I17 CBraMod pretrained (lag 5, LaBraM recipe), fold 1 (`ni01_new_member.py mi_cbm_ho5k_lag5 1 4,6,8,10`): closed
Alone 5k r3 1.02 / 1.11 / 1.21 / 1.15 at epochs 4 / 6 / 8 / 10 (five: 1.67); as a sixth member 5k r3 -4.1 / -3.1 / -3.3 / -1.2 +- 2.6 %,
r4 -6.6 / -4.3 / -4.1 / -5.4 +- 2.3 %, 14k r3 -4.6 / -5.7 / -11.5 / -8.6 +- 4 %. Negative at every epoch. Matches AdaBrain-Bench's
THINGS-EEG cross-subject ranking (LaBraM 26.1 > scratch transformer 24.9 > CBraMod 23.2 > EEGPT 19.7 > BIOT 4.7).

## I15b pretrained REVE as a sixth member of the lag five, folds 1 + 3 (`ni01_new_member.py mi_prev_ho5k_lag5 1,3 2..6`)
5k r3 +1.5 / +4.3 / +4.1 / +3.5 / +3.6 +- 1.7 % at epochs 2..6 (folds +4/+5 at @3, +4/+4 at @4); r4 -0.8 / +1.7 / +1.3 / +2.4 / +2.2 +- 1.5 %;
14k r3 -3.1 / -2.8 / -0.6 / +0.8 / +1.1 +- 2.6 % (folds opposite: -8 / +7 at @4). Same size as LaBraM at 5k, nothing at 14k; as a
seventh on top of LaBraM (ni05, folds 1 + 3, @4): 5k r3 +1.3 +- 1.6, r4 +1.3 +- 1.3, 14k -2.0 +- 2.3 %. Substitute, as in I15.

## I18 a second LaBraM as a seventh member (cached, folds 1 + 3, `ni05_stack_members.py 1,3 mi_plab_ho5k_lag5_jit3:10 ...`)
six + LaBraM lag+jitter @10 (a second seed of the strongest member): 5k r3 +1.8 +- 1.5 % (folds +0 / +3), r4 +1.9 +- 1.3 %, 14k r3
+0.8 +- 2.1 %. six + LaBraM @8: -0.5 / -0.1 / +0.8 %; + LaBraM @12: -2.5 / -0.9 / +2.0 %. A second seed is worth ~+1-2 %, extra
checkpoints of the same run nothing.

## Package: sub16 (08:50)
`neurips_challenge/submissions/sub16_ens6_lag5jit_labram.zip` (424 MB, sha256 a3d6afa83f70d439b1fe2fcb8d32c04ad33f2ff03f6b4b1dd227954bf821d9f0)
= the five sub14 members (lag + jitter finals) + `final/labram_optx_things_lag5jit` (stop 10). Fold-1 replay 33.33 (batch 64) /
33.30 (batch 37); contract check clean (88 s). Proxy expectation = sub15 at zero offset (I16: jitter LaBraM sixth +3.9 / +4.4 % at
5k r3), plus the jitter insurance of sub14. sub15 = no jitter anywhere, sub16 = jitter everywhere.

## P01 the misses are not near-misses (`np01_resolution.py 1,2,3`; lag five, cached, 5k)
P(target scores above its k-th nearest gallery neighbour), k = 1 / 10 / 100 / random image; image top-5; concept top-5 (chance .37);
wrong winner (top-1 when not the target) among the target's 50 nearest (chance 1.0 %); share of the queries won by 1 % of the gallery:
- r=1: .581 / .600 / .615 / .660; 0.748 %; 1.44 %; 1.5 %; 26.3 %
- r=3: .629 / .660 / .683 / .746; 1.460 %; 2.48 %; 2.2 %; 24.9 %   (median neighbour rank of the wrong winner 2,848 of 4,999)
- r=4: .641 / .673 / .699 / .770; 1.749 %; 2.91 %; 2.6 %; 25.4 %
- all trials of the image over 5 people (~20): .743 / .795 / .841 / .912; 6.17 %; 8.65 %; 6.1 %; 37.2 %
Graded resolution with no floor (the nearest neighbour is separated at .63 for r=3); the wrong winner is unrelated to the target and
is a hub image. Concept-level retrieval is only 1.7x image-level. No mechanism for hard negatives / within-concept contrast /
full softmax. Hubness grows with averaging (25 -> 37 %): it is a systematic property of the outputs, not trial noise.

## P02 hubs are a property of the images; novelty inside THINGS is cheap (`np02_hubs_novelty.py`, r=3, 5k, 3 folds)
- Wins per gallery image (top-1 for another target) correlate .67-.73 across folds (rank), hits per image .79-.82: disjoint
  people, separately trained models. Wins vs typicality -.25, vs embedding norm -.09, vs distance to training images .08, vs hits .25.
- The 50 most-winning images: 24.9 % of the wins and 16.4 % of the hits (target in 1 % of the queries); the 500 most-winning: 72 %
  of the wins, 59.5 % of the hits. Hubs are where the ensemble is right: consistent with E01 (every output re-weighting loses).
- Novelty = cosine to the nearest of the 11,540 trained images (quintile edges .855 / .876 / .892 / .908; same concept for 51 %):
  top-5 far -> near 1.99 / 1.13 / 1.29 / 1.17 / 1.72 (confounded with typicality, r .45); inside typicality thirds, far / mid / near:
  atypical 2.42 / 1.71 / 2.94, middle 1.13 / 1.25 / 1.30, typical 0.66 / 0.83 / 0.91; P(target > random) .728 -> .761 over the range.
  Same picture with the nearest trained image of ANOTHER concept. A far region of image space costs little inside THINGS.

## P03 what the pretrained member adds (`np03_pretrained_adds.py`, r=3, 5k, 3 folds)
- Top-5 alone: d6 1.268, rmix 1.271, SQF-X 1.103, REVE-scratch 1.215, SQF 0.976, LaBraM 1.360; five 1.460, six 1.527 (+4.6 +- 1.5 %).
- Correlation between members of the target's z-score among the gallery (per query): d6-rmix .84, d6-scr .79, SQF-X-SQF .84,
  token-conv .58-.68; LaBraM with d6 .79, rmix .82, scr .77, SQF-X .65, SQF .61 (mean .73; the five among themselves .67-.74).
  LaBraM is a token-family member with ordinary correlation, and the best single.
- Gain of six by thirds of people (weak / mid / strong): +1.4 +- 4.7 / +10.7 +- 3.3 / +2.3 +- 1.8 %; by typicality quintile
  (atypical -> typical) 2.72 -> 2.72, 1.67 -> 1.78, 1.29 -> 1.43, 1.06 -> 1.16, 0.56 -> 0.54. No new axes (r per band of gallery
  principal axes: five .443 / .229 / .109 / .046 / .013, LaBraM .437 / .222 / .106 / .045 / .013).
Pretraining = one more strong member, not a new kind of evidence (hence REVE substitutes, weaker CBraMod hurts).

## P04 last-layer retraining on people the body never saw (`np04_late_retrain.py mi_rmix_ho5k_lag5 1 checkpoint_ep10.pth`): closed
Body frozen, late part retrained (AdamW 1e-4, batch 128, 2,000 steps, the members' loss) on 3 held-out people x 11,540 trained
images; the 2 other held-out people scored on never-trained images (5k, r=3), paired against the intact member. Control: same on
3 trained people (7, 14, 16). Splits: train [2,4,5] / test [11,15] (intact 0.600) and train [5,11,15] / test [2,4] (intact 1.690).
| part | on unseen people, step 2,000 (split 1 / split 2) | control on trained people |
|---|---|---|
| head | -3 +- 10 / -6 +- 6 % | -8 +- 10 / +4 +- 5 % |
| last block + head | -37 +- 10 / -15 +- 6 % | -12 +- 11 / +10 +- 7 % |
| feed-forward of blocks 3-5 + head | -42 +- 10 / -17 +- 6 % | -22 +- 10 / +7 +- 7 % |
| all | -28 +- 11 / (-22 +- 6 at step 300) | -10 +- 12 % (step 1,000) |
Single-trial P(target > random) moves by -.001 .. -.012 only: the late layers learn the three people's private code and lose the
norm calibration. The DFR logic does not apply: nothing in the late layers is mis-fitted for new people (as M02 said).

## Running (09:10)
- P05 people scaling: `mi_rmix_ho5k_lag5_n8` (8 Alljoined people 1 6 8 10 13 16 18 20 + THINGS, 28-epoch cycle, stop 14, job 72520)
  and `mi_rmix_ho5k_lag5_n4` (1 8 13 18 + THINGS, 36-epoch cycle, stop 18, job 73734): same number of steps as the 15-person
  reference `mi_rmix_ho5k_lag5` (stop 10 of 20), fold 1.
- I19 `mi_plab_ho5k_lag5_ld65` fold 1: LaBraM with the authors' layer-wise lr decay 0.65 (new `train.py --layer_decay`), 518.

## Package: sub17 (09:25)
`neurips_challenge/submissions/sub17_ens7_lag5jit_labram2.zip` (415 MB, sha256 e83ee07dfb2fc86ac15c310c343cd50f9fb1a85dfe7f42cbc97c0e8a0f2b5f56)
= sub16 (five lag + jitter finals + LaBraM lag + jitter @10) + the LaBraM lag final @10 of sub15 as a seventh member (no new
training). Fold-1 replay 34.46 (batch 64) / 34.31 (batch 37) against 33.33 / 33.30 for sub16; contract check clean (75 s).
Proxy expectation over sub15 / sub16: +1.8 +- 1.5 % (I18). The user uploads.

## Alljoined-1 as extra data: closed by size (09:20, no run)
The cached export (`/nasbrain/p20fores/alljoined/alljoined1_120Hz`, October A20/A25) holds 960 NSD images x 8 people (49,540
trials, BioSemi, SOA ~0.8 s): 3 % more trials and 960 new images, against the 16,540 we train on. Zero-shot transfer kept ~20 %
of the above-chance signal in October. Not a data-scale lever whatever its alignment; no lag diagnostic run.

## P05 people scaling: the new-person score rises steeply with the number of same-cohort training people (10:05)
rmix lag recipe, fold 1, N Alljoined people + the 10 THINGS-EEG2 people, same number of steps (cycle and stop scaled by 25 / (N + 10)),
member alone on the fold's new people, never-trained images (`ni01_new_member.py <arm> 1 <epochs>`), 5k r3 / 5k r4 / 14k r3:
| Alljoined people | checkpoints | 5k r3 | 5k r4 | 14k r3 |
|---|---|---|---|---|
| 4 (1 8 13 18), `mi_rmix_ho5k_lag5_n4` | 12 / 14 / 16 / 18 of 18 | 0.864 / 0.912 / 0.776 / 0.760 | 1.052 / 1.040 / 0.932 / 0.840 | 0.336 / 0.348 / 0.328 / 0.268 |
| 8 (1 6 8 10 13 16 18 20), `mi_rmix_ho5k_lag5_n8` | 10 / 12 / 14 of 14 | 1.148 / 1.248 / 1.132 | 1.388 / 1.432 / 1.432 | 0.452 / 0.480 / 0.508 |
| 15 (reference `mi_rmix_ho5k_lag5`) | 8 / 10 / 12 | 1.428 / 1.424 / 1.336 | 1.604 / 1.668 / 1.616 | 0.620 / 0.672 / 0.612 |
Best checkpoint each: 0.91 / 1.25 / 1.43 (5k r3), 1.05 / 1.43 / 1.67 (r4), 0.35 / 0.51 / 0.67 (14k). 4 -> 8: +37 / +36 / +46 %;
8 -> 15 (0.91 doublings): +14 / +17 / +32 % = +16 / +18 / +36 % per doubling. SE per value ~4 % (5k), ~6 % (14k); checkpoint
spread ~5 %; one subset per N; small cohorts overfit earlier in steps (4 people: peak at 12-14 of 18).
- One doubling of same-cohort people ~ the whole gap to the top of the board. Finals (20 Alljoined people) vs fold models (15):
  +6 .. +14 % by this curve, invisible on the proxy.
- Not separated: people vs trials (more people = more trials). Control not run (15 people, half the repeats; needs a new flag).
- Running: `mi_rmix_ho5k_noth_n15` (15 Alljoined people, no THINGS-EEG2; 33-epoch cycle, stop 17, saves 13 / 15 / 17; job 72520).

## P06 THINGS-EEG1 is not rescued by a time shift (`np06_eeg1_lag.py <recovered export dir> 45,10,25,33`, 09:50): closed
Four EEG1 people rebuilt from raw with the October export (`git show pre-cleanup-2026-11-02:neurips_challenge/export_things_eeg1_to_alljoined.py`
+ `preprocess_eeg1.py`, copied to the session scratchpad; held-image trials cached as `results/things_eeg/mi/nc/eeg1_sub??_held.npz`),
zero-shot through rmix lag f1 @10, input shifted by s samples (content later by s, edge replicated), HELD images, 5k gallery.
Reference, new Alljoined people: single-trial top-5 0.734 %, P(target > random) .662; 4 people averaged per image 2.08 %, P .822.
EEG1: s=0: top-5 0.02-0.08 % (chance 0.10), P .49-.51, 4 people averaged P .501; best s=+2 (+17 ms): P .50-.54 per person, .539
averaged (top-5 0.18 %); s=+4: .534; negative shifts .47-.49; +6..+18: .51-.52. A faint signal at +17..+33 ms, an eighth of the
Alljoined effect after averaging 4 people. Not a timing problem: 10 Hz RSVP, 50 ms stimuli, one trial per image.

## P05b what the 10 THINGS-EEG2 people are worth, and what operator transfer buys at 8 people (11:00; fold 1, rmix alone, same steps)
| run | checkpoints | 5k r3 | 5k r4 | 14k r3 |
|---|---|---|---|---|
| 15 Alljoined, NO THINGS-EEG2 (`mi_rmix_ho5k_noth_n15`) | 13 / 15 / 17 of 17 | 1.208 / 1.108 / 1.032 | 1.328 / 1.232 / 1.272 | 0.540 / 0.468 / 0.456 |
| 8 Alljoined + THINGS, NO operator transfer (`mi_rmix_ho5k_lag5_n8_noop`) | 10 / 12 / 14 of 14 | 1.064 / 1.024 / 0.940 | 1.264 / 1.180 / 1.100 | 0.420 / 0.452 / 0.428 |
| reference: 8 Alljoined + THINGS, operator transfer | 10 / 12 / 14 | 1.148 / 1.248 / 1.132 | 1.388 / 1.432 / 1.432 | 0.452 / 0.480 / 0.508 |
| reference: 15 Alljoined + THINGS | 8 / 10 / 12 | 1.428 / 1.424 / 1.336 | 1.604 / 1.668 / 1.616 | 0.620 / 0.672 / 0.612 |
- 15 Alljoined without THINGS (best 1.21 / 1.33 / 0.54) = 8 Alljoined + 10 THINGS (1.25 / 1.43 / 0.51): the 10 lag-aligned THINGS-EEG2
  people are worth ~7 Alljoined people (exchange rate ~0.7), i.e. +18 / +26 / +24 % at 15 Alljoined people. The no-THINGS run was
  still falling at its first saved checkpoint (13 of 17), so its peak may be earlier and higher: 0.7 is an upper bound.
  Same paradigm on other hardware counts almost like same-cohort people; THINGS-EEG1 (other paradigm) counts for nothing (P06).
- Operator transfer at 8 people: best-of-saved +17 / +13 / +12 % (mean over checkpoints +17 / +20 / +11 %): as much as 8 -> 15 people
  at 5k. At 15 people, 80 repeats (October) singles gained ~+3 %: the augmentation substitutes for people and saturates.

## I19 LaBraM with the authors' layer-wise lr decay 0.65 (`--layer_decay`, `mi_plab_ho5k_lag5_ld65`, fold 1, 12:10): closed
Alone, 5k r3 / r4 / 14k r3 at epochs 6 / 8 / 10: 1.400 / 1.316 / 1.396, 1.692 / 1.640 / 1.596, 0.576 / 0.580 / 0.572, against 1.560 / 1.508
(r3 @8 / @10), 1.748 / 1.616 (r4), 0.664 / 0.644 (14k) for the flat-rate LaBraM on the same fold: about 10 % weaker alone. As a sixth
member: +0.5 / +2.6 / -1.7 % (5k r3), +1.4 / +2.5 / +3.5 % (r4), -3.4 / -0.6 / -0.6 % (14k), vs +2.9 / +3.1, +0.4 / +1.2, -2.9 / -2.3 for the
flat one: no better. As a seventh on top of LaBraM: -3.2 / +1.0 / 0.0 % (@10). Keeping the early layers near their pretrained values
does not help here (as I13 with a low flat rate): the LaBraM gain needs the full retrain.

## Board (user, 12 Nov 10:56): sub16 is first
sub16 (model name "sub3" on the board) 0.001258 aggregated top-5 (top-1 0.00032; per-epoch 0.000629 / 0.000156; 1,533 s), +19.4 % over
sub 2 (0.001054); proxy expectation was +10-11 %. JLShen 0.001245 (per-epoch top-5 0.000648, above ours; aggregation gain 1.92 vs
our 2.00; aggregated top-1 0.000363; 2,631 s), _ares_ 0.001221 (1,972 s), sazeglio 0.001217 (3,875 s), saltytaro 0.00118 (2,597 s).
The top 4 are inside one board SE (~4 %). Six members = 26 min on the grader: sub17 (seven) ~30 min.

## Q01-Q03 MSS (OpenNeuro ds005589) passes the zero-shot test (11:35)
32 people, 99 sessions, RSVP 100 ms image + 100 ms blank (authors' experiment_RSVP.py), 20 images of ONE class per sequence, 8,000
trials per session, 10,000 images (ImageNet 4,731 from the cluster's per-class tars, VOC2012 3,334, VOC2007 1,935 downloaded), 122
channels at 1,000 Hz (Curry .cdt, read directly). `nq01_mss_images.py` (images + DINOv2-g targets, /nasbrain/p20fores/mss_ds005589/
targets.npz), `nq02_mss_export.py` (average reference, 23 Alljoined electrodes copied + 9 spline-interpolated: CP5 P1 PO7 O1 Pz O2 PO8
P2 CP6, grader chain, -0.2..0.8 s at 120 Hz), `nq03_mss_zeroshot.py` (rmix lag f1 @10, 10k-image gallery, chance top-5 0.05 %).
Single trial top-5 / P(target > any image) / P(target > same-class image), sub-01 ses-01 and sub-02 ses-01, by input shift:
-6: .14 / .568 / .570 and .13 / .591 / .600; -4: .21 / .592 / .596 and .25 / .608 / .617; -3: .21 / .598 / .601 and .18 / .611 / .620;
-2: .23 / .599 / .603 and .18 / .610 / .619; 0: .10 / .580 / .589 and .16 / .587 / .598; +2: .08 / .546 / .554 and .09 / .545 / .555;
+6 and beyond: chance. Peak at -2..-3 samples (content must move EARLIER by ~25 ms). The same-class read equals the any-image read,
so the class-blocked sequences do not inflate it; .60-.62 = the within-concept level of new Alljoined people (P01: .603 at r=1).
Lag adopted: -3. Full cohort streaming (6 workers, raw deleted after export). Training path: `module/dataset.ExtraTrials`,
`train.py --extra_trials_dir /nasbrain/p20fores/mss_ds005589 --extra_shift -3` (ids 100 + XX, objects 100000 + image row).

## P05c operator transfer at 15 people: on half the time is enough (13:10; rmix f1 alone, epochs 8 / 10 / 12)
| operator transfer | 5k r3 | 5k r4 | 14k r3 |
|---|---|---|---|
| off (`mi_rmix_ho5k_lag5_op0`) | 1.336 / 1.336 / 1.300 | 1.636 / 1.628 / 1.584 | 0.612 / 0.556 / 0.608 |
| 0.5 (reference `mi_rmix_ho5k_lag5`) | 1.428 / 1.424 / 1.336 | 1.604 / 1.668 / 1.616 | 0.620 / 0.672 / 0.612 |
| 1.0 (`mi_rmix_ho5k_lag5_op1`) | 1.348 / 1.396 / 1.368 | 1.664 / 1.596 / 1.508 | 0.544 / 0.556 / 0.580 |
At 15 people the augmentation is worth ~+7 % (5k r3), +2 % (r4), +10 % (14k) best-of-saved, inside the noise of one model; always-on is
not better than half the time. With P05b (+12-17 % at 8 people): it substitutes for people and fades as they are added. Closed.

## MSS ingestion (13:10)
93 of 99 sessions exported (31 people, 744,000 trials, 5.4 GB) by 6 parallel workers in ~95 min; raw deleted after export. Six sessions
failed (3 header parse, 2 size mismatch where the forward and reverse workers met, 1 run with 1,340 events for 2,000 images):
`nq02_mss_export.py` now skips a damaged run and keeps the session; the six are re-downloaded one at a time (`meta/ingest_fix.log`).
Runs launched (fold 1, clean proxy, shift -3): `mi_rmix_ho5k_lag5_mssA` (51 sessions, 21 people, 518, 12:17; slow at first: the node's
page cache was taken by the ingestion), `mi_d6_ho5k_lag5_mssA` (56 sessions, 21 people, new mi_gpu job 74191 on 513, 12:20),
`mi_rmix_ho5k_lag5_mssF` (93 sessions, 31 people, 515, 13:11). Job 72520 (516) cancelled at 12:19, one hour before its limit.

## Q04 adding MSS with ONE contrastive loss halves the new-person score; one loss per image set restores it (13:50)
Per-epoch held-out score of the training log (new Alljoined people, 200 test images, sealed rule), fold 1, epochs 1, 2, ...:
| run | MSS | loss | operator transfer | score by epoch |
|---|---|---|---|---|
| rmix reference | - | - | 0.5 | 18.1 21.4 25.6 26.2 27.7 27.9 |
| rmix reference | - | - | off | 18.8 21.7 24.3 |
| REVE-d6 reference | - | - | 0.5 | 21.4 24.3 25.5 27.0 26.6 29.2 |
| `mi_rmix_ho5k_lag5_mssA` (51 sessions) | yes | joint | 0.5 | 9.4 |
| `mi_d6_ho5k_lag5_mssA` (56 sessions) | yes | joint | 0.5 | 10.1 11.5 12.8 13.9 13.9 |
| `mi_rmix_ho5k_lag5_mssDa` (93 sessions) | yes | joint | off | 9.9 11.1 |
| `mi_rmix_ho5k_lag5_mssDb` (93) | yes | joint | 0.5, MSS people skipped | 8.6 |
| `mi_rmix_ho5k_lag5_mssShuf` (99) | yes, targets shuffled | joint | 0.5 | 8.1 |
| `mi_rmix_ho5k_lag5_mssS` (99) | yes | one per image set | 0.5 | 18.5 |
| `mi_d6_ho5k_lag5_mssS` (99) | yes | one per image set | 0.5 | 20.5 |
- Not the augmentation (off or skipping MSS people: same drop), not what the model learns from MSS (shuffled MSS targets: same drop).
  The cause is cross-set negatives: MSS images in the batch are negatives of our trials and are rejected by an output component
  common to all our trials, which costs retrieval among THINGS images.
- Fix (default with `--extra_trials_dir`): the contrastive loss is computed inside each image set and weighted by its share of the
  batch (`train.py`, `--extra_joint_loss` restores the joint loss for controls). Operator transfer skips ids >= 100.
- The October Alljoined-1 result (-3.6 .. -9.8 from epoch 1, with a lower training loss; A25) used a joint loss over NSD and THINGS
  images: very likely the same artefact, not a property of that cohort.
- All 99 sessions are in (32 people, 782,000 trials; 5 damaged runs skipped inside 4 re-downloaded sessions).
Running: `mi_rmix_ho5k_lag5_mssS` f1 (515) and `mi_d6_ho5k_lag5_mssS` f1 (513), 12 epochs, saves 6 / 8 / 10; fold 3 of the rmix run on 518.

## Q05 MSS in training (one loss per image set, shift -3, all 99 sessions = 32 people, 782,000 trials): neutral (16:45)
Clean proxy, new Alljoined people, member alone and swapped for its twin in the lag five (paired, +- ~2.5 % at 5k, ~4 % at 14k).
| member, fold | epochs | alone 5k r3 | alone 5k r4 | alone 14k r3 | swap 5k r3 | swap 5k r4 | swap 14k r3 |
|---|---|---|---|---|---|---|---|
| rmix + MSS, f1 | 6 / 8 / 10 / 12 | 1.376 / 1.516 / 1.512 / 1.516 | 1.544 / 1.700 / 1.696 / 1.692 | 0.556 / 0.616 / 0.628 / 0.596 | 0.0 / -3.3 / -0.5 / +0.2 | -1.2 / +0.6 / -1.4 / -1.2 | -5.2 / -6.9 / -2.9 / -4.6 |
| rmix reference, f1 | 8 / 10 / 12 | 1.428 / 1.424 / 1.336 | 1.604 / 1.668 / 1.616 | 0.620 / 0.672 / 0.612 | | | |
| rmix + MSS, f3 | 6 / 8 / 10 / 12 | 1.324 / 1.412 / 1.416 / 1.492 | 1.584 / 1.664 / 1.680 / 1.832 | 0.496 / 0.516 / 0.556 / 0.592 | +1.2 / 0.0 / +2.0 / +3.7 | -3.3 / -3.5 / -0.6 / +0.2 | -3.9 / -2.2 / -2.2 / +0.6 |
| rmix reference, f3 | 8 / 10 / 12 | 1.412 / 1.456 / 1.360 | 1.704 / 1.688 / 1.640 | 0.608 / 0.640 / 0.612 | | | |
| REVE-d6 + MSS, f1 | 6 / 8 / 10 / 12 | 1.320 / 1.340 / 1.380 / 1.376 | 1.512 / 1.660 / 1.588 / 1.596 | - / 0.616 / 0.548 / 0.604 | +0.2 / +1.0 / +1.4 / -0.7 | -0.8 / +0.4 / -1.0 / -2.3 | 0.0 / -2.9 / -4.6 / -4.0 |
| REVE-d6 reference, f1 | 8 / 10 / 12 | 1.372 / 1.428 / 1.340 | 1.600 / 1.732 / 1.612 | 0.524 / 0.568 / 0.508 | | | |
- Alone: rmix +6 % (f1) and +2.5 % (f3) at 5k r3 best-of-saved, -7 % at 14k on both; REVE-d6 -3 % at 5k, +8 % at 14k. In the ensemble
  the swap is 0 +- 3 % everywhere. With MSS the members no longer decline after epoch 10 (rmix f3 still rising at 12).
- The model does learn MSS (in-sample, rmix + MSS @10 on three MSS sessions: single-trial P(target > any image) .83-.89 and
  P(target > same-class image) .75-.78, against .60-.64 for the model that never saw MSS), so the null is not a loading bug.
- Reading: 32 more people with the same presentation, on OTHER images (20 classes, class-blocked sequences, 9 interpolated
  electrodes), add nothing for new Alljoined people on THINGS images, where 10 THINGS-EEG2 people on the SAME images add 18-26 %.
  What the P05 curve measures is people on the shared image set, not people as such.

### Q05 MSS as extra training people, per-set loss, 12 epochs (12 Nov, 16:45)

Clean proxy, member alone, top-5 % (5k r=3 / 5k r=4 / 14k r=3). Reference = the same recipe without MSS.

| Member, fold | Epoch | with MSS | reference |
|---|---|---|---|
| rmix, f1 | 8 | 1.516 / 1.700 / 0.616 | 1.428 / 1.604 / 0.620 |
| rmix, f1 | 10 | 1.512 / 1.696 / 0.628 | 1.424 / 1.668 / 0.672 |
| rmix, f1 | 12 | 1.516 / 1.692 / 0.596 | 1.336 / 1.616 / 0.612 |
| rmix, f3 | 8 | 1.412 / 1.664 / 0.516 | 1.412 / 1.704 / 0.608 |
| rmix, f3 | 10 | 1.416 / 1.680 / 0.556 | 1.456 / 1.688 / 0.640 |
| rmix, f3 | 12 | 1.492 / 1.832 / 0.592 | 1.360 / 1.640 / 0.612 |
| REVE-d6, f1 | 8 | 1.340 / 1.660 / 0.616 | 1.372 / 1.600 / 0.524 |
| REVE-d6, f1 | 10 | 1.380 / 1.588 / 0.548 | 1.428 / 1.732 / 0.568 |
| REVE-d6, f1 | 12 | 1.376 / 1.596 / 0.604 | 1.340 / 1.612 / 0.508 |

Swapped for its reference (epoch 10) in the lag five, paired, % (5k r=3 / 5k r=4 / 14k r=3):
rmix f1 @12 +0.2 ± 2.5 / -1.2 ± 2.3 / -4.6 ± 4.1; rmix f3 @12 +3.7 ± 2.6 / +0.2 ± 2.2 / +0.6 ± 3.8; d6 f1 @10 +1.4 ± 2.9 / -1.0 ± 2.5 / -4.6 ± 4.4.

- 32 more people (782,000 trials, about 12 Alljoined people in trial count) move a member by 0 to +6 % at 5k against its best
  reference epoch and by 0 to -10 % at 14k; no ensemble swap leaves the noise. The people-scaling slope measured inside
  Alljoined (P05: +16..36 % per doubling) does not extend to this cohort.
- The rmix member with MSS no longer declines after epoch 10 (the reference does), so MSS acts as a regulariser at most.
- Confound still open: with one loss per image set, our trials are about 60 % of a 128 batch, so each has about 77 in-batch
  negatives instead of 128. Running: `mi_rmix_ho5k_lag5_mssB` (batch 208, folds 1 and 3; jobs 74191 on 513, 73733 on 518).
- SQF-X with MSS (fold 1, interactive GPU on 516) ends about 18:45. Its per-dataset BN puts MSS in the THINGS-EEG2 domain
  (`--tsconv_domain_bn_boundary 21`), a handicap to keep in mind when reading it.
- Job 73734 (515) is idle and kept unused (user: leave one Slurm GPU free).
- Where the change falls (`nq05_mss_where.py`, 5k r=3, held-out images binned by cosine to their nearest MSS image, far -> near,
  with MSS vs reference): rmix f1 2.40/1.10/1.44/1.28/1.36 vs 2.32/0.94/1.24/1.28/1.34; rmix f3 2.26/1.24/1.32/1.36/1.28 vs
  2.06/1.22/1.42/1.58/1.00; d6 f1 2.14/1.26/1.32/1.02/1.14 vs 2.26/1.24/0.98/1.36/1.30. No gradient with closeness to MSS
  content and no consistent person: the small differences are noise, not a content effect.
- Several MSS members at once (user's question; fold 1, rmix and d6 both swapped in the lag five, 5k r=3 / 5k r=4 / 14k r=3):
  rmix@12 + d6@12 +2.4 ± 3.3 / -2.7 ± 2.8 / -5.2 ± 5.3 %; rmix@12 + d6@10 -4.3 ± 3.2 / -0.4 ± 2.8 / -5.7 ± 5.1 %;
  rmix@10 + d6@10 -0.7 ± 3.3 / -1.6 ± 2.9 / -2.9 ± 5.3 %. No accumulation.
- 17:57 user: release 515 -> job 73734 cancelled (idle), keep-alive lines removed from `job_rmix.sh`.

### Q06 ds007964 (12 Nov, 18:00): 19 people on THINGS images (user: proceed even if MSS is negative; clean the data once closed)

- OpenNeuro ds007964 (CC0; Kim, Quek, Moerel et al., Sydney): 127 electrodes + reference FCz, 1000 Hz, BrainVision, one file
  per person (20 GB in all). RSVP 100 ms image + 200 ms blank, 5,700 presentations = 1,900 THINGS images (190 concepts x 10) x 3,
  85 target trials dropped. All 32 Alljoined electrodes recorded (FCz rebuilt by the average reference): no interpolation.
- Images: 1,540 are THINGS-EEG2 training images (501 of them in the proxy's held-out 5,000), 24 test images, 336 are among
  the 14k-gallery distractors. Targets taken from our existing files. Under `--holdout_images` only the 1,039 images the
  proxy trains on are kept (3,068 of 5,615 trials for sub-01); final models can use everything.
- `nq06_ds007964_export.py <sub>` (same chain as nq02) -> `/nasbrain/p20fores/ds007964/export/sub-XX.npz`, `targets.npz` with
  `obj`, `image` (training ids, so the positive mask pairs these trials with ours of the same image) and `subject_offset` 200.
  `ExtraTrials` reads those optional keys; with them the loss is the ordinary joint loss (same image set). `--extra_repeat N`
  repeats the cohort N times per epoch.
- Zero-shot through rmix lag f1 @10 (`nq07_ds007964_zeroshot.py`, gallery = its 1,900 images, chance top-5 0.26 %): single
  trial P(target > random) .59-.66, top-5 0.7-1.5 %; three trials averaged up to 2.5 % / .72. Best shift per person +3..+6
  samples (sub-03 +5/6, 05 +5, 06 +3, 07 +2/3, 08 +4): `--extra_shift 4`.

### Q08 other cohorts as donors of recording operators (12 Nov, 18:10)

- `nq08_op_donors.py build` keeps R^(1/2) (32 x 32 channel covariance root, shrink .05) of each foreign person;
  `--op_donors <npz>`: half of the operator-transfer partners are donors (`module/augment.py`). No image label needed.
- `nq08_op_donors.py check`, frozen rmix lag f1 @10, held-out people, 5k r=3: intact 1.304; through another held-out Alljoined
  person 1.388 (+6.4 ± 4.2 %); through a ds007964 donor (2 so far) 1.224 (-6.1 ± 4.0 %); through an MSS donor 1.012
  (-22.4 ± 4.4 %). The foreign operators are outside what training covers (MSS partly through its 9 interpolated electrodes).
- Prior: October HISTORY (old metric): random-orientation control = real operators ("generic spatial-mixing augmentation").
- Q05 batch-size confound, `mi_rmix_ho5k_lag5_mssB` f1 (batch 208, our trials back to about 128 per batch): alone @10 / @12
  1.524 / 1.512 (5k r=3), 1.584 / 1.576 (5k r=4), 0.604 / 0.628 (14k) = the batch-128 MSS run; swapped for rmix @10
  +1.7 ± 2.7 / -3.5 ± 2.3 / +2.9 ± 4.2 %. The smaller number of in-batch negatives was not hiding a gain.
- 18:27 launched `mi_rmix_ho5k_lag5_don` f1 (513): `--op_donors op_donors_41.npz` (32 MSS + 9 ds007964 people). Queued:
  `mi_rmix_ho5k_lag5_ds3` (`--extra_trials_dir ds007964 --extra_shift 4 --extra_repeat 3`) f1 on 518 and f3 on 516.
- `mi_rmix_ho5k_lag5_mssB` f3 (batch 208): alone @10 / @12 1.500 / 1.556 (5k r=3), 1.840 / 1.896 (r=4), 0.604 / 0.608 (14k)
  vs reference @10 1.456 / 1.688 / 0.640; swapped for rmix @12 +2.5 ± 2.5 / +2.3 ± 2.2 / -4.4 ± 3.9 %. Two folds pooled:
  about +2 % at 5k r=3, 0 at r=4, -1 % at 14k, each within one standard error. Batch size closed.
- Queued on 513 after the donor run: `mi_rmix_ho5k_lag5_mssFT` f1 = MSS-trained rmix @8 fine-tuned 4 epochs on our data only
  (lr 2e-4): MSS as pretraining. (A first attempt shared the 8-core interactive node with SQF-X and both crawled; stopped.)
- Q08 with all 19 ds007964 donors, frozen rmix lag f1 @10: through a ds007964 donor 1.280 (-1.8 ± 4.2 %), through an MSS donor
  1.012 (-22.4 ± 4.4 %). Real foreign heads with real electrodes are already covered; MSS operators are foreign mostly through
  the 9 interpolated electrodes.

### Q09 second clean proxy for ds007964 (12 Nov, 19:05; user: change the proxy exceptionally to reduce the overlap)

- First proxy: 501 cohort images are in the held-out 5,000 and 336 are 14k distractors -> the cohort trains on 1,039 of 1,900
  images (58,203 of 106,6xx trials; `mi_rmix_ho5k_lag5_ds3` f1, x3 = 13 % of the batches, running on 518).
- Second proxy: `--holdout_avoid <cohort targets.npz>` draws the 5,000 held-out images among the 15,000 training images that
  are not cohort images (seed 0; 1,502 shared with the first set); the cohort keeps every trial. Scoring `nq09_proxy2.py`
  (5k r=3 / r=4; about 14k with the distractors that are not cohort images). New references needed; members alone only.
- Runs (rmix lag recipe, 12 epochs): reference `mi_rmix_h2_lag5` f1 (516 after SQF-X) and f3 (518 after ds3 f1);
  `mi_rmix_h2_lag5_ds2` (cohort x2 = 15.6 % of the batches) f1 (513 after the donor run) and f3 (516 after the reference).
  Then `mi_rmix_ho5k_lag5_mssFT` on 513.
- SQF-X with MSS, f1 (`mi_sqfx_ho5k_lag5_mssS` @40): alone 1.192 / 1.404 / 0.512 vs reference 1.228 / 1.316 / 0.396; swapped
  for SQF-X -1.0 ± 2.4 / -3.5 ± 2.1 / -6.9 ± 3.5 %. Triple swap (rmix@12 + d6@10 + SQF-X@40 with MSS) +1.2 ± 3.5 / +1.2 ± 3.2 /
  -6.3 ± 5.3 %. No accumulation over three members.
- 19:10 user: 4 GPUs again -> job 74489 `mi_gpu2` (515).
- **Q09 artefact (19:50): a foreign cohort in ONE loss with our people costs retrieval, even on THINGS images.**
  `mi_rmix_ho5k_lag5_ds3` f1 (first proxy, joint loss, 13 % of the batches): sealed-rule per-epoch 14.7 16.7 19.8 20.3 21.4 21.9
  22.3 vs reference 18.1 21.4 25.6 26.2 27.7 27.9 29.1; train loss 3.85 vs 4.10 at epoch 7 (the cohort's trials near
  log(17) = chance among the cohort images of the batch: inferred, not measured). Same signature as Q04 (MSS joint loss):
  the cohort is readable from the EEG, its targets lie in its own image subset, so the model adds a cohort offset to the
  output; our trials get the opposite offset. Q04's "another image set" was too narrow: a subset of our images does it too.
  Fix: the loss is split per COHORT (`subject id >= 100`), not per image set. Runs stopped and removed: ds3 f1, ds2 f1.
  Relaunched with the per-cohort loss: `mi_rmix_h2_lag5_ds2` f1 (515), f3 queued on 513; references f1 (516), f3 (518).
- Fix confirmed at epoch 1 (second proxy, f1): `mi_rmix_h2_lag5_ds2` 18.5 vs reference `mi_rmix_h2_lag5` 17.7 (joint loss: 14.7 vs 18.1).
- Q08 donors, `mi_rmix_ho5k_lag5_don` f1 (41 donors, half of the partners): alone @10 1.440 / 1.728 / 0.612 vs reference
  1.424 / 1.668 / 0.672; swapped for rmix @10 +0.2 ± 2.5 / +0.0 ± 2.1 / -3.4 ± 3.9 %. Neutral: donor operators closed
  (as the October random-orientation control predicted).
- **MSS closed (20:00, user: "let's give up MSS").** Joint training with a per-set loss (rmix 2 folds, d6, SQF-X), batch 208
  (2 folds), double and triple swaps, donors: all within noise at 5k and 3-7 % low at 14k. `/nasbrain/p20fores/mss_ds005589`
  deleted (8.2 GB). Kept: scripts nq01-nq03, `op_donors_mss.npz`, the run directories `mi_*_mssS`, `mi_*_mssB`.
- 20:50 mid-training, second proxy, sealed-rule per-epoch: f1 ds2 18.5 23.6 25.6 27.8 27.9 29.0 29.3 vs reference 17.7 23.0 26.1
  28.1 29.2 28.9 29.1 29.2; f3 ds2 16.9 21.7 24.9 26.4 27.0 28.2 vs 16.6 21.6 24.4 25.1 25.7 25.6 25.9 28.0 28.0.
- `--extra_bank` (train.py `extra_bank_loss`): each extra-cohort trial is scored against ALL the cohort's images (1,900) instead
  of the ~20 cohort images of its batch. Smoke-tested 4,000 steps.
- **Night queue (21:00, user: wifi may drop around 22:00), detached (`setsid nohup`), one per GPU, each starts when its GPU has
  been free for 3 minutes (`wait_gpu.sh`); launcher `h2b.sh <fold> <variant>`; logs `logs/mi/queue_<gpu>.log`:**
  516: d6 reference f1 -> d6 reference f3 -> SQF-X reference f1 | 518: d6 + cohort x2 f1 -> f3 -> SQF-X + cohort x2 f1 |
  513: rmix bank f3 -> rmix cohort x4 f3 | 515: rmix bank f1 -> rmix cohort x4 f1.
  `h2_scorer.sh` (detached) scores each finished pair with `nq09_proxy2.py` -> `logs/mi/h2_scores.txt`.
- 21:10 user: also try weaker oversampling -> `ds1` (cohort x1, 8.4 % of the batches) runs first on the 513 (f3) and 515 (f1) queues,
  then bank, then x4.
- **Q09 fold 1, second proxy, rmix alone (5k r=3 / 5k r=4 / 14k r=3), cohort x2 with the per-cohort loss vs reference:**
  @8 1.600 / 1.784 / 0.596 vs 1.492 / 1.728 / 0.656; @10 1.536 / 1.852 / 0.660 vs 1.472 / 1.660 / 0.656 (+4.3 ± 4.8 /
  +11.6 ± 4.7 / +0.6 ± 7.7 %); @12 1.428 / 1.760 / 0.580 vs 1.392 / 1.716 / 0.572. Positive at 5k at every epoch (+3..+12 %),
  flat at 14k; one fold, each within about one to two standard errors.
  (`nq09_proxy2.py` had a device bug in the 14k gallery; fixed before the scorer's first pass.)
- **Q09 fold 3, same comparison:** @8 1.468 / 1.692 / 0.596 vs 1.496 / 1.684 / 0.536; @10 1.460 / 1.720 / 0.568 vs 1.512 / 1.716 /
  0.580 (-3.4 ± 4.6 / +0.2 ± 4.2 / -2.1 ± 7.9 %); @12 1.552 / 1.692 / 0.604 vs 1.492 / 1.648 / 0.568. Flat.
  Two folds x three epochs averaged: about +2 % (5k r=3), +3.5 % (5k r=4), +1 % (14k); no single cell beyond two standard errors.
  rmix + ds007964 x2 (15.6 % of the batches, per-cohort in-batch loss) = weak positive at most. Dose (x1, x4), the bank loss and
  REVE-d6 / SQF-X follow overnight (`logs/mi/h2_scores.txt`).
- MSS as pretraining, `mi_rmix_ho5k_lag5_mssFT` f1 (MSS-trained rmix @8, then 4 epochs on our data only, lr 2e-4): alone @2 / @4
  1.552 / 1.516 (5k r=3), 1.676 / 1.700 (r=4), 0.628 / 0.588 (14k) vs reference @10 1.424 / 1.668 / 0.672; swapped for rmix @4
  +1.2 ± 2.7 / -1.4 ± 2.5 / -1.1 ± 4.1 %. Neutral, like every other use of MSS. Last MSS run.
- 22:00: the 515 queue started `mi_rmix_h2_lag5_ds1` f1; all four GPUs are on the ds007964 night queue.

### Q10 ds004018 (12 Nov, 22:10): 16 people, 200 object images at 5 Hz (user approved it with ds007964)

- OpenNeuro ds004018 (CC0; Grootswagers, Robinson, Carlson 2019, Sydney): 63 electrodes + reference Cz, 1000 Hz, BrainVision.
  Run 1 = 5 Hz (100 ms image, 200 ms between onsets), 8,000 presentations = 200 segmented-object images x 40; run 2 = 20 Hz
  (not used). The bucket's current listing only holds sub-01; the snapshot endpoint
  `openneuro.org/crn/datasets/ds004018/snapshots/2.0.0/files/...` serves the others (`/nasbrain/p20fores/ds004018/get.sh`).
- `nq10_ds004018.py targets` (DINOv2-g layer 26 token mean; transparent background filled mid-grey) and `nq10_ds004018.py <sub>`
  (chain shared with nq06) -> `export/sub-XX.npz`, 8,000 trials each, subject ids 300 + XX. No image overlap with ours: first proxy.
- Zero-shot through rmix lag f1 @10 (`COHORT=... nq07`, gallery = its 200 images, chance top-5 2.5 %): single trial top-5 7.0-9.2 %,
  P(target > random) .63-.68 at shift +4 (people 01-04; +4..+6); the 40 trials of an image averaged: 17-20 %, P .79-.83.
  Better than MSS and ds007964 per trial. `--extra_shift 5`.
- **Night queue re-ordered (22:25; current runs kept):**
  516: d6 reference f1 (running) -> d6 reference f3 -> rmix + ds004018 f1 (`g18`, in-batch per-cohort) -> SQF-X reference f1
  518: d6 + ds007964 f1 (running) -> d6 + ds007964 f3 -> rmix + ds004018 f3 (`g18`) -> SQF-X + ds007964 f1
  513: rmix ds007964 x1 f3 (running) -> bank f3 -> rmix + ds004018 bank f3 (`g18bank`, 200-way) -> ds007964 x4 f3
  515: rmix ds007964 x1 f1 (running) -> bank f1 -> rmix + ds004018 bank f1 -> ds007964 x4 f1
  The scorer also runs `ni01_new_member.py` on the `mi_rmix_ho5k_lag5_g18*` runs (first proxy: alone and swapped into the five).
- **Q09 REVE-d6 f1 (second proxy), cohort x2 vs reference (5k r=3 / 5k r=4 / 14k):** @8 1.468 / 1.780 / 0.580 vs 1.564 / 1.748 /
  0.612; @10 1.460 / 1.720 / 0.564 vs 1.500 / 1.784 / 0.644 (-2.7 ± 4.7 / -3.6 ± 4.1 / -12.4 ± 7.4 %); @12 1.428 / 1.712 /
  0.568 vs 1.464 / 1.756 / 0.676. Slightly negative, although its sealed-rule per-epoch curve was above the reference
  (30.7 vs 29.3 at epoch 8): that per-epoch check does not predict the proxy here.
- **Q09 rmix x1 f3:** @8 1.472 / 1.660 / 0.552, @10 1.496 / 1.800 / 0.532, @12 1.424 / 1.764 / 0.600 vs reference 1.496 / 1.684 /
  0.536, 1.512 / 1.716 / 0.580, 1.492 / 1.648 / 0.568. Flat.
- Standing at 23:30: ds007964 with the in-batch per-cohort loss = no effect (rmix x2 two folds about +2 %, rmix x1 f3 flat,
  d6 x2 f1 about -3 %). Still to come: bank loss, x4, d6 f3, and ds004018.

### Q11 a lower-variance read of the proxy (12 Nov, 23:45)

- Top-5 at 5k is a rare event (1.3 % of the r=3 queries): ± 4-5 % relative per fold and run, too coarse for effects of a few
  percent. `nq11_sensitive.py <folds> <reference> <candidate>` reads the same cached outputs with the epochs 8/10/12 averaged
  in rank space per query, paired bootstrap over queries: top-5, top-20, top-100 and 1 - AUC (mean rank / 4999, lower = better).
  Query-resampling errors: top-5 ± 3-5 %, top-20 ± 2-3 %, top-100 ± 1 %, 1 - AUC ± 0.2-0.3 %. They do NOT include the
  training-seed variance (not measured on this proxy).
- ds007964, in-batch per-cohort loss:
  | Comparison | top-5 | top-20 | top-100 | 1 - AUC |
  |---|---|---|---|---|
  | rmix x2, folds 1+3 | -0.6 ± 3.3 % | +1.1 ± 1.8 % | +0.5 ± 0.9 % | +0.0 ± 0.2 % |
  | rmix x1, fold 3 | +0.9 ± 4.9 % | -1.1 ± 2.6 % | +1.9 ± 1.2 % | -1.0 ± 0.2 % |
  | REVE-d6 x2, fold 1 | -6.5 ± 4.8 % | +1.4 ± 2.7 % | -1.0 ± 1.2 % | +0.3 ± 0.3 % |
  A null at the 1 % level. Expected size if the cohort were worth as much per trial as THINGS-EEG2 (0.7 person per 66k trials):
  107k trials = about 1.1 person on top of 25 = +1-2 %: below what top-5 can resolve, at the edge of what top-100 / AUC can.
- **Q09 rmix x1, both folds (00:10):** f1 @8 / @10 / @12 1.524 / 1.464 / 1.480 (5k r=3) vs 1.492 / 1.472 / 1.392; Q11 read, folds 1+3:
  top-5 +0.8 ± 3.3 %, top-20 +1.1 ± 1.8 %, top-100 +1.6 ± 0.8 %, 1 - AUC -1.0 ± 0.2 %. A small consistent gain at x1 where x2 gave
  none (top-100 +0.5 ± 0.9, AUC +0.0 ± 0.2): the size expected from about one extra person-equivalent. Whether 1 % exceeds the
  seed noise is not known: `mi_rmix_h2_lag5_s7` (reference, `--seed 7`) queued on 515 (f1) and 513 (f3) after the x4 runs.

## Night of 12-13 Nov, 00:30-03:30: the cohort lead closes; new goal (user, 03:10): +10 % on the next submission, 4 GPUs

### Q09 / Q10 remaining cohort runs (scored by `h2_scorer.sh`, `logs/mi/h2_scores.txt`)
- **ds007964, bank loss** (each cohort trial against all 1,900 cohort images), rmix, second proxy, vs reference @10
  (5k r=3 / r=4 / 14k): f3 @10 1.384 / 1.712 / 0.496 (-8.5 ± 4.7 / -0.2 / -14.5 ± 7.6 %); f1 @10 1.452 / 1.760 / 0.524
  (-1.4 / +6.0 / -20.1 ± 6.4 %). Negative, clearly so at 14k: the 1,900-way term weighs too much on the shared head. Closed.
- **ds007964, REVE-d6 x2, f3**: @8 +1.4 / +2.0 / -9.3, @10 +2.2 / -2.3 / +3.7, @12 -7.9 / -10.7 / -9.3 %. Flat to negative.
- **ds004018** (first proxy, rmix x1 = 10 % of the batches), alone vs reference @10 1.424 / 1.668 / 0.672 (f1), 1.456 / 1.688 /
  0.640 (f3): in-batch per-cohort f1 @10 1.420 / 1.572 / 0.584, f3 1.384 / 1.840 / 0.564; bank (200-way) f1 1.548 / 1.644 /
  0.600, f3 1.428 / 1.716 / 0.596. Swapped for rmix in the lag five @10: in-batch -1.9 ± 2.3 / -1.4 / -4.0 (f1), -1.7 ± 2.4 /
  +2.5 / +1.7 (f3); bank -1.4 ± 2.5 / +0.4 / +0.6 (f1), -1.7 ± 2.4 / +2.9 ± 1.9 / +0.6 (f3). Neutral.
- **Verdict on external cohorts (MSS, ds007964, ds004018):** with a joint loss they wreck the member (cohort shortcut); with
  a per-cohort loss none moves the proxy beyond about 1 %. Read with trial counts: THINGS-EEG2 gives about 0.7 person per
  66k trials on our own images; 107k-128k foreign trials are worth about one person out of 25, i.e. +1-2 %, which is what
  ds007964 x1 shows (top-100 +1.6 ± 0.8 %, 1 - AUC -1.0 ± 0.2 %) and what top-5 cannot resolve. No route to +10 % here.
  Stopped at 03:20 to free the GPUs: ds007964 x4 (both folds, 1 h in), the SQF-X pair (1.5 h in). Seed references
  (`mi_rmix_h2_lag5_s7`) stay queued behind the new runs on 513 / 515.

### R01-R03 cheap probes on a frozen member (rmix lag f1 @10), all closed
- R01 `nr01_person_probe.py`: every held-out person decodes best at input shift 0 (or -1 within noise): no latency lever
  (re-confirms O05, which I should have re-read first). The output is exactly invariant to the input gain.
- R02 `nr02_more_members.py`: sub16's six + the night's neutral-data runs as extra copies of rmix / d6 / SQF-X (stand-ins
  for second seeds): 9 members top-5 -1.4 ± 2.7 %, 12 members -3.5 ± 2.8 % (5k r=3), top-100 +2.0-2.3 ± 0.7 %, 1 - AUC
  -1.4 ± 0.1 %. More copies of the same families sharpen the bulk of the ranking and do nothing for the top-5. Also a
  calibration: +2 % top-100 / -1.5 % ranking error can coexist with 0 % top-5.
- R03 `nr03_key_gain.py`: adding beta x the person's key (image-independent mean ERP) to each trial. Oracle key: beta
  +0.5 top-5 1.276, +1 1.116, -0.5 1.156, -1 0.924 vs intact 1.304: the optimum is the natural amplitude. Key from the
  person's ~12 batch-mates: cosine .20 with the oracle key, beta +0.5 1.188. A test batch cannot supply a usable key
  (key = 6 % of the trial norm); in-context person statistics from a handful of trials are too noisy. Closed.

### R04 repeat-averaging SNR augmentation (launched 03:25)
- Reasoning: the score is noise-limited per query (r=3 -> r=4 is +20 %; all ~20 trials of an image over 5 people: 6.2 % vs
  1.46 %), and the cleaner THINGS-EEG2 people help new Alljoined people. `--repeat_mix p` (module/dataset.py, unit-tested):
  a training trial is replaced with probability p by its mean with another repeat of the same person and image. The
  October "repeat-mean" lever was judged on the old seen-image, 80-repeat metric; not tested on the clean proxy.
- Runs (first proxy; scored by ni01 alone and swapped into the lag five): `mi_rmix_ho5k_lag5_rm50` f1 (515) and f3 (513),
  `mi_d6_ho5k_lag5_rm50` f1 (516), `mi_rmix_ho5k_lag5_rm100` f1 (518). Then the seed references on 513 / 515.

### R05 the official test-run trials are not noisier: their deficit sits only at the top of the ranking (`nr04_testrun_vs_image.py`)
E11 / E14 left open why test-run trials are ~45 % harder at top-5. Frozen rmix lag f1 @10, single trials, one common gallery
(200 test images + the 5k held-out images):
- P(target > random image): training-run trials .662, test-run trials .673 (per person .757/.777, .637/.654, .715/.704,
  .582/.629, .618/.600). Flat over image typicality in both sets. The test-run trials carry as much information.
- Target in the top n (training-run / test-run): top-5 0.72 / 0.39 %, top-20 2.32 / 1.40, top-100 8.2 / 6.1, top-500
  25.8 / 24.1, top half 71.7 / 74.0. The deficit exists only at the extreme top.
- Test images are more typical (cosine to the gallery mean .839 vs .822); re-weighting the training-run trials to the test
  typicality mix gives 0.57 % (of 0.72 -> 0.39). Test images are also under-represented among the winners of any query
  (2.2-2.6 % of the top-1 for 3.85 % of the gallery): images of never-trained concepts are not output attractors, so they
  rarely reach the top-5 against held-out images of trained concepts. A gallery-composition effect of the top-5 metric,
  not a property of the runs. It does not explain the sealed gap (E00): the sealed gallery holds only new images.
- **R04 result (05:10): repeat averaging is not a lever.** rmix f1 alone (5k r=3 / r=4 / 14k) vs reference @10 1.424 / 1.668 /
  0.672: p 0.5 @8 / @10 / @12 1.372 / 1.352 / 1.352, 1.660 / 1.532 / 1.496, 0.576 / 0.604 / 0.544; p 1.0 1.308 / 1.264 / 1.104,
  1.440 / 1.412 / 1.256, 0.532 / 0.556 / 0.508. Swapped for rmix: p 0.5 -2.9 / -1.2 / +3.1 ± 2.6 % (5k r=3), p 1.0 -0.5 / -2.9 /
  -1.2 %. As a seventh member of sub16's six (cached): p 1.0 @10 -0.9 ± 2.3 / +2.2 ± 2.2 / -1.2 ± 3.4 %; p 0.5 @12 -0.2 / +0.4 /
  -1.2 %; both -2.8 / +0.8 / -0.6 %. The variants learn faster (per-epoch curve ahead until epoch 5) and overfit sooner; their
  evidence is a little less correlated with rmix (z-score correlation .80-.85 against .88 for a second seed) but that buys
  nothing at the top of the ranking. Closed (d6 p 0.5 f1 and rmix f3 scores land in `h2_scores.txt` for the record).
- 05:15 `mi_plab_ho5k_lag5_jit3` fold 2 launched on 518 (the second LaBraM of sub17): gives the 7-vs-6 comparison a third fold
  (`logs/mi/ni05_plab2_f123.out`). Seed references `mi_rmix_h2_lag5_s7` running on 515 (f1) and 513 (f3).

## R06-R07 label-free latency alignment per batch: free at zero offset, recovers everything under an offset (13 Nov, 05:20-06:10)

Why: G01/G07 (11 Nov) showed that a cohort latency offset of +-4 samples would cost the models about 2x, the size of the
unexplained sealed gap (E00); jitter only softens it. The jittered sub16 gained +19.4 % on the board where the proxy at
zero offset predicted +10-11 %. Every cohort aligned this week had its own offset against Alljoined-1.6M (THINGS-EEG2 +5,
ds007964 +4/+5, ds004018 +5, MSS -3). The sealed cohort is a new recording campaign; its offset is unknown.

- **R06 a 64-trial batch reveals its own alignment without labels** (`nr05_autolag.py`, frozen rmix lag, held-out people).
  The mean output norm of the member over the batch, as a function of an input shift, peaks next to the decoding optimum:
  fold 1 mean curve 33.0 / 38.8 / 39.3 / 38.5 / 34.9 / 31.5 at -4 / -2 / -1 / 0 / +2 / +4; per-batch argmax -1.14 ± 0.61 (99 %
  within ±2 of the truth); fold 3 -1.43 ± 0.73 (95 %). The peak is 1.1-1.4 samples early on new Alljoined people in both
  folds. Cross-correlating the batch-mean ERP with the training mean key also works but is noisier and its bias changes
  with the fold (-1.0 ± 1.3, +0.4 ± 1.3).
  On cohorts with a known offset (`nr06_autolag_cohorts.py`, no labels): ds007964 +5.3 ± 1.5, ds004018 +5.6 ± 0.9 per batch,
  against labelled optima of +3..+6 and +4..+6.
- **R07 end to end** (`nr07_autolag_e2e.py 1`): sub16's composition on fold 1 (d6 / rmix / SQF-X lag + jitter, REVE-scratch,
  SQF, LaBraM), data delayed by d; detector = rmix lag-only, 9 candidates (-8..+8 step 2), vertex of the parabola through
  the three points around the maximum, + 1.5, rounded. Top-5 % 5k r=3 / r=4 / 14k r=3:
  | offset | as is | aligned | gain |
  |---|---|---|---|
  | -4 | 1.077 / 1.350 / 0.477 | 1.666 / 2.031 / 0.633 | +54.6 ± 9.1 / +50.4 ± 8.2 / +32.8 ± 11.6 % |
  | -2 | 1.470 / 1.786 / 0.613 | same | +13.4 ± 4.6 / +13.7 ± 4.2 / +3.3 ± 6.9 % |
  | 0 | 1.622 / 2.043 / 0.657 | same | +2.7 ± 2.3 / -0.6 ± 1.8 / -3.7 ± 4.0 % |
  | +2 | 1.606 / 1.914 / 0.637 | same | +3.7 ± 3.8 / +6.1 ± 3.3 / -0.6 ± 6.0 % |
  | +4 | 1.362 / 1.622 / 0.525 | same | +22.4 ± 6.5 / +25.2 ± 5.7 / +20.6 ± 10.4 % |
  Residual shift after alignment +0.40 ± 0.51 samples. Constants +1 and +2 and a 13-candidate argmax are within noise of this
  (argmax + 1 costs -4.4 ± 2.3 % at zero offset: its residual sits on the steep side, -0.14 ± 0.61).
- **Packages (06:05):** `submission.py` gains `align()` (used only when members.pt names a detector; batches under 32 trials
  are left untouched), `package_submission.py --detector / --bias / --verify <fold> --delay d`.
  `sub18_ens7_lag5jit_labram2_align.zip` = sub17 + alignment (435 MB, sha256 edcf1ed4...2830be);
  `sub19_ens6_lag5jit_labram_align.zip` = sub16 + alignment (412 MB, sha256 058cb082...758da50d): the clean A/B against the
  0.001258 already on the board. Detector = the lag-only final rmix (sub13's member1_gen.pt), shifts -8..+8 step 2, bias 1.5.
  Contract check clean for both (82 s / 53 s). Fold-1 replay (in-sample, 200-image gallery): sub18 34.54 / 34.34 at zero
  delay (sub17 34.46 / 34.31); stream delayed by +4: sub18 34.23 / 33.96, sub17 30.93 / 30.99. Time per 64-batch on the shared
  3080: sub16 110 ms, sub17 122 ms, sub18 156 ms (x1.42 of sub16: about 36 min on the grader).
- What it is worth on the board is unknown: nothing if the sealed cohort has no offset (cost about 0), +5-15 % at 2 samples,
  +20-50 % at 4. The board is the only measurement. Fold 3 end-to-end running (`logs/mi/nr07_f3.out`).
- **R08 the real packages on real cohorts with a real offset** (`nr08_package_on_cohorts.py`; 8 people x 4,000 trials each,
  64-trial batches mixing people; neither cohort trained these finals): top-5 %, single trial / 3 trials averaged:
  ds004018 (200-image gallery, chance 2.5 %): sub16 6.84 / 10.62 -> sub19 (sub16 + alignment) 7.78 / 12.19 (+14 % / +15 %);
  ds007964 (1,900-image gallery, chance 0.26 %): sub16 0.82 / 1.17 -> sub19 1.17 / 1.87 (+43 % / +60 %).
  The alignment step finds and corrects an offset it was never told about, in the deployed code path.
- R04 remaining scores: rmix p 0.5 f3 swapped for rmix @10 / @12 -0.5 / +0.2 ± 2.5 %; d6 p 0.5 f1 swapped for d6 +0.7 / +1.7 ± 2.8 %
  (alone 1.37 / 1.26 vs 1.43): neutral, as on fold 1.
- **R07 fold 3** (`nr07_autolag_e2e.py 3`; d6 / rmix / SQF-X lag-only + LaBraM, detector = fold-3 rmix lag-only, parabola + 1.5,
  residual +0.20 ± 0.53): offset 0: 1.642 / 2.079 / 0.693 -> 1.666 / 1.987 / 0.689 (+1.5 ± 2.2 / -4.4 ± 1.9 / -0.6 ± 3.5 %);
  offset -2: +19.2 ± 5.6 / +24.6 ± 5.3 / +27.4 ± 9.8 %; +2: +6.1 / +3.8 / +6.2 %; -4: +84 / +77 / +100 %; +4: +33 / +30 / +29 %.
  Two folds at zero offset: 5k r=3 +2.7 and +1.5, 5k r=4 -0.6 and -4.4, 14k -3.7 and -0.6: a cost of about 1 % on average,
  inside the noise. Fold 2 (the weakest people, six lag-only members) queued: `logs/mi/nr07_f2.out`.
- **R07 fold 2, the weakest people** (`nr05_autolag.py ... 2`, `nr07_autolag_e2e.py 2`; six lag-only members): here the norm of
  rmix peaks at -0.09 ± 0.77 (folds 1 / 3: -1.14, -1.43), so with the constant +1.5 the residual is +1.54 ± 0.72 and the
  zero-offset cost is -5.4 ± 4.5 / -8.7 ± 4.0 / -4.5 ± 6.5 % (5k r=3 / r=4 / 14k). Offsets: -2 +17.7 / +20.8 / +34.7 %,
  +2 +3.0 / -3.1 / +2.4 %, -4 +101 / +73 / +146 %, +4 +43 / +34 / +52 %.
  The calibration constant depends on the group of people (the peak sits 0.6 to 2.0 samples before each fold's own decoding
  optimum): the residual after alignment is uncertain by about ± 0.7 sample. Three folds at zero offset: 5k r=3 +2.7 / +1.5 /
  -5.4, r=4 -0.6 / -4.4 / -8.7, 14k -3.7 / -0.6 / -4.5 %: mean cost about 2.5 % with lag-only members (folds 2, 3), about
  0.5 % with the jittered composition of the packages (fold 1). +1.5 stays (centre of the three folds).
- **Seed noise of the second proxy** (`mi_rmix_h2_lag5_s7` vs `mi_rmix_h2_lag5`, same recipe): alone @8 / @10 / @12, f1 +6.0 /
  +3.3 / -4.9 % (5k r=3), +9.4 / +4.3 / +1.9 % (r=4), -6.7 / -7.9 / -3.0 % (14k); f3 -3.4 / -9.3 / -7.9, -4.0 / -7.5 / -4.4,
  +2.8 / -0.7 / -2.8 %. Q11 read, folds 1+3: top-5 -5.5 ± 3.1 %, top-20 +0.9 ± 1.8, top-100 +0.1 ± 0.8, 1 - AUC -0.1 ± 0.2 %.
  Two trainings of one recipe differ by 5-9 % in top-5 on a fold: no single-member top-5 difference below about 10 % on one
  fold means anything; top-100 and 1 - AUC are stable across seeds. Against this, ds007964 x1 (top-100 +1.6 ± 0.8, 1 - AUC
  -1.0 ± 0.2) is a real but tiny gain in the bulk of the ranking, invisible at top-5.
- **Seven vs six on three folds** (`ni05_stack_members.py 1,2,3 mi_plab_ho5k_lag5_jit3:10`, second LaBraM trained on fold 2
  tonight): six + LaBraM lag+jitter @10 vs six: 5k r=3 +1.1 ± 1.3 % (folds +0 / -1 / +3), 5k r=4 +1.7 ± 1.2 %, 14k r=3
  +1.6 ± 2.0 %, 14k r=4 -3.3 ± 1.7 % (folds -2 / -4 / -4). I18's +1.8 ± 1.5 % on two folds does not strengthen with a third:
  the seventh member is neutral within noise. sub17 ~ sub16. Consequence for the alignment test: sub19 (sub16 + alignment)
  is the upload to prefer, same expected score as sub18 and a clean comparison with the board's 0.001258.
- **R10 the person-dependence of the calibration is in the people, not in the detector** (`nr10_detector_choice.py`, norm curves
  of rmix, d6, SQF-X, REVE-scratch, SQF, LaBraM on folds 1-3): every architecture peaks 1.8-1.9 samples before the ensemble's
  decoding optimum for the people of folds 1 and 3 and 0.5-0.9 for those of fold 2 (range over folds 0.9-1.4 for every
  detector and every pair or triple; SQF-X alone 0.93 but noisy). Ensemble coherence and centred norms (`nr09`) are no
  steadier. The ensemble's decoding optimum sits at +0.90 / +0.68 / +0.63 samples (folds 1 / 3 / 2).
- **R11 mixed batches, as in the sealed stream** (`nr11_autolag_mixed.py`: one 64-batch from each fold = 15 people, one shift
  for the 192 trials, each fold decoded by its own models; 75,000 queries pooled; 5k r=3 / r=4 / 14k r=3): residual
  +0.70 ± 0.47 (= the decoding optimum). Delay 0: 1.482 / 1.833 / 0.629 -> 1.510 / 1.788 / 0.638 (+1.9 ± 1.9 / -2.5 ± 1.5 /
  +1.5 ± 3.0 %; folds +3.7 / +1.4 / +0.5 % at 5k r=3). Delay -2: +18.7 ± 3.5 / +21.4 ± 3.3 / +24.5 ± 5.7 %; +2: +6.5 ± 2.2 /
  +4.3 ± 2.1 / +7.2 ± 3.9 %; -4: +78 / +67 / +86 %; +4: +34 / +31 / +37 %.
  The fold-2 cost of R07 came from batches made of that fold's people only; with people mixed the estimate is the cohort
  average and the zero-delay cost is nil. The packages stay as built (rmix detector, bias 1.5).
- **R12 which sealed offset do our two board scores fit?** (`nr12_board_offset.py`, fold 1: the sub-2 composition against the
  sub16 composition under a common delay d; predicted ratio sub16 / sub 2 at r=1 and r=3.) Board: x1.135 (single trial),
  x1.194 (aggregated); board SE about 4 % each.
  | delay | 5k r=1 | 5k r=3 | 14k r=1 | 14k r=3 |
  |---|---|---|---|---|
  | -4 | 1.278 ± .054 | 1.043 ± .059 | 1.293 | 1.280 |
  | -3 | 1.185 ± .042 | 1.132 ± .056 | 1.264 | 1.143 |
  | -2 | 1.106 ± .034 | 1.136 ± .053 | 1.115 | 1.070 |
  | 0 | 1.082 ± .033 | 1.049 ± .041 | 1.045 ± .053 | 1.079 ± .072 |
  | +2 | 1.179 ± .039 | 1.166 ± .048 | 1.125 | 1.112 |
  | +3 | 1.184 ± .041 | 1.160 ± .051 | 1.244 | 1.200 |
  | +4 | 1.174 ± .043 | 1.219 ± .056 | 1.230 | 1.180 |
  Chi-square of the two board ratios against each row (proxy and board errors added), 5k / 14k: delay 0: 6.5 / 3.5; +2: 0.7 /
  1.0; +3: 0.9 / 1.6; +4: 0.5 / 1.3; -2: 1.0 / 2.5; -3: 1.4 / 2.4; -4: 8.4 / 2.7. The board is 3 to 18 times more likely under a
  delay of 2 to 4 samples than under none (one fold, a sub-2 stand-in at epoch 10, fold models instead of finals: suggestive,
  not proof). Under such a delay the mixed-batch alignment gain (R11) is +6.5 % (+2), +19 % (-2), +34 to +78 % (±4).
- 08:40 R12 on a second fold: training the sub16 composition on fold 3 (`mi_d6_ho5k_lag5_jit3` on 513, `mi_rmix_ho5k_lag5_jit3`
  on 515, `mi_sqfx_ho5k_lag5_jit3` on 518, `mi_scr_ho5k_lag5` then `mi_sqf_ho5k_lag5` on 516); `nr12_board_offset.py 3` runs
  by itself when all five are scored -> `logs/mi/nr12_f3.out` (about 13:30). Report doc updated with the alignment section.

# Round 9 (13 Nov, 12:00): test-time adaptation beyond timing (user: "What other components of the signal can we extract and adapt to besides timing?")
Ops: user 12:10 "you have 3 GPUs total for your experiments, including this one" -> 516 (interactive), 513 (74191), 515 (74489).
73733 (518) is left idle and not reused; its keep-alive ends by itself on 13 Nov 20:35. The user uploads sub19.
11:12 `mi_sqf_ho5k_lag5` f3 moved to 513 (chain `logs/mi/queue_513.log`, then `nr12_board_offset.py 3`); `mi_scr_ho5k_lag5` f3 finished on 516 at 12:15.
Closed before this round (do not repeat): key normalisation / key gain (N02, R03), output-side person offset (N06), covariance
re-rendering to the population (E03), per-person latency (O05, R01), trial confidence (M03, M06), RSVP neighbours (O02),
identity steering (M07), per-trial spatial noise filter (O04), bad-electrode repair (E20). Open with oracle headroom: a
per-person spatial operator fitted WITH labels (+14..18 % at r=3 on three folds, E17/E19), person-specific (E18).

## S01 oracle headroom of one-parameter input transforms (`ns01_transform_headroom.py`, launched 12:16)
Frozen sub16 composition of a fold; 31 settings: time scale about the onset, spectral tilt / band gains / cut-offs, spatial
smoothing or sharpening, re-reference, posterior gain, global gain, detrend, clip. Read (1) pooled against the identity,
(2) one setting per person chosen with labels (other half of the images) or by the detector norm (no labels).
Fold 1 on 515; `COHORT=ds004018` then `ds007964` (known shift applied first, fold-1 models) on 516.

## S02 averaging or choosing among shifted views of a trial (inline, cached `autolag_e2e_f{1,2,3}.npy`, sub16 composition, 5k r=3, 12:28)
- Mean of the ensemble output over input shifts -2..+2 against shift 0 (folds 1 / 2 / 3): top-5 +0.7 ± 2.3 / +2.7 ± 3.6 /
  +2.7 ± 2.8 %, top-100 -0.2 / +1.2 / +1.2 %, ranking error -0.2 / -0.4 / -0.5 %. Same profile as extra members of the same
  families (R02): the bulk of the ranking sharpens a little, the top-5 does not move beyond noise, at 5x the run time. Not a lever.
- Per trial, the view with the largest output norm: top-5 -5.9 ± 3.2 / -3.4 ± 4.9 / +1.7 ± 3.9 %, ranking error +0.7 / +2.2 /
  +1.5 % (worse). Norm gain and rank gain between views of one trial correlate at .03-.09: inside a trial the norm says
  nothing about which view decodes better. Softmax weighting over views: same as the plain mean.
- The "oracle best view per trial" (+76..139 %) is a best-of-five selection on noise (best shift is -2 or +2 for 30 % of the
  trials each, i.e. the two most different views), not a headroom.
- Verdict: the output norm is a COHORT-level criterion (it locates a common latency over 64 trials); it carries no
  per-trial information about the input transform. Per-trial view selection is closed.

## S03 BatchNorm statistics from the test batch (`ns03_test_time_bn.py 1`, the two members with BatchNorm, single member, r=3, 12:42)
Statistics = a x current 64-batch + (1 - a) x training. Top-5 against a = 0 (top-100 or ranking error where it resolves more):
| member | cohort | a .25 | a .5 | a 1 |
|---|---|---|---|---|
| SQF-X | fold 1 (ours) | +0.0 ± 2.2 % | -0.7 ± 2.7 % | -2.0 ± 3.2 % (top-100 -1.7 ± 0.8) |
| SQF-X | ds004018 | +1.1 ± 0.7 % | +0.7 ± 1.0 % | -0.4 ± 1.2 % |
| SQF-X | ds007964 | +1.9 ± 2.1 % | +3.1 ± 2.9 % | +5.1 ± 3.8 % (top-100 +5.5 ± 0.8) |
| SQF | fold 1 (ours) | +2.8 ± 2.8 % | +6.3 ± 3.6 % | +6.7 ± 4.4 % (top-100 +0.8 ± 1.0) |
| SQF | ds004018 | +2.3 ± 0.9 % | +4.0 ± 1.2 % | +2.0 ± 1.4 % |
| SQF | ds007964 | +0.2 ± 2.3 % | -0.2 ± 3.1 % | -3.1 ± 3.5 % (ranking error -0.7 ± 0.1) |
A real but small effect on cohorts recorded elsewhere (+1 to +5 % for one member, so about +1 % in a six-member sum where two
members have BatchNorm), nothing on ours. The training statistics already fit a foreign cohort once it is time-aligned. Not a lever.

### S01 results: fold 1 (ours) and ds004018 (another lab, time-aligned), sub16 composition, r=3 (13:05)
Sensitivity (pooled, against the identity; top-5 % / ranking error %), fold 1 | ds004018:
| setting | fold 1 top-5 | fold 1 ranking error | ds004018 top-5 | ds004018 ranking error |
|---|---|---|---|---|
| time scale 0.95 / 1.05 | -8.1 ± 3.8 / -10.4 ± 3.9 | +5.2 / +6.0 | -4.1 ± 1.2 / -4.7 ± 1.2 | +2.6 / +4.1 |
| time scale 0.9 / 1.1 | -45.4 / -35.1 | +19.6 / +16.9 | -17.5 / -19.3 | +12.5 / +15.9 |
| tilt f^-0.5 / f^+0.5 | -12.8 / -26.9 | +5.4 / +12.3 | -3.2 / -17.3 | +5.9 / +11.9 |
| high-pass 2 Hz | -13.3 ± 3.8 | +4.4 | -6.5 ± 1.1 | +5.1 |
| low-pass 30 Hz | -1.0 ± 1.7 | -0.1 | -0.2 ± 0.5 | 0.0 |
| band 1-7 Hz x0.5 / x2 | -21.5 / -8.6 | +9.6 / +4.6 | -12.2 / 0.0 | +8.6 / +3.1 |
| band 8-13 Hz x0.5 / x2 | -1.7 / -10.6 | +3.2 / +3.2 | -2.7 / -4.3 | +3.0 / +2.0 |
| band 14-30 Hz x0.5 / x2 | 0.0 / -13.8 | +0.7 / +6.9 | +1.2 ± 0.9 / -11.3 | -0.3 / +6.7 |
| band 31-60 Hz x0.5 / x2 | -0.2 / -4.9 | -0.1 / +0.5 | 0.0 / -0.9 | 0.0 / +0.3 |
| spatial -0.3 / +0.3 | -0.2 / +0.2 | 0.0 / +0.7 | +1.3 ± 0.6 / -2.5 | -0.7 / +1.9 |
| reference -0.5 / +0.5 / +1 | -1.7 / -2.0 / -8.4 | +0.3 / +0.2 / +1.2 | -1.7 / +1.4 ± 0.6 / +0.7 | +0.6 / 0.0 / +0.2 |
| posterior x0.7 / x1.4, gain x0.5 / x2, detrend, clip 5 / 10 | all within ±4 (noise) | ≤ +0.4 | all within ±2 | ≤ +0.7 |
- The members are sensitive to the time axis (a 5 % stretch costs 4-10 %, 10 % costs 20-45 %) and to the spectral shape below
  30 Hz (1-7 Hz content, tilt, a 2 Hz high-pass); they do not care about gamma, a 30 Hz low-pass, spatial blur, reference, gain.
- (1) No setting beats the identity on our people, and none beats it by more than +1.4 % on ds004018 once it is time-aligned:
  a cohort from another lab (other amplifier, 5 Hz RSVP) differs from ours along the time offset and on NO other of these axes.
- (2) One setting per person chosen WITH labels: fold 1 at most +1.5 ± 1.2 % (detrend), ranking error -0.6 % (band);
  ds004018 at most +1.8 ± 0.8 % (band), +1.4 ± 0.5 % (spatial). People do not differ along these axes either.
- The detector norm as a criterion: right for the time axis (identity picked for 5 / 5 of our people; it falls 6 % at a 5 %
  stretch), WRONG for the spectrum (a 2 Hz high-pass raises the norm 6-9 % and costs 6-13 %; it picks "1-7 Hz x2").
  The norm locates the response in time; it is not a general "decodability" read-out.
- Verdict: among one-parameter input components, timing is the only one worth adapting. Fold 2 and ds007964 running (replication).

## S04 member weights per cohort (`ns04_member_weights.py`, six members, r=3, 13:05): closed
Uniform sum against weights fitted with labels on half of the groups: fold 1 +2.7 ± 3.0 / -2.7 ± 2.7 % (ranking error -0.1 %),
ds004018 +0.7 / 0.0 % (-0.1 / 0.0), ds007964 +2.0 ± 3.7 / -2.0 ± 3.0 % (-0.7 / -0.9). A member's batch norm against its usual
value (ds004018: LaBraM 1.73, others 0.87-1.09; ds007964: 0.66-0.95) as a weight: -1.6 to -12.8 % on ds004018, -0.5 to
-9.4 % on ds007964. The uniform sum is the right weighting for a foreign cohort too. Leave-one-out: every member helps or is
neutral on every cohort (LaBraM: 0 on ours, -1.2 % on ds004018, -8.2 ± 2.1 % on ds007964 when removed).

## S02b stability of a trial's output under small input shifts as a confidence (inline, cached, folds 1-3, 13:15): closed
Consistency = cosine between the ensemble output at shift 0 and at ±1. Correlation with the trial quality (cosine to the
target): .29 / .27 / .30 against .45 / .43 / .45 for the norm; it shares .48-.55 with the norm and adds a partial r of
.06-.08. As a repeat weight (exp(b z), b ±.25, .5; with or without the norm regressed out): top-5 -4 to +1.7 % (noise),
ranking error +0.1 to +1.4 % (worse). Third confirmation of M03 / M06: the norm holds the usable confidence
(oracle exp(.5 z(q)) weights would give +44 to +51 %).

## S05 matching second moments of the input to the training population (`ns05_moment_match.py`, d6 + rmix + SQF-X jitter of fold 1, r=3, 13:30): closed
Distance of the cohort's channel covariance to the Alljoined training population (relative Frobenius): ours 0.29,
ds004018 0.75, ds007964 0.68. Top-5 against no transform (ranking error in brackets), statistics per cohort / batch / person:
| transform | fold 1 (ours) | ds004018 | ds007964 |
|---|---|---|---|
| covariance -> population | -0.5 / -6.1 ± 3.1 / +1.5 % (+0.6 / +1.1 / +0.1) | +1.1 ± 1.1 / +0.1 / +2.2 ± 1.2 % (-0.2 / +0.3 / -1.3) | -3.0 / -6.2 ± 3.4 / -2.3 % (+0.6 / +2.4 / +0.4) |
| per-electrode variance -> population | +2.3 ± 1.6 / +2.5 / +1.5 % (+0.1 / +0.3 / +0.2) | +0.2 / +0.8 / +1.3 ± 0.8 % (+0.6 each) | +0.9 / +0.9 / +1.3 % (0.0 / +0.1 / +0.1) |
| per-electrode spectrum -> population | -4.5 / -7.6 ± 3.6 / -7.6 % (+1.8 / +3.1 / +2.3) | +0.5 / -0.1 / -1.1 % (+3.0 / +2.9 / +2.9) | +2.1 / +0.2 / +5.5 ± 2.8 % (-0.7 / -0.4 / -1.4) |
Cohorts recorded elsewhere sit 2.5x further from the training covariance than our new people and still gain nothing from
being moved to it (operator transfer in training already covers that distance). Spectrum matching hurts ours and ds004018.

## S06 the packaged alignment step under other batch compositions (`ns06_align_batching.py 1|2|3`, 13:35)
Organisers (Discord, October): sealed batches "will be shuffled and may mix participants"; the warm-up loader served subject
after subject. Estimator of the package (lag-only rmix norm, grid -8..+8 step 2, vertex + 1.5), per-trial norms cached in
`s06_trialnorm_f<k>.npy`; decoding from `autolag_e2e_f<k>.npy` (fold 1 = the real sub16 composition; folds 2 and 3 = lag-only
members, no jitter, so their costs are upper bounds for the package). Against no alignment, zero delay:
| batches | fold 1 | fold 2 | fold 3 |
|---|---|---|---|
| mixed 64: residual shift | +0.40 (sd .51) | +1.54 (sd .72) | +0.20 (sd .53) |
| mixed 64: top-5 / top-100 / ranking error | +2.7 ± 2.3 / +0.1 / +0.1 % | -5.4 ± 4.5 / -4.8 ± 1.2 / +4.1 % | +1.5 ± 2.2 / +0.2 / +0.2 % |
| one person 64: residual shift per person | 0.0 +0.5 +0.7 +0.4 +1.8 | -3.3 +2.2 +1.2 +4.4 -1.6 | -0.1 +0.8 -0.3 +0.6 0.0 |
| one person 64: top-5 / top-100 / ranking error | -0.2 ± 2.5 / -2.0 ± 0.6 / +0.6 % | -12.2 ± 5.1 / -10.7 ± 1.3 / +7.4 % | +5.1 ± 2.5 / -1.5 ± 0.7 / +0.7 % |
- Batch size 32 / 128 behaves as 64 (fold 1). Under a delay of ±2 / ±4 the one-person batches recover nearly as much as
  mixed ones (fold 1, -4: +50 vs +55 %; +4: +19 vs +22 %; fold 2, -4: +86 vs +101 %).
- The weak point: the people of fold 2 (7, 14, 16, 17, 20 = the ones whose mean ERP is atypical, O05). With one-person
  batches the norm peak of each sits 1.2 to 4.4 samples from the decoding optimum, and a batch of fold-2 people alone is
  shifted +1.5 on average. Mixed with other people (nr11, 15 people) this averages out: delay 0 was -2.5 to +1.9 %.
- Consequence for sub19: fine if the sealed batches are shuffled; if they are served person by person and the cohort has
  atypical people, those people lose (up to -44 % for one of them here, lag-only members). Next: a criterion with less
  person bias (S07).

### S01 replication: fold 2 (lag-only six) and ds007964 (time-aligned), 14:10
- Fold 2 (the atypical people): no setting beats the identity pooled (best: low-pass 30 Hz top-100 +0.5 ± 0.6 %); one setting
  per person WITH labels: gain family top-5 +4.4 ± 2.0 %, top-100 +2.0 ± 0.6 %, ranking error -0.5 % (the BatchNorm members
  are not gain-invariant), every other family within noise. Same sensitivities as fold 1 (time scale ±5 %: -12 to -23 %).
- ds007964: no setting beats the identity pooled (ranking error at best -1.3 % for 1-7 Hz x0.5, top-5 -4.7 %). One setting
  per person WITH labels: band top-100 +4.3 ± 0.8 %, ranking error -3.3 %; time scale top-100 +2.9 ± 0.8 %, ranking error
  -2.1 %; top-5 within noise everywhere. By the detector norm: -10 % (band), -15 % (cut-off): wrong again outside the time axis.
- Verdict over four cohorts (folds 1, 2, ds004018, ds007964): after the time offset, no one-parameter input component is
  worth adapting, per cohort or per person. S01 closed.

## S07 other detector criteria (`ns07_detector_criteria.py 1|2|3`, detectors lag-only rmix + lag-only SQF-X, 14:00)
Estimate per 64-batch (vertex on the -8..+8 step 2 grid, no constant added). Decoding optimum: +0.90 / +0.63 / +0.68 (folds 1 / 2 / 3).
| criterion | mixed batches, mean (sd per batch), folds 1 / 2 / 3 | one-person batches, mean per person, fold 2 (people 7 14 16 17 20) | sd across people, folds 1 / 2 / 3 |
|---|---|---|---|
| norm of rmix (packaged) | -1.03 (.43) / +0.04 (.59) / -1.31 (.48) | -4.8 +0.8 -0.2 +4.3 -3.1 | 0.60 / 3.16 / 0.47 |
| the same, batch mean of the output removed | -1.06 / +0.04 / ~ | -4.9 +0.8 -0.2 +4.3 -3.2 | 0.60 / 3.21 / ~ |
| norm of SQF-X | -0.88 (.57) / -0.23 (1.04) / -0.76 (.58) | -3.8 +0.3 -0.6 +2.8 -0.8 | 0.43 / 2.12 / 0.70 |
| agreement rmix vs SQF-X (cosine per trial) | +0.47 (.85) / +0.20 (1.58) / -0.35 (.83) | -3.7 +1.3 +1.4 +4.6 -2.5 | 0.55 / 2.99 / 0.54 |
| agreement on the trial-specific part | +0.40 (.85) / +0.18 (1.49) / -0.33 (.80) | -3.9 +1.3 +1.3 +4.3 -2.4 | 0.66 / 2.91 / 0.63 |
- Decoding of each person against the input shift, six members, ranking error (inline, 13:50): EVERY person of folds 1 and 2
  is best at 0 (person 4 at -1, person 14 at +1), bootstrap sd 0.1-0.5 samples. So the displacement of people 7, 17, 20 is an
  error of the criteria, not a latency of those people (O05 / R01 confirmed with the finer metric).
- No criterion removes it: two architectures, their norms and their agreement, with or without the common output, put the
  same people at the same wrong place. Their curves: person 17 almost flat (peak 6 % above its mean against 18-43 % for
  typical people), person 20 weak (11 %), person 7 a broad plateau from -8 to -2 (20 %, looks confident).
- Agreement has a calibration that varies less across folds (distance to the optimum -0.43 / -0.43 / -1.03 against
  -1.93 / -0.59 / -1.99 for the norm) but twice the per-batch scatter and the run time of a second detector. Not adopted.

### S06b a confidence gate on the alignment step (`GATE=p ns06_align_batching.py`, act only if the curve peaks >= p above its mean)
| gate | fold 2, one-person batches, delay 0 (top-5 / ranking error) | fold 2 mixed, delay -4 | fold 1 mixed, delay -4 | fold 1 one person, delay -4 |
|---|---|---|---|---|
| none | -12.2 ± 5.1 % / +7.4 % | +100.7 % | +54.6 % | +50.2 % |
| 0.10 | -7.8 ± 5.0 % / +4.0 % | +97.8 % (97 % of trials acted on) | +54.6 % | +50.9 % |
| 0.15 | -4.7 ± 4.6 % / +2.6 % | +81.3 % (70 %) | +54.6 % | +46.5 % |
A gate at 0.10 removes the flat-curve people (17, 20) at no cost under a delay; it does not catch person 7. Worth adding
to a next package with the alignment step (two lines in `submission.py`); not built yet (waiting for the board result of sub19).

## R12 on fold 3 (`nr12_board_offset.py 3`, 14:10): the fold-1 hint of a sealed delay does NOT replicate
Board ratios sub16 / sub 2: x1.135 (single trial), x1.194 (aggregated). Fold 3 prediction with no delay: 5k x1.091 ± .031 /
x1.185 ± .045, 14k x1.128 ± .058 / x1.186 ± .072 -> chi-square 0.7 (5k), 0.0 (14k): no delay fits the board. Delay +2:
5k x1.202 / x1.168 (1.5), 14k x1.280 / x1.218 (3.4); +4: x1.249 / x1.324 (5.8); -2: x1.102 / x1.231 (0.6); -4: x1.258 / x1.156 (3.8).
Fold 1 had 6.5 / 3.5 at delay 0 because of one low cell (aggregated 5k x1.049); fold 3 gives x1.185 for the same cell.
Two folds together: no evidence for a sealed delay, none against a small one (±2). The "3 to 18 times more likely" of the
morning is withdrawn. sub19's expected gain rests only on the prior that the sealed recordings could be offset.
(Three failed launches of this script before it ran: GPU memory on a shared card, then two fold-3 name lookups.)

## S08 a supervised latency reader (`ns08_latency_reader.py 1|2|3`, 14:13-14:28)
A 3-layer 1-D conv net (flatten read-out, 17 classes) trained 12 epochs on single trials of the fold's 15 Alljoined training
people (8,000 each), each delayed by a known d in -8..+8 and seen through the central 104 samples (no edge artefact); single
trial within ±1 sample: 59-60 %. Batch estimate = vertex of the mean log-probability over the batch. Held-out people, delay 0:
| estimator | mixed 64-batches, mean (sd), folds 1 / 2 / 3 | one-person batches, mean per person, fold 2 | sd across people, folds 1 / 2 / 3 | per-batch sd within a person |
|---|---|---|---|---|
| member norm (packaged in sub19) | -1.12 (.50) / -0.07 (.67) / -1.40 (.61) | -5.0 +0.7 -0.3 +4.3 -3.2 | 0.64 / 3.21 / 0.49 | 0.47 / 1.14 / 0.70 |
| trained reader | -0.23 (.34) / -0.27 (.40) / +0.20 (.31) | +1.2 -1.1 -0.4 +0.1 +0.5 | 0.50 / 0.77 / 0.39 | 0.29 / 0.34 / 0.29 |
- It tracks an imposed delay: -4 -> -4.25 / -4.22 / -3.89; +4 -> +3.76 / +3.72 / +4.13; ±2 within 0.3.
- Decoding with the correction -round(estimate - 0.25), r=3, 5k, against no alignment (top-5; ranking error):
  delay 0: mixed +1.2 ± 2.3 / -1.7 ± 3.5 / +1.0 ± 1.4 % (+0.2 / +1.1 / +0.1 %), one person +1.2 / +3.4 ± 4.1 / +2.9 %
  (+0.2 / +0.5 / +0.2 %) -> the fold-2 one-person failure of the norm (-12.2 %, +7.4 %) is gone;
  delay -4: mixed +52 / +110 / +85 %, one person +52 / +114 / +86 % (norm: +55 / +101 / +84 and +50 / +86 / ~);
  delay +4 (target .75 run): +22 / +51 / +33 %. Target 0.75 costs top-100 -2 % at zero delay (it always shifts by +1); 0.25 or 0 does not.
- BUT on the raw external cohorts it is WRONG: ds004018 -3.4 ± 0.3 per batch (true offset about +5), ds007964 -1.8 ± 0.5
  (true +4 to +5), same for the three fold readers, 14 of 16 and 16 of 19 people negative. The member norm recovers both
  (+5.3, +5.6). The reader learnt timing cues of the Alljoined protocol (very likely the RSVP rhythm: SOA 215 ms here,
  200 and 300 ms there), not the image-locked response the decoders use. It is only valid for a cohort recorded with our
  protocol whose whole stream is delayed (a trigger or hardware delay), which is the sealed scenario but cannot be checked.
- Consensus rule (same script): act with the reader's value only when the member-norm estimate (+0.9) agrees with it
  within 2.5 samples, otherwise leave the batch. Mixed batches: 100 % agree on our folds -> same as the reader. One-person
  batches of fold 2: 66-72 % agree; delay 0: +0.3 ± 3.6 % (ranking error +0.4 %); delay -4: +100.7 %; delay -2: +22.8 %.
  On the external cohorts the two disagree by ~8 samples, so the rule leaves them alone (safe, no correction).
- Design for a next alignment package, if the board shows that sub19's alignment pays: consensus(reader, member norm),
  target 0.25, final reader trained on the 20 Alljoined people. Not built yet.
- Code ready: optional `gate` in `neurips_challenge/submission.py` align() and `package_submission.py --gate` (default 0 =
  sub19 behaviour). Report doc updated at 15:05 (rev 22, section "Update of 13 November, afternoon").

## Package candidate: sub20 = sub19 + reader consensus (14:35-15:00), NOT for upload before the board result of sub19
`neurips_challenge/submissions/sub20_ens6_lag5jit_labram_consensus.zip` (413 MB, sha256 c62864cf...b39b): the files of sub19 +
`reader.pt` (final reader, `ns08_latency_reader.py 0`, 20 Alljoined people, 0.5 MB) + members.pt keys reader / target 0.25 /
agree 2.5 / offset 0.9 + the new `submission.py` (align(): the batch is corrected with the reader's delay only when the
member-norm estimate agrees within 2.5 samples; otherwise untouched). `check_package.sh` passes (90 s).
Fold-1 replay of the real packages (`--verify 1 --delay d`, batch 64 / 37; finals on fold people, an easy metric):
| delay | sub16 | sub19 | sub20 |
|---|---|---|---|
| 0 | 33.33 / 33.30 | 33.49 / 33.40 | 33.46 / 33.35 |
| +2 | 33.30 / 33.25 | 33.40 / 33.29 | 33.36 / 33.37 |
| -2 | 33.12 / 33.10 | 33.47 / 33.30 | 33.30 / 33.23 |
| +4 | 31.23 / 31.16 | 33.02 / 32.90 | 33.18 / 33.03 |
| -4 | 29.81 / 29.63 | 33.16 / 33.09 | 33.10 / 33.02 |
External cohorts with the real packages (`nr08_package_on_cohorts.py`, 3 trials averaged): ds004018 sub16 10.62 %, sub19
12.19 %, sub20 10.62 %; ds007964 1.17 / 1.87 / 1.17 %: on another protocol the two estimates disagree and sub20 leaves the
data alone (it gives up sub19's gain there, by design).
Trade-off sub19 vs sub20: sub20 is safer if sealed batches are one person each and more exact under a pure delay of our
protocol; sub19 is the only one that corrects a cohort whose protocol differs from ours.

### S08 CORRECTION (15:35): the reader is NOT wrong on the external cohorts; I had the sign convention backwards
The reader estimates a DELAY (content later by d). The raw exports of ds004018 / ds007964 need a correction of +5 / +4
samples (content moved later), i.e. their delay is about -5 / -4. The reader reads -3.4 and -1.8: the right sign, short by
about 1.5 and 2.5 samples. Checked with the real package (`nr08_package_on_cohorts.py`, 3 trials averaged):
| package | ds004018 | ds007964 |
|---|---|---|
| sub16 (no alignment) | 10.62 % | 1.17 % |
| sub19 (member norm) | 12.19 % | 1.87 % |
| sub20 as built (consensus, agree 2.5) | 10.62 % | 1.17 % |
| consensus with agree 3.5 | 10.75 % | 1.21 % |
| reader alone (agree 99) | 12.44 % | 1.55 % |
So the reader generalises to other protocols about as well as the member norm (better on ds004018, worse on ds007964).
sub20 left both cohorts untouched because its two estimates differ there by 2.8 and 4.7 samples (the +0.9 offset between
them was calibrated on our people only), not because the reader fails. The statements "wrong on the two external cohorts"
and "it reads cues of our protocol" in S08 above, in the memory note and in the report doc (rev 22) are wrong.
Status: the user is not comfortable submitting sub20 (15:25); the alignment line is parked until the board result of sub19.

# Round 10 (13 Nov, 15:30): avenues not explored yet (user: "If test time adaptation is closed, try other avenues that have not been explored yet", 3 GPUs)
Checked against this log, NIGHT_LOG.md and HISTORY.md before choosing. Already closed and not repeated: loss variants
(SigLIP, MSE mix, full-bank negatives, temperature 0.1, concept masking), KD, DRO, batch size, other targets + bridge,
target augmentation, image mixup, repeat averaging, external cohorts, extra members, epoch budgets, weight soups, output
side, test-time adaptation. Raw-space generation of trials for unrecorded images: closed by argument in O03 (a linear
image -> EEG model holds ~5 image dimensions against ~60 for the decoders), not re-opened.

## U01 auxiliary image targets (never tried in this challenge; `train.py --aux_targets <dirs> --aux_lambda`)
Idea: the only supervision so far is the grader's target (DINOv2-g layer 26, token mean). Other feature spaces of the same
16,540 training images are on disk and have a different similarity structure (correlation of pairwise similarities with the
grader target: DINOv2-g L8 .50, L13 .63, InternViT-6B L28 .64, CLIP ViT-H final .48, EVA02 L35 .78). Each is predicted
from the input of `model.head` (rmix: the 256-d token mean) by one linear head with the same contrastive loss, training
only; the output head and the package are unchanged. Hoped effect: features of the image that the EEG carries and the
grader target under-weights (low-level layout, other semantics) are kept in the trunk and help separate images.
rmix, fold 1, lag 5, same recipe as `mi_rmix_ho5k_lag5` (reference at epochs 8 / 10 / 12), launched 15:45-15:52:
`mi_rmix_ho5k_lag5_auxlow` (L8 + L13, lambda .5 each, 516), `_auxsem` (InternViT L28 + CLIP ViT-H, .5 each, 513),
`_auxall` (all four, .25 each, 515). Node disks are full (other users): the three runs read the NAS.
- 16:02 queued behind the three runs (chains `logs/mi/queue_u01_*.log`, `wait_gpu.sh`): `mi_rmix_ho5k_lag5_auxpix` f1 (16x16 RGB
  thumbnails, z-scored, `image_feature/pixels16_z`, similarity structure almost independent of the grader target: r .10; 516),
  `mi_rmix_ho5k_lag5_auxcon` f1 (`--aux_concept_lambda 0.3`: cross-entropy over the 1,654 concepts from the same 256-d
  feature; 513), `mi_rmix_ho5k_lag5_auxall` f3 (replicate; 515). Scoring: `nu01_member_compare.py <fold> <ref> <cand> ...`
  (alone and in the sub16 six, top-5 / top-100 / ranking error).
- Disk note (user question 15:55): node disks are 98 GB, 57 GB system; ours 0.1-0.2 GB (516, 513) and 3.5 GB on 515 (the
  other session's scratchpad of 4-5 Nov, left alone). Staging needs ~32 GB free: the runs read the NAS through the page cache (17-21 it/s once warm).

## U02 is the sealed set harder because every sequence neighbour is a new image too? No (inline, `o02_seq_f1.npy`, 16:07)
In the sealed stream the images shown just before and after a trial are new images as well; in the proxy 70 % of them are
trained images, whose responses the members could explain away better. Fold 1, lag five, single trials of new people on
never-trained images (100,000), by the status of the neighbours:
| neighbours | n | top-5 % | top-100 % | ranking error |
|---|---|---|---|---|
| previous and next trained | 43,947 | 0.846 ± 0.044 | 9.08 ± 0.14 | 0.3278 ± 0.0013 |
| exactly one never-trained | 37,854 | 0.740 ± 0.044 | 8.78 ± 0.15 | 0.3280 ± 0.0014 |
| previous and next never-trained | 8,188 | 0.782 ± 0.097 | 9.45 ± 0.32 | 0.3318 ± 0.0031 |
| 0 / 1 / 2 / 3 of the four nearest never-trained | 18,983 / 33,029 / 21,090 / 6,084 | 0.901 / 0.784 / 0.740 / 0.855 | 9.25 / 8.92 / 8.99 / 8.94 | 0.3279 / 0.3277 / 0.3275 / 0.3243 |
No trend at top-100 or in the ranking error; the top-5 differences are within two standard errors and not monotone.
The novelty of the neighbours does not explain the gap between the proxy and the board. Closed.
- 17:11 U01 first read, fold 1, against `mi_rmix_ho5k_lag5` at the same epoch (`nu01_member_compare.py`; alone | in the sub16 six; top-5 / top-100 / ranking error, %):
  | arm | epoch 8 alone | epoch 8 in the six | epoch 10 alone | epoch 10 in the six |
  |---|---|---|---|---|
  | auxlow (L8 + L13) | -3.1 ± 4.4 / +0.5 ± 1.2 / -0.5 ± 0.3 | +1.5 ± 2.2 / +0.2 / 0.0 | +8.3 ± 5.2 / +2.1 ± 1.2 / -0.6 ± 0.3 | +2.0 ± 2.2 / +1.4 ± 0.5 / -0.1 |
  | auxsem (InternViT + CLIP) | -3.6 / -0.1 / -0.5 | -1.0 / 0.0 / -0.1 | +2.1 / -0.6 / +0.2 | -0.8 / +0.4 / 0.0 |
  | auxall (all four) | -1.7 / +0.6 / -0.3 | +1.3 / 0.0 / +0.1 | +6.1 / +2.0 / -0.4 | -0.5 / +0.2 / 0.0 |
  Inside the seed noise of one run (Q11: top-5 5-9 %, top-100 and ranking error up to 0.8 %). Neutral so far; the low-level
  targets are the only arm with a consistent sign. The queued fold-3 replicate of auxall was dropped (17:13) to free 515.
- 17:16 user: "When an experiment is neutral on fold 1, wait for fold 3 results before giving up." The fold-3 replicates are
  back in the queue: 515 chain = `nu03_image_bias.py` (short), then `mi_rmix_ho5k_lag5_auxlow` f3, then `_auxall` f3
  (`scratchpad/chain515.sh`, log `logs/mi/queue_u01_515.log`). 516: auxpix f1; 513: auxcon f1 (as queued at 16:02).

## U03 a correction of the output fitted on never-trained IMAGES (`nu03_image_bias.py`, queued 17:16 on 515)
M04 found that ranking new people's queries against the training people's mean output for an image, instead of its
embedding, gives +16-18 %: the members share an image-specific error on new images. The fixed gallery needs the inverse,
a map g on the output. M04 / M05 fitted linear maps on new PEOPLE; here g (linear, rank-64 or MLP residual) is fitted on
2,500 never-trained images with the 15 training people's outputs and tested on the held-out people and the other 2,500.
- 17:38 U01 fold 1, final epoch (12), against `mi_rmix_ho5k_lag5` final (alone 1.298 % top-5, 13.37 % top-100, ranking error .2622):
  | arm | alone: top-5 / top-100 / ranking error | in the six |
  |---|---|---|
  | auxlow (L8 + L13) | +3.1 ± 5.6 / +2.7 ± 1.3 / -0.4 ± 0.4 % | +0.3 ± 2.3 / +1.1 ± 0.6 / -0.1 ± 0.1 % |
  | auxsem (InternViT + CLIP) | +3.1 ± 5.7 / +1.3 ± 1.3 / -0.2 ± 0.4 % | -1.3 ± 2.6 / +1.3 ± 0.6 / -0.1 % |
  | auxall | +2.2 ± 5.4 / +3.4 ± 1.4 / -0.1 ± 0.3 % | -0.8 ± 2.5 / +0.8 ± 0.6 / 0.0 % |
  Over the three epochs every arm is >= 0 at top-100 from epoch 10 on (alone +1.3 to +3.4 %, in the six +0.2 to +1.4 %), the
  top-5 in the six stays inside ±2 %. A small positive in the bulk of the ranking for ONE member of six; user (17:20): a few
  percent is welcome, gains can be stacked over members. All six member families have `model.head` (REVE-d6 512-d, SQF-X
  320-d, LaBraM), so the option applies to each without new code. Fold 3 decides (auxlow f3 started 17:38 on 515).

### U03 result (17:38): the output correction fitted on never-trained images is strongly negative; closed
New people x 2,500 never-trained images B, r=3, six members: no correction 2.856 % top-5 (gallery 2,500). Corrections fitted on
150,000 trials of the 15 training people on the other 2,500 images: linear residual -22 to -28 %, rank-64 -15 to -21 %, MLP
-21 to -30 % (top-100 -8 to -14 %, ranking error +4 to +11 %), and they hurt the training people on B as well (1.6-1.7 % against
2.26 %). The fit loss stays at 7.4-7.5 (chance 7.8): nothing systematic is learnt. The M04 oracle itself (gallery = training
people's mean output for the B images) is +3.6 ± 6.6 % top-5 and -5.9 ± 1.6 % top-100 for the six-member sum: the shared
image-specific error M04 saw for rmix alone is not there for the ensemble. No output map to deploy.
- 17:42 queue for the evening: 516 auxpix f1 -> `mi_rmix_ho5k_lag5_auxsem` f3; 513 auxcon f1 -> `mi_d6_ho5k_lag5_jit3_auxlow` f1
  (the low-level targets in a second member, against `mi_d6_ho5k_lag5_jit3`); 515 auxlow f3 -> auxall f3. Scoring watcher:
  `logs/mi/nu01_f1_pixcon.out`, `logs/mi/nu01_f3_auxlow.out` (`nu01_member_compare.py` now builds the six on any fold).
- 19:19 U01 scores of the evening (`logs/mi/nu01_f1_pixcon.out`, `logs/mi/nu01_f3_auxlow.out`; alone | in the six; top-5 / top-100 / ranking error, %):
  | arm, fold | epoch 8 | epoch 10 | final (12) |
  |---|---|---|---|
  | auxlow f3, alone | -1.9 ± 4.3 / +1.2 ± 1.2 / +0.2 | -6.6 ± 4.3 / +1.4 / -0.1 | +3.0 ± 5.2 / +2.0 ± 1.4 / -0.7 ± 0.3 |
  | auxlow f3, in the six | +2.7 ± 2.2 / +1.4 ± 0.5 / 0.0 | -4.9 ± 2.1 / +1.3 ± 0.5 / -0.1 | +1.0 ± 2.3 / +1.2 ± 0.6 / -0.1 |
  | auxpix f1, alone | -4.7 / +0.2 / -0.5 | +6.4 ± 5.3 / +1.8 / -0.8 ± 0.3 | +16.0 ± 6.0 / +1.9 ± 1.4 / -0.5 |
  | auxpix f1, in the six | +0.3 / +0.1 / -0.1 | +3.6 ± 2.4 / +1.1 ± 0.5 / -0.2 | +1.0 ± 2.5 / +0.4 ± 0.6 / -0.1 |
  | auxcon f1, alone | -4.7 / -0.2 / 0.0 | +6.1 / -1.5 / -0.4 | +7.4 ± 5.7 / +0.3 / -0.6 |
  | auxcon f1, in the six | +2.6 / +0.4 / 0.0 | -0.5 / +0.6 / -0.1 | +0.5 ± 2.5 / +0.5 ± 0.6 / -0.1 |
  auxlow over two folds, in the six: top-100 +0.2 / +1.4 / +1.1 (f1) and +1.4 / +1.3 / +1.2 (f3): replicated, about +1.2 % from
  one member; top-5 +1.5 / +2.0 / +0.3 and +2.7 / -4.9 / +1.0, mean +0.4 %: not resolved. Pixels behave like the DINO low layers.
- 19:21 night queue (chains `scratchpad/night51{6,3,5}.sh`, logs `logs/mi/queue_night_*.log`): the low-level targets in EVERY
  member on fold 1, to read the stacked effect on the top-5 (`nu04_stack.py 1`): d6 (running, 513) -> SQF f1 -> LaBraM f1;
  516: auxsem f3 (running) -> SQF-X f1 -> REVE-scratch f1; 515: auxall f3 (running) -> auxpix f3 -> auxcon f3 -> d6 auxlow f3.
- 20:50 first stacked read, fold 1 (`nu04_stack.py 1`, `logs/mi/nu04_f1_a.out`; reference six 1.570 % top-5, 15.61 % top-100, .2405):
  | swapped in | top-5 | top-100 | ranking error |
  |---|---|---|---|
  | d6 + low-level targets (alone vs its reference: +1.7 ± 4.8 / 0.0 ± 1.3 / -0.8 ± 0.3) | +1.5 ± 2.6 % | -0.1 ± 0.6 % | -0.2 ± 0.1 % |
  | rmix + low-level targets (epoch 10) | +2.0 ± 2.2 % | +1.4 ± 0.5 % | -0.1 ± 0.1 % |
  | both | +6.9 ± 2.9 % | +1.3 ± 0.7 % | -0.3 ± 0.1 % |
  Two members swapped: 1.570 -> 1.678 % top-5. One fold, and the two single swaps add up to about +3.5 %, so part of the +6.9
  is luck; the ranking error moves the right way for every swap (-0.2, -0.1, -0.3). The remaining four members follow tonight.
- 21:00 fold-3 replicates of the semantic and combined arms (`logs/mi/nu01_f3_semall.out`; in the six, epoch 10 / final):
  auxsem top-5 -2.1 ± 2.1 / +1.9 ± 2.4 %, top-100 +1.2 / +0.9 ± 0.6 %, ranking error -0.2 / -0.3 ± 0.1 %; auxall top-5 -2.3 / -0.2 %,
  top-100 +0.6 / +0.4 %, ranking error -0.1 / -0.2 %. Alone, ranking error -0.8 / -1.2 % (auxsem) and -0.1 / -1.1 % (auxall).
  Same picture as fold 1 for every arm: the bulk of the ranking improves a little (top-100 +0.4 to +1.4 %, ranking error -0.1 to
  -0.3 % in the six), the top-5 of a one-member swap stays inside ±2.5 %. No arm separates from the others; low-level (L8 + L13)
  stays the one carried into the other members.

## U01 night results (14 Nov, 02:30): low-level targets in every member, and the fold-3 replicates
**All six members with the low-level targets (L8 + L13), fold 1** (`nu04_stack.py 1`, `logs/mi/nu04_f1_c.out`; reference six 1.570 % / 15.61 % / .2405):
| member swapped | alone vs its reference: top-5 / top-100 / ranking error | in the six |
|---|---|---|
| REVE-d6 | +1.7 ± 4.8 / 0.0 ± 1.3 / -0.8 ± 0.3 % | +1.5 ± 2.6 / -0.1 ± 0.6 / -0.2 ± 0.1 % |
| rmix (epoch 10) | +8.3 ± 5.2 / +2.1 ± 1.2 / -0.6 ± 0.3 % | +2.0 ± 2.2 / +1.4 ± 0.5 / -0.1 % |
| SQF-X | +0.7 ± 5.7 / -2.7 ± 1.3 / +0.2 ± 0.3 % | +1.3 ± 2.0 / -0.7 ± 0.5 / 0.0 % |
| REVE-scratch | -2.1 ± 5.9 / +0.2 ± 1.6 / +1.5 ± 0.4 % | +1.0 ± 2.7 / -0.6 ± 0.6 / +0.1 % |
| SQF | -12.5 ± 6.0 / +0.5 ± 1.6 / +1.1 ± 0.4 % | +0.5 ± 2.1 / +0.4 ± 0.5 / +0.1 % |
| LaBraM | +4.1 ± 5.7 / +0.1 ± 1.3 / -0.7 ± 0.4 % | +1.0 ± 2.6 / -0.2 ± 0.6 / -0.2 % |
| all six | | **+4.3 ± 3.7 / +1.9 ± 0.9 / -0.1 ± 0.2 %** |
| d6 + rmix + LaBraM (the three it helps alone) | | **+4.8 ± 3.4 / +1.7 ± 0.8 / -0.5 ± 0.1 %** |
| d6 + rmix | | +6.9 ± 2.9 / +1.3 ± 0.7 / -0.3 ± 0.1 % |
| d6 + rmix + SQF-X + SQF | | +2.6 ± 3.1 / +1.2 ± 0.7 / -0.1 ± 0.1 % |
- The targets help the three token-mean transformers alone (ranking error -0.6 to -0.8 %) and not the two conv members nor the
  full-depth REVE-scratch (ranking error +1.1 to +1.5 %). Keep them for d6, rmix, LaBraM only.
- Fold 3 (`nu04_f3_a.out`; reference six 1.706 % / 15.37 % / .2401): d6 + rmix with the targets -0.7 ± 2.5 / +1.0 ± 0.6 / -0.1 ± 0.1 %
  (d6 alone -6.8 ± 4.5 / +1.1 / -0.4; in the six -2.6 ± 2.1 / +0.2 / -0.1). The +6.9 % of fold 1 does not repeat at top-5.
**Pixel targets are the best arm, on both folds** (rmix, `nu01_f1_pixcon.out`, `nu01_f3_pixcon.out`; epoch 10 / final):
| | alone: top-5 | alone: ranking error | in the six: top-5 | in the six: top-100 | in the six: ranking error |
|---|---|---|---|---|---|
| fold 1 | +6.4 ± 5.3 / +16.0 ± 6.0 % | -0.8 / -0.5 ± 0.3 % | +3.6 ± 2.4 / +1.0 ± 2.5 % | +1.1 / +0.4 % | -0.2 / -0.1 ± 0.1 % |
| fold 3 | +5.2 ± 4.8 / +11.9 ± 5.7 % | -1.4 / -1.5 ± 0.3 % | +2.6 ± 2.3 / +6.0 ± 2.4 % | +1.5 / +0.8 % | -0.4 / -0.3 ± 0.1 % |
Four reads out of four positive at top-5, alone (mean +10 %) and in the six (mean +3.3 %); ranking error -0.5 to -1.5 % alone.
The lower the level of the target, the better: pixels > DINOv2 L8 + L13 > semantic spaces = concept head (auxcon f3 in the six:
-2.6 / +3.1 % top-5, +0.6 / 0.0 % top-100: neutral on two folds, closed).
Mixed compositions: fold 1, d6 low + rmix pixels + LaBraM low +3.3 ± 3.4 / +0.7 ± 0.7 / -0.5 ± 0.1 %; fold 3, d6 low + rmix pixels
+1.2 ± 2.7 / +1.2 ± 0.6 / -0.4 ± 0.1 %.
**Reading**: a real, small effect. The ranking error of the six improves by 0.4-0.5 % on both folds (4-5 standard errors), top-100 by
1-2 %, the top-5 point estimates are +1 to +5 % with errors of 2.5-3.5 %. Not +10 %, but the first training-side change since the
lag that moves the ensemble on two folds.
- 02:35 next batch (pixels into the members that benefit; chains `scratchpad/b51*.sh`, `c51*.sh`): 516 `mi_d6_ho5k_lag5_jit3_auxpix`
  f1 -> f3 -> `mi_rmix_ho5k_lag5_auxpix32` f1 -> f3 (32x32 thumbnails, `image_feature/pixels32_z`); 513 `mi_rmix_ho5k_lag5_auxlp`
  f1 -> f3 (pixels + L8 + L13) -> `mi_rmix_ho5k_lag5_auxpixl1` f1 -> f3 (pixels, lambda 1); 515 `mi_plab_ho5k_lag5_long_auxpix` f1 -> f3.

## U01 pixel targets in the members that benefit (14 Nov, 06:08; `logs/mi/nu_b1.out`, `nu_b2.out`, `nu_b3.out`)
| composition (twins with 16x16 pixel targets, lambda .5) | fold | top-5 | top-100 | ranking error |
|---|---|---|---|---|
| d6 + rmix + LaBraM | 1 | **+6.1 ± 3.2 %** (1.570 -> 1.666) | +1.2 ± 0.7 % | **-0.5 ± 0.1 %** |
| d6 + rmix | 1 | +5.4 ± 2.9 % | +0.8 ± 0.7 % | -0.4 ± 0.1 % |
| d6 + rmix | 3 | +2.8 ± 2.6 % (1.706 -> 1.754) | +1.0 ± 0.6 % | -0.5 ± 0.1 % |
| d6 only (alone vs its reference: f1 -1.1 ± 5.1 / +2.3 / -1.1 ± 0.3; f3 -1.0 / +0.8 / -0.6) | 1 / 3 | +3.3 ± 2.4 / 0.0 ± 2.2 % | +0.4 / +0.7 % | -0.2 / -0.2 % |
| LaBraM only (alone: -1.5 ± 5.4 / -0.4 / -0.8 ± 0.4) | 1 | -0.8 ± 2.7 % | -0.1 % | -0.1 % |
Two folds pooled for d6 + rmix: top-5 about +4.1 ± 1.9 %, ranking error -0.45 ± 0.07 %. Every member's ranking error improves alone
(-0.6 to -1.5 %). rmix with pixels + L8 + L13 (`auxlp`): alone ranking error -0.9 / -1.0 (f1), -1.1 / -1.8 (f3), top-100 +4.3 %
final on both folds; in the six top-5 +1.8 / +2.0 (f1), -4.0 / +0.5 (f3): not better than pixels alone in the six. Pixels alone kept.
- 06:09 queued on 515 after LaBraM pixels f3: `mi_rmix_ho5k_lag5_jit3_auxpix` f1 -> f3 (the packaged rmix recipe has the ±3 jitter;
  the pixel runs so far were without it), to test the exact composition a package would use:
  d6 jitter + pixels, rmix jitter + pixels, SQF-X jitter, REVE-scratch, SQF, LaBraM + pixels against the sub16 six.
- 06:10 finals queued behind the variant runs (used only if the exact-composition test on folds 1 and 3 is positive; chains
  `scratchpad/f513.sh`, `f516.sh`): 513 after auxpixl1 f3 -> `final/reve_d6_optx_things_lag5jitpix` (budget 12, ~3 h); 516 after auxpix32 f3
  -> `final/rmix_reg_optx_things_lag5jitpix` (budget 10, ~2 h) -> `final/labram_optx_things_lag5jitpix` (budget 10, ~4 h). Same recipes as
  the sub16 finals (`*_lag5jit`) + `--aux_targets image_feature/pixels16_z --aux_lambda 0.5`.
- 08:00 three pixel members on fold 3 (`logs/mi/nu_c2.out`): d6 + rmix + LaBraM +1.2 ± 2.8 / +2.3 ± 0.7 / -0.4 ± 0.1 % (top-5 / top-100 /
  ranking error; fold 1: +6.1 ± 3.2 / +1.2 / -0.5). LaBraM with pixels is neutral in the six on both folds (f1 -0.8 / -0.1 / -0.1, f3
  +0.7 / +0.6 / +0.1; alone ranking error -0.8 then +0.4): d6 + rmix carry the gain (f1 +5.4 ± 2.9, f3 +2.8 ± 2.6; ranking error -0.4 /
  -0.5). The LaBraM final was taken out of the queue (chain f516 stopped before it started); sub16's LaBraM stays.
- Pixel variants on fold 1 (`nu_c1.out`; rmix, epoch 10 / final; in the six): 32x32 thumbnails top-5 +3.6 / +3.0 ± 2.5 %, top-100
  +1.3 / +2.1 ± 0.6 %, ranking error -0.4 / -0.3 % (alone -1.7 / -1.7 ± 0.3 %); lambda 1: +2.8 / +0.8, +1.7 / +1.3, -0.3 / -0.2 (alone
  -1.1 / -0.9); reference arm 16x16 lambda .5: +3.6 / +1.0, +1.1 / +0.4, -0.2 / -0.1 (alone -0.8 / -0.5). 32x32 is at least as good;
  fold 3 of both variants is running. The rmix final waits for it (target = 16x16 or 32x32); the d6 final keeps 16x16 (tested on 2 folds).
- 09:07 pixel variants on fold 3 (`logs/mi/nu_d1.out`; rmix in the six, epoch 10 / final): 16x16 lambda .5 top-5 +2.6 / +6.0 ± 2.4 %, top-100
  +1.5 / +0.8 %, ranking error -0.4 / -0.3 %; 32x32 -3.1 ± 1.9 / +0.2 %, +1.4 / +0.8 %, -0.2 / -0.3 %; lambda 1 -3.5 / +2.4 %, +0.4 / +0.6 %,
  -0.2 / -0.4 %. Neither variant beats 16x16 lambda .5 over the two folds; closed after fold 3. Robust part: the ranking error of the six
  improves in all 12 reads (3 pixel recipes x 2 folds x 2 checkpoints), by 0.1 to 0.4 %.
- 09:08 `final/rmix_reg_optx_things_lag5jitpix` started on 516 (16x16, lambda .5); `final/reve_d6_optx_things_lag5jitpix` running on 513
  since 08:45. Then a THIRD fold for the top-5 estimate (fold 2, where the jitter references do not exist yet): 516 rmix jitter f2 ->
  rmix jitter + pixels f2; 515 (after rmix jitter + pixels f3) d6 jitter f2 -> d6 jitter + pixels f2.

## U01 the exact package composition (14 Nov, 11:00; `RMIX=_jit3 TWINS=0:_auxpix,1:_auxpix nu04_stack.py`, `logs/mi/nu_exact_f{1,3}.out`)
Reference = the true sub16 six of the fold (rmix WITH jitter); twins = d6 jitter + pixels, rmix jitter + pixels.
| fold | top-5 | top-100 | ranking error | members alone (ranking error) |
|---|---|---|---|---|
| 1 (1.622 -> 1.690 %) | +4.2 ± 2.8 % | +1.3 ± 0.6 % | -0.5 ± 0.1 % | d6 -1.1, rmix -1.1 ± 0.3 % |
| 3 (1.746 -> 1.662 %) | **-4.8 ± 2.5 %** | +0.7 ± 0.6 % | -0.3 ± 0.1 % | d6 -0.6, rmix -1.0 ± 0.3 % |
Pooled top-5: about -0.3 ± 1.9 %. All top-5 reads of two- or three-member pixel swaps so far: fold 1 +5.4, +6.1, +4.2; fold 3 +2.8, +1.2,
-4.8. The ranking error and the top-100 improve in every read (-0.3 to -0.5 %, +0.7 to +2.3 %); the top-5 does not follow reliably:
fold 1 says about +5 %, fold 3 about 0. Honest size of the lever for the board metric: +2 ± 2 %.
The two finals keep running (d6 at epoch 8 of 12, rmix at 9 of 10) and the fold-2 pairs follow (third fold for the top-5).
- 11:02 adding the two pixel twins as extra members (eight) instead of swapping: fold 1 +0.7 ± 2.4 / +0.7 / -0.4 %, fold 3 -4.1 ± 2.3 / +0.7 /
  -0.2 % (swap: +4.2 / -4.8). Same picture either way; on fold 3 the jitter + pixel rmix is the unlucky run (the no-jitter pixel rmix read
  +2.6 / +6.0 % in the six there). A single member swap moves the top-5 of the six by ±2-3 % from run to run.
- 11:05 the exact pixel composition at the extreme top, two folds pooled (cached outputs; the board's top-5 of ~40k is the quantile of a
  top-0.6 in the 5k gallery): 5k r=3 top-1 -0.5 ± 4.6 %, top-5 -0.5 ± 1.8 %, top-20 +1.4 ± 1.0 %, top-100 +1.0 ± 0.4 %; 5k r=4 +1.6 ± 3.5 /
  +0.2 ± 1.6 / +1.2 / +0.5 %; 14k r=3 +2.6 ± 7.5 / +1.5 ± 3.0 / +1.7 / +0.5 %; 14k r=4 +3.6 ± 5.8 / +1.0 ± 2.5 / -0.2 / +1.7 ± 0.6 %.
  **U01 verdict**: pixel targets improve the bulk of the ranking by about 1 % (top-100, ranking error; every read, both folds) and leave
  the top of the ranking unchanged within ±2 %. Harmless to include, not a reason to spend a submission. Fold 2 will be a third fold.

## U04 cross-person latent prediction without its sampler cost (queued 11:10 on 513, after the d6 final)
I09 (12 Nov) closed the JEPA-style loss "for cost": with 2 trials per image in a 128-trial batch the pairing halves the distinct images
(-9 to -14 % alone), and the cross-person loss won +7 to +10 % of it back (about 2 sigma) without passing the standard recipe. The version
that keeps 128 distinct images was never run: `mi_rmix_ho5k_lag5_xp256` f1 (batch 256 = 128 images x 2 people, multi-positive, `jepa=3`,
`--xperson_lambda 1`), then its control `mi_rmix_ho5k_lag5_pair256` f1 (same sampler, no cross-person loss). Reference: `mi_rmix_ho5k_lag5`.
Like U01 this is an auxiliary loss on the trunk; its target is another person's representation of the same image instead of image features.

## Package candidate: sub21 = sub16 with d6 and rmix retrained with pixel targets (14 Nov, 12:25); NOT recommended for upload on its own
`neurips_challenge/submissions/sub21_ens6_lag5jit_labram_pix.zip` (392 MB, sha256 6dbfdfad...2510580f): finals
`reve_d6_optx_things_lag5jitpix` (budget 12, 08:45-12:10) and `rmix_reg_optx_things_lag5jitpix` (budget 10, 09:07-11:25) = the sub16 recipes
+ `--aux_targets image_feature/pixels16_z --aux_lambda 0.5`; SQF-X, REVE-scratch, SQF, LaBraM = the sub16 files. No alignment step.
`check_package.sh` passes (80 s). Fold-1 replay (in-sample, functional check only): 33.27 / 33.11 at delay 0 (sub16 33.33 / 33.30), 31.19 / 31.18
at delay +4 (sub16 31.23 / 31.16). Offline expectation from folds 1 + 3: top of the ranking 0 ± 2 %, bulk +1 %.
- 12:20 ops: job 74191 (513) ended at its 2-day limit (12:19:49) and took the xp256 run with it (15 min in). Replacement `sbatch mi_gpu3` =
  job 74916 (513, same options: Brain_GPU, rtx3080, 8 CPU, 30G, 2 days, `--wrap "sleep 2d"`), xp256 -> pair256 relaunched on it 12:28
  (`scratchpad/i513.sh`). 74489 (515) ends at about 19:45 today; 73733 (518) ended by itself 13 Nov 21:39. Three GPUs in use: 516, 515, 513.

## U01 third fold and three-fold verdict (14 Nov, 14:40; `logs/mi/nu_exact_f2.out`, pooled inline)
Fold 2, exact composition (d6 jitter + pixels, rmix jitter + pixels; reference six with SQF-X without jitter, the only one trained there):
top-5 +8.1 ± 3.5 % (1.194 -> 1.290), top-100 +0.7 ± 0.8 %, ranking error 0.0 ± 0.1 %.
| three folds pooled | top-1 | top-5 | top-20 | top-100 | ranking error | top-5 per fold (1 / 2 / 3) |
|---|---|---|---|---|---|---|
| 5k, r=3 | -1.5 ± 3.9 % | +1.8 ± 1.6 % | +1.5 ± 0.8 % | +0.9 ± 0.4 % | -0.24 ± 0.06 % | +4.2 / +8.1 / -4.8 |
| 5k, r=4 | +1.1 ± 3.1 % | -0.1 ± 1.4 % | +1.9 ± 0.7 % | +0.4 ± 0.4 % | -0.35 ± 0.06 % | -1.4 / -0.8 / +1.7 |
| 14k, r=3 | +7.8 ± 6.8 % | -1.9 ± 2.5 % | +2.7 ± 1.3 % | +0.4 ± 0.6 % | -0.21 ± 0.06 % | 0.0 / -9.8 / +2.9 |
| 14k, r=4 | +0.7 ± 4.9 % | +1.6 ± 2.2 % | +0.1 ± 1.1 % | +1.7 ± 0.6 % | -0.32 ± 0.06 % | +1.0 / +3.0 / +0.9 |
**Final reading of U01**: ranking error -0.2 to -0.35 % (4-6 standard errors), top-20 about +1.5 %, top-100 about +1 %; top-5 and top-1 are
0 to +2 % and not distinguishable from zero. The fold-to-fold spread of the top-5 (-4.8 to +8.1 at 5k r=3, with opposite signs at r=4 or
14k) is seed noise of the swapped members, larger than the bootstrap error. sub21 is expected to equal sub16 within ±2 % on the board.
A fourth fold (all six references + the two twins on fold 4) launched 14:50: one more fold for this and any later comparison.

## U04 result (14:38): cross-person prediction at batch 256, fold 1: negative, closed
`mi_rmix_ho5k_lag5_xp256` alone: top-5 -11.3 / -10.5 ± 5.5 %, top-100 -13.0 / -15.8 %, ranking error +10.0 / +12.2 % (epoch 10 / final); its
control `pair256` -17.8 / -24.4 %, -15.8 / -19.9 %, +11.5 / +13.8 %; in the six both are within noise (-1.3 to +1.3 % top-5, ranking error
+0.7 to +0.8 %). The cross-person loss again wins back a part of what the paired sampler costs (+7 to +14 % over its control) and the pair
of them stays far below the standard recipe: with 2 trials per image the member is worse whatever the batch size (half the optimiser steps
at 256). Clearly negative alone on fold 1: closed without a fold 3.
- 15:00 `neurips_challenge/submissions/sub22_ens6_lag5jit_labram_pix_align.zip` (sha256 53df8907...2d603d96) = sub21 + the alignment step of sub19
  (`detector_gen.pt`, shifts -8..+8 step 2, bias 1.5; no reader, no gate). `check_package.sh` passes (56 s). Fold-1 replay 33.31 / 33.33 at
  delay 0, 32.35 at +4 (sub21 31.19, sub19 33.02). User asked for the recommended submission (14:55): answer depends on sub19's board score
  (sub19 clearly above 0.001258 -> sub22; otherwise sub21), expected +0 to 2 % either way; hold the last slot until close to the deadline.

# Round 11 (14 Nov, 15:10): synthesis, literature review and plan to the deadline (user: 2 uploads kept, 6 days, "detailed research proposal plan")
Page: https://claude.ai/artifact/3ePKWoR7yrtp2CXWbbVgiK (source in the session scratchpad, `track1-research-plan.html`). Four web searches
(EEG cross-subject, fMRI / MEG cross-subject, domain shift in noisy contrastive retrieval, competition practice), each filtered against this
log, NIGHT_LOG.md and HISTORY.md. Facts confirmed on the rules page: deadline Sat 21 Nov 23:59 CET; 5 sealed uploads, 1 per day; top 3 are
re-run from the committed config on 22-30 Nov and must land within 2 sigma; the final submission ships a method description declaring
external corpora and a compute estimate. Board 14 Nov 13:14 UTC: us 0.001258 first, 2nd 0.001245, 10th 0.001121.
Plan items (ids of the page): F1 fold 4 (running), F2 composition on 4 folds, F3 audit bundle (commit waits for the user);
T1 layout-preserving auxiliary targets (DINOv2-g L26 tokens on a 2x2 / 3x3 grid, Gabor energy), T2 soft labels (0.7 one-hot + 0.3 softmax of
image-image cosine; October's -4.1 was on trained images at 80 repeats), T3 RINCE, T4 last two epochs on Alljoined only, T5 per-person
low-rank residual on the last block with subject dropout; M1 LUNA member, M2 LaBraM two-run weight average; A1 ridge from image descriptions
to pre-head activations (ranks T1 targets), A2 what the pixel head changed, A3 retrain spread. Closed by the literature + our logs and not
re-opened: mean-of-repeats loss (pair sampler cost), blurred main targets, cross-person positives, bridges, soups, synthetic responses, more
cohorts. Upload logic proposed: first upload Mon/Tue = best package WITH alignment (board test of a sealed delay), second Fri 20 = final stack.
- 15:30 code for round 11 (uncommitted): `module/loss.py` ContrastiveLoss gains `soft_label` / `soft_tau` (target = (1 - a) one-hot +
  a softmax(image-image cosine / tau) over the OTHER batch images; with the own image inside the softmax it kept all the mass) and `gce_q`
  (generalised cross-entropy (1 - p^q) / q, the bounded form of the robust loss: RINCE as published is unbounded below with our
  unnormalised EEG outputs); `train.py --soft_label --soft_tau --gce_q --alljoined_after E` (THINGS-EEG2 trials dropped from every batch
  after epoch E). Self-check inline: a -> 0 and q -> 0 reproduce the plain loss.
- `scripts/mi/nu05_grid_tokens.py`: DINOv2-g hidden state 26 patch tokens on a 2x2 / 3x3 grid -> `image_feature/dinov2-giant_track1_grid{2,3}_z`
  (running on 515; its all-token mean matches the grader targets at cosine >= 0.997).
- Night list (scratchpad `night.list`, one detached worker per GPU: `w515.sh`, `w513.sh`, `w516.sh`, log `night.log`), rmix jitter recipe, folds
  1 and 3 each: soft3 (a 0.3, tau 0.05), auxgrid2, gce4, auxpixgrid2, aj8, soft3t02 (tau 0.02), gce2, auxgrid3. Workers start when the
  fold-4 chains end (515 ~17:30, 513 / 516 ~20:40). Slurm: 74991 `mi_gpu4` submitted with `-d afterany:74489` (starts when 74489 reaches
  its limit at ~19:40; three GPUs at every moment).
- 15:35 USER: no audit work for now ("just try to improve accuracy in a reliable way"); the alignment is not uploaded alone, it waits for
  other improvements to stack with, and a sealed delay is read from board gain minus local forecast. F3 deferred; page updated.
- 15:45 USER: rely on all four folds before a final decision; check that the focus on fold 1 did not make us overlook past results.
  Actions: (1) a read-only audit of this log and NIGHT_LOG (716-1015) for verdicts resting on one fold inside the noise (agent, report
  pending); (2) the four-fold reference of the six is completed: SQF-X jitter fold 2 was missing (only the lag-only one existed there),
  added to the night list in second position; fold 4 of all six is running; (3) folds 1 + 3 are from now on a screen only.
- 15:38 grid features saved: `dinov2-giant_track1_grid2_z` (6,144-d), `grid3_z` (13,824-d); all-token mean vs grader targets: cosine mean 0.9994, min 0.9825.

## Audit of single-fold verdicts (14 Nov, 15:50; user: "make sure our obsession with fold 1 has not led us to overlook past results")
Read-only agent audit of this log and NIGHT_LOG 716-1015 against the noise yardstick (alone +-10 %, in the ensemble +-5 % on one fold).
About 26 levers solid (>= 2 folds agreeing or far outside the noise). Undetermined or thin, by how much a re-read could change a decision:
1. Time jitter at zero shift (G01/G04/G06): in every package since sub14 on fold-1 reads only (d6 0.0 / rmix +2.3 / SQF-X -4.5 +- 2.1 %).
2. Cross-image mixup (NIGHT_LOG 953-959): fold 3 only, before the rank fix, positive at 5k (d6 +4.4 +- 1.6 % at its last epoch), closed "flat".
3. Temperature 0.1 / full-bank loss (NIGHT_LOG 723): one member (SQF-X), fold 1, generalised to all families.
4. Test repeats in the finals (F01/F01b): opposite-sign single-fold reads averaged to "neutral".
5. ds007964 x1 (Q09/Q11): rmix alone, 2 folds, positive on the stable metrics (top-100 +1.6 +- 0.8, 1 - AUC -1.0 +- 0.2), closed for size; dose non-monotone.
6. Auxiliary targets dropped for SQF-X / scratch / SQF on fold 1 with the L8 + L13 arm; pixels never run there.
7. REVE-scratch budget 12 vs 10: alone and ensemble reads disagree. Lower: things_noise, weight decay 0.3, functional operator transfer,
   adaptive filter (d6 never reported), things_filter (d6 f3 never reported), CPA random view, LaBraM lr / layer decay, `auxlp`.
Actions: zero-training re-reads on cached runs, stable metrics, in the six (`nu04_stack.py` gained REF= / SWAP= slots; log
`logs/mi/audit_r11.out`): A jitter vs lag-only on folds 1, 2, 3; B tr10 on folds 1, 3; C mixup fold 3; D scratch @10 vs @12 on 3 folds.
New runs appended to the night list: temperature 0.1 (rmix f1, f3), pixels in REVE-scratch and SQF-X (f1, f3).
- 16:00 USER: a 4th GPU is allowed. Job 75032 (mi_gpu5) submitted; worker w4.sh takes the night list on it.

### Audit re-reads on cached runs (16:15, `logs/mi/audit_r11.out`; sub16 six of the fold, 5k r=3; top-5 | top-100 | ranking error, %)
- **A. Time jitter is a GAIN at zero shift, on three folds** (it was logged as "neutral, insurance only" from fold-1 top-5 reads).
  Jitter members replaced by their lag-only twins (d6 + rmix + SQF-X on folds 1 and 3; d6 + rmix on fold 2):
  fold 1 -5.4 ± 2.9 | -1.3 ± 0.8 | +0.8 ± 0.2; fold 3 -8.7 ± 2.8 | -1.1 ± 0.7 | +0.8 ± 0.1; fold 2 -1.0 ± 3.8 | -0.7 ± 0.9 | +0.6 ± 0.1.
  Per member alone (lag-only vs jitter, ranking error): d6 +3.6 / +2.8 / +3.6 (f1 / f3 / f2), rmix +1.1 / +1.2 / -0.2, SQF-X 0.0 / +0.1.
  So jitter is worth about +5 % top-5 and -0.7 % ranking error in the six, mostly through d6. The proxy runs of REVE-scratch, SQF and
  the packaged LaBraM carry NO jitter (the finals of scratch and SQF do): the proxy six is not the sub16 six on those slots.
- B. Test repeats in training: neutral, now on stable metrics (fold 1, three members swapped: +3.4 ± 3.2 | -0.4 ± 0.7 | -0.1 ± 0.1;
  fold 3 d6: -0.9 ± 2.2 | +0.2 | -0.1). Verdict stands.
- C. Cross-image mixup (fold 3, old recipe without lag / jitter, exact ranks): in the six d6 @12 -3.3 ± 2.4 | +0.7 ± 0.6 | -0.2 ± 0.1,
  d6 @14 -0.7 | +0.8 | -0.2; rmix @12 +0.5 ± 2.3 | -0.7 | -0.1, rmix @14 +1.9 | +0.1 | -0.1. Alone d6 @12 ranking error -2.5 ± 0.3.
  Small and positive on the stable metrics, one fold: re-opened (rmix jitter + mixup on folds 1 and 3 queued).
- D. REVE-scratch @10 vs the packaged @12: alone top-100 +8.8 / +11.4 / +8.9, ranking error -5.9 / -6.8 / -5.4 (folds 1 / 2 / 3);
  in the six top-5 -2.7 ± 2.1 / +0.7 ± 2.2 / 0.0 ± 1.9, ranking error -0.4 / -0.5 / -0.2. The later epoch trades the bulk of the
  ranking for nothing at the top at 5k; the 14k reads of 10 Nov favoured @12 at the top. NOTE for every epoch or regularisation
  question: ranking error and the extreme top can move in opposite directions (memorisation sharpens the top, hurts the bulk), so
  ranking error alone must not decide those; pooled top-5 / top-1 over four folds and the 14k gallery decide.
- Follow-ups launched: E LaBraM + jitter vs packaged LaBraM (cached, 3 folds), F d6 @10 vs @12 (cached, 3 folds) -> `audit_r11b.out`.
  Queued at the front of the night list: rmix `--time_jitter 5` (f1, f3), d6 jitter 5 (f1, later f3), REVE-scratch jitter 3 (f1, f3),
  SQF jitter 3 (f1, f3), rmix jitter + image mixup (f1, f3). The list holds 32 runs on four GPUs.
- 16:25 NEW GOAL (user): a submission by Monday 16 Nov with a multi-fold forecast of +10 % over sub16; results must be significant on
  several folds. Reasoning after the jitter re-read: an augmentation judged "~0" in October on trained images is worth +5 % on never-trained
  images, mostly through REVE-d6, so the augmentation family is re-opened on the current proxy and screened on d6 AND rmix (a rmix-only
  screen would have missed most of the jitter effect). New flags (train.py `warp_mask`, self-checked): `--time_warp a` (content after the
  onset stretched by a random factor in [1 - a, 1 + a]; the members lose 4-20 % under a ±5 % stretch, S01) and `--time_mask n`.
  Night list rebuilt by priority (32 runs, old list kept as `night.old`): jitter 5 (rmix, d6), time warp 0.05, time mask 12, band drop
  0.5 (d6), layout targets, soft labels, GCE, image mixup, then SQF-X jitter f2 reference, temperature 0.1, Alljoined-only finish,
  pixels in scratch / SQF-X. Dropped: second variants (soft tau 0.02, GCE 0.2, grid 3) until the first shows a signal; proxy runs of
  scratch / SQF with jitter (their finals already have it; deltas are read slot by slot against the same six).
  Timetable: screens on folds 1 + 3 until Sunday morning, folds 2 + 4 of the survivors Sunday, finals Sunday night, package Monday.
- 16:27 audit re-reads E / F (`audit_r11b.out`). E LaBraM + jitter (same recipe, epoch 10) vs the packaged LaBraM: alone top-5 +13.9 ± 6.0 /
  +11.8 ± 7.1 / +11.0 ± 5.3 % (folds 1 / 2 / 3), ranking error -2.5 / -1.2 / -0.8; in the six +1.7 ± 2.5 / +0.3 ± 3.1 / +2.3 ± 2.3 %,
  ranking error -0.2 / 0.0 / +0.2. Jitter helps LaBraM alone on all three folds; about +1.5 % in the six. F d6 @10 vs @12: in the six
  -1.7 / +0.3 / -0.5 %, ranking error -0.3 / -0.2 / -0.1: no change of the budget.
  Pattern to keep in mind for the +10 % goal: a member that is 10 % better alone moves the six by ~1.5 %; jitter moved it by 5 % because
  it improved three members at once. A lever must apply to (nearly) all members to reach the target.
- `nu04_stack.py` saves ranks with `SAVE=<tag>` (5k and 14k); `nu06_pool.py <tag> <folds>` gives the pooled multi-fold forecast.
- 17:00 ops: w515 replaced by w515b (no run on 74489: a 90-min run after its chain would hit the 19:41 limit); detached scorer.sh scores each finished round-11 run in the six (logs/mi/r11/<recipe>_f<k>.out, ranks saved for nu06_pool.py).

## T2 soft labels, first read (17:57; rmix jitter + `--soft_label 0.3`, fold 1, `logs/mi/r11/`, pooled read `nu06_pool.py`)
Alone vs the rmix jitter reference: top-5 +6.5 ± 5.4 %, top-100 -6.1 ± 1.3 %, ranking error +10.5 ± 0.5 %. In the six (rmix slot only):
5k top-1 0.0 ± 5.5, top-5 +4.0 ± 2.7, top-20 +0.5 ± 1.3, top-100 -0.2 ± 0.7, ranking error +1.1 ± 0.1; 14k top-1 +4.5 ± 8.0, top-5 +3.0 ± 4.5,
top-20 +5.9 ± 2.4, top-100 +1.8 ± 1.0, ranking error +0.7 ± 0.1. The reverse profile of the regularisers: the bulk of the ranking gets
WORSE and the top gets better, from one member. One fold: fold 3 moved to the front, with d6 + soft labels (f1), temperature 0.1 (rmix f1,
tests whether this is only a sharper softmax) and a stronger dose (a 0.5, rmix f1).
- 19:50 USER: "focus on the soft labels lead in the immediate future". Night list re-ordered (40 runs): a soft-label block first: rmix
  soft3 f3 (running on 74991 since 19:42), d6 soft3 f1, rmix a 0.5 f1, rmix temperature 0.1 f1 (control), d6 soft3 f3, rmix soft3 f2,
  LaBraM jitter + soft3 f1, rmix soft3 + pixels f1, rmix soft3 tau 0.02 f1, rmix a 0.5 f3, REVE-scratch jitter + soft3 f1, SQF-X jitter + soft3 f1;
  then the fold-4 SQF reference and the earlier list (augmentations, layout targets, GCE, mixup, audit re-tests).
  Ops: the j516 chain's parent was stopped so the interactive GPU joins the list after its SQF-X fold-4 run (SQF fold 4 moved into the
  list). My first re-ordering script dropped 24 queued lines through a wrong field index; restored from the list written at 16:25, and the two
  entries already started (rmix jitter 5 f3 at 18:54, rmix soft3 f3 at 19:42) removed. No run was lost or duplicated.
  74991 (mi_gpu4, 515) started at 19:41 when 74489 reached its limit.
- 19:56 rmix `--time_jitter 5`, fold 1: alone top-5 -11.1 ± 4.2 %, ranking error +1.1 ± 0.3 (worse than jitter 3); in the six 5k top-5
  +2.2 ± 2.2, top-20 -1.2 ± 1.1, ranking error 0.0; 14k top-5 0.0 ± 3.5. No sign that a wider jitter helps rmix; fold 3 running, d6 later.

## V01 a small constant shift of the output toward the typical-image direction (20:00-20:40, `nv01_mean_shift.py`, cached outputs, no training)
Question: is the soft-label gain (top better, bulk worse) a shift toward the mean target? Soft labels spread ~20 % of the target mass over
the batch; the soft-label rmix has smaller norms (27 vs 36) and a larger common component (.386 vs .317).
Test: sub16 six of the fold, summed output S + c x (mean norm of S) x m, m = unit mean of the 11,540 TRAINED image targets (a constant a
package can ship). Folds 1-3 pooled, never-trained images, change against c = 0 in %:
| c | 5k r=3 top-1 / top-5 / top-20 / ranking error | top-5 per fold | 14k r=3 top-1 / top-5 / top-20 |
|---|---|---|---|
| -0.02 | +1.1 / -2.5 ± 1.3 / -2.3 / -0.1 | -1.5 / -2.3 / -3.4 | +3.9 / -1.5 ± 2.2 / -2.6 |
| +0.02 | +4.7 ± 3.5 / +2.3 ± 1.4 / +1.1 ± 0.7 / +1.0 | +2.5 / +3.7 / +1.1 | +5.9 ± 5.3 / +3.3 ± 2.3 / +3.1 ± 1.1 |
| +0.03 | +5.8 ± 4.1 / +4.0 ± 1.6 / +1.1 / +1.8 | +3.7 / +5.7 / +3.0 | +7.8 ± 6.1 / +2.7 ± 2.7 / +4.7 ± 1.4 |
| +0.05 | +5.5 ± 4.9 / +2.6 ± 2.0 / -0.2 / +3.9 | +2.7 / +3.0 / +2.3 | +17.6 ± 9.3 / +4.0 ± 3.4 / +4.8 ± 1.8 |
| +0.07 | +3.3 / -0.1 ± 2.3 / -2.9 / +6.4 | -2.5 / +1.7 / +0.9 | +27.5 ± 11.0 / +1.7 ± 3.7 / +4.9 ± 2.0 |
r=4: 5k top-5 +2.4 ± 1.0 (c .01), +2.6 ± 1.3 (.02), +1.8 (.03), -0.9 (.05); 14k +3.0 ± 1.6, +3.1 ± 2.1, +2.6, +4.0 ± 3.2.
Larger shifts are destructive in both directions (c ±0.2: -18 % / -51 % top-5). Same profile as the soft-label member: the top gains,
the bulk loses, more so in the larger gallery.
Controls: (1) the six's own image-independent mean output b (|mean output| = 0.32-0.37 of the mean norm, cosine with m only 0.04): a shift
along b has its optimum at 0 (±0.02: ±0.9 %). The lever is specific to the typical-image direction. (2) Gallery and targets restricted to one
typicality half of the 5k images: less typical half +0.6 ± 1.2 / +0.3 ± 1.4 % top-5 (c .02 / .03), more typical half +1.7 ± 1.0 / +2.7 ± 1.2 %.
So about half of the gain is a transfer between typicality bands (depends on the gallery's composition) and half sits inside the typical band;
no band loses at c = 0.02.
Reading: the ensemble under-scores typical images (P02: wrong winners are less typical, r -.25); a small typicality prior corrects it at
the top of the ranking. Candidate for a package as one stateless line (c = 0.02-0.03): +2 to +4 % top-5 on three folds (~2 SE), same sign
on every fold at 5k r=3 for c .02-.05. To do: fold 4; whether it adds to the soft-label members or replaces them; distractor galleries from
other image sets cannot test it (targets would all be THINGS and the prior would favour them trivially).
- 20:25 rmix jitter 5 fold 3: alone -5.3 ± 4.0 %; in the six -0.7 ± 1.8. Folds 1 + 3 pooled in the six: top-5 +0.7 ± 1.4, top-20 -1.1 ± 0.8,
  ranking error +0.1. Wider jitter closed for rmix on two folds; d6 jitter 5 stays in the list (jitter acts mostly through d6).

## T2 soft labels replicate on fold 3; they add to the output shift at 14k (21:11-21:30)
- rmix jitter + soft labels (a 0.3), fold 3: alone top-5 +1.6 ± 5.3 %, top-100 -2.3 ± 1.3, ranking error +7.3 ± 0.4; in the six top-5 +1.1 ± 2.3.
  Folds 1 + 3 pooled, rmix slot only (`nu06_pool.py mi_rmix_ho5k_lag5_jit3_soft3 1,3`): 5k top-1 +4.5 ± 4.1, top-5 +2.5 ± 1.7 (+4.0 / +1.1),
  top-20 +0.7 ± 0.9, ranking error +0.9 ± 0.1; 14k top-1 +14.3 ± 6.9, top-5 +3.6 ± 3.0 (+3.0 / +4.1), top-20 +3.6 ± 1.5, top-100 +1.3 ± 0.7.
- Soft-label rmix AND the V01 shift (`ALT=1:mi_rmix_ho5k_lag5_jit3_soft3 nv01_mean_shift.py 1,3`, against the unchanged six):
  5k top-5: swap alone +2.5 ± 1.7, + shift c .02 +3.3 ± 2.1, c .03 +3.8 ± 2.3 (shift alone on these folds +1.8 / +3.3);
  14k top-5: swap alone +3.6 ± 3.0, + c .02 +6.5 ± 3.7 (+5.5 / +7.6), + c .03 +8.6 ± 4.1 (+6.7 / +10.5); 14k top-20 +6.1 ± 1.8 / +6.6 ± 2.1
  (shift alone on these folds +3.6 / +2.7 top-5). Sub-additive at 5k, additive at 14k.
- The shift's gain grows with the gallery (expected top-5 in random sub-galleries from the 14k ranks, three folds, `GSIZE=1`):
  c .03: 500 0.0, 1,000 +0.7, 2,500 +1.9, 5,000 +2.6, 10,000 +2.4, 14,367 +2.7 %; c .05: -1.2, -0.2, +1.9, +3.4, +2.9, +4.0 %.
  The board gallery is ~40k: the proxy at 5k under-reads this family; 14k is the read to trust, and the trend points up beyond it.
- Running on all four GPUs since 21:04: d6 soft3 f1 (75032), d6 soft3 f3 (74991), rmix a 0.5 f1 (74916), rmix temperature 0.1 f1 (516).
- 21:30 gallery-size trend of soft-label rmix + shift (folds 1 + 3, expected top-5 in random sub-galleries): swap alone 500 +1.2, 2,500 +1.8,
  5,000 +3.3, 10,000 +4.9, 14,367 +3.6 %; + shift c .03: -0.6, +3.4, +5.4, +8.5, +8.6 %; c .05: -2.4, +3.0, +4.9, +6.5, +7.4 %.
- 21:35 list re-ordered for the Monday goal: soft labels (a 0.3) in EVERY member on folds 1 and 3 first, each as its reference recipe + the
  flag only (LaBraM `mi_plab_ho5k_lag5_long_soft3`, REVE-scratch `mi_scr_ho5k_lag5_soft3`, SQF-X `mi_sqfx_ho5k_lag5_jit3_soft3`, SQF
  `mi_sqf_ho5k_lag5_soft3`), rmix soft3 fold 2, SQF fold-4 reference; then dose / sharpness variants and the earlier list. Scorer patterns
  extended to the SQF and LaBraM slots. Timetable: all-member screen on folds 1 + 3 by ~Sunday 06:00, folds 2 + 4 by ~15:00, finals after.

## Soft labels: REVE-d6 replicates the 14k gain; dose and temperature controls (22:53)
- d6 jitter + soft labels (a 0.3), d6 slot only, folds 1 + 3 pooled: 5k top-1 +7.0 ± 4.2, top-5 +0.5 ± 1.8 (+0.2 / +0.7), top-20 +1.3 ± 0.9,
  ranking error +0.7 ± 0.1; 14k top-1 +6.5 ± 6.5, top-5 +4.8 ± 3.0 (+3.7 / +5.8), top-20 +5.2 ± 1.5, top-100 +1.0 ± 0.7. Alone: top-5 -4.0 / -3.6,
  ranking error +9.3 / +6.7. Same profile as rmix (14k top-5 +3.6 ± 3.0, top-20 +3.6 ± 1.5): two members, two folds each, all four 14k
  top-5 reads positive (+3.0, +4.1, +3.7, +5.8).
- Dose a 0.5 (rmix, fold 1): 5k top-5 +2.0 ± 3.3, 14k top-5 +3.7 ± 5.8, top-20 +8.3 ± 3.1, ranking error +3.5 / +2.7 (alone top-100 -23 %):
  no more at top-5 than a 0.3, a larger cost in the bulk. a 0.3 kept (fold 3 of 0.5 is running, for the record).
- Temperature 0.1 (rmix, fold 1): 5k top-5 -4.2 ± 2.1, 14k 0.0 ± 3.8, ranking error -0.2. Not the soft-label profile: the gain is not a sharper softmax.
- OPS 22:03-22:13: nine list entries of the form `env DIRECT=1 bash scripts/mi/nc_train.sh ...` exited at once under srun on the batch
  nodes ("/bin/env: bash: Key has expired": the home ticket of those jobs has expired and env's PATH search stops on it), so the workers
  skipped to the next rmix entries (no GPU idle, nothing trained wrongly). Fix: `scripts/mi/job_nc.sh` wrapper (no env), all list lines
  converted, the nine entries re-queued at the head: LaBraM, REVE-scratch, SQF-X, SQF with soft labels on folds 1 and 3, and the SQF fold-4 reference.
- 23:00 USER: three positive folds are enough to decide, no need for the fourth. List: fold 2 added for every soft-label member (d6 first), the SQF-X jitter fold-2 reference moved up, the SQF fold-4 reference moved to the end.

## Soft labels in rmix: third fold (23:47)
- Fold 2, rmix slot only: alone top-5 -9.5 ± 5.9, ranking error +10.4; in the six 5k top-5 +3.0 ± 3.3, top-100 +1.1, ranking error +1.2.
- THREE FOLDS pooled (`nu06_pool.py mi_rmix_ho5k_lag5_jit3_soft3 1,2,3`): 5k top-1 +4.0 ± 3.6, top-5 +2.6 ± 1.5 (+4.0 / +3.0 / +1.1), top-20
  +1.0 ± 0.8, top-100 +0.2, ranking error +1.0 ± 0.1; 14k top-1 +15.7 ± 6.7, top-5 +0.2 ± 2.4 (+3.0 / -7.7 / +4.1), top-20 +3.7 ± 1.3,
  top-100 +1.7 ± 0.6, ranking error +0.6. Three folds positive at 5k top-5; at 14k top-1 and top-20 are positive and top-5 is carried down
  by fold 2 (~125 hits there, SE ~9 %). Adopted for rmix by the three-fold rule.
- Soft labels + pixel targets in rmix (fold 1): 5k top-5 +4.2 ± 2.7 (soft alone +4.0), 14k top-5 +1.2 ± 4.9, top-20 +8.3 ± 2.5 (soft alone
  +5.9), top-100 +3.2 ± 1.1. The pixel head adds its bulk gain on top; fold 3 queued first, then ONE rmix final (soft, or soft + pixels).

## First stack forecast with soft labels (23:52; folds 1 + 3; `ALT=0:d6 soft3,1:rmix soft3,5:LaBraM jitter` in `nv01_mean_shift.py`, against the unchanged six)
| | 5k top-1 / top-5 (per fold) / top-20 | 14k top-1 / top-5 (per fold) / top-20 |
|---|---|---|
| three swaps, no shift | +5.5 ± 5.4 / +4.2 ± 2.6 (+5.4 / +3.0) / +0.6 | +22.1 ± 10.2 / +6.2 ± 4.3 (+6.7 / +5.8) / +7.2 ± 2.2 |
| + shift c .02 | +10.6 ± 6.3 / +1.2 ± 2.6 (0.0 / +2.3) / -1.0 | +23.4 ± 11.8 / +13.4 ± 4.7 (+14.0 / +12.8) / +8.4 ± 2.5 |
| + shift c .03 | +10.1 ± 6.6 / +1.8 ± 2.7 (+2.0 / +1.6) / -1.9 | +32.5 ± 13.0 / +12.5 ± 4.9 (+10.4 / +14.5) / +8.0 ± 2.6 |
Expected top-5 by sub-gallery size (from the 14k ranks), c .02: 500 +0.2, 1,000 +0.8, 2,500 +4.1, 5,000 +6.2, 10,000 +12.7, 14,367 +13.4 %.
Reading: with two soft-label members the swaps alone give +4 % (5k) / +6 % (14k); the shift adds at 14k (+13 %) and not at 5k (+1 to 2 %).
The two galleries disagree on whether the shift still adds once the members carry soft labels; errors 2.6 / 4.7 %. Check of a possible
artefact: the 9,367 extra distractors of the 14k gallery are as typical as the 5,000 targets (cosine to the mean trained target .8185 vs
.8214, sd .061), so the shift does not favour targets over distractors by construction.
Open until fold 2 of d6 (queued) and the other members: the forecast is between +4 % and +13 % on two folds.

## Soft labels: three folds for REVE-d6 and for the two-member stack (15 Nov, 02:02)
- d6 jitter + soft labels, fold 2 (d6 slot only): alone top-5 -3.0 ± 6.9, ranking error +10.5; in the six 5k top-5 +2.3 ± 3.2, top-100 +2.2 ± 0.8.
  Three folds pooled, d6 slot: 5k top-5 per fold +0.2 / +2.3 / +0.7; 14k top-1 +8.8 ± 6.9, top-5 +2.1 ± 2.4 (+3.7 / -4.2 / +5.8), top-20
  +4.6 ± 1.3, top-100 +1.5 ± 0.6.
- STACK on three folds (soft-label d6 + soft-label rmix + LaBraM with jitter, against the unchanged six):
  | | 5k top-1 / top-5 (per fold) / top-20 | 14k top-1 / top-5 (per fold) / top-20 |
  |---|---|---|
  | swaps, no shift | +1.1 ± 4.9 / +4.1 ± 2.2 (+5.4 / +4.0 / +3.0) / +0.8 | +23.5 ± 9.6 / +2.7 ± 3.6 (+6.7 / -5.6 / +5.8) / +7.9 ± 1.8 |
  | + shift c .02 | +2.9 ± 5.5 / +2.0 ± 2.2 (0.0 / +4.4 / +2.3) / -0.6 | +23.5 ± 10.5 / +7.9 ± 4.1 (+14.0 / -4.9 / +12.8) / +7.5 ± 2.0 |
  | + shift c .03 | +2.2 ± 5.7 / +1.8 ± 2.4 (+2.0 / +1.7 / +1.6) / -1.1 | +30.4 ± 11.3 / +6.3 ± 4.1 (+10.4 / -8.4 / +14.5) / +7.1 ± 2.1 |
  Three folds positive at 5k top-5 for the swaps (+3.0 to +5.4). At 14k the weak-people fold 2 is negative at top-5 in every soft-label
  read (~125 hits there) while top-1 and top-20 are clearly positive. Soft labels adopted for d6 and rmix (three-fold rule).
- Closed variants: soft labels + pixels (rmix f3 in the six -0.2 ± 2.6 vs +1.1 for soft alone; f1 +4.2 vs +4.0; pooled 5k +1.9 ± 1.8, 14k
  top-20 +5.7 ± 1.6): no top-5 gain over soft labels alone -> finals use soft labels only. Similarity temperature 0.02 (rmix f1: +4.0 ± 2.9,
  alone top-100 -14 %): same top, worse bulk. Dose 0.5 fold 3: +1.8 ± 2.9, ranking error +2.4. Default (a 0.3, tau 0.05) kept.
- FINALS queued at the head (02:05; `scripts/mi/job_final.sh`, logs `logs/mi/final_<recipe>_lag5jitsoft.log`):
  rmix_reg_optx_things @10 and reve_d6_optx_things @12, each `--things_lag 5 --time_jitter 3 --soft_label 0.3`.

## Lower-noise forecast: all four repeat triplets per (person, image) (02:12; `TRIPLES=1 nv01_mean_shift.py`, bootstrap over groups)
One triplet per group gave errors of 4-10 % at top-1..3 of the 14k gallery (the level that matches the board's top-5 of 40k), and an
outlier (+23.5 ± 9.6 % at 14k top-1). With the four leave-one-out triplets the errors halve. Stack = soft-label d6 + soft-label rmix +
LaBraM with jitter, three folds, against the unchanged six:
| | top-1 | top-2 | top-3 | top-5 (per fold) | top-10 | top-20 |
|---|---|---|---|---|---|---|
| 5k, no shift | +1.3 ± 2.7 | +0.6 ± 1.9 | +1.9 ± 1.5 | +2.2 ± 1.2 (+3.8 / +0.5 / +1.9) | +2.0 ± 0.9 | +1.2 ± 0.6 |
| 5k, shift .02 | +2.1 ± 3.0 | +1.2 ± 2.1 | +1.0 ± 1.7 | +0.8 ± 1.3 (+1.3 / -2.0 / +2.2) | +1.0 | -0.3 |
| 14k, no shift | +8.1 ± 4.4 | +4.2 ± 2.8 | +0.9 ± 2.4 | +3.7 ± 2.1 (+11.1 / -6.5 / +4.7) | +2.9 ± 1.4 | +4.5 ± 1.1 |
| 14k, shift .02 | +11.4 ± 5.1 | +5.6 ± 3.3 | +2.6 ± 2.7 | +4.8 ± 2.2 (+13.1 / -4.6 / +4.4) | +2.4 | +3.9 ± 1.2 |
Every cell of the no-shift stack is positive; the size is +2 % (5k) to +4 % (14k) at top-5 and +4 to +8 % at top-1 / top-2 of 14k. The
earlier single-triplet numbers (+4.1 % at 5k, +13 % at 14k with the shift on two folds) were on the high side of their noise. The shift
adds 1-3 points at 14k and costs ~1.4 at 5k: undecided. FROM NOW ON stack forecasts are read with TRIPLES=1.
Goal check: the forecast of the stack with two soft-label members is +2 to +5 %, not +10 %.
- 02:22 LaBraM + soft labels (reference recipe + flag), fold 1: alone top-5 +4.4 ± 6.1, top-100 -6.0, ranking error +10.2; in the six 5k top-5
  +3.5 ± 2.9, top-100 +0.8 ± 0.7, ranking error +0.9. Same profile as rmix and d6; third member family positive on its first fold.
- 02:22 add vs swap (TRIPLES, three folds, LaBraM jitter swapped in both): soft twins of d6 and rmix ADDED (8 members) 5k top-5 +2.1 ± 1.1
  (+1.2 / +1.8 / +3.0), 14k top-5 +4.1 ± 1.8 (+7.2 / +1.1 / +3.6), ranking error +1.6 / +1.1; swapped with the plain twins kept at weight 0.5:
  5k +2.7 ± 1.1 (+3.0 / +2.6 / +2.3), 14k +4.3 ± 1.9 (+10.8 / -3.3 / +4.1); plain swap (above): 5k +2.2 ± 1.2, 14k +3.7 ± 2.1 (+11.1 / -6.5 / +4.7).
  No difference in size; keeping the plain twins makes fold 2 less negative at 14k and halves the bulk cost. Option for the package
  (inference used 1,533 s of 3,600 s with six members), decided on the full stack.
- 02:12 final rmix_reg_optx_things @10 with soft labels started on 75032 (`logs/mi/final_rmix_reg_optx_things_lag5jitsoft.log`).

## 05:45 results, and a CORRECTION of the reference (15 Nov)
- LaBraM + soft labels fold 3: in the six 5k top-5 +1.8 ± 2.8 (fold 1 +3.5 ± 2.9); folds 1 + 3 pooled, LaBraM slot: 5k top-5 +2.6 ± 2.0, 14k
  top-5 +4.2 ± 3.5, top-20 +5.1 ± 1.6. REVE-scratch + soft labels fold 3: +4.4 ± 2.7 (fold 1 +3.0 ± 3.2); pooled 5k top-5 +3.7 ± 2.0, top-10
  +3.8 ± 1.5, 14k top-5 +2.4 ± 3.3, top-20 +6.4 ± 1.7. All ten fold reads of soft labels in the six are positive at 5k top-5 (rmix +4.0 / +3.0 /
  +1.1, d6 +0.2 / +2.3 / +0.7, LaBraM +3.5 / +1.8, scratch +3.0 / +4.4).
- Four soft-label members (d6, rmix, LaBraM, scratch) swapped, folds 1 + 3, TRIPLES: 5k top-1 +4.4 ± 4.0, top-5 +3.3 ± 1.9 (+2.3 / +4.3), top-100
  -2.7, ranking error +7.3; 14k top-1 +12.2 ± 7.0, top-2 +8.4 ± 4.6, top-3 +6.8 ± 3.7, top-5 +8.6 ± 3.0 (+14.2 / +3.7), top-20 +5.7 ± 1.7.
  The 5k top-5 gain saturates (the bulk cost compounds: ranking error +7 %) while the 14k top keeps growing.
- CORRECTION: sub16's LaBraM is `final/labram_optx_things_lag5jit`, i.e. it HAS jitter (log lines 1283, 1305). My statement of 14 Nov 16:30
  ("LaBraM in sub16 has no jitter") was wrong, and every "stack" forecast above that counted "LaBraM with jitter" as a swap over-stated the
  gain over sub16 by that swap (~+1.5 % at top-5). The proxy reference for the LaBraM slot must be `mi_plab_ho5k_lag5_jit3` (epoch 10), not
  `mi_plab_ho5k_lag5_long`. `nv01_mean_shift.py` gained `REF=slot:recipe[:ckpt]`; forecasts are recomputed against that reference
  (`REF=5:mi_plab_ho5k_lag5_jit3:checkpoint_test_best.pth`). The LaBraM soft-label read above (no-jitter recipe + flag against the no-jitter
  reference) stays valid as a measure of the flag; the deployable recipe is jitter + soft labels: its fold 2 replaces the queued
  no-jitter fold-2 run (`mi_plab_ho5k_lag5_jit3_soft3`), scored against `mi_plab_ho5k_lag5_jit3`.
- 04:15 final rmix soft labels done (`final/rmix_reg_optx_things_lag5jitsoft`); d6 final at epoch 9 of 12.

## Corrected forecasts against the sub16-like reference (LaBraM with jitter in the reference six; 05:54, TRIPLES)
| candidate (swaps) | folds | 5k top-1 / top-2 / top-5 (per fold) | 14k top-1 / top-2 / top-3 / top-5 (per fold) / top-20 |
|---|---|---|---|
| soft d6 + soft rmix | 1-3 | +3.1 ± 2.5 / +1.1 ± 1.8 / +1.9 ± 1.1 (+3.5 / +0.7 / +1.1) | +10.3 ± 4.2 / +6.9 ± 2.9 / +3.6 ± 2.3 / +3.6 ± 1.9 (+8.0 / -4.5 / +5.5) / +4.5 ± 1.0 |
| same + shift .02 | 1-3 | +3.9 ± 2.9 / +1.7 ± 2.0 / +0.4 ± 1.3 | +13.7 ± 4.9 / +8.3 ± 3.3 / +5.4 ± 2.6 / +4.7 ± 2.1 / +3.8 ± 1.1 |
| + soft REVE-scratch | 1, 3 | +4.8 ± 3.4 / +4.2 ± 2.5 / +2.7 ± 1.6 (+1.5 / +3.9) | +7.1 ± 5.4 / +7.1 ± 3.9 / +6.4 ± 3.2 / +7.9 ± 2.7 (+11.2 / +4.8) / +5.9 ± 1.5 |
| same + shift .02 | 1, 3 | +6.9 ± 3.9 / +5.6 ± 2.9 / +1.1 ± 1.8 | +10.8 ± 6.4 / +9.4 ± 4.5 / +10.9 ± 3.7 / +7.9 ± 3.0 / +4.2 ± 1.6 |
Reading: against the right reference, two soft-label members give +2 % (5k top-5) and +4 to +10 % at the top of 14k on three folds; a
third member (two folds) brings the top of 14k to +6 to +8 % without the shift and +8 to +11 % with it. LaBraM and the two SQF members
are not in these numbers yet.

## 09:30 (15 Nov) alignment replay with the soft-label composition; per-member reads with all triplets; finals queued
- ALIGNMENT REPLAY (user request; `nw01_e2e_cache.py 1 soft ...` + `E2E=soft ns06_align_batching.py 1`, log `logs/mi/w01_align_soft_f1.out`).
  Composition: d6 / rmix / SQF-X / REVE-scratch with soft labels, SQF, LaBraM jitter; detector = the packaged one (lag-only rmix, vertex + 1.5).
  Against no alignment, top-5 (5k, r=3): delay 0: mixed 64 -0.9 ± 1.7, mixed 32 +0.2 ± 1.9, mixed 128 0.0 ± 1.7, one person -0.2 ± 1.9
  (sub16 composition, same test: +2.7 ± 2.3 / +0.7 / +3.0 / -0.2); delay -2 / +2: +10.0 ± 4.3 / +5.7 ± 3.2; delay -4 / +4: +30.0 ± 6.4 /
  +15.6 ± 5.1 (sub16 composition +54.6 / +22.4: the new composition loses less under a delay by itself, LaBraM having jitter here).
  The step costs nothing at zero delay with the soft-label members and still recovers a delay. Folds 2 and 3 not replayed.
- Per-member soft-label reads with all triplets (folds 1 + 3, member alone in the six; 5k top-1 / top-5 | 14k top-1 / top-5 / top-20):
  LaBraM +(-1.8) ± 2.3 / +1.9 ± 1.0 | +0.9 / +4.0 ± 1.9 / +3.4 ± 0.9; SQF-X +3.6 ± 1.9 / +1.2 ± 0.9 | +7.4 ± 3.4 / +2.2 ± 1.3 / +3.0 ± 0.7;
  SQF +3.6 ± 2.0 / -0.1 ± 0.9 | +0.6 / +1.0 ± 1.5 / +3.3 ± 0.8. The single-triplet "SQF negative at top-5" (-3.7 / -2.3) was noise; SQF is
  ~0 at top-5 and slightly positive around it. LaBraM jitter + soft labels fold 2 (vs LaBraM jitter): in the six 5k top-5 +0.7 ± 3.3.
- fold-2 scoring bug avoided: the SQF-X jitter fold-2 reference is training, its directory exists without a checkpoint; `nu04_stack.py` and
  `nv01_mean_shift.py` now fall back to the lag-only SQF-X on a missing CHECKPOINT (not a missing directory).
- FINALS queued at the head (trained in parallel to the third folds to keep Monday; the composition is decided on the three-fold forecast):
  REVE-scratch @12 (started 09:25 on 75032), LaBraM @10 (`scripts/mi/job_final_gen.sh`, recipe checked against `final/labram_optx_things_lag5jit`),
  SQF-X @39; each lag 5 + jitter 3 + soft labels 0.3. The SQF soft-label fold-2 run (step 75032.35) was cancelled as redundant (SQF ~0 on two folds).

## 10:45 (15 Nov) alignment replay fold 3; swap vs union compositions (folds 1 + 3); inference-time budget
- ALIGNMENT REPLAY fold 3, same soft-label composition (`logs/mi/w01_align_soft_f3.out`): delay 0: mixed 64 -1.9 ± 1.7 (top-100 +0.5 ± 0.4),
  mixed 32 -1.5 ± 1.9, mixed 128 -0.6 ± 1.2, one person -0.2 ± 1.8; delay -2 / +2: +11.8 ± 3.9 / +2.5 ± 3.0; delay -4 / +4: +31.6 ± 6.3 /
  +16.1 ± 5.0. Residual shift +0.20 (sd 0.53). Two folds: zero-delay top-5 -0.9 and -1.9 (each within ~1 SE, top-100 +), delays recovered.
- COMPOSITIONS, folds 1 + 3, TRIPLES, reference = six with the no-jitter LaBraM on both sides (soft LaBraM of folds 1 / 3 has no jitter):
  | composition | 5k top-1 / top-5 (per fold) / top-100 / ranking error | 14k top-1 / top-2 / top-5 (per fold) / top-100 |
  |---|---|---|
  | 3 soft swapped (d6, rmix, scr) | +1.9 / +3.7 ± 1.6 (+3.6 / +3.7) / -0.5 / +4.4 | +8.5 / +6.3 / +9.0 ± 2.7 (+13.1 / +5.3) / +3.1 |
  | 3 soft swapped + their plain twins @0.5 | +2.8 / +4.9 ± 1.4 (+4.7 / +5.0) / +0.5 / +2.8 | +6.8 / +8.9 / +6.5 ± 2.3 (+11.0 / +2.5) / +3.1 |
  | 5 soft swapped (+ SQF-X, LaBraM) | +2.3 / +1.8 ± 2.0 (-1.0 / +4.4) / -4.6 / +9.9 | +8.5 / +3.6 / +6.6 ± 3.2 (+10.7 / +2.9) / +0.7 |
  | 5 soft swapped + 5 plain twins @0.5 | +5.8 / +4.4 ± 1.7 (+3.2 / +5.6) / -0.7 / +5.1 | +14.5 / +9.5 / +9.7 ± 2.8 (+15.8 / +4.1) / +2.8 |
  | 5 soft ADDED (11 members, weight 1) | +3.4 / +4.8 ± 1.4 (+4.6 / +5.0) / +0.4 / +3.0 | +10.8 / +10.1 / +8.0 ± 2.5 (+12.5 / +4.0) / +2.6 |
  | 5 soft added @0.5 | +4.6 / +4.7 ± 1.2 (+5.4 / +4.1) / +1.1 / +1.3 | +5.4 / +4.4 / +5.2 ± 1.9 (+9.9 / +1.0) / +3.1 |
  Swapping ALL five is the worst option (bulk cost compounds: ranking error +10 %, top-100 -4.6). Three swaps, or soft members next to the
  plain ones, read +4 to +5 % (5k) and +6.5 to +10 % (14k); differences between those rows are inside the noise. Shift .02 on the five-swap:
  5k top-5 0.0, 14k +5.8 (no help).
- TIME. Member forward at batch 64 on a (shared) 3080, ms: d6 10.8, rmix 6.2, SQF-X 2.1, REVE-scratch 41.6, SQF 1.8, LaBraM 7.8 (six: 70),
  detector 4.3 per view x 9 views = 39. Sealed ~1.3 M trials = ~21 k batches x 70 ms = 1,500 s: member forwards explain sub16's 1,533 s.
  Estimates by that ratio: sub16 + alignment ~2,400 s; six + 4 cheap soft twins (no second scratch) + alignment ~3,000 s; 11 members +
  alignment ~3,900 s (over 3,600; one other team's 3,875 s run was scored). A union must leave out the second REVE-scratch.

## 11:25 (15 Nov) union control; REVE-scratch fold 2; three-fold forecasts of the candidate compositions (TRIPLES, sub16-like reference)
- CONTROL (folds 1 + 3): five extra PLAIN members next to the six (d6 lag, rmix lag, SQF-X lag, scr @10, LaBraM jitter as second LaBraM):
  5k top-5 +0.7 ± 0.9, 14k +1.3 ± 1.5, top-100 +0.4 / +0.7. Five soft-label members in the same position: +4.8 ± 1.4 / +8.0 ± 2.5.
  The union gain is the soft labels (a different error profile), not the member count. Four soft added (no SQF-X): 5k +5.8 ± 1.4
  (+5.5 / +6.1), 14k +8.1 ± 2.4; six soft added (with SQF): +4.5 / +8.8.
- Time-feasible unions, folds 1 + 3 (no-jitter LaBraM on both sides): A = soft d6, rmix, LaBraM added + scr swapped: 5k +5.0 ± 1.5,
  14k +7.4 ± 2.5; B = same, scr plain: +4.2 ± 1.3 / +7.7 ± 2.3; C = A + soft SQF-X added: +5.9 ± 1.6 / +8.2 ± 2.6; D = A with SQF-X
  swapped: +4.3 / +8.4. All the same within noise.
- REVE-scratch + soft labels fold 2 (scorer, single triplet): alone +1.9 ± 8.0, in the six +1.3 ± 3.5, top-100 +1.4, ranking error +0.8.
  Three folds positive in the six: +3.0 / +1.3 / +4.4 (pooled 5k top-5 +3.0 ± 1.8, top-10 +4.4 ± 1.3; 14k top-5 +1.6 ± 2.6, top-20 +5.9).
- THREE FOLDS, reference = sub16-like six (LaBraM jitter). Soft LaBraM = `mi_plab_ho5k_lag5_long_soft3` on folds 1 / 3 (no jitter: a
  handicap for the candidate) and the jitter + soft run on fold 2 (cache symlinked under the `long_soft3` name for fold 2 in `<OUT>`).
  | composition | 5k top-1 / top-2 / top-5 (per fold) / top-100 / ranking error | 14k top-1 / top-2 / top-3 / top-5 (per fold) / top-100 |
  |---|---|---|
  | 3 soft swapped (d6, rmix, scr) | +3.5 / +2.6 / +1.7 ± 1.4 (+1.5 / -1.2 / +3.9) / -1.0 / +4.8 | +5.6 / +8.2 / +7.3 / +6.0 ± 2.3 (+11.2 / +0.9 / +4.8) / +2.5 |
  | same + shift .02 | +5.3 / +3.1 / +0.1 ± 1.5 (-0.8 / -2.8 / +3.0) / -3.3 / +7.4 | +10.6 / +10.0 / +10.3 / +5.7 ± 2.4 (+9.3 / -0.2 / +6.5) / +1.2 |
  | A: soft d6, rmix, LaBraM added, scr swapped (9) | +3.8 / +2.8 / +3.4 ± 1.3 (+3.2 / +1.5 / +5.1) / -0.1 / +3.5 | +7.3 / +10.0 / +7.0 / +6.1 ± 2.2 (+9.8 / +5.3 / +3.3) / +2.9 |
  | A + shift .02 | +8.0 / +3.2 / +1.9 ± 1.4 (-0.4 / -0.1 / +5.6) / -2.2 / +6.0 | +13.7 / +12.0 / +10.3 / +8.4 ± 2.4 (+10.7 / +6.3 / +7.9) / +1.9 |
  | B: soft d6, rmix, LaBraM added, scr plain (9) | +3.4 / +2.7 / +3.0 ± 1.1 (+4.1 / +2.8 / +2.1) / +0.2 / +2.3 | +5.8 / +8.0 / +6.1 / +7.6 ± 2.0 (+11.2 / +4.2 / +6.6) / +2.8 |
  | B + shift .02 | +5.3 / +3.3 / +2.6 ± 1.3 (+2.2 / +2.0 / +3.4) / -1.8 / +4.6 | +10.2 / +10.6 / +8.3 / +6.5 ± 2.2 (+7.2 / +5.3 / +6.6) / +1.9 |
  A and B are the first compositions with all three folds positive at 5k AND 14k (2.5-3.5 SE pooled). B needs no REVE-scratch soft
  final. Size: +3 % (5k) and +6 to +8 % (14k) at top-5; +6 to +10 % at top-1 / top-2 of 14k. Not +10 % at top-5 on the proxy.
  The pure swap is weaker on fold 2 and costs the bulk; the shift is still undecided (5k -0.4 to -1.5, 14k -1.1 to +2.3).
- OPS: the TRIPLES forecasts on the interactive node starve the local training's loaders (SQF-X soft fold 2 fell to 1 it/s): two low-value
  forecast rows stopped; run forecasts with few threads.

## 12:10 (15 Nov) V02 union screen: nothing else cached adds to composition B; epoch-budget transfer check queued
- `nv02_union_screen.py 1,3` with BASE = six + soft d6, rmix, LaBraM (B): each of the 36 cached variants present on folds 1 and 3 added as
  a tenth member (`logs/mi/v02_union_B_f13.out`, TRIPLES). Best: soft REVE-scratch 5k top-5 +1.5 ± 0.7 (+0.2 / +2.8), 14k -1.2 ± 1.1;
  second soft rmix (dose .5) +0.1 / -0.4; soft SQF-X -0.3 / -0.4; soft SQF -0.4 / -1.6. EVERY plain-type extra (lag-only twins, pixel /
  low-level / semantic auxiliary members, other epochs, jitter 5) reads 5k -1 to +1 and 14k -1 to -3 (most 2 SE, both folds): next to B
  a plain extra dilutes the soft members at the large gallery. B sits near the right soft : plain proportion (3 : 6); no cached model
  is a further complementary member.
- User (11:30): prefers six members at most, "if that costs accuracy it's not that important". Six- to eight-member compositions being
  forecast (drop slots with `ADD=slot:recipe@-1`).
- User question: is the final retrain procedure fully optimised? All data are used (20 + 10 people, 16,540 images, 10 of 80 test
  repeats); the EPOCH BUDGET is transferred from the fold models although a final has 1.77x the steps per epoch (15,975 vs 9,015): never
  verified whether the optimum is tied to the epoch (schedule position) or to the step count. Check queued at the head: rmix fold 1,
  standard 20-epoch cycle, checkpoints 6 / 8 / 10 / 12 / 14 / 16, on the 15 Alljoined training people (`mi_rmix_ho5k_lag5_jit3_e16`) and
  on 8 of them (`mi_rmix_ho5k_lag5_jit3_n8e16`, `scripts/mi/job_rmix_n8.sh`, `nc_train.sh` gained `ALLJ=`): same best epoch = the budget
  transfers; a later optimum with fewer steps per epoch = the finals should stop earlier. n8e16 started 11:52 on the local GPU.

## 14:40 (15 Nov) smaller compositions; epoch budget vs amount of data; PACKAGE sub23 built
- Three folds, TRIPLES, sub16-like reference, top-5 (per fold) at 5k | 14k, then 14k top-1 / top-2:
  B nine (six + soft d6, rmix, LaBraM)        +3.0 ± 1.1 (+4.1 / +2.8 / +2.1) | +7.6 ± 2.0 (+11.2 / +4.2 / +6.6) | +5.8 / +8.0
  S6 eight (six + soft rmix, LaBraM)          +3.6 ± 1.0 (+4.9 / +4.9 / +1.5) | +6.4 ± 1.7 (+9.5 / +4.4 / +5.0)  | +2.9 / +6.7
  S8 eight (B without REVE-scratch)           +2.1 ± 1.2 (+1.8 / +1.6 / +2.8) | +5.9 ± 2.1 (+9.0 / +7.2 / +2.1)  | +6.9 / +7.5
  S7 seven (B without REVE-scratch and SQF)   +0.7 ± 1.3 (+0.1 / +0.1 / +1.7) | +5.7 ± 2.3 (+10.9 / +2.7 / +2.9) | +4.2 / +7.7
  S1 six (d6, rmix, LaBraM swapped)           +2.0 ± 1.4 (+3.5 / -1.7 / +3.1) | +4.8 ± 2.3 (+11.4 / -1.9 / +3.6) | +13.1 / +9.7
  S3 six (d6, rmix, LaBraM pairs only)        -0.5 ± 1.4                       | +3.9 ± 2.4
  All within noise of each other; B is the only one with the largest 14k top-5 and every fold positive at both sizes. The user prefers
  six members but puts accuracy first: B is packaged; S1 is the six-member fallback (fold 2 negative).
- EPOCH BUDGET vs DATA (user question). `mi_rmix_ho5k_lag5_jit3_n8e16` (8 Alljoined people + THINGS, the standard 20-epoch cycle), alone
  on the clean proxy against the 15-person rmix @10, ranking error / top-100: ep6 +4.5 / -15.2, ep8 +4.6 / -13.1, ep10 +5.7 / -13.9,
  ep12 +7.6 / -16.0, ep14 +11.0 / -18.5, ep16 +15.0 / -22.8. With fewer people the optimum is EARLIER (6-8; 15 people: 10) and
  over-training costs fast. So more data moves the optimum later, not earlier: the finals (20 people, same epoch budgets) are not
  over-trained; if anything one or two epochs short, on the flat part of the curve. The 15-person run to epoch 16
  (`mi_rmix_ho5k_lag5_jit3_e16`, local, until ~15:45) gives the shape past 12.
- **sub23 = `neurips_challenge/submissions/sub23_ens9_lag5jit_soft3_align.zip`** (528 MB, sha256 e491958c…d84c): the six finals of sub16
  + `final/reve_d6_optx_things_lag5jitsoft` + `final/rmix_reg_optx_things_lag5jitsoft` + `final/labram_optx_things_lag5jitsoft` (each
  lag 5, jitter 3, soft labels 0.3, same budgets 12 / 10 / 10) + alignment detector (lag-only final rmix, shifts -8..+8 step 2, bias 1.5;
  `package_submission.py --bias` now takes a float). Contract check (`check_package.sh`) passes. `--verify 1`: delay 0 32.89 / 32.93
  (batch 64 / 37), delay +4 32.04 / 32.20 (trained people: a plumbing check, not a level). Time per 64-trial predict on one 3080:
  sub16 80.7 ms, sub19 110.1 ms, sub23 138.0 ms = x1.71 -> at most ~2,600 s on the grader (sub16: 1,533 s; limit 3,600 s).
  Forecast vs sub16: +3 % (5k top-5), +6 to +8 % (14k top-1 / top-2 / top-5), plus the alignment gain if the sealed data are delayed
  (zero-delay cost of the alignment with soft members: -0.9 ± 1.7 and -1.9 ± 1.7 on folds 1 / 3).

## 14:50 (15 Nov) SOFT LABELS CLOSED (user: wrap up, move to the remaining list)
- Weight of the three soft members in B (three folds, TRIPLES; 5k top-5 | 14k top-5 | 14k top-1 / top-2): 0.6: +3.1 | +5.3 (+7.2 / +0.8 /
  +6.9) | +0.8 / +3.9; 1 (packaged): +3.0 | +7.6 | +5.8 / +8.0; 1.5: +2.9 | +7.0 | +6.2 / +10.1; 2: +2.3 | +6.7 | +8.3 / +9.7. Flat from 1 to
  2, lower at 0.6: weight 1 stays.
- SQF-X soft fold 2 (scorer): in the six +3.6 ± 2.8 (three folds +3.2 / +3.6 / +2.3 as a swap) but as a tenth member next to B -0.3 / -0.4.
- Final verdict of the lead: soft labels are a top-of-ranking lever that pays as a UNION with the plain members; three soft members
  (d6, rmix, LaBraM) saturate it; dose, tau, weight, extra soft members, pixel + soft, output shift add nothing on top. sub23 is the
  deliverable. The REVE-scratch soft final (epoch 7 of 12, step 75032.40) was stopped as redundant at 14:37; the SQF-X soft final
  finishes anyway (`final/sqf_dbn_optx_things_lag5jitsoft`, unused).
- Queue now: time warp (d6 f1 running on 74991), layout targets grid2 (rmix f1 on 75032 since 14:37), then time mask, GCE, image mixup,
  band drop, jitter 5 on d6, Alljoined-only finish, pixels in scratch / SQF-X. The closed temperature-0.1 fold-3 run was dropped.
  New candidates are read two ways: as a swap in the six (scorer -> `logs/mi/r11`) and as an extra member next to B (`nv02_union_screen.py`).

## 16:05 (15 Nov) epoch budget vs data: CLOSED (the budgets transfer); time warp d6 fold 1 neutral
- `mi_rmix_ho5k_lag5_jit3_e16` (15 Alljoined people, fold 1), alone against its own epoch 10, ranking error / top-100: ep8 -1.1 / -0.2,
  ep10 0, ep12 +1.4 / -1.6, ep14 +3.7 / -3.4, ep16 +6.5 / -6.6 (top-5 flat within ±4; in the six ±1-3, ranking error -0.3 .. +0.7).
  Optimum 8-10 with 15 people, 6-8 with 8 people (n8e16 above): the optimum moves LATER by at most ~2 epochs when the people double.
  Finals (20 people) at the fold budget (10) sit on the optimum or within one epoch of it. No retrain. Question closed.
- Time warp ±5 % (`mi_d6_ho5k_lag5_jit3_warp5` f1): alone top-5 -5.1 ± 4.9, top-100 -1.3, ranking error -0.1; in the six -0.2 ± 2.3 / -0.1 /
  0.0. Neutral on fold 1; fold 3 stays queued (rule), rmix fold 1 running on 74991.
- Running 16:00: layout targets rmix f1 (75032), time mask d6 f1 (74916), time warp rmix f1 (74991), GCE rmix f1 (local).
  SQF-X soft final finished 15:06 (unused).
- 16:12 layout targets (`mi_rmix_ho5k_lag5_jit3_auxgrid2` f1, 2x2 grid of DINOv2 patch tokens as auxiliary targets): alone top-5 -5.7 ± 4.5,
  top-100 -0.6, ranking error -0.4; in the six -2.7 ± 2.2 / 0.0 / 0.0. Neutral (top-5 inside the seed noise); fold 3 queued.

## 17:00 (15 Nov) fold-1 screens: time mask, time warp (rmix), GCE: all neutral; none acts as a union member
- In the six, fold 1 (top-5 / top-100 / ranking error): time mask 12 on d6 -1.0 ± 2.2 / +0.7 / -0.2 (alone ranking error -1.8); time warp on
  rmix -2.7 ± 2.1 / +0.1 / +0.1; GCE q .4 on rmix +2.2 ± 2.4 / +0.1 / -0.2. All neutral -> fold 3 (queued).
- As a tenth member next to B (fold 1, `nv02_union_screen.py 1 <filter>`): warp rmix -1.2 / -1.7, warp d6 -1.5 / -2.9, grid2 -0.6 / -2.1,
  mask d6 -0.9 / -1.3, GCE -0.1 / -1.3 (5k / 14k top-5, ±0.9 / ±1.4): like every plain extra. None has the soft-label profile.
- User rule (16:15): fold 1 neutral -> fold 3; fold 3 negative -> stop, neutral -> fold 2; fold 2 neutral -> give up, positive -> fold 4.
- 17:20 OUTPUT SHIFT on B, PAIRED against B without it (`VS=1 ... nv01_mean_shift.py 1,2,3`, TRIPLES): c .02: 5k top-1 +1.8 ± 1.5, top-5
  -0.4 ± 0.7 (-1.9 / -0.8 / +1.2), top-100 -1.9; 14k top-1 +4.1 ± 2.3, top-2 +2.4 ± 1.6, top-3 +2.0 ± 1.4, top-5 -1.0 ± 1.0 (-3.6 / +1.1 /
  0.0), top-100 -0.9. c .03 / .04: 14k top-1 +5.1 ± 2.8 / +6.5 ± 3.5, top-5 +0.2 / +0.1, 5k top-5 -0.5 / -2.0. With the soft members in,
  the shift only moves hits from ranks 3-20 to ranks 1-2 (under 2 SE) and costs the bulk; top-5 is flat at both sizes. Not significant,
  and it rests on the sealed images being as typical as ours. CLOSED: not packaged.

## 19:05 (15 Nov) screens: all neutral so far
| lever | member, fold | in the six: top-5 / top-100 / ranking error | alone: ranking error | next (user rule) |
|---|---|---|---|---|
| time warp ±5 % | d6 f1 / f3 | -0.2 ± 2.3 / -0.1 / 0.0 ; +0.5 ± 2.1 / +0.3 / -0.1 | -0.1 ; -0.9 | fold 2 (queued) |
| time warp ±5 % | rmix f1 | -2.7 ± 2.1 / +0.1 / +0.1 | +0.6 | fold 3 running (74916) |
| time mask 12 | d6 f1 | -1.0 ± 2.2 / +0.7 / -0.2 | -1.8 | fold 3 running (local) |
| time mask 12 | rmix f1 | -2.2 ± 2.2 / -0.1 / -0.1 | +0.2 | fold 3 queued |
| layout targets grid2 | rmix f1 | -2.7 ± 2.2 / 0.0 / 0.0 | -0.4 | fold 3 running (74991) |
| GCE q .4 | rmix f1 | +2.2 ± 2.4 / +0.1 / -0.2 | +0.3 | fold 3 queued |
| image mixup .5 | rmix f1 | -0.2 ± 2.2 / +0.2 / -0.2 | -0.1 | fold 3 queued |
| band drop .5 | d6 f1 | -2.2 ± 2.4 / +0.1 / -0.4 | -3.0 (top-100 +3.2) | fold 3 queued |
Band drop and time mask improve the bulk of d6 alone (ranking error -3.0 / -1.8) with nothing at the top of the six: the regulariser
pattern again. Alljoined-only finish (`_aj8` rmix f1) running on 75032 since 18:00.

## 19:10 (15 Nov) new pretrained member: ST-EEGFormer (braindecode 1.6.1, `braindecode/STEEGFormer-small`)
Why: every augmentation screen is neutral; besides soft labels the only additions that ever paid in the ensemble are new pretrained
families (LaBraM +3.9 % on three folds). LUNA and ST-EEGFormer were on the plan page and never run. ST-EEGFormer-small: MAE-pretrained
ViT, 25 M, 512-wide, depth 8, 16-sample patches at 128 Hz (our 120 Hz window resampled to 128 = 8 patches x 32 electrodes = 256 tokens),
all 32 Alljoined electrode names in its channel vocabulary. `module/eeg_encoder/foundation.py` gained `STEEGFormer` (5 lines; features =
mean of the layer-normalised tokens -> linear head). Checks: forward finite, electrode permutation changes the output, TorchScript trace
exact. COST: 94 ms per 64-trial batch on the shared 3080 (REVE-scratch 42, d6 11): as a tenth member the package would pass the time
limit (138 + 94 ms -> ~4,400 s), so it could only enter by replacing REVE-scratch or under half precision. First question is its value:
`mi_stf_ho5k_lag5_jit3` fold 1 (LaBraM's schedule, lr 2e-4 as for the pretrained REVE, stop at 10, checkpoints 6 / 8) queued at the head.

## 19:15 (15 Nov) TIME LIMIT IS 7,200 s, AND WE ARE THIRD (user pointed at the results tab)
- Live competition API (`/api/competitions/17974/`): Sealed Final Phase `execution_time_limit` = **7200** (the repo copy of
  `competition_image.yaml`, Sept 24, says 3600: stale). Scored runs on the board: 6,424 s, 4,458 s, 3,875 s. Every "over the limit"
  statement of today (11 members, ST-EEGFormer as a tenth) was wrong: sub23 ~2,600 s, + ST-EEGFormer ~4,400 s, both fine.
- Board 15 Nov 19:10 (`results/things_eeg/mi/nc/board_2026-11-15.json`, `/api/phases/30149/get_leaderboard/`), aggregated top-5:
  1 sazeglio 0.001362 (14 Nov, 6,424 s; per-epoch top-5 0.000686), 2 qingyangli 0.001311 (13 Nov, 3,065 s), 3 viktorhugo (sub16)
  0.001258 (1,533 s), 4 JLShen 0.001245, 5 _ares_ 0.001221, 6 hongchaoshu 0.001198, 7 akgokce 0.001192. First place is +8.3 % above
  sub16, second +4.2 %; board SE ~4 %. sub23's forecast (+6 to +8 %) lands at about 0.00133-0.00136: level with first, not clear of it.
- LUNA added on the user's suggestion (`foundation.py` 'LUNA', `scripts/mi/job_luna.sh`): pretrained base (PulpBio/LUNA_base, 6.5 M + head,
  240-sample window = 6 patches of 40, electrodes pooled into 4 queries per patch, 3-D positions from the standard montage; 22 ms per
  batch) and a from-scratch variant with a wider electrode read (8 queries per 16-sample patch, 15 tokens x 512, 33 M). Checks: finite,
  electrode-permutation sensitive, bf16 within 1 %. TorchScript trace of LUNA fails on a device mismatch inside braindecode
  (`repeat_interleave` index on CPU): to fix only if a LUNA member earns a place. Queue head: ST-EEGFormer f1, LUNA pre f1, LUNA scratch f1.

## 19:40 (15 Nov) layout targets CLOSED (fold 3 negative); Alljoined-only finish fold 1 neutral; new members training
- Layout targets grid2, rmix fold 3: in the six top-5 -5.7 ± 2.2 (fold 1 -2.7 ± 2.2), top-100 +0.2, ranking error 0.0. Fold 3 negative
  -> closed (user rule); the pixel + grid runs are dropped from the queue.
- Alljoined-only finish (`--alljoined_after 8`, rmix f1): alone top-5 -7.6 ± 4.5, ranking error -0.5; in the six -2.2 ± 2.2 / -0.5 / -0.2.
  Neutral -> fold 3 stays queued.
- ST-EEGFormer f1 started 19:24 on 75032 (4.4 it/s, ~34 min per epoch, done ~01:10); LUNA pretrained f1 started 19:27 on 74991
  (10 it/s, ~15 min per epoch, done ~22:00).

## 20:10 (15 Nov) fold-3 reads: time mask d6 neutral (-> fold 2), time warp rmix negative (closed)
- Time mask 12, d6 fold 3: in the six -1.4 ± 1.9 / +0.3 / -0.3 (fold 1 -1.0 ± 2.2 / +0.7 / -0.2); alone ranking error -2.0 (f1 -1.8).
  Neutral on both -> fold 2 queued (`mi_d6_ho5k_lag5_jit3_mask12` 2).
- Time warp, rmix fold 3: in the six -3.4 ± 2.1 (fold 1 -2.7 ± 2.1), top-100 -0.1, ranking error 0.0: same sign twice, pooled about
  -3 ± 1.5 -> negative, closed for rmix. (d6: -0.2 / +0.5, neutral; its fold 2 is queued.)
- LUNA from scratch f1 started 20:00 on 74916; GCE rmix f3 started 20:01 (local). ST-EEGFormer: epoch 1 done, 4.7 it/s.

## 20:45 (15 Nov) new-encoder variants queued (user: "don't hesitate to tweak them or cross them or improve them")
- Per-epoch held-out curve (200 test images, fold 1): ST-EEGFormer 17.0 / 22.7 (ep 1 / 2; LaBraM 21.2 / 26.0, REVE-scratch 19.4 / 21.0);
  LUNA pretrained 15.5 / 21.1 / 23.5 / 23.0 (flattening: 4 queries per patch look too coarse); LUNA scratch 16.5 / 18.3.
- Queue head, ahead of the augmentation follow-ups (`scripts/mi/job_stf.sh`): ST-EEGFormer + soft labels (the union profile);
  ST-EEGFormer without its final LayerNorm (`--fm_arch_kwargs {"final_norm":false}`: the property that sets REVE apart from the other
  token models, October LOO); ST-EEGFormer at lr 5e-4 (LaBraM's rate). Layer-wise lr decay is not tried (closed on LaBraM, I19).

## 21:50 (15 Nov) LUNA pretrained (base), fold 1: weak alone, no value as an extra member -> this form closed
`mi_luna_ho5k_lag5_jit3` @10: alone top-5 1.105 % = -28.3 ± 4.5 against LaBraM, top-100 -12.8, ranking error +7.1. Swapped for LaBraM in
the six: +0.5 ± 2.8 / -0.9 / +0.4. As a 7th member next to the six (TRIPLES): 5k top-5 +1.5 ± 1.3, 14k -3.4 ± 2.0; as a 10th next to B:
-0.9 ± 1.0 / -4.0 ± 1.5. Too weak (4 queries per patch) and it costs the large gallery. The from-scratch variant with the finer
electrode read (`mi_lunas_ho5k_lag5_jit3`, due ~23:30) decides the family. ST-EEGFormer + soft labels started 21:45 on 74991.
- 22:36 GCE q .4, rmix fold 3: in the six -0.2 ± 2.2 / +0.1 / +0.2 (fold 1 +2.2 ± 2.4); alone ranking error +1.9. Neutral -> fold 2
  queued. Curves (200-way, fold 1): ST-EEGFormer 28.3 at epoch 6 (LaBraM 29.1); LUNA scratch 26.7 at epoch 10 (pretrained 23).
  ST-EEGFormer without final norm started 21:55 (local).

## 08:50 (16 Nov) morning: queue lost with node 516, rebuilt on the NAS; ST-EEGFormer is a real member; LUNA closed
- OPS. The interactive job moved from 516 to 518 at 08:20 (user). The queue (list, log, workers) lived in the node-local scratchpad of
  516 and is gone; the last training log before that was written at 03:57 (the four ST-EEGFormer runs and LUNA scratch finished; none
  of the augmentation follow-ups started; cause of the stop unknown, the log is lost). New queue in `logs/mi/queue/` (NAS):
  `night.list`, `night.log`, `worker.sh <tag>` (runs ON the GPU node as one srun step, mkdir lock), `fit.sh <hours> <cmd>` (a line runs
  only if the holder job has that much time left, else goes back to the end). Workers restarted 08:23 on 74916 / 75032 / 74991.
  Replacement holders 76146 / 76147 / 76148 (`mi_gpu6-8`, `-d afterany:` 74916 / 75032 / 74991, same options) start their own worker.
- NEW MEMBERS, fold 1 (`logs/mi/x01_new_members_f1.out`; alone vs LaBraM jitter | swapped for LaBraM in the six | 7th next to the six,
  5k / 14k top-5, TRIPLES | 10th next to B):
  | member | alone top-5 / top-100 / rank err | LaBraM slot | 7th | 10th next to B |
  |---|---|---|---|---|
  | ST-EEGFormer (lr 2e-4) | -8.8 ± 4.9 / -6.3 / +4.2 | +1.7 ± 2.7 | +3.1 ± 1.3 / +5.0 ± 2.1 | -0.2 ± 0.9 / -1.2 ± 1.7 |
  | ST-EEGFormer + soft labels | -8.6 ± 5.5 / -11.4 / +15.1 | +2.2 ± 3.1 | **+4.3 ± 1.6 / +7.1 ± 2.5** | **+1.5 ± 1.0 / +4.0 ± 1.7** (top-100 +0.5 / +1.8) |
  | ST-EEGFormer, no final norm | -8.3 ± 5.1 / -5.2 / +4.0 | +1.9 ± 2.6 | +2.3 ± 1.2 / +1.6 ± 1.9 | +0.8 / -0.3 |
  | ST-EEGFormer, lr 5e-4 | -5.7 ± 4.9 / -0.8 / +1.5 | +2.9 ± 2.8 | +1.9 ± 1.2 / +2.5 ± 1.9 | -0.7 / +1.3 |
  | LUNA scratch (8 queries, patch 16) | -24.7 ± 4.9 / -10.5 / +4.2 | -1.7 ± 3.0 | +1.0 ± 1.2 / -3.3 ± 2.1 | -0.2 / -3.5 ± 1.6 |
  ST-EEGFormer is a member of LaBraM's value (equal in its slot, positive as a 7th in all four variants); with soft labels it is the
  first model that adds on top of B (one fold). LUNA: both forms weak alone and negative at 14k -> family closed.
- Launched: ST-EEGFormer + soft labels folds 3 and 2 (queue head), ST-EEGFormer plain fold 3, and its FINAL in parallel
  (`final/steegformer_optx_things_lag5jitsoft` @10, interactive GPU 518, since 08:48). Augmentation follow-ups behind them.
- 08:55 fold-1 STACK with ST-EEGFormer, against the sub16-like reference (TRIPLES; fold 1 is B's best fold): B 5k top-5 +4.1 ± 1.9, 14k
  +11.2 ± 3.2; **B + ST soft +5.7 ± 2.2 / +15.7 ± 3.6** (14k top-2 +12.2, top-100 +3.8); B + ST soft + ST plain +6.8 / +14.3 (top-100
  +1.4 / +4.4, ranking error +2.5 / +1.5); B + ST soft @1.5 +7.5 / +16.0; six + ST plain + ST soft +6.7 / +7.5. ST soft adds about
  +1.5 (5k) and +4.5 (14k) to B on this fold. Folds 3 and 2 decide.
- OPS 08:40-08:50: the interactive node 518 is shared with the other session's CVPR CPU jobs (inc_causal x4 at 100 %): an ST-EEGFormer
  training there runs at 2 it/s (21 h for a final). The final was moved to 74991 (4.3 it/s, ~10.3 h, ends ~19:05; limit 19:40;
  `--save_epochs 8 9` as a fallback), after clearing the `.claim` of the aborted local run. Pitfall: on batch nodes `xargs kill` fails
  ("Key has expired": PATH search); use the shell builtin `kill $(...)`. ST soft f3 goes to 75032 (~09:45), ST soft f2 to the
  replacement 76146 (12:26). `logs/mi/queue/scorer.sh` (interactive GPU) scores finished runs, ST-EEGFormer ones also as extra members.

## 10:20 (16 Nov) fold-3 reads: time mask rmix negative (closed), band drop d6 neutral (-> fold 2); node 518 speed
- Time mask 12, rmix fold 3: in the six -4.4 ± 2.1 / 0.0 / -0.2 (fold 1 -2.2 ± 2.2): negative -> closed for rmix (d6: neutral twice, fold 2 parked).
- Band drop .5, d6 fold 3: in the six -1.6 ± 2.1 / -0.2 / -0.2 (fold 1 -2.2 ± 2.4 / +0.1 / -0.4); alone ranking error -1.3 (f1 -3.0).
  Neutral -> fold 2 parked (`mi_d6_ho5k_lag5_jit3_bd5` 2).
- Node 518 (interactive): the same rmix training runs at 11-15 it/s against 21-22 on the batch nodes, GPU utilisation ~31 %; four
  `inc_causal.py` processes of the other session hold 4 of the 8 cores (nice 10). "CPUs fully used" (my 08:50 statement) was overstated:
  half the cores, half the training speed. The GCE fold-2 run trains here (since 09:35) as the speed test; long runs stay on batch nodes.
- Queue: `fit.sh` now parks a line that does not fit in `night.deferred`; a worker that STARTS puts the parked lines at the head.
  Parked for 76146 (12:26): ST soft f2, ST plain f3, d6 warp f2, d6 mask f2, d6 band drop f2. ST soft f3 runs on 75032 since 09:47.

## 11:55 (16 Nov) GCE closed (fold 2 neutral), Alljoined-only finish closed (fold 3 negative)
- GCE q .4, rmix fold 2: in the six 0.0 ± 2.7 / +1.7 ± 0.7 / -0.2 (folds 1 / 3: +2.2 ± 2.4, -0.2 ± 2.2). Three folds neutral -> given up.
- Alljoined-only finish, rmix fold 3: alone top-5 -12.6 ± 4.0; in the six -3.9 ± 2.2 / -1.0 / +0.3 (fold 1 -2.2 ± 2.2). Negative -> closed.
- 11:10 the other session's CPU jobs on 518 ended; plain ST-EEGFormer fold 3 trains there since 11:26 at 4.7 it/s, then a local worker.
- Still open from the list: d6 time warp / time mask / band drop on fold 2 (parked for 76146), image mixup f3, jitter 5 on d6 f1.

## 16:05 (16 Nov) ST-EEGFormer + soft labels, fold 3: adds on top of B again -> two folds positive
- The 15:01 score failed (CUDA OOM: scorer at batch 512 next to the plain ST training on the interactive GPU); rerun on node 517
  (`logs/mi/queue/score_stf.sh`). Scorer now uses `BS=128` for ST runs (new env in `nc_lib.load`) and retries an OOM at its next pass.
- Fold 3 (`logs/mi/r11/mi_stf_ho5k_lag5_jit3_soft3_f3.out`, TRIPLES for the union reads): alone vs jitter LaBraM top-5 -0.8 ± 5.9;
  LaBraM slot of the six +4.3 ± 3.0; 7th next to the six 5k +5.3 ± 1.4 / 14k +7.0 ± 2.4; **10th next to B 5k +4.0 ± 1.1 / 14k +1.8 ± 1.6**
  (fold 1: +1.5 ± 1.0 / +4.0 ± 1.7).
- Stack against the sub16-like six, folds 1 + 3, TRIPLES (`logs/mi/x02_stack_{B,B_STsoft}_f13.out`):

| composition | 5k top-5 (f1 / f3) | 14k top-5 (f1 / f3) | 14k top-1 / top-2 | 14k top-100 |
|---|---|---|---|---|
| B (= sub23 members) | +3.1 ± 1.4 (+4.1 / +2.1) | +8.9 ± 2.2 (+11.2 / +6.6) | +7.4 / +6.8 | +2.6 |
| **B + ST-EEGFormer soft** | **+6.0 ± 1.5 (+5.7 / +6.3)** | **+12.0 ± 2.5 (+15.7 / +8.6)** | +8.0 / +11.1 | +4.1 |

  Gain by gallery size (hypergeometric from the 14k ranks), B + ST soft: 2.5k +6.2, 5k +7.7, 10k +10.4, 14k +12.0 (B: +4.4 / +4.9 / +6.6 / +8.9).
- Fold 2 (the weak fold of B: +2.8 / +4.2) trains on 76146, due ~18:00; the final ends ~19:00 on 74991.
- 16:20 weight of ST soft in B + ST soft, folds 1 + 3 (`x02_stack_B_STsoft15_f13.out`): @1.5 5k +6.8 ± 1.6 (+7.5 / +6.1), 14k +11.6 ± 2.6
  (+16.0 / +7.6) against @1 +6.0 / +12.0 -> no difference, weight 1 stays. Closed.
- 16:05 queued at the head: CBraMod pretrained + soft labels + jitter, fold 1 (`mi_cbm_ho5k_lag5_jit3_soft3`, LaBraM recipe). Plain CBraMod
  was negative as a sixth (I17), plain ST-EEGFormer adds nothing next to B either, its soft version does: does a soft member of a second
  new family add next to B + ST soft?

## 17:50 (16 Nov) ST-EEGFormer + soft labels: three folds positive -> goes into the package
- Fold 2 (`logs/mi/r11/mi_stf_ho5k_lag5_jit3_soft3_f2.out`): alone vs jitter LaBraM top-5 -28.5 ± 5.7 (weak alone on this fold); LaBraM slot
  +6.3 ± 3.8; 7th next to the six 5k +6.5 ± 2.0 / 14k +7.6 ± 3.1; **10th next to B 5k +0.8 ± 1.3 / 14k +2.0 ± 2.1**.
  10th next to B per fold (1 / 2 / 3): 5k +1.5 / +0.8 / +4.0, 14k +4.0 / +2.0 / +1.8: same sign on the three folds at both galleries.
- Plain ST-EEGFormer fold 3: alone -5.6 ± 4.9; 7th +1.4 ± 1.2 / +6.5 ± 2.0; 10th next to B +2.3 ± 1.0 / +3.0 ± 1.6 (fold 1: -0.2 / -1.2):
  mixed across folds, not packaged.
- Stack against the sub16-like six, folds 1 + 2 + 3, TRIPLES (`logs/mi/x02_stack_{B,B_STsoft}_f123.out`):

| composition | 5k top-5 (f1 / f2 / f3) | 14k top-5 (f1 / f2 / f3) | 14k top-1 / top-2 / top-10 | 14k top-100 |
|---|---|---|---|---|
| B (= sub23 members) | +3.0 ± 1.1 (+4.1 / +2.8 / +2.1) | +7.6 ± 2.0 (+11.2 / +4.2 / +6.6) | +5.8 / +8.0 / +4.5 | +2.8 |
| **B + ST-EEGFormer soft** | **+5.4 ± 1.2 (+5.7 / +3.6 / +6.3)** | **+10.4 ± 2.2 (+15.7 / +6.3 / +8.6)** | +5.4 / +11.8 / +7.6 | +3.8 |

  Decision: package sub23 + the ST soft final (10 members + detector) once `final/steegformer_optx_things_lag5jitsoft` ends (~18:35).

## 18:30 (16 Nov) sub24 built: sub23 + ST-EEGFormer soft (10 members + detector)
- `neurips_challenge/submissions/sub24_ens10_lag5jit_soft3_stf_align.zip` (629 MB, sha256 `9610d646f722b846cc0ad6526b273e56025cfa299963f12124955664d040dead`):
  sub23's nine members + `final/steegformer_optx_things_lag5jitsoft/gen/20261116-084825-gen` (`checkpoint_test_best.pth` = epoch 10, the
  fold budget), detector and alignment as sub23 (`--detector --bias 1.5`). TorchScript trace of ST-EEGFormer matches eager at batch 64 / 37 / 1.
- `--verify 1`: 33.94 / 33.75 (batch 64 / 37) at delay 0, 32.69 / 32.77 at +4 (sub23: 32.89 / 32.93 and 32.04 / 32.20).
- Time per 64-batch on a 3080 (node 515, `logs/mi/queue/time_pkg.py`, a small training on the same GPU): sub16 79.2 ms, sub23 142.7 ms,
  sub24 242.0 ms = x3.06 of sub16 -> at most ~4,700 s of the 7,200 s limit if the whole 1,533 s of sub16 scales with predict().
- `check_package.sh` passes (dev top-5 0.2633, 84.9 s).
- Forecast against sub16 (three folds, TRIPLES, without alignment): 5k top-5 +5.4 ± 1.2, 14k top-5 +10.4 ± 2.2.
- 18:30 time warp .05 on d6, fold 2: alone +16.5 ± 7.3, in the six +0.3 ± 2.9 / -1.2 ± 0.7 / +0.0 (folds 1 / 3: -0.2, +0.5). Three folds
  neutral -> given up (fold rule).

## 18:50 (16 Nov) after sub24: user uploads it Tuesday; NEW GOAL "another +10 %" for the next package (sub25)
- V02 union screen next to sub24's composition (B + ST soft = 10 members; `logs/mi/v02_union_BST_f13.out`, folds 1 + 3, 45 cached
  candidates): best plain ST-EEGFormer 5k +0.9 ± 0.7 / 14k -0.3 ± 1.1; everything else -0.3..-1.5 at 5k-mean and -0.3..-3.4 at 14k
  (soft SQF-X / SQF / scratch, pixel-target twins, lag-only twins, earlier checkpoints). The existing families are saturated at 10 members.
- Half precision at inference (`logs/mi/queue/time_amp.py`, per member, 64-batch, 3080): only ST-EEGFormer gains (99.3 -> 55.8 ms under
  fp16 autocast, cosine 1.00000); REVE / rmix / SQF / LaBraM traces unchanged. Members total 207 -> 165 ms. Members by cost: ST 99, REVE-scratch
  48, d6 13-14 (x2), LaBraM 9.5 (x2), rmix 5 (x2), SQF 2 (x2). Option kept for when the 7,200 s budget binds (autocast around the ST member only).
- Running / queued new families with soft labels (fold 1): CBraMod (518), EEGPT (queued).

## 19:20 (16 Nov) sub24's ten members: leave-one-out and backward elimination (three folds, TRIPLES, cached)
- Leave-one-out (`LOO=1 nv02_union_screen.py`, `logs/mi/v03_loo_BST_f123.out`; change of the ten when the member is dropped, 5k / 14k top-5):
  d6 +0.6 / +0.7, rmix +0.6 / +0.6, d6 soft +0.9 / -0.2, rmix soft +1.2 / -0.6 (the four REVE-token members are each redundant);
  LaBraM -0.5 / -0.3, REVE-scratch -0.1 / -1.0, LaBraM soft -1.2 / -1.7, SQF-X -1.8 / -1.1, SQF -1.7 / -1.3, **ST soft -2.2 / -2.6**
  (errors +-0.5 / +-0.9). No member is worth more than ~2.5 %: a plateau; ST-EEGFormer (the only new family) is the most valuable.
- Backward elimination (`nv03_backward.py`, `logs/mi/v03_backward_BST_f123.out`; against the ten; optimistic, chosen on the scored people):
  9 (- d6) +0.5 / +0.7; 8 (- d6 soft) +1.3 / +0.2; 7 (- rmix soft) +2.3 / -0.4; 6 (- SQF-X) +1.6 / -0.3; 5 (- REVE-scratch) +0.7 / -2.4;
  4 (- LaBraM soft) +0.3 / -3.3; 3 (- rmix) -2.6 / -4.9.
  -> eight = {rmix, SQF-X, REVE-scratch, SQF, LaBraM, rmix soft, LaBraM soft, ST soft} equals the ten on every fold at 14k (0.0 / +0.2 / +0.4);
  six = {rmix, REVE-scratch, SQF, LaBraM, LaBraM soft, ST soft} equals it within noise (14k -2.4 / -1.1 / +2.5). The d6 pair frees 27 ms
  per batch for new families. Not a reason to change sub24 (selection noise ~1 %); it is the base for sub25 compositions.
- Consequence for "another +10 %": more members of the present families cannot give it (V02 next to sub24: nothing adds). Slate: new
  families with soft labels (CBraMod running, EEGPT, pretrained REVE, recurrent queued), then whatever the board says about the alignment (sub24, Tuesday).
- 19:32 CBraMod pretrained + soft labels + jitter, fold 1 (`logs/mi/r11/mi_cbm_ho5k_lag5_jit3_soft3_f1.out`): alone -21.3 ± 5.0 vs jitter LaBraM;
  LaBraM slot -1.9 ± 3.1; 7th +1.6 ± 1.5 / -1.3 ± 2.2; 10th next to B -1.9 / -3.1; **11th next to sub24 +0.3 ± 1.0 / -4.6 ± 1.5** -> negative, closed.
  (CBraMod and LaBraM are both one-token-per-electrode models here: patch 200 = the whole second. Not a new error profile.)
- 19:45 queue for the night (fold 1, all with lag 5 + jitter 3 + soft labels .3, read as "11th next to sub24" by `scorer.sh`):
  EEGPT pretrained (518, running, ~5.5 h), pretrained REVE (`mi_prev_..._soft3`), recurrent (`mi_rec_..._soft3`),
  **LaBraM with the second stretched over four 200-sample patches** (`mi_plab4_..._soft3`, `--fm_arch_kwargs {"n_times":800}`: 32 x 4
  tokens instead of one token per electrode; `foundation.py` now accepts `n_times` on a pretrained model, the temporal embedding was
  already re-initialised), **ST-EEGFormer from scratch** (`mi_stfscr_..._soft3`, lr 1e-3: is it the pretraining or the electrode x
  time-patch geometry that makes ST the only member that adds?). Plus d6 jitter 5 f1 (old list).
  Not queued: ST-EEGFormer-base (86 M: ~17 h per fold, ~30 h final, and ~320 ms per batch would not fit the 7,200 s with the rest).
- 19:50 fold reads: **time mask 12 on d6, fold 2** in the six -1.3 ± 2.6 (alone +8.2 ± 6.4; folds 1 / 3: -1.0 / -1.4) -> three folds neutral,
  given up. **Image mixup on rmix, fold 3** in the six -1.6 ± 2.3 (alone -7.0 ± 4.2; fold 1 -0.2) -> neutral twice; NOT sent to fold 2:
  the leave-one-out of 19:20 shows the rmix and d6 slots are themselves redundant in the ten (dropping the member changes ±1 %), so no
  swap inside those slots can move the package. Same bound applies to d6 band drop (f2 being scored) and d6 jitter 5 (f1 running, last of the old list).
- GPUs 19:50: 518 EEGPT soft f1, 513 pretrained REVE soft f1, 517 recurrent soft f1, 515 (76148, started 19:41) d6 jitter 5 f1.
- 19:52 band drop .5 on d6, fold 2: in the six -1.0 ± 2.8 (alone +6.9 ± 7.0, top-100 +6.3; folds 1 / 3: -2.2 / -1.6) -> three folds neutral
  to negative, given up. The augmentation list of 14 Nov is finished: nothing beyond jitter and soft labels.
- 20:35 ops (user 20:25: new interactive node; "as soon as one of the 4 GPU releases, stop using it and switch to this node so we stay
  at 4"): session now in job 76654 on 519. In use: 513 (76146 pretrained REVE soft f1), 515 (76148 d6 jitter 5 f1), 517 (76147 recurrent
  soft f1, first to finish ~21:10), 518 (old interactive 76143: EEGPT soft f1 until ~01:00, `worker.sh local518`, `scorer.sh`).
  `logs/mi/queue/retire.sh` is the first line of `night.list`: the first worker to finish its run writes `logs/mi/queue/retired`, stops
  (its holder job stays idle on `sleep 2d`; NOT cancelled by me), and `wait519.sh` (detached on 519) starts `worker.sh local519`.
- 20:35 USER: cancel d6 jitter 5, prioritise scratch ST-EEGFormer. `mi_d6_ho5k_lag5_jit5` f1 killed at epoch 6 on 515 (partial run dir
  removed, not scored); its worker took `retire.sh` -> job 76148 (515) retired (idle holder, not cancelled); `worker.sh local519` started
  `mi_stfscr_ho5k_lag5_jit3_soft3` f1 at 20:34 on the new interactive node (3 it/s at start, cold page cache). Queue after it: LaBraM
  four patches. In use: 513, 517, 518, 519.
- 20:45 USER: "free 515" -> job 76148 cancelled (it was idle).
- 21:50 new families with soft labels, fold 1, as 11th next to sub24 (5k / 14k top-5):
  **recurrent** alone -40.0 ± 4.5 vs LaBraM; 11th -1.7 ± 1.0 / -1.7 ± 1.4 -> negative, closed.
  **pretrained REVE** (epoch 6) alone -0.3 ± 5.4; LaBraM slot +2.2 ± 2.8; 7th +2.1 / +1.5; 11th +0.2 ± 0.8 / -0.5 ± 1.3; earlier checkpoints
  as 11th (`logs/mi/x03_prev_soft_epochs_f1.out`): @3 +0.2 / -1.9, @4 +0.7 / -1.5, @5 +0.6 / -2.3 -> nothing at 5k, below zero at 14k at
  every checkpoint: closed without a fold 3 (four checkpoints agree).
  With CBraMod (19:32): three soft-label families that do not add. Open: EEGPT (518, ~01:00), LaBraM four patches (517, ~00:15),
  scratch ST-EEGFormer (519, ~02:00).
- 22:00 513 idle (list empty, its worker exited): fold 3 of the two layout tests pre-run there (`mi_plab4_..._soft3` f3, then
  `mi_stfscr_..._soft3` f3), so that a positive fold 1 has its second fold ready; wasted only if fold 1 is negative, on a GPU with nothing else to do.

## 22:45 (16 Nov) swaps as well as additions (user: "don't forget to try swapping instead of adding"); wide night queue (user: 4 GPUs busy until 10:00)
- `nv02_union_screen.py` gained `SWAPIN=<cached name>`: the candidate REPLACES each member of the base in turn. Fold 1, base = sub24's
  ten, 5k / 14k top-5 (`logs/mi/x04_swaps_f1.out`; errors ~ +-1.1 / +-1.7):
  - pretrained REVE soft: for d6 soft +1.3 / +0.8, for d6 +0.8 / +1.0, for rmix +0.7 / +1.3; other slots <= 0; for ST soft -1.3 / -4.1.
  - plain ST-EEGFormer: for rmix +2.6 / +0.1, for d6 +2.2 / 0.0, for rmix soft +3.1 / -1.7; for ST soft -1.7 / -5.0.
  - CBraMod soft: best for rmix +1.6 / -1.7, every slot negative at 14k; recurrent soft: every slot <= 0.
  The positive swaps are all INTO the redundant d6 / rmix slots and equal what dropping that member gives alone (19:20: +0.6..+1.3 / ~0)
  plus about +1: inside the noise. ST soft is never the member to replace (-3 to -7 at 14k).
- `scorer.sh` now runs on 519 (the 518 copy was stopped) and appends "swapped into each slot of sub24" to every new-family read.
- `foundation.py`: ST-EEGFormer takes `fm_arch_kwargs` (n_times on the pretrained model; patch_size / embed_dim / depth from scratch). All
  variants below build and run a forward pass on CPU. EEG-DINO dropped: its released weights have 19 electrode embeddings (we have 32).
- Night list (after `mi_stfscr` f3), all lag 5 + jitter 3 + soft labels .3, fold 1 unless noted:
  token layout: `mi_stf64` (pretrained ST, 4 time patches of 250 ms instead of 8), `mi_cbm4` (CBraMod, 4 patches), `mi_plab2` / `mi_plab8`
  (LaBraM 2 / 8 patches), `mi_stfscr32` (scratch ST, patch 32), `mi_stfscrn` (scratch ST, 256-wide depth 6: a 5 M cheap variant),
  `mi_plab4scr` (LaBraM 4 patches from scratch: layout x pretraining with `mi_stfscr`);
  others: `mi_stf_..._soft3s2` (second seed of the packaged ST recipe: what a mere re-run adds), `mi_tsc_..._soft3` (TSConv, never run with
  soft labels), `mi_luna_..._soft3` (pretrained LUNA), pre-runs `mi_plab4` f2 and `mi_egp` f3. About 40 GPU-hours; the four workers
  (513 step, 517, 518, 519) take the lines in order.
- 22:50 USER: the swaps of rmix / d6 by pretrained REVE soft and by plain ST-EEGFormer deserve more folds. Queued first: `mi_prev_..._soft3`
  f3, f2 and plain `mi_stf_ho5k_lag5_jit3` f2. Plain ST swap on folds 1 + 3 from cache (`logs/mi/x04_swap_stfplain_f13.out`, 5k / 14k top-5, f1 / f3):
  for rmix +1.3 ± 0.8 (+2.6 / +0.1) / +1.5 ± 1.2 (+0.1 / +2.9); for d6 +1.0 ± 0.8 (+2.2 / -0.1) / +1.4 ± 1.2 (0.0 / +2.8);
  for rmix soft +2.2 ± 0.9 (+3.1 / +1.4) / +0.9 ± 1.4 (-1.7 / +3.4); for d6 soft +1.3 ± 0.8 / +1.1 ± 1.3; other slots ~0 or negative;
  for ST soft -1.7 / -1.9. Two folds, +1 to +2 %, 1-2 standard errors; dropping the same member alone gives +0.6 / +0.6 (three folds).

## 01:55 (17 Nov) fold-1 reads: EEGPT dead, four-patch LaBraM neutral
- **EEGPT pretrained + soft** (`mi_egp_..._soft3` f1): alone top-5 0.208 % (-86.5 ± 1.9 vs LaBraM: it barely learns in this recipe); 11th next
  to sub24 -3.2 ± 1.3 / -5.2 ± 2.2; swapped into any slot -4.5 .. -8.7 -> closed (fold-3 pre-run removed from the list).
  Ops: its score never appeared because `nu04_stack.py` passed `bs=1000` explicitly (my `BS` env of 16 Nov only reached `nc_lib`), EEGPT
  needs > 4 GB at that batch and the scorer retried the OOM forever. `nu04_stack.py` now reads `BS`; `logs/mi/queue/score_one.sh <recipe>
  <fold>` scores one run on a chosen node.
- **LaBraM, four patches, soft** (`mi_plab4_..._soft3` f1): alone -11.9 ± 5.0 vs the one-patch LaBraM (jitter); LaBraM slot of the six +2.4 ± 3.0;
  7th +2.2 ± 1.4 / +2.8 ± 2.1; 10th next to B -1.0 / +1.3; **11th next to sub24 +0.3 ± 0.9 / -0.5 ± 1.5**; swaps: for d6 +0.9 / +0.8, for
  rmix +0.7 / +0.8, for one-patch LaBraM +0.5 / -0.1, for ST soft -2.5 / -2.6. Neutral -> fold 3 (pre-run, ends ~01:45).
  A time axis in LaBraM's tokens does not reproduce what ST-EEGFormer adds.

## 02:30 (17 Nov) scratch ST-EEGFormer negative: the PRETRAINING is what adds; four-patch LaBraM neutral on two folds; ST-EEGFormer-base started
- **Scratch ST-EEGFormer + soft** (`mi_stfscr_..._soft3` f1, lr 1e-3): alone -43.9 ± 4.4 vs LaBraM (pretrained ST soft: -9 .. -28); 7th -1.9 /
  -1.6; 11th next to sub24 -2.5 ± 1.1 / -5.1 ± 1.6 -> negative, closed. Its fold-3 pre-run was killed 15 min in (dir removed); the other
  scratch-layout lines (`mi_stfscr32`, `mi_stfscrn`, `mi_plab4scr`) are dropped from the list.
- **Four-patch LaBraM soft, fold 3**: alone +1.5 ± 5.5; 7th +4.8 / +7.2; 10th next to B +3.1 / -1.3; **11th next to sub24 +0.6 ± 0.9 /
  -1.4 ± 1.4**; swaps all <= +1.1 at 5k and <= 0 at 14k (fold 1: +0.3 / -0.5, swaps < +1). Two folds neutral with the 14k side below zero:
  closed (no fold 2); `mi_plab2`, `mi_plab8`, `mi_plab4` f2 dropped.
- Reading: neither the electrode x time-patch layout from scratch nor a time axis in LaBraM reproduces ST-EEGFormer's contribution; its
  pretrained weights do. Hence **ST-EEGFormer-base** (86 M, 768-wide, depth 12; `--fm_arch_kwargs {"hub":"braindecode/STEEGFormer-base"}`,
  new `hub` key in `foundation.py`): `mi_stfb_..._soft3` f1 on 519 from 02:25. Batch 128 does not fit a 3080 (batch 64: 7.1 GB peak,
  0.33 s per step) -> batch 64, lr 1e-4, ~17 h for 10 epochs; checkpoints at 6 and 8 give an early read (~12:30). Inference ~3x ST-small:
  only deployable in a lean composition (the six of 19:20 + fp16 on ST: ~285 ms per batch).
- Left in the list: `mi_stf64` (pretrained ST, 4 coarse patches), `mi_cbm4`, second seed of ST soft, TSConv soft, LUNA soft, ST soft .5.

## 03:15 (17 Nov) pretrained REVE soft on three folds (user: the swaps deserve more folds)
`logs/mi/x04_swap_prev_f123.out`, base = sub24's ten, TRIPLES, 5k / 14k top-5 (folds 1 / 2 / 3):
- as 11th: -0.2 ± 0.5 (+0.2 / -1.5 / +0.2) / -0.1 ± 0.8 (-0.5 / +0.7 / -0.4): nothing.
- **swapped for d6 soft: +1.1 ± 0.5 (+1.3 / +0.6 / +1.1) / +0.8 ± 0.9 (+0.8 / +1.6 / +0.4)**: the only slot positive on the three folds at
  both galleries; dropping d6 soft alone gives +0.9 / -0.2 (19:20), so the swap itself is worth ~+0.2 / +1.0.
- for d6 -0.4 / +0.9 (fold 2 -2.5 at 5k), for rmix soft +1.2 / -0.9, for rmix -0.5 / +0.4; LaBraM, SQF, SQF-X, scratch slots -0.7 .. -1.9;
  for ST soft -1.3 / -3.5.
- Reading: a consistent +1 % at most, in a redundant slot, for a member four times as slow as d6 (and a final to train). Kept as a
  candidate for the sub25 composition, to be read together with the plain ST-EEGFormer swap (fold 2 running on 513, ~07:15).
- 03:00 running: 519 ST-EEGFormer-base soft f1 (2.3-3.3 it/s, ends ~18:00-22:00), 518 `mi_stf64` f1, 517 `mi_cbm4` f1, 513 plain ST f2.

## 07:45 (17 Nov) plain ST-EEGFormer swap on three folds; compositions with the two swaps; finals queued
- **Plain ST-EEGFormer** in sub24's ten, folds 1 / 2 / 3 (`logs/mi/x04_swap_stfplain_f123.out`, 5k / 14k top-5):
  for rmix **+1.7 ± 0.7 (+2.6 / +2.7 / +0.1) / +1.7 ± 1.1 (+0.1 / +2.3 / +2.9)**; for d6 soft +2.1 ± 0.7 (+1.8 / +4.4 / +0.9) / +1.4 ± 1.2
  (-0.1 / +2.1 / +2.3); for d6 +1.1 / +1.5; for rmix soft +2.3 / +0.2; as 11th +1.3 ± 0.6 (+1.0 / +2.5 / +0.8) / +0.2 ± 0.9; for ST soft -1.1 / -2.3.
- Compositions against the sub16-like six, three folds, TRIPLES (`logs/mi/x05_C*_f123.out`):

| composition | 5k top-5 (f1 / f2 / f3) | 14k top-5 (f1 / f2 / f3) | 14k top-1 / top-2 / top-100 |
|---|---|---|---|
| sub24 (B + ST soft) | +5.4 ± 1.2 (+5.7 / +3.6 / +6.3) | +10.4 ± 2.2 (+15.7 / +6.3 / +8.6) | +5.4 / +11.8 / +3.8 |
| C1: rmix -> plain ST | +7.2 ± 1.3 (+8.5 / +6.4 / +6.4) | +12.3 ± 2.4 (+15.8 / +8.7 / +11.7) | +13.3 / +13.7 / +4.3 |
| C2: d6 soft -> pretrained REVE soft | +6.5 ± 1.2 (+7.1 / +4.3 / +7.5) | +11.4 ± 2.3 (+16.6 / +8.0 / +9.0) | +5.2 / +10.4 / +4.0 |
| **C3: both swaps** | **+7.3 ± 1.3 (+7.4 / +6.2 / +7.9)** | **+13.3 ± 2.4 (+17.6 / +8.0 / +13.1)** | +9.8 / +13.1 / +5.0 |
| C4: eight (no rmix, d6, d6 soft; + plain ST) | +6.8 ± 1.3 (+9.2 / +4.5 / +6.1) | +10.8 ± 2.4 (+14.1 / +2.3 / +13.9) | +8.5 / +14.8 / +5.2 |

  C3 is +1.8 % (5k) / +2.6 % (14k) over sub24, each fold >= sub24 at both galleries. Small, three folds agree -> both finals queued first:
  `final/steegformer_optx_things_lag5jit` (@10, ~10 h) and `final/reve_pre_optx_things_lag5jitsoft` (@6, ~3-4 h).
  NOTE: the pretrained-REVE soft folds ran at lr 1e-3 (the recipe default; the October / I15 pretrained REVE used 2e-4): the final keeps 1e-3.
  Inference: C3 adds ~140 ms per batch in fp32 (plain ST 99, pretrained REVE ~50, minus rmix and d6 soft): needs fp16 autocast on the two
  ST members (~300 ms per batch, ~5,800 s of 7,200 s by the sub16 scaling); to be measured on the package, and against sub24's real run time.
- Fold-1 reads (11th next to sub24): `mi_stf64` (pretrained ST, 4 coarse patches) +0.4 / -1.4, best swap for rmix +1.3 / +0.9 (what plain ST
  does, on one fold) -> not pursued; `mi_cbm4` -0.3 / -1.9, all swaps <= +0.4 -> closed.
- 07:50 half precision per member in the package: `submission.py` runs the members listed in `members.pt['amp']` under fp16 autocast
  (`package_submission.py --amp <indices>`). Copy of sub24 with ST soft in fp16: `--verify 1` 33.94 / 33.75 (identical to fp32), 252.6 ->
  191.1 ms per 64-batch on node 517 (a small training on the same GPU). sub24 itself is unchanged (its folder holds its own submission.py).
- 10:25 fold-1 reads (11th next to sub24): **TSConv soft** alone -41.6; 11th -1.1 / -3.6, no positive swap -> closed. **LUNA pretrained soft** alone
  -26.8; 11th -1.3 / -2.7, swaps negative at 14k -> closed. Finals running since 08:25 (plain ST, 517) and 09:27 (pretrained REVE soft, 513).
  USER 10:20: uploads sub24 now; "begin a new search program for other levers we have not tried. Stacking new encoders can only get us
  this far. We need bold new ideas."

# Round 12 (17 Nov, 10:30): a search outside encoder stacking (user: "begin a new search program for other levers we have not tried ... bold new ideas")
Re-read before choosing (this log, NIGHT_LOG, HISTORY): closed and NOT repeated: output maps / bank snap / push-away / hub debiasing
(E01, M05, N20, N47, N61), batch-context person statistics (R03: 12 batch-mates give a key at cosine .20), per-trial views (S02), one-
parameter input transforms (S01), member weights (S04, I02), concept novelty (E09, E11/12: 5-15 %), KD (old regime), cohorts as people.
Facts the slate is built on:
- **The sealed gallery is mostly OUTSIDE THINGS**: ~40k images, none public; THINGS has 9.4k images left (E00). Every proxy so far has a
  THINGS-only gallery (5k, 14k). MSS (ImageNet + VOC images) decoded zero-shot at the within-THINGS level (Q03: P(target > random) .60-.62).
- New-image generalisation is a steep lever (7.5k vs 11.5k training images: -18..-22 %), and the only lever that moved the package in round
  11 (soft labels) acts on how a trial relates to images OTHER than its own.
- The population output for a new image has cosine .07 with its embedding (Y02 below): the readable part is small; nothing on the output side.
Slate:
- **Y01 open images** (`ny01_open_images.py`, running on 518 since 10:33): grader-style targets of 40,000 COCO photographs + 50,000 ImageNet
  val images; seeded halves -> `open_bank45k.npy` (loss) and `open_gallery45k.npy` (proxy), `things_trained_bank.npy` (the 11,540 trained images).
- **Y02 open-bank loss** (`module/loss.py` `bank` / `bank_soft`, `train.py --open_bank f.npy [--open_bank_soft]`, self-checked): every trial is
  also scored against images nobody saw. `obs`: the soft-label share goes to batch AND bank images by similarity (the trial is told which
  UNSEEN images its image resembles: image coverage without EEG); `obn`: bank images as plain negatives (control); `tbs`: bank = the 11,540
  trained images (does the size of the neighbourhood matter, without outside images). rmix, fold 1, queued on 518 after Y01.
- **Y03 a 40k gallery with the sealed mix** (`nc_lib.gallery(big=2)`: 5k held-out + 9.4k THINGS + 25.6k COCO / ImageNet; `OPEN=1` in nv01 / nv02):
  forecasts so far stop at 14k and the 5k and 14k reads disagree (sub24 +5.4 vs +10.4). Compositions re-read there once Y01 ends.
- Y04 (waits for the board score of sub24): if the alignment pays on the sealed people, a richer per-batch cohort adaptation.
- **Y02-check, pull toward trained images** (`ny02_pull.py`, cache of M04, rmix lag f1; `logs/mi/y02_pull.out`): the population output pbar(j)
  of a new image, centred: cosine .069 with its own embedding, .052-.071 with the mean of its 1-100 nearest TRAINED images (which have
  cosine .60-.73 with d_j): no preferential pull. Repulsion from the trained-neighbour mean y - g|y|P: g .05 / .1 / .2 -> top-5 1.27 / 1.22 /
  0.92 vs 1.34, attraction g -.1 1.39, -.2 1.24 (noise ±.07): optimum 0. Closed again, now in the sealed-like regime.

## 11:30 (17 Nov) Y01 done; Y03 the 40k gallery with the sealed mix; sub25 requested (user: upload the two-swap package instead of sub24)
- Y01: `open_coco.npy` (40,000), `open_imagenet.npy` (50,000) saved 11:01 (27 min on 518); banks built. `mi_rmix_..._obs` f1 started 10:57 on 518.
- Y03 (`OPEN=1 nv01_mean_shift.py 1,2,3`, `logs/mi/y03_*_f123.out`; top-5 against the sub16-like six, TRIPLES; 40k = 5k held-out + 9.4k
  THINGS + 25.6k COCO / ImageNet):

| composition | 5k | 14k | **40k** (f1 / f2 / f3) | 40k top-1 / top-100 |
|---|---|---|---|---|
| sub24 | +5.4 ± 1.2 | +10.4 ± 2.2 | **+15.9 ± 3.2** (+11.3 / +19.4 / +18.4) | +16.5 / +12.4 |
| C3 = sub24 with both swaps | +7.3 ± 1.3 | +13.3 ± 2.4 | **+19.1 ± 3.6** (+11.5 / +21.6 / +25.6) | +23.4 / +13.3 |
| C5 = C3 without d6 (nine) | +6.9 ± 1.4 | +13.6 ± 2.5 | **+22.5 ± 3.8** (+16.0 / +22.3 / +30.1) | +22.1 / +14.2 |
| C6 = C3 without d6 and REVE-scratch (eight) | +6.2 ± 1.5 | +13.5 ± 2.6 | +23.2 ± 4.1 (+15.2 / +23.7 / +31.9) | +19.0 / +14.2 |

  The gains of round 11 GROW with the gallery (5k -> 14k -> 40k: sub24 +5 / +10 / +16), on every fold at 40k. Caveats: the targets are
  still THINGS images among mostly non-THINGS distractors (easier to reject than same-domain ones); the board's rank shape matched 5k r=3
  earlier (memory note of 10 Nov); the one calibration point (sub2 -> sub16) sat between the 5k and 14k reads.
  d6 is free to drop here too (C5 >= C3 at 14k / 40k); REVE-scratch within noise (C6).
- sub25: `logs/mi/queue/build_sub25.sh` (detached, log `logs/mi/build_sub25.out`) waits for the two finals (pretrained REVE soft ~14:10,
  plain ST ~17:30), then builds `sub25_ens10_...` (C3, ten members, as asked) and `sub25b_ens9_...` (C5, without d6), both with the two
  ST-EEGFormer members in fp16, replays fold 1 at delay 0 / +4, times them against sub16 / sub24, zips, contract check, sha256.
  Run-time estimate (fp16 on the two ST members): C3 ~296 ms per batch, C5 ~283, C6 ~235; sub16 = 79 ms = 1,533 s on the grader.
- 11:40 **Y02 `obs` (open bank inside the batch's softmax) is broken: killed at epoch 3.** Held-out dev top-5 per epoch 4.0 / 4.2 / 4.5 against
  12.5 / 16.2 / 19.3 for the in-batch soft run. Reading (not measured): COCO / ImageNet embeddings differ from THINGS ones by a constant
  direction, so one softmax over batch + bank is lowered by a constant "THINGS-ness" output: the image-set shortcut of Q04 (external cohorts
  in one loss), and a constant output offset wrecks the cosine ranking (V01: ±.2 is destructive). `obn` (bank as negatives, same softmax) dropped.
  New form `--open_bank_sep L` (self-checked): a softmax over the bank ALONE, target = softmax(image-bank cosine / tau): the trial's similarity
  profile over unseen images should be its image's profile (one loss per image set, as Q04's fix). Queue on 518: `tbs` (bank = the 11,540
  trained THINGS images, joint softmax: no domain gap), then `obr` (open bank, separate softmax, weight .5, with in-batch soft .3).
- 11:26 ST-EEGFormer-base, dev curve epochs 1-5: 12.4 / 16.8 / 18.4 / 19.2 / 19.6 against ST-small 12.9 / 16.9 / 18.1 / 19.8 / 20.0: identical so far.
  Its epoch-6 checkpoint (~12:00) is read as a member before deciding whether the remaining 5 h are worth the GPU.
- 12:10 **ST-EEGFormer-base soft, fold 1, epoch 6** (`logs/mi/x06_stfb_ep06_f1.out`): alone -15.3 ± 5.2 vs LaBraM; dev curve 12.4 / 16.8 / 18.4 /
  19.2 / 19.6 / 19.1 (small: .. 20.0 / 19.9); 11th next to sub24 +1.1 ± 0.9 / +0.8 ± 1.5; swapped for rmix +2.3 / +2.6 (plain ST-small does
  +2.6 / +0.1 there); **swapped for ST-small soft +0.4 / -2.6**. It is one more ST-EEGFormer, no better than the small one, at three times
  the inference time -> stopped at epoch 6.7 (redundant), 519 freed for round 12. Closed.
- `tbs` dev curve epochs 1-4: 11.3 / 15.0 / 17.8 / 19.5 (in-batch soft 12.5 / 16.2 / 19.3 / 19.8): trains normally. `obr` taken by 519.
- 12:15 queued behind `tbs` / `obr`: `xbs` (bank = the 9,367 never-trained THINGS images, joint softmax, soft: unseen images WITHOUT a domain
  gap; it contaminates the 14k / 40k proxy galleries, whose distractors these are, so it is read at 5k only) and `obr0` (open bank in its own
  softmax, no in-batch soft labels: the bank term alone).
- 13:00 `obr` (open bank in its own softmax, weight .5, + in-batch soft .3) killed at epoch 5: dev curve 7.7 / 9.7 / 12.6 / 13.2 / 13.0 against
  12.5 / 16.2 / 19.3 / 19.8 / 20.7: clearly negative at matched epochs (not the collapse of `obs`). Pulling a third of the gradient toward the
  nearest out-of-THINGS images costs the reading of THINGS images. `obr0` dropped; one low-dose run `obr01` (weight .1) started on 519.
  `tbs` finished: dev 21.3 at epoch 12 (in-batch soft 22.3); alone +3.0 ± 5.1 vs plain rmix, in the six +1.2 ± 2.7 (in-batch soft: +6.5, +4.0).
  `xbs` running on 518 since 12:52.
- 13:10 **`tbs` (soft share spread over the batch AND the 11,540 trained images), fold 1, in sub24's ten** (`logs/mi/r12_mi_rmix_ho5k_lag5_jit3_tbs_f1.out`;
  5k / 14k / 40k top-5): swapped for rmix soft +0.2 ± 0.8 / +0.3 ± 1.3 / **+3.8 ± 2.1**; for plain rmix +0.3 / +1.8 / +5.9 ± 2.2; as 11th
  -0.2 / -0.8 / +4.0 ± 2.1. Neutral at 5k and 14k, about two standard errors up at 40k in all three reads: one fold -> folds 3 and 2 queued
  (the 40k gallery is new: its fold-to-fold noise is not known yet; C3 vs C5 moved by ±3 there).

## 15:15 (17 Nov) open-bank family: unseen images as negatives teach the model to avoid the unseen; the 40k proxy rewards a THINGS-ward bias
- `xbs` (bank = the 9,367 never-trained THINGS images, joint softmax), fold 1: **alone at the clean 5k gallery top-5 -72.4 ± 3.0 %**, ranking error
  +59 %; in sub24's ten 5k +0.5 / -0.5 / -0.3 (for rmix / for rmix soft / 11th), 14k +18, 40k +35 (those galleries hold the bank images as
  distractors: the model was trained to rank them low = contamination, announced). The 5k collapse is the real result: images that are only
  ever negatives are never-shown concept-mates of everything, and the held-out images look exactly like them. With `obs` (collapse) and
  `obr` .5 (-35 % dev): **any softmax in which unseen images are pushed down teaches "the answer is never an unseen image"**. Closed.
- `obr01` (open bank in its own softmax, weight .1), fold 1: alone +4.1 ± 5.7 vs plain rmix; in sub24's ten, for rmix soft +0.5 / +0.9 /
  **+9.0 ± 2.3 at 40k**; 11th 0.0 / +0.9 / +10.4 ± 2.5. Nothing at 5k / 14k; the 40k gain is the SAME artifact from the other side: it has seen
  COCO / ImageNet-like embeddings ranked low, and the 40k distractors are the other half of those sets. Not a gain for a gallery whose
  TARGETS are outside THINGS. Not pursued.
- **Consequence for Y03**: a gallery of THINGS targets among non-THINGS distractors rewards any THINGS-ward component of the output. Soft
  labels enlarge exactly that component (|mean unit output| .386 vs .317, V01), so the 40k column (sub24 +16, C3 +19, C5 +22) is inflated
  by an amount this proxy cannot separate. The 5k and 14k reads stay the forecast. `tbs`'s +3.8 at 40k falls under the same doubt.
- Y04 (running on 519, `ny04_packages_out_of_things.py`, `logs/mi/y04_packages.out`): the packages sub19 (= sub16 + alignment), sub23, sub24
  on the only non-THINGS-image EEG on disk, ds004018 (16 people, 200 segmented objects, another lab), and on ds007964 (THINGS images,
  another lab) as control; galleries own / + 40k COCO-ImageNet / + 14.4k THINGS; paired bootstrap against sub19.

## 15:25 (17 Nov) Y04 the packages on two cohorts from other labs (`logs/mi/y04_packages.out`; r=3, paired bootstrap against sub19 = sub16 + alignment)
| cohort (targets) | gallery | sub19 top-5 | sub23 vs sub19 top-5 / top-20 / ranking error | sub24 vs sub19 top-5 / top-20 / ranking error |
|---|---|---|---|---|
| ds004018 (200 segmented objects, NOT THINGS; 16 people, 32,000 groups) | own 200 | 11.52 % | +0.4 ± 0.9 / -0.2 ± 0.4 / +0.9 | +2.0 ± 1.0 / -0.3 ± 0.5 / +1.1 |
| | + 14.4k THINGS | 0.441 % | +12.8 ± 7.0 / +15.6 ± 3.5 / -10.6 | +17.0 ± 7.8 / +31.2 ± 4.3 / -16.2 |
| | + 40k COCO / ImageNet | 0.244 % | +69 ± 16 / +37 ± 5 / -16.8 | +101 ± 20 / +58 ± 6 / -24.3 |
| ds007964 (1,900 THINGS images; 19 people, 34,270 groups) | own 1,900 | 1.762 % | +2.8 ± 3.2 / +4.6 ± 1.6 / -1.4 | **+11.9 ± 3.8 / +11.3 ± 1.8 / -3.2** |
| | + 14.4k THINGS | 0.149 % | +9.8 ± 13.0 / +7.1 ± 5.8 / -3.1 | +23.5 ± 15.0 / +23.2 ± 7.2 / -5.4 |
| | + 40k COCO / ImageNet | 0.079 % | +67 ± 28 / +37 ± 8 / -7.9 | +96 ± 36 / +57 ± 10 / -11.6 |
- **sub24's gain replicates on people from another lab in a single-source gallery**: ds007964 own 1,900 +11.9 ± 3.8 % top-5 (top-20 +11.3 ± 1.8,
  top-100 +8.1 ± 0.9); most of it comes with ST-EEGFormer (sub23 +2.8, sub24 +11.9): the pretrained member pays MORE on a foreign cohort than
  on held-out Alljoined people (+1-4 % next to B).
- On non-THINGS targets in their own small gallery (200 images: top-5 = the bulk of the ranking) nothing is lost and little gained (+0.4 / +2.0).
- Mixed-source galleries: the soft-label packages separate image sources far better (+13..+100 %), whatever the target's source. That is
  what inflated Y03's 40k column; it would only pay on the board if the sealed gallery mixes image sources.
- `tbs` fold 3 in sub24's ten: for rmix soft +0.4 / +2.3 / -1.3 (5k / 14k / 40k), 11th -0.3 / +0.3 / +2.0: fold 1's +3.8 at 40k does not
  repeat. Neutral on two folds; fold 2 (running on 518) closes it under the fold rule.

## 16:35 (17 Nov) ops: interactive node changed again (user) -> session in job 77299 on 515; `tbs` closed; Y05 started
- Lost with job 76654 (519): `worker.sh local519`, `scorer.sh`, the sub25 build watcher, the 513 step worker (its srun lived on 519). Nothing was
  training on 519. Restarted on 515: `build_sub25.sh local` (builds on this node's GPU; sub25b first), `scorer.sh`. Still running elsewhere:
  517 plain ST final (epoch 9 of 10, ends ~18:10), 518 `worker.sh local518` (idle, list empty). In use: 513, 515, 517, 518.
- `tbs` fold 2: alone -10.2 ± 6.1, in the six +2.9 ± 3.3 (folds 1 / 3: +1.2, -0.2); in sub24's ten fold 3 neutral (15:25). Three folds
  neutral -> given up. The open-bank family is closed in every form.
- **Y05 foreign-cohort screen** (`ny05_foreign_screen.py <cohort> <fold>`, lists in `nv02_union_lists.py`): every member of sub24's ten and every
  candidate of rounds 11-12, fold-k models, on ds007964 (19 people from another lab, own 1,900-image gallery): dropped / added / swapped,
  paired bootstrap. Fold-1 models on 515, fold-3 models on 513. Why: Y04 shows ST-EEGFormer is worth +9 % there against +1-4 % on held-out
  Alljoined people; the sealed people decode twice worse than held-out Alljoined people (E00), so the right composition may sit between.
