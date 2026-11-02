# ARCH_LOG: hybrid architecture search (2026-10-15 -)

Goal (user, 2026-10-15 11:10): empirically find how the signal should be processed by recombining TSConv / Squeezeformer /
REVE parts (conv vs attention at different places, attention axes, kernels, pooling, split paths). No hand-made
heuristics (frequency bands etc.), <= 5M parameters. Start from 5 designs, iterate fast, cut losers early.

Protocol: cv4 fold 1 (test subjects 2 4 5 11 15), THINGS added (`_things`), per-dataset BN (boundary 21), batch 1024,
40-epoch one-cycle lr 1e-3, same head as the best compact TSConv (Linear -> 64, residual MLP, LN, Linear -> 1536).
Metric: per-epoch sealed subject-aggregated top-5 (sealed_per_epoch.jsonl), compared at matched epochs with
`tsc_dbn40_bn64_things` (best 29.3, epochs 5/10/20/40: 21.7/26.7/25.8/28.8). Seed noise is about +-3 per epoch, so
judge on the mean over windows of epochs. Kill when clearly behind (memory: kill-losing-screens-early).
Reference fold-1 bests: TSC-bn64 29.3, SQF-dbn 28.5, REVE-d6 (19.7M) 29.4.
Already-dead designs (ALIGN_LOG A22-A27): attention BEFORE the spatial collapse, i.e. factorised space-time
(stf -5.4), electrode tokens instead of the spatial conv (etok -3.0), space attention per time step + flatten
(sta -1.7), REVE electrode x 0.2 s tokens (-1.9), small REVE (1.8M -1.5; 5.2M -0.9).
Code: ensemble_experiments/architectures/hybrid_encoders.py; recipes in cv4.sh (h*_things).

## H0 (11:45): five starting designs
| recipe | class | params | idea |
|---|---|---|---|
| hfa_things | FilterAttnTSConv | 0.39M | TSConv stem, then attention ACROSS the 40 filters (token = a filter's 36-step time course), flatten |
| hta_things | TimeAttnFlatTSConv | 0.73M | TSConv stem, 2 attention blocks over the 36 time tokens (d128), TSConv flatten readout instead of SQF's pooled stats |
| hvs_things | VirtualSensorFormer | 0.42M | REVE's tokenisation (20-sample patches, stride 10) on 16 LEARNED virtual sensors (1x1 spatial projection of the raw EEG), 4 blocks d96, mean pool |
| hds_things | DynSpatialTSConv | 0.27M | TSConv + a trial-adaptive spatial-filter residual: 8 learned queries attend over electrodes (keys from each electrode's filtered features), zero-init mix |
| hdp_things | DualPathTSConv | 0.35M | TSConv split in two: the slow stem (kernel 25, 25-sample box pool) + a fast path (kernel 9, pool 5) that keeps the content the 0.2 s box removes |
Rationale: every dead design put attention or tokens BEFORE the spatial collapse. Four of the five keep TSConv's
spatial conv and vary what happens after it (attention over filters / over time / readout); hvs tests REVE's
tokenisation after a learned spatial collapse; hdp tests splitting the temporal processing.

### H0 cuts (12:00)
fold-1 per-epoch sealed top-5 vs tsc_dbn40_bn64_things at matched epochs (base 15.3 19.4 20.0 22.7 21.7 26.2 24.8 26.8 24.4):
- hdp_things (dual slow/fast path) 14.6 15.8 16.3 21.3 21.6 19.4 21.8 ...: 9 ep mean -2.9, last5 -3.4 -> KILLED. Giving the
  head the fast path's 1296 extra unpooled features (kernel 9, pool 5) hurts: the 0.2 s box pool is a useful prior.
- hds_things (trial-adaptive spatial residual, zero-init = TSConv) 13.5 17.1 17.2 21.2 20.2 21.0 24.1 ...: 9 ep mean -2.3,
  last5 -2.5 -> KILLED. Input-dependent spatial filtering learns something that does not transfer to unseen subjects.
- hfa_things lost its first hour: node 516's warmed page cache was evicted by a concurrent grader_sim, the run read the
  NAS at 1.2 it/s (12 it/s once re-warmed with stage_eeg_cache.py). hta / hvs started 11:51.
- hfa_things (attention across the 40 filters) 13.8 17.8 20.3 18.8 21.3 21.5 21.2 23.4 22.4 24.4: 10 ep mean -2.3, last5
  -3.2 -> KILLED (12:14). Letting filters exchange information after the spatial conv does not help; flatten of
  independent filters is better.
- (12:10) the killed hdp/hds had been re-launched by their workers before the queue edit landed (orphaned alljoined_run.sh
  survived the cv4.sh kill) and ran 4-5 epochs next to hta/hvs on the same GPUs -> killed, dirs removed. hta/hvs
  epochs 1-5 ran on a shared GPU (slower, same numbers).
- hta_things (2 attention blocks over the 36 time tokens + flatten) 16.7 19.9 19.4 21.0 23.0 22.6 24.5 22.0 23.6 24.5 23.2
  23.8: 12 ep mean -1.3, last5 -2.6 (falling behind) -> KILLED (12:40).
- hvs_things (REVE tokens on 16 learned virtual sensors) 2.6 6.9 12.7 11.5 15.6 14.7 18.2: -10 -> KILLED.
H0 verdict: 0/5. Every design that adds processing AFTER TSConv's spatial conv (filter attention, time attention, a
fast path, adaptive spatial filters) loses 2-3 points; replacing the conv stem by patch tokens loses 10. TSConv's
flatten of 40 independent spatially-filtered, 0.2 s-smoothed time courses into a 64-d bottleneck is a strong optimum
for this data. H1 probes single knobs of TSConv itself (learned pool, 2x spatial filters, sepconv, query readout);
H2 the filter count, kernel and pooling.

## H2 (12:40) queued: hf80 / hf20 (filters), hp49 (pool 49 stride 4), hk49 / hk13 (temporal kernel)

## Control (13:15): bf16 BIASED every H0/H1 screen by ~-1.2
TSConvPlus with default knobs (= TSConv-bn64 re-implemented) vs the tsc arm, matched epochs:
hbase_things (bf16, as all h* runs) 9 ep -2.2 (last5 -1.7); hbase32_things (fp32 = the tsc arm) 8 ep -1.0 (last5 -0.6).
bf16 autocast costs these small conv models ~1.2 points; the residual -0.6..-1 is within seed noise. Corrected H0/H1
estimates (delta + 1.2): hta ~-0.1 (last5 -1.4), hfa ~-1.1, hds ~-1.1, hdp ~-1.7, hlp ~-1.9, hsm2 ~-1.1, hsep ~-1.1
(18 ep, last5 -1.8). Only hta was plausibly at par -> re-screened in fp32 (hta32). hbase (bf16) and hsep killed.
All h* recipes are now fp32 (cv4.sh); hbase32 runs to 40 epochs as the same-code-path reference.

## H1/H3 (13:55)
- hqr_things (8 learned queries cross-attend to the 36 time steps, fp32) 8 ep -4.1 (vs hbase32 -3.1) -> KILLED. A
  learned pooled readout loses the time-locked layout, as SQF's pooled stats do; flatten stays.
- hta32_things (hta in fp32) 7 ep +0.6 vs TSC-bn64, +1.5 vs hbase32: first design at/above par. H3 queued ahead of H2:
  attention kind (htsqf: Squeezeformer block with depthwise conv module; htgat: gated attention + SwiGLU; htdif:
  differential attention) and depth (htd4: 4 blocks; htd1: 1 block d64).
- (14:30) hta32_things 16.3 20.5 20.4 21.7 24.4 25.3 25.8 23.1 23.7 22.8 24.6 21.5 22.4 24.9 24.7 24.0 23.0 23.3: ahead at
  epochs 1-7, then flat at ~23.5 while TSC climbs (ep 14-18 -2.5) -> KILLED. Time attention after the stem learns faster
  and plateaus early: overfits the training subjects. htsqf shows the same early lead (ep 3: 26.2). htd4/htdif dropped;
  htd1 (1 block, d64, 0.34M) promoted, htgat moved behind H2.

## Status (14:55)
hbase32_things finished: 28.5 last / 28.6 best (TSC-bn64 original 28.8 / 29.3) -> same-code-path reference is sound.
Running: htsqf (17 ep, -0.2 vs hbase32, last5 -1.2), htd1 (10 ep, -0.15, last5 -0.3), hf80 (2 ep).
Queue: hf20, hp49, hk49, hk13, hsm2 (fp32 rerun), htgat. Watchdog overnight/arch_autokill.py (every 10 min) kills an
h* screen at >= 10 epochs when both its mean and last-5 delta vs hbase32 are <= -1.5, and logs it below.
Survivors run to 40 epochs; a design that only matches TSC solo is still a candidate for the ensemble (grader_sim --fp).
- (14:51) WATCHDOG BUG: arch_autokill.py scaled the (already %) curves by 100, so it killed htsqf (17 ep, really -0.2
  vs hbase32) and htd1 (10 ep, really -0.15) on its first pass. Fixed (+ a scale check on real curves); both
  re-queued from scratch at the front of the queue (dirs removed).
- [autokill 15:23] hp49_things: 10 ep, mean -6.77, last5 -7.88 vs hbase32 -> KILLED
  hp49 curve 8.8 14.7 14.4 16.7 13.7 14.2 14.1 17.6 18.9 19.6: coarser smoothing (0.4 s box, stride 4, 18 steps) loses ~7; with hdp (finer path -2) this brackets the 0.2 s box / stride 2 as the sweet spot. hf80 / hf20 track hbase32 at 10 ep.
- (16:25) htsqf_things (restart): 15 ep, -0.6 vs TSC (+0.05 vs hbase32), last5 -2.7 / -2.2: same early-lead-then-plateau as hta32 -> KILLED. Time attention after the stem: closed for now. hf80 (80 filters) 27 ep +0.1 vs TSC, +0.8 vs hbase32 (last5 +1.5): best H2 so far.

## H4 (17:00) queued: hf160 / hf120 (more filters; hf80 is the only positive axis so far: 36 ep +0.6 vs hbase32)
- (17:20) H2 final (best / last-5 mean): hbase32 28.6 / 28.28 | hf80 28.8 / 28.26 (par: the mid-run +0.6 vanished)
  | hf20 28.0 / 26.98 (-1.3). Filter count 20-80 is flat at the end; hf120 dropped, hf160 kept as the last check.

## H5 (18:10, user question) queued LAST: htfx_things = TimeFreqXTSConv (0.36M)
Never tried before: attention in the Fourier domain or time<->frequency cross-attention (only band-power branches, -9
LOSO, and FilterBankTangentNet, -5). Reasoning: the image information is the phase-locked, latency-sensitive evoked
response (sparse in time, phase-coded across frequencies), so amplitude spectra lose it, while spectra are good
recording/subject descriptors (the fingerprint router). Design: TSConv time tokens (content) cross-attend to 30
frequency-bin tokens of the trial's own log-amplitude spectrum (context), zero-init gate (= TSConv at init).
- [autokill 18:01] htd1_things: 36 ep, mean -1.54, last5 -3.12 vs hbase32 -> KILLED
- (20:40) hk49 (temporal kernel 49 = 0.4 s) final: best 28.0 / last-5 27.44 vs hbase32 28.6 / 28.28 (-0.8): a longer kernel does not help.

## H6 (20:50) queued: hmax (max pooling instead of the box average), hts2 (two stacked 13-tap temporal convs before the pool)
- (21:55) hk13 (temporal kernel 13 = 0.1 s) final: best 28.8 / last 28.1 vs hbase32 28.6 / 28.5: par. Kernel 13-25 flat, 49 worse.
- (21:55) htgat (gated time attention + flatten) at 24 ep: -0.95 vs hbase32, last-5 -1.56 (-1.57 vs TSC): the same
  time-attention plateau as hta32 / htsqf / htd1 -> KILLED by hand. Time attention after the TSConv stem: closed.

## R-series (21:50, user): REVE-style token variants, ~5M params, vs reve_d6w256_things (5.16M, fold 1 28.5)
TokenREVE (hybrid_encoders.py) = br3's conventions (REVE blocks: RMSNorm, no biases, GEGLU, no final norm; learned
positions; per-electrode trial z-score at 120 Hz; br3 at d512 was the best REVE clone, 29.3 on folds 1-2) at d256,
depth 6, REVE recipe (batch 128, lr 1e-3, 10 of 20 epochs, bf16, EMA). Only the tokenizer / readout changes:
| recipe | tokens | params | question |
|---|---|---|---|
| rvir_things | 64 learned virtual channels (linear spatial mix of the z-scored electrodes), whole-window tokens | 5.15M | does a TSConv-style spatial filter in front of REVE beat electrode tokens? (hvs -9 was a small LN/BN model with patches on the TSC recipe) |
| rmix_things | 32 electrode tokens + 5 all-electrode 0.2 s time-patch tokens | 5.34M | does REVE gain from explicit global time context? |
| rtime_things | 20 time-patch tokens (50 ms, all electrodes), time-locked flatten -> 64-d bottleneck head | 5.20M | a pure time-token transformer with TSConv's readout |
| relec_things | 32 electrode tokens (= br3 at d256) | 5.14M | control: separates the tokenizer from br3's conventions |
- (22:35) hmax (max pool instead of box average) 6 ep: -4.2 vs hbase32 -> KILLED. hsm2 (fp32 spatial x2) 9 ep +0.9 vs hbase32. hf160 final: last-5 -0.84 vs hbase32 (par / slightly worse): filters 20-160 closed.
- (23:30) rvir (REVE tokens on 64 learned virtual channels, 5.15M) 7 ep: -1.91 vs reve_d6w256, last-5 -1.36 -> KILLED. A learned spatial mix in front of REVE loses (cf. hvs -9): REVE wants the raw electrodes as tokens. hsm2 37 ep: +0.55 vs hbase32 (last-5 +0.24).
- (23:40) hsm2 (spatial multiplier 2, fp32) final: best 29.0 (ep 36) / last 28.0 vs hbase32 28.6 / 28.5; matched-epoch mean +0.55: par. H-series verdict so far: filters 20-160, kernel 13-25, spatial x2 are flat; pooling coarser / max, learned pool, sepconv, query readout, attention after the stem all lose. TSConv's stem + flatten is a flat local optimum on these axes.
- (00:45) hts2 (two stacked 13-tap temporal convs) 13 ep: -1.21 vs hbase32, last-5 -2.30 -> KILLED. rmix (electrode + 5 global time-patch tokens) 8 ep: +0.39 vs reve_d6w256, last-5 +1.14: first positive R-series signal.
- (01:35) R-series so far (fold 1, matched-epoch delta vs reve_d6w256_things): rmix final 10 ep +0.44 (last-5 +0.98);
  rtime 8 ep +0.29 (last-5 +0.84); relec (control: electrode tokens with the same conventions) 4 ep -0.95. Tokens that
  span all electrodes over a short time patch are what help, not the conventions. Queued the rtime iteration:
  rtime_mean (mean readout instead of the time-locked flatten), rtime12 (10 x 0.1 s patches), rtime3 (40 x 25 ms),
  plus rmix12 (rmix with 10 time patches) running.
- (01:55) rtime final 10 ep: +0.18 vs reve_d6w256 (last-5 +0.64). relec (control) 7 ep: -0.53 (last-5 +0.12) = catching up: the R-series gains (+0.2 to +0.4 mean) are within seed noise so far; relec's final decides. If they stay at par, rtime/rmix are still candidates as ENSEMBLE members (a different token axis than REVE's electrode tokens): needs 4 folds + ens_cache.
- (02:10) R-series fold-1 finals (last-3 mean / best): reve_d6w256 28.27 / 28.5 | relec (control) 28.07 / 28.5 |
  **rmix 28.83 / 29.3** | rtime 28.37 / 29.1 | full REVE-d6 (19M) 28.87 / 29.4. rmix (32 electrode tokens + 5 global
  0.2 s time-patch tokens) = full REVE-d6 at 1/4 of the parameters and +0.76 over its matched control. One fold of 5
  subjects: within noise -> queued rmix folds 2-4 (confirmation, and the 4 folds the ensemble test needs).
  rmix12 (10 x 0.1 s patches) 6 ep -1.58 -> KILLED.
- (02:50) rtime_mean (time tokens, REVE mean readout) 4 ep: -2.15 vs reve_d6w256 -> KILLED: with time tokens the time-locked flatten readout is what works (as for TSConv: hqr query readout -3/-4).
- (03:50) rmix fold 2: last-3 28.57 / last 28.7 vs reve_d6 27.00 / 26.6, reve_scratch 27.87, br3 (d512) 28.73. Folds 1-2 mean last-3: rmix 28.70 vs reve_d6 27.94 (+0.76) at 5.3M vs ~19M. Folds 3-4 running, then enscache_rmix (ensemble test).
- (04:40) rmix 4 folds (cv4_summary, 20 subjects): 28.05 (budget ep 6) / last 27.80 vs reve_d6 27.55 / 26.98 (+0.5 / +0.8), reve_scratch 28.15 / 27.77 (par), sqf_dbn 28.00, tsc_bn64 27.23. FIRST R-series design to hold up on all folds: equal to the best REVE single at ~1/4 of the parameters. Ensemble test (enscache_rmix + ens_search) next.
- (05:15) rmix ENSEMBLE test (ens_cache exact grader_sim --shuffle --fp, 20 subjects): rmix single 36.88 (best
  non-KD single; reve_d6 36.52). Swap REVE-d6 -> rmix in the reference trio: SQF-dbn + TSC-bn64 + rmix = **40.60 vs
  40.05 (+0.55, 15 better / 5 worse, Wilcoxon p 0.10)**; the honest cross-half pick (4 candidate subsets) picks it on
  both halves (40.60). Adding rmix as a 4th member: 40.38 (+0.33, 11/8). Candidate for the user: same 3-member
  structure, a 5.3M model instead of ~19M, +0.55 (not significant on 20 subjects). Not adopted without approval.
- (05:45) rtime patch size (fold 1, last-3): 25 ms (rtime3, ep 9) 27.53 | 50 ms (rtime) 28.37 | 0.1 s (rtime12) 28.07 vs reve_d6w256 28.27: pure time tokens = par at any patch size; the mix of electrode + time tokens (rmix 28.83) is what helps. Queued rmix iterations: rmix40 (3 x 0.33 s time tokens), rmixflat (flatten readout over the 37 tokens).

## H7 (07:00) queued: mixture-of-experts readout (motivated by SUBJECT_LOG D4: fine-tuning the head alone recovers
~90% of the per-person fine-tune headroom, +12.6 of +14.1). TSConvPlus readout='moe': K flatten -> 64 -> 1536 heads,
mixed per trial by softmax(Linear(flattened features)), zero-init gate (= the average of K heads at init). Trials
from different people can be read out by different heads without any person label. hmoe4 (K=4), hmoe8 (K=8), vs
hbase32. If it helps, the same readout goes on rmix.
- (07:15) rmix40 (3 x 0.33 s time tokens, 5.47M) final last-3 28.00 vs rmix 28.83: coarser loses; with rmix12 (finer) -1.6 the 0.2 s patch is the sweet spot. rmixflat 8 ep last-3 28.53 (last 29.3).
- (07:25) rmixflat (flatten readout over 37 tokens, 5.66M) final last-3 28.60 vs rmix 28.83: par; flatten adds nothing once electrode tokens are present. Queued rmixd8 (rmix at d224 x depth 8, ~5.2M; depth > width at 5M in A27).
- (08:15) H7 result: hmoe4 19 ep -0.91 vs hbase32 (last-5 -2.30; +0.9 at 7 ep), hmoe8 11 ep +0.75 (last-5 -1.58) ->
  both KILLED. Early lead, then plateau: K trial-gated readouts fit the TRAINING people's readouts instead of learning
  one that transfers. The per-person readout headroom (D4) is not reachable by giving the model more readouts.
- (08:20) queued rmix_s2 (seed 7 vs 33) on 4 folds + enscache_rmix_s2: is the +0.55 swap (40.60 vs 40.05) robust to the seed, and do two rmix seeds add as members? rmixd8 and htfx started 08:07.
- (08:45) htfx (time tokens cross-attend to the trial's spectrum tokens) 14 ep: -1.00 vs hbase32, last-5 -1.94 -> KILLED: same early-par-then-plateau as every attention-after-stem design.
- (09:55) rmixd8 (rmix at d224 x depth 8, 5.36M) fold 1: last 31.0, last-3 30.23 vs rmix 28.83 (+1.4), reve_d6 (19M) 28.87, reve_d6w256 28.27: best REVE-family fold-1 result so far. Depth over width again (A27). Queued folds 2-4 + enscache_rmixd8 ahead of rmix_s2 folds 3-4.
- (10:05) rmix_s2 (seed 7) fold 1: last-3 28.43 vs rmix (seed 33) 28.83: seed noise ~0.4 on fold 1; rmixd8 30.23 is +1.4 to +1.8 above both seeds.
- (11:20) rmixd8 fold 2 final: last-3 28.23 vs rmix 28.57 / rmix_s2 28.57 (reve_d6 27.00): the fold-1 +1.4 does NOT repeat (it led +1.0 at ep 6, then lost it). Folds 1-2 mean: rmixd8 29.23 vs rmix 28.70 (+0.5, within fold noise). Folds 3-4 + ensemble cache decide.
- (12:15) rmixd8 fold 3: 27.67 vs rmix 28.10 (reve_d6 26.67). Folds 1-3 mean last-3: rmixd8 28.71 vs rmix 28.50 (+0.2) vs reve_d6 27.51: depth 8 = depth 6 at equal size; the fold-1 +1.4 was fold noise. Both rmix depths beat REVE-d6 by ~1 on every fold.
- (12:30) rmixd8 4 folds (last-3 mean): 28.04 vs rmix 27.91 vs reve_d6 27.13 (folds: 30.23/28.23/27.67/26.03 vs 28.83/28.57/28.10/26.13). Depth 8 = depth 6 at equal size; both +0.8 over REVE-d6 on every fold. Ensemble cache running.
- (12:45) rmixd8 as an ensemble member (ens_cache, fp, 20 subjects): single 36.35 (rmix 36.88); swap for REVE-d6: 40.55 (+0.50, 12/4) vs rmix swap 40.60 (+0.55, 15/5); rmix + rmixd8 + SQF + TSC 40.38. Depth 8 = depth 6 in the ensemble too; the 5M rmix swap (+0.5) stands, two rmix depths together add nothing.
- (14:30) rmixd12 (d192 x 11, 5.3M) fold 1: last-3 29.33 (last 29.6) vs rmix 28.83 / rmixd8 30.23 (rmixd8 = rmix over 4 folds). Depth at equal size (6 / 8 / 11) is within fold noise: axis closed. Parameter-count ladder (10M / 10.6M / 13M) running.
- (14:55) SCALING ladder point 1: rmix_d12w256 (~10M, d256 x 12) fold 1 last-3 28.33 (best 29.4) vs rmix 5.3M 28.83: 2x the parameters = no gain (-0.5, within fold noise). Supports the user's 'capacity is not the binding constraint' reading. d8w320 (10.6M) and d16w256 (13M) next.
- (16:40) SCALING ladder point 2: rmix_d8w320 (~10.6M, d320 x 8) fold 1 last-3 29.20 (best 29.5) vs rmix 5.3M 28.83, rmix_d12w256 10M 28.33, rmixd8 5.4M 30.23 (= rmix over 4 folds). Fold 1 spans 28.3-30.2 at 5-10.6M with no size trend: capacity is not the lever. d16w256 (13M) is the last point.
- (19:30) SCALING ladder point 3 (last): rmix_d16w256 (~13M, d256 x 16) fold 1 last-3 28.53 (best 28.7).
  SCALING VERDICT (fold 1, last-3): 5.3M rmix 28.83 | 5.4M rmixd8 30.23 (4-fold = rmix) | 5.3M rmixd12 29.33 |
  10M d12w256 28.33 | 10.6M d8w320 29.20 | 13M d16w256 28.53 | 19M reve_d6 28.87. No trend with size (5-19M all within
  28.3-30.2 = fold noise): capacity is not the binding constraint at 25 training people. rmix (5.3M) stays the
  candidate (ensemble swap 40.60 vs 40.05); the rmix architecture search is settled.

## R-series round 3 (19:45): readout and time-token layout around rmix (size settled: not the lever)
| recipe | change vs rmix | question |
|---|---|---|
| rmix_split | readout = mean(electrode tokens) + time-locked flatten of the 5 time tokens only | rmixflat flattened all 37 tokens; time-locking should only apply to time tokens |
| rmix_ov | 9 overlapping 0.2 s time tokens (stride 0.1 s) | 0.1 s (rmix12) and 0.33 s (rmix40) non-overlapping both lost: keep the 0.2 s field, finer placement |
| rmix_ms | 5 x 0.2 s + 10 x 0.1 s time tokens | multi-scale time context |
Fold 1 vs rmix 28.83; fold noise ~0.5-1, so only >= +1 earns folds 2-4.
- (19:35) incident: a second align_dyn worker srun'd into job 61644 (whose original worker was still alive) relaunched rmix_split; its train.py rmtree'd the live run's files. Duplicate killed, corrupted run killed, rmix_split relaunched cleanly 19:29 by the original worker. rmix_ov (516) and rmix_ms (513, job 61953) unaffected.
- (22:10) R-series round 3 results (fold 1, last-3 vs rmix 28.83): rmix_split 27.40 (-1.43; mean -1.75) LOSES;
  rmix_ms 28.60 (-0.23) PAR; rmix_ov died at epoch 6 when the session restart cancelled its srun step (5 ep: -2.00,
  last-3 -2.47, already past the cut rule) -> counted as CUT, not rerun. Verdict: the token layout / readout around
  rmix is exhausted (patch size, overlap, multi-scale, readout, depth, size all flat or worse).
  GPUs now: 516 (interactive) + 513 (61953 holder); 61644 completed 21:03 (its worker found the queue empty).

## R-series round 4 (22:15): REVE x TSConv hybrid tokenizer
rmix's time tokens are linear patches of the raw trace. Round 4 makes them TSConv's stem output instead (temporal
conv 25 taps -> 0.2 s box pool stride 2 -> 40 spatial filters; the validated TSConv front end), one token per stem
step: rmix_tsc (36 tokens of 40 filters), rmix_tsc9 (avg-pooled x4 -> 9 tokens). Electrode tokens unchanged. Fold 1
vs rmix 28.83; >= +1 earns folds 2-4.
- (00:15) rmix_tsc (36 TSConv-stem time tokens) 6 ep: -1.50 vs rmix (last-3 -2.13) -> KILLED. rmix_tsc9 (9 pooled stem tokens) 6 ep: -0.58 (last-3 -1.00), continues.
- (00:25) queued rmix_sqf (REVE x SQF): rmix + Squeezeformer conv module (LayerNorm -> depthwise conv k3 -> GELU -> pointwise, zero-init) on the 5 time tokens after each of the 6 blocks; uses the idle 516 GPU while rmix_tsc9 finishes on 513.
- (00:40) rmix_tsc9 final fold 1: last-3 28.43 (last 28.3, best 28.9) vs rmix 28.83 (-0.40, mean -0.57): PAR. TSConv-stem time tokens = raw linear patches when pooled to 9 tokens, worse at 36 (rmix_tsc -1.5). REVE x TSConv: no gain. rmix_sqf (REVE x SQF) running.
- (02:30) rmix_sqf (REVE x SQF: Squeezeformer conv module on the time tokens after each block) final fold 1: see
  below; with rmix_tsc9 (REVE x TSConv) par and rmix_tsc -1.5, round 4 adds nothing over rmix.
  rmix_sqf last-3 28.33 (last 28.6) vs rmix 28.83 (-0.50, matched-epoch mean -1.3): PAR at the end, behind all run.
  ROUND 4 VERDICT: REVE x TSConv (stem time tokens) and REVE x SQF (conv module on time tokens) both add nothing.

## R-series final summary (2026-10-17)
Winner: rmix (TokenREVE, 32 electrode tokens + 5 all-electrode 0.2 s time-patch tokens, REVE blocks d256 x 6,
5.3M): 4 folds 28.05 vs REVE-d6 27.55; ensemble swap for REVE-d6 40.60 vs 40.05 (+0.55, 15/5, p .10) = candidate
for the user. Everything around it is flat or worse at fold-1 resolution (~+/-0.5-1): tokens (virtual -1.9,
electrode-only control par, time-only par, stem time tokens par / -1.5), patch layout (0.1 s -1.6, 0.33 s -0.8,
overlap -2.0, multi-scale par), readouts (flatten par, split -1.4, time-mean -2.2), depth at equal size (6 / 8 / 11
par over 4 folds), size (5 -> 10 / 10.6 / 13M no gain), SQF conv module par. Architecture search CLOSED.

## R-series round 5 (03:00): the attention mechanism (reopened per the standing goal; every R run so far used
plain full softmax attention over all 37 tokens). _SwitchBlock gains an optional boolean attn_mask and output gate
(defaults off: BridgeTransformer / all earlier runs unchanged).
| recipe | change vs rmix | question |
|---|---|---|
| rmix_bip | bipartite attention: a token attends to itself + the other type only (electrode <-> time) | is within-type attention needed, or is the gain all cross-type exchange? |
| rmix_gate | gated attention: attention output x sigmoid(W x), per token and channel | attention-sink / over-mixing fix (Qiu et al. 2025) |
| rmix_reg | + 4 learned register tokens | a shared workspace token set |
Fold 1 vs rmix 28.83; >= +1 earns folds 2-4. Starting architectures for the record: H0 = FilterAttnTSConv,
TimeAttnFlatTSConv, VirtualSensorFormer, DynSpatialTSConv, DualPathTSConv (+ TSConvPlus knobs), then R0 = rvir, rmix, rtime.
- (03:55) round 5: rmix_bip (bipartite attention) 7 ep -3.93 vs rmix (last-3 -4.33) -> KILLED: within-type attention
  is essential (electrode tokens must attend to other electrodes). rmix_gate (gated attention, 5.73M) 7 ep -1.33
  (last-3 -1.47) -> KILLED: cannot reach the +1 bar in 3 epochs. rmix_reg (4 registers) next.
- (05:30) rmix_reg (+4 learned register tokens) fold 1 final: last-3 29.80 (last 30.1, best 30.5) vs rmix 28.83 /
  rmix_s2 28.43 (+0.97 / +1.37). At the +1 bar; rmixd8 led fold 1 by more (30.23) and was par over 4 folds, so
  folds 2-4 + ensemble cache queued before any claim.
- (06:55) rmix_reg fold 2: last-3 28.10 vs rmix 28.57 / rmix_s2 28.57 (-0.47): the fold-1 +1.0 does NOT repeat (same pattern as rmixd8). Folds 1-2 mean 28.95 vs rmix 28.70 (+0.25). Folds 3-4 + ensemble cache continue.
- (07:10) rmix_reg fold 3: last-3 28.17 vs rmix 28.10 (+0.07). Folds 1-3 mean 28.69 vs rmix 28.50 (+0.19): registers = rmix within noise. Fold 4 + ensemble cache continue.
- (08:20) rmix_reg 4 folds (last-3): 29.80 / 28.10 / 28.17 / 26.40 = 28.12 vs rmix 27.91 (+0.21) vs reve_d6 27.13: registers = rmix within noise (like rmixd8, +0.13). Ensemble cache running for the member test.
- (08:45) rmix_reg as an ensemble member (ens_cache, fp, 20 subjects): single 37.35 (rmix 36.88, best non-KD single);
  swap for REVE-d6 40.42 (+0.38, 12/7) vs rmix swap 40.60 (+0.55, 15/5); both rmix + rmix_reg 40.15. Registers do
  not improve the ensemble. ROUND 5 VERDICT: bipartite -3.9 (within-type attention essential), gated -1.3, registers
  = rmix (+0.21 solo over 4 folds, no ensemble gain). rmix remains the winner; R-series closed (rounds 1-5).

## R-series round 6 (09:00): two untested axes of the brief on rmix
| recipe | change vs rmix | question |
|---|---|---|
| rmix_ep | each electrode split into 5 x 0.2 s patch tokens (160 tokens, REVE's fine tokenization, positions = electrode + patch), + the 5 global time tokens | attention along time WITHIN electrodes: can electrode c at t1 relate to electrode c' at t2? |
| rmix_tffn | split paths: attention joint over all 37 tokens, but separate FFN experts for electrode vs time tokens (d224 x 5, ~5M) | do the two token types want different nonlinear processing? |
Fold 1 vs rmix 28.83; >= +1 earns folds 2-4 (and after rmixd8 / rmix_reg, a fold-1 lead is not a claim).
- (10:10) rmix_ep (160 electrode x 0.2 s patch tokens + 5 time tokens) 4 ep: 9.6 / 18.6 / 20.1 / 21.7 = -7.1 vs rmix -> KILLED (also 3x slower per epoch). Whole-trace electrode tokens are essential: splitting an electrode's trace into patches destroys the per-electrode time-locked template the embedding learns. rmix_tffn 6 ep -1.03 continues.
- (11:00) rmix_tffn (split FFN experts per token type, d224 x 5, 5.56M) fold 1 final: last-3 28.73 (best 29.7) vs
  rmix 28.83 (-0.10): PAR. ROUND 6 VERDICT: electrode x patch tokens -7.1 (whole-trace electrode tokens essential);
  type-specific FFN experts par. rmix remains the winner.

## R-series closing summary after rounds 1-6 (2026-10-17 11:00)
rmix (32 whole-trace electrode tokens + 5 all-electrode 0.2 s time tokens, joint full softmax attention, REVE blocks
d256 x 6, mean readout, 5.3M) is a local optimum along every axis tried: tokenizer (virtual -1.9, electrode-only par,
time-only par, TSConv-stem par / -1.5, electrode x patch -7.1), time-token layout (0.1 s -1.6, 0.33 s -0.8, overlap
-2.0, multi-scale par), readout (flatten par, split -1.4, time-mean -2.2), depth at equal size (6 / 8 / 11 par, 4-fold),
size (5 -> 13M flat), SQF conv module par, attention (bipartite -3.9, gated -1.3, registers +0.2 / 4 folds, joint with
split FFN par). Two fold-1 leads (rmixd8 +1.4, rmix_reg +1.0) did not survive folds 2-4: single-fold leads of ~+1
are noise at this resolution.

## R-series round 7 (11:20): SQF's readout on rmix, 2-fold screen
rmix_apool: readout = Squeezeformer's attention-weighted mean + std over all 37 tokens (std carries the spread across
tokens; mean / flatten / split were tried, statistics pooling never). Screened on folds 1 + 2 in parallel (10
subjects; two fold-1 leads were noise), vs rmix 28.83 / 28.57.
- (11:50) round 7: rmix_apool (Squeezeformer attention-weighted mean + std readout) fold 1 final last-3 26.80 vs rmix
  28.83 (-2.03); fold 2 at 8 ep -1.07 (last-3) -> KILLED. Statistics pooling LOSES: rmix's plain token mean is the best
  readout (vs flatten par, split -1.4, time-mean -2.2, attention-stats -2.0 / -1.1).

## Round 8 (12:20): the SQF family gets rmix's ingredient
Every hybrid so far was built around REVE; the SQF ensemble member (sqf_dbn, 28.00) was never varied here.
sqfmix (SqfMix, hybrid_encoders.py): sqf_dbn unchanged (TSConv stem -> 36 time tokens -> 3 Squeezeformer blocks,
per-dataset BN, attention-stat pooling, same recipe: batch 1024, 40 ep, fp32) + 32 whole-trace electrode tokens in
the sequence; attention + FFN over all 68 tokens, conformer conv over the time tokens only. 2-fold screen (1 + 2) vs
sqf_dbn_things. If it helps, it is an ensemble-member upgrade (SQF-dbn is in the reference trio).
- (14:15) round 8: sqfmix (SQF-dbn + 32 electrode tokens) final, last-3 of 40 epochs: fold 1 23.90 vs sqf_dbn 27.50
  (-3.6), fold 2 26.17 (ep 39) vs 27.50 (-1.3). Led by +0.5 to +1.8 for the first ~25 epochs, then SQF-dbn kept
  improving and sqfmix did not: the electrode tokens speed up early learning but end worse (overfit the training
  people's electrode signatures?). LOSES. rmix's ingredient does not transfer to the Squeezeformer.

## Round 9 (14:25): attention across filters, REVE-side
rmix_filt: 32 whole-trace electrode tokens + 40 FILTER tokens (each TSConv stem filter's 36-step time course after
the spatial conv, Linear 36 -> d), joint REVE attention over 72 tokens. hfa (attention across filters after the
TSConv stem, TSConv readout) lost 1-2; here the filters meet REVE's electrode tokens instead. 2-fold screen vs rmix.
- (15:40) round 9: rmix_filt (32 electrode tokens + 40 TSConv-stem filter tokens) final, last-3: fold 1 28.57 vs rmix
  28.83 (-0.27), fold 2 28.37 vs 28.57 (-0.20). PAR (slow start, converges to rmix). Attention across learned filters
  is an equally good "global" token set as the 5 linear time patches, not a better one.

## Search status after 9 rounds (2026-10-17 15:40)
Families covered: TSConv (H-series), REVE (R rounds 1-7, 9), SQF (round 8). Every axis in the brief has been probed on
the current best: convolutions (filters, kernels, pooling, stacked, learned pool, sepconv, stem-as-tokenizer, SQF conv
module), attention placement and direction (filters, time, electrodes, filters x electrodes, within-electrode time),
attention kinds (full, bipartite, gated, registers, cross-attention to spectra), split paths (dual slow/fast, MoE
readouts, type-specific FFNs), readouts (flatten, query, mean, split, attention-stats), size (0.2M-13M). rmix holds.

## The five starting architectures (H0, 2026-10-15), named, with outcomes
| # | recipe / class | idea | result (fold 1 vs TSC-bn64) |
|---|---|---|---|
| 1 | hfa_things / FilterAttnTSConv | attention ACROSS the 40 TSConv filters after the stem | -2.3 -> killed |
| 2 | hta_things / TimeAttnFlatTSConv | attention over the 36 time tokens + TSConv's time-locked flatten | plateau ~-2 -> killed |
| 3 | hvs_things / VirtualSensorFormer | REVE tokenisation on 16 learned virtual sensors | -10 -> killed |
| 4 | hds_things / DynSpatialTSConv | trial-adaptive spatial filters (queries attend over electrodes) | -2.3 -> killed |
| 5 | hdp_things / DualPathTSConv | TSConv split into slow + fast paths | -2.9 -> killed |
Iteration lines from them: TSConvPlus knobs (H1-H7), then the REVE-side R-series (rmix = the winner) and SQF (round 8).

## Round 10 (16:00)
| recipe | change vs rmix | question |
|---|---|---|
| rmix_cemb | each electrode token = TSConv temporal filters on its trace (Conv1d 1 -> 16, 25 taps, 0.2 s box pool, flatten -> d) instead of one Linear(120 -> d) | does a filter-bank embedding of each electrode beat the single time-locked linear template? |
| rmix_union | 32 electrode + 5 time-patch + 40 stem filter tokens (77 tokens) | time tokens (rmix) and filter tokens (rmix_filt) tie: are they complementary? |
Fold 1 first, vs rmix 28.83; par-or-better gets fold 2.
- (16:05) STOPPED by the user: round 10 (rmix_cemb, rmix_union) dequeued before any run started; GPU job 61953 cancelled.
