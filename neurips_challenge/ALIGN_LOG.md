# EEG-image alignment levers: experiment log

Goal (user, 2026-10-12 night): improve the EEG->image alignment of the base encoders (TSConv-dbn40, REVE, Squeezeformer-dbn,
all + THINGS) by ~+3 points on average on cv4 (sealed rule, subject-aggregated top-5).

## Constraints from first principles
- The grader compares our 1536-d output with the FIXED DINOv2-giant final features of unseen images (cosine).
  Nothing can be applied to the gallery, so image-side tricks (other layers, blur, learned target maps) can only act
  as training-time targets or inductive biases. A learned LINEAR image-side map is foldable into the EEG head only
  if the gallery is not re-normalised after it, i.e. it adds no expressivity.
- EEG side: single trials at ~SNR << 1, 32 channels, 1 s; it carries a coarse, foveal, mostly mid-level image signal
  (early components ~ low-level contrast/colour/layout, later ~ category). The DINOv2 final CLS carries much that the
  EEG cannot see -> fitting all 1536 dims spends capacity on unpredictable directions.
- Current loss: EEG->image InfoNCE only (asymmetric), image targets L2-normalised, EEG outputs unnormalised (their
  norm = learned temperature), logit scale fixed at 1. Negatives: in-batch only (TSC/SQF 1023, REVE 127).

## Already dead on Alljoined cv4 (alljoined-cv4-protocol memory)
latency / montage jitter, repeat-grouped batches, repeat-mean loss (TSC), MixStyle, mixup (REVE-6 -0.7), soft
targets (-1 FBTN), PCA-subspace head (-7, uncentred), neighbour heads (TSC).

## Literature (2025-26)
- UBP (CVPR 2025, arXiv 2503.04207): uncertainty-aware blur of the image targets (system gap: foveal, fast
  presentation); +13.7 top-1 intra-subject THINGS-EEG.
- "Aligning What EEG Can See" (arXiv 2603.07077): final-layer targets mismatch; align with intermediate "EEG-visible"
  layers and fuse levels.
- SAMGA (arXiv 2604.17782): multi-layer visual target + coarse (MMD) -> fine alignment; inter-subject 34.4 top-1.
- Structured visual targets (arXiv 2609.36971): learned multi-view image target + MMD; LOSO THINGS-EEG2 35.3 top-1.
  Their image side is learned and the gallery passes through it (not possible for us at test time).

## Experiments
| id | lever | recipe | folds | result |
|---|---|---|---|---|
| A1 | global negatives: softmax over all 16540 frozen training-image targets instead of in-batch | reve_d6_gneg_things | 1 | **NEGATIVE**: matched epochs e1-e4 -1.2 / -2.2 / -2.9 / -1.8 vs reve_d6_things; stopped at e4 |
| A2 | low-rank bottleneck: encoder -> 64-d code -> linear lift to DINOv2 (image side identity) | reve_d6_bn64_things | 1 | **NEGATIVE**: e1-e3 -1.3 / -3.5 / -3.6; stopped |
| A3 | global negatives on TSConv | tsc_dbn40_gneg_things | 1-2 | queued (shared) |
| A4 | bottleneck 64 on TSConv | tsc_dbn40_bn64_things | 1-2 | queued |
| A5 | multi-level target: training-only head aligned to DINOv2 layer-16 mean tokens (lambda .5, 512-d) | tsc_dbn40_l16_things, reve_d6_l16_things | 1-2 | queued |
| A6 | image-view targets (UBP-like): DINOv2 of original / blur / low-res / mosaic / noise, one at random | tsc_dbn40_iaug_things, reve_d6_iaug_things | 1-2 | queued |
| A7 | relational (RKD/CKA-like) EEG-EEG vs image-image structure loss, lambda .5 | tsc_dbn40_rkd_things | 1-2 | queued |
| A8 | global negatives on Squeezeformer | sqf_dbn_gneg_things | 1-2 | queued |
Note: the available blurred DINOv2 targets are very close to the originals (mean cosine 0.986), so A6 is a mild
version of UBP. Queue: neurips_challenge/overnight/align_queue_1012.sh (2 sbatch workers after the THINGS-norm
screen + the interactive GPU after A1/A2). sqf_dbn_coral_things parked (claim by job 59605) to free GPU time.

### Notes (2026-10-12 ~23:00)
- **The Track-1 target is NOT the final layer**: neuralset HuggingFaceImage(layers=0.67, token_aggregation=mean,
  imsize=518) -> the HF DINOv2 processor cancels imsize (resize 256, centre-crop 224), hidden state
  int(.6667*40) = 26 of 40, mean over all tokens. Reproduced exactly (cosine 1.0000) by
  build_dinov2_multilayer.py. Neighbouring states: 25 .989, 27 .990, 20 .63, 40 .54, 8 .34 -> the target is
  already a late-intermediate layer; the earlier dinov2-giant_layer16_mean dir used other preprocessing.
- New targets extracted with the exact grader preprocessing: layers 8 / 13 / 20 / 33 / 40, and the layer-26 target
  of Gaussian-blurred images (sigma 4 / 8 / 16 px at the 518 viewing size; the old GaussianBlur aug was mild,
  cosine .986 to the original).
- Recipes replaced: l16 -> ml (training-only heads on layers 8/13/20/40), iaug -> blur (random sigma 0/4/8/16).
- User request: alternative losses. Added: --siglip (pairwise sigmoid, learnable bias), in-batch SoftCLIP
  (--soft_target_tau .02), --global_symmetric (image->EEG term, anti-hub), global soft targets (--global_soft_tau)
  and concept smoothing (--global_concept_eps).
- A1 early: reve_d6_gneg_things fold 1 epochs 1-2: -1.2 / -2.2 vs reve_d6_things. Hypothesis: the full softmax
  makes the 9 same-concept near-duplicates hard negatives that EEG cannot separate -> A3b concept smoothing.

### 2026-10-12 23:20
- THINGS input normalisations (folds 1-2, best-epoch delta vs reve_d6_things): CORAL recolouring -0.55
  (per-epoch -2.3..+0.7), common average reference +0.35 (-1.4..+3.3) -> both neutral.
- A9 added: repetition-average augmentation (--rep_avg_prob .3): 30% of training trials replaced by the mean of
  2-4 of the same subject's valid repeats of the image (higher-SNR view, same target; zero-padded repeats skipped).
- GPUs: 513/514 run tsc_dbn40_bn64_things folds 1/2 (queue order before the edit); 516 runs tsc_dbn40_ml_things.
- Gotcha: a claim owned by the worker's OWN job id is taken over (alljoined_run.sh ME == owner); park with the id of
  another long-running job.

### 2026-10-12 23:55
- **A4 tsc_dbn40_bn64_things looks POSITIVE early**: fold 1 matched epochs e1-e13 +3.3 +2.8 +0.7 +4.0 +0.3 +4.7
  +2.0 +5.7 +2.9 +3.6 +3.9 +4.7 +5.6; fold 2 e1-e4 +2.2 +0.7 -2.2 +1.5. Unlike REVE (A2, negative): TSConv's head
  is a very wide linear map (flattened conv features -> 1536), a low-rank code regularises it. Sweep queued:
  bn32, bn128, and sqf_dbn_bn64_things.
- Queueing: the 3 running workers parsed their recipe lists at start (edits do not reach them). New editable queue
  neurips_challenge/overnight/align_queue.txt + align_dyn.sh (re-read before every run), chained after each
  worker (aligndyn 60371/60372, and appended to the 516 script). Parked (claim owner 60326) to make room:
  tsc_dbn40_gsoft/gnegc/softclip/rkd; they sit at the end of align_queue.txt -> remove their .claim dirs when due.

### 2026-10-13 00:45
- tsc_dbn40_ml_things crashed at start on 516 (folds 1-2): the new layers' image_test.npy were written ~45 min
  after image_train.npy and the runs started in between. Claims cleared, requeued.
- A9 tsc_dbn40_ravg_things fold 1 early: e1-e6 +3.3 +0.6 +3.1 +3.5 +3.8 +3.3. Two different levers both ~+3 at
  matched early epochs on fold 1 -> suspect a weak baseline curve; **noise control queued first:
  tsc_dbn40_s2_things (seed 7 replicate of the baseline)**.
- A4 bn64 TSC fold 1 through e28: +2..+5.7 every epoch, best 29.30 vs baseline best 26.10; fold 2 through e19
  mostly +1..+3.5 (best so far 23.6 vs 24.4 at baseline e31).
- Scheduling: align_handover.sh cancels each stale-list worker as soon as its current run is scored, so the GPU
  passes to align_dyn.sh (priority list in align_queue.txt; dead claims are taken over).

### 2026-10-13 01:45
- **Noise control** tsc_dbn40_s2_things (seed 7) vs tsc_dbn40_things (seed 33), matched epochs: fold 1 e1-e16
  +0.7 -0.1 -1.1 +1.9 +2.4 +0.1 -2.8 +0.4 +1.2 -1.9 -0.6 +1.8 +1.3 +1.7 +2.1 -2.9 (mean +0.3, sd ~1.7); fold 2 e1-e7
  +1.7 +2.3 -2.4 +2.9 +1.1 -2.5 +0.7. => single matched epochs swing +-3; only sustained shifts count.
- **A4 tsc_dbn40_bn64_things DONE (folds 1-2)**: best epoch 29.30 vs 26.10 (fold 1, +3.2), 26.00 vs 24.40 (fold 2,
  +1.6) -> **+2.40**; fold 1 matched epochs average +3.6 over 28 epochs (beyond the control's +0.3 +- 1.7).
- A9 tsc_dbn40_ravg_things fold 1 through e27: +1 on average early, fading to ~0 -> within noise so far.
- Added: tsc_dbn40_bn64do_things (dropout .3 on the 64-d code), reve_d6_bn256_things (wider code for REVE).

### 2026-10-13 02:20
- A9 ravg TSC fold 1 final-ish: best 25.6 (e32) vs 26.1 -> DEAD (within the seed-control noise).
- A10 (new, first principles from A1): **false-negative elimination** --mask_concept_neg: same-concept,
  different-image pairs are removed from the in-batch softmax (THINGS train: 1654 concepts x 10 images; in a
  1024 batch ~49% of trials have a concept-mate, which EEG cannot tell apart; A1's full softmax made that worse).
  Recipes tsc_dbn40_mcn_things, tsc_dbn40_bn64mcn_things (+ bottleneck), reve_d6_mcn_things; queued first.

### 2026-10-13 10:20 -- overnight results (folds 1-2, cv4_summary paired vs tsc_dbn40_things, full trials, budget epoch)
| recipe | lever | delta | better/10 | Wilcoxon p | verdict |
|---|---|---|---|---|---|
| tsc_dbn40_s2_things | seed replicate (noise control) | -0.15 | 5 | 0.83 | control: noise ~0 |
| **tsc_dbn40_bn64_things** | **64-d bottleneck head** | **+2.40** | 8 | 0.062 | **promote to folds 3-4** |
| tsc_dbn40_bn64mcn_things | bottleneck + concept-negative masking | +2.30 | 8 | 0.016 | = bottleneck alone |
| tsc_dbn40_mcn_things | concept-negative masking | +0.50 | 6 | 0.42 | noise |
| tsc_dbn40_ravg_things | repetition-average augmentation (fold 1) | -0.50 | 2/5 | - | dead |
| tsc_dbn40_blur_things | UBP-like blurred targets (fold 1, per-epoch half trials) | -0.9 | - | - | dead |
| reve_d6_gneg_things / reve_d6_bn64_things | global negatives / 64-d bottleneck on REVE | -2..-3.6 per epoch | - | - | stopped, dead |
The interactive job 59605 ended ~10:08 (its GPU worker died with it, mid tsc_dbn40_ml_things). New interactive
job 60527 on 516: align_dyn worker restarted detached (setsid nohup srun --jobid=60527 --overlap).
Queue now: bn64 TSC folds 3-4, sqf_dbn_bn64, reve_d6_bn256, TSC bn128 / bn32, ml, bn64do, siglip, trunc.

### 2026-10-13 10:50
- A11 REVE flatten head (REVE's own query-pool + all tokens flattened -> ~66k-wide linear) with / without the 64-d
  bottleneck: the TSConv lesson says a bottleneck helps a WIDE head; REVE's default mean head is only 512 -> 1536.
- A12 NeuroCLIP-style soft targets (arXiv 2511.09250: removing them costs 4.4 inter-subject top-1): target = .7
  one-hot + .3 softmax(image-image cos / .05) (--soft_target_tau .05 --soft_mix_beta .3); the earlier soft-target
  attempts used the pure softmax (aggressive), and tau .02 would be ~one-hot (self mass .99) -> softclip dropped.

### 2026-10-13 11:30
- **A4 replicates on folds 3-4** (per-epoch, half trials): fold 3 e1-e23 mostly +1.5..+5.5 (best 27.2 vs 25.7),
  fold 4 e1-e21 mostly +1..+4.6 (best 25.7 vs 24.2). 4-fold best-epoch mean so far +1.95 (folds 3-4 still running).
- Head anatomy explains where the bottleneck helps: TSConv flattens conv features (~1.4k) straight into the
  1536-d head (wide, 2M weights) -> a 64-d code regularises it. REVE (mean-pooled 512 -> 1536) and Squeezeformer
  (attention-pooled mean+std -> 512 trunk -> 1536) already squeeze through narrow pooled vectors -> little to gain
  (REVE bn64 -3; SQF bn64 fold 1 mixed, ~+0.6 per epoch so far). Hence A11 (REVE's WIDE flatten head + bottleneck).
- CORRECTION (head shapes from the checkpoints): TSConv flattens to 1440 (40 filters x 36 steps); its head is
  Linear 1440->1536 + residual GELU Linear 1536->1536 + LayerNorm (~4.6M weights). bn64 makes it 1440->64, a 64x64
  residual block + LayerNorm(64), then 64->1536 (~0.2M). SQF: attention-pooled mean+std 320 -> 512 (+512 residual,
  LN) -> 1536; REVE: mean-pooled 512 -> 1536. So the TSConv gain is about head CAPACITY (a 4.6M full-rank head incl.
  a 1536x1536 residual MLP), not input width; SQF/REVE heads are already small/narrow.

## RESULT A4 (2026-10-13 12:05): tsc_dbn40_bn64_things ADOPTED
cv4_summary.py --ref tsc_dbn40_things, all 4 folds, full trials, budget epoch: **27.23 vs 24.68 = +2.55 over 20
subjects, 16/20 better, Wilcoxon p 0.004 -> ADOPT**. Budget epoch = 40 = last (still rising?) -> tsc_dbn60_bn64_things
queued. Mechanism: replaces TSConv's ~4.6M-weight full-rank head with a 64-d code (~0.2M).
- 12:40: tsc_dbn60_bn64_things (60-epoch bottleneck) dropped at the user's request (the bn64 curve is flat e30-e40: 26.78 / 27.05 / 27.23).

### A13 (2026-10-13 12:40): head anatomy -- what in the bottleneck helps?
Per-fold budget epochs (sealed per-epoch, half trials): tsc_dbn40_things best 35/31/35/19, bn64 28/32/35/40; no
early stopping (fixed 40-epoch one-cycle, budget epoch = best fold-mean epoch: 35 baseline, 40 bn64; bn64 flat
26.78 / 27.05 / 27.23 at e30 / 35 / 40).
TSConv's head ends with LayerNorm(1536) = the OUTPUT, so every trial's output norm is ~constant; with
unnormalised-EEG InfoNCE and repeat-averaging in the grader, that norm is the learned temperature / per-trial
confidence. bn64 moves the LN onto the 64-d code and adds a free 64->1536 lift. SQF (LN(512) -> Linear) and REVE
(Linear) already have a free output norm. Ablations (TSC folds 1-2 unless noted):
- tsc_dbn40_noln_things: full-width head, final LN removed (--tsconv_head_no_ln, new) -> free norm, full capacity
- tsc_dbn40_tb512_things / tb256: 512 / 256 trunk + LN + free lift (existing --tsconv_backbone_dim)
- tsc_dbn40_bn64noln_things: bottleneck without LN on the code
- reve_d6_mlphead_things: REVE mean tokens -> Linear, residual GELU MLP, LN, free lift (--fm_head mlp, new)
- 13:05: sqf_dbn_bn64_things STOPPED: fold 1 complete, late epochs e31-e40 all -1.6..-3.6 (budget region), fold 2 early +1.5 then likely the same -> the bottleneck does not help SQF's already-narrow head.
- 13:12: tsc_dbn40_noln_things STOPPED: fold 1 e1-e10 mean -1.66, last5 -3.86 -> freeing the output norm is NOT the mechanism; bn64's gain = head capacity.
- 13:55: reve_d6_mlphead_things STOPPED: fold 1 e1-e4 mean -8.55 (REVE's head must stay a single linear map; a residual-MLP+LN trunk collapses it). tsc_dbn40_tb512_things +2.12 / +1.46 over 14 epochs (folds 1/2) -> capacity story holds.
- 15:15: **tsc_dbn40_tb512_things +1.90 (8/10, p .088)** vs bn64 +2.40 on the same folds: a smaller head trunk
  helps more (capacity story; 512 < 1536 helps, 64 helps more) -> bn128 / bn32 running next.
  reve_d6_flatbn64_things fold 1 e1-e8: -2.0..+0.9 (mean -0.7): neutral so far. REVE rejects both a smaller (bn64,
  -3) and a bigger head (mlp, -8.6). Loss-side NeuroCLIP soft targets moved up for all three encoders
  (sqf_dbn_smix_things added).
- 15:40: reve_d6_flatbn64_things STOPPED: fold 1 full (10 ep) mean -0.90; the flatten head is not the REVE lever either (reve_d6_flat control dropped too). tsc_dbn40_bn128_things fold 1 +2.30 over 12 ep.
- 16:10: tsc_dbn40_smix_things STOPPED: fold 1 e1-e10 mean -4.06 (last5 -4.88). Similarity-smoothed targets make the EEG predict the neighbourhood average -> less discriminative (also -1 on FBTN before). REVE/SQF smix dropped. Next for the narrow-head encoders: multi-scale aux heads (reve_d6_ml, sqf_dbn_ml) and SigLIP.
- 16:50: BUG fixed: with --auxiliary_image_targets the test gallery was built from all concatenated target blocks (7680-d) -> crash at the first sealed eval (score_track1.score_modules, trial_cache, grader_sim now slice target_dims[0]). reve_d6_ml_things restarted.
- 17:35: **TSC width sweep (folds 1-2, official): bn64 +2.40, tb512 +1.90, bn128 +1.35** (6/10, p .25) -> 64 best
  so far; bn32 queued to bracket. reve_d6_ml fold 1/2 e1-3: -0.8 / -0.3; tsc_dbn40_siglip fold 1 e1-5: -1.0.
- Correction: sqf_dbn_bn64 kept SQF's 512 residual trunk and only added 512->64->1536 -- NOT the TSConv-bn64
  analogue (there the whole trunk incl. the residual MLP is 64). Queued **sqf_dbn_tb64_things**
  (--tsconv_backbone_dim 64: 320->64, residual 64, LN, 64->1536). New REVE head **attnstat** (SQF's attention
  mean+std pooling over tokens -> linear): reve_d6_attnstat_things.
- 17:55: tsc_dbn40_siglip_things STOPPED: fold 1 e1-e15 mean -0.21 (no effect); reve SigLIP dropped. reve_d6_ml e1-5: -0.2 / -0.4 (neutral, kept to e8).
- 18:20: reve_d6_ml_things STOPPED: e1-e7 fold 1 -0.34, fold 2 e1-e6 -0.20 (neutral at REVE's usual peak) -> multi-scale aux heads do nothing for REVE; sqf_dbn_ml dropped as well. sqf_dbn_tb64 fold 1 e1-5 +0.68.

### A14 (2026-10-13 18:25): regularise where the capacity sits
TSConv's over-capacity was its head (fixed by bn64). In REVE/SQF the capacity (and likely the overfitting) is in
the transformer / conformer blocks; their heads are already small, and head changes failed (REVE bn64 -3, mlp
-8.6, flatten+bn64 -0.9, multi-scale aux ~0; SQF bn64 ~0). REVE has no stochastic depth in our wrapper (LaBraM
only). Queued: sqf_dbn_ema_things (EMA .9995; SQF/TSC train without EMA), reve_d6_wd02_things (weight decay
.2 vs .05), reve_d6_chdrop_things (channel-dropout input aug, as in the scratch-LaBraM recipe),
tsc_dbn40_bn64ema_things. Plus sqf_dbn_tb64 (true bottleneck analogue) and reve_d6_attnstat (SQF pooling) running.
- 18:55: sqf_dbn_tb64_things STOPPED: fold 1 e1-e13 mean -0.11, fold 2 e1-e6 +0.32 -> no bottleneck variant helps SQF; head capacity is not its limit.
- 20:12: reve_d6_attnstat_things STOPPED (fold1 10 ep mean -1.92, fold2 9 ep -1.34): attention-stat pooling hurts REVE like every other REVE head change.
  sqf_dbn_ema_things fold1 19 ep mean +4.78 (last5 +3.80) -- first positive lever on SQF. EMA (REVE already has it, 0.9998) is
  the "regularise where the capacity sits" idea working. Promoted: sqf_dbn_ema folds 3-4 queued; tsc_dbn40_bn64ema moved up.
  Caveat: EMA flatters mid-cycle epochs; the verdict is at the budget epoch.
- 20:40 CORRECTION on sqf_dbn_ema: per-epoch curves show EMA reaches 28.9 at e5 and then plateaus at 26.4-28.6, while the
  baseline is noisy at high LR (16-25 mid-cycle) and only peaks at the end of the anneal (fold1 28.5 @ e34, fold2 27.7 @ e39).
  The +4.8 matched-epoch mean is EMA smoothing the high-LR phase, not a better endpoint (cv4 pairs screen 09-30: EMA on
  TSC/SQF +-0.2 at budget). A one-cycle anneal to LR~0 already averages the weights; EMA matters for REVE/LaBraM because those
  recipes stop mid-schedule (stop_after_epoch). Folds 3-4 un-queued; tsc bn64ema demoted. Possible use: SQF compute cut (e5 ~ e34).
- 20:50: reve_d6_wd02_things STOPPED (fold1 e2-4: -3.3/-3.6/-4.1 vs reve_d6, widening). Stronger wd hurts REVE.
- 21:15: sqf_dbn_ema fold2 stopped (fold1 endpoint decides EMA). Queued SCORE-style cross-subject multi-positive (--grouped_batch_sampler --multi_positive_loss) on all 3 encoders (lit scan: Alljoined +3/+6 top1/top5 in 2608.19134). TTA probe (time-shift / gain / channel-drop averaging) running on fold1 ckpts.
- 21:15 TTA probe DEAD (fold1 sealed_best ckpts, fraction .5; overnight/tta_probe.py): averaging predictions over augmented copies
  | encoder | none | shift +-1 | +-2 | +-4 | gain .8/1/1.25 | chdrop .1 x5 |
  | TSC bn64 | 29.3 | 28.5 | 28.1 | 26.3 | 29.1 | 27.9 |
  | REVE d6 | 29.2 | 29.6 | 29.4 | 29.1 | 29.2 | 26.2 |
  | SQF dbn | 28.5 | 28.0 | 28.1 | 27.4 | 28.3 | 27.5 |
  The grader's repeat averaging already removes the noise TTA would; shifts move conv models off their time-locked filters.
- 21:20: sqf_dbn_ema fold1 endpoint confirms the artifact: e33-35 deltas -0.6/-1.1/-0.5, best 28.9 vs 28.5 (+0.4). EMA CLOSED for SQF.
  reve_d6_mpos_things STOPPED (fold1 +0.3/-1.3/-3.8 at e1-3): at batch 128 the grouped sampler leaves 32 images = 4x fewer negatives.
  TSC/SQF (batch 1024 -> 256 images) mpos screens still queued. reve_d6_chdrop fold1 -0.17 over 4 ep (neutral, running).
- 21:35: A15 REVE stochastic depth: _DropPath wrapper on every REVE attention/FF branch (foundation.py, --fm_drop_path, linear 0..p over depth; self-checked). Queued reve_d6_dp1/dp2_things (p .1/.2) folds 1-2. Rationale: REVE has zero dropout and every head/loss change hurt it -> regularise the trunk, not the head. Also checked: loss uses un-normalised EEG x L2 image at fixed T=1, so the EEG norm is already a learned per-trial temperature (no temperature sweep needed).
- 22:15 A16 MORE SUBJECTS (THINGS-EEG1, 48 subj x 1 trial of EEG2's 16540 training images). First export (from the stored
  250 Hz epochs) SCRAPPED: those epochs crop the -0.2..0 prestimulus (data = 0..1.0 s), so the time axis was off by 0.2 s and
  the baseline sat on the response. Rebuilt from RAW BrainVision with the grader chain (export_things_eeg1_to_alljoined.py):
  Fz re-ref, 32 Alljoined channels, notch 50/60, 0.1-75 Hz, 120 Hz, RobustScaler per recording, -0.2..0.8, baseline, clamp 20.
  Packed 4 subjects per pseudo-subject sub-31..42 (rep slot = subject; training draws single reps -> no averaging).
  Check on sub-02: mean|x| .52 (THINGS-EEG2 .51, Alljoined .70). Occipital grand ERP: EEG1 periodic at 100 ms (10 Hz RSVP,
  50 ms stimuli) and ~half the amplitude; Alljoined and THINGS-EEG2 periodic at 200 ms (5 Hz). So EEG1 trials carry 2x the
  neighbour overlap and weaker evoked signal: screen decides if subject diversity outweighs it (user's caveat, 22:00).
- 22:20: tsc_dbn40_bn64mpos STOPPED (fold1 14 ep -0.66, last5 -0.90: no gain); reve_d6_chdrop fold1 done neutral (-0.11), fold2 dropped; sqf_dbn_mpos fold1 only.
- 23:05: reve_d6_dp1 STOPPED (fold1 6 ep -1.20, last5 -1.38), dp2 stopped with it: stochastic depth hurts REVE like dropout did not help it. EEG1 caches built (sub-31..42, eeg1_caches_ok); queued reve_d6_e1 / tsc_dbn40_bn64_e1 / sqf_dbn_e1 folds 1-2 first, d3 (EEG1 own BN) variants ready.
- 23:20: EEG1 runs were IO-bound off the NAS (REVE 5 it/s vs 26, TSC 2.6-3.4 s/it): 18.1 GiB > the stage script's half-of-30G warm limit. Warmed by hand (overnight/warm_e1.sh) -> REVE 25.6 it/s, TSC 8.2 it/s; stage_eeg_cache warm limit raised to 65%.
- 23:22: sqf_dbn_mpos STOPPED (fold1 14 ep mean ~-1.0, last5 ~-2.6). Cross-subject multi-positive (SCORE) is negative on all 3 encoders here -> CLOSED.
- 23:50: A16b fallback prepared: sub-43..54 = EEG1 minus each subject's grand-average epoch (removes the 10 Hz steady-state response; overnight/eeg1_meansub.py), recipes *_e1m_things. Early plain-EEG1: reve +0.1 (2 ep), tsc +0.0 (7 ep), sqf -0.9 (2 ep).
- 00:05: plain EEG1 early: reve -0.87 (3 ep), sqf -0.53 (6), tsc -0.44 (13). Queued e1m (mean-subtracted) x3 and d3 (own BN) tsc/sqf ahead of plain-e1 fold 2.
- 00:20: A16c queued: pt_e1_{reve,tsc,sqf} = pretrain on EEG1 sub-31..42 ONLY (no test subject ever in EEG1 -> one checkpoint serves all folds), then *_pte1_things = the base recipe fine-tuned from it (cv4.sh waits for the checkpoint). Lit scan on mismatched-SOA RSVP reuse running.
- 00:30 lit scan (EEG1 reuse): no paper pools THINGS-EEG1 with EEG2/Alljoined for retrieval (BrainFLORA 2507.09747: EEG1 rarely
  used, low SNR); Banville 2025 (2501.15322): 24->48 EEG1 subjects +0.004 R n.s. -> plain pooling ~neutral is expected.
  Robinson/Grootswagers/Carlson 2019: first peak ~120 ms invariant to SOA, later info shrinks at fast SOA -> EEG1 image info in
  ~0.05-0.35 s, later = next images. Fixed 10 Hz SOA makes timing deconvolution (unfold) ill-posed. Recommended: own norm
  domain + crop/fade late window + lower weight; pretraining. Built A16d sub-55..66 = EEG1 faded to 0 over 0.35-0.45 s
  (overnight/eeg1_window.py); recipes *_d3_e1w_things (own BN) and reve_d6_e1w_things.
- 00:45 OPS: editing cv4.sh IN PLACE while workers run it makes bash resume at a shifted byte offset (it re-ran a killed
  fold). From now on cv4.sh edits = write a copy + rename (new inode). Building on 516 evicted the REVE-e1 page cache
  (1.2 it/s): CPU builds now go to the node whose run uses the least data (513, pt_e1 6 GiB).
- 00:45 A17 FRAMEWORK (user, 00:35): epoch = A_s [sum_k R_d(img_{n+k}, t - k SOA_d) + SS_d] + noise_s. EEG1 adds no new
  images, only 48 brains (A_s, noise_s) + trials; its temporal style (10 Hz overlap, truncated late response) is the
  nuisance. => keep EEG1's subject style, give it the target paradigm's temporal style:
  A17a splice (sub-67..78): EEG1 early (<0.30 s) + cosine x-fade 0.30-0.40 + THINGS-EEG2 late, same image (segment recombination).
  A17b residual (sub-79..90): EEG1 early + P(EEG1) + [EEG2 late - P(EEG2)], P = ridge late<-early on EEG2 single trials.
  Same partner draws (seed 0); partners only EEG2 sub-21..30 (never test) -> fold-safe. Controls: e1w (faded), pte1.
- 00:45: user: keep the simple levers too -> re-queued *_d3_e1_things (EEG1 own BatchNorm path, plain EEG1) for SQF and TSC right after reve_d6_pte1.
- 02:30 EEG1-only pretraining scores CHANCE on Alljoined (pt_e1_reve 2.1-3.1, pt_e1_tsc 2.6-3.8 top-5; chance 2.5), loss
  4.73->4.68 only. Label-alignment check (overnight/reliab.py; per-image means 0.05-0.40 s, same- vs different-image
  score, z): EEG2 21 vs 22 +4.5 | EEG1 grp(1-4) vs grp(5-8) +0.6 | EEG1 grp vs EEG2 21 +3.6 | EEG2 21 vs Alljoined 01 -0.7
  | EEG1 vs Alljoined 01 +0.4. => EEG1 export is label/time-aligned (matches EEG2 at z 3.6) but very low SNR; raw
  cross-HEADSET (BrainVision vs Emotiv) similarity is nil even for EEG2, so source-only -> Alljoined at chance is expected;
  transfer only appears jointly. reve_d6_e1 fold1 final: best 28.6 vs base 29.4 (curve -0.75 mean): plain pooling ~ -1.
- 03:55: sqf_dbn_e1 (plain EEG1) STOPPED at 32/40 ep (mean -0.33, last5 -1.32). sqf_dbn_d3_e1 (EEG1 own BN) 13 ep mean +0.42, last5 +1.30. reve_d6_pte1: epoch-1 head start (+4.2) gone by e2; 8 ep mean +0.01.
- 04:25: reve_d6_pte1 (EEG1-only pretraining -> fine-tune) DONE: best 28.4 vs 29.4 (-1.0); head start vanishes by epoch 2. Pretraining on EEG1 CLOSED (tsc/sqf pte1 + pt_e1_sqf dropped; their pretrains were at chance on Alljoined anyway). Own-BN EEG1: sqf d3 18 ep +0.41, tsc d3 6 ep +0.77 (last5 +1.14).
- 05:50: sqf_dbn_d3_e1 fold1 33 ep mean +0.51, last5 +1.04 (only EEG1 variant consistently >0) -> fold 2 queued next. tsc d3_e1 28 ep -0.03; reve e1s (splice) 7 ep -0.47.
- 06:25: BEST-EPOCH verdicts: reve_d6_e1s (splice) 28.2 vs 29.4 (-1.2); sqf_dbn_d3_e1 (own BN) 27.5 vs 28.5 at 38/40 (-1.0; its mid-run lead was a high-LR artefact). Fold-2 replication + conv splice runs dropped. Remaining REVE controls: e1r (residual splice), e1w (faded window).
- 06:50 A16/A17 EEG1 SCOREBOARD (fold 1, best sealed epoch vs the same recipe without EEG1):
  | variant | TSC bn64 (29.3) | SQF dbn (28.5) | REVE d6 (29.4) |
  | plain pooling (e1)            | stopped e20, -0.38 curve | stopped e32, -0.33 curve | 28.6 (-0.8) |
  | EEG1 own BatchNorm (d3_e1)    | 29.7 (+0.4)  | 27.5 (-1.0) | n/a (no BN) |
  | EEG1-only pretrain -> FT      | dropped       | dropped      | 28.4 (-1.0) |
  | splice EEG1 early + EEG2 late | dropped       | dropped      | 28.2 (-1.2) |
  | residual splice / faded window | running (REVE) |
  All within seed noise or negative: no EEG1 lever found. EEG1-only models are at chance on Alljoined.
- 08:05: EEG1 line CLOSED (reve e1r 6 ep -0.32, reve e1w 4 ep -0.47, tsc d3_e1w 12 ep +0.12/last5 -1.04 stopped).
  A18 ENSEMBLE DISTILLATION (multi-target: DINO + teacher): teacher = mean L2-normalised prediction of the fold's 3 base
  models (tsc bn64, sqf dbn, reve d6; trained on exactly the fold's training subjects -> no test leakage) on every training
  trial (export_teacher_cv4.py; real subject ids so THINGS rows take the THINGS BN path). Student = same base recipe +
  --distill_lambda 1 --distill_tau 0.005 (Track-1 val-split: KD +3.2 sqfp, +2.2 FBTN). Recipes *_kd_things, folds 1-2 queued.
- 08:20 OPS: the 513/514 align_dyn workers run a pre-claim-check parse of the loop (bash parsed it >1 day ago), so they duplicate claimed runs; running entries are now removed from align_queue.txt instead of relying on claims. cv4.sh now claims *_kd_things runs before the ~30 min teacher build.
- 08:45: A18 reve_d6_kd fold1: e1 27.8 (base 20.9), e2 31.0 (base 25.5; base BEST 29.4). Queued reve kd folds 3-4 after 2.
- 08:55 OPS: KD runs on 514/516 crawled at ~1 it/s with 28 GB 'cached': stale EEG1 pages (active LRU list) inside the job's 30 GB memory cgroup kept evicting the freshly warmed working set. Fix: overnight/evict_eeg1.py (posix_fadvise DONTNEED on the EEG1 files), then re-warm (overnight/warm_kd.sh + cat teacher) -> TSC ~8 it/s, SQF ~7 it/s.
- 09:15: A18 fold1 so far: REVE KD best 33.0 @e5 vs base best 29.4 (+3.6; matched +4.6..+6.9 every epoch, REVE stops at e10 so not a mid-schedule artefact); SQF KD 8 ep matched +4.0 (judge at end); TSC bn64 KD 11 ep matched -1.06 (KD vs 64-d bottleneck?).
- 10:07 A18 fold1: REVE KD DONE best 34.2 vs 29.4 (+4.8; matched +5.1 mean). SQF KD 21/40 ep best 30.1 vs 28.5 (+1.6 so far; matched +3.9). TSC bn64 KD 30/40 ep best 27.7 vs 29.3 (matched -0.6) -> try kd3 (lambda .3). Fold-2 teacher building.
- 10:15: A18 fold1 FINAL REVE KD: best 34.2 vs 29.4 (+4.8). SQF KD 22 ep matched +3.9 (last5 +4.3). TSC bn64 KD (lambda 1) STOPPED at 31/40 (matched -0.60, best 27.2 vs 29.3) -> trying lambda 0.3 (tsc_dbn40_bn64_kd3_things).
- 10:15: queue = reve kd folds 3,4, sqf kd folds 2-4, then old head screens; EEG1 leftovers removed. SQF KD f1 22 ep best 30.1 vs 28.5.

## A19 (2026-10-14). Temporal occlusion probe (occlusion_probe.py; cv4 fold 1, sealed rule, fraction 0.5)
Replace a 100 ms window (occ) or everything outside a span (keep) by the mean epoch of training sub-01 (image-agnostic).
Onset = sample 24. Info lives at 0.1-0.5 s, peak 0.1-0.25 s, for all three encoders:
| encoder | none | worst occ (0.1-0.2 s) | occ 0.2-0.45 s | occ 0.65-0.8 s | keep 0-0.8 | keep 0.05-0.6 | keep 0-0.3 |
|---|---|---|---|---|---|---|---|
| TSC bn64 | 29.3 | 12.3 | 15-19 | 29.2-29.4 | 24.6 | 22.7 | 8.0 |
| SQF dbn  | 28.5 | 10.4 | 15-19.5 | 26.2-27.9 | 25.2 | 23.6 | 7.7 |
| REVE d6  | 29.2 | 10.3 | 18-22 | 27.7-27.8 | 27.6 | 23.2 | 9.1 |
Prestimulus matters to the convnets as trained (keep 0-0.8: -4.7 TSC / -3.3 SQF) but barely to REVE (-1.6). Keep-only drops
are partly distribution shift (never trained on templated spans); a crop needs a training run to judge. TSC head weight
norm per time position is flat (0.84-1.00): each of the 36 positions spans 0.4 s, none is pure noise.
Ensembles (grader_sim --shuffle --fp, 20 subj): REVE-scratch+SQF+TSC-bn64 39.73 (vs 39.80 old best); REVE-d6+SQF+TSC-bn64
40.05 (+0.25, noise). TSC KD (lam 1 / .3) negative -> closed.

## A20 (2026-10-14). Alljoined-1 (NEMAR nm000133) suitability
Data: 8 subj (13 sessions), BioSemi 64 ch, 960 NSD-shared COCO images x <=8 reps, 300 ms stimuli, REAL SOA ~0.80 s
(paper says 0.6-0.65); 59,540 single trials. CC-BY-NC-ND. Raw at /nasbrain/p20fores/alljoined/nm000133 (nemar-py 0.3.1 via
uvx; 0.3.0 fails), grader-chain export (avg ref over 64 -> our 32 ch) export_alljoined1.py -> alljoined1_120Hz/sub-XX.npz
(+ dinov2_track1.npz targets of the 960 NSD crops).
- Model-free ridge (repeat-mean EEG -> DINOv2, 7 subj x 800 img -> held-out subj x 160 img, top-5, chance 3.12):
  Alljoined-1 6.95 == Alljoined-1.6M (same protocol, 4-rep mean) 6.95 -> the data carries as much decodable info.
- Zero-shot cv4 models (zeroshot_alljoined1.py, 4 folds x 8 subj, 960-way top-5, chance 0.52): TSC 1.45, SQF 1.48,
  REVE 1.20, ens 1.55; early-only (-0.2..0.35 s kept) lower (ens 1.30).
- Matched (overnight/aj1_matched_ref.py, fold 1, 200-way, 8 reps, chance 2.50): in-domain unseen subj TSC 16.8 /
  SQF 18.5 / REVE 20.1 vs Alljoined-1 5.66 / 5.63 / 5.51 -> transfer keeps ~20% of the above-chance signal.
  Loss = paradigm/hardware + COCO-scene vs THINGS-object target domain (not separable without training).

## A21 (2026-10-14). REVE KD student, all 4 folds, grader_sim --shuffle --k 2 --fp (20 subjects, official metric)
| model | plain | old transform | fp router |
|---|---|---|---|
| reve_d6 (base) | 28.85 | 34.58 | 36.52 |
| reve_d6_kd (student) | 33.38 (+4.53) | 37.97 (+3.39) | 39.30 (+2.78) |
| REVE-d6 + SQF + TSC-bn64 (best ens, A19) | 33.73 | 39.20 | 40.05 |
| REVE-KD + SQF + TSC-bn64 (student replaces REVE) | 34.10 | 38.32 | 38.50 (-1.55) |
| REVE-KD + REVE-d6 + SQF + TSC-bn64 | 34.27 | 39.03 | 40.33 (+0.28, noise) |
=> the single student ~= the 3-model ensemble (39.30 vs 40.05 fp, 1/3 of the inference), but it is redundant WITH the
ensemble it imitates: swapping it in loses diversity (-1.55), adding it is noise. Router gain shrinks on the student
(+5.9 vs +7.7 over plain). Ensemble-level progress needs a better/different teacher, not more students of this one.

## A22 (2026-10-14 ~16:00). EEG-norm probe + architecture screens (fold 1 first; kill early if clearly negative)
norm_probe.py (fold-1 models, all fold-1 test trials; logit scale is FIXED at 1 in all 3 recipes, so the EEG norm is the
only temperature): norm CV 0.33 / 0.44 / 0.39 (TSC / SQF / REVE); AUC(norm -> correct single-trial top-5) 0.57 / 0.58 /
0.59; Spearman(norm, input RMS) -0.31 / -0.19 / -0.25 (high-amplitude = artefact-laden trials get SMALL norms); grader-style
repeat averaging raw (norm-weighted) vs per-trial unit vs norm^2: TSC 32.6 / 33.5 / 30.3, SQF 31.4 / 30.5 / 29.4,
REVE 31.4 / 30.9 / 30.1 -> the norm is a weak learned per-trial confidence (unlike THINGS LOSO, CV 0.04); reweighting
repeats by it is a wash.
Screens (code: train.py --norm_type + model.swap_batchnorm; ortho_encoders ffn switch; foundation chan_lora):
reve_d6_p40 (5x0.2 s patches/electrode), reve_d6_p50 (7 half-overlapping), reve_d6_clora (rank-8 per-electrode patch
correction), tsc/sqf GroupNorm(1,C) (gn), tsc GroupNorm(C,C) (in), sqf no FFN, tsc GELU, tsc unit-EEG + learnable
temperature control (l2). New worker job 61315 (514) after 60372 exited on an empty queue.
- A22 ideas 1-2 added (user-approved): `--latency_align 6` (model.LatencyAligner: per-trial learned shift +-50 ms,
  differentiable linear resampling, zero-init = identity; *_lat_things on all 3 bases) and `FactorizedSTFormer`
  (ortho_encoders: SQF's fused temporal stem per electrode, NO spatial conv, 32x36 electrode x time tokens, 3 blocks of
  divided space/time attention d64, electrode softmax pooling + attentive stats; 1.28M; batch 512 bf16 ~6.6 GB;
  stf_dbn_things vs sqf_dbn_things). Queued first. No previous run had attention before the spatial collapse with a
  temporal stem (ATM attends over raw electrode traces first: -1.9; EEG Conformer conv-then-attention: -9.65).

## A23 (2026-10-14 17:15). KD final verdict: ABANDONED (user rule: only the best-ensemble score counts)
grader_sim --shuffle --k 2 --fp, 20 subjects (best ensemble REVE-d6 + SQF-dbn + TSC-bn64 = 40.05):
SQF student swapped in 40.35 (+0.30, noise); SQF student added as 4th 39.78; both students + TSC 38.12 (-1.93);
REVE student swapped in 38.50 (A21); added 40.33. KD students are strong solo (REVE +4.5 plain / +2.8 fp) but copy
the teacher ensemble and remove diversity -> no ensemble gain. KD line closed; best ensemble stays 40.05.
- A22b (user, 17:40): sta_dbn_things (FactorizedSTFormer time_attn=False readout=flatten bn64: conv along time, attention
  across electrodes at each time step, TSConv time-locked readout; 0.36M), etok_dbn_things (ElectrodeTokenTSConv: TSConv
  step 4 -> 4 attention layers over 32 electrode tokens of 40x36 stem features; 1.84M), reve_d6lite_things (embed 160,
  4x40 heads, FFN ratio 2: 1.84M vs 19.8M), sqf_dbn_sf160_things (160 stem filters). Fix: train.py domain-BN validation
  now allows FactorizedSTFormer / ElectrodeTokenTSConv (stf_dbn_things looped on FAILED ~20 min). stf runs at 16 min/epoch
  (judge at matched epochs ~8-10). Never-run so far on Alljoined: brain2brain (cross-subject EEG<->EEG InfoNCE), RKD.
- 18:50 verdicts (fold 1, matched epochs vs base): tsc_dbn40_bn64lat (latency aligner) +0.03 over 39 ep -> CLOSED, and
  reve/sqf _lat dropped unrun (TSC is the most shift-sensitive model = its best case). reve_d6_p40 (electrode x time
  tokens) -1.87 over 7 ep -> KILLED, p50 dropped. stf_dbn (factorised space-time attention) -5.4 at 5 ep, 16 min/epoch
  -> KILLED; its lighter sibling sta_dbn (no time attention, TSConv readout) stays queued.
- 19:20 etok_dbn (TSConv step 4 -> REVE-style electrode-token attention) -3.0 over 10 ep (last5 -2.4) vs tsc bn64 -> KILLED.

## A24 (2026-10-14 ~19:45). Literature sweep (subagent) -> what transfers to a FIXED grader gallery
Gallery = DINOv2-g layer-26 mean token, scored by the grader -> target-side methods (PE-Core/InternViT intermediate-layer
query-pooled targets, ViEEG multi-view, HyFI hyperbolic) only as auxiliary targets (our multi-layer/blur aux failed);
hubness correction (NNN/QB-Norm, CSLS) needs gallery-side scores -> impossible in predict(). Transferable: SAMGA MMD
coarse stage (in train.py since earlier, never run on Alljoined) -> tsc_dbn40_bn64mmd / reve_d6_mmd (lambda .5, all epochs);
subject-conditioned FiLM + subject dropout + test-time marginalisation over training-subject codes (SAMGA/PictorialCortex)
= candidate next; ProCA (+1.5 on weak baselines), ZEBRA (fMRI recon only) second tier.
- 20:10 tsc_dbn40_bn64sfilm_things queued: existing subject_adapter (FiLM rank 8, never run on cv4) + new --subject_adapter_dropout 0.3 (ID 0 = padding/zero code = population path used at inference; dropout makes the trunk train on it).
- 20:20 verdicts: tsc_dbn40_bn64mmd (SAMGA MMD .5) -9.9 over 6 ep -> KILLED, reve_d6_mmd dropped (matching the EEG and image
  clouds fights the bottleneck's low-rank retrieval geometry). sta_dbn (per-time electrode attention, TSConv readout) -1.66
  over 8 ep -> KILLED. reve_d6lite (1.84M) finished -1.52 vs reve_d6 (19.8M): 10x smaller for -1.5.
- 20:30 sqf_dbn_sf160 (160 stem filters) -0.08 over 9 ep (last5 -0.9) -> KILLED (user: be more aggressive). Queue pruned to
  reve_d6_clora, tsc_bn64l2 (temperature control), tsc_bn64gelu. Held back unless tsc_bn64gn is positive: sqf_dbn_gn,
  tsc_bn64in. Dropped: sqf_dbn_noffn (an ablation, cannot help). New kill rule: TSC mean <= -1 after 5 ep, REVE after 4.
- 20:55 tsc_dbn40_bn64gn (per-trial GroupNorm(1,C) for every BN) -4.3 over 11 ep -> KILLED; sqf_dbn_gn / tsc_in dropped.
  tsc_dbn40_bn64sfilm (subject FiLM rank 8 + subject dropout .3) -0.75 over 10 ep (last5 -1.1) -> KILLED (aggressive rule).
  CORRECTION A20: Alljoined-1 has 49,540 single trials (not 59,540).

## A25 (2026-10-14 ~21:05). Alljoined-1 in the training mix (user)
Pipeline: dataset.add_extra_trials (extra single trials after the grid, own DINOv2 targets, object_idx 100000+row so no
positive-mask collision), train.py --extra_trials_dir/--extra_subject_offset 40, own BN path via
--tsconv_domain_bn_boundary2 41, optional --extra_input_adapter (model.ExtraInputAdapter: identity-init depthwise
25-tap temporal filter per electrode + 1x1 electrode mix, applied to subjects 41..48 only; never at test). Smoke-tested
(mixed batch 1/21/41/48, tertiary BN, adapter identity + grads). Screens (fold 1): tsc_dbn40_bn64_aj1, reve_d6_aj1,
sqf_dbn_aj1, tsc_dbn40_bn64_aj1a, reve_d6_aj1a. 4th worker job 61505 (node 518) per user.
- 21:15 A26 queued: REVE ~5M budget: reve_d2w416 (4.87M), reve_d6w256 (5.16M), reve_d10w192 (4.75M), interleaved with the A25 Alljoined-1 screens.

## A27 (2026-10-15 morning). Overnight verdicts, cv4 fold 1 (best epoch vs base best; mean = matched-epoch mean diff)
| run | best vs base | mean | verdict |
|---|---|---|---|
| tsc_dbn40_bn64_aj1 (Alljoined-1, own BN) | 19.5 vs 29.3 (-9.8) | -9.2 | SUSPICIOUS (3% extra data) |
| tsc_dbn40_bn64_aj1a (+ input adapter) | 19.3 vs 29.3 (-10.0) | -9.0 | adapter changes nothing |
| sqf_dbn_aj1 | 21.0 vs 28.5 (-7.5) | -5.5 | |
| reve_d6_aj1 (no BN) | 25.8 vs 29.4 (-3.6) | -3.5 | |
| reve_d6w256 (5.16M) | 28.5 vs 29.4 (-0.9) | -0.5 | best small REVE, 4x fewer params |
| reve_d10w192 (4.75M) | 28.4 vs 29.4 (-1.0) | -0.9 | |
| reve_d2w416 (4.87M) | 27.0 vs 29.4 (-2.4) | -2.1 | depth matters more than width |
| reve_d6lite (1.84M) | 27.9 vs 29.4 (-1.5) | -1.5 | |
| tsc_dbn40_bn64l2 (unit EEG + learnable T, init .05) | 21.6 vs 29.3 (-7.7) | -8.7 | per-trial norm freedom is essential |
| tsc_dbn40_bn64gelu | 28.3 vs 29.3 (-1.0) | -0.6 | closed |
Alljoined-1 drop is ~5 points from epoch 1 (9.8 vs 15.3) with LOWER train loss; configs identical except the extra flags;
other same-night runs normal -> control tsc_dbn40_bn64_dupctl (49,540 sub-01 trials with THINGS targets through the SAME
extra path, subject id 1) running to separate a pipeline bug from a real Alljoined-1 effect. reve_d6_aj1a killed
(uninterpretable until then); reve_d6_clora (died at ep 4 with the 2026-10-14 session) re-queued.
Ops: workers 60371 (time limit) / 61315 / 61505 exited on "queue empty" (claims of the dead 61490 worker looked alive);
new worker 61634 (513). A stale .claim owned by the interactive job blocks other workers: delete it by hand.

## A28 (2026-10-15 12:00). Norm-robust batch-transform statistics: CLOSED (no gain)
overnight/robust_stats_sim.py: batch_transform_fp's mean / fingerprint regression / top-k directions estimated on
norm-robust copies of each member's batch, applied to the raw outputs (best ensemble, grader_sim --shuffle --fp, 20 subj).
base (Q = P) 40.05 (reproduces) | unit trials x mean norm 40.08 (+0.03, 8/20 better, 11 worse) | norm capped at the
batch median 39.57 (-0.47, 5/20). Large-norm trials do not distort the statistics; keep the transform as is.
Head geometry (overnight/head_geometry.py, fold 1): behind the final LN the output norm is set by where the code puts its
fixed energy (gain spread s1/s_med 3.6 TSC / 15.6 SQF); the pre-LN magnitude the LN discards is ANTI-correlated with
|e| (rho -0.35 TSC, -0.91 SQF): loud trials are routed to low-gain code directions.

## A29 (2026-10-15 18:40). Ensemble search over the 10 best 4-fold models, with the full sealed post-processing
ens_cache.py = exact per-member cache of grader_sim --shuffle (fp router included; the pipeline is linear after each
member's transform), validated: current best REVE-d6 + SQF-dbn + TSC-bn64 = 40.05 fp / 33.73 plain / 39.20 k2 (=A23),
REVE-KD student alone 39.30 (= A21). ens_search.py: greedy + exhaustive sizes 3-6 (792 subsets) + an honest estimate
(subset picked on folds 1-2 subjects, scored on folds 3-4, and vice versa).
Singles (fp): reve_d6_kd 39.30, reve_d6 36.52, reve_scratch 36.27, sw0 36.23, sqf_dbn_kd 35.65, lab_s10 35.08,
plab_gain 34.48, reve_z 34.45, sqf_dbn 34.40, tsc_bn64 34.17.
| subset | fp | vs 40.05 (subjects +/-, Wilcoxon p) | honest (cross-half) for its size |
|---|---|---|---|
| best 3: reve_d6_kd + sqf_dbn + reve_d6 | 40.60 | | 39.58 |
| best 4: reve_d6_kd + sqf_dbn_kd + reve_scratch + tsc_bn64 | 40.75 | +0.70 (13/6, p .13) | 40.08 |
| best 5: + reve_d6 | **40.85** | +0.80 (12/6, p .08) | 39.95 |
| best 6: sqf_dbn_kd + sqf_dbn + sw0 + reve_d6 + tsc_bn64 + lab_s10 | 40.82 | +0.78 (9/9, p .17) | 40.40 |
| current 3 + reve_scratch (not selected, the natural 4th) | 40.70 | +0.65 (13/4, p .053) | - |
Reading: the searched maxima (40.6-40.85) shrink to 39.6-40.4 out of sample: selection noise is ~0.5-1 point at
20 subjects. No subset is a clear win; the most credible step is adding the 22-layer scratch REVE as a 4th member
(+0.65, 13/4, not picked from hundreds of subsets). KD students enter the top subsets but do not change the
picture (A23 stands: no reliable KD gain).
- (21:00) USER DECISION: keep the 3-member reference REVE-d6 + SQF-dbn + TSC-bn64 (40.05): simpler; the +0.65 4th member is not adopted.

## A30 (2026-10-15 21:15, user idea) queued: members "talk" through a proxy gallery (poe_sim.py)
predict() has no gallery, so the 16,540 training-image targets act as candidates: per trial, product of experts over
the members' cosines -> consensus embedding; variants agree (reweight members by agreement with the consensus) and mix
(pull the summed prediction toward it); base must reproduce 40.05. Prior evidence: label-free per-trial weighting incl.
member agreement R^2 0.04 (ROUTING_LOG E4).
- A30 result (poe_sim.py, grader_sim --shuffle --fp, 3 reference members, 20 subjects): base 40.05 (reproduced).
  agree (members reweighted by agreement with the product-of-experts consensus over the 16,540 training-image proxy
  gallery) t .02/.05 x b 5/20: -0.05 to -0.15 (6-9 better / 8-10 worse) = null. mix (pull toward the consensus
  embedding) -21 to -31 (0/20): the test images are not in the proxy gallery, so its consensus is a wrong image.
  CLOSED: joint member combination via a proxy gallery gives nothing over independent post-processing + sum.
- (05:15) ARCH_LOG R-series: rmix (TokenREVE mixed tokens, 5.3M) swapped for REVE-d6 in the reference trio: 40.60 vs 40.05 (+0.55, 15/5, p 0.10). Awaiting the user's decision.
