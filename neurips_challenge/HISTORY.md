**Track-1 history (NeurIPS 2026), strict summary.** Most script names below no longer exist: their code is in git tag `pre-cleanup-2026-11-02`. Condensed from the former ALIGN/ARCH/AUG/ROUTING/SUBJECT/TTA logs (full text in the same tag); E/T/A/D ids refer to their entries.

## 1. Setting
- Task: single-trial EEG (32-ch Emotiv, 120 Hz, -0.2..0.8 s) -> 1536-d DINOv2-giant target, cosine-ranked by the grader against a fixed gallery.
- Sealed phase: unseen people AND unseen images, ~11 people; `predict(X)` on shuffled 64-trial batches mixing people; no subject id, no gallery, no state across calls; batch statistics allowed, batch size may change.
- Metric: official subject-aggregated top-5 (raw predictions averaged per (person, image), ranked, per-person top-5 averaged; chance 2.5 at 200-way).
- cv4 (CV4_PROTOCOL.md): 4 frozen folds x 5 held-out Alljoined people; train on the other 15 + THINGS-EEG2 sub-21..30; 20 people in all. Score with `tta_cache.py` + `lean_tta.py`.
- Warm-up leaderboard numbers (subjects seen in training, single epochs) are a different measurement: never compare them with cv4.

## 2. Current best
- **optx5 = 44.25** (folds 47.90 / 42.40 / 46.70 / 40.00): `reve_d6_optx_things`, `reve_scratch_optx_things`, `rmix_reg_optx_things`, `sqf_dbn_optx_things`, `tsc_dbn40_bn64_optx_things` (all `--op_transfer_prob 0.5`).
- Lean TTA (`score_track1.batch_transform_wknn`): per-member batch-mean-norm scaling; centre each trial on its K=4 nearest batch-mates in the 656-d `eeg_fingerprint` (log variance, channel correlations, band powers) whitened by the within-person covariance (WCCN .3, fit offline on the training Alljoined people), c=1; sum members; ONE uncentred Ledoit-Wolf shrink of the sum, d = p/(p+n).
- Baseline without operator transfer: base5 (`*_things` twins) = 44.02 (47.20 / 41.80 / 46.60 / 40.50). Recipes in `cv4.sh`, usage in README.md.

## 3. What moved the number (chronological; cv4, 20 people)
| lever | measured delta | cv4 score after |
|---|---|---|
| THINGS-EEG2 in training + per-dataset BatchNorm (boundary 21) | SQF +4.3 (sqf_dbn 28.00); TSC -3.4 without dbn, neutral with; REVE (no BN) +2.4/+2.2 (E29) | plain 3-ens 33.63 (E25) |
| stateless batch transform: subtract .5 batch mean, shrink top-2 centred PCs x.5 (E25) | +5.00 (20/0) | 38.62 |
| EEG-fingerprint LDA router `batch_transform_fp` p6 c1 k3, LDA on Alljoined+THINGS (E43) | +1.18 (13/5) | 39.80 |
| TSC 64-d bottleneck head (A4: single 24.68 -> 27.23, +2.55, 16/20, p .004); REVE-d6 replaces REVE-scratch | ens +0.25 (noise) | 40.05 (reference until 10-20) |
| rmix TokenREVE (32 electrode + 5 global 0.2 s time tokens, 5.3M) / rmix_reg (+4 registers) | single 28.05 / 28.12 vs REVE-d6 27.55; router swap +0.55 (15/5, p .10) | member of base5 |
| wknn: K4 batch-mates in WCCN-whitened fingerprint space, c .7, + old k3 shrink (T5/T17) | +2.10 (15/5), live | 42.15 |
| one uncentred LW shrink (d .93, c .85) instead of k/b/a, all 10 cached members, no selection (T29) | +1.69 (vs 42.33, same 10 under wknn); +3.73 vs router; rich fp 44.52 | 44.02 (live) |
| KD members banned, <=5 members: lean pipeline (656-d fp, c 1, shrink of the SUM, d = p/(p+n)) on base5 (T40/T43) | +0.03 | 44.05 replay / 44.02 (aug_ens reference) |
| channel-operator transfer on all 5 members (AUG 10-27) | +0.23 (folds +0.7/+0.6/+0.1/-0.5); singles +0.8 avg | **44.25** |

## 4. Dead ends (result vs the reference of the time)
**Architecture** (fold-1 screens vs TSC-bn64 29.3 / REVE-d6 29.4 / rmix 28.83 unless noted)
- Attention or tokens BEFORE the spatial collapse: factorised space-time -5.4, electrode tokens on TSConv -3.0, per-time electrode attention -1.7, REVE electrode x 0.2 s patches -1.9, rmix electrode x patch tokens -7.1, learned virtual sensors -10 (hvs) / -1.9 (rvir).
- Processing AFTER the TSConv stem: filter attention -2.3, time attention (hta/htsqf/htd1/htgat) early lead then plateau -1..-2.5, adaptive spatial filters -2.3, slow+fast dual path -2.9, query readout -4.1, MoE readouts (K 4/8) lead then -1.6..-2.3, time<->spectrum cross-attention -1.0.
- TSConv knobs (vs fp32 hbase32 28.6): filters 20-160 flat, kernel 13-25 flat / 49 -0.8, spatial x2 par, 0.4 s pool -6.8, max pool -4.2, stacked convs -1.2, GELU -1.0, GroupNorm -4.3; head width bn64 +2.40 > tb512 +1.90 > bn128 +1.35; final-LN removal -1.66; unit-norm EEG + learnable temperature -7.7.
- REVE: bn64 head -3, MLP head -8.6, flatten+bn64 -0.9, attention-stat pooling -1.9, multi-layer aux heads ~0, wd .2 -3..-4, drop-path -1.2, channel dropout ~0, 1.8M -1.5 / 5.2M -0.9; latency aligner (TSC) +0.03.
- rmix neighbourhood: 5-13M flat (capacity not binding), depth 6/8/11 par (4-fold), registers +0.21 4-fold with no ens gain, bipartite attn -3.9, gated -1.3, time patches 0.1 s -1.6 / 0.33 s -0.8 / overlap -2.0 / multi-scale par, readouts flatten par / split -1.4 / time-mean -2.2 / attn-stats -2.0, TSConv-stem tokens -1.5 (36) / par (9), SQF conv module par, filter tokens par, per-type FFN par.
- SQF: bottleneck (bn64, tb64) ~0, 160 stem filters par, + electrode tokens (sqfmix) -3.6 / -1.3.
- New 6th-member designs (fold 1/2 gain over base5): braindecode EEGNeX 0.0, CTNet/ATCNet weak, BIOT chance; GRU gruw +0.8/-0.6; FourierFormer +0.6/0.0 (variants -1.0..-1.1); DilatedConvNeXt +0.2; SpecPatchFormer -0.9/0.0; late-window specialists -0.2..-1.1. Useful 6th needs lean single >= ~35 AND hit phi <= ~.5 vs REVE family.
- Older families as members (ATM, LaBraM scratch/pretrained, pretrained REVE, dualstream, Bridge sw0, mixer, EEGNet, FBTN, Conformer, GCT): every swap <= base5 (sw0 43.77, ATM <= 43.8). TSConvPlus hk13 / hsm2 as 6th, 4-fold: 44.23 / 43.97 vs optx5 44.25.

**Augmentation / regularisation / schedule** (REVE-d6 fold 1 unless noted)
- Mixing: mixup -0.7; raw SubjectMix TSC -0.25 / SQF -1.75; same-dataset SubjectMix -0.3 (ens +0.2); SCORE cross-subject multi-positive negative on all 3; MixStyle dead.
- Identity-band edits: noise transplant -1.47; 8-30 Hz band swap -0.97; hyperalignment conversion not run (cross-subject cross-cov is 2-4 dims: maps are random or identity).
- Input views: low-pass < 8 Hz -3.6 (ens -0.4..-1.0); latency/montage jitter, gain aug, channel dropout ~0; repetition-average aug -0.5; CAR +0.35 / THINGS->Alljoined CORAL -0.55.
- Optimisers: SAM 4-fold ens 44.10 vs 44.02; SWA -0.7; EMA on SQF/TSC +-0.2 at budget; V-REx -4.05; wd .01 -2.1; negative-correlation penalty lambda 1/3 neutral (2-fold 44.25/44.80 vs 44.50).
- Schedules: REVE full anneal overfits (oc10 -1.8; full 20-ep cycle collapses to 16-21 by e20); peak lr 5e-4 -1.3; SQF 60 ep -2.1, TSC 60 ep +0.2; batch 256: REVE singles +1.0 but ens 43.85, TSC/SQF -2.7/-6.4.
- Neighbour (DINOv2-cluster) batches: 4-fold 43.83; with optx 43.90 (2-fold vs 44.50).
- Op-transfer variants (fold-1 ens, optx 47.50): geodesic extrapolation 47.80, per-band 47.20, random-orientation CONTROL 48.00, p=1 48.00, cross-dataset 47.30 => a generic spatial-mixing augmentation; rmix_reg optx on 20-ep cycle 4-fold 44.23.
- Losses: SigLIP REVE -2.5 / TSC -0.21; +0.1 MSE -2.73; softmax over all 16,540 training targets -1.2..-2.9; concept-negative masking +0.50 (noise).

**Alignment targets / InternViT / bridges**
- Target-side training: NeuroCLIP soft targets -4.06; UBP-like blurred targets -0.9; multi-layer DINOv2 aux heads ~0; SAMGA MMD -9.9.
- Train on another target, then linear bridge to the grader layer: CLIP RN50 -7.9 as 6th, DINOv2 L40 -3.3, L16 own 38.1 / bridged +6th <= 40.80; InternViT L28 own-space 43.75 but every bridge (ridge, codebook 3%, per-trial and image-mean stacking, PCA, TTA-then-bridge 24.05) loses ~14-18.
- Target diversity through training: warm start from IV28 (single 37.20, best optx5 swap 44.90 vs 45.15), fixed IV28 aux head REVE w1 38.70 / w2 37.35, TSC / SQF optx5+1 45.35 / 44.55 vs 45.15 (f1-2): noise. IV28's own-space edge = an easier gallery.

**Distillation & boosting**
- KD from the fold's 3-member ensemble: REVE student +4.53 plain / +2.78 router solo, ens swap -1.55, add +0.28; SQF student swap +0.30; TSC KD negative. Later, under the LW shrink, the KD trio REVE-KD+SQF-KD+rmix-reg = 44.12 live (T34): excluded by user rule (no KD).
- Boosting member: per-trial hardness weights -0.7..-12 (noise is single-trial); image-level ibst20 +1.40/-0.80 (2-fold ~+0.3), cancelled.

**Extra data**
- THINGS-EEG1 (48 people, 10 Hz RSVP; raw rebuilt with the grader chain): pooling REVE -0.8; own BN TSC +0.4 / SQF -1.0; EEG1-only pretrain -> FT -1.0 (EEG1-only models at chance on Alljoined); early/late splice -1.2; residual splice / faded window noise. Its identities raise router purity only (.33 -> .36).
- Alljoined-1 (nm000133, 8 people, BioSemi, COCO): in training TSC -9.8 / SQF -7.5 / REVE -3.6 (real: duplicate-path control -0.22); input adapter no change; zero-shot ens 1.55 at 960-way (chance 0.52).
- Curation: dropping weak people 9/17/19 never helped; no-THINGS member twins as swaps 42.35-43.67.

**Subject / affinity routing, person adaptation**
- Subject-level expert routing / member reweighting: oracle +1.0 (E2), honest -0.28 (E3). Affinity is real but only grand-ERP similarity predicts it (r .25).
- Per-trial quality / vote weights: R^2 .04; |y|^g, agreement, reliability oracle +0.23; L2 per trial -0.05..-2.1.
- Output-space per-person corrections (rotation / linear, even label-fitted) never beat centring (D1); static SAGE / gallery whitening / ridge maps -5..-25 (E17-E19); static output subspaces do not transfer across models (E30/E33).
- Per-person headroom sits in the readout (oracle fine-tune: head +12.6, all +14.1, LoRA r16 +11.9, 40x40 filter mix +7.3), but few-shot (6-512 labelled trials) -13..-27; unlabelled filter recolouring null; transforms inside the head -1.7 best; per-person maps share ~1 direction (D9).
- Learned person adaptation in the model: subject FiLM + dropout -0.75; training under the batch transform -3.5..-5.5.
- Learned identity embedders: AM-softmax purity .33-.36 vs LDA .29 (+0.50 s11, router refit on 61 ids null), supcon .26, temporal-contrastive .61 vs WCCN .77; ARM-lite affinity 40.33 vs 42.15; model-internal features +0.012 purity.

**TTA variants** (vs wknn 42.15 / lean 44.05 of the time)
- Grouping estimators: exp/Bayes weights, GP, PLDA, SNN, mutual-kNN, adaptive K, spectral clustering (purity .47-.59), similarity-profile re-ID (purity +.03, -0.25), bias-corrected c > 1: none beats the even K4 mean.
- Fingerprints: rich (+fine, alpha/beta/gamma corr, gamma power) helps only at 10 members (base5 43.85, optx5 44.15); tangent-space -0.3/-0.5; regression after kNN worse.
- Shrinks: centred LW, analytic LW d (-> 38.3), full/partial whitening (monotone worse), target-prior shrink -3..-4, per-dim power law -1..-2.2, per-person 2nd-order shrink (worse even with oracle groups), joint-member shrink = per-member, equal votes = noise.
- Input/model TTA: shift averaging -0.65, mirror -2.35, MC-dropout -0.6, AdaBN +0.1-0.3 over centring; per-person latency oracle only +1.05.
- Gallery / proxy: pull toward training targets -3..-37; mean-target removal ~0; proxy-gallery PoE: agree null, mix -21..-31; cross-person same-image pooling: oracle 46.5-53.8 but label-free -2..-5.5 (pair precision ~1% vs 0.5% chance; break-even needs p >= .25-.3 at full recall).

**Ensembling / selection**
- Subset search optimism 0.5-2: A29 40.85 -> 39.95 nested; T49 44.50 -> 42.20; nested greedy -0.42 / -0.51 vs base5 / optx5; greedy on all 20 tops at 4 members 44.55.
- More members: 17 non-KD 43.95; second seeds sd7 TSC 44.12 / SQF 44.35; nested per-member weights +0.14 (flip between splits); agreement-weighted aggregation dead; 70-swap scan max +0.55 = noise.

## 5. Structural findings / traps
- Ensemble is limited by error correlation: LOO base5 -TSC -1.07, -rmix-reg -0.14; REVE-family hit phi .74-.75 vs .52-.58 with SQF/TSC; any-member top-5 57.6 but 75% of misses hit by no member; ~50-80% of a single's gain vanishes in the ensemble. Correlation follows the architecture: a REVE trained toward InternViT stays at phi .65.
- Judge a lever only by a 4-fold swap into the lean-TTA ensemble (`lean_tta.py`). Fold-1 and 2-fold leads routinely vanish (rmixd8 +1.4, rmix_reg +1.0, hk13 +1.2 -> +0.3); matched-epoch seed noise +-3/epoch; fold-1 ens swaps scatter +-0.7; 20-person resolution ~1 point (1 image = 0.5 per person).
- checkpoint_sealed_best / budget epoch is selected on the test people: known and accepted (final retrain uses the 4-fold-optimal epoch). Do not re-raise.
- Grader target = DINOv2-g hidden state 26 of 40, mean over tokens, HF resize 256 + crop 224 (reproduced at cos 1.0000). The scorer averages RAW predictions, so trial norm = vote weight: never L2-normalise trials; the logit scale is fixed at 1, so the EEG norm is the learned confidence. Permute channels by name.
- TTA ceiling: perfect in-batch grouping caps base5 at 44.93 (c 1) / 45.12 (c .85); purity gains above ~.77 buy <= +0.25. At N=11 (s11 testbed) purity drops .74 -> .44 and gains shrink: check sealed-like N on s11. Lean TTA beats the router at every B 16-256.
- Signal: cross-subject image info is < 8 Hz (ridge probe full 12.3, < 4 Hz 12.6, 13-30 Hz 4.0); identity is the alpha/beta power topography; info window 0.1-0.5 s (peak 0.1-0.25); models are time-locked (REVE halves at a 50 ms shift).
- sub-09 (~10) and sub-17 (~15) are intrinsically hard: no lag, polarity or channel permutation helps; lifting both to ~45 = +3 on the mean.
- REVE family needs the e10 stop of its 20-epoch cycle; bf16 costs small conv models ~1.2 (h* recipes run fp32); EMA flatters mid-cycle epochs; capacity 5-19M is flat.
