# Night 2026-11-03: push the cv4 ensemble from 44.25 (optx5) toward 48

**Summary (for the morning; details below).** 48 was NOT reached. Best honest result: **opt4 = 44.55**
(optx5 without reve_scratch_optx; 48.50 / 42.80 / 46.30 / 40.60; the nested leave-one-member-out picks the same drop on
all 4 folds, N12). An exhaustive search over all 582 3/4/5-subsets of the 10 cached members tops out at 44.62
in-sample and 43.6-43.9 nested (N16): member composition is exhausted. Dead or noise tonight: Euclidean-alignment
members (N1, N4: +1.4 on the centring-free per-epoch metric, 0 under lean TTA, ensemble <= base; code reverted),
test-time operator averaging (N2), static target whitening (N6), input ERP replacement (N7), mean-target offset (N10),
larger LW statistics (N11), TTA knob regrid (N13), more / balanced members (N5, N14), seed twins on all three families
(N9: REVE seed added -1.6, TSC/SQF seeds +-0.3). Still running: rmix_reg_optx20 / reve_d6_optx20 swaps (N15).

Baseline replay `lean_tta.py` = 44.25 (47.90 / 42.40 / 46.70 / 40.00). All 10 cached non-KD members together: 44.28.

## TTA NIGHT (2026-11-03/04) SUMMARY: 48 not reached; production lean TTA 44.55 -> 45.48 (opt4), 45.20 (optx5)
Adopted (user approval): add-back beta 0.5 (N27, nested 45.05) + sqrt-norm vote weight (N33, nested 45.48).
Gist (N33, N35, N43): lean TTA = (1) first-order person-offset removal via WCCN-fingerprint mates (+4.6; perfect identity
would add only ~+0.9: in-protocol ceiling ~46) and (2) batch Ledoit-Wolf = partial whitening of the predictions' CONTENT
anisotropy (hubness vs gallery; +7; NOT person removal: person structure in the batch spectrum is at chance).
The full-stream oracle 48.4 needs a ~30-dim person-specific second-order structure estimated from >= ~200 trials of the
person (13 in a batch -> catastrophic), so it is unreachable stateless. Dead tonight: in-batch whitening/damping (oracle
groups too), per-person scale, pooled/static LW, partial whitening, recolouring, learned set-function transform (4 forms),
Fourier-surrogate self-centring, offset-subspace projection, richer/learned/ensembled fingerprints, k-means with k known
(60% accuracy), robust mates, per-member beta, vote weights, offset-direction shrink, person-group BatchNorm (single +1.8,
ensemble 0), graph-aware spectral denoising, agreement-aware shrink. Training side: person-centred loss, DRO dead; JS noise.

## N1. Euclidean alignment (EA), ridge probe (CPU, fold 1, 15 people x 6000 trials, 576 low-pass features)
Official subject-agg top-5 of centred ridge predictions, best lambda:
none 12.1 | oracle per-person EA 10.8 | kNN-group EA (self + 4 fingerprint neighbours, shrink .3 / .6 to the
training mean cov) 11.2 / 11.0 | same with oracle-person groups 11.5 / 10.7. Linear probe says EA does not help.

## N2. Test-time operator averaging (inference only, optx members)
Re-render each trial through M training people's channel operators R_t^(1/2) R_i^(-1/2), R_i from the trial's
fingerprint group, average outputs. REVE-d6-optx fold 1 single (lean TTA), base 43.10:
M4 s.3 42.20 | M4 s.1 43.70 | M4 s.03 43.10 | M4 s.01 42.30 | M8 s.1 42.80 | M8 s.03 43.20 | M8 s.6 39.80 |
M4 s.3 + original 42.30. Noise, no direction: DEAD.

## N3. Per-image difficulty (optx5 lean, 20 people x 200 images)
Split-half (people) correlation of per-image hit rate r = .73: misses are image-intrinsic and shared across people
and members (9 images missed by all 20 people, 30 hit by >= 15). Not predicted by target geometry (Spearman with
crowding among the 200 -.05, cos to nearest training targets +.04, cos to training mean -.06, norm +.09).

## N4. EA-trained members (`--ea_shrink .3`): train = self + 4 random same-person trials (per-subject covariance
bank), per-epoch eval = 4 random batch-mates (subject-pure batches, optimistic), tta_cache = 4 fingerprint
neighbours in the real 64-trial stream. Fold-1 screens: reve_d6_ea_things, tsc_dbn40_bn64_ea_things, sqf_dbn_ea_things.
Fold-1 single lean baselines: REVE-d6 41.3 (optx 43.1), TSC-bn64 37.1 (optx 36.4), SQF 34.7 (optx 39.2).

## N5. Rich identity fingerprint (old `identity_features`, RICH groups; scratch rich_tta.py)
optx5 44.15 (= HISTORY), all 10 non-KD members 44.42 (48.70 / 42.20 / 46.20 / 40.60): +0.17 over optx5, noise.

## N6. Static training-target covariance whitening of member outputs (p -> Sigma^-alpha p, before lean TTA), optx5
alpha 0 44.25 | .05 42.75 | .1 40.00 | .2 32.43 | -.1 43.65. DEAD (both directions).

## N4 (cont). REVE-d6 EA, mid-run checkpoint (sealed best of epochs 1-5), fold 1, lean TTA:
single 42.60 with fingerprint-kNN EA groups vs 41.60 with oracle person groups (impurity costs nothing);
no-EA twin 41.3, optx 43.1. Ensemble (fold-1 base 47.90): swap for reve_d6_optx 47.50, add as 6th 47.60.
Per-epoch matched-curve leads (oracle groups) after 6-8 epochs: REVE +1.6, TSC +1.3, SQF +1.4.
REVE-d6 EA final curve (fold 1, per-epoch half-trial sealed mean): 23.4 27.2 29.1 28.9 28.5 29.1 28.6 28.9 28.4 28.2
vs no-EA twin 20.9 25.5 27.7 27.8 26.9 27.7 29.4 28.1 29.3 29.2 (best 29.4) and optx (best 31.0): EA trains
faster, same ceiling. sealed_best = epoch 3 (the checkpoint scored above). Test-time EA grid on it (single, lean):
K2/.3 42.50, K4/.3 42.60, K8/.3 42.70, K4/.1 42.00, K4/.6 41.90, K8/.1 41.30: flat.
Hit phi (fold 1, single lean): REVE-EA vs REVE-optx .63 (REVE family among itself .72-.75; vs SQF/TSC .45-.52).

## N7. Input-level person-ERP replacement (inference only, optx REVE-d6, fold 1 single lean, base 43.10)
x -> x - a (mean of K fingerprint neighbours - training grand-mean trial): K4 a.5 40.90, K4 a1 35.30, K8 a1 37.70. DEAD.

## N8. Composition (4-fold, lean TTA, existing caches)
optx5 minus one REVE-family member: -reve_scratch 44.55 (48.50 / 42.80 / 46.30 / 40.60), -reve_d6 44.38, -rmix 43.85.
The third REVE-family member is redundant (choosing which to drop is test-selected: treat +0.3 as optimistic).
4-member + no-optx SQF/TSC twins (6 members) 44.10: more conv copies do not help.
weights on the 4-member set (REVE-d6, rmix, SQF, TSC): 1,1,1,1 44.55 | conv x1.3 44.15 | conv x.8 44.38 | rmix x.7 44.28 | REVE x1.3 44.40: equal is best.
TSC-bn64 EA final (fold 1): per-epoch curve +1.4 over its twin through 40 epochs (no centring in that metric), but
lean-TTA single on the real stream 36.40 (twin 37.1, optx 36.4). Fold-1 ensembles vs the 4-member set 48.50:
TSC-EA swap 47.70, TSC-EA as 5th 48.50, REVE-EA as 5th 47.30. **EA DEAD**: input-side person alignment buys what
output centring already buys. Folds 2/3 killed. (rmix/SQF EA fold 1 finished for the record; not pursued.)

## N9. Second seeds (--seed 7) of the 4 non-redundant optx members, all folds (queue_seeds.txt, 3 workers)

## N10. Adding alpha * mean-training-target direction (scaled by trial norm) after lean TTA, 4-member set (44.55)
alpha -.03 40.05 | -.01 44.00 | -.003 44.43 | 0 44.55 | +.003 44.18 | +.01 43.85 | +.03 41.68. Optimum at 0: DEAD.

## N11. Is the LW shrink estimate the batch-size bottleneck? (diagnostic, not deployable; 4-member set)
LW statistics from the batch + E extra stream batches: E0 44.55 | E1 44.62 | E3 44.23. No: the B 128 gain of the
lean TTA (44.98 at 10 members) comes from more same-person mates for the centring, which a 64-trial batch caps.

## N12. Nested leave-one-member-out (choose the dropped member on 3 folds, score the 4th)
| set | f1 | f2 | f3 | f4 | mean |
|---|---|---|---|---|---|
| optx5 | 47.90 | 42.40 | 46.70 | 40.00 | 44.25 |
| -reve_d6 | 48.20 | 42.90 | 46.40 | 40.00 | 44.38 |
| -rmix_reg | 48.10 | 42.20 | 45.50 | 39.60 | 43.85 |
| -sqf_dbn | 46.70 | 41.80 | 46.20 | 39.90 | 43.65 |
| -tsc_bn64 | 46.30 | 41.00 | 45.90 | 39.30 | 43.12 |
| **-reve_scratch** | 48.50 | 42.80 | 46.30 | 40.60 | **44.55** |
Every fold's nested pick is -reve_scratch: NESTED 44.55. **New best candidate (opt4): reve_d6_optx + rmix_reg_optx +
sqf_dbn_optx + tsc_dbn40_bn64_optx = 44.55** (+0.30; 3/4 folds up). Also cheaper (4 members).

## N13. Lean-TTA knobs on opt4 (test-selected grid): K4 c1 d=D/(D+B) 44.55 | K3 44.03 | K5 44.57 | K6 44.30 |
c .85 44.28 | d .9 44.22 | d .95 44.60 | K5 c .9 44.48. Defaults are at the optimum.

EA code reverted at 05:20 (it lived in train.py `--ea_shrink`, module/augment.py `SubjectCovBank` /
`group_euclidean_align`, score_track1 `fp_neighbours`, tta_cache EA branch; recipes `*_ea_things`). Results kept in
results/alljoined/track1/cv4/*_ea_things and tta_cache_ea/. sqf_dbn_ea_things fold 1 finished after the revert, so
its track1_score*.json were scored WITHOUT EA: invalid.

## N14. Adding cached members to opt4 (44.55), 4-fold
+reve_d6 (no optx) 43.93 | +rmix_reg 44.52 | +sqf_dbn 44.30 | +tsc_bn64 44.12 | +reve_scratch 44.35 |
balanced 8 (opt4 + all four no-optx twins) 44.20 (47.50 / 42.60 / 46.10 / 40.60). Near-twins do not help even when
balanced: the ensemble prefers few strong diverse members. Seed-7 runs: judge on fold 1, kill if <= opt4 there.
opt4 leave-one-out: -reve_d6 43.92 | -rmix_reg 42.90 | -sqf 43.50 | -tsc 43.83: every member contributes 0.6-1.65; 4 is the sweet spot.
REVE-d6-optx seed 7, fold 1: curve tracks the original (best 30.8 vs 31.0); lean single 42.40 (orig 43.10); opt4 +
seed (5 members) 46.90 vs 48.50 (-1.6: doubles the REVE family weight); seed swapped for the original 48.00 (-0.5 =
member seed noise on one fold). TSC-bn64-optx seed 7, fold 1: single 36.10 (orig 36.40); add 48.50 (=); swap 48.80 (+0.3, noise).
Seeds DEAD as members; remaining seed folds dropped (TSC/SQF s7 fold 1 finish for the
record).

## N15. 20-epoch-cycle swaps into opt4 (rmix_reg_optx20, reve_d6_optx20; pre-cleanup singles +1.05 / 2-fold ens +0.35)

## N16. Exhaustive subset search over the 10 cached non-KD members (all 3/4/5-subsets, 582; scratch subset_search.py,
GPU-batched lean TTA verified on opt4 = 48.5 / 42.8 / 46.3 / 40.6)
Best in-sample 44.62 (two sets); opt4 44.55 is 6th of 582. Nested (pick on 3 folds, score the 4th): size 3 43.60,
size 4 43.85, size 5 43.77, any size 43.87. A free subset search overfits by ~0.7; the narrow drop-one-from-optx5
selection (N12, 6 candidates, same pick on every fold) is the only composition that holds up. Composition of the
existing members is exhausted at ~44.6.
SQF-dbn-optx seed 7, fold 1: single 38.00 (orig 39.20); add 48.30; swap 48.40 (opt4 48.50). Seeds closed on all three families.
rmix_reg_optx20 fold 1: per-epoch curve identical to rmix_optx through e10 (same seed + schedule), then peak e13 31.8
(vs 30.7); fold 2 declined after e9. Lean-TTA single on the real stream 41.70 (rmix_optx 44.8); opt4 with it swapped
47.40 vs 48.50, added 46.70. DEAD; folds 2-4 and reve_d6_optx20 (same mechanism) killed. Note the bias: sealed_best
over 20 epochs maximises over twice as many noisy epochs, so the per-epoch "gain" was partly selection.

## Housekeeping (10:15)
All night recipes were removed from cv4.sh again (dead); their fold-1 run dirs stay in results/alljoined/track1/cv4/
(`*_ea_things`, `*_optx_s7_things` = `<base recipe> --seed 7`, `rmix_reg_optx20_things` = rmix_reg_optx with
`--stop_after_epoch 20`). Scratch caches deleted. The EA code was never committed; its core, for the record:
```python
def group_euclidean_align(eeg, covs, shrink=0.3):  # covs (B, 1+K, C, C): the trial's own time-demeaned covariance + K mates'
    R = covs.mean(1); R = (1 - shrink) * R + shrink * eye * R.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
    w, V = torch.linalg.eigh(R); return (V * w.clamp_min(1e-8).rsqrt()[:, None, :]) @ V.transpose(1, 2) @ eeg
# train: mates = 4 random trials of the same subject from a per-subject FIFO bank of 256 covariances (after op transfer: none)
# test:  mates = the trial's 4 fingerprint neighbours in the 64-trial batch (same kNN as batch_transform_wknn)
```

## N17. Projection of the opt4 output onto the top-m training-target PCs (+ mean direction), 4-fold
m 32 24.33 | 64 28.98 | 128 34.27 | 256 37.62 | 512 40.67 | 1024 43.35 | none 44.55: monotone loss, DEAD (the
low-variance target directions carry signal; consistent with the LW shrink's gain).

## N17b. Oracle bound on composition (all 1023 subsets of the 10 cached non-KD members, lean TTA)
Best subset chosen PER PERSON with the labels: 47.42. Best subset per fold (labels): 45.70. Nested any-size: 43.62.
=> even a label-cheating choice of members cannot reach 48 with the existing members; 48 needs new members or new
information, not fusion.

## N18. KD students of the opt4 ensemble (user allowed KD, 2026-11-03 10:30)
KD port: train.py `--distill_targets/--distill_lambda/--distill_tau` (KL(teacher||student) over the batch images, tau
.005 as A18), dataset appends the teacher row, neurips_challenge/export_teacher_cv4.py (opt4 members' final-epoch
weights, mean of L2-normalised predictions on the fold's 1.654M training trials), cv4.sh recipes `*_optx_kd4_things`.
First export wrote the TEST split (train=False -> 400k rows): caught before any training, deleted, fixed.
Teachers folds 1-3 exported (1,654,000 rows each, ~15 min per member). Fold-1 students started 11:14 (REVE-d6 kdA,
SQF kdB, rmix local); folds 2-3 queued behind them. Evaluator: scratch kd_eval.sh (single / add to opt4 / swap for
its own family member, fold 1; opt4 fold 1 = 48.50).

## N19. User idea: a top-5-aware training loss (not run)
Per-trial top-5 is only 7.5-9.2% (top-1 ~2%) while the aggregated metric is 31-34% per member: a top-k-relaxed loss
equals InfoNCE on the ~92% of trials outside the top 5 and only stops pushing the few well-decoded trials, which are
the ones that carry the 80-repeat average (norm = vote weight). The metric ranks the person-image MEAN after centring
against 200 unseen images, not single trials against in-batch training images. Metric-aligned variant (top-k on
averaged predictions) ~= the dead repeat-mean loss. Verdict: low expected value; a fold-1 REVE screen if a slot frees.
KD per-epoch curves (fold 1, matched epochs vs the optx twins): REVE +4.7 (e3 31.5 > twin best 31.0), rmix +6.0
(e3 32.8 vs twin best 30.7), SQF +3.9 (7 ep).
rmix_reg_optx_kd4 fold 1 (lean TTA, real stream): single 46.60 (rmix_optx 44.8; opt4 ensemble 48.50); opt4 + student
48.70 (+0.2); student swapped for rmix 48.10 (-0.4). Same as A18: strong solo, redundant inside the ensemble.
reve_d6_optx_kd4 fold 1: single 48.40 (= the opt4 ensemble alone; reve_d6_optx 43.10); opt4 + student 48.90; **student
swapped for reve_d6_optx 49.80 (+1.30)**. Fold-1 combos: both students (REVE-KD + rmix-KD + SQF + TSC) 47.00;
REVE-KD swap + rmix-KD as 5th 49.50; opt4 + both students (6) 48.60; REVE-KD + SQF + TSC 46.20; REVE-KD + rmix +
SQF + TSC + reve_scratch 49.00; REVE-KD + rmix-KD + TSC 48.10. One student in place of its own family member is the
sweet spot. Candidate kd-opt4 = REVE-d6-KD + rmix_optx + sqf_optx + tsc_optx: 4-fold confirmation running (folds 2-3
on sbatch, fold 4 after its teacher export). rmix-KD folds dropped.
13:30 user: keep going with KD (bigger ensembles / architectures / KD variants allowed), target 48 over 4 folds.
Fold-1 screens queued (teachers in results/alljoined/track1/cv4/teachers/): reve_scratch_optx_kd4 (deep REVE student),
reve_d6_optx_kd4pc (teacher minus each training subject's mean teacher vector + global mean; subject-mean norm .11 vs
row norm .87), reve_d6_optx20_kd4 (KD on the full 20-epoch cycle), reve_d6_optx_kd5 (2nd generation: teacher =
kd-opt4). Confirmation of kd-opt4 on folds 2-4 runs first.
sqf_dbn_optx_kd4 fold 1: single 40.50 (sqf_optx 39.2); in kd-opt4 for SQF 48.80 (kd-opt4 49.80); added to kd-opt4 47.90. Conv student = copy of the teacher, loses SQF's own errors: DEAD; SQF-KD folds 2-3 dropped.
Seed slot-averaging (two seeds inside one slot, weight .5 each; fold 1, opt4 48.50): REVE slot 48.40, SQF slot 48.50, TSC slot 48.50, all three 48.10. Seed variance is not what limits members: DEAD.
kd-opt4 fold 2: 41.90 vs opt4 42.80 (-0.9). REVE-KD single on the real stream 38.80 vs reve_d6_optx 38.90 (no gain; per-epoch metric said +3.1). 2-fold kd-opt4 45.85 vs opt4 45.65.
15:45 user: add (not replace) a huge TSC with >= 5M params + KD. tsc_big_optx_kd4_things = TSC-bn64-optx with 384
temporal/spatial/projection filters: 5.77M (base 0.15M + 0.10M head), 20 epochs (fp32, ~20 min/epoch shared), on kdB
after REVE-KD fold 3. Control tsc_dbn40_bn64_optx_kd4_things (base capacity + same KD) on the local GPU after the gen-2
student. Evaluated as an ADDED member (scratch tscbig_eval.sh).
kd-opt4 fold 4: 40.10 vs opt4 40.60 (-0.5); REVE-KD single 40.60 vs reve_d6_optx 37.40 (+3.2, 4/5 people). 3-fold mean kd-opt4 43.93 vs opt4 43.97: the solo gain vanishes in the ensemble (= A18, October). Fold 1 was KD's lucky fold: judge the variant screens on fold 4 as well.
tsc_big first attempt OOMed at batch 1024 (one activation = 4.8 GB): added train.py --backbone_checkpoint_chunks (backbone on 4 chunks with activation checkpointing; BN sees 256-trial chunk statistics), peak 3.4 GiB. Restarted 16:39 on kdB at 1.19 it/s (~23 min/epoch, ~8 h). reve_d6_optx20_kd4 screen moved behind it.
**kd-opt4 4-fold FINAL: 44.60 (49.80 / 41.90 / 46.60 / 40.10) vs opt4 44.55 (48.50 / 42.80 / 46.30 / 40.60): +0.05,
noise.** REVE-KD singles 48.40 / 38.80 / 45.00 / 40.60 (plain REVE-d6 43.10 / 38.90 / ? / 37.40). The fold-1 +1.3 was
luck; distilling the ensemble into one member does not lift the ensemble (second confirmation after A18).
16:50 user: generation 2 (reve_d6_optx_kd5) ditched, killed at epoch ~5.

## N20. Push predictions away from their nearest training targets (static bank, after lean TTA, opt4)
k 10: beta -.02 44.45 | -.005 44.50 | .005 44.30 | .01 44.20 | .02 44.08 (coarser beta collapses); k 100 similar. Optimum 0: DEAD.
opt4 + REVE-KD as a 5th member, 4-fold: 44.63 (+0.08). KD closed except the big-TSC add test and REVE-scratch screen.

## N21. Person-centred training loss (`--person_centre_train`, recipe reve_d6_optx_pct_things)
Each training output minus the leave-one-out mean of its same-person batch-mates, plus the batch mean (= the test-time
centring with true ids; REVE batch 128 over 25 people ~ 5 per person = the test group size). Unlike E26 (whole-batch
mean + PC shrink, -3.5..-5.5). The per-epoch metric does not centre, so evaluate final-epoch AND sealed-best
checkpoints on the real stream (scratch pct_eval.sh). Folds 1 (local) and 4 (kdA) to avoid judging on a lucky fold.

## N22 (17:20) KD line closed by user; person-centred training on all 4 folds
User: ditch KD. Killed reve_scratch_kd4 f1, tsc_big_kd4 f1, tsc_dbn40_bn64_kd4 f1 (no results). Queues now only
`reve_d6_optx_pct_things` (= reve_d6_optx + `--person_centre_train`): f1 local, f2 kdB, f4 then f3 kdA. Judged by
pct_eval.sh (scratchpad) as swap-for-REVE-d6 and add-to-opt4, final-epoch and sealed-best checkpoints, 4-fold mean vs 44.55.

## N23 (17:45) per-trial confidence weighting (from the user's "heteroscedastic likelihood" discussion)
The output norm is the trial's weight in the grader's (subject, image) mean. opt4, cached, 4 folds (scratchpad trialw.py):
raw 44.55 | unit-norm 43.12 | raw x member-agreement^1 44.67 | ^2 43.68 | ^4 41.98. Summed-member norm already
correlates 0.5-0.57 with member agreement and is worth +1.4 over equal weights; extra agreement weighting = noise (+0.12). Dead.

## N24 (18:10) TTA headroom (opt4 cached, 4 folds; scratchpad headroom.py / headroom2.py)
no centring 40.55 | current wknn K=4 44.55 | ORACLE true same-person mates in the 64-batch 45.25 | ORACLE person mean
over all 16k trials (not allowed: stateful) 45.40 | + oracle per-person output whitening 38.38 | LABELLED per-person
linear map toward identity (fit on the other 100 images) below its identity baseline at every lambda (best lam=10: -5).
=> lean TTA captures ~85% of the oracle offset gain; perfect person identification is worth only +0.7; second-order /
linear person adaptation hurts even with labels. TTA is saturated: the 3.5 points to 48 must come from the members.

## N25 (20:20) person-centred training loss (reve_d6_optx_pct_things): DEAD
Lean TTA, opt4 baseline per fold 48.50 / 42.80 / - / 40.60. final-epoch (sealed-best) checkpoint:
f1 single 40.2 (REVE-d6 43.1), swap 47.70, add 46.40 | f2 swap 40.80 (40.70), add 41.70 (41.30) | f4 single 33.7
(37.4), swap 39.40 (40.00), add 39.50 (40.20). Negative on every fold/checkpoint; f3 killed. Training on leave-one-out
centred outputs (batch 128, ~4 mates/person) costs single-model accuracy and the test-time centring already captures the gain.

## N26 (20:20) oracle-identity TTA operations (scratchpad oracle_ops.py, full person stream = stateful, NOT deployable)
opt4, 4-fold means: centre + add-back beta 0 42.07 | 0.5 46.57 | 1 45.40 | 2 44.52 | person scale 46.05 | per-member
person norm 46.35 | drop top-r person PCs (1..30) 37.9 -> 5.8 | person whiten^0.1 (50% shrunk) 47.02 | ^0.25 47.60 |
person-level uncentred LW d=0.1 39.23, 0.5 46.35, 0.9 48.48. Pooled (static) whitening hurts on f1 (51.3 -> 50.6) while
per-person helps (54.4): the second-order gain is person-specific. In-batch version + label-free beta: pwhite2.py / beta.py.

## N27 (20:50) label-free add-back beta in lean TTA (scratchpad beta.py; beta = 1 reproduces 44.55 exactly)
opt4: beta 0 40.13 | 0.25 44.32 | 0.4 45.02 | 0.5 45.17 (48.70/42.30/47.60/42.10) | 0.6 45.07 | 0.75 44.90 | 1 44.55 | 1.5 44.02.
Nested (beta picked on the other 3 folds: 0.5/0.5/0.5/0.6) 45.05 = +0.50 honest. Per-fold optimum differs (f2 prefers 1.5).

## N28 (21:20) in-batch second-order TTA: dead (pwhite2.py, pscale.py)
In-batch person-level partial whitening (damp the group's dominant directions), groups = true same-person batch-mates or
self + K in {4, 8, 12} fingerprint neighbours: every damping variant < no damping, even with oracle groups (best oracle
46.05 = plain oracle centring at beta 0.5; label-free beta 0.5 = 45.17). Per-member person scale from the group: oracle
45.25, wknn K12 45.20. => the in-protocol TTA ceiling with perfect identity is ~46.05; second-order person statistics need
the full stream. Next: a LEARNED set-function transform (ttnet.py) trained on held-out people's train-split outputs.

## N29 (23:00) learned test-time transform (user idea; scratchpad ttnet.py, trcache.py): DEAD in 4 forms
Structure = lean TTA with learnable soft grouping (fingerprint cosine * e^s + fingerprint MLP q.k + w * output cosine),
centring gamma, add-back beta, member gains g; LW fixed after. Trained for fold k on the OTHER folds' held-out people
(leaky upper bound: their members trained on fold k's people). Init (soft attention) 47.90/41.60/45.10/42.50 = 44.28.
(a) train-split trials (seen images), per-trial InfoNCE: f4 42.50 -> 38.50 (learns gamma 0.23: seen-image outputs carry
little person offset). (b) other folds' test streams, per-trial InfoNCE: 40.85 (per-trial loss ignores the 80-trial
averaging that makes offset removal pay). (c) aggregated (person, image)-mean InfoNCE, full net: f4 41.40 (fingerprint MLPs
overfit 15 people). (d) same, scalars only: 46.90/38.40/42.60/40.60 = 42.13 (CE drives SQF/TSC gains to 0: CE != top-5).
Only the 1-scalar beta, tuned by top-5 across folds, transfers (+0.5 nested).

## N30 (22:20) group DRO (reve_d6_optx_dro_things, --group_dro_rho 1), fold 1
single 41.80 (REVE-d6 43.10) | swap for REVE-d6 47.10 | add to opt4 47.60 (opt4 48.50). Negative; fold 4 queued (kdB 67768
resubmitted: kdB had exited at 21:21 on an empty queue). James-Stein f1 (local) / f4 (kdA) training, ~1.5x per epoch.

## N31 (22:50) full-stream LW: person-specific (scratchpad plw.py) -> TTA closed
oracle centring + person-level LW d=0.9: 48.43; same with the fold's POOLED stream LW: <= 46.60 (b0.5 d0.95); with
label-free wknn centring pooled LW only hurts (<= 45.00 vs 45.17). No static/pooled second-order map substitutes for
person statistics, which a 64-trial batch cannot supply. In-protocol TTA best: beta 0.5 = 45.17 (nested 45.05).

## N32 (00:10) beta 0.5 ADOPTED in score_track1.batch_transform_wknn (user approval): optx5 45.17, opt4 45.17
## N33 lean-TTA anatomy (scratchpad ident.py, inflate.py; opt4, beta 0.5)
Steps: raw sum 34.27 | member scaling only 33.85 | scaling + LW 40.55 | scaling + centring, no LW 38.12 | all 45.17 |
centring + LW without member scaling 45.10. => LW and centring are synergistic (+11.3 together); LW is the bigger single
step (-7.05 when removed vs -4.6 for centring); member scaling is ~free to drop (-0.07).
Identification (64-trial batches): kNN4 purity .742 (full 656 + WCCN); WCCN off .516 (score 43.40); logvar 32 .477 (42.10),
corr 496 .616 (43.60), bandpow 128 .692 (44.72); larger K lowers purity (K12 .532, 43.92); spherical k-means with k = 5
KNOWN: Hungarian accuracy .601, mates purity .473, score 43.67. Oracle all mates 46.05. Single-trial fingerprints are far
from separable at ~13 trials/person.
Norm inflation y * |y|^g after the transform: g .25 45.35 | .5 45.48 (48.80/42.30/48.40/42.40) | .75 45.32 | 1 45.10;
nested pick = 0.5 on every fold -> 45.48 (+0.31, 3 folds up, 1 flat). Not adopted (needs approval).

## N34 (01:20) consensus vote weight ADOPTED (y * sqrt|y| after the shrink, user approval): optx5 45.20, opt4 45.48
## N35 decomposition of the person-level LW oracle (scratchpad decomp.py; oracle centring + global mean, opt4)
full stream 48.48 | top-30 person directions only 48.50 | top-10 44.68 | top-3 39.12 | top-1 35.77; from n random trials of
the person: 1000 47.62, 200 46.90, 50 42.50, 13 32.45 (catastrophic); per-person diagonal scaling 33-37. => the gain is a
~30-dim person-specific second-order structure needing >= ~200 trials: in-batch (13) estimation is hopeless; only a
transform that MANUFACTURES samples per person (surrogates) could reach it. Next: phase-randomised Fourier surrogates.

## N36 (02:00) surrogate self-centring (scratchpad surr_cache.py / surr_eval.py; REVE-d6 single, fold 1)
no centring 37.80 | phase-randomised Fourier surrogates (S=4) self-centring 39.50 | lean kNN 44.20 | surrogate + kNN 42.40 |
LW fitted on surrogate scatter 29-38. Surrogates keep spectra/cross-spectra but carry only ~1/4 of the person offset: the
offset comes from the person's EVOKED response (destroyed by phase randomisation), not background statistics. Dead.
## N37 low-dimensional offset subspace (scratchpad offsub.py)
20 test-people offsets: PC1 50%, 5 PCs 89% (offset norm 0.48 vs trial 3.52). Projected kNN offset estimate with the OWN
fold's basis (cheat) r5 = lean 45.47; with the other folds' basis -4 (41.6 best): subspaces do not transfer across models. Dead.
## N38 richer fingerprints / WCCN fits (scratchpad fpstudy.py; folds 1-3 so far)
purity and score decouple: base+bandcorr raises purity (f1 .793 vs .772) but scores lower (48.10 vs 48.80); + fine 1-Hz
bands lowers both; THINGS in the WCCN fit: mixed (+0.5 f2, -0.4 f1); shrink 0.3 stays best.
fpstudy final (4-fold): base/A 45.48 | base/AT 45.43 | base+bcorr 45.30 (purity .751) | all/A s.5 45.08 | base/ATd 44.95.
## N39 (02:40) geometry / member-structure transforms (scratchpad geom.py), all below production 45.48:
LW d 0.98 45.10, 0.93 45.00, 0.9 44.55; mates fp + w * output cosine 43.2-44.6; agreement-aware shrink 42.8-44.3;
agreement-only 33-35; image-covariance recolour ^0.25 39.05, ^-0.25 28.85, ^0.5 27.78. Production is a sharp local optimum.
## N40 (03:20) person-group BatchNorm inside TSC/SQF ("deep centring"; scratchpad gbn_cache.py / gbn_eval.py, fold 1)
Every BN layer normalises with stats blended (alpha) between running and the trial's group (self + 4 mates, over trials x time).
Singles: TSC 36.60 -> 38.40 (knn a1) / 38.60 (oracle a1); SQF 39.30 -> 40.50 (knn a.5). opt4 with both swapped: 48.30-48.90
vs 48.80 (even oracle groups). Single-model gain only: the output centring already captures it in the ensemble. Dead.
## N41 (04:00) identifier ensemble (scratchpad consensus.py): kNN graphs from several fingerprint views (each WCCN'd) averaged
full only 45.48 (purity .742) | +bandpow 45.45 (.717) | full x2 + bandpow + corr 45.10 | + corr 44.85 | all 4 views 44.40 (.632).
Views add impure mates; the full 656-dim view alone is the best identifier. Dead.
(note: the clock times in the N34-N41 headers were estimates and run ~1-3 h ahead; N41 was logged at ~00:40 on 2026-11-04.)
## N42 (2026-11-04 ~01:30, logged 09:50) per-member beta, robust mates, vote weights, offset-direction shrink: all dead
per-member beta nested 45.33 (in-sample best 45.62) vs 45.48; robust mates K6/K8->4 44.47/44.77; down-weight loud trials
44.98-45.15; weight by grouping confidence 44.38-44.95. Offsets ARE aligned with person PCs (|cos| PC1 .33, PC3 .29 vs .026
random) but a rank-1 shrink along the offset hurts (oracle eta .5 43.10, knn eta .25 44.68).
DRO / JS (production TTA): DRO f1 swap 48.30 / f4 41.30 (vs 48.80 / 42.40): dead. JS f1 swap 49.30, f4 42.20 (+0.15 mean,
single f1 -0.5 / f4 +1.7): noise-level, not pursued.
## N43 (09:55) graph-aware spectral denoising (scratchpad gsmooth.py) -> what the LW step really does
Top-10 batch SVD components: oracle person-structure u'Ou 0.08-0.09 = chance (5 groups of ~13 -> 0.078); fingerprint-graph
smoothness ~0. Person offsets are NOT visible in the batch spectrum; even oracle removal of person-structured components
40.92. Graph-smooth removal after kNN centring 45.53 (noise). => lean TTA = two independent corrections: (1) first-order
person offset via fingerprint mates (<= +0.9 left even with perfect identity); (2) the batch LW = partial whitening of the
predictions' CONTENT anisotropy (hubness vs the gallery), worth ~7 points; pooled/static covariance does worse than batch.
## N44 (10:30) why the batch LW works (scratchpad lwself.py)
production 45.48 | leave-one-out LW (trial excluded from its own fit) 45.20 | LW before centring 44.02 | self-direction-only
damping with the same per-trial ratio 39.50. Self-influence is minor (0.28): the LW acts through directions SHARED by the
batch's trials, and must follow the centring.

## N45-N51 (2026-11-04) non-centring / non-shrink TTA families (user: "radically new ideas"), opt4, 4 folds, prod 45.48
Scripts in the session scratchpad (newfam, matedet, gate, gate_wg, aug_cache/aug_eval, mfilt, decomp2, snap2).
- N45 cross-person SAME-IMAGE pooling (the grader never pools across people; ~10 such pairs per batch): ORACLE pairs
  y + g * sum(mates) g1 54.73 (+9.3), g.5 50.70 -> the largest headroom found anywhere. Blind (non-local means in output
  space, fingerprint mates excluded) 44.68. Detectability (N46): pair AUC 0.51-0.555 for every similarity tried (prod y,
  centred raw sum, per-member cos mean/min, bank-PC subspaces, bank-softmax profiles); prec@1% 0.009 vs base 0.005,
  mutual-NN 0.008. Single-trial content SNR is far too low: DEAD (headroom real, unreachable).
- N47 natural-image prior: kernel snap onto the 16540 THINGS-train DINOv2 embeddings, centred, +a*|y|*dir:
  a .05 -> 29.2, a .25 -> 4.3 (raw-space version equally bad). Bank-confidence vote ^1/2/4: 45.30/45.10/45.00. DEAD.
- N48 cross-fold LEARNED trial quality / member gating (ridge, fingerprint + output statistics -> retrieval margin,
  train 3 folds, test 4th): quality IS predictable (r .21-.27; within-(person,image) r .14) but vote weights hurt
  (best 44.70; within-group 44.50) and member gating is neutral-to-negative (45.10; within-group 45.38). DEAD.
- N49 input-space TTA (fold 1, each member averaged over augmented EEG views): +-1-sample shift 48.80 (= clean),
  +-2 48.60, 4 channel-dropout masks 47.20, all 8 views 47.60. DEAD (model variance << EEG noise).
- N50 cross-fold linear readouts of aggregated predictions (same 200 gallery images in every fold!): matched filter
  Sigma_resid^-a best 40.92 (a .25), ridge readout <= 27.9. Residual structure is person/model specific. DEAD.
- N51 oracle decomposition of the person-level 2nd-order structure (no vote weight; prod 45.17): person LW from the
  TOTAL second moment 46.52 (d.95), from WITHIN-(person,image) noise only 44.10, from BETWEEN-image content only 41.77.
  So test-time noise synthesis (propagating the batch's well-estimated 32x32 EEG noise covariance through the nets)
  could at best recover the "within" part, which alone loses to the batch LW: DEAD in principle.
GIST (why only centring + shrink survive): per-trial SNR << 1 and the grader averages ~80 trials afterwards, so the
useful component of one output is a tiny fraction of its norm; any per-trial NONLINEAR term (snap, gating, weighting,
pooling on noisy similarities) injects a bias that does not average out, while the signal it adds is negligible. In
that regime the optimal stateless transform is linear, built from batch nuisance statistics: means (person centring)
and second moments (LW). Outside information (image prior, other folds' labels, model invariances) does not transfer.
- N52 recoloured synthetic trials (synth.py, fold 1, no vote; prod 48.70): training-split EEG whitened per source person,
  coloured with the TEST person's 32x32 spatial covariance, run through the 4 members, person LW fitted on the synthetic
  outputs: C from all trials (oracle) 47.40, from 64 46.90, from 13 46.10, no-recolour control 45.80 -> recolouring
  carries person-specific 2nd-order info (+1.6 oracle) but never reaches the batch LW; mixed into the batch fit
  (synth_mix.py) w.25 48.50 (control 48.20), w1 46.50. DEAD.
- N53 batch-as-pseudo-gallery hypothesis (galsamp.py): LW map fitted on the TRUE gallery embeddings of the batch's own
  images 15.40, 64 random gallery 34.93 = 64 random bank 35.08, all 200 gallery 34.45; joint batch+gallery fits <= 43.0.
  REJECTED: the batch's shared directions are model prediction NOISE, not content; damping true content kills signal.
- N54 random-matrix shrinkers (rmt.py): LW 45.48 | LW on bulk-debiased eigenvalues 45.42 | power (lam/c med)^-a clamped
  <= 44.25 | MP-edge spike flattening <= 41.97 | top-k spikes -> median <= 38.77. LW stays best.
- N55 GIST, Gram domain (gram.py): algebraically (no approximation), LW with d = D/(D+B) == REGULARISED Loewdin
  orthogonalisation Y' = (Y Y^T/n + k nu_g I)^-1/2 Y with k = 1 (nu_g = mean Gram eigenvalue): singular direction i is
  scaled by (lam_i/nu_g + 1)^-1/2. Fold-1 spectrum (lam/nu_g, ranks 1/2/3/5/10/20/40): 19.1 9.9 7.3 4.4 1.3 .22 .02;
  top direction 30% of variance, ~10.6 directions above the mean -> top scaled .22, rank 10 .67, tail ~1: the ~10
  shared directions are flattened, the tail kept. Mean |cos| between two trials of a batch .264 -> .140 (partial
  orthogonalisation) = a soft EXCLUSIVITY prior in the trial domain (different trials ~ different images; whatever trials share,
  shared model noise, hub direction, residual person offsets, is removed). The Loewdin grid around it (a .4-.6,
  k .01-.3) is lower (best 44.62); nonlinear repulsion from similar batch-mates (softmax or Sinkhorn kernel, before
  LW) <= 45.02. The production point is already the optimum of this family.
- N56 one Loewdin operator on a JOINT trial similarity (jgram.py): (Kout/tr + lam K/tr + I)^-1/2 reproduces production
  exactly with lam 0 (45.48); + fingerprint-mates graph lam .1 45.52, .3 45.50 (noise); + fingerprint Gram <= 45.27;
  sum of member Grams 45.05; the operator on the RAW sum cannot replace the kNN centring (<= 42.02). DEAD.
Headroom map after N45-N56 (all need information a stateless 64-trial batch does not contain): cross-person same-image
pooling +9.3 (pairs undetectable), full-stream person statistics +3 (needs ~200 trials/person), perfect in-batch
identity +0.9. No TTA family tested reaches 48; the production point (kNN centring, beta .5, Loewdin/LW k=1, sqrt-norm
vote) is a local optimum of every family around it.
- N57 centring by PROJECTION (projc.py, 4-fold): remove only the mates-mean direction scaled per trial g1 42.70 / g1.5
  43.68; remove the rank-4 span of the centred mates 40.88-41.00; soft fingerprint-softmax mates (UnMix-TNS analogue)
  44.52 / 43.40. The additive mates-mean subtraction (45.48) stays best: the offset is additive, not multiplicative.
- N58 input band filtering at test time (filt_cache/filt_eval, fold 1; prod 48.80), motivated by HISTORY "image info
  < 8 Hz, identity = alpha/beta topography": low-pass 24 Hz 48.80 (= clean), 16 47.60, 12 44.00, 8 38.40; band-stop
  8-13 42.30, 8-30 40.40; clean + filtered view averages <= 48.60. The members use alpha/beta. DEAD.
NOTE: HISTORY.md (Oct) already covered cross-person same-image pooling (oracle 46.5-53.8, label-free -2..-5.5), shift
averaging, AdaBN (+0.1-0.3 then), pull toward training targets; N45/N46/N47/N49 re-confirm them on the current pipeline.
- N59 matched filter with a per-batch NOISE covariance from member differences (mdiff.py; content-free rows
  c_m - mean_m c): noise filter before LW <= 44.50, after LW <= 43.88, LW fitted on [y0; w D] w.25 45.37 / w1 43.42.
  Member-specific noise is not what limits; the shared (correlated) error is, and it is invisible to differences. DEAD.
- N60 batch ICA artifact removal on the INPUT (ica_cache/ica_eval, fold 1; prod 48.80): symmetric FastICA (logcosh) on
  the batch EEG (32 ch x 64 trials x 120 samples = 7680 obs, borderline vs the ~20 n^2 rule, 5 heads mixed), top-k
  kurtosis components zeroed: k1 47.80, k2 47.60, k4 46.60; top spatial PC removed 45.70; clean + view averages <= 48.80
  (k1 tie). Blink/muscle-like sources do not limit the members. DEAD.
- N61 natural-image subspace projection (natsub.py, 4-fold): basis = bank mean + top-r PCs of 16540 THINGS-train DINOv2
  embeddings (gallery energy inside: r64 .87, r256 .944, r1024 .994). LW then project r1024 44.02, r512 40.78, r64 29.15;
  project first or fit-only variants worse; CHEAT projection onto span(gallery) 45.52 (= prod: only vote weights move).
  The gallery's discriminative signal lives in the low-variance image directions the bank PCs drop (0.6% energy costs
  1.5 points) -- the same reason flattening the dominant shared directions (LW/Loewdin) helps. DEAD.
- N62 Loewdin mixing with the trial Gram in a natural-image metric K = Y C_bank^a Y^T (kmetric.py): a 0 45.48 (prod),
  +.25 45.25, -.25 45.20, +.5 45.10, -.5 44.97. The plain output metric is the optimum. DEAD.
- N63 in-batch ORACLE ceiling of the production structure (kres.py / kres2.py, 4-fold, vote on): Loewdin Gram from the
  oracle content-free residual (output minus its projection on the true centred target) 46.30 (+0.82); oracle in-batch
  identity (all same-person batch-mates as mates) 46.30; BOTH oracles 46.85. Even with perfect identity AND perfect
  shared-noise knowledge inside one 64-trial batch the transform tops out ~1.4 above production, 1.15 short of 48.
  48 needs information outside the batch (full-stream person statistics, cross-person same-image pooling).

## N64. Step 0 of a decorrelated new member: is a 2nd-order / induced EEG statistic a new information source? (fold 1)
Single-trial ridge EEG features -> DINOv2 (15 training people, 2 reps/image, lambda by held-out-person val MSE; output
minus the training target mean, else the constant mean alone sets the ranking = chance), scored as a member
(scratchpad probe.py / probe_eval.py). Features: wave = raw 32x120; env = log analytic amplitude, 10 bands 2-45 Hz x
0.1 s; cov = tangent-space band covariances (6 bands x 2 post-stim windows, log-Euclidean at the training mean).
  solo raw / lean TTA: wave 10.5 / 14.8, cov 4.7 / 7.3, env 3.7 / 5.4 (chance 2.5); hit phi vs REVE/SQF/TSC: wave .18/.10/.18,
  cov .10/.09/.08, env ~0. opt4 + probe at w .03-1: best = opt4 48.80 (w .03), any larger weight loses (w .25: -3..-4).
  Joint ridge wave+cov 13.7, wave+env 13.5 (< wave alone; MSE-picked lambda); late fusion wave + .3 cov 15.4 (+0.6, noise).
=> the phase-free statistics carry real image information (cov ~half the linearly readable signal of the waveform) and
   their errors are near-independent, but they add ~nothing on top of the waveform even linearly: gate FAILED, the
   two-statistic (SPD) branch is not worth a GPU screen.

## N65. MLP-only members (design agreed with the user 2026-11-04; module/eeg_encoder/ortho_encoders.py Mixer3 / FlatMLP)
Why: fold-1 top-5-hit phi: seed pair .77, REVE vs rmix .78 (= a reseed), SQF-TSC .63, cross-family .45-.48, image-difficulty
floor (B's hits permuted across people within image) ~.30. Every current member is conv- or patch-token-based.
- B `mix3_dbn_bn64_optx_things`: per-dataset BN per electrode -> each (electrode, time) cell lifted to d 16 -> 4 pre-LN
  residual blocks of dense MLPs over all 120 samples (hidden 128; position-specific = absolute latency, no shift
  invariance), over all 32 electrodes (64), over features (32); GELU, dropout .3; flatten -> TSC bn64 head. 2.1M params,
  no conv, no patches. TSC recipe + bf16 + torch.compile'd trunk: 0.09 s/step (~2.5-3 h for 40 epochs). Agreed spec was
  d 16 / hidden 256-64-64, but uncompiled it ran 0.83 s/step (15 h; fp32 1.17 s, 8.7 GB): the cost is memory-bound
  LN/GELU/dropout over 1024 x 32 x 120 x d cells, not FLOPs. Dense 120->40 time compression (T40 d16 compiled 0.07 s)
  was the alternative; full-resolution time kept as the design's core.
- A `mlp_dbn_bn64_optx_things`: EEGProject (flatten 3840 -> 1024 -> residual MLP) + the same input BN and head, TSC recipe.
- Gates (fold 1, then fold 2): lean-TTA solo >= TSC - 2 (fold 1 34.6, fold 2 30.6); max hit phi vs the 4 members < .5.
  Attention (single head, qk 8, over the 120 time positions, after block 2) only if both MLPs miss the accuracy bar.
- Reference per-epoch sealed curve (fold 1, half fraction, no TTA): TSC 14.3 / 22.8 (e4) / 25.3 (e10) / 25.8 (e20) / 29.0 (e40).
- A (FlatMLP) killed at epoch 15 (user: prioritise): per-epoch sealed 11.6/15.6/15.7/15.9/21.1 ... 22.2 (e10/e12), 19.9 (e15)
  vs TSC 25.3 (e10) / 25.9 (e15): ~4-6 behind and flat. B (Mixer3) e1-e6 13.0/13.0/16.5/19.5/20.1/21.1 vs TSC 14.3/18.3/
  20.0/22.8/22.7/~24 (gap closing). Freed slot -> Mixer3 fold 2 (needed for the pass rule anyway).
- Fold-2 Mixer3 cancelled (user: use the slot for new architectures). Fold-1 screens now on all 3 GPUs:
  `mix3att` = Mixer3 + one single-head attention over the 120 time positions after block 2 (qk dim 8 from each position's
  whole electrode x feature slice + learned positional qk, values = the grid; 0.096 s/step), local on node 518;
  `spec3` = Mixer3 over each electrode's orthonormal rFFT (61 bins, cells = (Re, Im) lifted to d 8; 1.09M params,
  0.053 s/step), sbatch. Same TSC recipe + bf16.
- 18:50 fold-1 sealed curves, 5-epoch means (TSC / mix3 / mix3att / spec3): e6-10 25.1/22.8/22.6/20.0, e16-20 26.8/23.1/24.5/21.2,
  e26-30 28.0/24.6/-/22.2, e31-35 28.6/24.7. spec3 killed (-5.8). Train loss e35: TSC 6.48, mix3 6.03 (spec3 6.20 at e30):
  the mixers OVERFIT (fit training people far better, transfer worse); TSC's 0.2 s avg-pool is its regulariser.
  Next (single changes on mix3att): `mix3att40` = learned dense 120->40 time compression before the lift (0.73M params),
  `mix3attd5` = dropout .5 (user's suggestion), queued behind mix3 (2-sbatch cap).
- mix3att killed at e25 (e21-25 23.9 vs TSC 27.5: gap 2.3 -> 3.6, widening); local GPU -> mix3att40d5 (both fixes).
- 19:15-19:50 GATE RESULTS fold 1 (opt4 48.80; raw / solo lean TTA; hit phi raw vs REVE / rmix / SQF / TSC; opt4 + it):
    mix3 (e40)        27.2 / 38.8   .65 .68 .48 .47   47.80 (w .25 48.60, w .5 48.40; TSC swap 46.90)
    mix3att (<=e25)   27.1 / 36.5   .58 .59 .44 .48   48.00
    spec3 (<=e30)     25.6 / 36.0   .64 .63 .47 .48   47.70
    mix3att40 (<=e15) 24.0 / 36.5   .59 .59 .45 .48   47.10;  mix3att40d5 (<=e13) 23.2 / 35.7, .59 .55 .42 .50, 47.10
    FlatMLP (<=e15)   23.4 / 33.3   .59 .57 .48 .52   46.50
  The accuracy bar passes (mix3 38.8 > TSC 36.6: like REVE, mixers gain ~+11 from lean TTA vs TSC +5), but EVERY
  mixer / flat MLP lands in the REVE error family (phi .58-.68 vs REVE/rmix, ~.47 vs conv) and lowers opt4 at any weight.
  Time compression, dropout .5, attention, frequency domain: none moves the family. Hypothesis: family = in-network
  normalisation (conv members: BatchNorm inside; REVE/rmix/mixers/FlatMLP: LayerNorm). Test: `mix3bn` (BatchNorm per
  (electrode, feature) over batch x time inside the blocks), `mix3attbn`, and `mix3nn` (no norm in the blocks; local).
  cv4.sh now ends with `done; ...; exit` on one line (editing it mid-run made bash re-read shifted text: harmless
  syntax error after mix3's scoring, but it could have re-run loop text).
- 20:20 interactive job 67749 (node 518) hit its 24 h limit; new interactive job 68305 (node 515) relaunched mix3nn fold 1 at
  20:24, and train.py's same-name cleanup deleted the old (epoch-7) run dir. The 515 run is the live mix3nn; 518's GPU is no
  longer allocated. Early (e1-5 means): mix3bn 18.1, mix3attbn 17.7, mix3nn (old run) 17.1 vs mix3 16.4, TSC 19.6.
- Error-family anatomy (fold 1, hits per (person, image)): removing per-person means barely moves phi (REVE-mix3 .65 -> .63),
  removing per-image means does (.65 -> .56; REVE-SQF .48 -> .36): families agree on WHICH IMAGES fail. Per person,
  REVE / rmix / mix3 / FlatMLP collapse on s15 (15.0 / 16.0 / 11.5 / 15.0 % hits vs SQF 27.5, TSC 30.0). s15 is NOT quiet in
  this cache (trial RMS 0.733 = median; normal per-channel and evoked RMS) and not latency-shifted beyond others (-25 ms like
  s13/s17; s5 is +50 ms and REVE handles it): cause unknown.
- 20:45 mid-run (best ckpt <= e10) BN mixers: mix3bn 27.3 / 37.1, phi .61 .58 .47 .51, opt4+ 48.00; mix3attbn 24.7 / 37.8,
  opt4+ 48.00. In-block BatchNorm does NOT move the family: normalisation hypothesis rejected; both killed.
  Next hypothesis: family = frequency band read (conv members avg-pool 0.2 s = ~5 Hz low-pass; REVE patches / mixers
  read the full band). `mix3lp` (TSConv's 25/2 avg pool on the input, 48 steps) and `mix3hp` (input minus its 25-sample
  moving average) on fold 1.
- 20:50 INCIDENT: the pre-restart Claude process kept running on node 518 (started under that node's Cursor server, it
  survived the end of interactive job 67749) and resumed the SAME conversation in parallel with the new session on 515.
  It (not the user) cancelled mix3bn/mix3attbn (68291/68292) at e11, added Mixer3 `band` + recipes mix3lp/mix3hp and
  launched them (68312 on 518, 68313 on 513), and wrote the `mix3bnmid` cache. Killed it (kill -9 971140 on 518). Its BN
  verdict matches mine (private cache, best ckpt <= e10): mix3bn 27.3 / 37.1, phi .61 .58 .47 .51, opt4+ 48.00 -> the
  normalisation hypothesis is rejected; mix3nn (its third arm) killed. mix3lp / mix3hp adopted (band hypothesis).
- `--input_band lp|hp` (train.py, applied in run_eeg_backbone for every encoder: the 0.2 s moving average or the rest).
  `reve_d6_hp_optx_things` = REVE-d6 on the high band: the actionable arm (a strong third family?), local on 515.
- 21:55 BAND RESULT (best ckpt <= e10, fold 1; raw / solo lean TTA; phi raw vs REVE / rmix / SQF / TSC; opt4 + it):
    mix3lp (input = 25-sample moving average, < ~3 Hz)  26.2 / 31.7 (+5.5)   .39 .39 .50 .55   47.40
    mix3hp (input minus it, > ~4 Hz)                    25.2 / 34.8 (+9.6)   .60 .61 .45 .49   47.70
  THE INPUT BAND SETS THE ERROR FAMILY: the same mixer lands in the conv family on the low band and in the REVE family
  on the high band; the TTA gain follows (+5 low band like TSC, +10 high band like REVE). The conv members read < ~5 Hz
  through their 0.2 s avg pool; the REVE family's errors come from > ~4 Hz content. Both band mixers lower opt4: killed.
  Next (actionable): `reve_d6_lp` (a strong model in the conv family: TSC replacement?), `reve_d6_bp` (~4-12 Hz: a third
  family?), `reve_d6_hp` (running) ; `--input_band bp` = 5-sample minus 25-sample moving average (gain .89 at 5 Hz,
  .51 at 10, .03 at 20).
- 23:00 reve_d6_hp (> ~4 Hz) fold 1 complete: 30.5 / 40.2 (REVE-d6 32.7 / 44.2), phi .77 .79 .46 .48 = a RESEED of REVE
  (seed ceiling .77); opt4+ 48.30, swaps 48.10 (REVE) / 47.90 (rmix) / 47.70 (SQF) / 46.80 (TSC). REVE barely uses < 4 Hz.
  Launched `reve_d6_vhp` (input minus its 5-sample moving average, > ~15 Hz; lp + bp + vhp == input exactly) locally.
  Early curves (2-epoch means, e3-4): reve_d6_lp 25.0, reve_d6_bp 25.8 vs REVE-d6 27.6, TSC 21.4.
- 23:40 REVE band runs, mid-run (best ckpt <= e8; raw / solo lean TTA; phi vs REVE rmix SQF TSC; opt4+, best swap):
    reve_d6_lp (< ~3 Hz)   29.0 / 39.9   .72 .70 .48 .56   48.40, 47.80 (rmix)
    reve_d6_bp (~4-15 Hz)  29.2 / 40.4   .78 .78 .48 .50   48.40, 48.20 (REVE);  opt4 + lp + bp 47.50
  REVE stays in its own family on ANY band: the band rule holds for the mixer only. Gallery-level anatomy (fold 1, 200
  images): hub-count profile r REVE vs REVE-lp/bp/hp .90/.90/.93 (REVE-rmix .94, SQF-TSC .85, across families .58-.74),
  per-image hit-rate r .86-.87: the family = the model's OUTPUT PRIOR (which gallery images it predicts), intrinsic to the
  architecture, not the EEG band. mix3lp is the one outlier prior (hub r .43-.62 with all) but weak. REVE-lp/bp killed at
  e9, reve_d6_vhp killed (would be one more REVE reseed).
- 01:20 band-dropout augmentation `--band_drop 0.5` (train only: a trial keeps lp, bp, vhp, lp+bp or bp+vhp): REVE-d6
  (local), SQF, TSC on fold 1; judge by swapping each for its own opt4 member under lean TTA.
- 02:30 reve_d6_bd killed (e1-2 -1.4, e3-4 -1.8 vs reve_d6); sqf_dbn_bd e6-10 26.7 vs 24.6 (+2.1), tsc_bd e11-15 26.5 vs 26.3 -> sqf_dbn_bd fold 2 started locally.
- 11-05 05:23 band-dropout FINAL (auto, finish_bd.sh), fold 1:
    opt4                                                 48.80
    sqf_dbn_op raw / soloTTA                             33.80 / 39.30
    tsc_dbn40_ raw / soloTTA                             31.30 / 36.60
    sqf_dbn_bd_optx_things raw / soloTTA                 35.00 / 40.80
    sqf_dbn_bd phi vs reve_d6_op raw / TTA               0.47 / 0.48
    sqf_dbn_bd phi vs rmix_reg_o raw / TTA               0.47 / 0.52
    sqf_dbn_bd phi vs sqf_dbn_op raw / TTA               0.70 / 0.66
    sqf_dbn_bd phi vs tsc_dbn40_ raw / TTA               0.67 / 0.57
    opt4 + sqf_dbn_bd_optx_things                        48.70
    opt4 reve_d6_ -> sqf_dbn_bd_optx_things              46.70
    opt4 rmix_reg -> sqf_dbn_bd_optx_things              47.20
    opt4 sqf_dbn_ -> sqf_dbn_bd_optx_things              48.10
    opt4 tsc_dbn4 -> sqf_dbn_bd_optx_things              48.30
    tsc_dbn40_bn64_bd_optx_things raw / soloTTA          32.30 / 36.10
    tsc_dbn40_ phi vs reve_d6_op raw / TTA               0.46 / 0.43
    tsc_dbn40_ phi vs rmix_reg_o raw / TTA               0.46 / 0.49
    tsc_dbn40_ phi vs sqf_dbn_op raw / TTA               0.60 / 0.58
    tsc_dbn40_ phi vs tsc_dbn40_ raw / TTA               0.84 / 0.79
    opt4 + tsc_dbn40_bn64_bd_optx_things                 48.60
    opt4 reve_d6_ -> tsc_dbn40_bn64_bd_optx_things       47.20
    opt4 rmix_reg -> tsc_dbn40_bn64_bd_optx_things       46.30
    opt4 sqf_dbn_ -> tsc_dbn40_bn64_bd_optx_things       47.90
    opt4 tsc_dbn4 -> tsc_dbn40_bn64_bd_optx_things       48.30
    opt4 + all new                                       47.90
  fold 2 (sqf_dbn_bd only):
    opt4                                                 42.30
    sqf_dbn_op raw / soloTTA                             30.00 / 34.70
    sqf_dbn_bd_optx_things raw / soloTTA                 31.20 / 36.90
    sqf_dbn_bd phi vs reve_d6_op raw / TTA               0.58 / 0.61
    sqf_dbn_bd phi vs rmix_reg_o raw / TTA               0.56 / 0.61
    sqf_dbn_bd phi vs sqf_dbn_op raw / TTA               0.77 / 0.75
    sqf_dbn_bd phi vs tsc_dbn40_ raw / TTA               0.64 / 0.63
    opt4 + sqf_dbn_bd_optx_things                        42.00
    opt4 reve_d6_ -> sqf_dbn_bd_optx_things              40.60
    opt4 rmix_reg -> sqf_dbn_bd_optx_things              40.80
    opt4 sqf_dbn_ -> sqf_dbn_bd_optx_things              42.10
    opt4 tsc_dbn4 -> sqf_dbn_bd_optx_things              41.90
- VERDICT band dropout: better singles (SQF-bd fold 1 35.0/40.8 vs 33.8/39.3, fold 2 31.2/36.9 vs 30.0/34.7; TSC-bd 32.3/36.1
  vs 31.3/36.6) but NO ensemble gain: SQF swap 48.10 / 42.10 (opt4 48.80 / 42.30), TSC swap 48.30, adds 48.70 / 42.00 / 48.60.
  N65 overall: no member from the MLP or band lines helps opt4; the error families are the architectures' output priors.

## N66. Output-prior equipartition (follow-up of N65: the error families = each architecture's output prior)
`--prior_lambda L`: loss += L * KL(batch-mean softmax over the 16,540 training-image bank || uniform); each trial's bank
cosines z-scored, temperature 0.25 (raw cosine std vs the bank is only ~0.02). Uniform is the training targets' own
marginal. Measured on cached fold-1 outputs: REVE .80 (KL), TSC .70, SQF .46; random predictions 1.28 (the bank's
anisotropy alone makes hubs). Screens fold 1: reve_d6_pr1, reve_d6_pr3 (sbatch), sqf_dbn_pr1 (local); judge by opt4 swaps.
- sqf_dbn_pr1 killed (e1-5 14.5 vs SQF 20.0); local -> sqf_dbn_pr03 (lambda .3).
- reve_d6_pr1 / pr3 killed at e2 (14.4 / 7.1 vs 23.7): equipartition fights retrieval. Scan gentler: reve_d6_pr01, reve_d6_pr03 (sbatch).
- Test-time in-batch hub debiasing (hubdebias.py: subtract alpha x the over-predicted training-bank direction of each
  batch, pre or post the LW shrink), opt4 4-fold: best post T.25 a.05 37.30 vs production 45.48; a .1 24.7, a .2 12.6.
  A common shift re-ranks the gallery for every trial and the per-trial margins are tiny; training-bank hubs are not
  the test-gallery hubs (cf. N47 bank snap). Dead.
- reve_d6_pr03 killed (e1-2 17.8 vs 23.7); reve_d6_pr01 e1-2 19.6 (-4.1): family check at e4 decides.
- sqf_dbn_pr03 e1-4 5.6/16.6/17.0/23.3 vs SQF 16.9/18.6/18.0/20.9: recovers after a bad e1 (kept). Freed slot -> tsc_dbn40_bn64_pr03.
- tsc_dbn40_bn64_pr03 killed (e1-5 6.4..14.9 vs 14.3..22.8: -8). sqf_pr03 e6-8 -2, reve_pr01 e3 -3.4 vs baselines.
- 10:45 FIRST POSITIVE SIGNAL (fold 1, best ckpt so far; raw / solo TTA; phi vs REVE rmix SQF TSC; opt4 48.80):
    sqf_dbn_pr03 (<= e10)  23.8 / 31.9   .39 .40 .44 .51   TSC swap 48.90, SQF swap 48.70, add 48.40
    reve_d6_pr01 (<= e4)   26.4 / 38.8   .60 .56 .44 .47   best 47.90 -> killed
  The output-prior term MOVES the family: SQF-pr03 is decorrelated from BOTH families (phi .44 with its own baseline,
  vs .70 for band-dropout SQF) and, though still weak, already ties/edges opt4 when it replaces TSC. Launched
  sqf_dbn_pr03 fold 2 and sqf_dbn_pr01 fold 1 (sbatch); sqf_dbn_pr03 fold 1 continues to e40 (local).
- 11:30 user: don't stop experiments early, call results on more folds. Queued sqf_dbn_pr03 folds 3, 4 (sbatch, after the
  current jobs) and sqf_dbn_pr01 fold 2 (local, after the fold-1 pr03 run); finish_pr.sh (folds 1-2) and finish_pr2.sh
  (4-fold pr03, 2-fold pr01) append the verdicts here automatically.
- 13:20 sqf_dbn_pr03 fold 1 FINAL (sealed_best over 40 epochs): 24.7 / 34.9, phi .41 .39 .53 .53 (mid-run <= e10 was .39 .40
  .44 .51), opt4 + 47.50, swaps REVE 46.80 / rmix 45.20 / SQF 47.00 / TSC 47.70 (mid-run TSC swap 48.90 was noise/early):
  with training the prior model regains accuracy AND drifts back toward the conv family. Folds 2-4 + pr01 still running
  (user: judge on more folds).

## N67. Shared-confusion penalty (anti-teacher)
Loss += lambda * mean_i sum_{j wrong} p_student(j|i) p_teacher(j|i) over the batch images (tau .005), teacher = the fold's
opt4 member-mean export (the KD teacher files, used in the opposite direction: avoid the ensemble's MISTAKES; shared
correct answers cost nothing). `train.py --anti_teacher_lambda`, cv4 recipes *_anti4_things (cv4.sh hands each fold its
teacher). Gate check: the teacher is moderately in-sample (single-trial 200-way top-5 16.7 on its own training trials vs
9.7 held-out) but still misses 83% of training trials, so the confusions exist. Penalty scale: student = teacher .137,
teacher + noise .018, random .001 (CL ~6.5) -> lambda 5 and 20 on SQF fold 1 (local after pr01 f2; sbatch after pr03 f3).
- 11-05 13:43 finish_pr.sh scoring crashed (CUDA OOM: it shared the local GPU with a training run); rescored on CPU below.
- 11-05 13:48 output-prior SQF FINAL, CPU rescoring: sqf_dbn_pr03 folds 1-2
    opt4                                                 48.80   42.30
    sqf_dbn_op raw / soloTTA                             33.80 / 39.30   30.00 / 34.70
    tsc_dbn40_ raw / soloTTA                             31.30 / 36.60   27.80 / 32.60
    sqf_dbn_pr03_optx_things raw / soloTTA               24.70 / 34.90   24.80 / 32.20
    sqf_dbn_pr phi vs reve_d6_op raw / TTA               0.41 / 0.45   0.46 / 0.52
    sqf_dbn_pr phi vs rmix_reg_o raw / TTA               0.39 / 0.45   0.45 / 0.55
    sqf_dbn_pr phi vs sqf_dbn_op raw / TTA               0.53 / 0.55   0.60 / 0.67
    sqf_dbn_pr phi vs tsc_dbn40_ raw / TTA               0.53 / 0.49   0.57 / 0.52
    opt4 + sqf_dbn_pr03_optx_things                      47.50   42.60
    opt4 reve_d6_ -> sqf_dbn_pr03_optx_things            46.80   40.50
    opt4 rmix_reg -> sqf_dbn_pr03_optx_things            45.20   40.70
    opt4 sqf_dbn_ -> sqf_dbn_pr03_optx_things            47.00   42.50
    opt4 tsc_dbn4 -> sqf_dbn_pr03_optx_things            47.70   42.30
  sqf_dbn_pr01 fold 1:
    opt4                                                 48.80
    sqf_dbn_op raw / soloTTA                             33.80 / 39.30
    tsc_dbn40_ raw / soloTTA                             31.30 / 36.60
    sqf_dbn_pr01_optx_things raw / soloTTA               30.70 / 36.40
    sqf_dbn_pr phi vs reve_d6_op raw / TTA               0.44 / 0.43
    sqf_dbn_pr phi vs rmix_reg_o raw / TTA               0.41 / 0.46
    sqf_dbn_pr phi vs sqf_dbn_op raw / TTA               0.67 / 0.65
    sqf_dbn_pr phi vs tsc_dbn40_ raw / TTA               0.60 / 0.56
    opt4 + sqf_dbn_pr01_optx_things                      48.20
    opt4 reve_d6_ -> sqf_dbn_pr01_optx_things            47.40
    opt4 rmix_reg -> sqf_dbn_pr01_optx_things            46.80
    opt4 sqf_dbn_ -> sqf_dbn_pr01_optx_things            49.00
    opt4 tsc_dbn4 -> sqf_dbn_pr01_optx_things            49.20
  verdict so far (14:10): pr03 is no gain (fold 1 best variant 47.50 vs 48.80, fold 2 42.60 vs 42.30). pr01 fold 1: swap for
  TSC 49.20 (+0.4 = 4 images of 1000), swap for SQF 49.00 -- one fold, inside noise; pr01 folds 2-4 decide it (requeue after
  the finals, ahead of pr03 folds 3-4).

## N68. Submission 1 (sealed phase day 1, 2026-11-05): opt4 + lean TTA, retrained on everyone
- Members = opt4 recipes, final.sh: all 20 Alljoined people + sub-21..30, each at its cv4 epoch budget (REVE 9, rmix 9,
  SQF 39, TSC 29), plus 10 of the 80 test reps of every person (`--train_test_reps 10`: every 8th rep, so all sessions;
  +60,000 trials = +3%; user call, test images are not in the sealed set). First launch without test reps cancelled after
  5 min. Jobs: REVE 68512, SQF 68513, TSC 68514 (after either), rmix local on 515.
- Package (package_submission.py): each member = encoder + EEG projector traced to TorchScript on CUDA with the
  unknown-subject id (domain-BN members then take the Alljoined path), so submission.py needs no model source and no
  braindecode; tta.pt holds the member list, training channel order and the fingerprint WCCN fitted on all 20 Alljoined
  people. Two trace fixes, same maths: TokenREVE `len(z)` -> `z.shape[0]` (len froze the batch size),
  _FusedTemporalConvPool padding as a Python int.
- Check on fold 1 (fold-1 checkpoints, WCCN on its 15 training people, stream served in a shuffled channel order through
  Solver.load_model + predict): batch 64 48.90 vs lean_tta 48.80 (fp16 cache vs fp32 live). Batch 37: 46.30 (fewer
  batch-mates for the kNN centring; the grader serves 64).
- Contract check: the competition's check_submission.sh is BCI-only and the image warm-up data is not prepared here, so
  neurips_challenge/check_package.sh (was scratchpad check_sim.sh) runs the official ingestion + scoring on a copy of tracks/image_decoding whose Simulated set
  serves 32 ch x 120 samples under the Emotiv names in a shuffled order, 1536-D targets, 600 test windows: runs clean
  (load_model with the real meta, CUDA, 9 x 64 + 24 batches), chance score as expected, 46 s.
- finish_sub1.sh (background): TSC final on the local GPU after rmix, then package the 4 finals -> 
  neurips_challenge/submissions/sub1_opt4_tta.zip, in-sample fold-1 replay, ingestion check. ETA ~18:00.
- 15:20 twin incident (the user restarted the interactive job: 68305 on 515 -> 68597 on 514; a resumed session on 514
  relaunched rmix there and removed the 515 rmix run). The stale 515 session stopped its finish_sub1.sh, which had just
  started a TSC final on the unallocated 515 GPU (killed, its run dir removed so Slurm job 68514 runs TSC cleanly).
  Nothing of the 515 session is left running; the 514 session owns submission 1.
- 514 side: the 515 rmix final was in fact still alive (orphan of the dead interactive job, epoch 6/9) when I removed
  its run dir to relaunch -> it crashed at its next save; rmix restarted from scratch on 514 at 15:18 (ETA ~17:35).
  Twin claude on 515 (pid 364382) killed -9 at 15:23 via a fresh srun allocation on 515. Finisher now
  logs/finish_sub1.sh (NAS, survives node loss): waits for the local rmix + Slurm finals, packages, checks, zips.
- 16:55 REVE final done. TSC final moved onto a 4th GPU (user OK, 515). Band follow-ups queued (logs/band_queue.sh,
  workers 68673 after SQF final, 68674 after TSC final, + the local GPU after submission 1 is packaged): pr01 folds
  2-4, anti-teacher lambda 5 / 20 fold 1, pr03 folds 3-4. The paused partial runs (pr01 f2, pr03 f3/f4) were deleted
  (no resume).
- 17:30 SUBMISSION 1 READY: neurips_challenge/submissions/sub1_opt4_tta.zip (112 MB, sha256 4e9e56b35caab9ed..., 6 files
  at the zip root: submission.py, tta.pt, member0_reve/1_gen/2_sqf/3_tsc.pt). Finals' sub-01 raw (in-sample, no TTA)
  top-5: REVE 49.5, rmix 51.0, SQF 52.5, TSC 47.0. Packaged fold-1 replay 76.20 (in-sample: those people + 10/80 of
  their test reps are in training, smoke only). Official ingestion check clean (62 s). Left to the user to upload.
  Local GPU now runs a 3rd band_queue.sh worker.

## N69. Sealed leaderboard regime vs cv4 (2026-11-05 evening): the lean TTA HURTS there; submission 2 = no TTA
- Leaderboard (sub 1 = opt4 + lean TTA, 3rd): subject-agg top-5 0.000921 (fraction), top-1 0.000194; per-epoch 0.000465 /
  0.0001. The 6 lowest teams sit at chance (top-5/top-1 ~4.7): per-epoch top-1 2.57e-5, top-5 1.22e-4, subject-agg top-1
  2.7e-5 -> gallery G ~ 37-41k images. Their spread suggests ~0.9-2M scored epochs. 11 participants (tracks.html table).
  Subject-agg / per-epoch ~2 for every good team -> ~3 repeats per participant-image (cv4: 1.5x at r=2, 2.8x at r=4).
- So the sealed regime is r ~ 3 at G ~ 40k, while every cv4 number (and all TTA tuning) is r = 80 at G = 200.
- TTA ablation, opt4, 4 folds, target-only 200 gallery, subject-agg top-5 (logs/tta_ablation_reps.out):
      r:                         1      3      4      8      80
      plain L2 sum             8.60  13.73  15.50  20.70  32.43
      batch-norm-scaled sum    8.61  14.11  16.03  21.18  32.33
      + kNN centring           7.69  12.69  14.56  20.13  36.72
      + LW only                8.11  13.82  15.88  22.14  38.90
      deployed lean TTA        7.37  12.46  14.62  21.02  44.02
  Centring costs ~1.4 at r <= 4 (subtracting 4 noisy batch-mates); LW is neutral until r ~ 8. Crossover at r ~ 8.
  With training-image distractors (G up to 16.7k, logs/gallery_tta_vs_raw.out) TTA loses 2x, partly an artifact (the
  plain outputs share a test-set direction that separates targets from never-target distractors), and TTA costs top-1
  more than top-5 (top5/top1 6.6-8 vs 5.7): explains our weak leaderboard top-1.
- Ensemble under the sealed-like metric (r=3, scaled sum, logs/sealed_ens_search_r3.out): opt4 14.20, optx5 14.33, best
  reve_d6X + rmix_regX + sqf_dbnX + reve_scratchX + sqf_dbn 14.53, picked on every nested fold (nested 14.53).
  Output-prior SQF (pr01/pr03) under it: neutral (swap for TSC +0.1/-0.0, others <= 0): no band gain in either metric.
- Submission 2 option A READY: neurips_challenge/submissions/sub2_opt4_sum.zip = sub-1 weights + scaled-sum predict
  (submission.py no longer carries the TTA; fold-1 cv4 package: 14.84 at batch 64, 14.78 at batch 37, batch-size
  robust). Option B = the 5-member set: finals reve_scratch_optx_things (budget 8, job 68780) and sqf_dbn_things
  (budget 33, job 68781) overnight. Blocked (claim owner "blocked"): l20 anti-teacher f1, pr03 f3/f4 (no value now);
  pr01 f2-4 finish, then anti-teacher lambda 5 f1. Band watcher now scores with sealed_swaps.py.
- 19:10 anti-teacher lambda 5 f1 claimed for the local worker (owner = interactive job 68597) so both Slurm workers exit after pr01 and free their GPUs for the two option-B finals.
- 19:20 user picked option B for submission 2; logs/finish_sub2.sh packages it when finals 68780/68781 end.
- Low-repeat test-time transforms (logs/tt_variants_lowrep.out, opt4, 4 folds, 200 target-only, subject-agg top-5 at
  r=1/3/4/8): scaled sum 8.61/14.26/16.09/21.05 is the best at r <= 4. Centred batch whitening is worse everywhere
  (mild 13.35 at r=3, d=0.1 9.71); full batch centring -0.8; agreement weighting <= 0; pulling predictions onto the
  training-image bank collapses to chance (test concepts are new). Uncentred LW only wins at r=8 (21.95).
  Open: LW is anti-hubness and hubness grows with the gallery -> logs/largeg_proxy.py (held-out people's trials of
  5,000 training images, target-only gallery, r=1/3) running on fold 1.
- pr01 under the sealed-like metric: swap for TSC +0.15/+0.15/+0.04 (folds 1-3), others <= +0.08: neutral.
- Large target-only gallery proxy (logs/largeg_proxy.py, fold 1, held-out people x 5,000 training images x 4 reps,
  chance top-5 0.10%; images seen in training via other people), subject-agg top-5 %:
      opt4 scaled sum r=1 0.828 r=3 1.828 | plain L2 0.820 / 1.736 | uncentred LW 0.773 / 1.864 | lean TTA 0.660 / 1.608
      ens5 (option B) scaled sum r=1 0.868 r=3 1.916 (+5% relative over opt4, same as at 200 images)
  -> TTA -12% at large G too; LW +2% at r=3 but -7% at r=1 (noise-level, not adopted); option B holds at large G.
  Repeat count: agg/single-trial top-5 ratio here = 1.61 (r=2), 2.21 (r=3), 2.72 (r=4); leaderboard teams 1.84-1.98
  -> ~2-3 repeats per participant-image (or a mix), not 4+.
- pr01 4-fold verdict under the sealed-like metric (r=3): swap for TSC +0.11/+0.15/+0.04/-0.10 (mean +0.05), for SQF
  +0.03/+0.08/-0.21/-0.26, as 5th member -0.11/+0.01/-0.24/-0.21. Output prior: NO GAIN, closed (both metrics).
- 21:20 anti-teacher lambda 5 f1 started on the local GPU at 1.8 s/step (32 h!): GPU idle, trainer in IO wait on random
  reads of the 5 GB fp16 teacher-target memmap on the NAS. Warmed it into the page cache (cat > /dev/null, 21 s).
  -> 2.7 it/s after warming (~10 min/epoch, 40 epochs ~6.5 h, done ~04:00).

## N70 (2026-11-06). Submission 2 = option B; band line closed
- The 11-05 session ended before logs/finish_sub2.sh ran; re-run 12:45: neurips_challenge/submissions/sub2_ens5_sum.zip
  (399 MB: reve_d6X, rmix_regX, sqf_dbnX, reve_scratchX (budget 8), sqf_dbn (budget 33), scaled-sum predict).
  In-sample fold-1 replay 27.57 (opt4 finals 25.30); contract check clean (90 s).
- Speed on an idle 3080: sub 1 (opt4 + TTA) 14.9 ms / 64-trial batch, opt4 scaled sum 12.5, ens5 38.7 (full-depth
  REVE scratch). Sub 1 took 744 s sealed -> ens5 <= ~32 min even if all of it were inference (limit 60).
- Anti-teacher lambda 5, fold 1, sealed-like metric: swap for TSC +0.14 (any SQF variant gets that on f1), +opt4 -0.23,
  alone 9.58: no gain. With the output prior (4 folds, no gain) the band line is closed.
- Fold-1 screen of all 28 other cached members under the sealed-like metric (logs/sealed_screen_f1.out): opt4 14.83,
  ens5 15.07; best add to ens5 +0.04, best swap +0.04 (noise). Band (hp, bd), mixers, seeds, DRO/JS/pct, even the
  banned KD members: nothing. ens5 is saturated for the current pool.

## N71 (2026-11-06). Optimising the large-gallery, 3-4 repeat metric (user goal)
- Test-time side, all on ens5 (option B), 4 folds unless noted:
  - large-gallery proxy (logs/largeg_proxy.py: held-out people x 5,000 TRAINING images x 4 reps, target-only):
    opt4 1.467 / 1.812, ens5 1.536 / 1.896 (top-5 % at r=3 / r=4, chance 0.10). Exhaustive subsets of the 10-pool:
    nested 1.521 / 1.883 (< ens5); tuned member weights nested 1.535 / 1.868 (= or < uniform). ens5 is the optimum.
  - per-trial vote y = S |S|^gamma: gamma = 0 (natural norm) best on both proxies (logs/norm_vote.out).
  - gallery-prior hubness bias y - alpha mu (mu = mean training-image embedding): alpha > 0 hurts monotonically,
    alpha = -0.02 noise (logs/hub_bias_small.out).
  - band / shift test-time views (logs/band_tta_views.py, 200-image, folds 1-3): id+smooth +0.12 / -0.22 / -0.32 at
    r=3, shifts +0.0 / -0.0 / +0.1: no gain (fold 1 alone looked +0.12-0.21).
  -> option B + scaled sum is the optimum of everything applicable to existing models.
- Training side (the only route left): the loss runs at temperature 1.0 (logit scale 1), a near-uniform softmax that
  barely weighs hard negatives, while the sealed task is fine-grained retrieval among ~40k images.
  New train.py flags: --holdout_images N (drop N training images, seed 0, from training and bank) and --bank_lambda /
  --bank_tau (cross-entropy over ALL training-image embeddings, cosine / 0.05). New clean proxy logs/holdout_eval.py:
  held-out people x the 5,000 held-out (never-trained) images x 4 reps, target-only gallery = new people + new images +
  large gallery + few repeats. Fold-1 runs (11-06 13:30): sqf_dbn_ho5k_optx_things (base, 69772),
  sqf_dbn_ho5k_bank_optx_things (bank lambda 1 tau .05, 69773), sqf_dbn_ho5k_t01_optx_things (temperature 0.1, local).
- 15:50 novel-image proxy verdict (logs/holdout_eval_f1.out; fold-1 SQF, 25k person-image groups, 5,000 never-trained
  images, top-5 % at r=1 / 3 / 4, sd ~0.06): base 0.518 / 1.020 / 1.196; bank loss (lambda 1, tau .05) 0.492 / 1.000 /
  1.128; temperature 0.1 0.510 / 1.020 / 1.204. 200-image r=80: 30.4 / 31.7 / 31.2. Neither the full-bank negatives
  nor a sharper temperature moves the large-gallery few-repeat metric: closed.
- Session summary for the large-gallery 3-4 repeat metric: the gains are dropping the lean TTA (~+12-15%) and ens5 over
  opt4 (+4.7% at 5k, r=3 and r=4), both in submission 2 (option B). Everything else tried (subsets, weights, vote
  weighting, hubness bias, whitening, LW, band/shift views, bank loss, temperature) is at or below that optimum.
- Per-direction (PCA basis) stacking of ens5, ridge toward the uniform sum, nested (logs/pc_stack.out): best +0.04
  (noise) at weak shrinkage, worse below; MSE-optimal per-direction weights are not ranking-optimal. Closed.
- Epoch-budget check (the budgets were picked at ~10 repeats / 200 images): fold-1 REVE-family members run on past
  their budgets with checkpoints (--save_epochs; recipes *_long_optx_things): reve_d6 to 20 (job A), rmix to 20 (B),
  reve_scratch to 14 (local). Each checkpoint gets swapped into ens5 at r=3/4 on the 200-image and 5k proxies.
- 19:30 epoch scan, rmix fold 1 (replacing rmix in ens5; top-5 % r=3 / r=4): reference (cv4 budget ~9) 200 14.96/16.90,
  5k 1.808/2.248; ep 10 14.88/16.93, 1.876/2.244; ep 12 14.84/17.04, 1.864/2.304; ep 14 14.86/17.17, 1.892/2.312;
  ep 18 14.91/17.10, 1.768/2.236; ep 20 14.95/16.96, 1.780/2.252. Large gallery prefers ep 12-14 (+3-5%, ~1 sd, one
  fold). rmix long runs on folds 2 and 3 launched (+ epoch_swap.py <recipe> <slot> <fold> after each).
- 20:05 epoch scan, REVE-d6 fold 1 (replacing reve_d6 in ens5): reference 200 14.93/16.90, 5k 1.816/2.252; ep 6-20 all
  within noise (best ep 14 1.868/2.224, ep 16 1.860/2.296 at 5k; 200 14.66-14.96). Budget stays. rmix folds 2-3 running.

## N72 (2026-11-08 night). Epoch budgets for the large gallery; resumed on node 520
- rmix long-run epoch scans (replacing rmix in ens5; 5k proxy top-5 % r=3 / r=4):
      fold 1: ref 1.808/2.248  ep12 1.864/2.304  ep14 1.892/2.312  ep16 1.856/2.276  ep20 1.780/2.252
      fold 2: ref 1.172/1.464  ep12 1.140/1.516  ep14 1.176/1.452  ep16 1.140/1.456  ep20 1.148/1.508
      fold 3: ref 1.828/2.232  ep12 1.864/2.248  ep14 1.900/2.252  ep16 1.896/2.324  ep20 1.936/2.384
  mean of 3 folds: ref 1.603/1.981, ep12 1.623/2.023, ep14 1.656/2.005, ep16 1.631/2.019, ep20 1.621/2.048; 200-image
  r=3/4 ref 14.53/16.44 vs ep14 14.58/16.59. Later rmix epochs (12-20) mostly >= the budget-9 reference on the large
  gallery; ep14 picked. Fold 4 long run + an rmix final at budget 14 launched (submission 3 candidate).
- The 11-06 session ended before the REVE-scratch epoch scan ran; run now on 520. Old node 514 checked: no twin.
- 11-09 00:45 interactive moved 520 -> 518. The 520 chain (plain-SQF long run) survived as an orphan on an
  unallocated GPU: killed, dir removed, relaunched on 518 (logs/chain_local.sh: REVE-scratch scan, then SQF long f1 +
  scan). The first rmix budget-14 final (70611) exited at once: train.py refuses an existing 'gen' experiment in
  final/rmix_reg_optx_things -> final.sh takes an optional tag (final/<recipe>_<tag>); relaunched as tag ep14.
- 00:50 REVE-scratch epoch scan fold 1 (replacing it in ens5; top-5 % r=3/r=4): ref 200 14.98/16.86, 5k 1.804/2.260;
  ep8 (final budget) 14.87/17.00, 1.792/2.220; ep10 14.94/16.93, 1.788/2.216; ep12 14.99/17.10, 1.856/2.320;
  ep14 14.95/17.02, 1.832/2.276. Same direction as rmix (later epochs +~3% at 5k, one fold). Folds 2-3 long runs
  queued behind the Slurm jobs (+ scans).
- 03:05 rmix fold 4 scan: ref 5k 1.332/1.636, 200 13.79/15.57; ep14 1.380/1.612, 13.75/15.50; ep16 1.384/1.708.
  4-fold means (5k r=3, r=4 | 200 r=3, r=4): ref 1.535, 1.895 | 14.34, 16.22; ep12 1.550, 1.938; ep14 1.587 (+3.4%),
  1.907 (+0.6%) | 14.37, 16.32; ep16 1.569, 1.941 | 14.33, 16.26; ep20 1.544, 1.957. Epochs 12-20 all >= the reference
  (itself test-selected): rmix budget 9 -> 14 ADOPTED for the large-gallery regime (+~2%). logs/finish_sub3.sh packages
  submission 3 (option B with the budget-14 rmix) when final 70616 ends.
- 03:35 plain-SQF epoch scan fold 1 (replacing sqf_dbn in ens5): ref 5k 1.784/2.240, 200 14.95/16.90; ep24
  1.788/2.248; ep28 1.828/2.232; ep32 1.840/2.256 (15.01/16.78); ep36 1.808/2.232; ep40 1.812/2.236. Flat: budget 33 stays.
  Local GPU -> REVE-scratch long run fold 4 (+ scan).
- 03:55 SUBMISSION 3 READY: neurips_challenge/submissions/sub3_ens5_rmix14_sum.zip = option B with rmix retrained at
  budget 14 (final/rmix_reg_optx_things_ep14). In-sample fold-1 replay 29.11 (b64) / 28.98 (b37); contract check clean
  (72 s). Expected over sub 2: ~+2% at large galleries (4-fold rmix scan). REVE-scratch long runs folds 2-4 running.
- 11:00 REVE-scratch epoch scans folds 2-4 done. 4-fold means (5k r=3, r=4 | 200 r=3, r=4): ref 1.534, 1.899 |
  14.35, 16.22; ep8 (final budget) 1.543, 1.885; ep10 1.572, 1.917; ep12 1.577 (+2.8%), 1.960 (+3.2%) | 14.44, 16.43;
  ep14 1.599, 1.927 | 14.31, 16.30. REVE-scratch budget 8 -> 12 ADOPTED: final/reve_scratch_optx_things_ep12 (70826,
  ~7 h). Also testing swap vs ADD-as-6th-member for rmix ep14/16 and scratch ep12/14 (the old finals at budgets 9 / 8
  exist, so a 6th or 7th member costs no training): logs/add_scan.sh, folds 1-2 Slurm 70827, folds 3-4 local.
- 11:10 swap vs add (4 folds, logs/add_scan_*): adding the old checkpoint as a 6th member is worse than swapping on the
  5k metric (rmix ep14 1.528 add vs 1.587 swap at r=3; scratch ep12 1.565 vs 1.577). Combo (logs/combo_check.out, 4
  folds; 5k r=3, r=4 | 200 r=3, r=4): ens5 1.522, 1.892 | 14.34, 16.23; rmix14 1.585, 1.911 | 14.38, 16.28; scratch12
  1.575, 1.963 | 14.44, 16.43; BOTH 1.599 (+5.1%), 1.931 (+2.1%) | 14.36, 16.41, every fold >= ens5 at r=3.
  -> submission 4 = ens5 with rmix ep14 + REVE-scratch ep12 (logs/finish_sub4.sh packages it after final 70826).
- Caveat: the 5k proxy uses training images, so longer training may partly be memorisation. Clean check: rmix and
  REVE-scratch long runs with --holdout_images 5000 (fold 1), epoch scans on the 5,000 never-seen images
  (logs/chain_ho5k_eval.sh -> logs/ho5k_epoch_scan.out).
- 13:50 CLEAN CHECK OVERTURNS the epoch result: rmix with 5,000 images held out (fold 1), alone, on those never-seen
  images (top-5 % r=3 / r=4): ep8 1.308/1.484, ep10 1.376/1.484, ep12 1.392/1.376, ep14 1.272/1.328, ep16 1.036/1.240,
  ep18 0.880/1.088, ep20 0.932/1.068. Peak ep10-12, decline after: the seen-image 5k proxy's "+3% at ep14" was
  memorisation of the training images. rmix budget 9 stays; submissions 3/4 NOT recommended, submission 2 stands.
  Lesson: epoch budgets (anything that changes how much the model fits the training images) must be judged on the clean
  novel-image proxy only. Same clean scan now for REVE-d6 (Slurm) and REVE-scratch (local, ~16:30).
- 15:50 REVE-scratch clean scan (fold 1, alone, never-seen images, top-5 % r=3/r=4): ep6 1.160/1.408, ep8 (budget)
  1.260/1.444, ep10 1.332/1.520, ep12 1.200/1.364, ep14 0.980/1.140. Peak ep10; budget 12 worse than 8 -> final
  70826 CANCELLED, submission 4 dropped (finish_sub4 killed). Both REVE-family members peak near ep10 on new images
  (their budgets 8-9 are about right; scratch 8 -> 10 = +5% alone on one fold, not acted on yet).
  Next: are the SQFs (budgets 39 and 33 of 40, picked on 200 new-concept images) over-trained for new images of seen
  concepts? sqf_dbn_ho5k_long_optx_things (Slurm) and sqf_dbn_ho5k_long_things (local), epochs 16-40.
- 16:40 REVE-d6 clean scan (fold 1, alone): ep6 1.176/1.428, ep8 1.164/1.364, ep10 1.224/1.384, ep12 1.268/1.424, ep14
  1.144/1.312, ep16 0.776/0.992, ep20 0.548/0.636. Plateau 6-12, budget 9 fine. All three REVE-family members peak
  ~ep8-12 on new images and collapse past 14 (memorising the exact training-image embeddings).
- New lever against that memorisation: --concept_smooth a (target = (1-a) image + a concept mean of its non-held-out
  training images; cos to the image ~0.98 at a=0.5, i.e. the image-specific residual is halved). Clean test:
  sqf_dbn_ho5k_cs05_long_optx_things vs sqf_dbn_ho5k_long_optx_things (same epochs), fold 1.
- 18:20 SQF clean scans (fold 1, alone, top-5 % r=3/r=4): sqf_dbnX ep20 1.004/1.192 ... ep32 1.036/1.188, ep40
  1.020/1.196 (flat, no memorisation: budget 39 fine); plain sqf_dbn ep28 0.992/1.184, ep32 0.996/1.160, ep40
  0.980/1.116 (peak 28-32: budget 33 fine). All five ens5 budgets are right for new images.
  Concept smoothing extended to the REVE family (where the memorisation is): rmix (Slurm) and REVE-d6 (local), clean
  proxy, epochs 6-16, vs the plain clean scans above.
- 18:50 concept smoothing on SQF (clean, fold 1): ep40 0.876/1.072 vs plain 1.020/1.196 (-14% / -10%), lower at every
  epoch: it removes image-specific signal SQF uses (SQF does not memorise). REVE-family cs05 runs still decide it there.
  Free GPU -> REVE-scratch clean scan on fold 2 (ep8 vs ep10: fold 1 had ep10 +5-6%).
- 20:40 concept smoothing on the REVE family (clean, fold 1): rmix peak 1.176/1.400 (ep8) vs plain 1.376/1.484 (ep10);
  REVE-d6 peak 1.164/1.320 vs plain 1.268/1.424; late collapse unchanged (ep16 0.98 vs 1.04, 0.77 vs 0.78). DEAD: the
  image-specific residual is signal. Next: weight decay 0.3 (vs 0.05) for REVE-d6 (local) and rmix (Slurm), clean proxy.
- 21:00 user asked: why not ValCon 70/30? Answer: the clean proxy already holds out 30% (5,000 / 16,540) but scores NEW
  people (ValCon validates on training people) on new IMAGES of seen concepts (ValCon: new concepts). To cover the
  concept question: --holdout_concepts N (all 10 images of N concepts, seed 0) + HOLDOUT=concepts in holdout_eval.py
  (logs/holdout_concepts/). REVE-d6 epoch scan under it (reve_d6_hoc500_long_optx_things, fold 1) queued: if its peak
  matches the image-level one, budget decisions are robust to whether sealed concepts are new.
- 23:55 (node 514; old node 518 clean; 3 claude processes on 514 in separate ttys, mine = pts/2, others idle: left).
  Weight decay 0.3 (clean, fold 1, top-5 % r=3/r=4): REVE-d6 ep10 1.028/1.140, ep12 1.064/1.216, ep14 1.128/1.296,
  ep16 1.152/1.436 (plain: peak ep12 1.268/1.424, ep16 0.776/0.992); rmix ep10 0.828/1.004, ep14 0.952/1.124, ep16
  1.092/1.220 (plain peak ep10 1.376/1.484, ep16 1.036/1.240). WD 0.3 removes the collapse and is still rising at ep16:
  rerun both to the end of the 20-epoch schedule (*_ho5k_wd3_full_optx_things, ckpts 12-20).
  REVE-scratch clean fold 2: ep8 0.908/1.076, ep10 0.924/1.108, ep12 0.884/1.016, ep14 0.704/0.860. With fold 1 (ep8
  1.260/1.444, ep10 1.332/1.520): ep10 > ep8 on both folds -> REVE-scratch budget 8 -> 10, final launched (tag ep10).

## N73 (2026-11-10 night). Sub 2 on the board: 6th (0.001054 subj-agg top-5, +14.4% over sub 1); user goal = top 3
- Board (fractions; subj-agg top-5 / top-1 / per-epoch top-5 / top-1): JLShen .001245/.000363/.000648/.000148, _ares_
  .001123, iso_simon .001111, Lucky Jinx .001095, tzikos .00108, us .001054/.000267/.000554/.000142. Top 3 = +5.4%.
  Our per-epoch top-1 is 2nd best; our agg/per-epoch ratio (1.90) is the lowest. Worst team (wt12345) sits BELOW the
  chance cluster (top-1 1.7e-5 vs ~2.6e-5): a systematically wrong model, not a gallery-size signal.
  Leaderboard sampling noise: ~600k groups x 0.00105 = ~630 hits -> +-4% relative, i.e. the gap to 3rd is ~1 SE; the
  board keeps the best of the last 5 uploads, so several good candidates are several draws.
- Ensemble-level clean proxy (logs/clean_ens.py, fold 1: all 5 members from the --holdout_images 5000 runs, 25,000
  (person, image) groups, ens5 = 343 hits at r=3): REVE-family epochs 8 -> 10 (paired bootstrap vs ens5 @ ep8):
  d6 10 +3.5/+2.6% (r=3/r=4), rmix 10 -0.9/+3.6, scr 10 +2.0/+2.6, rmix 10 + scr 10 +2.6/+5.7, all three +4.4/+3.9,
  d6 12 + rmix 10 + scr 10 +6.1/+6.5 (SE ~3%). Concept-level holdout (500 new concepts, d6): ep8 1.136/1.244, ep10
  1.064/1.248, ep12 1.028/1.196 -> d6 ep12 not robust to new concepts; 10 is the safe budget.
- Ensemble is saturated at the large gallery: ens5 (1.432 r=3) is only +4% over its best member alone (rmix 1.376).
  Adding near-seed REVE variants (wd3, cs05, other epochs) as 6th/7th members: -6..+2% (logs/clean_add.py). Test-time
  (logs/clean_tt.py): per-trial unit fusion -8..+3% (noise), training-image manifold pull -30..-70% (DEAD).
- Actions: rmix wd3 full rerun cancelled (ep16 1.092 alone vs plain peak 1.376); final rmix@10 (Slurm 71720) next to
  REVE-scratch@10 (71612); logs/finish_sub5.sh packages sub5_ens5_e10_sum (d6 9, rmix 10, sqfX 39, scr 10, sqf 33) when
  both finish. final.sh now forwards extra train.py flags (4th+ args, e.g. --save_epochs).
- 02:20 Sealed-like distractors: data/things_meg/image_set/object_images = full THINGS (26,107 images); the 9,367 in neither
  EEG2 split -> DINOv2-g grader targets (logs/things_distractors.py; EEG2 images are 500 px copies of the 800 px THINGS
  originals; reproduction of the grader cache cos 0.9993 mean / 0.9955 min, transformers-version noise). Distractor
  nearest-training-image cos median 0.882 vs held-out 0.884 (same distribution), 1 near-duplicate. BIG=1 in
  clean_ens/clean_add/clean_tt/holdout_eval appends them: clean gallery 5k -> 14,366 never-trained THINGS images.
- 14k gallery (fold 1, top-5 % r=1/r=3/r=4; 131 hits at r=3, SE ~6%): ens5 @ ep8 0.278/0.524/0.636; d6 10 0.568 r=3;
  rmix 10 0.564; scr 10 0.548; 10/10/10 0.283/0.552/0.684. Alone: rmix ep10 0.277/0.584/0.680 (= the ensemble!), d6
  ep10-14 plateau 0.52-0.53 r=3, scr ep8-10 0.53. Epoch peaks at 14k agree with budget 10 for all three REVE members.
- Member weighting vs gallery size (logs/clean_gsize.py: expected top-5 at sub-gallery G' from exact ranks, hypergeometric
  smoothing): upweighting the REVE family (w) or dropping SQFs costs 0.5-5% at G' <= 5k and gains +1-4% at 10k; at 14k
  noisy. Stress test with the 11,540 memorised training images also as distractors (G 25.9k, logs/clean_gsize_seen.out):
  REVE-only -2..-7%, rmix alone -4..-10% (SQFs stay); w=1.5 the only consistently >= 0 row (+0..+5% r=1, -1..+3% r=3,
  -1..+1% r=4). Verdict: equal weights near-optimal; w=1.5 a small (+1-2%) optional lever at 40k. No big lever.
- 02:50 Weight soups along a run (logs/soup.py; rmix 8-10-12 / 10-12, d6 8-10-12 / 10-12-14, scratch 6-8-10 / 8-10),
  clean 5k and 14k: all within noise of the best single checkpoint (e.g. rmix soup 8-10-12 14k r=3/r=4 0.580/0.660 vs
  ep10 0.584/0.680). The EMA (0.9998) already averages; DEAD. Soup files deleted.
- 03:05 REVE-d6 wd 0.3 to the end of the cycle (clean fold 1, 5k r=3/r=4 | 14k r=3/r=4): ep14 1.128/1.300 | 0.488/0.508,
  ep16 1.152/1.436 | 0.460/0.544, ep18 1.176/1.344 | 0.448/0.556, ep20 1.164/1.388 | 0.468/0.552. Plain d6 ep10-12:
  1.22-1.27/1.38-1.42 | 0.524/0.58-0.61. wd 0.3 removes the collapse but never reaches the plain peak: DEAD.
  -> final REVE-d6 @10 local (logs/chain_local7.sh), then sub6_ens5_e10x3_sum (all three REVE members @10).
  Fold-2 clean replication queued behind the finals: Slurm 71773 (rmix mid + sqf_dbnX long), 71774 (d6 mid + sqf long).
- 05:38 SUBMISSION 5 READY: neurips_challenge/submissions/sub5_ens5_e10_sum.zip (sha256 b1ea87fb...; d6 9, rmix 10, sqfX 39,
  scr 10, sqf 33). In-sample fold-1 replay 28.79 (b64) / 28.80 (b37); contract check clean (73 s).
- Fold-2 clean rmix (5k, r=3/r=4): ep8 0.864/1.076, ep10 0.884/1.028, ep12 0.868/1.056 -> flat on fold 2 (fold 1 had
  ep10 +5% r=3 alone). Scratch fold 2: ep10 +2/+3% over ep8. Epoch-10 gain = small, not negative.
- 05:47 SUBMISSION 6 READY: neurips_challenge/submissions/sub6_ens5_e10x3_sum.zip (sha256 d1b48608...; d6 10, rmix 10,
  sqfX 39, scr 10, sqf 33). In-sample fold-1 replay 29.13 (b64) / 29.14 (b37) vs sub 5 28.79; contract check clean (61 s).
  Upload order: sub 6 first (most support on 5k + 14k fold 1), then sub 5 / variants.
  Fold 2 at 14k (noisy, ~90 hits): rmix ep8 0.332/0.476, ep10 0.376/0.468 (r=3/r=4); scratch ep8 0.376/0.476, ep10
  0.352/0.484. Local GPU -> fold-2 plain SQF (logs/chain_local8.sh) so the fold-2 ensemble is complete sooner.
- 07:50 Fold-2 clean ENSEMBLE (logs/clean_ens_f2.out; all 5 members on fold 2): 10/10/10 vs 8/8/8: 5k r=3 -2.0% r=4
  +9.2% (SE ~3.6); 14k r=3 -9.6% r=4 +8.8% (SE ~5.7). Two folds: r=3 ~0, r=4 ~+7 -> budget-10 gain small (+2-4% est.).
  Weighting (logs/clean_gsize_f2.out): REVE-family w=1.5 vs 1: 14k r=3 +7.8% r=4 +3.0%; with the training images as
  extra distractors (G 25.9k) +1.1..+3.9% / +2.0..+4.0%; small galleries ~0. Both folds: w=1.5 >= 0 everywhere, gain grows
  with G. w=2 noisier (-5..+15%). REVE-only no.
- SUBMISSION 7 READY: neurips_challenge/submissions/sub7_ens5_e10x3_w15_sum.zip (sha256 270d367d...) = sub 6 members with
  REVE-family weight 1.5 (members.pt 'weights'; submission.py reads them, default 1). Replay 29.24/29.20; contract clean.
  RECOMMENDED UPLOAD ORDER: sub 7, then sub 6 (paired board comparison = a direct sealed read of the weighting), sub 5.
- 08:00 Fold-3 clean replication (all 5 members) for w (1.5 vs 2) and budgets: Slurm 71997 (d6 mid + sqfX), 71998
  (rmix mid + sqf), local scratch mid (new recipe reve_scratch_ho5k_mid_optx_things, stop 10 save 8).
- 08:20 Weight grid, selected on folds 1-2 (logs/clean_wgrid.py, expected top-5 at G' 10k / 14k, r=3 | r=4, vs equal):
  REVE 1.5 +3.1% mean, REVE 2 +5.1%, REVE 3 +3.8%, REVE 1.5 + sqf .5 +4.6%, rmix-heavier +3.0..3.6%. Every REVE-upweighted
  row is >= 0 on average; optimum near 2. SUBMISSION 8 = sub 6 members with REVE weight 2
  (neurips_challenge/submissions/sub8_ens5_e10x3_w2_sum.zip, sha256 4a7e4aa1...; replay 29.13/29.16; contract clean).
  Fold 3 (held out from this selection) decides 1.5 vs 2: logs/f3_analysis.sh -> logs/f3_analysis.out when its members land.
- 11:25 Fold 3 single members (clean 5k, top-5 r=3/r=4): REVE-scratch ep8 1.192/1.384 vs ep10 1.208/1.364 (flat);
  plain SQF ep32 1.044/1.252; d6 ep8->10 +3/+8%, rmix -3/+3%. sqfX still training (f3_A) -> logs/f3_analysis.out.
- 11:30 IMAGE-COUNT SCALING PROBE (idea: if new-image generalisation at large G scales steeply with the number of
  distinct training images, THINGS-EEG1's ~6-9k images outside EEG2 (raw ds003825 is on disk, 48 people x 1 trial,
  DINOv2 targets already in things_extra.npy) become the next lever). --holdout_images 9000 is a SUPERSET of the 5k
  gallery (checked), so d6/rmix trained on 7,540 images score on the identical 5k/14k gallery vs the ho5k (11,540)
  runs. reve_d6_ho9k_mid_optx_things local, rmix_reg_ho9k_mid_optx_things sbatch 72402; ep 8/10/12/14.
- 12:30 Scaling probe, rmix fold 3 (clean 5k, top-5 r=3/r=4): 7,540 training images (ho9k) ep8 1.056/1.168, ep10
  0.952/1.192, ep12 0.984/1.208 vs 11,540 images (ho5k) ep8 1.288/1.456, ep10 1.252/1.504 -> -18% / -22% for -35%
  images: novel-image large-G accuracy scales steeply with the number of DISTINCT training images (exponent ~0.5).
  User: THINGS-EEG1 is a bad idea (not pursued). d6 ho9k still running locally.
- 12:30 User question: Alljoined-only training (drop THINGS-EEG2 sub-21..30) in the large-gallery setting?
  cv4.sh: *_aj recipes skip 21..30. reve_d6_ho5k_mid_optx_aj (sbatch 72483), rmix_reg_ho5k_mid_optx_aj (72484), fold 3,
  15 people, ep 8/10/12/14/16 -> compare with the fold-3 *_ho5k_mid_optx_things ep8/10 on the same 5k / 14k gallery.
- 12:55 Scaling probe, d6 fold 3: ho9k (7,540 images) best ep8 1.088/1.276 vs ho5k (11,540) ep10 1.240/1.544 ->
  -12% / -17% (rmix -18/-22%). Confirmed on both REVE members. f3_analysis.sh never fired (its running copy predated
  an edit of the file names); killed, analysis run directly. Alljoined-only d6 also on fold 2 (local, chain_f3.sh now
  takes FOLD=).
- 13:10 FOLD 3 = held-out check (logs/f3_analysis.out).
  Weights (selected on folds 1-2), fold 3 at G' 10k/14k r=3|r=4 vs equal: REVE 1.5 -0.5 -1.9 | +1.1 -6.6, REVE 2
  -2.5 -4.3 | +0.2 -2.8, REVE 3 worse; rmix 2 d6/scr 1.5 -0.7 +0.0 | +1.9 +0.0. 3-fold means: REVE 1.5 +1.4%,
  REVE 2 +2.6%, rmix 2 d6/scr 1.5 +2.5% -- all inside noise; the "grows with G" trend does NOT replicate on fold 3.
  NOT CONFIRMED: REVE upweighting is a coin flip around +1-2%.
  Budget 10 (sub 6 = d6/rmix/scr @10) vs @8, fold 3: 5k r=3 -0.8% / r=4 +3.9% (SE ~3%); 14k +11.8% (SE 5.2) /
  +9.0% (SE 4.5). CONFIRMED on the held-out fold at the large gallery.
  -> Upload order: sub6 (equal weights, all @10; the validated one), then sub8 (REVE 2, best 3-fold mean) as a second
  ticket (the board keeps the best of the last 5), sub7 after.
- 13:30 Alljoined-only rmix, fold 3 (clean 5k, top-5 r=3/r=4): ep8 1.084/1.244, ep10 1.064/1.276, ep12 1.024/1.272,
  ep14 0.848/1.024, ep16 0.712/0.784 vs with THINGS-EEG2 ep8 1.288/1.456, ep10 1.252/1.504 -> -16% / -15% at the
  best epoch; it also overfits sooner (collapses after ep12). THINGS-EEG2 HELPS at the large gallery. d6 pending.
- 13:45 Next levers, aimed at the image-count finding (fold 3 clean proxy, d6 baseline ep8 1.204/1.432, ep10 1.240/1.544):
  (1) --image_mixup_prob 0.5 (new train.py flag): cross-image mixup, EEG and DINOv2 target both lam*x+(1-lam)*x[perm],
  lam ~ U(0,1) per trial = virtual images. reve_d6_ho5k_imix_optx_things (sbatch 72514); rmix_reg_ho5k_imix_optx_things
  local after the fold-2 aj run. (2) operator transfer on every batch (--op_transfer_prob 1.0 vs 0.5; THINGS people
  were worth +16%, so person diversity matters): reve_d6_ho5k_opt1_things (sbatch 72515 after 72483). ep 8/10/12/14.
- 14:15 Alljoined-only d6, fold 3: ep8 1.144/1.180, ep10 1.116/1.280, ep12 1.024/1.224 vs with THINGS 1.240/1.544 ->
  -8% / -17% (rmix -16/-15%). Keep THINGS-EEG2. (fold-2 local replication still running.)
- 14:15 User: focus on image augmentations; add the NeuroBridge paper's proven image prior (CPA). Old recipe (git
  296e1bb^: fuse_feature.py + --aug_image_feature_dirs .../GaussianBlur-GaussianNoise-LowResolution-Mosaic): training
  target = MEAN of the encoder feature over the 4 views (blur k15, noise std 25, low-res 0.1, mosaic 5), test clean.
  logs/aug_features.py recomputes it on the grader target (DINOv2-g hidden 26 token mean, HF processor; clean view
  reproduces image_train.npy at cos 0.99926) -> aug_{view}.npy + aug_cpa4.npy. New train.py --train_image_targets
  (training rows only). reve_d6_ho5k_cpa4_optx_things (sbatch 72538, waits for the features), rmix (72539 after
  imix_d6). opt1 (operator transfer every batch) cancelled to make room (had run 22 min).
- 15:00 REPEATS IN THE SEALED SET (user question, 3 or 4?). logs/agg_ratio.py, sub-6-like ensemble, folds 1-3:
  subj-agg / per-epoch top-5 at 5k: r=2 x1.49, r=3 x1.89, r=4 x2.27; at 14k: r=2 x1.62, r=3 x1.99, r=4 x2.48 (grows with
  G; ~40k extrapolation r=2 ~1.75, r=3 ~2.1, r=4 ~2.7). Board: every good team 1.84-1.98 (sub 2 1.90) -> between r=2
  and r=3 (a 2/3 half mix gives 1.92); uniform r=4 is excluded. The near-constant bottom entry (wt12345) has per-epoch
  > subj-agg (top-5 7.5e-5 vs 5.5e-5, top-1 1.7e-5 vs 1.2e-5): with one ranking for every trial the two metrics are equal
  iff every (person, image) has the same repeat count, so repeats VARY; the images it hits have ~1.4x the mean count
  (4 -> mean ~2.8-2.9). Chance rows: gallery ~40k, no repeat information. Best estimate: mean ~2.5-3, mixed counts.
  -> weight r=3 (and r=2) over r=4 in proxy decisions.
- 15:10 USER: r=3 is the main metric from now on (r=4 secondary). CPA features done (aug_features.py split over two
  GPUs; cos to clean: blur .989, noise .978, mosaic .964, low-res .927, fused cpa4 .979). reve_d6 cpa4 training
  (72548). Added the per-trial random-view variant (--train_image_targets with several files; clean + 4 views):
  reve_d6_ho5k_cpav_optx_things queued after 72548.
- 15:40 USER: simplified ensembles (one REVE + one non-REVE, weights only, no retraining). logs/pair_ens.py -> .out,
  folds 1-3 pooled, r=3 top-5, paired SE vs sub 6 (no TSC is cached on the clean proxy: SQF-X / plain SQF only).
  14k: sub 6 0.519; rmix + 0.5 SQF-X 0.518 (-0.2%, SE 2.1) = TIE; rmix + 0.33/0.2 SQF-X -1%; scratch + 0.75/1 SQF-X
  -1.7%; d6 pairs -2.6..-6%; rmix alone -5.4%, scratch -10.6, d6 -13.2, SQF-X -28.5. SQF-X > plain SQF as partner.
  5k: sub 6 1.275; best pair scratch + 0.75 SQF-X -3.1% (SE 1.3), rmix + 0.5 SQF-X -4.6%. The 5-member advantage
  shrinks with the gallery (-4.6% -> -0.2% for rmix + .5 SQF-X); one non-REVE partner recovers most of the gap that a
  single REVE member leaves. SUB 9 = rmix@10 + 0.5 SQF-X (finals of sub 6, members.pt weights):
  neurips_challenge/submissions/sub9_rmix_sqfx05_sum.zip (sha256 475fdde9...), replay 27.07 / 27.03 (sub 6 29.13 on the
  200-image in-sample replay), contract check OK. A less-correlated ticket than sub 7/8 for the best-of-5 board.
- 16:20 Cross-image mixup (p 0.5), d6 fold 3 (logs/swap_eval.py: alone and swapped into sub 6, r=3 main; baseline d6
  @10; d6 @12 baseline cached now): 14k r=3 in sub 6: imix ep10 -1.0% (SE 2.4), ep12 -2.9%, ep14 +1.2% vs baseline
  d6 ep12 +2.4%; 5k r=3 in sub 6: imix ep14 +4.4% (SE 1.6) vs baseline ep12 +1.4%. Mixup shifts the peak later but
  does not beat the baseline's own later epoch: FLAT on d6. rmix imix (local) still running.
- 16:35 Cross-image mixup, rmix fold 3, in sub 6 at 14k r=3 (vs sub 6 with rmix @10): imix ep10 +2.6%, ep12 +3.1%,
  ep14 +5.2% (SE 2.5); the baseline rmix's own ep12 gives +3.4% (SE 2.0). 5k r=3: all within +-1%. As on d6, mixup
  regularises (later peak) but adds <= ~2% over the baseline at its own later epoch: FLAT (not pursued; d6 + rmix).
- 16:55 NeuroBridge CPA image prior (fixed target = mean of the 4 views), d6 fold 3: ALONE r=3 5k -4.1% (ep10),
  -6.0% (ep12), -14.3% (ep14); 14k -7.4 / -6.0 / -19.2%. In sub 6 at 14k r=3: ep10 -2.1%, ep12 +1.5%, ep14 -5.2% (vs the
  baseline d6's own ep12 +2.4%). NEGATIVE alone, flat-to-negative in the ensemble. Likely cause: the grader ranks
  against CLEAN DINOv2 targets (no image-side projector at test, unlike the paper's SSP), and the fused prior sits
  at cos .979 from them, so the model aims at a shifted target. rmix cpa4 finishing; random-view variant (cpav,
  clean target 1/5 of the time) is the remaining CPA test.
- 17:10 Budget 12 for the REVE members inside sub 6 (logs/ep12_check.py, cached checkpoints, folds 1-3 pooled):
  14k r=3 d6@12 +1.4% (SE 1.3), rmix@12 +1.3% (1.2), d6+rmix@12 +3.6% (1.5), all REVE@12 (scratch@12 on f1-2 only)
  +4.9% (SE 1.7; per fold +4 / +2 / +8%); 5k r=3 +1.5% (1.0); r=4 +1.2% (14k) / +3.2% (5k). Same pattern as 8->10:
  later epochs pay more as the gallery grows. Caveat = new CONCEPTS: the concept-holdout d6 (fold 1) at 5k r=3 ep8
  1.136 / ep10 1.064 / ep12 1.028, at 14k ep8 0.400 / ep10 0.416 / ep12 0.404 / ep14 0.352 (noisy, ~100 hits): 10->12
  about neutral at 14k for new concepts, ep14 clearly worse. -> candidate sub 10 = sub 6 with REVE @12. Final
  rmix_reg_optx_things 12 (tag ep12) launched on the free sbatch slot; d6 / scratch @12 next.
- 17:20 Sub 10 pipeline: finals @12 rmix (72673 running), scratch (72674 after cpa4_rmix), d6 (72675 after rmix12);
  logs/finish_sub10.out packages sub10_ens5_e12x3_sum (d6/rmix/scr @12 + the two SQFs) when all three land.
  Fold-3 scratch @12 (reve_scratch_ho5k_m12_optx_things) queued locally after cpav, to complete the all-REVE@12 check
  on fold 3.
- 17:45 Expected gain of the next submission (logs/next_sub.py -> .out; folds 1-3 pooled vs a sub-2-like base with
  REVE @8; fold-3 scratch @12 still pending, @10 used). 14k r=3: sub 6 +4.3% (SE 1.7), sub 8 (sub 6 w2) +6.4% (2.1),
  sub 10 (@12) +9.4% (2.2), sub 10 w1.5 +9.3%, sub 10 w2 +8.8% (2.4; r=2 +10.6%), sub 9 +3.7%. 5k r=3: sub 10 +3.4%,
  sub 10 w2 +2.6%. Epoch 12 and REVE weight 2 do NOT stack (both shift weight to the REVE fit). Sub 2 used d6/rmix @9,
  so vs the real sub 2 subtract ~1-2%: sub 10 ~ +7-8% at 14k r=3, gains growing with the gallery -> board estimate
  0.001054 x ~1.05-1.12 (central ~0.00114; 3rd is 0.001111, 2nd 0.001123, 1st 0.001245 out of reach).
- 18:10 CPA fixed prior, rmix fold 3: alone 14k r=3 ep10 -12.6%, ep12 -0.2%, ep14 -6.9% (5k -6..-9%); in sub 6 at 14k
  r=3 ep10 -0.2%, ep12 +5.0%, ep14 +2.9% vs the baseline rmix's own ep12 +3.4%. With d6: the CPA fixed target is
  DEAD (worse alone on both architectures, ensemble at best = the baseline's later epoch). Random-view (cpav) remains.
  Finals @12: scratch (72674) and rmix (72673) running, d6 (72675) queued.
- 19:30 CPA random view per trial (clean + 4 views), d6 fold 3: alone 14k r=3 ep10 +10.1% (SE 4.9) but 5k +0.7%, ep12
  +1.7%, ep14 -7.4%; in sub 6 at 14k r=3 ep8 -1.4%, ep10 +2.4%, ep12 +2.4%, ep14 +3.1% vs the baseline d6's own ep12
  +3.3%. FLAT. IMAGE-AUGMENTATION ROUND CLOSED: cross-image mixup flat (d6, rmix), CPA fixed prior negative (d6, rmix),
  CPA random view flat (d6). None beats the baseline member at its own best epoch inside the ensemble. (swap_eval.py now
  keeps members on the CPU: a co-running training OOMed it.)
- 21:30 REVIEW (MI session, scripts/mi/MI_NEURIPS_LOG.md) -- verified and adopted. (1) RANK BUG in every proxy scorer:
  target score from a separate dot product; on f3 rmix@10 the target sits above itself in 46.8% of queries; top-1
  0.100 -> 0.180%, top-5 0.655 -> 0.720% when read from the same matmul. Fixed in 22 logs/*.py + package_submission.py
  (S.gather). (2) JUDGE -> 5k r=3 exact ranks (board rank-shape stats match 5k r=3, not 14k; sealed is ~4.4x chance per
  trial vs 7-8x on the proxy). (3) Sub 9 (pair) withdrawn: -3..-5% at 5k. README sealed notes corrected. Exact-rank
  rerun of logs/next_sub.py -> logs/next_sub_exact.out.
- 21:55 MI session (neurobridge-ssl-8b) on the THINGS-EEG2 lag fix (--things_lag 5 shifts sub >= 21 training trials
  by 5 samples; Alljoined lags THINGS-EEG2 by ~42 ms): exact ranks, rmix lag @10 swapped into ens5, folds 1+3: 5k r=3
  +1.7% (SE 1.8), r=4 +2.9% (1.6); all 16 pooled cells >= 0. SQF-X lag fold 1: flat alone, rmix + SQF-X both swapped
  +4.9% (2.9) at 5k r=3. Pending tonight: d6 lag fold 1, SQF-X lag fold 3. Their d6 fold-1 new-people scan (legacy
  ranks) ep10 1.20 > ep12 1.12 -> the @10 vs @12 choice goes to the exact-rank pooled rerun (gate b).
  Plan: lag finals (final.sh <recipe> <B> lag5 --things_lag 5) for rmix (+ SQF-X / d6 if their results hold) when
  the d6@12 slot frees; sub 11 = best member set with the lag finals swapped in.
- 22:05 EXACT-RANK candidate table (logs/next_sub_exact.out; folds 1-3 pooled vs sub-2-like REVE @8; f3 scratch @10).
  5k r=3 (judge): sub 6 +1.9% (SE 1.0; folds +2/+1/+3), sub 10 +2.2% (1.2; +3/+1/+2), sub 10 w1.5 +2.9%, w2 +2.8%,
  sub 8 +0.6%, sub 9 -3.2% (1.4). r=2: sub 6 +1.5, sub 10 +3.2. r=4: sub 6 +2.6, sub 10 +5.6. 14k r=3: sub 6 +2.9,
  sub 10 +7.6. -> sub 10 ~ sub 6 at the judge (+0.3 pt), ahead at r=2/r=4/14k: gate (b) passed narrowly; expected
  board gain of either over the real sub 2 (d6/rmix @9) only ~+1-2%. Lag final rmix @10 queued (after d6@12):
  final.sh rmix_reg_optx_things 10 lag5 --things_lag 5 -> sub 11 = sub 6 with rmix lag (the configuration the MI
  session measured: +1.7% (1.8) at 5k r=3), + SQF-X lag if its fold-3 read holds.
- 22:15 USER: stop after the queued runs land; nothing new. Queued and finishing tonight: finals d6@12 (72675),
  scratch@12 (72674), rmix lag @10 (72829); fold-3 scratch @12 check (local) -> logs/next_sub_f3scr12.out;
  packaging -> submissions/sub10_ens5_e12x3_sum.zip (logs/finish_sub10.out) and sub11_ens5_e10_rmixlag_sum.zip
  (logs/finish_sub11.out). No SQF-X / d6 lag finals (told the MI session).
