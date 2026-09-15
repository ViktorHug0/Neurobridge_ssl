# Ensemble experiments — inventory of runs, measurements, and file locations

Status date: 2026-09-07. Compiled from `ensemble_results_report.md`,
`analysis/ensemble_results_analysis_20260823.md`, `validation/README.md`, the per-track
`*.md` notes, and a re-scan of `results/things_eeg/`. Numbers marked *(rescanned)* were
recomputed from the per-fold result CSVs for this document; all others are quoted from the
files cited in the same row or section.

---

## 1. Measurement conventions used by every number below

- Dataset/task: THINGS-EEG2, inter-subject LOSO (train on 9 source subjects, test on the
  held-out one), 200-way closed-set retrieval.
- Metric: `best top1 acc` (best-epoch top-1 on the held-out subject), plain cosine.
- No SAW, CSLS, Sinkhorn, Procrustes, or other test-set adaptation is applied in this
  track. The transductive SATTC family is a separate track.
- Fusion: each member's 200×200 cosine score matrix is standardized per query (`row_z`),
  then uniformly averaged. Both EEG and image sides are L2-normalized before the dot
  product. Rules implemented in `retrieval_fusion.py`: `raw`, `probability`, `row_z`,
  `row_z_probability`, `rank`, `vote`.
- Three checkpoint-selection protocols appear in the results tree:

| Name | Checkpoint chosen by |
|---|---|
| test-selected | minimum held-out test loss on the LOSO subject |
| ValCon | held-out source concepts (10%, seed 20260822) |
| LOSO-subject val | a held-out source subject (8 training subjects remain) |

- Measured difference between protocols, 10 matched ATM/TSConv InternViT-depth arms:
  test-selected exceeds ValCon by **+2.235pp** on average (per-arm range +0.25 to +4.05).
  For fixed committees the measured differences were +2.65, +1.70, and +2.35pp.
- Reported seed SD on this benchmark is ±0.20 (agent memory `current-best-model`).

---

## 2. Fixed-pool committee sweeps (test-selected)

Exhaustive all-ten-fold selection over the 27-arm roster, row-z uniform mean
(`analysis/extended_roster_z_sweep.json`, `analysis/target_matrix_test_extended_z.json`):

| k | top-1 | members | combos evaluated |
|---:|---:|---|---:|
| 1 | 37.05 | `ge100` | 27 |
| 2 | 44.10 | `atm_iv_group_e75`, `iv33_group_e75` | 351 |
| 3 | 46.55 | `atm25`, `iv33_group_e75`, `sqf_bigg` | 2,925 |
| 4 | 48.20 | `sqf28`, `atm_vith`, `atm_iv_group_e75`, `iv33_group_e75` | 17,550 |
| 5 | 48.85 | `atm25`, `iv35`, `atm_iv_group_e75`, `iv33_group_e75`, `sqf_bigg` | 80,730 |
| 6 | 49.50 | + `atm_vith` | 296,010 |

Per-k marginals: +7.05, +2.45, +1.65, +0.65, +0.65.

k=4 committee fold scores: `59.5 57.0 36.5 40.0 47.0 42.0 48.0 34.5 52.5 65.0`.
This set is the selected winner on all ten leave-one-fold-out nine-fold subsets of its
declared 10-arm pool. Drop-one accuracies: `atm_vith` removed 45.30, `atm_iv_group_e75`
removed 45.65, `iv33_group_e75` removed 44.75, `sqf28` removed 44.95 (drops 2.55–3.45).
Individual-member oracle 62.65. Conditional accuracy by number of individually correct
members: 0 → 5.0% (747 trials), 1 → 37.8% (429), 2 → 83.2% (351), 3 → 100% (270),
4 → 100% (203). 37 trials are answered correctly where no member ranks the truth first.

ValCon roster and matched test-selected controls
(`analysis/target_matrix_valcon_extended_z.json`, `target_matrix_ctl_extended_z.json`):

| k | ValCon | matched test-selected control |
|---:|---:|---:|
| 1 | 33.40 | 36.35 |
| 2 | 41.40 | 44.60 |
| 3 | 43.75 | 46.00 |
| 4 | 44.90 | 47.75 |
| 5 | 45.15 | 48.40 |
| 6 | 45.65 | 48.90 |

Nested leave-one-fold-out member selection, by pool: mixed 16-arm 44.45, InternViT-only
13-arm 44.50, depth-only pools 43.00–45.50. Best all-fold complex (transform + weight +
member) rule 48.65, nested LOFO 44.35. A k=5 all-ten figure of 45.50 became 44.25 when the
candidate set was widened by two arms that did not enter the winner. At k=4, nested
selection returns 7 distinct member sets across the 10 folds.

Historical fixed sets: `atm_iv` + `tsconv_eva` + `tsconv_vith` = 44.60 raw-mean
(nested check returns the same triple on every fold subset); reference quartet
(`atm_iv`, `ge100`, `tsconv_eva`, `tsconv_vith`) = 45.35 fixed, nested-LOFO 43.95;
frozen five with row-z + signed power 1.25 = 47.00 all-ten and 46.64 on the seven subjects
not used to pick it (`test_selected/frozen_fusion_baseline.json`).

---

## 3. Diversity-axis measurements

Pairs drawn from stored sweeps; score correlation is Pearson between per-query
standardized matrices over all ten subjects; pair gain is against the better solo member
(`analysis/ensemble_evidence_audit.py`):

| Axis changed | n pairs | score corr | correctness corr | mean pair gain |
|---|---:|---:|---:|---:|
| Seed only | 3 | 0.956 | 0.653 | +1.58 |
| Encoder only (same InternViT-28) | 10 | 0.869 | 0.402 | +1.43 |
| ATM, image target differs | 6 | 0.893 | 0.470 | +2.70 |
| TSConv, image target differs | 15 | 0.899 | 0.458 | +2.76 |
| ATM, InternViT depth differs | 10 | 0.915 | 0.542 | +2.60 |
| TSConv, InternViT depth differs | 10 | 0.918 | 0.497 | +3.20 |
| Encoder and target both differ | 23 | 0.814 | 0.339 | +5.17 |

Single controlled pair, same InternViT-28 target: ATM-28 35.20 + TSConv-28 36.10 → 40.65
row-z (+4.55 over the better solo), score corr 0.851, member oracle 49.95.

Encoder composition, 12-arm depth grid (6 InternViT layers × {ATM, TSConv}), 220 triples:
ATM-vs-TSConv composition accounts for R² = 0.69 of ensemble variance; depth spread
r = +0.08 at k=3 and +0.41 at k=2; mean solo of members r = +0.297. k=5 group means by
(TSConv + ATM) count: 5+0 39.48, 4+1 40.95, 3+2 42.64, 2+3 43.78, 1+4 43.96, 0+5 41.06;
the best all-single-encoder 5-arm set ranks 701st of 792.

Depth-only committees:

| k | best all-ATM | best all-TSConv |
|---:|---:|---:|
| 1 | 36.50 | 36.10 |
| 2 | 38.65 | 39.75 |
| 3 | 39.70 (`25+31+35`) | 41.15 (`28+31+33`) |
| 4 | 39.90 | 41.35 |
| 5 | 40.15 | 41.40 |
| 6 | 39.85 | 40.70 |

Mixed ATM/TSConv depths at k=3: 45.00 (`ATM-28 + TSConv-33 + TSConv-35`).

Pairwise complementarity study, 45-model test-selected roster, 990 pairs, fixed row-z
(`analysis/pair_complementarity_testselected.{csv,json}`). Spearman ρ with gain over the
pair mean: score corr −0.920, margin corr −0.910, correctness corr −0.907, wrong-winner
agreement −0.904, prediction disagreement +0.896, oracle headroom +0.669. For gain over
the stronger member, oracle headroom gives ρ = 0.898. The 990 pairs share 45 models and
10 subjects, so they are not independent observations.

Cross-backbone additions: adding `atm_eva` (32.15) and `tsconv_eva` (33.95) to the 12-arm
pool moves nested LOFO by +0.00/+0.25/+1.05/−1.25/+0.00 for k=2..6 (mean +0.01);
`atm_eva` does not enter a top-20 combination. `EVA02-E-14_layer35_mean` has one extracted
layer, so no depth ladder exists for it.

Parameter-budget replication (`synthetic_subjects/best_small_ensembles.md`): the winning
triple retrained at bb=128/fd=128 gives 42.25 at 2.34M params (0.49× the 4.754M baseline),
+6.35 over the 35.90 baseline, p=0.0039. The parameter-matched three-seed control gives
34.65. Difference 7.60pp, p=0.0020, 10/10 folds. Post-shrink solos: `atm_iv` 35.20,
`eva` 33.50, `vith` 31.65, `eva` seeds 3301/3302 32.60/32.85.

---

## 4. Seed-ensemble controls

Three matched runs per encoder, row-z (`analysis/report_figures/seed_ensembles.png`):

| | TSConv | ATM |
|---|---:|---:|
| seed 3300 / 3301 / 3302 solo | 35.05 / 34.85 / 34.60 | 35.20 / 34.40 / 35.15 |
| 3-seed row-z ensemble | 37.10 | 37.45 |
| difference vs best seed | +2.05 | +2.25 |
| mean pairwise score corr | 0.956 | 0.960 |
| fold-level 95% CI on the difference | `[-0.03, +4.33]` | `[-0.22, +4.72]` |
| individual-member oracle | 46.80 | 47.45 |

ATM committee per-subject: `42.0 46.5 26.0 29.0 46.0 23.0 33.0 35.0 45.0 49.0`.
A separate seed-3300 export named `pair` exists and is not numerically identical to the
`p3300` dump used in the audited committee (open provenance item, `ensemble_results_report.md`).

---

## 5. Training-side interventions (`decorrelated_models/`)

`train_twins.py` jointly trains two branches with individual multi-positive losses plus
`β·L_ensemble` (row-z mean) plus `λ·L_div` (negative-only squared score correlation), or
`γ_rescue` (detached soft responsibilities favoring the branch with the lower loss).
Both branches' initialization seeds are set explicitly, then a single training RNG is
restored so all arms see identical batches. Protocol writeup: `decorrelated_models.md`.
Reported metric is `pair_top1`.

### 5.1 Twin-TSConv correlation leash, test-selected, n=10 per arm *(rescanned)*

| λ | β | member a | member b | pair top-1 | gain over best | score corr | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | 35.55 | 37.40 | 37.55 | −0.35 | 0.96 | 44.60 |
| 0.01 | 0.5 | 34.80 | 34.85 | 36.25 | +0.60 | 0.94 | 43.70 |
| 0.05 | 0 | 35.15 | 34.65 | 38.25 | +2.50 | 0.96 | 43.55 |
| 0.05 | 0.5 | 35.15 | 35.25 | 37.70 | +1.15 | 0.94 | 43.30 |
| 0.10 | 0 | 35.00 | 34.70 | 36.90 | +0.80 | 0.95 | 43.55 |
| 0.10 | 0.5 | 34.65 | 35.05 | 36.55 | +0.65 | 0.93 | 44.00 |
| 0.50 | 0 | 35.15 | 34.40 | 36.70 | +0.30 | 0.57 | 44.50 |
| 0.50 | 0.5 | 33.80 | 33.95 | 35.35 | +0.20 | 0.62 | 42.45 |

### 5.2 Heterogeneous ATM+TSConv, batch 512, test-selected, n=10 per arm *(rescanned)*

| arm | member a | member b | pair top-1 | gain over best | score corr | oracle | headroom realized |
|---|---:|---:|---:|---:|---:|---:|---:|
| λ=0 (control) | 34.35 | 34.95 | 40.15 | +2.90 | 0.850 | 48.75 | 0.148 |
| λ=0.05 | 35.20 | 36.20 | 41.45 | +2.80 | 0.852 | 48.70 | 0.16 |
| λ=0.10 | 32.95 | 35.35 | 40.40 | +3.90 | 0.828 | 48.70 | 0.212 |
| λ=0.25 | 34.20 | 35.80 | 40.90 | +3.40 | 0.565 | 49.35 | 0.269 |
| γ_rescue=0.10 | 34.25 | 34.40 | 40.75 | +4.10 | 0.851 | 48.15 | 0.288 |
| γ_rescue=0.30 | 33.30 | 34.75 | 41.90 | +5.20 | 0.846 | 47.80 | 0.393 |
| γ_rescue=0.50 | 34.10 | 34.65 | 41.65 | +4.35 | 0.848 | 48.35 | 0.379 |

### 5.3 Rescue dose response under ValCon, n=10 per arm *(rescanned)*

| γ_rescue | member a | member b | pair top-1 | gain over best | score corr | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 30.55 | 31.85 | 38.10 | +4.25 | 0.83 | 45.05 |
| 0.10 | 32.05 | 32.25 | 38.55 | +4.30 | 0.83 | 45.70 |
| 0.30 | 30.30 | 32.30 | 37.75 | +4.05 | 0.83 | 44.70 |
| 0.50 | 30.40 | 31.60 | 37.60 | +3.25 | 0.83 | 44.45 |

The γ ordering under ValCon differs from the test-selected ordering in §5.2:
test-selected peaks at γ=0.30, ValCon peaks at γ=0.10 with γ=0.30 and γ=0.50 below the
γ=0 control. `ensemble_results_report.md` §5 and `decorrelated_models.md` were written
before these four arms completed and describe the test-selected ordering only.

### 5.4 Stochastic member specialization, n=10 *(rescanned)*

Arm `stoch_spec05_cdrop10_keep75_b05` (spectral gain SD 0.5, channel dropout 0.10,
per-query member keep 0.75, β=0.5, batch 512, all nine source subjects retained):
members 32.85 / 33.60, pair **35.40**, score corr 0.94, oracle 41.65, gain over best +0.60.
The matched `joint_b05_control` was not run past a one-fold smoke
(`decorrelated_models/stochastic_full_data.md`, execution note 2026-09-02); the one-fold
smoke recorded control pair 22.5 vs stochastic 22.0
(`results/.../stochastic_full_data_smoke/stochastic_vs_control.csv`).

### 5.5 Loss-component diagnostics

`analysis/decor_atm_tsconv_l005_loss_components.{csv,png}`,
`analysis/fusion_beta_training_curves.{csv,png,pdf}`,
`analysis/plot_decorrelation_loss_components.py`,
`analysis/plot_fusion_beta_training_curves.py`.

---

## 6. Source-subject roster perturbation

### 6.1 Cohort controls, 3 members each

| roster per member | ensemble top-1 | mean score corr | best of six fixed rules |
|---|---:|---:|---:|
| disjoint 3-of-9 | 31.80 | 0.778 | 31.85 (raw probability) |
| overlapping 6-of-9 | 35.70 | 0.896 | 35.85 (raw / probability) |
| full 9-of-9, cyclic soft weights | 38.20 | 0.957 | 38.25 (row-z probability) |

Largest rule-to-rule spread: 0.15pp. Manifests in
`results/things_eeg/subject_cohort_bagging/testselected_{triplets,overlap6,softweights_d030}/`.

### 6.2 Balanced 7-of-9 bagging, 8 members, 80 folds complete 2026-09-02

| quantity | top-1 |
|---|---:|
| mean individual member | 32.60 |
| best member per subject (diagnostic) | 36.80 |
| predeclared row-z ensemble | 37.85 |
| raw-score ensemble (diagnostic) | 38.10 |
| probability ensemble (diagnostic) | 38.05 |
| rank / vote (diagnostic) | 37.80 / 37.40 |
| member oracle | 62.10 |

Mean pairwise score corr 0.921. Row-z gain over mean member +5.25, over best member +1.05.
Per-subject table: `results/things_eeg/subject_cohort_bagging/testselected_balanced7x8/balanced7x8_ensemble.csv`.
Fused committee matrices written to `fused_scores/balanced7x8-subXX.npz`.
Protocol: `balanced_subject_bagging.md`. Code: `balanced_subject_bagging.py`,
`analyze_balanced_subject_bagging.py`, `run_balanced_subject_bag_fold.sh`.

### 6.3 Cohort members inside the depth roster

Adding the three 6-of-9 cohort members, or the fused cohort committee as a single
meta-member, to the 12-arm depth roster returns identical winners at every k
(43.90 / 45.00 / 45.40 / 45.85 / 46.15 for k=2..6 in all three pools:
`depth12`, `depth12_plus_members`, `depth12_plus_committee`).
Cohort solos: `overlap6_c0` 31.10, `c1` 30.15, `c2` 31.40, committee 35.70.
`analysis/cohort_roster_integration.json`, `analysis/analyze_cohort_roster_integration.py`.

### 6.4 Subject-specialist ensemble (one model per source subject)

Nine single-subject experts fused per target subject: mean source solo 5.32, best source
9.80, nine-source ensemble 13.55 top-1 / 36.75 top-5, mean pairwise score corr 0.383.
`results/things_eeg/subject_specialist_ensemble/testselected/nine_source_ensemble.csv`,
`transfer_top1.csv`, `transfer_top5.csv`; code `evaluate_subject_specialist_ensemble.py`,
`subject_specialist_ensemble.sbatch`.

---

## 7. Routing and learned fusion

| Run | matched uniform | learned | oracle | job |
|---|---:|---:|---:|---|
| Source-only R0 router | 45.45 | 44.50 | 66.65 | `48953` |
| Pooled all-ten weight router | 47.20 | 46.80 | 67.05 | `49005` |
| Pooled linear candidate residual | 47.20 | 43.45 | 67.05 | `49006_0` |
| Pooled MLP candidate residual (25→32→32→1) | 47.20 | 47.30 | 67.05 | `49006_1` |
| Legacy MoE router, variants A/B/C | 45.35 | 43.90 / 44.20 / 44.00 | — | `legacy/scratch_claude/router_moe_results.json` |

Router inputs (frozen in `test_selected/pooled_router_config.json`): per query and member,
max, top-1/top-2 margin, top-5 and top-10 means, normalized entropy, row norm, top-1 vote
support, similarity to member consensus, two consensus-at-winner scores. Architecture
10→16→1 shared MLP plus one bias per member, softmax weights with a 50% uniform floor,
loss = smooth hard-negative top-1 margin + 0.25 retrieval CE + 0.02 KL to uniform,
300 epochs AdamW, seed 3300. Subject ID is never an input. R0 improved NLL while top-1
fell 0.95pp. The pooled router's per-subject pattern gains on subjects 6–8 and loses on
1, 2, 4, 10.

Confidence-based expert selection: max-confidence routing gives 42.20–42.45 against
45.65 raw uniform on the router pool, and 42.15–42.30 against 46.95 raw / 48.20 row-z on
the k=4 committee. Correctness AUC of query-local confidence: 0.675 (top-1 margin),
0.699 (top-1 vs top-5 gap). EEG row norm does not predict a correct top-1 (AUC 0.46–0.53
across 10 arms; within-fold CV 0.037–0.049).

Also measured and recorded in `validation/README.md`: visual-cluster reliability
estimated from the other nine subjects reached 47.20 as a global diagnostic and did not
improve the seven untouched folds when its hyperparameters were picked on folds 1/3/5;
per-query model-subset selection, top-gap routing, and agreement/centrality weighting did
not exceed the frozen 47.00 rule.

Code: `test_selected/learned_global_router.py`, `learned_pooled_router.py`,
`learned_pooled_candidate_scorer.py`, `run_router_subject_oof.py`,
`dump_router_source_scores.py`, `build_router_manifest.py`, `global_rule_search.py`,
`final_ensemble_audit.py`, `auto_promote_after_gate.py`, `candidate_bias.py`,
`score_target_gate.py`; configs `router_config.json`, `pooled_router_config.json`,
`pooled_candidate_config.json`, `router_member_pool.json`, `router_member_recipes.json`.

---

## 8. Architecture arms

### 8.1 Full-size arms, all ten folds, test-selected *(rescanned solos)*

| arm | solo |
|---|---:|
| `fast_tsconv_squeezeformer_pairwise50` (exported as `sqf28`) | 36.70 |
| `encv3_multiscale_sqf` | 34.90 |
| `sqf_bigg` | 34.10 |
| `inductive_v2/ranked_ts_bigru` | 33.70 |
| `inductive_v2/multiscale_ts_mixer` | 33.60 |
| `encv3_ranked_perceiver` | 33.05 |
| `ortho_arch/mixer_group` | 32.75 |
| `sqf_vith` | 30.70 |
| `target_transfer/multi_eva` | 30.40 |
| `ortho_arch/perceiver_group` | 29.60 |
| `encv3_multiscale_bissm` | 29.50 |
| `temporal_conformer_pairwise50` | 29.10 |
| `target_transfer/multi_vith` | 28.45 |
| `ortho_arch/convgru_group` | 26.85 |
| `convnext1d_pairwise50` | 26.00 |
| `ortho_arch/spec_group` | 24.20 |

`sqf28` and `inductive_v2_multiscale_ts_mixer` appear in §2 winners; the others do not.
Rejected architecture implementations and their launchers were deleted; the seven
maintained encoders are listed in `architectures/README.md` (`FastTSConvSqueezeformer`,
`TinySqueezeformer`, `MultiScaleTSMixer`, `TinyMultiScaleTSMixer`,
`TinyGatedChannelTransformer`, `TinyDifferentialChannelTransformer`,
`TinyGraphDiffusionNet`), implemented in `architectures/ortho_encoders.py` with
`architectures/smoke.py`.

### 8.2 Joint / concatenated visual targets

`multibackbone5_group_e100` (one linear projector onto concatenated
InternViT + EVA + ViT-H + BigG + DINO) scored 42.5 / 23.0 / 32.5 on folds 1/3/5
(mean 32.67) and entered no best fixed size-2–5 combination; its remaining seven folds
were not trained (`validation/README.md`, Gate 2). ValCon multi-target solos:
`multi_iv28_valcon` 31.40, `multi_eva_valcon` 27.60, `multi_vith_valcon` 26.60.

### 8.3 Alignment-dimension arms (same fold-specific frozen TSConv encoder, fresh projectors, 10 epochs)

Solos: fd32 32.85, fd64 33.45, fd128 35.15, fd256 34.65, fd512 34.60, fd1024 34.70.
Mean off-diagonal score correlation 0.985. Fixed all-six ensemble 35.55, member oracle
46.85, post-hoc best k=2 36.20, k=3 35.90, k=4 36.25.
`analysis/frozen_alignment_dim_ensemble.json`, `analysis/analyze_alignment_dim_ensemble.py`,
`test_selected/run_frozen_alignment_dim.sh`.

---

## 9. Compute-matched tiny-model track

Protocol and FLOP accounting in `tiny_compute_ensemble.md`. Ten tiny encoders total
132.115M forward FLOPs/sample against 137.211M for one full TSConv + one full ATM (96.3%);
with projectors 140.80M vs 142.39M (98.9%); with batch-1024 score products ~143.42M vs
~143.70M (99.8%). Parameters 1.837M vs 3.507M. Ten-epoch RTX-3080 calibration (job 50302):
full TSConv 293s (29.3 s/epoch), TinyTSConv 199s (19.9), full ATM 462s (46.2),
TinyATM 195s (19.5) — ten small runs take 2.61× the realized GPU time of the two full runs.

Completion *(rescanned)*: `tiny_tsconv_seed3300` and `tiny_atm_seed4300` have 10 folds;
seeds 3301–3304 and 4301–4304 have 4 folds each, so the predeclared Tiny-10 primary
result is not computable yet.

Tiny reference pair, 10 folds
(`results/things_eeg/tiny_compute_ensemble/testselected_internvit28/tiny_reference_pair.csv`):
TinyTSConv 29.75, TinyATM 29.50, pair 34.60 top-1 / 69.60 top-5. Full-pair reference 40.65.

Other tiny arms *(rescanned)*: `tiny_squeezeformer` 33.10 (n=10),
`tiny_multiscale_mixer` 33.45 (n=10), `tiny_graph_diffusion` 30.20 (n=10),
`tiny40_tsconv` 29.67 / `tiny40_atm` 29.67 (n=3), `ultratiny_tsconv` 26.33 /
`ultratiny_atm` 29.50 (n=3). `tiny_gated_attention` arms A–E have run directories but no
`result.csv` written.

Analyzers: `analyze_tiny_compute_ensemble.py`, `analyze_tiny_squeezeformer.py`,
`analyze_tiny_multiscale_mixer.py`, `analyze_tiny_reference_pair.py`,
`analyze_tiny40_reference_pair.py`, `analyze_ultratiny_reference_pair.py`.
Launchers: `tiny_*.sbatch`, `run_tiny_*_fold.sh`, `calibrate_tiny_compute.sbatch`.

---

## 10. Fusion-rule and TTA measurements

- Six fixed rules over the completed cohort committees: maximum spread 0.15pp (§6.1).
- Label-free fusion sweep on the reference quartet: 45.35 → 45.45.
- Freezing one rule across the k=5 pool: `raw` 42.99 vs `row_softmax4` 43.94 mean
  (−0.95), −0.70 at the max, group ordering unchanged.
- Deterministic temporal-shift averaging `[-1,0,1]`: ATM fold 1 48.0 → 47.5; tested
  ensembles −2 to −2.5pp; group ViT-H folds 1/3/5 43.0/20.5/32.5 → 42.0/19.0/32.0.
  Entered no best fixed size-3–5 ensemble.
- Un-normalized EEG side during fusion: −0.76 nested on average across k=2..5
  (+0.30 / −1.90 / −0.70 / −0.75) on the 10-arm concept-val depth pool; normalizing
  neither side gives ~22–23. Mean |eeg| is ~18–21 for TSConv arms and ~9.6–10.2 for ATM
  arms. Solo accuracy identical to 0.0000pp either way.

---

## 11. Gate history and selection-transfer measurements (`validation/README.md`)

- Stated objective of that ledger: a 2–5-member plain-cosine ensemble at ≥50.00% mean
  top-1 under one global rule. Its final three checklist items are unchecked.
- Gate 1 (`48849`), folds 1/3/5: `eva35_group_e100` 36.0/21.5/32.0;
  `vith10_group_e100` 43.0/20.5/32.5. The best fixed gate ensemble containing group ViT-H
  scored 48.83 on folds 1/3/5 vs 47.83 for the pre-gate rule; EVA entered no best gate
  combination. Decision recorded in `test_selected/gate1_promotion.json`.
- Promoted ViT-H arm on all ten subjects: `43.0 44.0 20.5 32.0 32.5 30.0 30.0 24.0 26.0 42.5`
  (33.45 mean). Its five-fold-selected five-member rule scored 48.20 on subjects 1–5 and
  45.95 over all ten; a separately selected rule reached 49.00 on subjects 1–5 and 42.90
  on subjects 6–10.
- Gate 2 (`48853`): ATM–InternViT 48.0 (fold 1), 22.5 (fold 3), 36.5 (fold 5);
  five-backbone arm rejected as in §8.2.
- Completed 50-fold e75 batch (`48896`, `48897`, `48902`, `48912`–`48919`), all-ten solos:
  `iv_vith_dino_aux025_group_e75` 36.40, `iv25_31_aux025_group_e75` 35.85,
  `iv33_group_e75` 35.65, `atm_iv_group_e75` 35.55, `atm_vith_group_e75` 32.80,
  `bigg27_group_e75` 32.65, `dino16_group_e75` 30.90. No arm exceeds `ge100` (37.05).
  Best all-ten post-hoc fixed rule after this batch: 47.40. Union oracle coverage rose
  from 66.50 (frozen five) to 75.30 with the seven new arms while fixed fusion stayed
  near 47.
- Recorded infrastructure events: Slurm tasks exiting 127 after writing results,
  unsatisfiable `afterok` chains, and CPU-only recovery exports (`48880`, `48881`,
  `48883`, `48892`, `48900`, `48901`).
- ValCon vs LOSO-subject validation, matched three architectures: `ATM/InternViT-28`
  29.15 → 33.40; `TSConv-group/InternViT-33` 30.75 → 33.30; `TSConv/BigG` 30.10 → 30.10;
  mean solo 30.00 → 32.27. Committees: two-member 37.70 → 41.40; three-member 40.20 → 43.15.
- ValCon arm solos (`results/things_eeg/honest_ensemble/`, all n=10): `atm_iv_valcon` 33.40,
  `atm31_valcon` 33.35, `iv33g_valcon` 33.30, `iv31_valcon` 32.95, `iv28_valcon` 32.60,
  `atm33_valcon` 32.40, `atm35_valcon` 31.50, `iv33_valcon` 31.45, `multi_iv28_valcon` 31.40,
  `iv35_valcon` 31.20, `sqf_iv28_valcon` 31.20, `sqf_bigg_valcon` 31.05, `iv33g_valsub` 30.75,
  `tsconv_bigg_valsub` 30.10, `tsconv_bigg_valcon` 30.10, `atm25_valcon` 29.85,
  `iv25_valcon` 29.80, `atm_iv_valsub` 29.15, `sqf_vith_valcon` 28.65,
  `multi_eva_valcon` 27.60, `multi_vith_valcon` 26.60.

---

## 12. `inductive_covariance/` — per-query covariance alignment grid

Flags: `--inductive_covariance_align`, `--inductive_covariance_alpha`,
`--inductive_covariance_shrinkage`, `--inductive_covariance_ref_trials`,
`--inductive_covariance_reference_cache` (implemented in `module/inductive_covariance.py`,
applied in `train.py` before the EEG encoder). 50 epochs, 2048 reference trials.

3×3 grid, 4 subjects (`single_tsconv_screen/grid_summary.csv`):

| α | shrinkage | mean best top-1 |
|---:|---:|---:|
| 0.2 | 0.50 | 32.375 |
| 0.2 | 0.25 | 30.375 |
| 0.2 | 0.10 | 30.250 |
| 0.4 | 0.50 | 30.250 |
| 0.4 | 0.25 | 29.375 |
| 0.6 | 0.50 | 28.500 |
| 0.4 | 0.10 | 27.125 |
| 0.6 | 0.25 | 26.750 |
| 0.6 | 0.10 | 26.250 |

Low-α / high-shrinkage rescreen, subjects 1–5
(`lowalpha_highshrink_screen/grid_summary.csv`): (0.05, 0.95) 36.4, (0.05, 0.90) 35.9,
(0.05, 0.80) 35.2, (0.10, 0.90) 35.1, (0.10, 0.95) 34.9, (0.05, 0.70) 34.7;
(0.20, 0.95) 32.75 at n=2, (0.10, 0.70) 29.75 and (0.10, 0.80) 29.25 at n=2.
Baseline pairwise-SubjectMix subjects 1–5: `50.5 43.0 27.0 29.5 29.5` (35.9 mean).
Scripts: `inductive_covariance/run_grid_shard.sh`, `summarize_grid.py`,
`grid_shard.sbatch`, `lowalpha_shard_{513,514}.sbatch`.

---

## 13. Incomplete runs

| Track | Fold status | Numbers so far |
|---|---|---|
| Tiny-10 primary ensemble | 2 arms × 10 folds, 8 arms × 4 folds | primary result not computable |
| ValCon β-fusion wave (`fusion_wave/`) | n=6 / 6 / 4 *(rescanned)* | β=0.10 pair 35.17, β=0.30 36.92, β=0.50 37.25 |
| Frozen-fusion wave (`frozen_fusion_wave/`, 6 arms) | n=2 on two arms, 0 on four | `atm→tsconv` b030 43.75, b100 43.00 |
| `heterogeneous_wave` aggregation | 10 folds present for all 7 arms; `inter_subject_summary.csv` written for only 2 | §5.2 values recomputed from per-fold CSVs |
| `tiny_gated_attention` arms A–E | run dirs present, no `result.csv` | — |
| `tiny40_*` / `ultratiny_*` reference pairs | n=3 | §9 |
| `validation/README.md` ≥50.00% objective | 3 checklist items open | highest recorded: 47.00 frozen rule, 48.20 declared-pool k=4, 49.50 post-hoc k=6 |

---

## 14. File map

| Path | Contents |
|---|---|
| `ensemble_results_report.md` | report-ready summary; its §5 rescue section predates the ValCon arms in §5.3 |
| `analysis/ensemble_results_analysis_20260823.md` | full read-only audit of the repo as of 2026-08-23: protocol taxonomy, mechanism model, negative controls |
| `analysis/ensemble_evidence_audit.py` | reproduces the audit's calculations from stored dumps |
| `analysis/target_matrix_{test,valcon,ctl}_{base,extended}_z.json` | k=1..6 sweeps behind §2 |
| `analysis/extended_roster_z_sweep.{py,json,sbatch}` | 27-arm roster sweep |
| `analysis/cohort_roster_integration.{py,json}` | §6.3 |
| `analysis/frozen_alignment_dim_ensemble.json`, `analyze_alignment_dim_ensemble.py` | §8.3 |
| `analysis/pair_complementarity_testselected.{csv,json,pdf,png}`, `pair_complementarity_scatter.py` | 990-pair study |
| `analysis/report_figures/`, `generate_report_figures.py` | figures used by the report |
| `analysis/architecture_target_ensemble_audit.sbatch` | architecture × target audit launcher |
| `validation/README.md` | ensemble-to-50 ledger: objective, gates, job IDs, promotions, rejections, checklist |
| `validation/{dump_scores.py,score_pool.py,pool.json,gate_pool.json}`, `run_concept_val.sh`, `run_group_target_{val,refit}.sh`, `valcon_grid_famA.sbatch`, `val_selected_waves.sbatch` | ValCon / LOSO-val runners and scorers |
| `test_selected/` | training arms, score dumping, fixed-rule searches, router code and configs, gate promotion records |
| `decorrelated_models/` | `train_twins.py`, `losses.py`, `stochastic_views.py`, `smoke.py`, the λ / β / γ / stochastic wave launchers, `decorrelated_models.md`, `stochastic_full_data.md` |
| `architectures/` | `ortho_encoders.py`, `smoke.py`, `run_fast_tsconv_squeezeformer.sh`, `architecture_target_matrix.sbatch`, `README.md` |
| `synthetic_subjects/` | seed / repetition / shrunk-model / meta-subject utilities, `best_small_ensembles.md`, `score_npz_ensemble.py`, `evaluate_repetition_ensemble.py` |
| `balanced_subject_bagging.{py,md,sbatch}`, `analyze_balanced_subject_bagging.py`, `run_balanced_subject_bag*.sh` | §6.2 |
| `subject_cohort_bagging.sbatch`, `subject_overlap6_bagging.sbatch`, `subject_soft_weight_bagging.sbatch`, `analyze_subject_cohort_bagging.py`, `run_subject_cohort_fold.sh` | §6.1 |
| `subject_specialist_ensemble.sbatch`, `evaluate_subject_specialist_ensemble.py` | §6.4 |
| `inductive_covariance/` | §12 |
| `retrieval_fusion.py` | shared fusion rules and ensemble diagnostics |
| `presentation/ensemble_results_presentation.tex` | Beamer source for the ensemble-results deck |
| `legacy/scratch_claude/` | superseded searches: `ens_search.py`, `ens_search45.py`, `ens_diverse.py`, `router_moe.py`, `ens45_results.json`, `router_moe_results.json`, `manifest.json`, `solo.json` |
| `results/things_eeg/synthetic_subjects/ensemble_screen/dumps/` | 66-arm score dump pool |
| `results/things_eeg/ensemble50_testselected/` | test-selected arms, `rule_final_ledger16.json` |
| `results/things_eeg/honest_ensemble/` | ValCon and LOSO-val arms |
| `results/things_eeg/{decorrelated_models,subject_cohort_bagging,subject_specialist_ensemble,tiny_*,ensemble_arch_candidates,ensemble_alignment_dim_frozen,ortho_arch,inductive_covariance}/` | per-track run outputs |

The former top-level path `new_architectures_for_ensemble` is a compatibility symlink to
`ensemble_experiments`; `ensemble50_experiments` and `honest_ensemble_experiments` links
were removed.

Related agent-memory entries: `cross-backbone-ensembles`, `ensemble-16-arm-pool`,
`eeg-embeddings-unnormalized`, `current-best-model`, `arm-names-encode-channels-not-layers`,
`group-vs-pairwise-subjectmix-is-a-wash`, `results-tree-scan-before-claiming-absent`.
