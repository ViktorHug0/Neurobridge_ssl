# Cleanup plan, draft 2026-11-02 (nothing deleted yet)

Scope you approved:
- Q1 (b): the challenge is reduced to the best ensemble, plus the abandoned lines across the whole repo; anything a paper still cites stays.
- Q2 (a): a snapshot commit and tag go first.
- Q3 (b): failed-run artefacts are deleted; every `*_LOG.md` stays.
- Q4 (b): dead branches inside kept files are stripped.
- Q5 (b): anything that reproduces a number in a paper or submission stays.

"Best ensemble" means **optx5** (cv4 44.25), made of five recipes:
- `reve_d6_optx_things`
- `rmix_reg_optx_things`
- `sqf_dbn_optx_things`
- `tsc_dbn40_bn64_optx_things`
- `reve_scratch_optx_things`

It is scored with the lean TTA. The five base recipes (`*_things` without `optx`) stay as its comparison baseline.

Round-2 answers: Q1 (b) commit text only and delete `logs/`. Q2 (a) keep the 3 submission zips and the build scripts. Q3 (a) unprotect and delete the intra/sparse/ensemble-head results, keep the 3 LOSO drivers. Q4 keep the notes. Q5 strip dead code everywhere, protected files included: a flag is dead when no kept code passes a non-default value; default-valued mentions are removed from every script.

---

## 0. Safety net (step 1, before any deletion)

- Commit the current tree on `transductive_experiments` and tag it `pre-cleanup-2026-11-02`.
- About 7 GB is untracked: `submissions/` weights/zips about 4 GB, `logs/` 2.5 GB. **[Q1]** decides what goes into that commit.
- External worktrees (`/nasbrain/p20fores/apr_code_*`, `SATTC-CVPR2026`, `MindEyeV2`) are **not touched**.

## 1. KEEP: closure of the best ensemble

**Shell:**
- `neurips_challenge/cv4.sh`
- `alljoined_run.sh`
- `overnight/align_dyn.sh`
- `overnight/aug_queue.txt`, trimmed to the 10 recipes

**Scoring and TTA:** `score_track1.py`, `tta_cache.py`, `tta_greedy.py`, `aug_ens.py`, `add6.py`, `tta_lab.py`, `grader_sim.py`, `group_purity.py`, `group_features.py`, `cv4_ensemble.py`, `cv4_summary.py`, `official_metrics.py`, `neighbour_lda.py`, `stage_eeg_cache.py`.
- `tta_lab.py`, `grader_sim.py` and `score_track1.py` get stripped (section 4).

**Data builders:** `build_alljoined_cache.py`, `build_120hz_cache.py`, `validate_120hz_cache.py`, `export_things_to_alljoined.py`, `build_dinov2_targets.py`.

**`trial_routing.py` and `trial_features.py`:** both go once `FOLDS` (a constant) is inlined into `grader_sim.py` and `neighbour_lda.py`. That also removes an import-time side effect that loads a 4 GB npz.

**Docs:** `README.md`, `CV4_PROTOCOL.md`, and all `*_LOG.md`.

**Shared:**
- `train.py`, everything in `module/` that train.py imports, `module/eeg_encoder/test_domain_conv.py` (the self-check for DomainBatchNorm)
- `ensemble_experiments/__init__.py`, `ensemble_experiments/architectures/{__init__,ortho_encoders,hybrid_encoders,alljoined_encoders}.py`

## 2. KEEP: backs a number in the CVPR paper or the NeurIPS paper/rebuttal

All of these stay:
- `CVPR_paper/` (all of it, including `scripts/`)
- `Neurips_paper_and_rebuttal/`, `neurips_author_response_period/`, `next_paper/` (active), `author-kit-CVPR2027-v1-latex/`
- `module/transductive/` (CVPR Tab. 2), `module/util.py`, `module/eeg_encoder/foundation.py` (LaBraM/CBraMod, CVPR + rebuttal)
- `module/image_augmentation.py` (used by extract_feature)
- `evaluate.py`, `compute_avg_results.py`, `extract_feature.py`, `extract_internvit_layers.py`, `preprocess_meg.py`, `preprocess_eeg.py`
- `ensemble_experiments/validation/` (`run_concept_val.sh` and `val_selected_waves.sbatch` trained CVPR seed 3300)
- `scripts/alljoined/`, `scripts/things_meg/`, `scripts/nsd_sage/`, `scripts/things_eeg/{tta_rebuttal,neurips_grid,patches}/`
- `scripts/things_eeg` top level:
  - all `inter-subject-SAATC*.sh`
  - `inter-subject-tsconv-*`, `inter-subject-foundation.sh`, `inter-subject-alljoined.sh`
  - `aggregate_loso_val_results.py`, `reproduce_ablation_with_mixup.sh`
  - `tta_sattc_oat_sweep.sh`, `tta_alljoined_oat_sweep.sh`, `apply-tta-to-sweep.sh`
  - `encoder_grid_ablation.*`, `progressive_sattc_candidate_sweep.py`, `run_paper_progressive_plot.sh`
  - `plot_progressive_sattc_results.py`, `run_sattc_sensitivity_sweep.py`, `plot_sattc_sensitivity.py`
  - `holdout_probe_projector_mixup.sh`, `compare_probe_mixup_nomix.py`, `plot_holdout_probe_comparison.py`
  - the SubjectMix control launchers (`run_subjectmix_controls.sh`, `run_smooth_paperbase.sh`, `run_smooth_kernel_sweep.sh`, `run_mixcontrols_paperbase.sh`, `subjectmix_rebuttal.sh`, `mixcontrols*.sbatch`, `fixedmean.sbatch`)
  - `extract_internvit_layers.sbatch`, `download_eeg.sh`
  - the `foundation_*.log` files (git-tracked)

## 3. DELETE: whole files

### 3a. `neurips_challenge/` top level (75 files, ~9.5k lines; all failed lines or superseded tools)

- **Routing / LDA / fingerprint era (superseded by the lean TTA):** route_final, route_study, routing_pool, neighbour_centring, centring, centring_pca, centring_pca2, centring_stream, static_sage, static_whiten, feature_coral, feature_transform, hyperalign_maps, calib_map, filter_maps, ens_transform, postproc_probe, poe_sim, pc1_diag, mixed_batch, mixed_sweep, n11.py, s11.sh, ens11.py, warmup_transform_check, tta_arm, tta_backbone, tta_ens, tta_ssl, adabn_compare, trial_norm, trial_cache, train_split_cache, occlusion_probe, lag_probe, grand_erp, diag_input_scale, official_order_probe, verify_grader.
- **Subject affinity / scaling / identity:** affinity.sh, affinity_subsets.txt, affinity_labelfree, affinity_summary, affinity_trial_cache, nsubject_scaling, scaling.sh, scaling_summary, subject_embed, subject_geometry, subject_separability, eeg1_identity_cache.
- **KD / boosting / warm start / soups (all dead):** export_teacher, export_teacher_cv4, boost_weights, warmstart_ckpt, soup, ens_cache, ens_search, cache_track1_scores, pretrain_denoise.
- **Aug / alignment rounds:** aug_cmp, aug_status, aug_swapscan, build_aug_targets, build_alljoined_neighbors, build_dinov2_multilayer, align_curves, align_running, progress, collect_results, launch_sweep.sh, sweep_cell.sh.
- **THINGS-EEG1 / Alljoined-1 lines (no lever helped):** export_alljoined1, export_things_eeg1_to_alljoined, zeroshot_alljoined1. Also root `preprocess_eeg1.py` and `scripts/things_eeg/{preprocess_eeg1.sbatch, verify_eeg1_export.py, eeg1_intra_report.py}`.
- **September warm-up launchers:** run_recipe.sh, run_reve.sh, run_reve_repro.sh, run_track1_tsconv.sh. **[Q2]** decides `ensemble_track1.py` and `submissions/`.
- **Docs:** TTA_RESEARCH.md, AUG_LITERATURE.md and challenge_innovations.md are notes, not logs. **[Q4]**

### 3b. `neurips_challenge/overnight/` (all but 2 files: ~70 scripts, the 51 MB `logs/`, `bank_train_dinov2.npy`)

Everything except `align_dyn.sh` and `aug_queue.txt`. That means all chain/tail/rescue/reve_*/sqf_*/tsc_*/lpft/kdx/distill/warm_kd/ft_oracle*/ens_cache_job/tta_*/s11/scaling/eeg1_*/arch_* scripts and the probe `.py` files.

### 3c. `ensemble_experiments/` (abandoned ensemble line, ~190 code files, ~20k lines)

Delete everything except `__init__.py`, `architectures/` (trimmed) and `validation/`:
- **Fusion:** fusion_c012*, fusion_private_stems*, fusion_three/, FUSION_C012_LAUNCH.md
- **F1 replay:** f1_replay_snapshot/, f1_checkpoint_replay*, F1_*.md
- **Tiny line:** tiny_*, ultratiny_*, run_tiny*_fold.sh, analyze_tiny*/ultratiny*
- **Bagging:** balanced_subject_bagging*, subject_cohort/overlap6/soft_weight/specialist bagging, run_balanced*/run_subject_cohort*
- **Single-file experiments:** full_sharing* + full_sharing_v2/, spherical_fusion*, nash_*, iv28_control*, tsconv_bb128_valcon*, retrieval_fusion.py, evaluate_subject_specialist_ensemble.py
- **Folders:** analysis/, test_selected/, synthetic_subjects/, compact_followup/, compact_valcon/, decorrelated_models/, tiny_sharing/, mechanism_study/, inductive_covariance/, legacy/, presentation/, recaps/, ENSEMBLE_RECAP.md
- **[Q3]** decides the leftover `validation/` files that back no number (pool.json, gate_pool.json, score_pool, run_group_target_*).
- **In `architectures/`:** delete `braindecode_encoders.py`, `recurrent_encoders.py` and `own_encoders.py`, and drop their registry imports in ortho_encoders.py:998-1010.

### 3d. `scripts/things_eeg/` (abandoned, ~75 files, ~11k lines)

- **Folders:** sparse_clip/, identity/ (its xadv flag no longer even exists), transductive_methods/ (superseded by CVPR r1_transductive.py)
- **April ablation sweeps:** projector_size_sweep*.sh, projector_only_sweep_backbone512.sh, multipos_loss_sweep.sh, inter-subject-train-saw-sweep.sh, inter-subject-mixup*.sh, ultimate-ablation.sh, plot_featdim_sweep.py, plot_dropout_mixup_barplot.py, plot_random_eeg_grid.py, tta_diagnostics.py, barplot.py (its figure is not in any tex)
- **Transfer family:** session_split_transfer_experiment.py, session-test-split-transfer-sweep.sh, test-split-transfer-sweep.sh, transfer_calibration_experiment.py, disjoint_tta_transfer_curve.py, run_disjoint_tta_transfer_curve.sh, inductive_tta_transfer_suite.py, run_inductive_tta_transfer_suite.sh
- **Intra line:** intra-subjects.sh, run_intra_subject_tta_sweep.py, intra_insights*, intra_pooled*, loso_insights*, pooled_select.py, intra_ensemble.py. **[Q3]** decides the protected intra results.
- **Misc:** meta_subject_report.py, run_subject_scaling.sh, subject_scaling.sbatch, subject_scaling_report.py, run_samga_ref.sbatch, run_samga_nosmooth.sh, run_geometry_encoder.sh, run_phase0_augfeats.sh, resswap.sbatch, semantic_*_screen.sbatch, strong.sbatch, inter-subjects-xavg-512.sh, watch_xavg.sh, image_feature_extract.sh
- **[Q3]** decides the reference drivers `inter-subjects.sh`, `inter-subjects-all.sh` and `inter-subject-one-train.sh`.

### 3e. Reconstruction / captions / misc (abandoned, no paper number)

- `reconstruction_experiments/` (all, 33 files)
- `module/lowlevel.py`, `module/reconstruct.py`, `module/recon_metrics.py`
- `module/stable_adamw.py` (only failed cv4 recipes use it)
- `train_lowlevel.py`, `precompute_vae_latents.py`, `fuse_feature.py`
- `analysis/` (root: 2 draw_* scripts + 2 PNGs)
- `graphs/`, `output/` (770 MB of old logs + recon outputs), `slides/` (TransCLIP beamer). **[Q4]** decides `papers/` (md reading notes).
- `new_architectures_for_ensemble` (symlink to `ensemble_experiments/architectures`, unused)

## 4. Strip inside kept files

Rule: a flag is dead when no kept code (optx5 + section-2 launchers + CVPR scripts) passes it a **non-default** value. Dead flags and their code go, and every script that passes them at the default value is rewritten without them, protected scripts included. Helpers/classes go when nothing kept references them.

- **`neurips_challenge/cv4.sh`:**
  - Keep the 10 recipe lines and the `*_things` block. That leaves about 60 of 528 lines.
  - Delete the other ~326 case lines, the `ens_cache_job`/`ft_oracle` dispatch (l.14), and the glob cases 399-407 (xIV28w, wsIV28, _sd, _t[A-Z], bd_).
  - Also delete the post-recipe blocks 433-512 (kd, kdx, bst, e1*, aj1, nc, pt_e1, reve_ft guard).
- **`tta_cache.py`:** remove `TRACK1`/`bridge()`/the `_raw.npy` path and the teacher/mc8/backbone-dump branches.
- **`tta_lab.py`:** keep `load`, `mn`, `lw_shrink`, `wccn_affinity`, `BS` and `DEV`; drop the ~490 other lines (METHODS, router/oracle/knn/spectral/bayes/gp/pca, `main`). Alternatively fold the 6 kept functions into `tta_greedy.py` and delete `tta_lab.py`.
- **`grader_sim.py`:** drop `main` (l.52-131).
- **`score_track1.py`:**
  - drop `batch_transform`/`batch_transform_fp`/`identity_features`/`RICH`/`FP_AB` once the two importers above are trimmed;
  - keep `batch_transform_wknn` + `fit_fingerprint_wccn` + `lw_shrink` (the deploy path) and `self_test`;
  - revert my `build_modules` aux-target fix if `--auxiliary_*` goes from train.py.
- **`train.py`:**
  - remove the challenge-only dead flags and their code: `--auxiliary_fixed_targets`, `--init_checkpoint`, `--boost_*`, `--distill_*`, `--anticorr*`, op_transfer variants (bands, cross_dataset, rotate, extrapolate), `--optimizer stableadamw`, hyperalign / cross-dataset translation, global negatives, siglip, mask_concept_neg, loss_trunc, repeat_mean, mmd, b2b, relational, rsa, iso, matryoshka, confidence head, neighbour heads, LoRA, spatial_mix_bank, `_things_coral_maps`, SharedSpaceProjector;
  - keep everything a kept paper launcher still passes (subject_mixup incl. `group`, sattc/eval_mode, data_average, grouped sampler, multi_positive, ATM). The grep decides each one.
- **`module/eeg_encoder/model.py`:** drop classes referenced by nothing kept: EEGNet, EEGProject (check the paper encoder grid first), Conformer, EEGTransformer, TSConv30, LatencyAligner, ExtraInputAdapter, plus the TSConv_parameterizable router/power/decode branches if unreferenced.
  - `TSConv` stays only if a paper encoder grid uses it (lowlevel.py goes).
- **`module/loss.py`, `module/projector.py`, `module/dataset.py`:** drop only what the grep shows unreferenced (multi_positive and data_average are paper-used, so they likely stay).
- **`ensemble_experiments/architectures/`:** drop the unused encoders (FactorizedSTFormer, mixers, GraphDiffusion, FBTN, …; ortho_encoders l.259-984; hybrid_encoders all but TokenREVE; alljoined_encoders all but `_SwitchBlock`).
- **Check after stripping:** rebuild the TTA cache for optx5 fold 1 from the checkpoints, then confirm the lean-TTA score is unchanged (47.90 f1; 44.25 over 4 folds), plus a 1-epoch smoke run of each of the 5 recipes and `evaluate.py` on one CVPR run.

## 5. DELETE: artefacts (about 270 GB)

- **`results/alljoined/track1/cv4/`** (230 GB). Keep the 10 recipe dirs (~13 GB) and their `tta_cache/` files (~31 GB). Delete everything else, about 186 GB:
  - 327 failed-recipe dirs
  - `teachers/` 38 GB
  - `warmstart/`
  - `ensemble_cache/`, `enscache_*`
  - the 67 GB of `tta_cache/` belonging to failed members
  - the `*_optx_sd7` second seeds (44.35 = noise)
- **`results/alljoined/track1/{affinity, scaling, s11, heldout, drop3, pooled, routing, debug}`** (~48 GB). `results/alljoined/inter-subjects` (NeurIPS appendix) stays.
- **`logs/`** (2.5 GB, 2,099 untracked slurm outputs) **[Q1]**.
- **`results/things_eeg/neurips_track1/`** (31 GB, September warm-up) and **`neurips_challenge/submissions/`** (3.9 GB) **[Q2]**.

## 6. Docs to update after the delete

- `AGENTS.md` and `neurips_challenge/README.md`: point at the optx5 path only.
- `PROTECTED.md`: it is stale (says tta_rebuttal has 27 files, now 31; protects intra, sparse and ensemble "live head" results that no paper uses). **[Q3]**
