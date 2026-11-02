# CVPR 2027 — decisions and experiment checklist

**Deadlines** (official CFP, AoE): enrollment **2026-11-10**, paper **2026-11-16**, supplementary **2026-11-23**. Venue decided: **CVPR 2027**.

**Compute:**
- 2 rtx3080 sbatch GPU workers for this project, on top of the GPU used by another agent.
- GPU work goes through `CVPR_paper/queue/gpu_queue.txt`, consumed by `CVPR_paper/scripts/gpu_worker.sh`.
- CPU work runs as `srun --jobid=<interactive> --overlap` steps.
- Record every run in `EXPERIMENT_LOG.md`.

---

## Decisions

### 2026-10-21 update (supersedes the D3/D4 wording below)

- **D13 — Venue and name.** Venue is CVPR 2027. The method is called **SAGE** (drop "-TTA").
- **D14 — SubjectMix.** Do not push it in this paper; it may become a separate workshop paper. It stays only as part of the base training recipe.
- **D15 — Main-text protocol = P-main.**
  - Checkpoints are selected on the **test subject** (lowest test loss), as in SATTC, SCORE, SVTL and every prior cross-subject paper. This keeps the main tables comparable.
  - Calibration hyperparameters are **never** chosen on the target subject. They are always selected cross-fold (Q5a): for fold *s*, pick the config maximising the mean accuracy over the other 9 folds.
  - Competitors get the same treatment.
- **D16 — P-clean comes later.** ValCon / LOSO-val checkpoints, Alljoined and MEG come near the end. They back the pitfalls section and the supplementary, not the main tables.
- **D17 — Streaming adds a prequential score.** Each arriving trial is decoded at arrival with the current calibration, which is then re-fitted including it. This is reported next to the buffer-transductive score.
- **D18 — Immediate order of work:**
  1. a solid reproduced set of R1 (200×200) numbers, beating 77%;
  2. intra-subject base and SAGE results (competitors report them);
  3. our best models in the Phase 4 regimes (beyond the 1-to-1 mapping);
  4. clone competitors and evaluate them in their setting and in ours;
  5. then P-clean (ValCon), Alljoined, MEG.

### 2026-10-20

- **D1 — Focus.** The paper is about label-free transductive subject-and-session calibration. The paradigm comes first, the method second. Inductive results are context.
- **D2 — Framework is transductive.**
  - The gallery is fixed and known (same assumption as zero-shot retrieval).
  - In streaming, queries arrive one at a time and need not cover the gallery. The calibration is re-fitted on the growing unlabeled buffer.
  - This is the comparable setting for SATTC, SCORE and SVTL. Frozen carry-over (Exp A) is a secondary result.
- **D3 — No peeking on calibration hyperparameters**, for every method. Checkpoint selection follows D15/D16.
- **D4 — Rerun everything** through one code path. No NeurIPS number is reused.
- **D5 — No lucky single runs as headlines.** `beta_k9` (36.70) and `group_e100` (37.0) did not replicate. Headline only multi-seed means.
- **D6 — Stronger base encoders for R1.** Target ≥ 80% after SAGE. Candidates: ATM (incl. the 128-d variant), rmix, Squeezeformer, REVE-from-scratch.
- **D7 — Reproduce competitors twice:** in their own conditions (checked against their reported numbers), then under our protocol.
- **D8 — Framing.**
  - We formalize a setting BCI uses informally and EEG-to-image benchmarks ignore.
  - The gallery is a second point cloud, so alignment is cross-modal to the known stimuli.
  - We name the hidden assumptions A1–A4.
  - Test-peeking selection is flagged as a community problem (already partly addressed in the NeurIPS paper with LOSO-val).
- **D9 — No strong plateau claim.** Argue complementarity: the calibration gain grows with base quality.
- **D10 — Hungarian pitfall dropped.** It is unverifiable; on the main model Hungarian scores 51.4 vs 68.45 for SAGE. It stays as a baseline only.
- **D11 — Limitations.** Dataset-level ones: RSVP epoching ambiguity, averaged repetitions, one exemplar per concept, 10 subjects.
- **D12 — Scope.** Intra-subject is evaluated too; cross-subject stays the main case.
- **D19 — "Prior only" baseline and framing (user, 2026-10-20).** Every R1 table carries a row for plain cosine + Hungarian decode, with no adaptation (64.5 on ATM-L31, about SCORE 65.2). The text says the correspondence prior alone gives SCORE-level accuracy and SAGE adds +17.8 on top. This goes in the intro contributions, the R1 section and the pitfalls (P8).

## Open questions

- **Q2 — Double-blind (user: will do).** Make `github.com/ViktorHug0/Neurobridge_ssl` private or anonymize it before submission.
- ~~Q7 — SATTC memory~~ → user approved one rtx3090 node for the SATTC reproduction (2026-10-20).

## Protocols

**P-main (main text, D15).**
- `--select_best_on test`, `--save_weights`.
- Calibration hyperparameters by cross-fold selection: the `loso_fold_cv` column of `run_honest_tta_selection.py`, on a wide grid.
- Report the `paper_default` and `oracle_test` columns only as references, never as headlines.

**P-clean (supplementary / pitfalls, D16).**
- ValCon checkpoint selection: `--val_concept_ratio 0.1 --val_concept_seed 20260822 --select_best_on val`. Add `--save_testsel_checkpoint` to also get the P-main checkpoint from the same run. ValCon runs drop 10% of training concepts.
- Calibration hyperparameters cross-fold, as in P-main.

**Both protocols.**
- At least 3 training seeds; mean ± std; per-subject tables; paired Wilcoxon tests.
- Harm rate: fraction of subjects below R0, for every regime and budget.
- One TTA code path (P0.2).

---

## Phase 0 — Infrastructure (CPU, do first)

- [x] **P0.1** Download the CVPR 2027 author kit into `CVPR_paper/` and check the page limit (historically 8 pages + references). DONE 2026-10-21: cvpr.sty + ieeenat_fullname.bst from github.com/cvpr-org/author-kit (CVPR 2026 kit, latest); limit = 8 pages excluding references.
- [ ] **P0.2** Unify the TTA code path (P-clean item 5).
  - Regression target: `honest_tta_selection_merged` `paper_default` per-subject numbers on cached embeddings.
  - Then explain the 68.8 (May) vs 68.3 (July) drift.
- [ ] **P0.3** Embedding cache.
  - For every (encoder, seed, fold), store per-repetition test embeddings, gallery embeddings and ValCon/val embeddings in one `npz` format, so that every regime runs on CPU.
  - Reuse the format of `synthetic_subjects/ensemble_screen/dumps/` if it fits.
- [ ] **P0.4** Cross-fold hyperparameter selection utility (Q5a), generic over methods, so competitors get the same treatment.
- [ ] **P0.5** Docs fixed 2026-10-20 (`AGENTS.md`, `PROTECTED.md`): the main NeurIPS checkpoints were deleted, not absent.
- [x] **P0.6** Protect the new clean checkpoints in `PROTECTED.md` as soon as they exist. DONE 2026-10-21.

## Phase 1 — Base encoders (GPU, P-main)

**Launch infrastructure.**
- Arms are defined in `CVPR_paper/scripts/loso_fold.sh` (one LOSO fold per call). Outputs go to `results/things_eeg/cvpr/loso/<arm>/seed<S>/`.
- Two `gpu_worker.sh` sbatch jobs consume the queue.

**Screen rule.**
- Run folds 1, 3 and 8 (strong / weak / weakest subject) at seed 3300 first.
- Kill any arm that is clearly behind at matched epochs.
- Survivors get 10 folds × 3 seeds.

**Arms.**
- [x] **E1.1 TSConv mainline** [DONE 2 seeds (3300, 3301; 3302 ditched 2026-10-21): plain 35.53 +- 0.25 -> SAGE 67.30 +- 0.57]
  - Recipe of the NeurIPS model: k30/pool51/do0.5, fd512/bb1024, pairwise SubjectMix α=0.5, InternViT L28, 50 epochs.
  - Queued: seed 3300 on all folds, then 3301 and 3302.
  - Also serves as the reproduction check for current `train.py` (target ≈ 35.9 plain).
- [x] **E1.2 ATM, NeurIPS shape** [s3300 only, by design (same-encoder dimension test); 10 folds DONE: plain 30.05 -> SAGE 78.70 cross-fold; SCORE 55.55; log 06:35] (fd512/bb1024, 50 epochs; the arm behind the 77.2): queued, seed 3300 on all folds, then 3301 and 3302.
- [x] **E1.2b ATM, 128-d** [MAIN MODEL; DONE 2 seeds: plain 35.70 +- 0.71 -> SAGE 83.00 +- 0.99; 100 epochs (vs the 50-epoch base recipe) KEPT by user decision 2026-10-21; report it in the paper] (`atm128`: bb128/fd128, 100 epochs, the `atm_iv_valcon` shape)
  - Promoted to the front of the queue after the E2.0 screen.
  - The ValCon-trained `atm_iv_valcon` test-selected checkpoint reaches **plain 36.35 → SAGE 80.75** (cross-fold selection).
  - Queued: seed 3300 on folds 1/3/8, then the rest. Add seeds 3301 and 3302 once the screen confirms.
- [x] **E1.3 rmix** [screen DONE: 74.0 oracle fold 1, dropped (kill-losing-screens)] (`TokenREVE`, `tokens=mixed`, `patch=50` = 0.2 s at 250 Hz, 5.9M params)
  - Builds on 63 ch × 250 samples (smoke-tested 2026-10-20).
  - Recipe: THINGS mainline plus one-cycle lr 1e-3, wd 0.05, clip 1.0, EMA 0.999 with warmup, bf16.
  - Screen queued on folds 1/3/8.
- [x] **E1.4 Squeezeformer** [screen DONE: 73/64.5 on folds 1/3, dropped] (`OrthoFastTSConvSqueezeformer`, 75 epochs, `sqf_iv28_valcon` recipe, 2.4M params): screen queued on folds 1/3/8.
- [x] **E1.5 REVE-from-scratch** [screen DONE: 41.5/58, dropped] (depth 6, 19.6M params, `trial_zscore`, default 63 THINGS channel names, `fm_resample` default)
  - Batch 504 to fit 10 GB, plus the token recipe.
  - Screen queued on folds 1/3/8. If it OOMs, lower the batch.
- [~] **E1.6 Grid encoders** [QUEUED 2026-10-21 15:00: EEGNet/EEGProject/EEGConformer, s3300, folds 1/3/8 + cross-fold SAGE screen; LaBraM skipped (no LOSO recipe)] (EEGConformer, EEGNet, EEGProject, LaBraM full-FT) at 1 seed, for the base-quality analysis.
- [x] **E1.7 Image targets.** DONE 2026-10-21: atm128 L31 vs L28 on folds 1/3/8, SAGE cross-fold 89.0 vs 89.0 (per subject 91/97/79 vs 91.5/94.5/81), plain 35.2 vs 35.0 -> keep L28. Keep InternViT-6B L28. Re-check L25/L31 only for the winning encoder; existing ValCon arms `atm25/31/33/35_valcon` are in the E2.0 wide screen.
- [ ] **E1.8 ValCon vs LOSO-val vs test selection** (P-clean, later, D16).
  - Prior single-seed evidence: ValCon beats LOSO-val (ATM 29.15 → 33.40, TSConv-group 30.75 → 33.30, TSConv/bigG unchanged at 30.10).

## Phase 2 — Regime R1 (200×200, bijective) with SAGE (CPU)

- [x] **E2.0 Screen of existing test-selected checkpoints** (`honest_ensemble/*_valcon` keep `checkpoint_testsel_best.pth`)
  - Scripts: `CVPR_paper/scripts/tta_screen_existing{,_wide}.sh`; plain baselines from `CVPR_paper/scripts/testsel_plain.py`.
  - Selection: `loso_fold_cv` (cross-fold).
  - Done: `atm_iv_valcon` 36.35 → 80.75. The shrink parameter picked the 0.85 grid edge, so a wide grid (0.5–0.97) is chained for ATM L25/28/31/33/35, TSConv L28 and Squeezeformer.

- [x] **E2.1 Calibration hyperparameter grid** [DONE: wide grid, cross-fold, per encoder and seed (tta_screen_loso)] per encoder, using cross-fold selection (Q5a).
  - Include the rotation shrinkage α ∈ {0.4 … 1.0}; α<1 has never been tested on the batch regime.
  - Make the grid wider than the old test-centred one.
- [x] **E2.2 Main table.** [FINAL 2 seeds: results/things_eeg/cvpr/final/main_table.md; s3300 detail: plain 35.20, CSLS 47.5, SATTC 45.45, SVTL 63.9, SCORE 65.6, prior-only 66.2, SAGE 82.30, 10/10 vs SCORE p=0.002, harm 0/10; tsconv s3300 battery running; seeds 3301/3302 pending] Encoders × {R0, SAGE-TTA} for 3 seeds, mean ± std, per-subject results, harm rate, Wilcoxon tests.
- [ ] **E2.3 Ensemble?** Out of scope by default (D9). Revisit only if a reviewer-style question demands it.
- [x] **E2.4 Repetition (SNR) sweep** DONE (atm31, atm_iv): SAGE-SCORE -2 at q=5, 0 at q=10, +5 at 20, +17 at 80; see log. (`r1_ablate.py --reps_sweep`, running on atm31/atm_iv ValCon): q ∈ {1, 5, 10, 20, 40, 60, 80} on the best two encoders, giving R0 and TTA accuracy as a function of q.
- [x] **E2.5 Component ablation** (DONE atm31_valcon, see log; repeat on final P-main model), with correct labels (`r1_ablate.py`, running; sub-01 smoke in log): [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]
  - −whitening, −CSLS, −Sinkhorn (Sinkhorn replaced by plain scores but Procrustes kept), −Procrustes, and the old "structural" removal of both.
  - Fix the CSLS row so it does not also change k.
- [x] **E2.6 Correspondence estimators** (DONE atm31_valcon: Sinkhorn 82.3 > softmax 71.2 > top5 70.6 > Hungarian 63.3 > MNN 60.2 > argmax 56.4; plain+Hungarian DECODE 64.5 = mandatory baseline) (`r1_ablate.py` plan_* and +hung rows; NOTE plain+Hungarian decode = 76.5 on sub-01, must be a baseline row): Sinkhorn / softmax / top-k / argmax / Hungarian, and SAGE→Hungarian. Old suite: `sattc_hungarian` 66.95 < `full_sattc` 68.45. [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]

## Phase 3 — Competitors, own conditions then ours (CPU, plus GPU only where needed)

- [x] **E3.1 SATTC** [final atm128: 45.45 vs plain 35.20; OWN CONDITIONS DONE: 11.39 top-1 / 28.11 top-5 over their 9 test subjects (paper 14.8; rebuilt ViT-H features)] — UNDER OUR PROTOCOL DONE on atm31_valcon: 45.35 top-1 (plain 36.05, SCORE 65.20, SAGE 82.30), wrapper validated exactly against their logged sub-01/02; own-conditions run still training (job 63601). Next: run the wrapper on the final P-main model and on R2-R4. (official code: https://github.com/QunjieHuang/SATTC-CVPR2026). Own-conditions run is job 63601 on an rtx3090: memory-only loader patch, `--data_path` passed explicitly.
  - **Under our protocol:** do not hand-port the calibration. It is spread over the 1,200-line `evaluate_model` (SAW → Ada-CSLS → structural expert → PoE, with many config defaults).
  - Instead, call their own `evaluate_model` with a pass-through model and a dataloader that yields our embeddings.
  - Validate the wrapper first: run it on embeddings from their saved `sub-XX_final.pth` and require that it reproduces their logged per-subject numbers. Only then apply it to our encoders and to R2–R4.
  - Reproduce their table (ATM base 9.2 → 14.8 top-1) in their conditions, then on our encoders under P-clean, R1–R4.
- [x] **E3.2 SCORE** [DONE on all final models] (arXiv 2608.19134, no code found). Deployment step reimplemented in `CVPR_paper/scripts/label_free.py`; R1 run chained (`r1_competitors.sh`).
  - Reimplement the deployment step: moment matching → CSLS (k=10) mutual-NN landmarks (12–160) with margin weights → identity-regularised orthogonal Procrustes → CSLS retrieval.
  - Their training-time "simulated unseen-subject episodes" are a training method. Decide whether to reimplement them (cost) or apply only the deployment step to our encoders, and state that choice.
  - Their own conditions: SAMGA base 26.22 → 53.23. If SAMGA cannot be reproduced, reproduce the *delta* on a base of similar accuracy and say so.
- [x] **E3.3 SVTL refinement** [DONE on all final models] (arXiv 2609.36971, contemporaneous). Repository is a placeholder; reimplemented as SCORE with 64 landmarks (`label_free.py`). Same recipe family as SCORE: moment matching + CSLS mutual-NN landmarks + regularised orthogonal alignment. Reimplement it; their reported result is 35.3 → 48.1.
- [x] **E3.4 Transductive VLM baselines** DONE atm31 2026-10-21: TransCLIP 55.1, EM-Dir 58.5, hard k-means 57.5, Hungarian 64.5, Sinkhorn 66.5 vs SAGE 82.3 (cross-fold configs); final model queued. (`r1_transductive.py`, cross-fold config selection, smoke running): EM-Dirichlet, TransCLIP, plain Sinkhorn, CSLS-only, Hungarian (`module/transductive.py`, `scripts/things_eeg/transductive_methods/`). Rerun on clean embeddings. [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]
- [x] **E3.5 BCI-style baselines:** DONE atm31: AdaBN +1.9 plain (37.9), TENT +0, re-centering destructive; SAGE +46. Full EA retrain not run. [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]
  - Euclidean Alignment on the raw target EEG (label-free, input space, needs a forward pass).
  - Riemannian re-centering.
  - BatchNorm-statistics adaptation (AdaBN) — `neurips_challenge/adabn_compare.py` may be reusable *as code only*.
  - TENT-style entropy minimisation on retrieval logits.
- [x] **E3.6 SAGE vs SCORE mechanism ablation** (DONE: MNN 1-step 59.5 vs SAGE 82.3; 1-step Sinkhorn 68.8) (`r1_ablate.py` mnn / mnn_1step / 1step rows): one-step Procrustes on mutual-NN landmarks vs iterated Sinkhorn–Procrustes. This is the core novelty argument. [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]

## Phase 4 — Protocol regimes (CPU)

- [x] **E4.1 R2 streaming grid.** DONE atm31 (buffer + prequential; see log night 9). Repeat on final P-main model. `CVPR_paper/scripts/stream_grid.py` is running on `atm31_valcon` (buffer score plus prequential `@new`). [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]
  - K ∈ {25, 50, 100, 150, 200} × q ∈ {5, 10, 20, 40}; refit every m arrivals; score the buffer (D2); 3 stream orders × random menus.
  - Run on P-clean models with P-clean hyperparameters, for SAGE, SATTC, SCORE and SVTL.
  - Old driver: `scripts/things_eeg/tta_rebuttal/run_streaming_refit.py --acausal`, `run_acausal_menu_grid.sh` (the script that built the table is missing; rebuild it).
  - Fix the "50 refits" claim (it was 25 for K ≤ 50 / 150).
  - Optional (Q6): prequential column.
- [x] **E4.2 Damping schedule.** DONE: alpha=u/(u+3) picked by all 10 folds; prequential 59.6 vs 56.1 (alpha=1); first refit 35.7 vs 17.3 (plain 39.2); end buffer unchanged. SUPERSEDED by E4.2b (2026-10-21 01:25): FULL damping (the same alpha also on SAW whitening and CSLS) u/(u+3): LOSO preq 60.8 (+1.2 vs rotation-only, 9/10), intra 87.5 vs plain 82.4 (+5.1, 10/10), first checkpoint at plain level in both. ADOPTED as the streaming default.
  - Choose the ramp constants on the validation folds (the old `ramp_grid` 0.8·u/(u+1.5) was not selected on held-out data).
  - Report crossover time and cold-start harm with and without damping.
- [x] **E4.3 R3 coverage and multiplicity.** DONE atm31: SAGE < plain for U<=100, < SCORE for U<=150 (see log); fix search CLOSED 2026-10-21 (v1-v4 + UOT, 5 families: best label-free variant ties/trails SCORE on the 12-cell mean; SAW is culprit 1, the forced bijection culprit 2). Still to run on the final model (queued in final_analyses.sh). [FINAL atm128 DONE: same map, SCORE 38.8 vs SAGE 29.1 mean; SAGE wins R1 + U=200]
  - U/K sweep, many-to-one R, and Dirichlet-imbalanced query counts (Veilleux et al. 2021 protocol).
  - Arms: balanced / free / adaptive ρ, plus oracle support as an upper bound.
  - Old: `run_partial_coverage*.py`.
- [x] **E4.4 R4 open gallery.** DONE atm31: SAGE < SCORE for every D>0, < plain at D=1654; fix search CLOSED (see E4.3). Still to run on the final model (queued). [FINAL atm128 DONE: D1654 SAGE 12.6 vs SCORE 27.6]
  - Add D distractors from the 16,540 training images (old: `run_openset_streaming.py`).
  - Test the fix: unbalanced OT (dummy sink column) and/or a top-N′ shortlist from R0 scores. If neither works, report it as a limitation.
- [x] **E4.5 Carry-over (secondary):** COVERED by E6.2 sage_cross (final atm128: calibration fitted on another session, applied FROZEN to fresh trials: 52.69 vs plain 28.55 vs own-session 56.01 = 88% of the gain carries over; log 04:10). No separate run. the calibration fitted on a stream is frozen and applied to fresh trials of the same menu (Exp A, `run_calibrate_then_deploy.py`), reported at matched SNR.

## Phase 5 — Label-equivalence (GPU-light)

- [x] **E5.1 Supervised calibration baselines.** DONE (label_equiv2, matched CSLS): unlabeled SAGE 82.3 ties supervised rotation on ALL 16,540 labeled target pairs (80.6, p=0.49); labels+SAGE 90.6 (E5.2 done too). Encoder fine-tuning not run (GPU). Old note follows: Running `run_inductive_rotation_sweep.py` on atm31_valcon (n = 25..all labeled 4-rep TRAIN-split trials, 3 seeds, alpha grid; alpha to be chosen cross-fold in analysis). Head/full fine-tuning not run (needs GPU); the supervised linear maps are the same function class as SAGE's. [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]
  - Use n labeled target trials from the target subject's *training* split (concepts disjoint from the test set), with n ∈ {25, 50, 100, 200, 500, 1000, all}.
  - Methods: supervised Procrustes, a linear map, head fine-tuning, and full fine-tuning (with early stopping on a target validation split).
  - Report n* = the number of labels needed to match SAGE-TTA with a given unlabeled budget.
  - Old code: `run_inductive_rotation_sweep.py` (supervised rotation; its α was chosen on test — reselect).
- [x] **E5.2 Labels + unlabeled stream combined**: 90.6 with all labels (+8.35 over SAGE, 10/10); hurts with n<=200. [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]

## Phase 6 — Intra-subject and cross-session

- [x] **E6.1 Intra-subject clean.** atm128_oc plain 87.75 -> SAGE 99.10 cross-fold (SCORE 93.0); atm_oc 99.45, epj_oc 99.5, tsc_oc 98.75, sqf_oc 94.45 cross-fold.
  - Base: EEGProject one-cycle (`epj_oc`, inductive 87.85 two-seed) and TSConv one-cycle, selected on ValCon instead of test loss.
  - Apply SAGE-TTA with cross-fold hyperparameters. Old: 85.25 → 99.05, single seed, test-selected (`intra-subjects/tta_grid_sweep_20260503-150419`).
  - Use it as the ceiling and as a demonstration of the correspondence prior.
- [x] **E6.2 Cross-session.** DONE on atm31 (LOSO): own-session SAGE 53.4 vs other-session frozen 50.3 (+3.1, 10/10, p=0.002); see log. Optional: repeat on intra models. Test-split sessions RECOVERED (preprocessing shuffle seed 20200220, verified by fingerprint clustering; see log). Train-split sessions are NOT recoverable (only the pairing {1,3}/{0,2}). Running `session_calib.py` (own-session vs cross-session vs mixed fits, 20-rep queries). [FINAL atm128 s3300 DONE 2026-10-21, log 04:10-04:30]
  - First verify whether the preprocessed `.npy` files keep the session index. The test set's 80 repetitions are expected to span 4 sessions.
  - If they do: train on sessions 1–3 and calibrate label-free on session 4 ("every morning is a new subject").
- [x] **E6.3 Intra-subject regimes:** DONE atm128_oc (map holds; full damping +5.1 over plain). R2 and R3 on intra models, to check whether the protocol conclusions hold when the base is strong.

## Phase 7 — Other datasets and modalities

- [ ] **E7.1 Alljoined-1.6M LOSO**, clean.
  - Recompute the baseline: the paper's 8.0 has no on-disk source, and the existing no-mix runs score 10.1 and 9.2.
  - Then SubjectMix, then TTA with cross-fold hyperparameters. Note the "seed3302" directory is really seed 3310.
- [ ] **E7.2 THINGS-MEG**, clean. Resolve the swapped appendix parameter strings; discard the steps=1 / iters=1 test artefacts.
- [ ] **E7.3 NSD / MindEye2.**
  - Report only the fit-on-1000 CV result (image retrieval 68.4 → 84.8, 1000-way).
  - Either redo the per-pool 300-way CV (PCA rank was picked on test; the rotation can memorise the assignment at k/n = 1.7), or drop the 300-way numbers.
  - Brain retrieval is inflated by the bijection; lead with image retrieval.

## Phase 8 — Analyses for the paper

- [x] **E8.1 Pitfall table** DONE 2026-10-21 (log 04:35; P1 from atm31 ValCon, rest final atm128). (P1–P7 in `paper.tex`), recomputed on P-clean models: inflation in points for each.
- [x] **E8.2 Complementarity.** DONE 2026-10-21 (log 05:05): gain vs R0 null within model (pooled rho -0.13, p=0.28) and across models (rho -0.36, p=0.43; ATM +45-48 vs TSConv/SQF +28-32 at equal R0); gain rises with q (SNR). Intra rho -0.86 is a ceiling artefact. Calibration gain vs R0 accuracy across encoders × seeds × q × datasets. Report honest correlations; per-subject "weak subjects benefit more" is not supported so far (ρ = −0.32, p = 0.36).
- [x] **E8.3 Geometry diagnostics:** DONE atm31 (angle 79 deg, gap 0.33->0.53, hubness 2.26->1.03); final model queued. rotation angle spectrum, residual after alignment, hubness before and after. These give direct structural evidence (the old Rekw-Q5 request) without causal claims. [FINAL atm128 DONE: angle median 71.8, gap 0.325->0.537, hub skew 2.27->1.05]
- [x] **E8.4 Compute cost** per calibration step on CPU. SAGE 64.5 ms / 200-query batch on 1 core (SCORE 5.7 ms), see log 2026-10-21 00:35.

## Phase 9 — Writing

- [~] Fill `paper.tex` sections. FIRST DRAFT 2026-10-21 (paper.tex, 7.7 pages + refs; skeleton kept as paper_skeleton.tex; figures: regimes.pdf, stream.pdf via scripts/make_figures.py). Missing: teaser/protocol figure, Alljoined/MEG, supplementary. Figures: protocol diagram, R2 curves, R3/R4 curves, label-equivalence curve, complementarity scatter.
- [ ] Supplementary: per-subject tables, sensitivity, competitor reproduction details, encoder details.
- [ ] Final number audit: every number in the paper traced to a path in `EXPERIMENT_LOG.md`.

---

## Priority order (D18)

1. **P-main R1 numbers.**
   - E2.0: CPU screen of existing checkpoints (running).
   - E1.1, E1.2, E1.2b: retrains (queued).
   - New encoders E1.3–E1.5: screens queued.
2. **Intra-subject** (E6.1): CPU screen of the existing one-cycle models (chained after E2.0); then ATM intra.
3. **Phase 4 regimes** on the best 2–3 models (E4.1 with the prequential column, E4.3, E4.4).
4. **Competitors** (E3.1–E3.5): SATTC cloned at `/nasbrain/p20fores/SATTC-CVPR2026` (commit 0cdfd1b); SCORE and SVTL to reimplement.
5. **P-clean** (E1.8), Alljoined (E7.1), MEG (E7.2), label-equivalence (E5.1), NSD (E7.3).
6. **Writing.**
