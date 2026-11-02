# CVPR 2027 paper — experiment log

Newest entries go at the top. For every run, record:
- date;
- experiment ID (from `checklist.md`);
- command or script;
- output path;
- protocol (checkpoint selection, how hyperparameters were selected, seeds);
- result;
- verdict.

Only runs under protocol P-clean (`checklist.md`) can feed the paper. Older numbers are listed in the provenance section below as context only.

## 2026-10-23 14:15 — E3.2d faithful-recovery SCORE variants (30 epochs) — QUEUED (strengthening the baseline)

- **Gap.** Our SCORE training adds +0.95 over Standard at 30 epochs (E3.2c / 30-epoch entry). The paper reports
  +3.1. E3.2c isolated the recovery loss as the component that hurts.
- **Two untested readings of the paper**, both new flags in the SAMGA fork `train.py`:
  - `--score_rec_tau 0.07`: the paper's τ_rec = 0.07 on L_rec ONLY. Before, L_rec used SAMGA's scale (τ ≈ 0.37). The
      earlier `scoret` arm put τ = 0.07 on L_MP too, which failed (20.3).
  - `--score_rec_start_epoch 20`: L_rec only in stage 2, after the MMD warm-up.
- **Arms** (`samga_fold.sh`): `score30rt` and `score30s2`, seed 3300, 10 folds each, 30 epochs, last epoch.
- **GPU.** New sbatch workers 64319 (515) and 64320 (517), a step in 63835 (502), and nohup on interactive 64236 (514).
- **CPU.** `score_variants_eval.py` (plain + SCORE deploy) when 10 folds exist.
- **Decision rule.** If either variant reaches SCORE − Standard ≈ +3 (Standard 30 epochs = 26.70) and adapted ≥ 52,
  it becomes the SCORE (original) column: rerun its battery, SAGE included. Otherwise report E3.2c / E3.2d as the
  reproduction caveat.

## 2026-10-23 11:55 — E8.8b diversity vs volume — DONE: both matter; diversity dominates at low subject counts

Folds 1/3/8, each arm's own 3-fold params.

| Arm (data volume) | Subjects × concepts | plain | Hungarian | SCORE | SAGE | ff_SAGE | SAGE − SCORE | RSA | oracle (centred) |
|---|---|---|---|---|---|---|---|---|---|
| full (1) | 9 × all | 35.0 | 71.7 | 71.8 | 89.0 | 0.83 | +17.2 | 0.70 | 97.5 |
| `n6` (2/3) | 6 × all | 24.5 | 58.2 | 60.5 | 76.3 | 0.69 | +15.8 | 0.69 | 95.5 |
| `v6` (2/3) | 9 × 2/3 | 30.8 | 60.5 | 61.2 | 77.8 | 0.68 | +16.7 | 0.65 | 96.0 |
| `n3` (1/3) | 3 × all | 16.7 | 32.8 | 34.3 | 39.7 | 0.28 | +5.3 | 0.62 | 68.8 |
| `v3` (1/3) | 9 × 1/3 | 25.0 | 51.0 | 48.8 | 61.5 | 0.49 | +12.7 | 0.65 | 95.7 |

- **At 1/3 volume, diversity is decisive.**
  - 9 subjects beat 3 subjects by +21.8 SAGE and ff 0.49 vs 0.28.
  - The rotation ceiling collapses with only 3 source subjects (oracle 68.8 vs 95.7). The EEG space learnt from 3
    people does not admit a good rotation onto the image space for a new person.
- **At 2/3 volume, diversity no longer matters:** 6 vs 9 subjects gives 76.3 vs 77.8, ff 0.69 vs 0.68.
- **Volume matters independently:** with 9 subjects, ff goes 0.49 → 0.68 → 0.83 for 1/3 → 2/3 → all concepts.
- **RSA tracks less cleanly here** (v3 0.65 > n3 0.62 fits; v6 0.65 < n6 0.69 does not), but the oracle ceiling
  separates the n3 failure clearly.
- **Paper:** "TTA needs a sufficiently diverse source: below ~6 subjects the learnt space stops being rotatable onto
  the target space for a new subject. Above that, more data per subject keeps increasing the TTA gain." This replaces
  E8.8's confounded reading.

## 2026-10-23 10:15 — E4.7 streaming (R2): support pilot NEGATIVE; label-free SCORE→SAGE switch DOMINATES

**Support arms** (`stream_grid.py --support`: `full1_3s` / `full1_3sf` = `full1_3` fitted on SCORE's top-1 support of
the buffer, balanced or free marginal). Pilot on atm128 s3300, subject 1, q = 40, K = 100 / 200, 50 checkpoints.
- Early in the stream (first 30 %): at K = 100, `full1_3` 51.7 → `full1_3s` 60.6, but SCORE gets 66.1.
- Over the whole stream the arms are unchanged (at-arrival K = 100: 74.0 / 76.2 vs SCORE 76.8).
- Reading. SAGE's early R2 deficit comes from small buffers and the damping (α = u / (u + 3) ≈ 0 early), not from
  gallery support. Not pursued.

**Hybrid** (CPU, existing `stream50` rows: ATM-128 × 3 seeds × 10 subjects × K 25–200 × q 5–40 × 3 orders × 50
checkpoints). Score each arriving chunk with SCORE while u = t/K < u*, else with SAGE (`full1_3`). u is label-free:
buffer size over menu size.
- u* chosen leave-one-subject-out from the grid {0, .05, .1, .2, .3, .5, .75, 1, 1.5, 2, ∞}: it is u* = 1.5 in all
  10 folds.
- Prequential at arrival, mean of 10 subjects: hybrid 63.47, SAGE 62.40, SCORE 56.42.
  - The hybrid beats SAGE on 9/10 subjects (exception sub-4: 46.75 vs 47.05) and SCORE on 10/10.
- By (K, q) cell, pooled u* = 1.5: hybrid ≥ SCORE in all 20 cells. Hybrid / SAGE / SCORE:

  | K | q | Hybrid | SAGE | SCORE |
  |---|---|---|---|---|
  | 100 | 20 | 70.6 | 68.3 | 63.4 |
  | 100 | 40 | 70.6 | 67.2 | 67.9 |
  | 200 | 40 | 60.6 | 57.1 | 58.2 |
  | 25 | 40 | 79.9 | 81.6 | 77.8 |

  At K = 25, q = 40 the hybrid is slightly below SAGE.
- **Paper.** "Deployment recipe: SCORE's one-step map while the buffer is under 1.5× the menu, SAGE afterwards. The
  switch point is label-free and stable under subject cross-validation; it removes the only R2 cells where SCORE beat
  SAGE." SCORE's early strength is folded in, so present it as a schedule, not as SAGE alone.

## 2026-10-23 09:55 — E8.8b source-diversity vs data-VOLUME control — QUEUED (exploration reopened, goal still active)

- **Code.** `EEGPreImageDataset.keep_objects(fraction, seed=0)` plus `train.py --train_object_fraction`: train on a
  fixed random subset of the training concepts, the same subset for every subject. Unit-checked on sub-01: 1654 → 551
  concepts, local object ids 0–550, features and EEG sliced consistently.
- **Arms.** `atm128_v3` (9 subjects × 1/3 of the concepts = 49,590 samples, vs `atm128_n3` 3 × all = 49,620) and
  `atm128_v6` (2/3 of the concepts, vs `atm128_n6`). Folds 1/3/8, plus `screen3_eval`.
- **Question.** At equal data volume, does subject diversity (9 subjects) give a larger TTA gain than more data per
  subject (3 subjects)? This decides whether E8.8 is about diversity or volume.
- **Queue.** These are appended after the last SCORE-seed folds; the GPU queue had just emptied.

## 2026-10-23 09:25 (updated 12:05) — EXPLORATION CONCLUDED (user goal: SAGE-vs-SCORE edge + TTA-paradigm insights)

**Paper-ready findings** (details in the entries below):

*SAGE's edge over SCORE*
1. R1: SAGE > SCORE on 110 / 110 subject × encoder pairs, 9 encoders. Margin +8.6 to +23.2; minimum +3.5 (E8.4b).
2. Strongest SCORE baseline we can build: SAGE beats SCORE's REPORTED 53.23 by ~+15–17 on SCORE's own stack.
   - 50-epoch stack: SAGE 68.9.
   - 30-epoch stack: SAGE 70.3, vs SCORE 48.6 on the same encoder.
3. R4 open gallery: SAGE + Hungarian support ≥ SCORE at every D on ATM-128, keeping +18 at D = 0. On the SCORE stack
   it wins at D ≤ 200 and ties within 0.8 beyond (E4.5).
4. R3 partial coverage: top-1 support + free marginal brings SAGE from −21 to ±2 of SCORE (E4.5).
5. The edge needs ≥ 20 averaged repetitions and grows with SNR (E8.10 reps sweep).
6. SAGE's fixed point is init-insensitive: SCORE-init +0.3 (E4.6).
6b. R2 deployment recipe (E4.7): SCORE while buffer < 1.5 × menu, then SAGE. The switch point is label-free;
    LOSO always picks 1.5. Prequential 63.5 vs SAGE 62.4 vs SCORE 56.4; ≥ SCORE in all 20 (K, q) cells.

*Which encoders suit TTA*
7. Zero-shot accuracy does not predict the TTA gain across architectures (rho 0.00).
8. RSA (EEG–target relational agreement) predicts SAGE's gain:
   - across architectures, rho 0.87;
   - across target spaces and source sizes, 0.96;
   - across target depth, 0.94.
9. RSA is a diagnostic, not a lever: an RSA loss raises RSA, but SAGE is unchanged (E8.5).
10. No label-free encoder predictor: GW, mutual-NN, hubness and effective rank of EEG all fail (E8.9).
11. The target layer is first-order. Intermediate layers (InternViT L28–31, bigG L27, EVA L35, DINOv2 L20) give
    ff 0.63–0.83. Final or near-final layers (CLIP ViT-H final, InternViT L44, DINOv2 L40) give ff 0.17–0.26 and kill
    SAGE's edge (+1–3). This replicates on TSConv (E8.7 / 7b / 7c / 7d).
12. Image-only screen: the target gallery's effective rank predicts ff (rho −0.81, 11 targets). Coarse, non-monotone
    within a model.
13. Source diversity vs volume (E8.8 / E8.8b), at equal volume:
    - 3 source subjects collapse TTA: ff 0.28 vs 0.49 with 9 subjects × 1/3 of the concepts; the rotation ceiling
      is 69 vs 96.
    - At ≥ 6 subjects diversity saturates, and more data per subject keeps helping (ff 0.49 → 0.68 → 0.83).
14. Checkpoint choice: zero-shot-best ≈ TTA-best for ATM-128 / TSConv / rmix. Exception: SAMGA's stage-1 peak (E8.10).
15. Embedding dimension: post-hoc PCA never helps; SAGE uses ~64 dims (E8.6).

**Not pursued** (judged low value, or too invasive before the draft):
- a trained feature-dim sweep (E8.6 says no post-hoc effect);
- a streaming (R2) support variant (pilot negative, see E4.7);
- a recovery-loss fix for SCORE (no authors' code; speculative).

**Still running, automatic:** SCORE (original) seeds 3301 / 3302 (13 / 20 folds), then their last-epoch and best-epoch
batteries, 50-fit streams, and `seed_table.py --arms atm128 score_orig score_orig_tb` → `final/main_table_3seeds.md`.

## 2026-10-23 09:20 — E8.7d final-layer effect replicates on a second architecture (TSConv) — DONE

`analysis/rsa_screen/tsconv_{f138,iv44}_cv3.csv`. Folds 1/3/8, each arm's own 3-fold params. `tsconv_f138` = symlinks to
the existing `tsconv` s3300 folds.

| Target | plain | Hungarian | SCORE | SAGE | ff_SAGE | SAGE − SCORE | RSA | oracle (centred) |
|---|---|---|---|---|---|---|---|---|
| TSConv, InternViT L28 | 34.2 | 49.5 | 45.5 | 59.0 | 0.38 | +13.5 | 0.64 | 98.3 |
| TSConv, InternViT L44 | 17.7 | 24.3 | 24.5 | 26.2 | 0.10 | +1.7 | 0.54 | 98.3 |

- Same collapse as on ATM-128 (ff 0.83 → 0.17): the final-layer effect is not architecture-specific. RSA drops again.
- On TSConv the oracle rotation ceiling stays at 98 for both layers. So the loss is in pseudo-label / matching quality
  (Hungarian 49.5 → 24.3), not in rotatability.
- Final-layer evidence now: 3 vision models (CLIP, InternViT, DINOv2) × 2 EEG architectures (ATM-128, TSConv).

## 2026-10-23 08:15 — E8.7c InternViT target-depth curve — DONE; image-only predictor of TTA suitability

**Depth curve.** `analysis/rsa_screen/atm128_<arm>_cv3.csv`. ATM-128 recipe, folds 1/3/8, each arm's own 3-fold
params.

| Layer | plain | Hungarian | SCORE | SAGE | ff_SAGE | SAGE − SCORE | RSA | oracle (centred) |
|---|---|---|---|---|---|---|---|---|
| 20 | 31.2 | 62.3 | 63.3 | 77.3 | 0.67 | +14.0 | 0.692 | 91.0 |
| 28 | 35.0 | 71.7 | 71.8 | 89.0 | 0.83 | +17.2 | 0.696 | 97.5 |
| 31 | 35.2 | 71.3 | 70.5 | 89.0 | 0.83 | +18.5 | 0.697 | 98.3 |
| 36 | 33.7 | 64.7 | 63.8 | 75.7 | 0.63 | +11.8 | 0.678 | 96.0 |
| 39 | 27.7 | 53.8 | 55.7 | 66.0 | 0.53 | +10.3 | 0.645 | 97.5 |
| 44 | 19.7 | 31.7 | 32.8 | 33.7 | 0.17 | +0.8 | 0.580 | 81.3 |

- An inverted U, peaking at L28–31; our L28 choice sits at the peak.
- Spearman: RSA ~ ff = 0.94; zero-shot ~ ff = 0.89 (within one model, quality moves together).

**Image-only predictor** (`analysis/target_space_stats.csv`). Statistics of the 200 test-image target features alone,
before any EEG training, across 11 target spaces.

| Target | PR | ff_SAGE |
|---|---|---|
| InternViT L20 | 12 | 0.67 |
| InternViT L28 | 18 | 0.83 |
| InternViT L31 | 23 | 0.83 |
| InternViT L36 | 27 | 0.63 |
| InternViT L39 | 29 | 0.53 |
| InternViT L44 (final) | 46 | 0.17 |
| CLIP ViT-H final | 67 | 0.17 |
| DINOv2 L20 | 40 | 0.63 |
| DINOv2 L40 | 78 | 0.26 |
| OpenCLIP bigG L27 | 44 | 0.67 |
| EVA02-E L35 | 40 | 0.63 |

- *Participation ratio* (effective rank) of the centred gallery vs ff_SAGE: Spearman −0.81; vs RSA: −0.81. All final
  or near-final layers have PR ≥ 46 and ff ≤ 0.26; all good targets have PR ≤ 44.
- *Mean pairwise cosine (anisotropy):* +0.63. *Hubness skew:* ~0.
- *Not monotone within InternViT:* L20 has the lowest PR, 12, but ff 0.67 < L28.
- *Reading.* The EEG encoder mirrors only a low-dimensional relational structure: SAGE uses ~64 dims (E8.6), and EEG
  PR is ~11–15 (E8.4). Targets whose semantics spread over many dimensions (final layers) cannot be mirrored: low RSA,
  low TTA gain. A coarse, label-free pre-training screen ("avoid high-PR targets"), not a precise selector.

## 2026-10-23 01:20 — E8.7 / E8.8 target space and source diversity — DONE

`analysis/rsa_screen/atm128_<arm>_cv3.csv`. ATM-128 recipe, seed 3300, folds 1/3/8. Each arm uses its own 3-fold
cross-validated SAGE params. ff = fraction of zero-shot errors fixed.

| Arm | plain | Hungarian | SCORE | SAGE | ff_SAGE | ff_SCORE | SAGE − SCORE | RSA | oracle (centred) |
|---|---|---|---|---|---|---|---|---|---|
| InternViT-6B L28 (ours, `atm128_f138`) | 35.0 | 71.7 | 71.8 | 89.0 | 0.83 | 0.57 | +17.2 | 0.70 | 97.5 |
| OpenCLIP bigG L27 | 31.5 | 64.2 | 60.8 | 77.3 | 0.67 | 0.43 | +16.5 | 0.69 | 94.2 |
| EVA02-E L35 | 31.8 | 60.7 | 62.0 | 75.0 | 0.63 | 0.44 | +13.0 | 0.67 | 95.2 |
| DINOv2-giant L20 | 28.8 | 60.8 | 57.5 | 73.8 | 0.63 | 0.40 | +16.3 | 0.68 | 96.2 |
| CLIP ViT-H-14 final (field default) | 14.8 | 23.8 | 27.5 | 29.0 | 0.17 | 0.15 | +1.5 | 0.52 | 79.5 |
| 6 / 9 source subjects | 24.5 | 58.2 | 60.5 | 76.3 | 0.69 | 0.48 | +15.8 | 0.69 | 95.5 |
| 3 / 9 source subjects | 16.7 | 32.8 | 34.3 | 39.7 | 0.28 | 0.21 | +5.3 | 0.62 | 68.8 |

- **Target space is a first-order TTA factor.**
  - The intermediate layers of large vision models (InternViT L28, bigG L27, EVA L35, DINOv2 L20) all give
    ff_SAGE 0.63–0.83.
  - The field's default CLIP ViT-H FINAL embedding collapses (zero-shot 14.8, ff 0.17, RSA 0.52).
  - Caveat: the recipe (128-d, τ, epochs) was tuned on InternViT L28, so part of the CLIP gap may be recipe mismatch.
    But all other targets used the same recipe and stay ≥ 0.63.
- **Source diversity amplifies TTA, it does not substitute for it.** ff_SAGE goes 0.28 → 0.69 → 0.83 with 3 / 6 / 9
  source subjects; SAGE − SCORE goes +5 → +16 → +17. Confound: fewer subjects = less data.
- **RSA again predicts TTA gain across these 7 training interventions** (same architecture): Spearman RSA ~ ff_SAGE =
  0.96, RSA ~ SAGE = 0.93.
  - Zero-shot also predicts here (0.75), since these interventions move quality broadly; across ARCHITECTURES it did
    not (E8.4: 0.00).
  - SAGE's edge over SCORE collapses exactly where RSA is low (CLIP final +1.5, 3 subjects +5.3). Mechanism: SAGE
    exploits relational structure that SCORE's single step cannot.
- **01:30 QUEUED E8.7b** (ahead of the SCORE seeds; 6 folds + `screen3_eval` waiters). Final vs intermediate layer
  of the same model, to tell whether CLIP's collapse is "final layer" or "CLIP":
  - `atm128_iv44`: InternViT-6B L44, vs L28 = ours;
  - `atm128_dino40`: DINOv2-giant L40, vs L20.
  - **04:05 InternViT result: the collapse is the FINAL LAYER, not CLIP.** `atm128_iv44` (InternViT-6B L44):

    | InternViT layer | plain | Hungarian | SCORE | SAGE | ff_SAGE | SAGE − SCORE | RSA | oracle |
    |---|---|---|---|---|---|---|---|---|
    | L44 (final) | 19.7 | 31.7 | 32.8 | 33.7 | 0.17 | +0.8 | 0.58 | 81.3 |
    | L28 | 35.0 | | | 89.0 | 0.83 | | 0.70 | |

    The same model's final layer reproduces CLIP ViT-H final's failure (ff 0.17, RSA 0.52).
  - **04:45 DINOv2 confirms it.** `atm128_dino40` (L40, near-final):

    | DINOv2 layer | plain | Hungarian | SCORE | SAGE | ff_SAGE | SAGE − SCORE | RSA |
    |---|---|---|---|---|---|---|---|
    | L40 | 20.2 | 37.0 | 38.5 | 41.0 | 0.26 | +2.5 | 0.59 |
    | L20 | 28.8 | | | 73.8 | 0.63 | +16.3 | 0.68 |

    3 / 3 models: a final or near-final target layer kills both the TTA gain and SAGE's edge, and RSA flags it.
    Paper: "train the EEG encoder against an intermediate layer if you intend to adapt at test time".
  - **04:10 QUEUED E8.7c, InternViT target-depth curve.** `atm128_iv{20,36,39}` (ATM-128 recipe, folds 1/3/8, 9 folds),
    ahead of the SCORE seeds. Together with L28 (`atm128_f138`), L31 (`atm128_l31_f138` = symlinks to the existing
    10-fold `atm128_l31` folds 1/3/8) and L44, that is 6 depths. Each gets `screen3_eval` (own 3-fold params; ff, RSA,
    SAGE − SCORE).
  - Diversity vs data-volume control (9 subjects × 1/3 of the images) NOT run: it needs an image-subset option in
    the dataset and grouped sampler. The confound is reported as a caveat.
- **Paper:** "the EEG–target geometric agreement (RSA) is the common currency: it varies with architecture (E8.4),
  target space and source diversity (E8.7 / E8.8), and predicts SAGE's gain in all three. Raising it directly with a
  loss does not (E8.5)."

## 2026-10-22 20:55 — STATUS (what is still running, all automatic)

- **GPU queue**, 4 workers: 64084 (node 515), 64085 (517), a step in 63835 (502, 3090), and a nohup worker on 514.
  - E8.7 target spaces + E8.8 source diversity: 4 / 18 folds done, the rest ≈ 01:30.
  - Then the SCORE (original) seeds 3301 / 3302: 20 folds, ≈ 07:00.
- **CPU waiters**, which fire automatically:
  - `screen3_eval.sh` for `atm128_{cliph,dino20,bigg27,eva35,n3,n6}` → `analysis/rsa_screen/<arm>_cv3.csv`. Compare with
    `atm128_f138_cv3`: SAGE 89.0, plain 35.0, Hungarian 71.7, SCORE 71.8, RSA 0.70.
  - The SCORE-seed batteries (last-epoch and test-best), streams, and `seed_table.py` → `final/main_table_3seeds.md`.
- **Findings so far today (E8.x / E4.x)**, in paper order:
  1. SAGE > SCORE on 110 / 110 subject-encoder pairs (R1).
  2. SAGE + Hungarian support ≥ SCORE in R4 on ATM-128 at all D. On the SCORE stack it is ≥ SCORE at D ≤ 200 and
     within 0.8 below for D ≥ 400. A top-1 support with a free marginal brings R3 to within ±2.
  3. SAGE's edge needs ≥ 20 repetitions.
  4. RSA predicts the gain across encoders but is not a training lever; within a run it is flat.
  5. Zero-shot accuracy predicts nothing at the encoder level. No label-free predictor found (GW negative).
  6. The zero-shot-selected checkpoint ≈ the TTA-optimal one, except SAMGA's stage-1 peak.
  7. SAGE is insensitive to initialisation (SCORE-init +0.3).
  8. Post-hoc PCA never helps; SAGE needs ~64 dims.

## 2026-10-22 20:00 — INCIDENT (fixed): FINAL_CKPT leaked into unrelated CPU-queue items

**What happened.** `final_analyses.sh` starts 4 inner `cpu_worker.sh` that also pop OTHER queue items. They inherited
the battery's `FINAL_CKPT`.
- The `score30_orig` battery runs with `FINAL_CKPT=checkpoint_last.pth`.
- So `screen3_eval.sh` (atm128_f138 / rsa1 / rsa5) ran `tta_screen_loso.sh` on `checkpoint_last.pth`, which does not
  exist for these runs. They FAILED (FileNotFoundError); no silent wrong numbers.

**Audit of other items popped since 11:20** (the first last-epoch battery, `samga_orig_intra_last`):
- `regimes_support`, `epoch_curve_eval`, `score_variants_eval` and `rsa_screen_eval` all set `SAGE_CHECKPOINT_NAME`
  explicitly, so they are unaffected.
- `stream50.sh` takes its checkpoint as an argument.
- The 10-fold batteries pass `FINAL_CKPT` explicitly as a prefix, which overrides the inherited value.

**Fix.**
- `final_analyses.sh` starts its inner workers with `env -u FINAL_CKPT -u INTRA`.
- `screen3_eval.sh` sets `FINAL_CKPT=checkpoint_test_best.pth` explicitly.
- All 9 screen3 waiters were killed and requeued (one process each, verified).

## 2026-10-22 19:50 — E8.10 TTA vs training length — DONE: the SCORE-stack effect does NOT generalise

`analysis/epoch_curve/<arm>_epXXX.csv`, folds 1/3/8. SAGE params are frozen from the base arm's 10-fold screen.
Means over the 3 folds:

**ATM-128:**

| Epoch | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 20.7 | 24.3 | 25.8 | 26.2 | 30.3 | 30.3 | 29.0 | 32.8 | 28.5 | 30.3 | 32.2 |
| SAGE | 49.7 | 65.0 | 81.3 | 85.7 | 86.5 | 90.0 | 87.3 | 88.2 | 88.8 | 90.5 | 87.7 |
| RSA | 0.64 | 0.68 | 0.71 | 0.70 | (flat from ep 20) | | | | | | |
| oracle (centred) | 72 | 82.5 | 90 | 92 | 95 | 95 | 96 | 97 | 98 | 98 | 98 |

**TSConv:**

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| plain | 23.5 | 27.0 | 30.2 | 31.3 | 31.7 | 32.3 | 31.7 | 28.8 |
| SAGE | 32.5 | 52.3 | 54.3 | 58.7 | 61.5 | 61.0 | 60.2 | 57.5 |

**rmix:**

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| plain | 15.5 | 21.0 | 21.0 | 20.8 | 22.8 | 21.3 | 21.8 | 22.2 |
| SAGE | 46.7 | 62.7 | 63.5 | 60.5 | 63.7 | 61.5 | 61.8 | 60.5 |

Hungarian on rmix peaks at ep 10 (59.2), then declines. RSA is flat at 0.64; the oracle stalls at ~85.

- **Within-run Spearman (across checkpoints) of SAGE with each statistic:**

  | Arm | plain | Hungarian | SCORE | RSA | oracle |
  |---|---|---|---|---|---|
  | ATM-128 | +0.74 | +0.84 | +0.93 | +0.01 | +0.82 |
  | TSConv | +0.93 | +0.81 | +0.76 | +0.55 | +0.65 |
  | rmix | +0.51 | +0.66 | +0.69 | +0.18 | +0.18 |

- **SAGE at the best-zero-shot checkpoint vs best SAGE:** ATM-128 88.2 vs 90.5; TSConv 61.0 vs 61.5; rmix 63.7 vs 63.7.
  For these encoders, zero-shot checkpoint selection is ≈ TTA-optimal.
- **The SCORE-stack exception** (zero-shot best ~ep 14 → SAGE 59.85; ep 30 → 70.3; ep 50 → 68.9) is specific to SAMGA's
  two-stage schedule. Its zero-shot peak falls inside the MMD warm-up stage (≤ ep 20); SAGE prefers the stage-2
  geometry. Do NOT claim "over-train for TTA" in general.
- **RSA along training** jumps early (ATM-128: 0.64 → 0.71 by ep 20), then is flat while SAGE keeps rising. So RSA is
  an encoder-level property, consistent with E8.5's "diagnostic, not lever". Within a run, matching quality (Hungarian,
  SCORE) tracks SAGE best.
- **Paper value:** a "practitioner's note", with one exception to report (SAMGA's stage-1 peak).

## 2026-10-22 17:25 — 30-epoch SAMGA/SCORE (user's reproduction regime) — GPU DONE

Last-epoch zero-shot, 10 subjects:

| Run | Ours | Paper |
|---|---|---|
| `samga30` (Standard) | 26.70 (best epoch 31.85) | 26.22 |
| `score30` | 27.65 | 29.33 |
| SCORE − Standard | +0.95 ± 1.04 (p = .38, 7/10 wins) | +3.11 |

- At 30 epochs both rows sit closest to the paper.
- The SCORE training gain stays ~1/3 of the reported one and not significant.
- Pending: SCORE-deploy on both (`score_variants_eval.py`) and the `score30` battery (SAGE on it).
- **17:50 adapted numbers** (R1):

  | Run | plain | SCORE deploy | SAGE |
  |---|---|---|---|
  | `samga30` | 26.70 | 50.60 | |
  | `score30` | 27.65 | 48.60 | 70.30 (`final/score30_orig_s3300`) |

  - The SCORE paper reports 53.23 adapted. Our closest is 50.60, from the Standard stack at 30 epochs.
  - Against the paper's 53.23, SAGE on `score30` is +17.1; against SCORE on the same encoder, +21.7.
- **SCORE-stack trajectory (E8.10 data point):**

  | Checkpoint | Zero-shot | SCORE | SAGE |
  |---|---|---|---|
  | ~ep 14 (best zero-shot) | 33.7 | 49.65 | 59.85 |
  | ep 30 | 27.65 | 48.6 | 70.3 |
  | ep 50 | 24.5 | 47.15 | 68.9 |

  SAGE peaks well AFTER the zero-shot optimum; SCORE tracks zero-shot.

## 2026-10-22 16:22 — E4.5 result (full: ATM-128 × 3 seeds × 10 subjects, R3 3 draws) + 30-epoch SCORE

**R4 open gallery** (top-1 by distractor count D):

| D | SAGE | SCORE | `sage_hung` (Hungarian support) |
|---|---|---|---|
| 0 | 84.6 | 66.5 | 84.6 |
| 200 | 46.2 | 53.6 | 56.2 |
| 400 | 30.5 | 45.8 | 46.6 |
| 800 | 18.6 | 36.5 | 37.1 |
| 1654 | 11.7 | 27.3 | 27.4 |

- `sage_hung` ≥ SCORE at every D, with no cost at D = 0. Iterating it (`sage_hung_it`) adds nothing.
- The top-1 / top-2 supports (`sage_s1` 65.5, `sage_s2` 79.7 at D = 0) cost accuracy without distractors.

**R3 partial coverage.** Best SAGE variant: `sage_s1f` (top-1 support + free marginal). SCORE → `sage_s1f`:

| U | Uniform law | Dirichlet law |
|---|---|---|
| 25 | 31.9 → 30.7 | 25.8 → 24.2 |
| 50 | 35.4 → 34.5 | 34.3 → 32.6 |
| 75 | 36.1 → 36.5 | 34.5 → 33.9 |
| 100 | 35.9 → 36.2 | 34.5 → 35.4 |
| 150 | 38.8 → 39.9 | 38.0 → 38.1 |
| 200 | 39.2 → 41.2 | 38.5 → 40.4 |

Plain SAGE: 10.3 / 9.2 at U = 25. So SAGE goes from −21 to within ±2 of SCORE.

**Label-free regime switch.** Tried: \|top-2 support\| ≥ 0.9 \|G\| and n_queries > 1.2 \|top-1 support\|. It misfires at
R3 U = 100–150 (worst −6) and at R4 D = 200. NOT adopted.
- Paper framing: the deployer KNOWS whether stimuli repeat (an assumption of the paradigm, not a tuned switch).
  - distinct queries → Hungarian support;
  - repeated queries → top-1 support + free marginal.
- SCORE-stack rows (`score_orig_s3300`) are still running; the R4 claim is currently ATM-128 only.
- **19:55 SCORE stack (`score_orig_s3300`, last epoch, 10 subjects).**
  - *R4, D = 0 / 200 / 400 / 800 / 1654:*

    | Method | 0 | 200 | 400 | 800 | 1654 |
    |---|---|---|---|---|---|
    | SAGE | 68.9 | 32.6 | 20.9 | 11.9 | 6.6 |
    | SCORE | 47.2 | 35.9 | 28.9 | 22.2 | 16.1 |
    | `sage_hung` | 68.9 | 37.2 | 28.8 | 21.4 | 15.5 |

  - *R3:* `sage_s1f` is within ±1.5 of SCORE at every U (U = 25: 16.4 vs 17.4 Dirichlet, 18.7 vs 20.0 uniform).
  - *R4 claim, both stacks:* `sage_hung` wins clearly at D ≤ 200 and ties SCORE at D ≥ 400 (−0.8 to +0.8). On the
    weaker SCORE stack the high-D tie is slightly negative. Wording: "matches SCORE under heavy distractor load, while
    keeping the bijective-regime lead".

**30-epoch SCORE (`score30`, last epoch).** plain 27.65, SCORE deploy 48.6. Compare:

| Run | plain | SCORE deploy |
|---|---|---|
| `score30` (30 epochs) | 27.65 | 48.6 |
| SCORE, 50 epochs | 24.50 | 47.15 |
| paper | 29.33 | 53.23 |

The `score30` battery (SAGE on it) is running.

## 2026-10-22 16:02 — E8.4b SAGE vs SCORE across ALL encoders (R1) — headline: 110 / 110 subject-encoder pairs

From `analysis/encoder_gain_diagnostics.csv` (11 arms × 10 subjects, each arm's cross-fold SAGE params, SCORE
deployment on the same embeddings). SAGE − SCORE, mean [min]:

| Arm | Seeds | SAGE − SCORE |
|---|---|---|
| ATM-128 | 2 | +16.7 / +17.0 [10.0, 14.0] |
| ATM | 1 | +23.2 [15.0] |
| Conformer | 1 | +19.4 [10.0] |
| EEGNet | 1 | +16.4 [5.0] |
| EEGProject | 1 | +13.4 [6.0] |
| rmix | 1 | +8.6 [3.5] |
| SCORE stack | 1 | +21.8 [10.0] |
| Squeezeformer | 1 | +15.4 [8.0] |
| TSConv | 2 | +17.1 / +17.1 [10.0, 6.0] |

- SAGE > SCORE on every one of the 110 subject-encoder pairs; the smallest margin is +3.5.
- Paper: "encoder-agnostic: SAGE beats SCORE on every subject of every one of 9 encoders (11 runs)".

## 2026-10-22 16:00 — E8.9 label-free encoder diagnostic (entropic Gromov–Wasserstein) — DONE: NEGATIVE

`CVPR_paper/scripts/gw_label_free.py` → `analysis/gw_label_free.csv`.
- *Method.* EEG query cloud vs image gallery cloud, square loss on centred cosine distances, eps 0.002, 60 outer
  iterations, uniform marginals. Self-check: a rotated copy has GW ≈ 0 and its plan recovers the pairing.
- *Results* (9 encoders, Spearman vs SAGE's fraction of errors fixed, encoder / subject level):

  | Statistic | Labels? | rho | Reading |
  |---|---|---|---|
  | GW discrepancy | label-free | −0.28 / −0.29 | does NOT predict; TSConv and Conformer are the most isometric by GW, yet gain little |
  | GW-plan matching accuracy | labels needed to score it | +0.46 / +0.50 (vs SAGE top-1: +0.70 / +0.51) | relational structure alone recovers 1.4–8.4 % of pairs (chance 0.5 %; ATM-128 highest) |
  | RSA (for reference, labels) | labels | +0.87 / +0.59 | |

- *Verdict.* No label-free encoder-selection statistic found (mutual-NN fraction, effective rank, hubness, GW: all
  weak). Encoder suitability for TTA is visible only with labels, through RSA. Use it on labelled source or validation
  subjects.

## 2026-10-22 15:38 — E8.10 TTA vs training length: the zero-shot-optimal checkpoint is NOT the TTA-optimal one — QUEUED

**Trigger.** The SCORE stack has the same run under two checkpoints (R1, 10 subjects):

| Checkpoint | Zero-shot | SCORE | SAGE |
|---|---|---|---|
| best epoch (test-selected, ~ep 14) | 33.70 | 49.65 | 59.85 |
| last epoch (50) | 24.50 | 47.15 | 68.90 |

- The over-trained checkpoint loses 9 zero-shot points but gains 9 SAGE points. SCORE barely moves.
- Encoders whose selected epochs are EARLY (rmix: epochs 7–34; TSConv: 9–47) are exactly the low-SAGE-gain ones.
  ATM-128's selected epochs are mostly late: 61–95, with two outliers at 28 and 47.
- Our P-main protocol selects checkpoints by zero-shot test loss, so it may under-sell SAGE on every encoder.

**Repetition sweep** (existing batteries, SAGE − SCORE at 1/5/10/20/40/80 reps):
- ≈ 0 at ≤ 10 reps on every encoder; the edge opens at 20 and grows to +16–23 at 80.
- Exceptions: intra-subject +4 and the SCORE-stack best-epoch checkpoint +10.

**Queued.**
- *Code:* `train.py --save_epoch_checkpoints E...` saves `checkpoint_epXXX.pth`, the same (EMA-swapped) weights that are
  evaluated.
- *GPU:* arms `atm128_ep` (epochs 5–100), `tsconv_ep` (5–50) and `rmix_ep` (5–50). Same recipes and seed as the
  originals, folds 1/3/8, 9 folds. They sit after `samga30`, ahead of E8.7 / E8.8.
- *CPU:* `epoch_curve_eval.sh` gives plain / Hungarian / SCORE / SAGE / RSA / oracle per checkpoint. SAGE params are
  frozen from the base arm's 10-fold screen. Outputs `analysis/epoch_curve/<arm>_epXXX.csv`.
- *Bonus:* `score30` gives the SCORE stack's epoch-30 point (tb ≈ 14 / 30 / 50).

## 2026-10-22 15:35 — E4.6 SAGE initialised from SCORE's map — DONE (CPU): SAGE's fixed point is init-insensitive

`CVPR_paper/scripts/sage_score_init.py` → `analysis/sage_score_init.csv`. R1 cached embeddings, each arm's own
cross-fold params. SAGE's loop is started from SCORE-mapped queries (`score_fit`) instead of raw queries:

| Arm | SCORE | SAGE | SAGE from SCORE init |
|---|---|---|---|
| ATM-128 | 65.6 | 82.3 | 83.5 |
| ATM | 55.55 | 78.7 | 78.95 |
| rmix | 54.35 | 62.95 | 64.4 |
| Squeezeformer | 52.5 | 67.9 | 68.35 |
| TSConv | 50.65 | 67.7 | 67.65 |
| SCORE stack | 47.15 | 68.9 | 68.6 |

- Changes are −0.3 to +1.5 pts.
- SAGE's iterations reach the same basin from a 48 % CSLS start or a 66 % SCORE start. SCORE's one-step recovery is
  subsumed, not complementary.
- Paper: "SAGE's gain is not a better initialisation; the loop converges to the same solution from SCORE's estimate."

## 2026-10-22 15:30 — E4.5 SAGE + label-free SUPPORT ESTIMATION for R3/R4 — PILOT DONE, full run QUEUED

**Motivation.** The existing regimes show SAGE's weak spots (ATM-128, 3 seeds; SAGE vs SCORE).

| Regime | SAGE | SCORE |
|---|---|---|
| R4, D = 400 | 30.5 | 45.8 |
| R4, D = 1654 | 11.7 | 27.3 |
| R3, U ≤ 100 | below | above |

The cause is SAGE's bijective prior: balanced Sinkhorn gives every gallery item, distractors included, mass.
`sage_free` / `sage_adapt` (unbalanced marginals) help little.

**Method.** `CVPR_paper/scripts/regimes_support.py`, same queries, distractors and draw seeds as `regimes.py`. Prune
the gallery label-free, then run SAGE on the support only:

| Variant | Support |
|---|---|
| `sage_s1` / `sage_s2` | union of SCORE's top-1 / top-2 items |
| `sage_s1f` | `sage_s1` with rho = 0 |
| `sage_it1` | `sage_s1`, re-estimated twice under SAGE's map |
| `sage_hung` | the N items a rectangular Hungarian on SCORE's scores gives the N queries (assumes distinct concepts per query: exact in R4) |
| `sage_hung_it` | `sage_hung`, iterated twice |
| `sage_auto` | `sage_s1` if \|top-2 support\| < 0.9 \|G\|, else plain SAGE |

**Pilot** (atm128 s3300, subjects 1 + 3, 1 draw). R4 top-1 by distractor count D:

| D | SAGE | SCORE | `sage_hung` | `sage_hung_it` | `sage_auto` | `sage_s1` |
|---|---|---|---|---|---|---|
| 0 | 91.8 | 77.2 | 91.8 | 91.8 | 91.8 | 77.8 |
| 200 | 52.2 | 67.2 | 70.8 | 71.8 | 68.0 | 68.0 |
| 400 | 37.5 | 60.0 | 60.0 | 60.2 | 58.2 | 58.2 |
| 800 | 24.0 | 48.5 | 48.8 | 48.2 | 47.5 | 47.5 |
| 1654 | 14.0 | 34.0 | 33.5 | 34.0 | 34.0 | 34.0 |

- **`sage_hung` ≥ SCORE at every D, with no loss at D = 0.** With an exact 200 × 200 gallery the Hungarian keeps
  everything.
- **Ceiling.** The top-1 support recall falls 82.8 → 71 → 64 → 52 → 36.5 %. SAGE on the support converts nearly every
  recalled item: `sage_s1` 34.0 at a recall of 36.5. The bottleneck at high D is support recall, i.e. the
  embedding's ability to separate the true items from distractors. That caps every method.
- **Re-entry hurts.** Re-scoring the full gallery under the fitted map with top-2 re-entry lets distractors back in
  (24.2). Iterating the Hungarian adds ≤ +1.
- **R3** (multiplicity, U concepts present):
  - support pruning lifts SAGE a lot at small U: uniform U = 25 from 9.8 to 29.9 (`sage_s1`) and 34.9 (`sage_s1f`);
  - it stays ≤ SCORE (36.4) for U ≤ 100;
  - at U = 200, SAGE ≥ SCORE.
  - This is an honest limitation: with few distinct concepts, SCORE's single regularised step is more robust than
    SAGE's iterative loop.

**Queued (CPU).** Full run: atm128 s3300 / s3301 / s3302 and score_orig s3300 (`checkpoint_last`), 10 subjects,
3 draws → `final/<arm>/regimes_support/part_*/support_rows.csv`.

## 2026-10-22 15:15 — New goal (user): experiments showing SAGE's edge over SCORE, or TTA-paradigm insights (which encoders suit TTA)

### E8.6 post-hoc PCA dimension sweep — DONE (CPU): dimension is NOT a post-hoc knob

`CVPR_paper/scripts/dim_sweep_pca.py` → `analysis/dim_sweep_pca.csv`.
- *Method.* Cached R1 embeddings of 10 arms. Queries and gallery are projected on the top-k principal axes of the test
  GALLERY (label-free). Then plain / Hungarian / SCORE / SAGE, with each arm's own cross-fold params.
- *Results* (SAGE, mean of 10 subjects; ks 8/16/32/64/128/199/full):
  - Every encoder gains monotonically with k up to 64, then is flat.
  - The full space is equal or best everywhere except rmix (k = 128: 65.0 vs full 63.0).
  - SAGE needs about 64 dims: ATM-128 k = 32 → 73.4, k = 64 → 82.0, full → 82.3. Hungarian / SCORE saturate at 32–64.
- *Reading.*
  - TTA-relevant information lives in ~64 directions.
  - The 512-d encoders gain nothing from truncation. ATM-128's edge over ATM-512 (SAGE 82.3 vs 78.7) is not a
    "fewer dims" effect that PCA can mimic.
  - A training-dim sweep would be needed to test trained compactness. Not queued; low priority.

### E8.7 / E8.8 queued (GPU, ahead of the SCORE seeds, after `score30` / `samga30`)

- **E8.7 which image TARGET space suits TTA.** ATM-128 recipe with the target swapped (`run_fold.sh` arms):
  - `atm128_cliph`: CLIP ViT-H-14 final, the field's default;
  - `atm128_dino20`: DINOv2-giant L20;
  - `atm128_bigg27`: OpenCLIP bigG L27;
  - `atm128_eva35`: EVA02-E L35.
  vs InternViT-6B L28 (ours). Folds 1/3/8.
- **E8.8 source diversity.** `atm128_n3` / `atm128_n6`: ATM-128 trained on only the first 3 / 6 of the 9 source
  subjects (confound: also less data). Question: does TTA gain grow as the source shrinks, i.e. does TTA substitute for
  source diversity? Folds 1/3/8.
- **Evaluation** (`screen3_eval.sh <arm>`, CPU waiters): SAGE cross-fold screen on folds 1/3/8 (each fold's params
  chosen on the other two), then `rsa_screen_eval.py` → `analysis/rsa_screen/<arm>_cv3.csv`. These are each arm's own
  params, which matters for other target spaces: the Sinkhorn τ is score-scale dependent.
- **Like-for-like 3-fold baselines** for the same protocol: `atm128_f138` (symlinks to `atm128` s3300 folds 1/3/8),
  `atm128_rsa1`, `atm128_rsa5`.
- 8 more polling CPU workers in 63835 (waiter slots).
- *ETA:* 18 folds after ~16 short SAMGA folds → ~23:00.

## 2026-10-22 15:05 — E8.5 RSA training-loss screen — DONE: RSA is a correlate, not a reliable lever

`analysis/rsa_screen/<arm>.csv`, folds 1/3/8. SAGE uses the params frozen from `atm128_s3300`. Means over the 3 folds:

| Arm | plain | Hungarian | SCORE | SAGE | RSA | oracle (centred) |
|---|---|---|---|---|---|---|
| ATM-128 s3300 / s3301 / s3302 | 33.5 (32.3–35.0) | 69.6 | 69.4 | 87.3 (84.7–89.0) | 0.693 (0.688–0.696) | 96.1 |
| + RSA loss, λ = 1 | 33.0 | 68.2 | 69.7 | 90.2 | 0.766 | 82.8 |
| + RSA loss, λ = 5 | 32.0 | 68.2 | 65.7 | 83.2 | 0.753 | 72.8 |

Per subject, SAGE (seed mean → λ = 1 / λ = 5):

| Subject | Seed mean | λ = 1 | λ = 5 |
|---|---|---|---|
| 1 | 95.0 | 92.5 | 91.0 |
| 3 | 87.7 | 96.0 | 83.5 |
| 8 | 79.2 | 82.0 | 75.0 |

- **The loss moves its target.** RSA goes from 0.69 to 0.75–0.77, far outside the seed range. Zero-shot and Hungarian
  are unchanged.
- **SAGE does not follow.**
  - λ = 1: +2.9 over the seed mean, +1.2 over the best seed. Mixed per subject (−2.5, +8.3, +2.8). That is within
    seed noise at n = 3 folds.
  - λ = 5: −4.1.
- **The rotation ceiling falls.** The oracle in-sample Procrustes, centred, falls 96 → 83 → 73 with λ. Pushing RSA
  reshapes the cloud in a way that is NOT more isometric to the image cloud. Pearson RSA is invariant to global scale
  and offset but not to anisotropic or monotone distortions, and the loss exploits those.
- **Reading.** Across encoders, RSA predicts SAGE's gain (ρ = 0.87, E8.4). Optimising RSA directly does not deliver
  it (Goodhart). RSA marks encoders whose geometry is already near-isometric to the image space; it is not a
  training lever in this form.
- **Paper wording:** "RSA is a diagnostic for encoder selection; directly maximising it does not transfer."
- **Possible follow-up, low priority:** λ = 1 on all 10 folds, to settle +2.9 vs noise.
- **19:56 own-params check** (`screen3_eval.sh`: each arm's own 3-fold cross-validated SAGE params, same s3300 folds):

  | Arm | SAGE | Per subject (1 / 3 / 8) |
  |---|---|---|
  | baseline `atm128_f138` | 89.0 | 91.5 / 94.5 / 81.0 |
  | λ = 1 | 89.0 | 92.5 / 94.0 / 80.5 |
  | λ = 5 | 84.0 | |

  With its own params, λ = 1 equals the baseline exactly. The +2.9 above was a parameter-transfer artefact. Verdict
  confirmed: the RSA loss gives no SAGE gain. Follow-up dropped.

## 2026-10-22 12:00 — E8.5 RSA training-loss screen QUEUED FIRST; SCORE seeds demoted; 30-epoch SAMGA cut assessed

**RSA screen.** User: seeds are not needed for the first draft, so the RSA-loss screens go first.
- *Loss:* new `--rsa_lambda` (`module/loss.py` `rsa_loss`) = 1 − Pearson corr of the off-diagonal centred EEG-EEG vs
  image-image cosine matrices of the batch. Same-image pairs (`positive_mask`) are dropped: with grouped batches
  (9 subjects per image) they would dominate. The image side is detached.
- *Self-check:* 0 for identical clouds, ~0 for a rotated copy (rotation-invariant), ~1 for random clouds.
  `--relational_lambda` (RKD KL) already existed, was never run, and is not used here.
- *Arms* (`run_fold.sh`): `atm128_rsa1` (λ = 1) and `atm128_rsa5` (λ = 5), ATM-128 recipe, seed 3300, folds 1/3/8.
  These are 6 folds at the queue head; started 11:54 on 4 GPUs.
- *SCORE-seed folds:* the 4 running ones (seed 3301 folds 1–4) were killed. That is 4 deliberate failures, so
  `failed.txt` is now at 23. Partial dirs were deleted; the lines were requeued at the end.
- *Evaluation:* `CVPR_paper/scripts/rsa_screen_eval.py` gives plain, Hungarian, SCORE, SAGE, RSA and oracle in-sample
  Procrustes. SAGE uses the per-subject params frozen from `atm128_s3300`'s cross-fold screen. Outputs
  `analysis/rsa_screen/<arm>.csv`.
  - *Validated* on atm128 s3300: plain / SCORE / SAGE equal `final/atm128_s3300/r1_competitors.csv` on folds 1/3/8.
  - *Baseline (folds 1/3/8, mean).* The seed spread is the noise floor for the screen: SAGE 84.7–89.0, RSA
    0.688–0.696.

    | Seed | plain | Hungarian | SCORE | SAGE | RSA | oracle |
    |---|---|---|---|---|---|---|
    | s3300 | 35.0 | 71.7 | 71.8 | 88.2 | 0.696 | 97.8 |
    | s3301 | 32.3 | 64.3 | 66.3 | 84.7 | 0.688 | 94.2 |
    | s3302 | 33.2 | 72.8 | 70.2 | 89.0 | 0.694 | 96.2 |

**30-epoch SAMGA cut (user question).** SAMGA has no LR scheduler: stage 1 is 20 epochs, then a constant stage-2 LR.
So a `--num_epochs 30` run with the same seed follows the same trajectory, and its last epoch is epoch 30 of the
existing logs. Mean test top-1 ± SE over 10 subjects:

| Epoch | SAMGA | SCORE | `scoremp` |
|---|---|---|---|
| 15 | 29.1 | 29.3 | |
| 30 | 26.6 ± 2.2 | 26.7 ± 1.7 | 26.8 |
| 50 | 26.4 | 24.5 | |

- At 30 epochs, Standard matches their 26.22, but SCORE stays 2.6 below their 29.33.
- No epoch gives SCORE − Standard ≈ +3.1; the largest is +0.2. The epoch cut moves both rows together and cannot
  reproduce their training gain. At 15 epochs, SCORE = 29.3 (= theirs), but Standard = 29.1 (vs their 26.22).
- The adapted (SCORE-deploy) number at epoch 30 needs a 30-epoch retrain (no epoch-30 checkpoint saved).
- **12:25 QUEUED (user: "yes pls queue it").**
  - *GPU:* `samga_fold.sh` variants `score30` (full SCORE training, `--num_epochs 30`) and `samga30` (Standard anchor,
    30 epochs), seed 3300, 10 folds each. They sit right after the 6 RSA folds and ahead of the SCORE seeds.
  - *CPU waiters:*
    - `final_analyses.sh score30_orig 3300` with `FINAL_CKPT=checkpoint_last.pth` (symlink `loso/score30_orig`): full
      R1–R4 battery, including SAGE on it.
    - `score_variants_eval.py` on `score30` and `samga30` (plain + SCORE deploy).
  - 4 extra polling CPU workers were started in job 63835, because all 10 slots were held by long waiters.
  - *ETA:* RSA screen ≈ 14:30–15:00, then 20 short folds (~35 min each) ≈ 17:30, battery ≈ 19:00.

## 2026-10-22 11:45 — E3.2c closest SAMGA/SCORE run to SCORE's reported numbers (user question)

`CVPR_paper/scripts/score_variants_eval.py` (plain + SCORE deployment, R1 80 reps; it reproduces the battery's
27.0 / 49.5 on score sub-01). Outputs `analysis/score_variants/<variant>_<ckpt>.csv`. SCORE paper:
- complete pipeline: 29.33 zero-shot → 53.23 adapted;
- Standard + deployment only: 26.22 → 48.75.

**10-fold variants** (zero-shot → SCORE-adapted; gap to 29.33 / 53.23):

| Variant | Checkpoint | Zero-shot → adapted | Gap |
|---|---|---|---|
| `scoremp` | last | 27.30 → 50.00 | −2.03 / −3.23 (closest overall) |
| `scoremp` | test_best | 33.20 → 50.30 | +3.87 / −2.93 (highest adapted) |
| `samga` | last | 26.35 → 49.65 | −2.98 / −3.58 |
| `samga` | test_best | 32.40 → 49.50 | |
| `score` | test_best | 33.70 → 49.65 | |
| `score` | last | 24.50 → 47.15 | |
| `scorerec` | test_best | 31.80 → 44.40 | |
| `scorerec` | last | 24.75 → 43.10 | |

3-fold variants (`scoreg`, `scoregm`, `scorelr`, `scoret`) are all worse (adapted ≤ 45.7).

- **No run reaches 53.23.** Adapted accuracy saturates at ~50 whatever the checkpoint, even at 33–34 zero-shot. So our
  training never adds to the deployment gain; the paper's training adds +4.48.
- **The deployment step reproduces SCORE's Table 2 row:** Standard last 26.35 → 49.65, vs their 26.22 → 48.75.
- **SAGE on the same stacks (last epoch):** SAMGA 67.75 (+18.1 over SCORE deploy 49.65); full SCORE 68.90 (+21.75 over
  47.15). Against SCORE's REPORTED 53.23: +14.5 / +15.7.

## 2026-10-22 11:40 — E8.4 why SAGE gives +23 over SCORE on ATM but +9 on rmix (user question)

**Setup.** `analysis/procrustes_ceiling.csv`, scratchpad script `procrustes_ceiling.py`. It uses the R1 embedding caches
of 11 arms (9 encoders). Same dimension for the two arms in question: ATM and rmix are both 512-d.

| Encoder | Zero-shot | Hungarian (= pseudo-label precision) | SAGE | SAGE − Hungarian | Oracle in-sample Procrustes | RSA |
|---|---|---|---|---|---|---|
| ATM | 30.05 | 59.05 | 78.70 | +19.65 | 97.90 | 0.69 |
| rmix | 31.00 | 57.20 | 62.95 | +5.75 | 83.85 | 0.65 |

- **Where it is not.** Zero-shot and matching quality are equal. A single Procrustes step on the Hungarian pairs is also
  equal (62.0 vs 57.5).
- **Where it is: the GEOMETRIC step.**
  - Oracle in-sample Procrustes = an orthogonal map fit on the TRUE pairs of the same 200 test items, i.e. SAGE's
    transductive ceiling. It is 97.9 vs 83.85: even with perfect labels, a rotation aligns rmix ~14 pts worse. That is
    ~14 of the 15.75-pt SAGE difference.
  - rmix has the lowest RSA of all encoders, 0.65, tied with sqf. RSA is the agreement of the EEG–EEG and image–image
    similarity structure, i.e. how close an isometry can come.
- **Across 9 encoders (encoder means; n = 9, so indicative only):**

  | Model of SAGE top-1 | R² | LOO-RMSE |
  |---|---|---|
  | Hungarian precision alone | 0.51 | 6.8 |
  | oracle ceiling alone | 0.16 | 7.6 |
  | Hungarian + oracle | 0.64 | 7.5 |
  | **Hungarian + RSA** | **0.91** | **2.8** |

  RSA vs SAGE − Hungarian: Spearman 0.78. Zero-shot vs SAGE − Hungarian: −0.23.
- **Reading.** SAGE ≈ (quality of its matching prior) + (how isometric the EEG and image geometries are).
  - ATM is good on both.
  - rmix matches well but is not isometric to the image space, so the rotation stage adds little.
  - Conformer is the opposite: ceiling 99.2, but weak matching (48.3).
- **Caveats.**
  - 9 encoders and 2 predictors; the RSA range is narrow (0.65–0.70).
  - The oracle ceiling is not comparable across dimensions (128-d ATM-128 vs 512-d), but it is fair for ATM vs rmix.

## 2026-10-22 11:20 — P2 UNPARKED (SCORE (original) seeds 3301/3302) + protocol-parity batteries — QUEUED

**Why unparked.**
- Per-epoch curves, mean over 10 subjects, ep 5/10/15/20/30/40/50:

  | Arm | ep 5 | ep 10 | ep 15 | ep 20 | ep 30 | ep 40 | ep 50 | Peak | Mean ep 41–50 |
  |---|---|---|---|---|---|---|---|---|---|
  | Standard | 22.2 | 28.7 | 29.1 | 27.8 | 26.6 | 27.4 | 26.4 | 29.1 @ ep 15 | 26.34 |
  | Full SCORE | 24.0 | 28.9 | 29.3 | 27.1 | 26.7 | 25.9 | 24.5 | 30.8 @ ep 14 | 25.29 |
  | `scoremp` | — | — | — | — | — | — | — | 29.8 @ ep 12 | 26.70 |
  | `scorerec` | — | — | — | — | — | — | — | 29.2 @ ep 14 | 24.70 |

- Every arm peaks in stage 1 (MMD warm-up, epochs ≤ 20), then decays. The recovery term steepens the decay.
- Under the P-main checkpoint rule used for ATM-128 (test-selected), our full SCORE is +1.30 over Standard (7/10). That is
  the paper's direction.
- So the reimplementation is not broken under our protocol. Its last-epoch deficit is a late-training decay.
- That makes 3 seeds informative for the main table. The GPUs were idle.

**Queued.**
- *GPU:* 20 folds of `samga_fold.sh score {3301,3302} <sub>`. Workers:
  - 64084 (node 515) and 64085 (517), new sbatch jobs on 3080s;
  - a step in 63835 (502, 3090);
  - a nohup worker on interactive 514.
- *CPU:*
  - `score_orig_tb`: symlink `loso/score_orig_tb` → same SCORE runs. Battery with `FINAL_CKPT=checkpoint_test_best.pth`
    (protocol parity with ATM-128) for seeds 3300 / 3301 / 3302. The output dirs are `final/score_orig_tb_s*`; the
    last-epoch `final/score_orig_s*` dirs are untouched.
  - `samga_orig_intra_last`: symlink to the same intra runs. Battery with `INTRA=1 FINAL_CKPT=checkpoint_last.pth`
    (SCORE's protocol), next to the existing test-best intra battery.
  - `score_orig` last-epoch battery + 50-fit stream per new seed (the original P2 lines).
  - Then `seed_table.py --arms atm128 score_orig score_orig_tb` → `main_table_3seeds.md`.
- *ETA:* 20 folds / 4 GPUs at ~55 min ≈ 16:00, then batteries ≈ 18:00.

## 2026-10-22 11:15 — E3.2c SCORE component ablation (10 folds) — DONE: the recovery loss is the culprit

`samga_fold.sh {scoremp,scorerec} 3300 <sub>`, `competitors/score_original/{scoremp,scorerec}/seed3300`. Same SAMGA
stack and recipe as Standard. Top-1, mean of 10 subjects.

| Arm | Last epoch (SCORE's protocol) | Best epoch (test-selected) |
|---|---|---|
| SAMGA Standard | 26.35 | 32.40 |
| Full SCORE (multi-positive + recovery) | 24.50 (−1.85 ± 1.08, p = .12, 2/10 wins) | 33.70 (+1.30 ± 0.89) |
| Multi-positive only (`scoremp`) | 27.35 (+1.00 ± 0.93, p = .31, 6/10) | 33.25 (+0.85 ± 0.68) |
| Recovery only (`scorerec`) | 24.75 (−1.60 ± 0.97, p = .14, 3/10) | 31.85 (−0.55 ± 0.96) |

Paired difference vs Standard ± SE; paired t-test over the 10 subjects.

- **Multi-positive minus recovery, last epoch: +2.60 ± 0.73, p = .006.** This is the only significant contrast. Adding
  our recovery loss costs ~2.6 pts at the final epoch, whether or not the multi-positive term is present (full 24.50 ≈
  rec-only 24.75).
- **At the best epoch the penalty mostly disappears.** Full SCORE is +1.30 over Standard. The recovery term seems to
  hurt LATE training (degradation over epochs, not a lower ceiling). Last-epoch reporting, SCORE's protocol, exposes it.
- **Still short of the paper.** Even multi-positive only (27.35) is below SCORE's reported full-method 29.33 ± 0.57, so
  the multi-positive term does not close the whole gap either. The paper reports +3.11 for full SCORE over Standard; we
  get +1.00 for multi-positive alone.
- **Hypotheses, untested:**
  1. Our recovery loss (frozen-landmark Procrustes residual, λ = 0.03, k = 10, ρ = 0.1) differs from theirs in
     detail: when landmarks are recomputed, gradient flow through the Procrustes map, normalisation.
  2. Its weight should be scheduled or warmed up.
- **Status of P2 (SCORE (original) seeds 3301/3302): stays PARKED.** Rerunning seeds with a component known to cost
  ~2.6 pts would only add seeds to a known-handicapped baseline. User decision: (a) report SCORE (original) at 1 seed
  with this ablation as a disclosed reproduction caveat; (b) contact the SCORE authors for code; (c) try a recovery-loss
  fix (e.g. stop-gradient through the map, warm-up), ~3 h GPU per 10-fold variant.

## 2026-10-22 09:15 — First-draft tables and figures REGENERATED with 3 ATM-128 seeds

`draft_tables.py` → `final/draft_tables.md`. The previous 2-seed version is in the scratchpad, not kept.
`stream_figs.py` and `make_figures.py` were rerun; the ATM-128 arms now pool 15 stream parts (3 seeds).

**R1, ATM-128 (3 seeds):**

| Method | Top-1 |
|---|---|
| Zero-shot | 35.78 |
| CSLS | 48.12 |
| SATTC | 45.23 |
| Hungarian | 66.17 |
| Sinkhorn | 66.87 |
| TransCLIP | 54.98 |
| SVTL | 65.03 |
| SCORE | 66.48 |
| SAGE | 84.58 |

**R2, pooled (at arrival / strict-causal / end buffer):**

| Arm | At arrival | Strict-causal | End buffer |
|---|---|---|---|
| SAGE | 62.4 | 60.9 | 77.4 |
| SCORE (ATM-128) | 56.4 | 55.7 | 60.1 |
| SCORE (original) | 41.1 | 40.4 | 44.5 |

- R2 is within 0.3 of the 2-seed numbers. The qualitative picture is unchanged: SCORE leads prequentially at large K and
  small buffers, and SAGE overtakes late.
- R4 at D = 0 is now 84.6 for SAGE. R3 and R4 otherwise move by less than 1 pt.
- The SCORE (original) column is still 1 seed: P2 is parked pending E3.2c.

## 2026-10-22 08:55 — E8.4 11-stack encoder-gain diagnostics — DONE (weakens the 9-stack story)

`analysis/encoder_gain_diagnostics.csv` (`error_decomposition.py`). The 11 arms are 9 encoders; ATM-128 and TSConv
have 2 seeds each, averaged per encoder. ff = fraction of zero-shot errors fixed.

**Fraction fixed by SAGE, per encoder:**

| Encoder | ff |
|---|---|
| rmix | 0.47 |
| EEGNet | 0.48 |
| Squeezeformer | 0.50 |
| EEGProject | 0.50 |
| TSConv | 0.50 |
| Conformer | 0.56 |
| SCORE stack | 0.60 |
| ATM | 0.70 |
| ATM-128 | 0.74 |

Hungarian ff runs 0.26 (SCORE stack) to 0.48 (ATM-128).

**Spearman rho vs ff.** Encoder level: n = 9, two-sided p < .05 needs |rho| ≥ 0.70. Subject level: n = 110, NOT
independent (same encoders).

| Predictor | Hungarian enc / sub | Sinkhorn enc / sub | SCORE enc / sub | SAGE enc / sub |
|---|---|---|---|---|
| zero-shot nn | 0.27 / 0.40 | 0.03 / 0.36 | 0.13 / 0.28 | 0.00 / 0.41 |
| rev_nn − nn | 0.40 / 0.40 | 0.63 / 0.51 | 0.67 / 0.57 | 0.70 / 0.52 |
| rev_nn | 0.28 / 0.63 | 0.17 / 0.65 | 0.32 / 0.60 | 0.58 / 0.71 |
| colcent | 0.40 / 0.73 | 0.42 / 0.73 | 0.53 / 0.71 | 0.23 / 0.65 |
| hung_colshuf | 0.57 / 0.65 | 0.47 / 0.67 | 0.55 / 0.62 | 0.45 / 0.70 |
| RSA | 0.13 / 0.32 | 0.33 / 0.34 | 0.43 / 0.40 | 0.87 / 0.59 |
| reciprocity | −0.22 / −0.11 | −0.63 / −0.16 | −0.58 / −0.24 | −0.15 / −0.05 |
| pr_q, hub_skew | ~0 | | | |

**Verdict.**

1. *Robust.* Zero-shot accuracy does not predict the gain at the encoder level (rho 0.00–0.27). The squeezeformer
   counterexample: 36.5 zero-shot, ff_SAGE 0.50; ATM-128 35.7, 0.74.
2. *Weakened.* The 9-stack claim "rev_nn − nn predicts gain with rho 0.78–0.92" does NOT survive the two new
   encoders. It is now 0.63–0.70 for Sinkhorn / SCORE / SAGE (only SAGE reaches p < .05) and 0.40 for Hungarian.
   In the paper, call it suggestive, not established.
3. *Strongest single predictor for SAGE: RSA (rho 0.87).* But its range is tiny (0.65–0.70 across encoders), so it
   could be fragile. Do not headline it without a CI (bootstrap over subjects).
4. *Hungarian gain.* No predictor reaches significance at the encoder level. Subject-level colcent / rev_nn / column-
   shuffle null are ~0.65–0.73. The column structure (per-image offsets plus reverse retrievability) carries the
   signal, consistent with the shuffle-null result.

**09:05 robustness (done; inline script, seed 0).** 1000× bootstrap over subjects (the same resampled subjects for all
encoders), giving a 95% CI on the encoder-level rho. Also leave-one-encoder-out (LOO), the min..max of 9 values.
- RSA → ff_SAGE: +0.87 [+0.58, +0.97], LOO +0.81..+0.90. Holds up under both subject resampling and dropping any encoder.
- rev_nn − nn → ff_SAGE: +0.70 [+0.48, +0.90], LOO +0.57..+0.81. → ff_SCORE: +0.67 [+0.48, +0.87], LOO +0.52..+0.83.
  → ff_Sinkhorn: +0.63 [+0.27, +0.80].
- Hungarian: no predictor has a CI clear of 0 except rev_nn − nn (+0.40 [+0.17, +0.65], LOO down to +0.14) and colcent (+0.40 [+0.28, +0.72]).
- zero-shot nn: every CI includes 0 (SAGE +0.00 [−0.23, +0.33]).
- Caveat: the bootstrap covers subject sampling only. Encoder sampling (n = 9, 3 of them ATM-family) is covered only
  by LOO.
- Paper-safe statement: "Zero-shot accuracy does not predict calibration gain. Image-image vs EEG-EEG relational
  agreement (RSA) does, for SAGE (rho = 0.87, 9 encoders, stable under leave-one-out)."

## 2026-10-22 08:45 — E8.4 rmix R1 battery — DONE; 11-stack diagnostics pass firing

`final/rmix_s3300`. Top-1, mean of 10 subjects:

| Method | Top-1 |
|---|---|
| plain | 31.00 |
| CSLS k=3 | 38.95 |
| SATTC | 31.75 |
| TransCLIP | 47.45 |
| SVTL | 53.60 |
| SCORE | 54.35 |
| Hungarian | 57.20 |
| Sinkhorn | 58.45 |
| SAGE | 62.95 |

- Compared with Squeezeformer: rmix is 5.5 pts LOWER at zero-shot (31.0 vs 36.5), but Hungarian is HIGHER (57.2 vs
  53.0) and so are Sinkhorn and SCORE.
- So zero-shot accuracy does not even give the ORDER of two encoders after a bijective prior.
- SAGE (62.95) gains less over Hungarian on rmix (+5.8) than on any other stack (sqf +14.9, ATM-128 ~+18). Whether
  the Procrustes step helps depends on more than the Hungarian quality.
- The `error_decomposition.py` waiter (11 arms, seed 3302 not included) fires within 10 min. Output
  `analysis/encoder_gain_diagnostics.csv`.

## 2026-10-22 08:20 — E8.4 Squeezeformer R1 battery — DONE (key data point)

`final/sqf_s3300/{r1_competitors,sattc,r1_transductive}.csv`. Top-1, mean of 10 subjects:

| Method | Top-1 |
|---|---|
| plain | 36.50 |
| CSLS k=3 | 43.20 |
| SATTC | 25.15 |
| TransCLIP | 47.55 |
| SVTL | 51.95 |
| SCORE | 52.50 |
| Hungarian | 53.00 |
| Sinkhorn | 53.00 |
| SAGE | 67.90 |

- Squeezeformer matches or beats ATM-128 at zero-shot (36.50 vs 35.78, 3-seed mean). It gains far less from calibration:
  SAGE 67.90 vs 84.58, Hungarian 53.00 vs ~66, SCORE 52.50 vs 66.48.
- Fraction of zero-shot errors fixed by SAGE: 0.49 (sqf) vs 0.76 (ATM-128). This is the cleanest single
  counterexample to "calibration gain follows zero-shot accuracy" in the encoder study. Same zero-shot, a 17-point
  SAGE gap. The encoder-level statistics (reverse NN, RSA, column-centred NN) come in the 11-stack pass.

## 2026-10-22 06:45 — P4 rmix seed 3300 — GPU DONE (10/10); SCORE component ablation STARTED

- **rmix:** `loso/rmix/seed3300`. Zero-shot best top-1 is 30.95 (last epoch 29.15). Its `encoder_battery.sh` waiter
  fires next. After that: the 11-stack `error_decomposition.py` pass.
- **SCORE ablation:** `scoremp` / `scorerec`, 10 folds each, started 06:34–06:44. 4 folds are in flight and 16 are
  queued, at ~55 min per fold with 4 GPUs. Expected done ≈ 11:15, later than the 10:30 estimate because folds slow down
  under contention.

## 2026-10-22 06:15 — E2.2 3-seed main table (ATM-128, R1) — DONE

`seed_table.py --arms atm128 --seeds 3300 3301 3302` → `final/main_table_3seeds.md` (04:59). Mean ± std over seeds:

| Method | Mean ± std |
|---|---|
| plain | 35.78 ± 0.52 |
| CSLS k=3 | 48.12 ± 0.63 |
| SATTC | 45.23 ± 0.56 |
| SVTL | 65.03 ± 1.01 |
| SCORE | 66.48 ± 0.78 |
| prior-only (plain+Hungarian) | 66.17 ± 0.15 |
| SAGE | 84.58 ± 2.83 |

- SAGE wins on 10/10 subjects against every method (Wilcoxon p = 0.002 on seed-averaged accuracies).
- Harm (SAGE below plain): 0/30 subject-seed pairs.
- Per-seed SAGE is 82.30 / 83.70 / 87.75. Zero-shot is flat (35.2 / 36.2 / 35.95), so SAGE's seed spread is ~5×
  zero-shot's. Report the std, and do not quote the seed-3302 value alone.
- The table has no Hungarian / Sinkhorn / TransCLIP columns (`seed_table.py` method list). `draft_tables.py` pools all
  seeds automatically for those.

## 2026-10-22 05:54 — P4 Squeezeformer seed 3300 — GPU DONE (10/10), encoder battery firing

`loso/sqf/seed3300`, 10 folds. Zero-shot best top-1 (test-selected) 36.50; last epoch 33.30.
- This is above ATM-128 (35.80, mean of 3 seeds). It is a single seed, so the gap is within seed noise.
- It makes Squeezeformer a strong point for E8.4: zero-shot equal to or above ATM-128, so its calibration gain tests
  whether gain tracks zero-shot accuracy.
- `encoder_battery.sh sqf 3300` (waiter since 21:37, job 63835) runs the R1 battery. The 11-stack `error_decomposition.py`
  pass follows once rmix is done.

## 2026-10-22 04:46 — P1 ATM-128 seed 3302 — GPU DONE (10/10), battery + 50-fit streams running

Zero-shot best top-1 (test-selected, P-main): seed 3300 35.25, seed 3301 36.20, seed 3302 35.95. Mean 35.80 over the 3 seeds.
Last epoch: 30.95 / 31.90 / 32.00. Folds 9-10 took ~90 min each, against ~45 min earlier. GPU and CPU contention on 4 jobs.
The CPU battery (`final_analyses.sh atm128 3302`) and the 50-fit SAGE/SCORE streams fired automatically at 04:46.
The 3-seed table waits on them.

## 2026-10-22 04:20 — P3 SAMGA intra-subject (`intra.sh` verbatim, seed 2025) — GPU DONE, battery running

`samga_fold.sh intra 2025 <sub>`, 10 subjects, ~15 min each (03:40-04:18). Output
`competitors/score_original/intra/seed2025` (symlink `intra/samga_orig_intra`). SAMGA's own objective (no SCORE loss),
EEGProject, 17 channels, 60 epochs.
- Inductive: last epoch top-1 90.55 / top-5 99.00; best-epoch (test-selected) top-1 92.40. Mean of 10 subjects.
- Above our EEGProject intra bar (85.2) and our one-cycle variant (87.85).
- Battery `INTRA=1 FINAL_CKPT=checkpoint_test_best.pth final_analyses.sh samga_orig_intra 2025` started 04:17 on job 63835
  (log `logs/cvpr/final/samga_orig_intra_s2025_battery.log`). It uses test-best for parity with the ATM intra bases (P-main),
  not SCORE's last-epoch rule. Rerun with `checkpoint_last.pth` if the intra column must follow SCORE's protocol.

## 2026-10-21 22:35 — E8.4 first cross-encoder result (8 stacks; Conformer / Squeezeformer / rmix pending)

`analysis/encoder_gain_diagnostics_partial.csv` (`error_decomposition.py`; ATM-128 x2, ATM, TSConv x2, EEGNet,
EEGProject, SCORE stack). Gain = fraction of zero-shot errors fixed, (acc - plain) / (100 - plain).
- Fraction fixed: SAGE 0.73-0.75 (ATM-128), 0.70 (ATM), 0.60 (SCORE stack), 0.48-0.51 (TSConv, EEGNet, EEGProject);
  Hungarian 0.48 / 0.42 / 0.26 / 0.26-0.34; SCORE 0.47 / 0.37 / 0.30 / 0.23-0.32.
- Zero-shot top-1 barely predicts it (encoder-level Spearman 0.05-0.43). Best predictors: reverse NN minus NN
  (encoder 0.55-0.88, subject 0.43-0.66), RSA (0.40-0.83 / 0.33-0.55), column-centred NN (0.60-0.83 / 0.72-0.78),
  reverse NN (0.50-0.76 / 0.62-0.72), mutual-NN precision (0.40-0.67 / 0.47-0.62). Reciprocity weak and negative
  (-0.17 to -0.62 / -0.10 to -0.27); effective rank and hub skew do not predict.
- CORRECTION to my earlier wording: reverse NN accuracy, mutual-NN precision, RSA, reciprocity and column-centred
  NN all need the true pairing, i.e. LABELS. They are not label-free at test time. They are usable as an encoder-
  selection diagnostic on labelled SOURCE / validation subjects (not the target). Truly label-free statistics here
  (mutual-NN fraction, effective rank, hub skew) predict weakly or not at all.
- Caveats: 8 stacks, 3 of them ATM-family; subject-level points are not independent across arms.
- 23:30 + EEGConformer (9 stacks; zero-shot 26.9, reverse NN 29.1, Hungarian 48.3, SCORE 47.8, SAGE 67.1, SATTC 21.4;
  fraction fixed SAGE 0.55 / Hungarian 0.30 / SCORE 0.29). Encoder-level Spearman vs fraction fixed (Hungarian,
  Sinkhorn, SCORE, SAGE): zero-shot 0.43 / 0.12 / 0.15 / 0.33; reverse NN - NN 0.58 / 0.87 / 0.92 / 0.78; RSA 0.45 /
  0.77 / 0.82 / 0.88; column-centred NN 0.60 / 0.52 / 0.53 / 0.78; reciprocity -0.25 / -0.62 / -0.65 / -0.45.
  Subject level (n=90) unchanged in pattern (column-centred NN 0.72-0.76, reverse NN 0.62-0.72). Stable.

## 2026-10-21 20:40 — E8.4 encoder study extended: Squeezeformer and rmix to 10 folds — QUEUED (overnight, user)

- Zero-shot (P-main, test-selected, s3300) of the screened encoders: Squeezeformer 33.7 on folds 1/3/8 (= ATM-128's
  33.5 on the same folds, but SAGE only 73 / 64.5 on folds 1/3: the same equal-zero-shot / unequal-gain pattern as
  TSConv), rmix 26.0 (fold 1), REVE-6 28.8 (folds 1/3). New common encoders (10 folds): EEGNet 25.1, EEGProject 27.5,
  EEGConformer ~25.1 (8 folds).
- GPU (after ATM-128 seed 3302 and SAMGA intra, before the held SCORE seeds): `run_fold.sh sqf 3300` folds
  2,4,5,6,7,9,10 (~34 min each) and `run_fold.sh rmix 3300` folds 2-10 (~27 min each). CPU: `encoder_battery.sh sqf`
  / `rmix`, then one final `error_decomposition.py` pass over 11 encoder stacks (ATM-128 x2, ATM, TSConv x2, EEGNet,
  EEGProject, EEGConformer, Squeezeformer, rmix, SCORE stack) -> `analysis/encoder_gain_diagnostics.csv`.
- `error_decomposition.py` rewritten as the consolidated diagnostics (NN / reverse NN / mutual-NN, column-centred NN,
  Hungarian and its row/column shuffle nulls, reciprocity, RSA, participation ratios, hub skew) joined with every
  method's R1 accuracy; the invalid split-half oracle-rotation column was dropped. Reproduces the earlier numbers.

## 2026-10-21 20:20 — E1.8 / P-clean: full battery on ATM-128 ValCon (val-selected) — QUEUED (low priority, user)

`honest_ensemble/atm_iv_valcon/seed3300` (ATM bb128/fd128, 100 ep, InternViT L28, ValCon 10% concepts held out,
10 folds) = the main model's recipe with VALIDATION-selected checkpoints (`checkpoint_test_best.pth` in ValCon runs;
`checkpoint_testsel_best.pth` = test-selected). Symlink `cvpr/loso/atm128_valcon`; `final_analyses.sh atm128_valcon
3300` with FINAL_CKPT=checkpoint_test_best.pth (val-selected) at the END of the CPU queue -> `final/atm128_valcon_s3300`.
Run dir added to PROTECTED.md.

## 2026-10-21 19:50 — E3.2c why our SCORE (original) reproduction is ~5 pts low — INVESTIGATION OPENED (user)

Gap: zero-shot final epoch 24.50 vs SCORE 29.33 +- 0.57 (3 seeds; our gap ~8 of their stds); complete 47.15 vs
53.23. User: the 8-bit features are not the suspect; asked whether epoch selection explains it.
Evidence so far (sources: `CVPR_paper/related_papers/2608.19134.html.txt`, SAMGA `2604.17782v1.pdf`, code):
- Epoch: SCORE l.578 "train each model for 50 epochs, and report the final epoch" -> our protocol matches theirs.
  Our mean curve (10 folds) peaks at epoch 14 (30.75, also the min test loss) and decays to 24.50 at 50;
  per-fold min-test-loss epochs give 33.70 (SAMGA's own paper: test-selected with early stopping, 34.4). So the stack
  reaches SAMGA's level; the problem is the post-peak decay, which SCORE's final-epoch number apparently avoids.
- Data: SAMGA follows Song et al. (NICE) preprocessing (0-1000 ms, 200 ms baseline, 250 Hz, MVNN, averaged) = our
  NICE release; SCORE cites "the processed release of Wu et al. (2025)" (UBP). Not a strong suspect.
- Our SCORE-loss reimplementation (prime suspects):
  (a) batches: SAMGA's inter.sh loader shuffles all (subject, image) samples (average, not random), so a 1024 batch has
      ~0.06 same-image partners per sample -> our L_MP is ~ InfoNCE. SCORE: "the image-to-EEG term supplies
      cross-subject supervision because each image is paired with responses from multiple source subjects", and
      Table 4 gives multi-positive +2.41 -> they must batch several subjects per image.
  (b) scale: SAMGA averages the two directions, (L_ei + L_ie)/2; we SUM them (Eq. 1 as written). Adam removes the global
      scale, but in stage 1 this doubles alignment vs the MMD term.
- Decisive anchor: SAMGA "Standard" folds (seed 3300, final epoch; expect 26.22 +- 1.08) run next on the GPUs.
  If they match, the gap is in (a)/(b); if not, in the base pipeline. SCORE seeds 3301/3302 HELD at the end of the
  queue meanwhile (`OVERNIGHT_PLAN.md`).
- 20:05 diagnostics prepared and queued right after the anchor (seed 3300, folds 1/3/8, compared with our current
  `score` folds 1/3/8): `scoreg` = SCORE + image-grouped batches (SAMGA `train.py --score_grouped`: each batch = 113
  images x all 9 source subjects = 1017 samples, 146 batches/epoch; index mapping verified on CPU: every group holds
  the same (object, image) across subjects with distinct EEG), `scoregm` = scoreg + averaged L_MP
  (`--score_mp_mean`). Flags default off: queued `score`/`samga` runs unchanged.
- 20:15 user hypothesis: a lower lr peaks later, so the fixed 50-epoch budget lands nearer the peak. SCORE states no
  lr (main text l.577-584 only "We use SAMGA ... 50 epochs ... final epoch"; supplement not public). inter.sh = 50
  epochs, lr 1e-4, stage-2 5e-5 (SCORE's "50 epochs" matches inter.sh, not SAMGA's paper: 60 epochs + early stop).
  Queued `scorelr` (SCORE with lr 5e-5 / stage-2 2.5e-5), folds 1/3/8, after scoreg / scoregm. The SAMGA Standard
  anchor's per-epoch curve also tests it: does plain SAMGA peak early and decay too?
- 21:55 PRELIMINARY anchor (SAMGA "Standard", 6/10 folds: subjects 1,2,3,4,5,7): final epoch 24.83 vs our SCORE
  23.83 on the same subjects (our SCORE 10-subject mean 24.50, so these 6 are ~0.7 harder: Standard projects to ~25.5
  over 10 vs SCORE's reported 26.22 +- 1.08). Best epoch 31.17 vs 32.83. Reading: the BASE pipeline reproduces
  SCORE's "Standard" row within their std; our SCORE training adds ~-1 instead of their +3.1 -> the gap is in our
  reimplementation of SCORE's training (prime suspect: image-grouped batches, their multi-positive row +2.41).
  Per-epoch: Standard also peaks early (sub-01: 35.0 at epoch 13) and decays less (final 30.0 vs SCORE 27.0).
- 22:35 ANCHOR COMPLETE (10/10, seed 3300, final epoch): SAMGA "Standard" top-1 26.35 / top-5 57.35 vs SCORE's
  reported Standard 26.22 +- 1.08 / 57.98 +- 0.88 -> the base pipeline (code, NICE data, 8-bit features, final-epoch
  protocol) REPRODUCES SCORE's baseline row. Our SCORE training: 24.50 / 57.10 vs their 29.33 / 62.42, i.e. -1.85
  vs Standard where they report +3.11. CONCLUSION: the gap is entirely in our reimplementation of SCORE's training
  (Eqs. 1-2 / batching). Per subject (Standard, ours): 1 30.0/27.0, 2 29.0/31.5, 3 25.0/22.0, 4 22.5/21.5,
  5 19.5/16.5, 6 29.5/29.5, 7 23.0/24.5, 8 28.0/27.0, 9 16.5/15.0, 10 40.5/30.5. Best epoch: 32.40 vs 33.70.
  SCORE seeds 3301/3302 (current version) stay on HOLD; the scoreg / scoregm / scorelr diagnostics decide the fix.
- 23:00 `scoreg` (image-grouped batches) folds 1/3/8 final: 29.5 / 20.0 / 27.0 = 25.5 vs our SCORE 25.3 and
  Standard 27.7 on the same folds -> grouped batches do NOT fix it (peak moves later, epoch ~31 vs ~14).
- 23:05 NEW LEAD, temperature: inter.sh passes --softplus without --t_learnable, so SAMGA's logit scale is FIXED at
  softplus(log(1/0.07)) = log(1 + 1/0.07) = 2.727 (effective tau ~ 0.37). Our L_MP and L_rec reused it. SCORE
  defines its own losses with S_ij = q^T g / tau, tau = tau_rec = 0.07 -> scale 14.3 (5.2x sharper). The Standard
  row is unaffected (SAMGA's own loss and scale), consistent with Standard reproducing while our SCORE does not.
  Added SAMGA `train.py --score_tau` (fixed scale 1/tau for L_MP and L_rec; default 0 = old behaviour) and variants
  `scoret` (tau 0.07) and `scoretg` (tau 0.07 + grouped), folds 1/3/8, at the FRONT of the GPU queue.
- 23:55 `scoregm` (grouped + averaged L_MP) folds 1/3/8 final 30.0 / 22.0 / 24.0 = 25.3 -> no fix either. Batching
  and the loss-direction scale are ruled out. Waiting for scorelr / scoret / scoretg.
- 00:45 `scoret` (fixed tau 0.07, scale 14.3) folds 1/3/8 final 25.0 / 14.0 / 22.0 = 20.3 -> much WORSE than our
  SCORE (25.3) and Standard (27.7): SAMGA's fixed softplus scale (2.73) is the right one; temperature lead REJECTED.
  `scoretg` (tau 0.07 + grouped) killed mid-training (kill-losing-screens; its two components each failed).
  `scorelr` (lr halved) fold 1: 32.5 (Standard 30.0, ours 27.0); folds 3, 8 running. atm128 seed 3302 started.
- 01:45 `scorelr` folds 1/3/8 final 32.5 / 18.5 / 25.0 = 25.3 -> no fix. Summary of the 3-fold diagnostics (mean of
  folds 1/3/8, final epoch): Standard 27.7 | our SCORE 25.3 | grouped 25.5 | grouped + mean 25.3 | lr/2 25.3 | tau 0.07
  20.3. Noise: the same fold moves 3-6 pts between training runs (sub-03: 25.0 / 22.0 / 20.0 / 22.0 / 18.5), so a
  3-fold comparison resolves only ~5-pt effects (only tau's -5 is real).
- 10-fold paired stats: our SCORE - Standard = -1.85 +- 1.08 (se), p = 0.12 (t) / 0.12 (Wilcoxon): not significantly
  different from Standard; SCORE claims +3.11, i.e. we are ~4.6 se short of their training gain.
- 01:50 NEXT (overnight, powered): component ablation on all 10 folds, seed 3300 — `scoremp` (multi-positive only,
  SAMGA batching) and `scorerec` (SAMGA objective + recovery loss only). SCORE Table 4: + multi-positive +2.41,
  + simulated recovery +0.70; a component below Standard flags our implementation of it. These replace the 20 HELD
  SCORE seed folds, parked in `CVPR_paper/queue/overnight/parked_score_seeds.txt` (re-queue only with a fixed
  version). Their CPU waiters (score_orig 3301/3302 batteries, streams) removed; 3-seed table re-queued for ATM-128
  only. (Incident: a local pkill matched its own shell via the echo text; the remote cleanup on 502 had already run,
  the queue had emptied into workers, nothing lost.)

## 2026-10-21 19:05 — Overnight plan QUEUED (`CVPR_paper/OVERNIGHT_PLAN.md`)

ATM-128 seed 3302 (9 folds) + battery + 50-fit streams; SCORE (original) seeds 3301/3302 (20 folds) + batteries +
streams; SAMGA intra (10) + battery; 3-seed table (`final/main_table_3seeds.md`). Behind E1.6 encoders and SAMGA
"Standard" in the GPU queue.

## 2026-10-21 19:00 — FIRST-DRAFT DATA COMPLETE (R1-R4 x {SCORE (original), SCORE on ATM-128, SAGE on ATM-128})

Tables: `results/things_eeg/cvpr/final/draft_tables.md` (`CVPR_paper/scripts/draft_tables.py`). Figures:
`CVPR_paper/figures/stream_grid_{linear,log}.{pdf,png}` (`stream_figs.py`, 50 refits, buffer / at-arrival /
strict-causal), `CVPR_paper/figures/regimes.pdf` (`make_figures.py`, now with a dotted SCORE (original) line, legend
moved above the panels). ATM-128 = 2 seeds (3300, 3301), SCORE stack = seed 3300 (final epoch).
- R1: ATM-128 zero-shot 35.7 / CSLS 48.1 / SATTC 45.0 / Hungarian 66.1 / Sinkhorn 67.1 / TransCLIP 55.1 / SVTL 64.6 /
  SCORE 66.2 / SAGE 83.0. SCORE stack: 24.5 / 34.7 / 23.5 / 43.8 / 46.0 / 38.3 / 46.5 / SCORE (original) 47.2 /
  SAGE 68.9.
- R2 (pooled, at-arrival / strict-causal / end buffer): SAGE ATM-128 62.3 / 60.8 / 77.4; SCORE ATM-128 56.2 / 55.5
  / 59.8; SCORE (original) 41.1 / 40.4 / 44.5; zero-shot 40.0 (ATM-128), 31.2 (SAMGA).
- R3 (uniform, U=25 -> 200): SAGE ATM-128 9.8 -> 44.9; SCORE ATM-128 30.1 -> 39.1; SCORE (original) 19.9 -> 25.1;
  zero-shot ~25.5 (ATM-128), ~18 (SAMGA). R4 (D=0 -> 1654): SAGE 83.0 -> 11.9; SCORE ATM-128 66.2 -> 28.0;
  SCORE (original) 47.2 -> 16.1.
- Hungarian / SATTC are R1-only baselines (bijection / their 200x200 evaluation).
- Open for the paper: palette unification (SCORE orange in regimes.pdf, blue in stream_grid); SCORE (original)
  reproduction is ~5-6 pts below the paper (zero-shot 24.5 vs 29.3, complete 47.2 vs 53.2).

## 2026-10-21 15:45 — E8.4 why equal zero-shot encoders gain differently (ATM vs TSConv) — EXPLORATORY

User question: ATM-128 ~ TSConv in zero-shot, but much better with SCORE and SAGE; explanation? Cached R1 embeddings
(`emb_cache/<arm>`, 80-rep queries, seed-3300 encoding), 10 subjects; `CVPR_paper/scripts/error_decomposition.py`
-> `results/things_eeg/cvpr/analysis/error_decomposition.csv`; RSA -> `analysis/rsa_structure.csv`.

| arm | fwd top-1 | top-5 | img->query top-1 | MNN prec. | colcent | Hungarian | Sinkhorn | CSLS | SCORE | SAGE | SAGE - Sinkhorn | RSA r | PR q / g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atm128 s3300 | 35.2 | 68.0 | 45.7 | 65.6 | 38.8 | 66.2 | 67.2 | 47.5 | 65.6 | 82.3 | +15.1 | 0.698 | 13.0 / 10.7 |
| atm (fd512) | 30.1 | 63.9 | 32.5 | 51.0 | 30.1 | 59.1 | 57.9 | 41.7 | 55.6 | 78.7 | +20.9 | 0.688 | 13.1 / 9.7 |
| atm64 | 34.5 | 65.7 | 42.3 | 59.8 | 38.4 | 63.1 | — | 44.7 | 61.5 | 77.7 | — | 0.696 | 10.8 / 9.4 |
| tsconv s3300 | 35.7 | 68.7 | 33.4 | 55.4 | 28.8 | 51.8 | 52.8 | 42.5 | 50.7 | 67.7 | +15.0 | 0.663 | 14.9 / 10.7 |
| tsconv s3301 | 35.4 | 69.9 | 32.4 | 56.6 | 29.2 | 53.6 | 53.0 | 41.9 | 49.9 | 66.9 | +13.9 | 0.657 | 15.6 / 11.5 |
| atm31_valcon | 36.1 | — | 45.1 | 66.5 | 39.9 | 64.5 | — | — | — | — | — | 0.688 | 13.4 / 11.8 |

Findings so far:
- The ATM-TSConv gap is created by the BIJECTIVE ASSIGNMENT ALONE (Hungarian on raw cosines, no geometry change:
  66.2 vs 51.8 at equal top-1 AND top-5). SAGE's gain on top of Sinkhorn is the same for both (+15).
- Not explained by per-image additive offsets (hubness): column-centring helps ATM +3.6, HURTS TSConv -7.
- Not explained by per-query/per-image score scale: row/column/both z-scoring leaves Hungarian unchanged (+-3).
- Strong separators at equal forward accuracy: REVERSE retrieval (image -> query top-1 45.7 vs 33.4), mutual-NN
  precision (65.6 vs 55.4 = SCORE's landmark quality), and, more weakly, relational structure (RSA between
  query-query and image-image similarities 0.69-0.70 for every ATM vs 0.66 for TSConv); TSConv's query cloud has a
  higher effective rank (participation ratio 15 vs 13) at the same gallery rank -> extra query-only dimensions.
- Caveats: two architecture families only (all ATM variants vs TSConv): confounded with family. My split-half
  ORACLE-rotation column is invalid for 512-d encoders (100 pairs cannot determine a 512-d rotation) — not reported.
- 16:20 literature check (subagent; `CVPR_paper/notes/planted_matching_literature.md`): verified planted-matching /
  database-alignment sources; key: Dai, Cullina & Kiyavash arXiv:2307.02459 compare per-row max vs MLE (MLE needs a
  weaker signal; the gap vanishes when the gallery is much larger than the query set). No theory for correlated /
  structured score noise, none at finite n.
- 16:25 nulls (`analysis/reciprocity_iid.csv`, `analysis/shuffle_nulls.csv`):
  | arm | NN | reverse NN | Hungarian | Hung., i.i.d. resample | Hung., rows kept (cols shuffled) | Hung., cols kept | reciprocity |
  |---|---|---|---|---|---|---|---|
  | atm128 s3300 | 35.2 | 45.7 | 66.2 | 29.5 (NN 30.3) | 36.1 | 48.5 | 0.35 |
  | atm (fd512) | 30.1 | 32.5 | 59.1 | 21.4 (NN 22.3) | 31.0 | 33.5 | 0.40 |
  | atm64 | 34.5 | 42.3 | 63.1 | 27.6 | 34.8 | 44.6 | 0.35 |
  | tsconv s3300 | 35.7 | 33.4 | 51.8 | 26.2 (NN 26.2) | 36.8 | 33.6 | 0.47 |
  | tsconv s3301 | 35.4 | 32.4 | 53.6 | 26.0 | 35.9 | 33.3 | 0.45 |
  Reading: with i.i.d. entries (same diagonal / off-diagonal values) the bijection gains NOTHING; given only one
  direction's structure (the other shuffled) it adds ~1-3 pts over that direction's NN. The whole assignment gain
  (+16 TSConv, +29-31 ATM) comes from the JOINT row-column consistency of the real matrix (a low-rank geometric
  score matrix, s_ij = <q_i, g_j>). Reciprocity corr(S_ij - S_ii, S_ji - S_jj) is higher for TSConv (0.46) than ATM
  (0.35-0.40), the direction the theory predicts for a smaller gain; exact 2-cycle errors are < 0.5% for all.
- 19:15 Gaussian database-alignment model (Dai et al. setting; `analysis/gaussian_alignment_model.csv`): per subject,
  canonical correlations between the top-k PCs of the (normalised) query and image clouds, then simulate
  y = rho x + sqrt(1 - rho^2) z, inner-product scores, n = 200. k = 16: simulated NN 38.9 (ATM-128) / 35.7
  (TSConv) vs real 35.2 / 35.7; simulated Hungarian 55.8 / 49.6 vs real 66.2 / 51.8. The model reproduces a large
  MLE-over-NN gain and the ordering, but only ~6 of the 14-pt encoder gap. Caveats: in-sample CCA on 200 pairs is
  biased upward and grows with k (MI 3.5 / 6.5 / 11.7 at k 8 / 16 / 32); Gaussian, equal-weight dims. PRELIMINARY;
  needs held-out CCA (e.g. repetition blocks or train-split EEG) before any claim.
- Next (proposed to user): more encoders (sqf / rmix / reve6 checkpoints exist for folds 1/3/8; E1.6 EEGNet /
  EEGProject / Conformer queued; SAMGA-TSConv stack), all adaptation methods, a predictive model of the gain, and a
  controlled noise-structure intervention.

## 2026-10-21 15:00 — E1.6 grid encoders (EEGNet, EEGProject, EEGConformer) — QUEUED

For the base-quality analysis (E8.2) and the encoder-study contribution. The old InternViT-L28 architecture sweep
(`inter-subjects/eeg_encoder_architecture_sweep_*_20260429-221100`) has no checkpoints left (cleanup), so retrain.
- Arms `eegnet`, `eegproject`, `conformer` in `run_fold.sh`: P-main mainline recipe (SubjectMix, multi-positive,
  test-selected checkpoint, 50 epochs), seed 3300, screen folds 1/3/8 (same protocol as the rmix/sqf/reve6 screens).
  GPU queue, after the SCORE-original folds.
- CPU waiters run `tta_screen_loso.sh <arm> 3300` (cross-fold SAGE on the 3 folds) when the folds land.
- LaBraM full-FT NOT run: no working LOSO recipe (earlier transfers 21.5-31 vs 45, memory loso-track1-transfer);
  would need its own recipe search.
- 15:55 user: focus the E8.4 encoder study on these three common encoders. Upgraded from a 3-fold screen to ALL
  10 folds, moved to the FRONT of the GPU queue (behind only the running SCORE folds; SAMGA "Standard" after them).
  3-fold screen waiters replaced by `CVPR_paper/scripts/encoder_battery.sh <arm>` (10-fold cross-fold SAGE screen,
  then R1 only: r1_eval, SATTC, r1_transductive (Hungarian/Sinkhorn/TransCLIP/EM), geometry); then
  `error_decomposition.py` + reverse-retrieval / MNN / RSA diagnostics across ATM-128, ATM, TSConv, these three and
  the SAMGA-TSConv stack.

## 2026-10-21 14:30 — E4.3 result: 50-checkpoint streams, ATM-128 (2 seeds), SAGE full damping vs SCORE — DONE

`final/atm128_s330{0,1}/stream50_{sage,score}`; figures `CVPR_paper/figures/stream_grid_{linear,log}.{pdf,png}`.
Whole-stream mean of chunk accuracies (= end of the cumulative prequential curve); 10 subjects x 2 seeds x 3 orders.
@new = transductive at arrival, @prev = strict-causal, end = buffer at the last checkpoint.

| K | q | SAGE @new / @prev / end | SCORE @new / @prev / end | plain |
|---|---|---|---|---|
| 25 | 5 / 10 / 20 / 40 | 71.1/71.8/87.0, 78.3/80.5/95.3, 80.6/82.2/96.2, 81.6/80.5/96.6 | 59.8/59.8/62.6, 72.5/72.7/77.2, 78.2/76.7/85.4, 77.4/74.2/86.9 | 48.6, 57.9, 63.7, 67.2 |
| 50 | 5 / 10 / 20 / 40 | 61.1/60.3/74.9, 71.0/70.9/89.2, 74.7/74.0/92.8, 73.4/71.0/93.2 | 47.6/47.5/48.9, 62.0/61.4/65.1, 71.4/70.2/77.0, 72.6/70.0/80.7 | 35.8, 44.2, 50.1, 54.4 |
| 100 | 5 / 10 / 20 / 40 | 45.9/45.1/55.3, 62.0/60.2/76.7, 68.1/65.9/85.7, 66.8/63.0/87.1 | 35.5/35.4/35.7, 51.2/51.0/53.2, 63.0/62.2/66.8, 67.8/66.1/73.8 | 26.8, 35.1, 40.9, 43.5 |
| 150 | 5 / 10 / 20 / 40 | 38.2/37.0/44.2, 53.9/52.2/66.4, 61.8/58.9/79.7, 61.4/57.4/82.2 | 29.4/29.4/29.7, 44.1/44.1/45.8, 56.1/55.8/59.6, 62.2/61.4/67.0 | 21.5, 29.1, 34.8, 38.4 |
| 200 | 5 / 10 / 20 / 40 | 32.6/31.6/37.1, 48.2/46.2/58.0, 57.8/54.5/73.4, 57.0/52.8/76.1 | 25.4/25.2/25.6, 39.2/39.0/40.2, 51.1/51.0/53.6, 57.8/57.2/62.3 | 18.5, 25.4, 30.6, 33.9 |
| pooled | | 62.3 / 60.8 / 77.4 | 56.2 / 55.5 / 59.9 | 40.0 |

Verdict:
- End-of-stream buffer: SAGE far above SCORE everywhere (+17.5 pooled).
- Prequential: SAGE wins every q <= 20 cell. At q = 40 (only 2 queries per concept, i.e. 2K queries in the whole
  stream) and K >= 100, SCORE wins: strict-causal 66.1 vs 63.0 (K 100), 61.4 vs 57.4 (150), 57.2 vs 52.8 (200).
  SAGE's map needs many queries; SCORE's identity-regularised map saturates early.
- Strict-causal costs SAGE 1.5 pts pooled (up to 4.2 at K 200, q 40), SCORE 0.7.
- Early stream: SCORE collapses on tiny buffers (K 25, q 40, first chunk ~8 queries: 6%, and a strict-causal dip to
  ~35% around 30 s); damped SAGE stays at plain level from the first chunk.
- For the paper: report @prev as the honest online number; the R2 claim must be qualified (high-SNR, few-query
  streams favour SCORE prequentially).

## 2026-10-21 13:35 — E4.3 streaming figures: 50 checkpoints, strict-causal score, recording-time axis — QUEUED

User request: accuracy vs recording time (s), one panel per K, hue = method, tint = q, buffer and cumulative
prequential versions, linear and log x; zero-shot included; colour-blindness check.
- `stream_grid.py`: new `<arm>@prev` rows = strict-causal score (chunk decoded by the map frozen at the PREVIOUS
  checkpoint; CSLS gallery-side hub term from that fit's buffer only, `label_free.csls_ref`; plain cosine for the
  first chunk). `label_free.score_deploy` split into `score_fit` (frozen map) + CSLS; selfcheck `csls_ref(q,q)==csls`.
  Equivalence vs the old code: SCORE rows identical (max diff 0.0); SAGE identical under the same BLAS thread count.
- Finding (reproducibility): the SAGE fit on tiny buffers (48 queries, d=128) depends on the BLAS THREAD COUNT
  (map differs at 1/2/4 threads; 2 threads reproduces the 04:01 rows exactly, 4 threads moves full1_3 at t=48 from
  37.5 to 41.7). Floating-point sensitivity of an ill-posed fit, not a code change. Reruns pin 2 threads.
- `CVPR_paper/scripts/stream50.sh <arm> <seed> <sage|score> [ckpt]` (waits for 10 folds + params, then queues 5
  parts, `--n_checkpoints 50`): atm128 s3300/s3301 SAGE (`--alphas 1.0 --fullramps 1:3`) and SCORE; score_orig
  s3300/s3301 SCORE (checkpoint_last); atm64 s3300 SAGE and SCORE -> `final/<arm>_s<S>/stream50_<kind>/`.
- `CVPR_paper/scripts/stream_figs.py` -> `CVPR_paper/figures/stream_grid_{linear,log}.{pdf,png}`. Colours: lightness-
  matched one-hue ramps (OKLCH L .76/.62/.48/.34) red `#fd8e86,#e34948,#b0081d,#6d010d`, blue (reference sequential
  steps 250/400/550/700), grey. Validator: each ramp passes `--ordinal`; red vs blue at every q tint passes CVD (worst
  dE 15.0 protan at the lightest tint) and normal vision (>= 21). Grey vs the lightest/darkest red and blue tints is
  BELOW the normal-vision floor (9.7-14.2) -> zero-shot drawn thin (relief); line style = encoder (solid ATM-128,
  dashed ATM-64, dotted SAMGA).
- Preview on the OLD 25-checkpoint ATM-128 rows (no @prev yet): in recording time SCORE leads SAGE early at large K
  and high q, and SAGE overtakes late; at K=200, q=40 the cumulative prequential ends almost tied (~57 vs ~58).
  The pooled prequential average in the table hides this.

## 2026-10-21 12:45 — E4.x SCORE in R2 (streaming), all P-main arms — DONE

`stream_grid.py --score_only` (SCORE's deployment step re-fitted on the buffer at every checkpoint; same menus
25-200, reps 5/10/20/40, 3 orders, 25 checkpoints as the SAGE stream) -> `final/<arm>_s<S>/stream_score/`,
`intra_analyses/atm128_oc_s2025/stream_score/`. Mean over 10 subjects. preq = prequential (`@new`), first / end =
buffer top-1 at the first / last checkpoint. SAGE damped = full1_3 (adopted), from the existing `stream/` runs.

| arm | plain preq | SCORE preq / first / end | SAGE undamped preq / first / end | SAGE damped preq / first / end | damped - SCORE (preq) |
|---|---|---|---|---|---|
| atm128 s3300 | 39.8 | 56.8 / 42.1 / 60.2 | 58.4 / 17.9 / 78.2 | 63.0 / 41.5 / 77.6 | +6.1, 10/10, p=0.002 |
| atm128 s3301 | 40.2 | 56.2 / 41.4 / 59.5 | 57.9 / 17.9 / 77.6 | 62.4 / 42.1 / 77.1 | +6.2, 10/10, p=0.002 |
| tsconv s3300 | 36.2 | 41.3 / 31.2 / 43.9 | 43.8 / 13.4 / 58.3 | 48.0 / 34.6 / 58.9 | +6.7, 10/10, p=0.002 |
| tsconv s3301 | 36.2 | 41.6 / 32.6 / 44.2 | (SAGE stream of this seed: 5/10 subjects so far, requeued items still running) | | |
| atm (fd512) s3300 | 35.5 | 50.0 / 36.7 / 52.8 | 55.0 / 15.4 / 73.7 | 59.0 / 36.1 / 73.5 | +9.0, 10/10, p=0.002 |
| intra atm128_oc s2025 | 82.4 | 86.0 / 76.1 / 87.9 | 77.0 / 29.7 / 92.8 | 87.5 / 81.6 / 92.7 | +1.6, 10/10, p=0.002 |

Verdict: damped SAGE beats SCORE prequentially on every arm (10/10 subjects). SCORE does not collapse at small
buffers the way undamped SAGE does (its first checkpoint is at or slightly above plain on LOSO), but it plateaus far
lower (end 44-60 vs 59-78). On intra, SCORE's first checkpoint (76.1) is below plain (82.2); damped SAGE stays at 81.6.

## 2026-10-21 12:05 — E3.2b SCORE in its ORIGINAL stack (SAMGA official code + SCORE recovery-aware training) — QUEUED

User request: a "SCORE (original)" column in every setting where SCORE is reported on our best encoder (anchor to
SCORE's paper; encoder choice is a contribution). Plus SCORE in R2 (queued earlier: `stream_score/`).
- Code: `/nasbrain/p20fores/SAMGA` (github LinJiang8/SAMGA, branch `score_repro`). SCORE has no code: `score_lib.py`
  reimplements Eq. 1 (multi-positive InfoNCE) and Eq. 2 (recovery loss, lambda_rec 0.03, map fitted without gradient,
  k 10, rho 0.1, tau 0.07); selfcheck agrees with `label_free.score_deploy` (1.0). `train.py` flags `--score_mp
  --score_rec_weight`. `module/dataset.py` reads the NICE dict files. `RandomSmooth.apply_all`: vectorised frozen
  smoothing prior (the original per-sample Python loop took >25 min per subject); selfcheck = the loop to 2.4e-7, same
  RNG order.
- Features: InternViT-6B layers 20/24/28/32/36 (SAMGA's router set); 20/24/32/36 extracted by
  `CVPR_paper/scripts/extract_samga_layers.sh` (NeuroBridge venv: bitsandbytes; layer 28 re-extracted as a check).
- `CVPR_paper/scripts/samga_fold.sh <score|samga|intra> <seed> <sub>`: inter.sh verbatim + `--early_stop_patience 0`
  (SCORE reports the FINAL epoch -> `checkpoint_last.pth`) + fixed seed; intra = intra.sh verbatim (EEGProject, 17 ch,
  test-selected, early stopping = their protocol). Recovery loss needs >1 training subject, so intra SCORE (original)
  = SAMGA intra model + SCORE deployment.
- Queued (GPU): score seeds 3300, 3301; samga (standard, anchor = SCORE Table 2 "Standard") seed 3300; intra seed
  2025. Expected anchors: complete SCORE 53.23 top-1 transductive, inductive 29.33; standard SAMGA 26.22 / +recovery
  48.75.
- Evaluation: `shared.py` SAMGA adapter (TSConv/EEGProject, router global weights). Validated on a 2-epoch smoke
  checkpoint: reproduces SAMGA's own test top-1/top-5 (1.50/7.50) exactly. Fixed `_test_image_matrix` for the
  stacked [N,1,K,D] gallery and the projector dim (SAMGA's evaluate_config overwrites feature_dim).
- Symlinks `loso/score_orig`, `loso/samga_orig`, `intra/samga_orig_intra` -> `competitors/score_original/*`; CPU-queue
  waiters run the FULL battery (`final_analyses.sh`, now with `SAGE_CHECKPOINT_NAME` override, `INTRA=1` mode and
  a `stream_score/` SCORE R2 arm) when 10 folds exist. So SAGE is also evaluated on SCORE's own encoder (bonus).
- 14:45 first real fold (score, seed 3300, sub-01): final epoch top-1 27.0 / top-5 63.5 (best epoch 14: 37.0 —
  not used, SCORE reports the final epoch). Adapter on checkpoint_last reproduces 27.0 / 63.5 exactly; SCORE
  deployment on it 49.5 (SCORE's complete-pipeline average 53.23; inductive 29.33). ~45 min per fold (stage-2 epochs
  ~48 s with the recovery loss).
- 16:42 all 10 folds (seed 3300). Zero-shot top-1 FINAL epoch (SCORE's protocol) 24.50 (per subject 1-10: 27.0 31.5
  22.0 21.5 16.5 29.5 24.5 27.0 15.0 30.5) vs SCORE paper 29.33 (recovery-aware) / 26.22 (standard). Best-epoch
  (test-selected) 33.70, close to SAMGA's own 34.4. So our final-epoch reproduction is ~4.8 below their reported
  number; candidates (unverified): seed, our reimplementation of Eqs. 1-2 (no code released), 8-bit InternViT
  features, NICE preprocessing instead of SAMGA's EEG files. Report the gap honestly in the paper.
- 17:49 R1 on the SCORE stack (`final/score_orig_s3300/r1_competitors.csv`, checkpoint_last, 80-rep queries):
  zero-shot 24.50, CSLS 34.65, SVTL 46.45, SCORE complete (= SCORE (original)) 47.15 (paper 53.23, -6.1, consistent
  with the -4.8 zero-shot gap), SAGE on SCORE's own encoder 68.90 (cross-fold; oracle 72.0). SAGE beats SCORE on its
  own stack by +21.8.
- 15:05 user: seed replications (SCORE seed 3301, all 3302) and the SAMGA intra-subject run are DEFERRED to an
  overnight run, not needed for the first draft. Removed from the queues (folds not started, waiters killed);
  the exact lines are kept in `CVPR_paper/queue/deferred_overnight.txt`. Still running for the draft: SCORE seed
  3300 (10 folds) + battery + 50-fit SCORE stream; SAMGA standard seed 3300 (reproduction anchor) + battery.
- Result: pending.

## 2026-10-21 11:45 — ATM-64 (user test: does a smaller alignment space make the SAGE rotation easier?) — QUEUED

- Arm `atm64` in `CVPR_paper/scripts/run_fold.sh`: identical to atm128 (ATM, mainline recipe, SubjectMix,
  multi-positive, test-selected checkpoint) except `--eeg_backbone_dim 64 --feature_dim 64` (user: backbone too) and
  **50 epochs** (user: iterate faster). Confound: atm128 trained 100 epochs. (Folds 1-2 first started with backbone
  128 at 11:42; killed at 11:48 and requeued with backbone 64; their failed.txt rows are these kills.)
- Seed 3300, 10 LOSO folds -> `results/things_eeg/cvpr/loso/atm64/seed3300`. GPU workers 63794 (node 517) and the 3090
  of job 63766 (node 502), queued ahead of the SAMGA/SCORE-original folds.
- CPU queue waiter runs `final_analyses.sh atm64 3300` when 10 folds exist: cross-fold SAGE screen, R1 competitors,
  ablations, reps sweep, R3/R4 regimes, R2 stream with full damping arms -> `results/things_eeg/cvpr/final/atm64_s3300`,
  log `logs/cvpr/final/atm64_s3300_battery.log`.
- 14:10 all 10 folds done. Inductive best top-1 (P-main, test-selected) 34.50 (per subject 23.5 29.0 41.0 47.0 41.5
  23.5 20.5 29.5 38.5 51.0) vs ATM-128 35.25 (s3300) / 36.20 (s3301) at 100 epochs.
- 14:15 INCIDENT: the first battery run failed (`checkpoint_testsel_best.pth` not found): my `SAGE_CHECKPOINT_NAME`
  override in final_analyses.sh inherited cpu_worker.sh's legacy default. Override renamed `FINAL_CKPT` (also in
  tta_screen_loso.sh / tta_screen_intra_cvpr.sh); atm64, score_orig, samga_orig, samga_orig_intra waiters killed
  and requeued with `FINAL_CKPT`.
- R1 SAGE screen (cross-fold, `tta_screen_loso/atm64_s3300`): loso_fold_cv 77.65 (oracle 80.65, paper_default 68.05)
  vs ATM-128 s3300 82.30 (oracle 86.15). Gain over plain +43.2 vs +47.0. A 64-d alignment space does NOT make the
  rotation fit easier on R1 (confound: 50 vs 100 epochs; ATM-64 plain is only 0.75 below). R2/R3/R4: battery running.
- 14:40 DITCHED by user (R1 result negative). Battery and ATM-64 streams killed; rows already written are kept
  but not used (`final/atm64_s3300/stream50_score` complete, the rest partial).
- INCIDENT 14:40: killing the ATM-64 items made the last own worker of CPU job 63766 (node 502) find an empty queue
  and exit -> job COMPLETED -> all its srun steps died: GPU worker (SCORE fold 3300/3, requeued, partial dir
  deleted), the four SCORE-original/SAMGA battery waiters and the two score_orig stream50 drivers (requeued).
  Fix: `cpu_worker.sh` POLL=1 (sleeps instead of exiting); `cpu_batch.sh` runs polling workers + optional GPU worker
  (GPU=1). Replacement node job 63835 (3090, 28 CPU, 12 workers, 2 days).

---

## 2026-10-21 (10:16) — Seed 3302 DITCHED (user decision); batch jobs ended; 2-seed main table is final for v1

- **User decision:** drop the third seed and focus on the remaining experiments for the paper's first version.
  - Seed-3302 folds were removed from the GPU queue (backup: `/tmp/claude-265679/gpu_queue_before_ditch3302.txt`), and the running ones killed (EXIT 143).
  - Partial dirs remain under `loso/{atm128,tsconv}/seed3302/` (1–2 finished folds each). They are unused; ignore them.
- **Incident:** with the GPU queue empty, each batch GPU worker exited ("queue empty"), which ENDED batch jobs 63639 / 63565 / 63566 (10:14:03). Every step inside them died too: CPU workers, drivers, waiters.
  - Lost, all secondary: 15 in-flight seed-3301 analyses (tsconv: transductive, 5 stream, 2 regimes; atm128: reps, transductive, BCI ×2, label-eq ×2, stream part 1_2).
  - Requeued, with 3 CPU workers on the interactive job 63758 (the only allocation left).
  - **Lesson:** never empty the GPU queue while CPU work lives inside the batch jobs. Put a `sleep infinity` placeholder in the queue, or run CPU work in its own allocation.
- **Main table: 2 seeds is final for v1** (3300 + 3301; `seed_table.py`): atm128 SAGE 83.00 ± 0.99; tsconv 67.30 ± 0.57 (full table in the 10:15 entry). All main-table inputs for seed 3301 (competitors, SATTC, ablation) were complete before the incident.
- The atm fd512 s3300 battery was complete (21/21) before the incident.

## 2026-10-21 (10:15) — Morning: interactive job restarted (63645 → 63758); seed 3301 done for both encoders

- **Restart impact: none.** The old GPU worker on 514 survived (it is still training atm128 3302 fold 2). The batch-job steps (waiters, drivers, GPU workers on 515/517/523) were untouched.
- Added 5 CPU workers (2 on 515, 2 on 517, 1 on new 63758) for the 20-item CPU backlog; 523 was already at load 23/28.
- **No failures overnight:** 167 CPU items with rc=0, no GPU fold with EXIT≠0 since 03:00.
- **Seed 3301 (10 folds each; P-main, cross-fold screen):**
  - atm128: plain 36.20 → SAGE **83.70** (oracle 86.65);
  - tsconv: plain 35.35 → SAGE **66.90** (oracle 70.7).
- **Two-seed table (`seed_table.py`, seeds 3300 + 3301; std over seeds):**

| Method | atm128 | tsconv |
|---|---|---|
| plain | 35.70 ± 0.71 | 35.53 ± 0.25 |
| CSLS-3 | 48.12 ± 0.88 | 42.17 ± 0.39 |
| SATTC | (s3301 pending) | 24.57 ± 0.53 |
| SVTL | 64.62 ± 1.03 | 49.67 ± 0.39 |
| SCORE | 66.17 ± 0.81 | 50.25 ± 0.57 |
| prior only | (s3301 pending) | 52.67 ± 1.24 |
| **SAGE** | **83.00 ± 0.99** | **67.30 ± 0.57** |

  - SAGE beats every method on 10/10 subjects (seed-averaged, p=0.002). Harm 0/20 subject-seeds for both encoders.
- **Remaining:** seed 3302 (atm128: 1/10 done, 3 training; tsconv: 2/10 done, 2 training; 12 folds queued). ETA: folds ~14:30, batteries + auto 3-seed table ~16:30.

## 2026-10-21 (07:40) — Overnight status: what is done, what still runs, what I deliberately did not run

**Done and logged (P-main, cross-fold SAGE params, 10 subjects):**
- Final models, seed 3300, with full batteries:
  - atm128 (MAIN): 35.2 → 82.3;
  - tsconv: 35.7 → 67.7;
  - atm fd512: 30.05 → 78.7.
- Competitors: SCORE, SVTL, SATTC (wrapper and own conditions), transductive VLM heads, BCI baselines, prior-only control.
- Ablations, reps sweep, session calibration, carry-over (E4.5 via E6.2), label-equivalence, geometry.
- Regimes R3/R4 (fix line closed after 5 families), streaming with the new FULL damping (adopted; confirmed in 4 settings).
- Intra-subject (99.1; regimes; streaming).
- E1.7 L31 = L28.
- E8.1 pitfall table; E8.2 complementarity; E8.4 compute cost.

**Still running, fully automated:**
- seeds 3301/3302 of atm128 and tsconv (26 GPU folds, ETA ~15:00), each followed by its battery;
- the auto 3-seed main table (see the 04:50 entry);
- the rest of the atm fd512 battery.

**Deliberately NOT run (judgement; easy to queue if wanted):**
- E1.6 extra encoder grid (EEGConformer / EEGNet / EEGProject / LaBraM): the encoder-family effect is already shown by ATM vs TSConv vs SQF (E8.2).
- E1.8 ValCon vs LOSO-val (D16, "later").
- E2.3 ensembles (out of scope, D9).
- AllJoined / MEG (per instructions).

## 2026-10-21 (04:50) — Overnight automation for the remaining scheduled runs (read this first)

- **GPU queue:** 35 folds left (atm128 seeds 3301/3302, tsconv seeds 3301/3302), on 4 nodes (514, 515, 517, 523; the SATTC node is released). ETA about 15:00–16:00.
- **Waiters** (steps of job 63639; logs in `logs/cvpr/final/`):
  - `driver_atm_s3300.out`: atm fd512 battery (same-encoder 128-d vs 512-d test) after its last folds.
  - `driver_{atm128,tsconv}_s{3301,3302}.out`: full battery per seed, each with 4 CPU workers.
  - `main_table_when_done.sh`: when all 6 seed batteries (atm128, tsconv × 3 seeds) have `r1_competitors.csv`, it writes `results/things_eeg/cvpr/final/main_table.md` (`seed_table.py`: mean ± std over seeds, harm rate, Wilcoxon on seed-averaged subjects) and inserts it at the TOP of this log as an "AUTO" entry (unreviewed).
- **Checking for failures:** `grep "rc=[1-9]" CVPR_paper/queue/cpu_done.txt` and `grep "EXIT [1-9]" logs/cvpr/worker_*.out`.

## 2026-10-21 (06:35) — E1.2 atm fd512 (NeurIPS shape) s3300, 10 folds: plain 30.05 → SAGE 78.70; same-encoder dimension test

- Battery partly done (`final/atm_s3300/`). Screen: cross-fold **78.70** ± 12.37, oracle 81.6, paper default 75.65.

| Method | atm fd512 (bb1024/fd512, 50 ep) | atm128 (bb128/fd128, 100 ep) |
|---|---|---|
| plain | 30.05 | 35.20 |
| CSLS-3 | 41.65 | 47.50 |
| SATTC (wrapper) | 28.20 (< plain) | 45.45 |
| SVTL | 54.40 | 63.90 |
| SCORE | 55.55 | 65.60 |
| prior only | 59.05 | 66.20 |
| **SAGE** | **78.70** | **82.30** |
| SAGE − SCORE | +23.15 (10/10, p=0.002) | +16.70 |
| SAGE − prior only | +19.65 | +16.10 |

- **Same encoder family, different dimension:** SAGE's gain is the same (+48.65 vs +47.1). SCORE's gain is +25.5 (512-d) vs +30.4 (128-d).
  - The earlier "SCORE gains half as much on 512-d encoders" was mostly an encoder-type confound. Holding the encoder fixed, the dimension effect on SCORE is −5 points.
- The 128-d / 100-epoch recipe is better at both stages (+5.15 plain, +3.6 SAGE): it remains the main model.
- The NeurIPS-shape model under P-main clean protocol gives 78.70 cross-fold, vs the NeurIPS paper's 77.2 (test-tuned defaults): consistent.
- SATTC's calibration hurts here too (−1.85 vs plain), as on TSConv. It only helps atm128 (+10.25).

## 2026-10-21 (05:05) — E8.2 complementarity: the gain tracks SNR and encoder family, NOT weak zero-shot

- Script: `CVPR_paper/scripts/complementarity.py`. Per-subject (plain, SAGE) from all 7 inter-subject r1_eval tables (final atm128 / tsconv + 5 ValCon arms) plus intra atm128_oc; the q sweeps of the final models.

| Analysis | Spearman ρ (gain vs R0) | p |
|---|---|---|
| within model over subjects, pooled after removing model means (inter, n=70) | −0.13 | 0.28 |
| per inter model (n=10 each) | −0.48 … +0.27 | all > 0.15 |
| across inter models (model means, n=7) | −0.36 | 0.43 |
| intra atm128_oc over subjects | −0.86 | 0.002 (ceiling: gain ≤ 100 − R0) |

- **Across models:** all have R0 32–36, but gains are 44.9–48.5 for the four ATM arms and 28.1–32.0 for TSConv / SQF. The encoder family determines the gain, not zero-shot accuracy.
- **Across q (same model):** the gain grows monotonically with SNR.
  - atm128: q=5 +6.1, q=10 +15.1, q=20 +29.0, q=40 +41.1, q=80 +47.1.
  - tsconv: +1.0, +4.5, +11.5, +20.4, +32.0.
  - At q=1 the gain is ≈ 0 (−0.3 / −0.2).
- **Verdict:** "weak subjects / weak decoders benefit more" is NOT supported (as before: ρ=−0.32, p=0.36). The calibration AMPLIFIES an existing signal: it needs query SNR and an encoder whose geometry is a near-rigid transform of the gallery's.
  - The intra ρ=−0.86 is a ceiling artefact and must not be reported as complementarity.
  - Paper: the complementarity scatter (gain vs R0 by model, coloured by encoder family) plus the q curves.

## 2026-10-21 (05:00) — tsconv s3300 final battery (second encoder): same structure, three encoder-specific differences

Outputs in `results/things_eeg/cvpr/final/tsconv_s3300/`. 10 subjects, P-main, cross-fold SAGE params.

| Method | Top-1 | SAGE − method (10/10, p=0.002 for all) |
|---|---|---|
| plain | 35.70 | +32.00 |
| CSLS-3 | 42.45 | +25.25 |
| SATTC (wrapper) | **24.95** | +42.75 |
| SVTL | 49.95 | +17.75 |
| SCORE | 50.65 | +17.05 |
| prior only (plain + Hungarian) | 51.80 | +15.90 |
| **SAGE** | **67.70** | |

**Replicates the atm128 findings:**
- SCORE ≈ prior only; SAGE adds +15.9 over the prior (atm128: +16.1).
- Ablation ranking: −procrustes 40.55 < 1 step 50.0 < −csls 54.2 < −whiten 58.6 < full 67.7. Plan: softmax 62.05, Hungarian 47.05, MNN 39.4.
- Reps crossover: SAGE ≈ SCORE at q ≤ 5 (14.9 vs 13.8), then +1.6 at q=10, +5.2 at q=20, +17.1 at q=80.
- Own-session calibration: own − cross = **+3.81, 10/10, p=0.002** (37.99 vs 34.18; plain 25.71).
- Geometry: raw gap 0.161 → 0.322 after SAGE; hubness skew 1.43 → 1.02; median angle 86.7°.

**TSConv-specific:**
1. **SATTC HARMS TSConv: 24.95 vs plain 35.70, 10/10 subjects harmed** (−10.75). Their calibration is tuned for their ATM pipeline. Report it as is: the drop-in wrapper is the same code validated on their own runs.
2. **AdaBN / TENT HARM TSConv** (TSConv has BatchNorm in the temporal-spatial stem):
   - plain −4.7 (8/10 harmed, p=0.02);
   - after SAGE −6.75 (**10/10 harmed**, p=0.002).
   - On ATM, AdaBN was +2.65 plain / neutral after SAGE. BN statistics re-estimated on 200 averaged test trials are a poor estimate. Re-centering is destructive (31.8 after SAGE).
3. **Label equivalence is lower on TSConv:** n* ≈ 1000–5000 labeled pairs, against ≥ 16,540 on ATM.
   - Supervised rotation (CSLS): n=1000 61.7 (SAGE +5.98, 8/10, p=0.04); n=5000 71.5 (+3.8 over SAGE, n.s.); all 16,540 74.0 (+6.25 over SAGE, 8/10, p=0.02).
   - Labels + SAGE: 80.0 (all labels).
   - Consistent with the smaller raw true-pair gap (0.161 vs 0.325): TSConv's geometry is harder to recover label-free.
- HP ensembling gives nothing on TSConv (m=10: 68.0 vs 67.7; m=20 66.4). The atm128 gain (+1.65) is encoder-specific, so don't claim it.

**Regimes (tsconv, 10 subjects, 5 draws)**

| Cell | plain | SCORE | SAGE | SAGE free |
|---|---|---|---|---|
| R1 (D=0) | 35.7 | 50.6 | **67.7** | 62.0 |
| D=200 | 26.6 | **38.2** | 35.1 | 31.6 |
| D=800 | 16.6 | **24.2** | 13.3 | 13.2 |
| D=1654 | 11.4 | **17.8** | 7.1 | 6.8 |
| uni U=25 | **20.7** | 18.7 | 8.5 | 11.0 |
| uni U=200 | 20.4 | 23.3 | **25.8** | 25.8 |
| dir U=200 | 20.5 | 23.6 | **26.0** | 25.7 |
| MEAN (17 cells) | 21.1 | **24.8** | 21.0 | 21.1 |

- Same map as ATM. On TSConv SAGE is already below SCORE at D=200, and below plain from D=800.

**Streaming (tsconv; damping constant fixed beforehand)**

| Arm | Prequential | First checkpoint | End of stream |
|---|---|---|---|
| plain | 36.2 | 34.7 | 36.2 |
| SAGE α=1 | 43.8 | 13.4 | 58.3 |
| rotation-only ramp u/(u+3) | 46.6 | 27.6 | 58.7 |
| **full damping u/(u+3) (adopted)** | **48.0** | **34.6** | 58.9 |

- Full u/(u+3):
  - vs plain: +11.85, 10/10, p=0.002;
  - vs rotation-only: +1.45, 10/10, p=0.002;
  - vs undamped: +4.22, 10/10.
  - First checkpoint −0.15 vs plain (7/10 ≥ plain): no cold-start harm.
- **Full damping now confirmed on 4 settings** (atm31 LOSO, intra atm128_oc, atm128 final, tsconv final), always with the same constant.

## 2026-10-21 (04:40) — tsconv s3300, 10 folds: plain 35.70 → SAGE 67.70; battery started

- **E1.1 P-main plain** (best top-1): 50.5, 43.0, 27.0, 29.5, 29.5, 36.0, 32.5, 25.0, 36.0, 48.0 → **35.70**. This matches atm128's 35.25.
- **SAGE cross-fold 67.70 ± 12.55** (oracle 71.3, gap 3.6; paper default 67.7; val_subject 63.1).
  - Per subject: 83.0, 74.0, 65.0, 55.0, 69.5, 75.0, 64.0, 41.0, 69.0, 81.5.
- **Encoder inversion replicates under P-main:** equal zero-shot (35.7 vs 35.25), but SAGE gives +32.0 on TSConv vs +47.1 on ATM-128.
- **Label-free target-only selectors COLLAPSE on TSConv** (kappa / MNN / sharpness / hubness: 40.7–41.4, vs cross-fold 67.7). On atm128 they were within 1–4 points.
  - Another P2-type warning: label-free selection is encoder-dependent and can fail silently. Cross-fold selection is robust on both.
- The battery (`final/tsconv_s3300`) is queued, with 7 extra CPU workers.

## 2026-10-21 (04:35) — E8.1 pitfall table, recomputed on P-clean models (final atm128 s3300 unless noted)

| Pitfall | Assumption violated | Quantified on our pipeline | Fix |
|---|---|---|---|
| P1 test-peeking checkpoint | checkpoint chosen without target labels | zero-shot +2.7 inflation (36.0 test-sel vs 33.35 val-sel, atm31_valcon); after SAGE the clean checkpoint is ≥ (84.2 val-sel vs 82.3 test-sel; atm_iv 83.1 vs 81.25) | validation-selected checkpoint; it costs nothing after calibration |
| P2 test-tuned calibration hyper-parameters | hyper-parameters chosen without target labels | oracle 86.15 vs cross-fold 82.30: +3.85 inflation. Best label-free target-only selector (kappa agreement) 81.15; paper default 80.65 | cross-fold selection on other subjects |
| P3 bijection prior under partial coverage | every gallery item appears exactly once | SAGE − plain at U=25: −15.6 (10/10 subjects harmed, worst −24.7); U=50 −8.7 (10/10); break-even U≈100 (+2.0, 3/10 harmed); U=200 +20.2 | declare the regime; SCORE-type one-shot step off the bijection |
| P4 open gallery | no distractors | SAGE − plain at D=200 +18.0; D=800 −0.3 (5/10 harmed); D=1654 −1.6 (6/10 harmed). SCORE stays above plain at every D (27.6 vs 14.2 at 1654) | same as P3; support estimation is open (oracle support +56 on atm31) |
| P5 query-SNR confound | the same repetition count in train-time claims and deployment | SAGE − plain: q=5 +6.1, q=10 +15.1, q=20 +29.0, q=40 +41.1, q=80 +47.1. A q=80 gain quoted for a q=10 deployment overstates it 3× | report q with every number; repetitions sweep |
| P6 cold start | the calibration batch is large | streaming first refit, undamped: 17.9 vs plain 39.5 (−21.6); intra −52.5 | full damping α=u/(u+3): +1.9 over plain at the first refit, +23.2 prequential |
| P7 mean-only reporting | the mean represents subjects | D=800: mean −0.3 looks like a tie, but 5/10 subjects lose, the worst −12.0. R1: harm 0/10 | per-subject results, harm rate, paired Wilcoxon |
| P8 crediting the prior to the method | gains come from adaptation | one-to-one prior alone (plain + Hungarian) 66.2 ≈ SCORE 65.6; SAGE 82.3 = +16.1 over the prior | always report the prior-only control under R1 |

- Every number traces to an entry above: P1 (night 9 / 00:10), P2 (03:25), P3/P4/P7 (`final/atm128_s3300/regimes`), P5 (`reps.csv`), P6 (`stream/`, intra `intra_analyses/.../stream`), P8 (`ablate.csv`).

## 2026-10-21 (04:15) — E3.1 SATTC under ITS OWN conditions: DONE (job 63601; node NOT replaced, per the 4-node cap)

- Job: official code (commit 0cdfd1b), seed 3300, their LOSO protocol. Total runtime 8h19m, COMPLETED 0:0.
- Their protocol holds sub-06 out as the tune-stage validation subject (`--val_subject sub-06` default), so the final stage tests **9 subjects**.
- Final stage, last epoch (15), SATTC on, per subject:

| sub | 01 | 02 | 03 | 04 | 05 | 07 | 08 | 09 | 10 | **mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 12.0 | 12.0 | 8.5 | 9.5 | 10.5 | 10.0 | 12.0 | 10.5 | 17.5 | **11.39** |
| top-5 | 31.0 | 30.0 | 22.5 | 30.0 | 23.0 | 22.5 | 25.5 | 28.5 | 40.0 | **28.11** |

- Source: `results/things_eeg/cvpr/competitors/sattc/own_conditions_seed3300/run_sattc_loso_sattc/final_SATTC_sub-*.csv`.
- **Their paper: 14.8 top-1 (ATM-based LOSO with SATTC; base 9.2). Our reproduction: 11.4 (−3.4).**
  - Known source of the gap: their ViT-H-14 features are not distributed. I rebuilt them (`sattc_vith_features.py`, train-transform assumption unverified), so their absolute base is not exactly reproducible.
  - Single seed.
- In context:
  - Under their conditions (their encoder, recipe and ViT-H targets), the whole pipeline is at ~11–15% top-1.
  - Under OUR protocol, their calibration as a drop-in on our final embeddings (validated wrapper) gives 45.45 vs plain 35.20 (+10.25), against SAGE 82.30.
  - For the paper: report both. The "own conditions" number shows the reproduction is faithful within the expected feature gap. The wrapper number is the apples-to-apples comparison.

## 2026-10-21 (04:10) — FINAL MODEL atm128 s3300 battery, part 1 (P-main, test-selected checkpoints, cross-fold SAGE params)

Outputs are in `results/things_eeg/cvpr/final/atm128_s3300/` and logs in `logs/cvpr/final/atm128_s3300/`. 10 subjects; 80-repetition queries.

### E2.2 / E3.x main R1 table (`r1_competitors.csv`, `sattc.csv`)

| Method | Top-1 | Top-5 | SAGE − method (wins, Wilcoxon) |
|---|---|---|---|
| plain | 35.20 ± 7.90 | 68.00 | +47.10 (10/10, p=0.002) |
| CSLS-3 | 47.50 | 78.75 | +34.80 (10/10, p=0.002) |
| CSLS-10 | 48.00 | 78.90 | |
| SATTC (official calibration, validated wrapper) | 45.45 | 76.75 | |
| SVTL | 63.90 | 92.55 | +18.40 (10/10, p=0.002) |
| SCORE | 65.60 ± 9.59 | 93.05 | +16.70 (10/10, min +10.0, p=0.002) |
| **Prior only (plain + Hungarian)** | **66.20 ± 13.66** | | +16.10 |
| **SAGE** | **82.30 ± 10.36** | 93.30 | |
| SAGE + HP ensembling (top-10 cross-fold configs) | 83.95 | | +1.65 over SAGE |

- **Harm rate: 0/10** (SAGE < plain never happens).
- The "prior only" framing replicates on the final model: SCORE (65.6) ≈ one-to-one prior alone (66.2). SAGE adds +16.1 on top of the prior.
- (r1_eval's plain is 35.20 from re-encoded single-trial blocks; train.py's averaged-trial best top-1 is 35.25.)

### E2.4–E2.6 / E3.6 ablation (`ablate.csv`)

| Variant | Top-1 |
|---|---|
| full SAGE | 82.30 |
| −whiten | 78.55 |
| −csls | 74.15 |
| 1 step | 71.15 |
| −procrustes | 54.65 |
| plan softmax / top-5 / MNN / argmax / Hungarian | 72.25 / 71.75 / 60.00 / 57.70 / 65.85 |
| α 0.4 / 0.6 / 0.8 | 77.80 / 82.25 / 82.60 |
| plain + Hungarian | 66.20 |
| CSLS-3 + Hungarian | 66.20 |
| full + Hungarian | 82.10 |

- The ranking of components matches atm31: Procrustes > iteration > CSLS > whitening.
- Whitening matters less on atm128 (−3.75 vs −9.6 on atm31).

### E2.5 repetitions sweep (`reps.csv`)

| q | plain | CSLS-3 | SCORE | SAGE | SAGE − SCORE |
|---|---|---|---|---|---|
| 1 | 5.89 | 6.18 | 5.85 | 5.60 | −0.25 |
| 5 | 18.38 | 21.61 | 24.85 | 24.51 | −0.3 |
| 10 | 25.05 | 31.24 | 38.52 | 40.11 | +1.6 |
| 20 | 30.46 | 38.88 | 51.32 | 59.41 | +8.1 |
| 40 | 33.12 | 44.65 | 60.32 | 74.22 | +13.9 |
| 60 | 35.15 | 46.85 | 63.75 | 79.95 | +16.2 |
| 80 | 35.20 | 47.50 | 65.60 | 82.30 | +16.7 |

- Same crossover as atm31: SAGE ≈ SCORE at q ≤ 5, and its advantage grows with SNR. At q=1 nothing helps.

### E6.2 session calibration (`session_calib.csv`)

| Arm | Top-1 |
|---|---|
| plain (20 reps, own session) | 28.55 |
| SAGE fitted on another session | 52.69 |
| **SAGE fitted on its own session** | **56.01** |
| plain, 20 reps mixed across sessions | 30.35 |
| SAGE, 20 reps mixed across sessions | 59.04 |

- Own minus cross: **+3.33, 9/10, p=0.004** (atm31: +3.15). Session-specific calibration replicates.
- The mixed arm is better for plain too (+1.8): cross-session averaging cancels session noise. So it is an SNR reference, not a calibration arm.
- **E4.5 carry-over is answered by this experiment.** The sage_cross arm IS calibrate-then-freeze-then-deploy on fresh trials, from a different session. It keeps (52.69 − 28.55)/(56.01 − 28.55) = **88% of the gain**. No separate `run_calibrate_then_deploy` run is needed.

### E8.3 geometry (`geometry.csv`)

| Quantity | plain / raw | after SAW | after SAGE |
|---|---|---|---|
| true-pair cosine | 0.514 | 0.350 | 0.540 |
| true-pair gap | 0.325 | 0.354 | 0.537 |
| hubness skew | 2.27 | 1.53 (CSLS) | 1.05 |
| hubness max | 38 | 22.5 (CSLS) | 15.2 |

- Rotation: median angle 71.8°; 77% of planes turn more than 30°.
- corr(gain, raw gap) = −0.53: subjects with a smaller initial gap gain more. corr(gain, plain top-1) = −0.34.

### E3.5 BCI-style baselines (`bci_*.csv`)

| Adaptation | plain | CSLS-3 | + SAGE |
|---|---|---|---|
| none | 35.20 | 47.50 | 82.30 |
| AdaBN | 37.85 | 49.55 | 82.40 |
| TENT (on AdaBN) | 37.90 | 49.40 | 82.50 |
| input re-centering, no retrain | 23.05 | 29.10 | 47.30 |

- AdaBN gives +2.65 on plain; TENT adds nothing on top. After SAGE both are neutral (+0.1/+0.2; on atm31 AdaBN was −1.35).
- Re-centering stays destructive. Same conclusions as on atm31.

### E4.3 / E4.4 regimes on the final model (`regimes/`, 10 subjects, 5 draws)

| Cell | plain | CSLS-3 | SCORE | SVTL | SAGE | SAGE free |
|---|---|---|---|---|---|---|
| R1 (D=0) | 35.2 | 47.5 | 65.6 | 63.9 | **82.3** | 72.2 |
| R4 D=200 | 28.2 | 36.6 | **54.6** | 53.4 | 46.2 | 44.8 |
| R4 D=800 | 19.1 | 25.4 | **37.4** | 36.8 | 18.8 | 23.6 |
| R4 D=1654 | 14.2 | 18.4 | **27.6** | 27.5 | 12.6 | 14.8 |
| R3 uniform U=25 | 25.6 | 24.0 | **30.7** | 30.7 | 10.1 | 17.8 |
| R3 uniform U=100 | 25.4 | 27.9 | **36.9** | 36.5 | 27.3 | 33.2 |
| R3 uniform U=200 | 25.4 | 29.6 | 39.8 | 38.8 | **45.6** | 43.5 |
| R3 Dirichlet U=25 | 25.5 | 23.1 | **28.5** | 28.5 | 9.1 | 15.1 |
| R3 Dirichlet U=200 | 25.9 | 29.6 | 38.8 | 38.3 | 40.8 | **41.5** |
| MEAN (17 cells) | 25.2 | 28.5 | **38.8** | 38.3 | 29.1 | 32.8 |

- The regime map replicates exactly on the final model:
  - SAGE wins the bijection (+16.7 over SCORE) and full coverage (U=200, +5.8 uniform);
  - it loses partial coverage and open galleries, falling below plain at D=1654 and U ≤ 50.

### E4.1 / E4.2b streaming on the final model (`stream/`, 10 subjects; damping arms label-free)

| Arm | Prequential | First checkpoint | End of stream |
|---|---|---|---|
| plain | 39.8 | 39.5 | 39.8 |
| SAGE α=1 | 58.4 | 17.9 | 78.2 |
| rotation-only ramp u/(u+3) | 62.0 | 36.8 | 77.6 |
| **full damping u/(u+3) (ADOPTED default, fixed before this run)** | **63.0** | **41.5** | 77.6 |
| full u/(u+1.5) (cross-fold pick here, 10/10) | 63.0 | 41.1 | 78.4 |

- **Adopted default (full u/(u+3)) vs plain: +23.16 prequential, 10/10, p=0.002.**
  - vs rotation-only ramp: +1.00, 8/10, p=0.010.
  - vs undamped: +4.60, 10/10, p=0.002.
  - First checkpoint +1.94 over plain (9/10 ≥ plain): no cold-start harm.
- The cross-fold pick (u/(u+1.5)) gives the same prequential 62.86. The adopted constant was fixed before the final model was evaluated, so the paper reports u/(u+3) as a held-out confirmation.

### E3.4 transductive VLM baselines on the final model (`r1_transductive.csv`, cross-fold configs)

| Method | Top-1 |
|---|---|
| TransCLIP | 55.65 |
| hard k-means | 57.60 |
| hard EM-Dirichlet | 57.90 |
| EM-Dirichlet | 58.35 |
| Hungarian | 66.20 |
| Sinkhorn decode | 67.20 |
| **SAGE** | **82.30** |

- Same picture as atm31: everything sits at or below the one-to-one prior (66–67), and SAGE is +15.1 above the best.

### E5.1 / E5.2 label-equivalence on the final model (`label_equiv2_*.csv`, 3 seeds per n)

| n labeled pairs (target subject, train split) | sup. rotation, cos | sup. rotation, CSLS | sup. ridge, CSLS | sup. rotation + SAGE |
|---|---|---|---|---|
| 50 | 19.2 | 25.4 | 13.2 | 54.0 |
| 200 | 40.0 | 52.1 | 33.1 | 81.8 |
| 1000 | 58.0 | 71.6 | 57.2 | 89.7 |
| 5000 | 65.3 | 78.7 | 64.0 | 92.2 |
| 16540 (all) | 67.2 | 81.0 | 64.8 | **93.0** |
| unlabeled SAGE (reference) | | **82.3** | | |

- **Unlabeled SAGE ≥ a supervised rotation fitted on ALL 16,540 labeled pairs** (matched CSLS scoring): +1.25, 7/10, p=0.34, n.s.
  - This replicates atm31 (82.3 vs 80.6, p=0.49).
  - Label-equivalent n* ≥ 16,540 pairs, i.e. 66k single trials at 4-repetition averaging.
- **Labels + SAGE: 93.0, +10.7 over SAGE alone, 10/10, p=0.002.** Complementary: calibration on the test batch adds on top of supervised alignment.
- Few labels then SAGE (n ≤ 200) is ≤ SAGE alone: a noisy rotation is a worse start than the identity (as on atm31).

The atm128 s3300 battery is COMPLETE.

## 2026-10-21 (03:25) — atm128 seed 3300: all 10 folds DONE; final battery started

- **E1.2b P-main plain** (best top-1 = test-selected epoch): 41.5, 40.5, 32.0, 26.0, 39.0, 23.5, 27.5, 31.5, 43.0, 48.0 → **mean 35.25** (last epoch 30.95).
  - Source: `results/things_eeg/cvpr/loso/atm128/seed3300/inter_subject_summary.csv`.
  - For comparison: the ValCon atm31 test-selected checkpoint gives 36.05 and atm_iv 35.x.
- `final_analyses.sh atm128 3300` started at 03:23. The full cross-fold screen runs first, then the battery is queued.
- **E2.0 P-main main number (atm128 s3300, 10 folds, wide grid, `loso_fold_cv`): plain 35.25 → SAGE 82.30 ± 10.92 cross-fold.**
  - Oracle 86.15; val_subject 82.25; paper default 80.65.
  - Per subject (cross-fold): 91.5, 78.0, 92.0, 58.0, 85.0, 88.5, 75.5, 81.0, 78.0, 95.5.
  - Output: `tta_screen_loso/atm128_s3300/folds_1_2_3_4_5_6_7_8_9_10/`.
  - **Coincidence checked.** The cross-fold (82.30) and oracle (86.15) MEANS equal atm31_valcon's to the second decimal, but the per-subject values differ (oracle sub-04 61.5 vs 70.5, sub-10 98.0 vs 92.5).
    - I recomputed both independently from `grid_scores.csv`; the source dir in `experiment_config.json` is `cvpr/loso/atm128/seed3300`.
    - Not a loading bug.
  - The 3-fold screen (89.0 on folds 1/3/8) was optimistic: those are easy folds. Over 10 folds atm128 ties the ValCon ATM, and both clear the NeurIPS 77.2.
- tsconv s3300: 8/10 folds done (plain mean so far 36.06; fold 10 = 48.0, so the fold-10 fix works). A waiter (`logs/cvpr/final/driver_tsconv_s3300.out`) starts its battery when folds 7 and 9 land.

## 2026-10-21 (02:25) — E3.4 transductive VLM baselines on R1: DONE

- Setup: `r1_transductive.py`, atm31_valcon, 10 subjects. Each head has a small grid (optionally after SAW); the config is chosen cross-fold (mean over the other subjects). Output: `results/things_eeg/cvpr/r1_transductive/atm31_valcon.csv` (+ `_grid.csv`).

| Method | Top-1 | Top-5 |
|---|---|---|
| plain (ref) | 36.05 | |
| TransCLIP (official blocks) | 55.10 | 74.60 |
| hard k-means | 57.45 | 82.95 |
| hard EM-Dirichlet | 57.85 | 83.00 |
| EM-Dirichlet | 58.45 | 62.55 |
| Hungarian (one-shot) | 64.50 | 78.60 |
| Sinkhorn decode (one-shot) | 66.45 | 89.85 |
| SCORE (ref) | 65.20 | |
| **SAGE (ref)** | **82.30** | |

- The transductive VLM heads, which infer class priors and clusters, all land between CSLS (48) and the one-to-one prior (64.5–66.5).
- The best of them is 15.9 points below SAGE. Hungarian reproduces the "prior only" row exactly (64.5).
- So the prior explains ~29 points over plain, and SAGE's iterated alignment adds ~16–18 on top, whichever decoding exploits the prior.
- The checkpoint was fixed (CPU-only rerun, see Ops 01:25).

## 2026-10-21 (01:55) — E1.7 image-target layer, same encoder: L31 = L28 → keep L28

- `tta_screen_loso.sh atm128_l31` on folds 1/3/8. Plain 35.17 vs L28 35.00; SAGE cross-fold **89.0 vs 89.0** (per subject 91.0/97.0/79.0 vs 91.5/94.5/81.0).
- No reason to switch; L28 (the paper's target) stays. The l31 arm gets no more folds.

## 2026-10-21 (01:25) — E4.2b FULL damping: POSITIVE in both settings; ADOPTED as SAGE's streaming default

`stream_grid.py --fullramps` damps whitening, rotation and CSLS with the same label-free α = cap·u/(u+const), u = t/K (see the 00:35 entry). 10 subjects each; selection is cross-fold over subjects on prequential accuracy. Summary: `stream_summary.py`.

| Setting | Arm | Prequential | First checkpoint | End of stream |
|---|---|---|---|---|
| intra atm128_oc (plain 87.75 R1) | plain | 82.4 | 82.2 | 82.4 |
| | SAGE α=1 | 77.0 | 29.7 | 92.8 |
| | rotation-only ramp u/(u+3) (old E4.2) | 84.1 | 66.7 | 92.6 |
| | **full u/(u+3)** | **87.5** | **81.6** | 92.7 |
| | full u/(u+6) (cross-fold pick, 10/10 folds) | 88.0 | 82.2 | 92.2 |
| LOSO atm31_valcon | plain | 39.4 | 39.2 | 39.4 |
| | SAGE α=1 | 56.1 | 17.3 | 75.1 |
| | rotation-only ramp u/(u+3) | 59.6 | 35.7 | 74.9 |
| | **full u/(u+3)** (cross-fold pick, 10/10 folds) | **60.8** | **41.0** | 74.8 |

- **full u/(u+3) vs the rotation-only ramp:**
  - intra: +3.44 prequential, 10/10, p=0.002;
  - LOSO: +1.20, 9/10, p=0.02.
- **vs plain:**
  - intra: +5.12, 10/10, p=0.002;
  - LOSO: +21.42, 10/10, p=0.002.
- **vs undamped SAGE:**
  - intra: +10.5;
  - LOSO: +4.8, 10/10.
- **The cold-start harm is gone.** First checkpoint vs plain: −0.7 intra (full u/(u+6) +0.0), and +1.7 on LOSO (8/10 ≥ plain). Previously it was −15 and −3.5 with the rotation-only ramp, and −52 / −22 undamped.
- **No end-of-stream cost** (92.7 vs 92.8; 74.8 vs 75.1).
- **One constant works in both settings:**
  - LOSO cross-fold picks u/(u+3), the same constant as E4.2;
  - intra picks u/(u+6), which is only +0.5 better than u/(u+3).
- **ADOPTED:** SAGE in streaming = full damping α = u/(u+3) on all three steps. It is label-free and needs only the buffer-to-menu ratio. The method section describes this as one damping parameter that interpolates SAGE from the identity (plain cosine) to full calibration.
- The final-model battery runs full ramps 1:1.5 / 1:3 / 1:6 / 1:12 on atm128.

### Ops

- **`r1_transductive.py` failed again** (rc=1 at 00:21) with CUDA OOM: the EM-Dirichlet code moves tensors to a visible GPU, and 523's GPUs are training.
  - Requeued with `CUDA_VISIBLE_DEVICES=` (CPU only); the final battery has the same prefix.
  - Missed for an hour because the monitor did not flag it. The monitor now emits only rc≠0 lines plus fold and driver events.
- **`run_fold.sh` bug: fold 10 could never run** (EXIT 1 instantly, 01:38). Under `set -e`, `TRAIN=$(for …; do [ $s -ne $SUB ] && echo …; done)` returns the status of the loop's LAST test, which is false only when SUB=10.
  - Fixed with `; true`; tested.
  - atm128 3300 fold 10 requeued at the top of the GPU queue; it starts on the next free GPU, ~01:50. The tsconv/atm fold-10 items are covered by the same fix.
  - The final atm128 battery now starts at ~03:30 instead of ~03:15.
  - The CPU-only rerun then failed on TransCLIP's upstream `init_sigma`, which hard-codes `.cuda()`. `r1_transductive.py` now makes `Tensor.cuda` a no-op when no GPU is visible (third-party code untouched). Rerun started 01:38.
- Queued `tta_screen_loso.sh atm128_l31` on the finished folds 1/3/8 (E1.7: L31 vs L28 image targets, same encoder). The plain means are equal: 35.17 vs 35.00.

---

## 2026-10-21 (00:35) — E8.4 compute cost; E6.3 intra regimes queued

### E8.4 compute cost: DONE (`CVPR_paper/scripts/compute_cost.py`)

- Setup: atm31_valcon sub-01, cached 80-rep embeddings (200 × 128), median of 20 runs, node 514 CPU.

| Method | 1 thread | 4 threads (loaded node) |
|---|---|---|
| plain cosine | 0.2 ms | 0.2 ms |
| CSLS-3 | 0.4 ms | 0.5 ms |
| SCORE deployment | 5.7 ms | 7.3 ms |
| SAGE (16 steps) | **64.5 ms** | 87.5 ms |

- SAGE costs about 65 ms per 200-query batch on one CPU core, about 11× SCORE.
- That is negligible against acquisition (80 repetitions × 200 images). No GPU is needed.
- Extra threads don't help at this size (BLAS overhead; the node was busy).

### E6.3 intra-subject regimes: QUEUED

- atm128_oc s2025 with the intra cross-fold params:
  - `r1_eval` (SCORE / SVTL / CSLS vs SAGE, intra);
  - `regimes.py` (R3/R4, 5 parts);
  - `stream_grid` with α=1 plus the damped 1:3 arm (5 parts).
- Outputs: `results/things_eeg/cvpr/intra_analyses/atm128_oc_s2025/`. The question: do the regime conclusions hold when the base model is strong (plain 87.75)?

### Intra-subject R1 competitor table (atm128_oc s2025, `intra_analyses/.../r1_competitors.csv`): DONE

| Method | Top-1 (mean ± sd over 10) | Top-5 |
|---|---|---|
| plain | 87.75 ± 3.66 | 98.80 |
| CSLS-3 | 94.25 ± 2.10 | 99.60 |
| CSLS-10 | 92.05 | 99.45 |
| SCORE | 93.00 ± 3.07 | 99.25 |
| SVTL | 90.90 | 99.10 |
| **SAGE** | **99.10 ± 1.66** | 99.65 |

- SAGE reproduces the screen's cross-fold number exactly (99.10).
- The error rate drops from 12.25% (plain) to 0.9%, against 7.0% for SCORE (8× fewer errors).
- Intra-subject, SCORE is BELOW plain CSLS-3: its moment matching has little shift to correct, and it costs neighbours.
- SAGE is 100% on 6 of 10 subjects (1, 2, 3, 6, 7, 10); its minimum is 94.5 (sub-05). SAGE beats SCORE on every subject.

### E6.3 intra-subject regimes R3/R4 (atm128_oc s2025, 10 subjects, 5 draws): DONE. The regime map HOLDS with a strong base

| Cell | plain | CSLS-3 | SCORE | SVTL | SAGE | SAGE free | atm31 (LOSO) SAGE / SCORE |
|---|---|---|---|---|---|---|---|
| R1 (D=0) | 87.8 | 94.2 | 93.0 | 90.9 | **99.1** | 94.6 | 82.3 / 65.2 |
| R4 D=200 | 82.6 | **88.6** | 87.6 | 85.6 | 76.4 | 84.6 | 42.8 / 52.4 |
| R4 D=800 | 74.6 | **79.5** | 78.8 | 76.0 | 53.0 | 72.0 | 19.0 / 35.2 |
| R4 D=1654 | 67.0 | **72.2** | **72.2** | 69.5 | 42.8 | 63.0 | 10.2 / 26.4 |
| R3 uniform U=25 | **72.8** | 60.7 | 67.5 | 67.5 | 16.1 | 36.7 | 10.3 / 29.1 |
| R3 uniform U=100 | 72.0 | 69.9 | **75.3** | 74.6 | 46.6 | 65.3 | 25.8 / 36.5 |
| R3 uniform U=200 | 72.8 | 77.0 | **79.2** | 76.4 | 74.5 | 77.3 | 42.1 / 38.7 |
| R3 Dirichlet U=25 | **73.2** | 60.9 | 65.1 | 65.1 | 15.7 | 35.3 | 8.6 / 27.5 |
| MEAN (17 cells) | 74.3 | 73.1 | **76.3** | 74.9 | 49.3 | 65.2 | 28.0 / 37.6 |

- With a strong base, SAGE's bijection prior is still harmful off the bijection. It loses 56 points to plain at U=25 and 29 points to SCORE at D=1654.
- So the failure is not a weak-base artefact: it is the wrong prior.
- At small U, even CSLS and SCORE fall below plain intra-subject: any hubness or column-balancing correction hurts when few concepts are present.
- R3 at U=200 is full coverage, but each concept appears 1–4 times (uniform law) as distinct 10-repetition blocks, so N ≈ 500 queries. It is not the R1 bijection: multiplicity plus low-SNR queries. That explains SAGE < SCORE there too (`regimes.py:r3_stream`).
- Paper: this is the strongest evidence for the "declare your regime" message. The same pitfall table entry (P3/P4) holds both intra and inter.

### E6.3 intra-subject R2 streaming (atm128_oc s2025, 10 subjects; damping u/(u+3) chosen on atm31 LOSO, applied unchanged): DONE

- Summary script: `CVPR_paper/scripts/stream_summary.py`. It reproduces the atm31 E4.2 numbers exactly (59.6 / 35.7 / 74.9).

| Arm | Prequential | First checkpoint | End-of-stream buffer |
|---|---|---|---|
| plain | 82.4 | 82.2 | 82.4 |
| SAGE α=1 | 77.0 | 29.7 | 92.8 |
| **SAGE + damping u/(u+3)** | **84.1** | 66.7 | 92.6 |

- With a strong base, UNDAMPED SAGE streaming is a net LOSS: prequential −5.35 vs plain, 0/10 subjects, p=0.002. The cold start (29.7 vs 82.2) dominates.
- Damping transfers from inter to intra without re-tuning:
  - prequential **+1.69 vs plain, 10/10, p=0.002**;
  - +7.03 vs α=1, 10/10, p=0.002;
  - same end-of-stream gain (+10).
- The first checkpoint is still 15 points below plain, so the stronger the base, the more conservative the ramp should be. The full ramp grid (adds const 12 and 24) is queued intra (`intra_analyses/.../stream_damping`), with selection cross-fold within intra.
- **Intra ramp grid: DONE.**
  - Prequential is flat across const 3–12 (84.0–84.8); the LOSO-chosen u/(u+3) is at 84.1, 0.7 from the best.
  - BUT the first checkpoint stays at 60–69 for EVERY ramp, even u/(u+24) where α≈0 early, against plain 82.2.
  - Diagnosis (`stream_grid.py`): the ramp damps only the ROTATION (geodesic α). SAW whitening and CSLS are always applied in full, and on a tiny buffer they are the R3-like failure.
- **New arm `--fullramps cap:const`:** the same α damps all three steps:
  - whitening, via the feature blend (1−α)·q̂ + α·SAW(q)̂;
  - rotation, via the geodesic α;
  - scoring, via (1−α)·cos + α·CSLS.
  - Endpoint check on sub-01: α=1 equals `sage_a1` exactly. α=0 is within one query (≤1.6 points) of plain, the residue of reflection blocks in the geodesic.
  - Smoke: at t=16 SAGE gets 25.0, full u/(u+3) 81.25, plain 81.25.
  - Queued: 7 full ramps + ramp 1:3 + α=1 on atm128_oc intra and atm31_valcon LOSO (5 parts each). Outputs: `intra_analyses/.../stream_full`, `stream_full/atm31_valcon`.


## 2026-10-21 (00:20) — final-model battery wired; unbalanced-Sinkhorn family for R3/R4

- **`CVPR_paper/scripts/final_analyses.sh <arm> [seed]`** (new). It needs all 10 folds, then:
  1. runs the full cross-fold SAGE screen (`tta_screen_loso.sh`);
  2. queues r1_eval (plain / CSLS / SCORE / SVTL / SAGE), the SATTC wrapper, ablation, reps sweep, session calibration, geometry, r1_transductive, BCI baselines, label-equivalence, regimes, and streaming with α=1 plus the adopted damping 1:3;
  3. turns itself into a CPU worker.
  - Also chained after geometry: HP ensembling (`sage_hpens.py`, top-m cross-fold configs). r1_transductive uses its own embedding cache (`emb_cache/<t>_b`), so two jobs never write the same npz concurrently.
  - Outputs go to `results/things_eeg/cvpr/final/<arm>_s<seed>/`, logs to `logs/cvpr/final/`, all with `checkpoint_test_best.pth` (P-main).
  - A waiter step on 523 starts it for **atm128 s3300** as soon as the 10th fold lands (`logs/cvpr/final/driver_atm128_s3300.out`).
- **`regimes_fix4.py --family uot`** (new family, queued in 5 parts, output `fix4_uot_rows.csv`): unbalanced Sinkhorn, i.e. a KL column penalty instead of v4's hard cap.
  - Rows are exact; columns are scaled by (N/K / colsum)^ρ, with ρ ∈ {0.25, 0.5, 0.75, 0.9} × saw/mm × 4/16 steps.
  - ρ=1 is balanced SAGE and ρ=0 is the free plan, so this is the one-knob interpolation that v2 never tested (v2 had only the two endpoints).
  - Selfcheck passes: ρ=1 gives the same argmax as balanced Sinkhorn, and ρ=0 equals the row softmax.
  - **Bug in the first run (discarded):** I compounded the column scaling on P (`P *= (b/colsum)^ρ` every iteration). That converges to the BALANCED plan for any ρ>0, so all ρ gave identical numbers.
    - Fixed to Chizat's scaling: v = (b / Kᵀu)^ρ is recomputed from the kernel each iteration, in float64.
    - Added a selfcheck for intermediate ρ (column imbalance strictly decreases from ρ=0 to 0.5 to 1).
    - Rerun queued.
    - Rerun DONE (10 subjects): **NEGATIVE**.
      - All 10 folds pick ρ=0.25 (mm, 4 steps), at 35.2 mean vs SCORE 37.8 and the v4 cap 35.8. Per cell: D0 73.4, D200 53.8, D800 31.0, D1654 22.0, uni25 20.4, uni200 42.1, dir25 18.4, dir200 41.1.
      - Monotone in ρ: lower ρ helps small U and large D, and costs R1 (ρ=0.9 saw 16 steps: D0 81.6, mean 28.2).
- **R3/R4 fix line CLOSED** after five families (v1–v4 + UOT). The best label-free iterated variant ties or trails SCORE on average. It wins only D=200 (+1.4 to +4) and U=200.
  - Paper position unchanged: SAGE when the protocol declares a one-to-one batch; a one-shot conservative step off it; support estimation as the open problem (oracle +56 at D=1654).

### E4.3/E4.4 fix v4, capacity-constrained Sinkhorn: DONE, NEGATIVE

- atm31_valcon, 10 subjects × 2 draws; 12 cells (R4 D ∈ {0, 200, 800, 1654}; R3 uniform/Dirichlet U ∈ {25, 50, 100, 200}).
- Summary script: `CVPR_paper/scripts/fix_summary.py`.

| Method | D0 | D200 | D800 | D1654 | uni25 | uni100 | uni200 | dir25 | dir200 | MEAN (12 cells) |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 36.0 | 27.7 | 17.0 | 12.4 | 25.7 | 25.1 | 25.1 | 25.1 | 25.1 | 24.7 |
| SCORE | 65.2 | 52.4 | 35.2 | **26.4** | **31.8** | **35.8** | 38.3 | **29.0** | 37.9 | **37.8** |
| cap mm κ4 4 steps (cross-fold pick, 10/10 folds) | 71.9 | **54.0** | 31.4 | 22.4 | 22.4 | 34.1 | 41.6 | 20.2 | **41.0** | 35.8 |
| cap saw κ1 16 steps (≈ published SAGE in R1) | **81.6** | 43.0 | 21.0 | 14.9 | 16.5 | 25.5 | 41.8 | 14.2 | 38.1 | 29.7 |

- In R4 the cap almost never binds: D>0 cells are identical across κ (N<K, so cap ≥ 1, and soft column masses rarely exceed 1). It behaves like the free plan.
- The loss to SCORE comes from small U and large D, the same cells as in v2.
- Verdict: dropped. Four fix families (v1 shortlist, v2 whitening × plan, v3 SCORE support, v4 capacity) now say the same thing: off the bijection the iterated rotation fits wrong correspondences, and SCORE's one-shot step is the safer choice.

## 2026-10-21 (00:35) — intra SAGE, BCI baselines (10 subjects), transductive rerun

### E6.1 intra-subject SAGE on atm128_oc: DONE

- Wide grid, cross-fold over the other subjects' own models, test-selected checkpoints.
- Result: **plain 87.75 → SAGE 99.10 cross-fold** (oracle 99.45; paper default 97.75). Near saturation: the "ceiling / correspondence-prior demo" for the paper.
- Old intra models (epj_oc / tsc_oc / sqf_oc / epj_oc_s2026) and atm_oc are still running.

### E3.5 BCI-style label-free baselines on atm31_valcon: DONE (10 subjects; `results/things_eeg/cvpr/bci_baselines/`)

| Adaptation | plain | CSLS-3 | + SAGE |
|---|---|---|---|
| none | 36.05 | 48.05 | 82.25 |
| AdaBN | 37.90 | 48.65 | 80.90 |
| TENT (on AdaBN) | 37.90 | 48.55 | 80.90 |
| input re-centering, no retrain | 23.25 | 29.45 | 43.50 |

- Standard BCI-style label-free adaptation is worth **+1.9 zero-shot** (AdaBN). TENT adds 0 on top.
- AdaBN slightly HURTS after SAGE (−1.35).
- Test-time-only EA-style re-centering is destructive.
- vs SAGE: +46 points from the embedding-space calibration. Full EA (source re-alignment + retrain) not run.

### Ops / bug

- `r1_transductive.py` lost its whole run at the final save: I never created the output directory. Fixed with `makedirs`; requeued (embeddings cached).
- Queued `regimes_fix4.py` (capacity-constrained Sinkhorn) in 5 parts.

## 2026-10-21 (00:30) — SAGE improvement search on R1: one small win (HP ensembling); everything else negative

### `sage_plus.py` (atm31_valcon, 10 subjects; each family's own knob chosen cross-fold)

| Family | Variants (top-1) | Cross-fold |
|---|---|---|
| whitening | SAW 82.30, SAW both sides 71.20, CORAL 72.85 | 82.30 (SAW) |
| τ annealing | 1×: 82.30, 2×: 78.65, 4×: 72.05, 8×: 66.20 | 82.30 (off) |
| steps | 16: 82.30, 32: 81.80, 64: 81.65 | 82.30 |
| CSLS k | 1: 82.05, 3: 82.30, 5: 82.15, 10: 82.00 | 82.30 |
| ridge refinement after the loop | 0 / 0.03 / 0.1 / 0.3 / 1: 82.30 / 82.20 / 82.40 / 82.45 / 82.40 | 82.15 |

- The published loop sits at a local optimum of these knobs. Annealing (softassign-style) and CORAL clearly HURT.

### Label-free choice among the top-m cross-fold configs (offline, from `grid_scores.csv`)

- kappa / hubness / mutual-NN / sharpness over the top-3/5/10 configs changes results by −1.1 to +1.5 depending on the model. Not robust; dropped.

### Hyperparameter ensembling (`CVPR_paper/scripts/sage_hpens.py`): small positive

- Average the row-z-scored final score matrices of the top-m configs, ranked on the other subjects.

| m | 1 (= SAGE) | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Top-1 | 82.30 | 83.20 | 83.35 | 83.35 | **83.80** |

- +1.5 at m=20. Label-free, with no selection beyond the cross-fold ranking. A candidate "SAGE-ens" option for R1; costs 20× compute (still under 2 s on CPU).
- **User (2026-10-21):** "R1 is fine; if you can keep the advantage in the other settings that would be amazing; scheduled runs stay the priority". R1 improvement work STOPS here.

### Next: capacity-constrained correspondence for R3/R4 (`regimes_fix4.py`)

- **Diagnosis:** in R4 the query batch is still one-to-one, but onto an UNKNOWN subset of the gallery. Balanced Sinkhorn forces mass onto distractors; the free plan drops the prior.
- **Fix candidate:** capacity-constrained Sinkhorn. Rows are fully assigned; columns receive ≤ cap = κ·max(1, N/K), where N/K is observable. With κ=1 it reduces exactly to balanced SAGE when N = K·cap (R1, and R3 at full coverage).
- **Grid:** whitening saw / mm × κ ∈ {1, 1.5, 2, 4} × steps 4 / 16. One config is selected cross-fold over ALL cells, including R1.

## 2026-10-21 (00:20) — Pitfall P1 quantified: the clean checkpoint costs NOTHING after SAGE

### `pitfall_p1_valsel/` (SAGE wide grid, cross-fold, on the VALIDATION-selected checkpoint of the ValCon runs)

| Model | plain, val-selected | plain, test-selected | SAGE, val-selected | SAGE, test-selected |
|---|---|---|---|---|
| ATM L31 (atm31_valcon) | 33.35 | 36.00 | **84.20** | 82.30 |
| ATM L28 bb128 (atm_iv_valcon) | 33.40 | 36.35 | **83.10** | 81.25 |

- SAGE val-selected minus test-selected: +1.90 (7/10, Wilcoxon p = 0.15) and +1.85 (8/10, p = 0.10).
- Oracle-tuned SAGE: 85.6 / 86.05.
- **Reading:**
  - Test-peeking checkpoint selection inflates the ZERO-SHOT number by about 2.7–3.0 points (P1 confirmed).
  - After calibration the clean checkpoint is NOT worse; if anything it is slightly better. The epoch with the lowest test loss is chosen for plain retrieval, not for the post-calibration geometry.
  - So the honest protocol (validation concepts for the checkpoint, other folds for the hyperparameters) has no cost for SAGE.
- **Paper implication (flag for the user, D15):**
  - The main text can report test-selected numbers (as competitors do) AND state that the fully clean protocol gives ≥ the same SAGE accuracy (84.2 vs 82.3).
  - This is a strong honesty point. P-clean could even be the headline.
- **P2 (test-tuned hyperparameters) on the same models:** oracle 86.15 vs cross-fold 82.30 (test-selected checkpoint); 85.6 vs 84.2 (val-selected).

## 2026-10-21 (00:15) — E4.2 damping schedule: POSITIVE (SAGE improvement for streaming)

### `stream_damping/atm31_valcon` (10 subjects; the same grid as E4.1, plus 11 label-free ramps α(u) = cap·u/(u+const), u = t/K)

- **Selection:** cross-fold over subjects, on prequential accuracy averaged over K, q, orders and checkpoints. All 10 folds pick **cap = 1, const = 3**, i.e. α = u/(u+3).

| Arm | Prequential | First checkpoint | End-of-stream buffer |
|---|---|---|---|
| plain | 39.4 | 39.2 | 39.4 |
| SAGE α=1 | 56.1 | 17.3 | 75.1 |
| SAGE α=0.5 | 57.6 | 24.8 | 74.0 |
| **SAGE + damping u/(u+3)** | **59.6** | **35.7** | **74.9** |

- Damping removes most of the cold-start harm (first refit 35.7 vs 17.3; plain is 39.2) at no end-of-stream cost.
- Prequential gain over α=1: **+3.58, 10/10 subjects, Wilcoxon p = 0.002**; over α=0.5: **+2.06, 10/10, p = 0.002**.
- This replaces the old test-chosen ramp 0.8·u/(u+1.5); the cross-fold choice is more conservative early.
- **Adopt as SAGE's streaming default** (method section: budget-dependent damping, label-free).

## 2026-10-21 (00:10) — intra ATM bases, BCI baselines smoke, P1 queued

### E6.1 intra-subject ATM bases: DONE (`run_intra.sh`, seed 2025, test-selected epoch)

- Recipe: 17 occipito-parietal channels, one-cycle lr 1e-3, 50 epochs; about 1.5 min per subject on a 3080.

| Arm | sub 1–10 plain top-1 | Mean |
|---|---|---|
| atm128_oc (bb128 / fd128) | 90.5 88.5 88.5 82.5 83.5 93.5 84.0 90.5 84.5 91.5 | **87.75** |
| atm_oc (fd512) | 89.5 85.5 89.5 86.0 81.0 92.0 84.0 90.0 80.5 91.0 | 86.9 |

- These match the old intra best, EEGProject one-cycle (`epj_oc`, 87.85 two-seed).
- SAGE screens are running: `tta_screen_intra_cvpr.sh`, plus the old epj_oc / tsc_oc / sqf_oc / epj_oc_s2026 via `tta_screen_intra.sh`.

### E3.5 BCI-style baselines (`CVPR_paper/scripts/bci_baselines.py`), sub-01 smoke; 10 subjects queued

| Adaptation | plain | CSLS-3 | +SAGE |
|---|---|---|---|
| none | 42.0 | 57.0 | 89.5 |
| AdaBN (target-batch BN stats) | 42.5 | 56.0 | 89.5 |
| TENT (BN affine, entropy, 10 steps) | 43.0 | 55.5 | 89.5 |
| input re-centering to the source mean (no retrain) | 22.5 | 27.0 | 32.0 |

- AdaBN and TENT have about zero effect: the encoder's BatchNorms are not where the subject shift lives (or they are already well normalised by SubjectMix training).
- Test-time-only Euclidean-alignment re-centering is destructive without source re-alignment and retraining. Full EA (retrain) is not run, since it needs about 10 GPU-hours.

### Pitfall P1 (test-peeking checkpoint) queued

- SAGE wide screen on the VALIDATION-selected checkpoint (`checkpoint_test_best.pth` in ValCon runs) of atm31_valcon and atm_iv_valcon.
- The comparison is the test-selected 82.30 / 81.25.

## 2026-10-21 (00:00) — E5.1/E5.2 label-equivalence DONE

### `label_equiv2.py` on atm31_valcon (10 subjects × 3 label draws; `results/things_eeg/cvpr/label_equiv2/atm31_valcon_*.csv`)

- **Setup.**
  - Labels: n labeled TRAIN-split pairs of the TARGET subject (4-repetition averages; concepts disjoint from the test set).
  - Test: the 200 80-repetition test queries.
  - Supervised maps are scored with the same CSLS as SAGE (matched scoring).

| n labeled | 50 | 200 | 1000 | 5000 | 16540 (all) |
|---|---|---|---|---|---|
| supervised rotation, cosine | 18.4 | 38.6 | 57.4 | 65.5 | 67.6 |
| supervised rotation, CSLS | 24.2 | 50.6 | 71.1 | 78.7 | **80.6** |
| ridge map, CSLS | 12.8 | 33.9 | 57.0 | 63.4 | 65.2 |
| **labels, then SAGE on the test batch (E5.2)** | 46.5 | 78.7 | 87.0 | 90.0 | **90.6** |
| unlabeled SAGE (reference) | 82.3 | 82.3 | 82.3 | 82.3 | 82.3 |

- References: plain 36.0, CSLS-3 48.0.
- **Headline:** label-free SAGE on the 200 unlabeled test queries **ties a supervised orthogonal map fitted on the subject's ENTIRE labeled training session** (16,540 pairs, i.e. 66,160 trials): +1.65 for SAGE, 6/10 subjects, Wilcoxon p = 0.49. At n = 5000 SAGE is +3.6 (7/10, p = 0.11). So n* ≥ 16,540 labeled trials for this map class.
- **Labels and SAGE are complementary:** supervised + SAGE = 90.6 (+8.35 over SAGE, 10/10 subjects). With few labels (n ≤ 200) the noisy supervised rotation is a worse starting point than the identity (78.7 < 82.3).
- **Caveats to state:**
  - The supervised baselines are linear/orthogonal maps (the same map class as SAGE). Encoder fine-tuning on target labels was NOT run.
  - Labels are 4-repetition averages, which is realistic for a calibration session but lower SNR than the test queries.
  - SAGE is transductive on the test batch; the supervised map is not (except for the CSLS re-ranking, now matched).

## 2026-10-20 (night, 9) — R2 streaming grid DONE; damping schedule launched

### E4.1 streaming grid on atm31_valcon: DONE (10 subjects × 3 stream orders; `results/things_eeg/cvpr/stream_grid/atm31_valcon/stream_rows.csv`)

- **Setup.**
  - A fixed menu of K images; trials arrive one at a time as q-repetition averages, 80/q per concept.
  - SAGE is refitted at 25 checkpoints, with cross-fold R1 parameters.
  - "Buffer" = the whole buffer scored at the end of the stream (transductive, D2).
  - "Prequential" = each trial scored by the first fit that includes it, averaged over the stream (D17).

| q | K | plain | buffer α=0.5 | buffer α=1 | prequential α=0.5 | prequential α=1 |
|---|---|---|---|---|---|---|
| 5 | 25 | 46.2 | 79.9 | 81.6 | 67.3 | 67.4 |
| 5 | 200 | 18.1 | 32.7 | 33.2 | 29.4 | 28.9 |
| 10 | 25 | 54.8 | 91.3 | 92.0 | 72.4 | 71.8 |
| 10 | 100 | 34.6 | 70.7 | 73.3 | 57.6 | 57.7 |
| 10 | 200 | 25.0 | 52.3 | 53.8 | 45.0 | 44.8 |
| 20 | 200 | 30.6 | 68.7 | 69.9 | 55.5 | 54.1 |
| 40 | 25 | 63.0 | 96.0 | 96.0 | 66.9 | 63.0 |
| 40 | 200 | 34.3 | 78.5 | 78.6 | 57.7 | 54.3 |

- Large gains everywhere, including prequentially: what the user sees at arrival time is +10 to +23 points over plain.
- **Cold start is the cost.** The first refit is below plain by up to −58 points (q=40, K=25, a 2-trial buffer). SAGE overtakes plain after 4–40% of the stream; the crossover comes later for small menus and high q, where the buffer holds few trials at the first checkpoints.
- **α=0.5 halves the cold-start harm and wins prequentially** at q ≥ 20 (q=40, K=200: 57.7 vs 54.3), while α=1 wins slightly on the final buffer.
- This motivates E4.2, a budget-dependent damping schedule.
- Note: the stream eventually covers all K concepts with 80/q copies each, i.e. the R3 full-coverage case, where SAGE is in its regime. That is consistent with the regime map.

### E4.2 damping schedule: launched

- `stream_grid.py --ramps cap:const` adds label-free arms with α(u) = cap·u/(u+const), where u = t/K is the buffer-to-menu ratio.
- 11 (cap, const) pairs with cap ∈ {0.6, 0.8, 1} and const ∈ {0.5 … 6}, plus the fixed α ∈ {0.5, 1}.
- (cap, const) will be selected cross-fold on prequential accuracy. This replaces the old test-chosen 0.8·u/(u+1.5).
- 5 parts: `results/things_eeg/cvpr/stream_damping/atm31_valcon/part_*`.

## 2026-10-20 (night, 8) — fix search concluded; label-equivalence (matched scoring); last screens

### E4.3/E4.4 fix search, full 10 subjects × 2 draws: NO label-free fix beats SCORE off the bijection

Outputs: `regimes_fix/atm31_valcon/part_*/fix_rows.csv` (v2: 54 variants) and `regimes_fix3/.../fix3_rows.csv` (v3). Columns are R4 D ∈ {0, 200, 800, 1654} and R3 uniform/Dirichlet U. The mean is over 11 cells.

| Method | D=0 | D=200 | D=800 | D=1654 | uni25 | uni100 | uni200 | dir25 | dir200 | MEAN |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 36.0 | 27.7 | 17.0 | 12.4 | 25.7 | 25.1 | 25.1 | 25.1 | 25.1 | 24.5 |
| SCORE | 65.2 | 52.4 | 35.2 | 26.4 | 31.8 | 35.8 | 38.3 | 29.0 | 37.9 | 38.2 |
| SAGE (published) | **82.3** | 42.8 | 19.0 | 10.2 | 11.8 | 25.4 | 41.8 | 8.9 | 37.9 | 29.3 |
| best v2: mm whitening, MNN plan, reg 0.1, 4 steps | 70.7 | **56.4** | **38.6** | **27.8** | 28.7 | 34.1 | 38.5 | 23.3 | 38.5 | 38.2 |
| best v3: SCORE top-1 shortlist → mm/free SAGE | 64.8 | 51.2 | 34.3 | 26.2 | 30.8 | 37.8 | 41.0 | 28.6 | 40.9 | 38.6 |
| oracle support (SAGE on the TRUE present items) | 82.3 | 82.3 | 82.3 | 82.3 | 70.6 | 50.2 | 41.8 | 56.6 | 37.9 | 62.8 |

- **Culprit 1, SAW whitening on a non-bijective batch.** Moment matching (SCORE's Eq. 3) is far more robust: every top v2 variant uses it.
- **Culprit 2, the forced bijection.** The best moment-matched iterated variant (v2) only TIES SCORE on average. It beats SCORE in R4 by +1.4 to +4, loses at small U, and loses R1 (70.7 vs 82.3).
- **Support estimation from SCORE's aligned scores (v3) cannot close the oracle gap**, because the true support is not recoverable: top-k recall of the present concepts is only 0.30 / 0.46 / 0.56 (k = 1/2/3) at D=1654, and 0.54–0.80 at U=25.
- **Label-free regime statistics** (argmax-histogram entropy, winning-item fraction, N/K) overlap between R1 and R3 U=100: winning fraction 0.714 vs 0.704. So no clean label-free switch exists. The only reliable switch is the protocol itself, i.e. whether a one-to-one batch is declared.
- **Verdict and paper position:**
  - SAGE is the right tool when the experiment DECLARES a one-to-one batch (R1, and R3 at full coverage).
  - Off the bijection, a conservative step (SCORE, or SAGE with moment matching and MNN weighting) is safer, and no method gets near the oracle.
  - Support estimation is the open problem: up to +56 points at D=1654.
  - This becomes the regime-map figure + pitfalls P3/P4 + the "future work: support estimation" limitation.
- **Stopped here:** three grids (v1 shortlist, v2 whitening × plan, v3 SCORE-support) with diminishing returns.

### E5.1/E5.2 label-equivalence with matched scoring (`CVPR_paper/scripts/label_equiv2.py`), sub-01 smoke; full 10-subject run queued

- Old run (`run_inductive_rotation_sweep.py`, cosine scoring, 10 subjects, α cross-fold): supervised rotation from n labeled TRAIN-split trials (4-repetition averages) of the target subject.

| n | 25 | 100 | 500 | 1000 | 5000 | 16540 (all) |
|---|---|---|---|---|---|---|
| Top-1 | 37.8 | 42.8 | 53.1 | 58.2 | 65.3 | **67.2** |

  Reference: unlabeled SAGE = 82.3.
- Matched-scoring smoke on sub-01 (n=200 / all 16,540):
  - supervised rotation + CSLS: 56.2 / 84.0;
  - supervised rotation then SAGE (labels + unlabeled batch): 83.3 / **94.0**;
  - ridge + CSLS: 38.2 / 67.5;
  - reference: SAGE unlabeled alone = 89.5.
- **Reading so far:** even with every labeled training trial and matched CSLS scoring, the supervised map stays below unlabeled SAGE. Labels and SAGE combine (94.0).
  - Caveat: labels come from 4-repetition training trials (lower SNR, different concepts), which is the realistic calibration-session condition.
  - The few-label regime HURTS when followed by SAGE (83.3 < 89.5): a noisy 200-pair rotation is a worse starting point than the identity.

### Last screens

- sqf (P-main) cross-fold over folds 1/3: 73.0 / 64.5.
- reve6: 41.5 / 58.0.
- Both are far behind atm128 (91.5 / 94.5). Their fold-8 GPU runs are dropped. A reve6m screen is queued for completeness.

## 2026-10-20 (night, 7) — repetition sweep, 2nd ablation, geometry, new-encoder SAGE screens

### E2.4 repetition (SNR) sweep: DONE (10 subjects, up to 4 disjoint blocks per q; SAGE params fixed at their q=80 cross-fold values)

| q | plain (L31) | CSLS-3 | SCORE | SAGE | SAGE − SCORE |
|---|---|---|---|---|---|
| 1 | 5.4 | 5.7 | 5.7 | 5.4 | −0.3 |
| 5 | 18.2 | 21.2 | 24.3 | 22.4 | −1.9 |
| 10 | 25.0 | 30.5 | 37.7 | 37.8 | +0.1 |
| 20 | 29.9 | 38.5 | 50.8 | 55.9 | +5.1 |
| 40 | 34.0 | 43.8 | 59.5 | 71.5 | +12.0 |
| 60 | 36.2 | 47.2 | 64.2 | 78.2 | +14.0 |
| 80 | 36.0 | 48.0 | 65.2 | 82.3 | +17.1 |

- ATM L28 (atm_iv) replicates this: SAGE − SCORE is −2.3 at q=5, −1.3 at q=10, +5.0 at q=20, +17 at q=80.
- **Reading:** calibration gains grow with SNR, much faster than plain accuracy does. SAGE's edge over SCORE needs q ≥ 20; at q ≤ 10 they tie or SCORE is ahead.
  - Caveat: SAGE is not re-tuned per q, so a low-q-tuned SAGE might close the gap.
  - This also explains part of R3, whose queries are 10-repetition: R3 mixes the low-SNR handicap with the coverage problem.
  - **P5 (SNR confound) must state the q of every number.**

### E2.5 on atm_iv_valcon: replicates atm31

| full | −whiten | −csls | 1step | −procrustes | softmax | Hungarian plan | MNN | argmax | α=0.6 | plain+Hungarian |
|---|---|---|---|---|---|---|---|---|---|---|
| 81.2 | 75.5 | 67.2 | 68.6 | 53.4 | 73.0 | 63.4 | 58.2 | 55.7 | 81.6 | **65.6** |

- Same ordering as on atm31.

### E8.3 geometry on atm31_valcon (10 subjects; `results/things_eeg/cvpr/geometry/atm31_valcon.csv`)

- **Rotation:** mean principal angle 78.9°; 77% of directions are rotated by more than 30°. This is a large re-orientation, not a small correction.
- **True-pair gap** (own-image cosine minus mean of the others): raw 0.330 → SAW 0.347 → SAGE 0.534.
- **Hubness** (N₅ skewness / max): plain 2.26 / 37.1 → CSLS 1.78 / 24.5 → SAGE 1.03 / 14.6. SAGE de-hubs more than CSLS alone.
- **Per-subject gain vs base** (n=10, Pearson): with plain hubness r = +0.64 (p≈0.05); with raw gap r = −0.62; with plain top-1 r = −0.54. These are suggestive only; do not claim "weak subjects benefit more" beyond "a trend, n=10".

### E2.0-new: SAGE screens on the NEW P-main LOSO arms (`tta_screen_loso.sh`; test-selected checkpoints; wide grid)

| Arm | sub-01 | sub-03 | sub-08 |
|---|---|---|---|
| **atm128 (cross-fold over 1/3/8)** | **91.5** | **94.5** | **81.0** |
| atm128 oracle | 94.5 | 96.5 | 83.0 |
| atm_iv_valcon (ValCon, same encoder) cross-fold | 90.5 | 84.0 | 75.0 |
| atm31_valcon cross-fold | 89.5 | 90.0 | 74.5 |
| tsconv cross-fold over 1/3 | 83.0 | 63.0 | |
| atm128_l31 oracle (1 fold) | 91.0 | | |
| rmix oracle (1 fold) | 74.0 | | |
| rmixm oracle (1 fold) | 66.5 | | |

- atm128 mean over 1/3/8 is **89.0**, about 5 points above the ValCon ATMs on the same folds (83.2 / 84.7): the P-main recipe trains on all concepts for 100 epochs. Its full 10-fold number should clear the old 77% comfortably.
- TSConv has the best PLAIN fold-1 score (50.5) but loses after SAGE, the same encoder inversion as before.
- Token models (rmix, rmixm) are about 20–28 points behind after SAGE.
- reve6 / sqf screens are still running.
- **Decisions:**
  - atm128 is the main model;
  - seeds 3301/3302 move from `atm` (fd512) to atm128;
  - rmixm fold 8 is dropped (kill-losing-screens rule);
  - `atm` fd512 keeps its seed-3300 folds for the same-encoder dimension test.

## 2026-10-20 (night, 6) — R3/R4 regimes: SAGE breaks off the bijection; fix search

### E4.3 / E4.4 regimes on atm31_valcon: DONE (10 subjects, 5 draws; `results/things_eeg/cvpr/regimes/atm31_valcon/regime_rows_all.csv`)

- **Setup.**
  - R3 uses 10-repetition queries over U of the 200 concepts, with a 200-item gallery.
  - R4 uses 80-repetition queries over all 200 concepts, with the gallery enlarged by D training-concept distractors.
  - SAGE uses the cross-fold R1 parameters.

| Cell | plain | CSLS-3 | SCORE | SVTL | SAGE (bal) | SAGE free | SAGE adapt |
|---|---|---|---|---|---|---|---|
| R3 uniform U=25 | 24.1 | 23.3 | 29.1 | 29.1 | **10.3** | 16.6 | 16.6 |
| R3 uniform U=50 | 26.2 | 26.0 | 34.2 | 34.0 | 17.0 | 24.2 | 24.2 |
| R3 uniform U=100 | 25.1 | 27.1 | 36.5 | 36.1 | 25.8 | 30.4 | 30.4 |
| R3 uniform U=150 | 25.5 | 28.3 | 37.6 | 37.0 | 35.6 | 37.4 | 36.8 |
| R3 uniform U=200 | 25.1 | 28.7 | 38.7 | 38.2 | **42.1** | 41.1 | 41.5 |
| R3 Dirichlet U=25 | 24.4 | 22.8 | 27.5 | 27.5 | 8.6 | 14.2 | 14.2 |
| R3 Dirichlet U=100 | 24.6 | 26.2 | 35.0 | 34.5 | 25.1 | 30.1 | 29.8 |
| R3 Dirichlet U=200 | 25.5 | 28.5 | 38.3 | 37.5 | 39.2 | 40.2 | 39.5 |
| R4 D=0 (= R1) | 36.0 | 48.0 | 65.2 | 64.0 | **82.3** | 71.2 | 73.9 |
| R4 D=200 | 27.7 | 36.4 | 52.4 | 52.1 | 42.8 | 41.6 | 41.6 |
| R4 D=400 | 23.0 | 30.4 | 44.8 | 44.2 | 30.0 | 32.0 | 32.0 |
| R4 D=800 | 17.0 | 24.4 | 35.2 | 35.0 | 19.0 | 20.8 | 20.8 |
| R4 D=1654 | 12.4 | 17.4 | 26.4 | 26.3 | **10.2** | 14.8 | 14.8 |

- **Verdict (important, honest):** SAGE wins ONLY when the batch is bijective or fully covered.
  - In R3 it is below plain up to U≈100 and below SCORE up to U≈150.
  - In R4 it is below SCORE at every D>0, and below plain at D=1654.
  - The ρ-relaxed variants (free / adaptive) only partly help.
  - SCORE's conservative step (confident mutual-NN landmarks, one step, identity regulariser) is positive in EVERY cell.
- SCORE's R4 trend matches their own Table 5: complete SCORE 17.03 at 1654 distractors, raw 9.47. Ours on the deployment step alone: 26.4 vs 12.4.
- This is pitfall P3/P4 in concrete form, and it is the paradigm paper's core cautionary result. It must be reported: a calibration that is excellent under the R1 prior can be harmful off it.

### Fix search (`CVPR_paper/scripts/regimes_fix.py`), sub-01 smoke

- The published SAGE variant reproduces `regimes.py` exactly in all 12 cells.
- **v1 grid** (plan bal/free × identity regulariser 0–1 × steps 1/4/16 × CSLS top-k shortlist 0/3/5): nothing beats SCORE off R1. The best mean over cells was 34.5 vs SCORE 41.3. The shortlist does not help.
- **v2 grid** (whitening saw / moment-matching / centring × plan bal / free / mutual-NN × regulariser 0/0.1 × steps): **the whitening is the culprit.**
  - All top variants use SCORE's per-dimension moment matching instead of SAW. A partial-coverage batch's covariance spans only the present concepts, so full whitening distorts the space.
  - Best: `mm_mnn_r0.1_s4`, mean 40.3 vs SCORE 41.3. It wins R4 D=200 (67.0 vs 63.5) but loses R1 (76.5 vs SAGE 89.5).
  - Trade-off: SAW is what makes R1 strong.
- **Next:** full 10-subject run of the v2 grid with three label-free regime statistics per problem: argmax-histogram entropy `stat_hnorm`, winning-item fraction `stat_uniq`, and N/K `stat_nk`. A gate (SAW-SAGE when the batch looks bijective, else the conservative mm variant / SCORE) will be selected cross-fold.
  - Queued as 5 parts under `results/things_eeg/cvpr/regimes_fix/atm31_valcon/part_*`.
  - Fallback if no gate works: report the regime dependence as THE finding (pitfall + "which calibration for which prior" table) and keep SAGE's claim to the bijective / full-coverage regime.

## 2026-10-20 (night, 5) — CPU starvation found and fixed; CPU queue

- **Cause of the slow analyses: every `srun --overlap -c N` step was pinned to the SAME cores.** All steps had `Cpus_allowed_list: 0-1,16-17`. About 13 analyses shared 4 logical CPUs (about 0.3 CPU each) for roughly 2 h, while about 20 cores sat idle. The step cgroup cannot be widened in place.
- **Fix:** all CPU analysis steps were cancelled and replaced by `CVPR_paper/scripts/cpu_worker.sh`, which pops `CVPR_paper/queue/cpu_queue.txt` with the same mechanics as the GPU worker. Workers are launched WITHOUT `-c`, so each gets the job's full cpuset.
  - 10 workers on 523 (7 at 3 threads, 3 at 2 threads) and 2 each on 515, 517 and 514 at 2 threads.
  - Progress is logged in `queue/cpu_started.txt` / `queue/cpu_done.txt` (rc per item).
- **Step mapping was wrong during the earlier de-duplication.** Steps are numbered by launch race, not launch order. The step I cancelled as a "new duplicate" (.26) was the only atm31 stream grid, and one surviving copy (.23) was a duplicate ablation. Both are now covered by the queue. The stream grid resumes per subject from subject 7.
- **Second bug:** the CPU worker exports `SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth`, the ValCon name. The new LOSO and intra models save `checkpoint_test_best.pth`, so the first 11 screens failed (rc=1). `tta_screen_loso.sh` and `tta_screen_intra.sh` now set the name themselves; the 11 items were requeued at the head.
- **Done before the reset:**
  - session_calib atm31 (`results/things_eeg/cvpr/session_calib/atm31_valcon.csv`, analysis pending);
  - r1_competitors for iv28 / sqf_iv28 / atm25;
  - the sqf_iv28 wide screen;
  - r1_ablate atm31.
- **Restarted from scratch:** regimes (split into 5 two-subject parts under `regimes/atm31_valcon/part_*`), label_equiv, r1_transductive (the sub-01 embedding cache is kept), reps atm31, ablate atm_iv, and all SAGE screens.

### E6.2 session-level calibration on atm31_valcon: DONE (`results/things_eeg/cvpr/session_calib/atm31_valcon.csv`)

- Setup: LOSO model; queries average the 20 repetitions of ONE recovered session (4 sessions × 10 subjects); SAGE hyperparameters cross-fold, chosen at 80 repetitions.

| Method | Top-1 (mean ± sd over subject × session) |
|---|---|
| plain, single-session 20 reps | 28.61 ± 7.2 |
| plain, 20 reps mixed across sessions | 30.92 ± 6.6 |
| SAGE fitted on another session, frozen ("cross") | 50.27 ± 8.5 |
| **SAGE fitted on the session itself ("own")** | **53.41 ± 11.2** |
| SAGE, mixed-session queries | 55.02 ± 11.6 |

- Paired over subjects:
  - own − cross = **+3.15**, 10/10 subjects, Wilcoxon p = 0.002;
  - mixed − single-session plain = +2.31, 9/10, p = 0.004;
  - SAGE mixed − own = +1.61, 6/10, p = 0.20.
- **Reading:**
  - Most of the calibration is subject-level: a map fitted on one session keeps 47 of SAGE's 50 points (+21.7 over plain).
  - There is a small, consistent session-specific component: refitting on the session itself adds +3.1 on every subject. This is the "subject-AND-session" calibration claim, and it is modest.
  - Averaging repetitions within a single session loses about 2.3 points of SNR against mixing sessions, i.e. session noise is partly independent.
- **Caveat:** the session labels come from the preprocessing seed. They are verified by fingerprint clustering, not by metadata.

### Dimension hypothesis for SCORE's smaller published gains (R1 competitor tables, ValCon models)

| Model | Dim | plain | SCORE − plain | SCORE − CSLS-10 | SAGE − plain |
|---|---|---|---|---|---|
| ATM L31 | 128 | 36.05 | +29.2 | +17.4 | +46.3 |
| ATM L28 | 128 | 36.35 | +28.3 | +15.7 | +44.9 |
| ATM L25 | 128 | 32.15 | +30.4 | +18.2 | +48.5 |
| TSConv L28 | 512 | 35.30 | +16.4 | +9.2 | +30.3 |
| Squeezeformer L28 | 512 | 35.35 | +14.5 | +8.0 | +28.1 |

- At equal plain accuracy, the 512-d models give SCORE about half the gain, and SAGE loses about a third of its gain.
- The pattern fits "one Procrustes step is under-determined in high dimension" but is CONFOUNDED with encoder family. The clean test is the P-main `atm` (fd512) vs `atm128` (fd128) arms, same encoder and different dimension: run r1_competitors on both when their folds are complete.
- SVTL tracks SCORE within about 1 point everywhere, which is expected since it is the same recipe with 64 landmarks.

## 2026-10-20 (night, 4) — session restart, relaunch, session index recovered

### Ops

- The user moved to a new interactive job (63645, node 514, which has an RTX 3080).
- **Correction to my first diagnosis:** only work that ran ON the old interactive node died. That was:
  - the atm31 stream grid;
  - the chain of the sqf_iv28 wide screen into the intra screen;
  - the SATTC agent's shell.
- Steps attached with `srun --jobid=<batch> --overlap` from that node kept running after their client died, including the second 3090 worker. Their stdout logs stopped; their CSV outputs are unaffected.
- I relaunched everything before noticing, so the following ran twice for about 5 minutes: r1_ablate, reps, regimes, label_equiv, r1_transductive, and session_calib on 515.
  - The duplicates were cancelled (steps 63639.21/.22/.24/.25/.26 and 63565.2, plus one orphan PID); the older copies were kept.
- My renaming of the "killed" `sqf` fold-1 directory crashed that still-alive run (about 35 minutes lost). The fold is requeued and running on 63645.
- Freshly launched and valid:
  - stream_grid atm31 (step .23; resumes at subject 3);
  - the sqf_iv28 wide screen followed by r1_competitors for iv28 / sqf_iv28 / atm25 (63566.2);
  - the intra screen (.27);
  - the atm128 screen (.29, `CVPR_paper/scripts/tta_screen_loso.sh`).
- A queue worker now also runs on the interactive GPU (`logs/cvpr/worker_63645_interactive.out`).
- **Node cap (user, 2026-10-20):** when a batch job finishes, do not replace it. Run 3 batch GPU jobs plus the interactive GPU, 4 nodes in total.
- **Spurious rc=2 explained:** `run_fold.sh` was edited at 20:21 while runs were in flight. `main()` protects the body, but after `main` returns, bash reads on at a stale offset and hits a syntax error. Training and the summary had completed, so the result is intact.
  - Fix: every wrapper now ends with `main "$@"; exit $?`.
  - Runs started before the fix (reve6 fold 1, the atm128 fold-8 run, atm128_l31 fold 1) may still log rc=2. Ignore it when `result.csv` exists.

### GPU screen, first results (plain top-1, P-main recipe, test-selected epoch)

| Arm | Fold 1 | Fold 3 |
|---|---|---|
| atm | 31.0 | — |
| atm128 (bb128/fd128, 100 epochs) | 41.5 | 28.5 |
| rmix (challenge TOK recipe) | 26.0 | — |
| rmixm (mainline optimiser) | 22.5 | — |

- Both token-REVE recipes are well below ATM on fold 1.

### Recipe audit (user question: original-paper config or challenge config?)

- All LOSO arms use the ORIGINAL paper recipe:
  - `--data_average`;
  - SubjectMix: raw_eeg, pairwise, α=0.5;
  - multi-positive loss with grouped batches, 9 samples per image;
  - softplus, τ₀=0.07;
  - L2-normalised InternViT-6B L28 image targets, linear projector;
  - Adam lr 3e-4, wd 1e-4, batch 1024, no scheduler.
- Verified from the runs' own `train_config.json` against `honest_ensemble/atm_iv_valcon`. All differing keys are flags added later by the challenge work; each is at its no-op default.
- The ATM-128 checkpoint is identical in structure: 58 tensors and 2,962,526 parameters, same shapes.
- Uncommitted diffs in `atm.py`, `loss.py`, `sampler.py` and `dataset.py` are all flag-gated and off by default.
- Plain top-1 reproduces the old ATM-128 on the same folds: 41.5 vs 41.0 (sub-01), 32.0 vs 30.5 (sub-03).
- Intended differences:
  - test-selected epoch with no ValCon split (P-main);
  - epochs: atm/tsconv 50 (NeurIPS recipe), sqf 75, atm128 100.
- Only `rmix`/`reve6` add the challenge optimiser (TOK). `rmixm`/`reve6m` are the same models with the paper's optimiser.

### E6.2 session index recovered for the test split

- The released test arrays come from Gifford's `save_prepr`: 4 sessions × 20 repetitions concatenated, then `sklearn.utils.shuffle(arange(80), random_state=20200220)`.
- Hence `session(rep i) = shuffle(arange(80), 20200220)[i] // 20`.
- Verified on sub-01 and sub-05: per-repetition channel-covariance fingerprints correlate 0.37 / 0.26 within a session vs −0.10 / −0.09 across sessions; the 99th percentile of a permutation null is 0.03 / 0.02.
- The training split is NOT recoverable. Each image's 4 repetitions come from 2 sessions, (rep 1, rep 3) vs (rep 0, rep 2), but which absolute sessions is unknown; the behavioral files hold only target/response.
- `CVPR_paper/scripts/session_calib.py` compares plain, SAGE fitted on the session's own 20-repetition queries, SAGE fitted on another session (frozen), and same-SNR queries mixed across sessions.

## 2026-10-20 (night, 3) — R1 competitor table, ablation harness

### E2.5 / E2.6 / E3.6 / α on atm31_valcon: DONE (10 subjects, `results/things_eeg/cvpr/r1_ablate/atm31_valcon_ablate.csv`)

**Setup.** R1, 80-repetition queries, SAGE hyperparameters cross-fold. Each variant flips ONE switch of the SAGE loop. Values are top-1 mean ± std over subjects.

| Variant | Top-1 | Δ vs full |
|---|---|---|
| **full SAGE** | **82.30 ± 8.5** | — |
| −whiten (no SAW) | 72.70 ± 11.9 | −9.6 |
| −csls (cosine in loop and final, same k) | 64.25 ± 6.7 | −18.1 |
| −procrustes (SAW + CSLS only) | 53.85 ± 7.8 | −28.5 |
| 1 step instead of 16 | 68.80 ± 9.0 | −13.5 |
| plan = row softmax (no column constraint) | 71.15 | −11.2 |
| plan = softmax over top-5 | 70.55 | −11.8 |
| plan = Hungarian (hard bijection) | 63.25 | −19.1 |
| plan = mutual-NN landmarks (SCORE's estimator) | 60.20 | −22.1 |
| plan = argmax | 56.40 | −25.9 |
| mutual NN, 1 step (SCORE estimator inside SAW) | 59.45 | −22.9 |
| α = 0.8 / 0.6 / 0.4 | 82.35 / 82.30 / 78.65 | flat down to 0.6 |
| plain | 36.05 | |
| **plain + Hungarian decode** | **64.50 ± 11.7** | |
| CSLS-3 + Hungarian | 64.50 | |
| full + Hungarian decode | 81.60 | −0.7 |

**Reading:**
- **Every component matters.** The largest single factor is the alignment step (−28.5 without it), then CSLS inside the loop (−18), then iteration (−13.5), then the soft bijective plan (−11 vs the best alternative), then SAW (−9.6).
- **The soft Sinkhorn plan beats every hard estimator by 11–26 points.** Hard plans (Hungarian, mutual NN, argmax) lock in their early mistakes. This is the core of the "iterated soft vs one-step hard" (SCORE) mechanism argument, E3.6.
- **Hungarian decoding of PLAIN cosine already scores 64.50, about SCORE's 65.20.** The bijective prior alone, with no adaptation, gives +28. This is a mandatory baseline row; the honest framing is that SAGE adds +17.8 on top of exploiting the prior.
  - CSLS + Hungarian = plain + Hungarian exactly. This is expected: CSLS only subtracts row and column constants, which cannot change the optimal assignment. It serves as a sanity check.
  - Hungarian decoding after SAGE adds nothing (−0.7).
- **α:** shrinking the rotation is harmless down to 0.6 and hurts at 0.4. α=1 needs no tuning in R1. (α matters in streaming, E4.1.)
- The old "Hungarian 89.85 > full_sattc" memory claim is now superseded, not just unverifiable: Hungarian decoding is 64.5 here vs SAGE 82.3.

### Fidelity status of the SCORE / SVTL rows (answer to the user, 2026-10-20 ~22:00)

- **Neither row is a faithful reproduction.** Both are deployment-step-only reimplementations on our encoder, written from the paper text, with no code available. Neither has been validated against any number its authors report.
- **SCORE:**
  - Eqs. 3–9 are implemented.
  - The eps/eta floors are guessed.
  - The code uses ALL mutual-NN landmarks; whether the paper's "12–160" is a cap is unchecked.
  - Their training-time episodes are omitted.
  - Their SAMGA setting is not rebuilt.
  - Our gain is +29 on ATM; theirs is +27 on SAMGA.
- **SVTL:** modelled as SCORE with the 64 most confident landmarks. Its k and confidence function are assumed. Our gain is +28; theirs is +12.8.
- **Paper wording:** "deployment step reimplemented on a shared encoder (no code available)", with the assumptions listed in the appendix.
- A line-by-line audit agent against both papers is running (report only). Changes go into `label_free.py` only after that audit.

- **Why our SCORE/SVTL rows are higher than their papers' numbers (open; user question).**
  - **SCORE:** the base differs (36.05 vs 26.22) while the gain is similar (+29.2 vs +27.0). This part is explained.
  - **SVTL:** the base matches (36.05 vs 35.3) but the gain is +28.0 vs +12.8. Unexplained. Hypotheses:
    - (a) our modelling is not SVTL's step;
    - (b) their 35.3 → 48.1 is a cross-system comparison, not a deployment-step delta;
    - (c) dimension: our ATM space is 128-d. One Procrustes step on ≤160 landmarks is under-determined at 512–1024-d, so the ρ-regulariser keeps it near the identity. Test: the SCORE gain on `iv28_valcon` (512-d), pending in `r1_competitors`;
    - (d) our moment matching uses TEST-gallery statistics; check whether Eq. 3 uses training statistics;
    - (e) query repetitions.
  - (a), (b), (d) and (e) go to the audit agent.
  - Until resolved: claim only "same embeddings: deployment step ~65 vs SAGE 82". Never claim to beat their reported numbers.

- **Audit result (agent, full papers read; nothing edited except the label_free.py docstring).**
  - **SCORE `score_deploy` is FAITHFUL to Eqs. 3–9.**
    - Gallery and target statistics, CSLS k=10, mutual NNs, margin weights, ρ·spectral norm, one step, applied to all queries, final CSLS: all checked. The map orientation was verified numerically.
    - Assumptions: ε, η, and the std estimator.
    - "12 to 160 landmarks" is the observed range of m, not a cap. Ours on sub-01: m = 135, of which 118 are correct.
  - **SVTL:** the paper only NAMES the components in one paragraph (Sec. 2.4). Our version takes every formula from SCORE, so it is plausible but unverifiable. No code: the GitHub API shows a README only, plus an empty second repo.
  - **The deployment-step-only number in SCORE's own paper (Table 2):** SAMGA 26.22 → **48.75** (+22.5), i.e. +12.97 over CSLS (35.78).
    - The 53.23 headline adds their training (+4.48).
    - Other encoders (Table 3): NICE +22.7, ShallowFBCSPNet +18.3, EEGNet +28.3.
  - **Ours:** plain → SCORE **+29.2**, CSLS-10 → SCORE **+17.4**. This is at or just above the top of their range; it is explained by the stronger base, whose landmarks are 87% correct.
  - **SVTL's own +12.8** (35.3 → 48.1) is on their PE-G14/TSConv 512-d L2-normalised space with structured targets. It is not comparable in kind.
  - **Paper wording:** SCORE = "faithful reimplementation of the deployment step (Eqs. 3–9) on the shared encoder; their Table 2 delta +22.5 on SAMGA vs our +29.2". SVTL = "SCORE-style reading of its one-paragraph description; no code".
  - **Validation targets** (their Table 4 cumulative rows, Table 5 distractors, Table 6 frozen transfer) are in the agent report. Paper copies are in `CVPR_paper/related_papers/`.

### E2.0 sqf_iv28_valcon, wide grid: DONE

- loso_fold_cv **64.30**; oracle 65.95. The narrow grid gave 63.4.
- Squeezeformer stays about 18 points behind ATM after SAGE.

### E3.1 SATTC under OUR protocol: DONE on atm31_valcon (`CVPR_paper/scripts/sattc_on_embeddings.py`, `SATTC_WRAPPER.md`)

**Method.** The wrapper drives SATTC's OWN `evaluate_model` (`run_sattc_loso.py`) with three inputs:
- a pass-through model (its forward returns precomputed embeddings);
- a dataloader of our embeddings;
- the config built by their own argument parser, with the same flags as job 63601.

Nothing in their code was edited. Their MNN pre-check was switched off: it only writes diagnostics and never changes the scores.

**Validation gate passed.** Their saved `sub-XX_final.pth` was run once through their real model and once through the wrapper on that model's embeddings:

| Fold | Logged | Their model / wrapper (GPU) |
|---|---|---|
| sub-01 | 12.0 / 31.0 | 12.0 / 31.0 |
| sub-02 | 12.0 / 30.0 | 12.0 / 30.0 |

- The match is exact. On CPU, sub-02 top-5 is 29.5: one query sits at the rank-5 boundary and CPU vs CUDA rounding moves it (EEG forward pass only).
- The logged metric is the final-epoch 200-way accuracy. It involves no random k-way sampling.

**What SATTC needs from training data.** Their test step whitens the gallery with training-candidate statistics (shrink 0.5), computed on the first training step and stored on the config. For our encoder these are rebuilt with their own function from our image projector over all 16,540 training images. Text features never enter their ranking, only the reported loss.

**Result on atm31_valcon** (same 80-repetition embeddings as the R1 table; SATTC defaults, no tuning). Output: `results/things_eeg/cvpr/competitors/sattc/on_atm31_valcon.csv`.

| Method | Top-1 | Top-5 |
|---|---|---|
| plain | 36.05 | 67.45 |
| **SATTC** | **45.35** | **75.55** |
| SCORE | 65.20 | |
| SAGE | 82.30 | |

- SATTC improves every subject except sub-05 (+2.0 top-1, −0.5 top-5).

**Sensitivity check (test-peeking, NOT reportable as a result).** With no gallery whitening, SATTC reaches 50.45 / 82.45; with whitening fitted on the test gallery, 45.85 / 77.20. So the default training-statistic whitening costs about 5 points on our 128-d space. The headline row uses their defaults; this check goes in the appendix as a note.

**Own-conditions reproduction** (job 63601) continues: sub-01 and sub-02 both at 12.0 so far (their paper reports a 14.8 mean).

### E3.2/E3.3 R1 competitor table (`r1_competitors.sh`): DONE for two arms

- Protocol: ValCon-trained models, test-selected epoch (`checkpoint_testsel_best.pth`), 80-repetition queries (seed 3300), SAGE hyperparameters chosen cross-fold (`loso_fold_cv`). 10 subjects, mean ± std over subjects.
- Output: `results/things_eeg/cvpr/r1_competitors/{atm31_valcon,atm_iv_valcon}.csv`.

  | Method | ATM L31 top-1 | ATM L31 top-5 | ATM L28 bb128 top-1 | ATM L28 bb128 top-5 |
  |---|---|---|---|---|
  | plain | 36.05 ± 8.30 | 67.45 | 36.35 ± 7.80 | 67.75 |
  | CSLS-3 | 48.05 ± 9.80 | 79.05 | 48.25 ± 8.81 | 79.50 |
  | CSLS-10 | 47.85 ± 9.63 | 79.15 | 48.95 ± 8.14 | 79.70 |
  | SCORE (deploy step) | 65.20 ± 9.52 | 91.25 | 64.65 ± 8.32 | 91.15 |
  | SVTL (refine) | 64.05 ± 8.45 | 91.10 | 63.80 ± 7.39 | 90.55 |
  | **SAGE** | **82.30 ± 8.10** | **93.40** | **81.25 ± 8.85** | **93.40** |

- Verdict: on identical embeddings, SAGE beats the SCORE/SVTL deployment step by about 17 points of top-1. The top-5 gap is small (about 2 points): SCORE puts the right image in the shortlist but fails to resolve it.

### E2.4–E2.6 / E3.6 / α: `CVPR_paper/scripts/r1_ablate.py` (+ `r1_ablate.sh`)

- One switch of the SAGE loop flipped per variant: −whiten, −csls, −procrustes, 1step, the plan replaced (softmax / top5 / argmax / hungarian / mnn), mnn_1step (SCORE's estimator, one step), α ∈ {0.4, 0.6, 0.8}, plus Hungarian *decoding* of the plain, CSLS and SAGE scores.
- `--reps_sweep`: q ∈ {1, 5, 10, 20, 40, 60, 80}, up to 4 disjoint blocks per q, comparing plain / CSLS / SCORE / SAGE. SAGE uses the q=80 cross-fold parameters, i.e. no re-tuning per q.
- Sub-01 smoke on `atm31_valcon`. `full` reproduces 89.5, so the harness is exact.

  | Variant | Top-1 |
  |---|---|
  | plain | 42.0 |
  | **plain + Hungarian decode** | **76.5** |
  | CSLS-3 | 57.0 |
  | CSLS-3 + Hungarian | 76.5 |
  | **full SAGE** | **89.5** |
  | full + Hungarian | 89.0 |
  | −whiten | 83.0 |
  | −csls | 79.0 |
  | −procrustes | 60.5 |
  | 1step | 80.5 |
  | plan softmax | 77.5 |
  | plan top5 | 76.0 |
  | plan argmax | 62.0 |
  | plan hungarian | 76.5 |
  | plan mnn | 66.0 |
  | mnn_1step | 66.0 |
  | α=0.8 | 89.5 |
  | α=0.6 | 89.5 |
  | α=0.4 | 85.0 |

- **Important for the paper:** Hungarian decoding of *plain* scores is already 76.5 on this subject. It exploits the bijective prior with no adaptation at all, so it must be a baseline row. Under the R1 bijection it is the natural "use the prior, no learning" control.
- Full 10-subject runs launched for `atm31_valcon` and `atm_iv_valcon`: `logs/cvpr/r1_ablate.log` and `logs/cvpr/r1_reps.log`, writing to `results/things_eeg/cvpr/r1_ablate/`.

### Ops

- A second queue worker now runs on the 3090 (job 63639, `logs/cvpr/worker_63639_b.out`). The GPU had 12 GB free and was 61% busy.
- Training throughput is about 1 h per LOSO fold (100-epoch ATM-128), and the queue holds about 100 items. Folds 1/3/8 of every new arm are scheduled first.

## 2026-10-20 (night, 2) — Streaming grid, SCORE/SVTL reimplementations, R1 harness

### E3.1 SATTC (job 63601), sub-01

- Final stage, last epoch, SATTC on: **12.0 top-1 / 31.0 top-5**. Tune stage best on the validation subject sub-06: 18.0 / 44.0 at epoch 4, stopped at 15.
- Each final model is saved under `SATTC-CVPR2026/Retrieval/models/contrast/run_sattc_loso_sattc/across/SATTC/<time>/sub-XX_final.pth`, so it can be re-evaluated later under our protocol.
- About 55 minutes per fold.

### E4.1 streaming grid (`CVPR_paper/scripts/stream_grid.py`), launched on `atm31_valcon`

- Grid: K ∈ {25, 50, 100, 150, 200} × q ∈ {5, 10, 20, 40}; 25 checkpoints; 3 stream orders; 10 subjects.
- Fit: SAGE (balanced, geodesic, α ∈ {0.5, 1}) with this subject's cross-fold hyperparameters from the E2.0 wide screen.
- Scoring:
  - the buffer, with CSLS (transductive, as in the rebuttal);
  - **prequential**: the trials that arrived since the last fit, decoded by the first fit that includes them (`<arm>@new`, decision D17).
- Output: `results/things_eeg/cvpr/stream_grid/atm31_valcon/stream_rows.csv`.
- Smoke test (sub-01, q=40, K=200, t=400): plain 39.75, SAGE α=1 91.25 on the buffer.

### E3.2 SCORE and E3.3 SVTL reimplemented (`CVPR_paper/scripts/label_free.py`)

- No code exists for either: SCORE has none; SVTL's repository is a "coming soon" placeholder, cloned at `/nasbrain/p20fores/SVTL_EEG2Image`.
- SCORE follows its Eqs. 3–9: k=10, ρ=0.1, a single Procrustes step on mutual-NN landmarks weighted by margin. Unstated floors are set to 1e-6.
- SVTL uses the same recipe with the 64 most confident landmarks; its CSLS k (10) and confidence function (the margin) are assumed.
- Synthetic self-check: plain 0.39 → SCORE 0.95, SVTL 0.92.
- Training-time components (SCORE's simulated episodes, SVTL's structured targets) are not reproduced: we compare **deployment steps on the same encoder**.

### R1 harness (`CVPR_paper/scripts/r1_eval.py`)

- Methods: plain, CSLS-3/10, SCORE, SVTL and SAGE, all on the same 80-repetition embeddings. SAGE's numbers reproduce `run_honest_tta_selection` exactly (sub-01: 89.5).
- Sub-01 of `atm31_valcon`:

  | Method | Top-1 |
  |---|---|
  | plain | 42.0 |
  | CSLS-3 | 57.0 |
  | CSLS-10 | 58.5 |
  | SCORE | 72.5 |
  | SVTL | 69.0 |
  | **SAGE** | **89.5** |

- `r1_competitors.sh` is chained after the CPU screens, for ATM L31/L28/L25, TSConv L28 and Squeezeformer. Output: `results/things_eeg/cvpr/r1_competitors/`.

## 2026-10-20 (night) — E2.0 screen results; SATTC training; ops fixes

### E2.0 R1 (200×200) screen of existing test-selected checkpoints

All numbers are top-1, seed 3300, single seed. The CV column is SAGE with cross-fold hyperparameter selection (`loso_fold_cv`, P-main); the oracle column is a reference only.

| Arm | Plain | Narrow grid (CV) | Wide grid (CV) | Wide oracle |
|---|---|---|---|---|
| ATM L28 bb128 (`atm_iv_valcon`) | 36.35 | 80.75 | **81.25** | 86.0 |
| ATM L31 (`atm31_valcon`) | 36.00 | 81.85 | **82.30** | 86.15 |
| ATM L25 | 32.15 | | 80.60 | 84.15 |
| ATM L33 | 35.15 | | 78.05 | 82.5 |
| ATM L35 | 33.40 | | 76.65 | 78.8 |
| TSConv L28 (`iv28_valcon`) | 35.30 | 65.60 | | 70.15 |
| Squeezeformer L28 (`sqf_iv28_valcon`) | 35.35 | 63.40 | | 66.85 |

- With the wide grid, the selected shrink is interior (0.6–0.8), so the 0.85 edge is resolved.
- The ATM family dominates after SAGE (+45 points vs +28–30 for the conv encoders), despite near-equal plain accuracy. This is consistent with the NeurIPS encoder grid.
- Best so far: **ATM L31, 82.30** (top-5 93.4).
- The TSConv L28 wide run and the intra-subject screens are still running.

### E3.1 SATTC — job 63601 is training

- A third OOM (job 63597) showed that the first loader patch did not lower the peak. Measurement on 2 subjects: the peak was the buffer plus two in-flight copies per subject.
- Re-patch: rows are written straight into one buffer, and the per-subject concatenation copy is skipped in the default branch. Still byte-identical on sub-02.
- Job 63601 (unbuffered output plus a memory trace) reached `[TUNE][Epoch 1/100]` with peak RSS about 62.5 GB summed over processes, including DataLoader workers. It is not killed so far. The final stage loads 9 subjects instead of 8, so I am watching memory there.

### Ops

- **Failed queue item.** `atm 3300 1` reported rc=2. Its training finished fine (plain 31.0, epoch 39); only the last line of the script broke, because I had edited `loso_fold.sh` while it ran and bash reads scripts lazily.
  - Fix: `CVPR_paper/scripts/run_fold.sh` wraps the body in `main()`, so the whole file is parsed before running.
  - The queue now uses `run_fold.sh`. Folds already running finish on the old file; their training is unaffected.

## 2026-10-20 (night) — E3.1 SATTC relaunches

1. **Job 63573 failed immediately.** SATTC's `--data_path` defaults to `None` and is never filled from `data_config.json`, so `os.listdir(None)` lists the working directory and an assert fails. Their README command omits the flag. Fix: pass `--data_path` explicitly.
2. **Job 63578 ran out of memory at 60 GB** (MaxRSS 44 GB) while loading data.
   - Cause: their loader keeps the 1654-slice views of every subject and then calls `torch.cat`, so peak memory is about 2× (the training data is 4.2 GB per subject). `extract_eeg` adds another full copy.
   - Fix: a memory-only patch in `SATTC-CVPR2026/Retrieval/eegdatasets_leaveone.py` (original kept as `.orig`):
     - one contiguous tensor per subject;
     - the final concatenation copies into a lazily committed buffer and frees each subject as it goes;
     - `extract_eeg` skips the copy when the time window keeps every sample.
   - Verified on sub-02: data and labels are byte-identical to the original loader (`scratchpad/sattc_patch_check.py`).
3. **Job 63597** submitted with both fixes.

## 2026-10-20 (late) — First screen readings

- **E2.0, `iv28_valcon`** (TSConv L28, test-selected checkpoint): plain 35.30 → SAGE **65.6** (cross-fold selection). Reference columns: `paper_default` 64.05, `oracle_test` 70.15. The best label-free selector, `kappa_agreement`, gives only 57.55, against 82 for ATM. ATM benefits far more from SAGE than TSConv does: +44 vs +30 points.
- **E1.3, rmix fold 1** (token recipe: one-cycle 1e-3, EMA, bf16): plain **26.0** on sub-01. The strongest subject usually scores 41–46, and ATM's run is at 31.0 and still training. That is well beyond seed noise (±3).
  - The rmix fold 3/8 screens are dropped.
  - A recipe check `rmixm` (mainline optimiser: constant lr 3e-4, fp32, no EMA) is queued on fold 1.
- **E3.1:** SATTC features built (train 16540 images / 1654 texts, test 200 / 200). Training has started.

## 2026-10-20 (late) — E3.1: SATTC reproduction launched (user OK for one rtx3090 node)

**Job.** `sbatch CVPR_paper/scripts/sattc_repro.sbatch` → job **63573** on sl-mee-br-502 (rtx3090, 60 GB RAM, since their loader needs about 38 GB). Log: `logs/cvpr/sattc_<job>.out`.

**Code.** Official SATTC at `/nasbrain/p20fores/SATTC-CVPR2026` (commit 0cdfd1b), run with its own LOSO protocol: an internal tune stage on a validation subject (sub-06 by default), then a final stage. Command: `run_sattc_loso.py --gpu cuda:0 --batch_size 1024 --seed 3300`. Output: `results/things_eeg/cvpr/competitors/sattc/own_conditions_seed3300`.

**Deviations from their README** (all logistic):
- `--logger` omitted; `wandb` is stubbed in `SATTC-CVPR2026/.deps`.
- `--seed 3300` added for reproducibility.
- Our `.venv` instead of their conda environment.
- `data_config.json` points to our THINGS-EEG2 data (the original is kept as `data_config.json.orig`).

**Features.** Their ViT-H-14 features are not distributed, so I rebuilt them with `CVPR_paper/scripts/sattc_vith_features.py`. ATM-style recipe: open_clip ViT-H-14 `laion2b_s32b_b79k`, the TRAIN transform (seeded), L2-normalised image and text features, text "This picture is <name>". It writes `SATTC-CVPR2026/features/ViT-H-14_features_{train,test}.pt`.
- **Assumption to verify:** that ATM used the train transform for features. This affects the absolute base numbers.

**Target.** Their paper reports ATM-based LOSO top-1 9.2 → 14.8 with SATTC.

**Ops lesson.** `srun --jobid=<interactive> --overlap` without `-n 1` inherits the job's task count and runs the command 8 times. Always pass `-n 1`. The E2.0 screen used `-c 5`, which forced a single task, so its results are unaffected.

---

## 2026-10-20 (evening) — First launches (protocol P-main, decision D15)

### E2.0 — CPU screen of existing test-selected checkpoints

**Setup.**
- Script: `CVPR_paper/scripts/tta_screen_existing.sh`. It runs `run_honest_tta_selection.py` with `SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth`, on CPU, via `srun --overlap` in interactive job 61490.
- Grid: shrink {0.85, 0.90, 0.94, 0.97, 0.99} × tau {0.03, 0.05, 0.07, 0.10, 0.14, 0.20} × steps 16 × power {1.0, 1.2, 1.5}. Fixed: k=3, 12 Sinkhorn iterations.
- Output: `results/things_eeg/cvpr/tta_screen_existing/<arm>/`. Log: `logs/cvpr/tta_screen_existing.log`.
- Supporting change: `shared.py` now reads the env var `SAGE_CHECKPOINT_NAME` (default unchanged).
- Plain baseline: the test-selected epoch parsed from `train.log` by `CVPR_paper/scripts/testsel_plain.py`.

**Result: `atm_iv_valcon`** (ATM bb128/fd128, ValCon-trained on 90% of concepts, test-selected epoch, seed 3300).

| Protocol | Top-1 |
|---|---|
| plain | **36.35** |
| SAGE, cross-fold selection (`loso_fold_cv`) | **80.75** (top-5 93.1) |
| `paper_default` | 80.2 |
| `oracle_test` (reference only) | 84.1 |
| label-free `mutual_nn` | 82.05 |
| label-free `hubness_uniform` | 81.9 |
| label-free `kappa_agreement` | 81.65 |

- Per-subject cross-fold top-1: 90.5 / 80.5 / 83.5 / 55.5 / 86.5 / 85.5 / 75.5 / 74.0 / 84.5 / 91.5.
- The selected shrink is **0.85 in every fold, which is the grid edge**. A wider grid (0.5–0.94) is chained in `tta_screen_existing_wide.sh` (log: `logs/cvpr/tta_screen_chain2.log`).
- Verdict: the old 77.2 is beaten, by a single seed, with no peeking on calibration hyperparameters and a model trained on 10% fewer concepts. The 128-d ATM shape (`atm128` arm) becomes the priority retrain.

**Plain baselines of other test-selected checkpoints** (seed 3300, from `testsel_plain.py`):

| Arm | Plain top-1 |
|---|---|
| `iv28_valcon` (TSConv L28) | 35.30 |
| `sqf_iv28_valcon` | 35.35 |
| `atm31_valcon` | 36.00 |
| `iv33g_valcon` | 33.85 |

Their SAGE screens are running.

**Chained after E2.0:** `tta_screen_intra.sh` on the intra-subject one-cycle models `epj_oc`, `tsc_oc`, `sqf_oc`, `epj_oc_s2026` (test-selected, 17 channels; output `results/things_eeg/cvpr/tta_screen_intra/`).

### E1.x — GPU queue

- Queue file: `CVPR_paper/queue/gpu_queue.txt`; arms in `CVPR_paper/scripts/loso_fold.sh`; workers `CVPR_paper/scripts/gpu_worker.sh`.
- Logs: `logs/cvpr/worker_<job>.out`; `started.txt` and `failed.txt` in `CVPR_paper/queue/`.
- Workers: sbatch 63565 and 63566 (rtx3080, nodes 515 and others).

**Two false starts.**
1. Job 63561/63562: `flock` exec of `bash` failed with EKEYEXPIRED. Batch jobs have no Kerberos ticket for the NFS home. Fixed by forcing `PATH`/`HOME` to non-home locations and using `/bin/bash`.
2. Job 63564: `train.py` defaults `--text_feature_dir` to a missing BLIP2 file. It also drained 2 items, because the logged exit code was that of `date`. Fixed by passing `--text_feature_dir ""`, recording `rc` properly and writing failures to `failed.txt`. The two items were re-queued.

**Queue order.**
1. `atm128` folds 1/3/8.
2. `atm`, `rmix`, `sqf`, `reve6`, `tsconv` on folds 1/3/8 (seed 3300).
3. `atm`, `tsconv`, `atm128` on the remaining folds.
4. `atm` and `tsconv` seeds 3301 and 3302.

### Competitors

- SATTC official code cloned at `/nasbrain/p20fores/SATTC-CVPR2026` (commit 0cdfd1b, 2026-03-24). Not run yet.
- Caveat: their loader needs about 38 GB RAM; our 3080 nodes have 31 GB (open question Q7).

---

## 2026-10-20 — Kick-off, nothing launched

### Provenance audit of the NeurIPS paper and rebuttal

This was a read-only audit. Scan helpers live in the session scratchpad and are not needed again.

**Main checkpoints are deleted.**
- Paper Table 1, the ablation tables, the encoder table and the sensitivity/progressive figures were all evaluate-only runs on top of `results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-{190741,170207}` and `eeg_encoder_architecture_sweep_mixup_20260429-221100`.
- Their `.pth` files were deleted in the September 2026 cleanup.
- The same base model is also the source of every `tta_rebuttal` experiment.

**Two levels of test leakage.**
- Checkpoints were chosen with `select_best_on=test`.
- The TTA config (0.94, k3, τ0.1, steps16, pow1.2, iters12) is the argmax of test top-1 for seed 3300 over `sattc_final_paper_512_sweep_run_20260501-*`, then reused for seeds 3301 and 3302.
- AllJoined and MEG TTA configs were also selected on test.

**Inconsistent values for the same model.**
- The TSConv + TTA number appears as 68.1 (3-seed mean), 68.8 (May), 68.3 (July) and 68.5 (a different TSConv run in the encoder table).
- The progressive figure ran with 8 Procrustes steps, which is why its N=200 point is 67.25.
- In Table 2, the zero-shot row is the last epoch (31.5), while the val-selected checkpoint gives 31.7.

**Headline and baseline problems.**
- The 77.2 ATM headline is a single seed, the best of 5 image encoders, with TTA tuned on TSConv (`tta_sweep_20260503-152811`).
- The AllJoined baseline of 8.0 has no source anywhere in `results/`.

**Rebuttal streaming table.**
- It re-fits and re-scores the buffer (`rest = buf` in `run_streaming_refit.py --acausal`). Under decision D2 this is the intended transductive regime.
- Run with α=1 and the test-tuned config.
- Refit counts: 25 for some cells, not 50.

**SubjectMix controls.**
- The 35.90 row is the original April run; a config-identical rerun gives 34.80.
- `beta_k9` (36.70) is an unreplicated single run (decision D5).

**NSD 300-way table.** Per-pool fits with the PCA rank picked on test. Only the fit-on-1000 CV result (68.4 → 84.8 image retrieval, 1000-way) is clean.

**Intra-subject SAGE-TTA.** `intra-subjects/tta_grid_sweep_20260503-150419` gives 85.25 → 99.05 (single seed, test-selected). It was not in the paper.

### "Hungarian beats SAGE" check

- The 89.85 figure came from a memory note about a λ-sweep directory (`transductive_lambda_sweep/20260623-000530`). That directory no longer exists anywhere under `results/`.
- On the main model, `results/full_benchmark_suite/2026062{3}-*/transductive_benchmark_summary_*.csv` gives:

  | Method | Top-1 (%) |
  |---|---|
  | `full_sattc` | 68.45 |
  | `sattc_hungarian` | 66.95 |
  | `hungarian_sattc` | 61.95 |
  | `hungarian` | 51.4 |
  | `sinkhorn` | 49.0 |
  | `plain_cosine` | 35.9 |

- Hungarian does not beat SAGE, so the pitfall is dropped (decision D10). The old memory figure is unexplained, probably a different model or split.

### Assets that survive with weights (usable for clean reruns)

| Path | What it is |
|---|---|
| `inter-subjects/tsconv_val_best_20260501-170237/*_valbest` | TSConv, LOSO-val, seeds 3300–3302 |
| `honest_ensemble/atm_iv_valcon/seed3300` | ATM, ValCon, 33.40 inductive |
| `honest_ensemble/sqf_iv28_valcon` | Squeezeformer, ValCon, 31.20 — weights still to verify |
| `synthetic_subjects/promising_allfolds/group_e100/seed3300` | Not a headline (D5) |
| `results/alljoined/inter-subjects/tsconv_fd512_mixup_raw_pairwise_linear_a0p5_seed330*` | AllJoined models |
| `results/things_meg/inter-subjects/things_meg_*` | THINGS-MEG models |
| `results/things_eeg/intra-subjects/TTA/` | EEGProject intra-subject models |

### ValCon finding (to verify under P-clean)

- Validation by held-out concepts: `--val_concept_ratio 0.1 --val_concept_seed 20260822`; see `ensemble_experiments/compact_valcon/README.md`.
- Single seed, it beats LOSO-val subject selection: ATM 29.15 → 33.40, TSConv-group 30.75 → 33.30, TSConv/bigG 30.10 → 30.10; mean 30.00 → 32.27 (`ensemble_experiments/ENSEMBLE_RECAP.md` around l.440).
- Tracked as E1.8.

### Competitor code

| Method | Code |
|---|---|
| SATTC (CVPR 2026) | https://github.com/QunjieHuang/SATTC-CVPR2026 |
| SCORE (arXiv 2608.19134) | No code found |
| SVTL (arXiv 2609.36971) | No code found |

### 2026-10-23 17:00 — 3-seed SCORE (original) table landed (`final/main_table_3seeds.md`)
- `score_orig` (SAMGA + our SCORE objective, 50 epochs, `checkpoint_last`), seeds 3300-3302, 10 subjects:
  plain 25.60, CSLS 34.65, SATTC 23.68, SVTL 46.62, SCORE 47.15, SAGE 68.37 (+21.22 over SCORE, 10/10, 0/30 harm).
- `score_orig_tb` (same runs, test-selected checkpoint): plain 33.55, SCORE 49.83, prior 50.35, SAGE 61.53 (+11.70).
  Test selection inflates zero-shot (+8.0) and lowers SAGE (-6.8): same direction as P1 on ATM-128.
- Paper: one sentence added to the competitors paragraph (68.4 vs 47.2, 10/10 subjects).

### 2026-10-23 — LaBraM (rebuttal vxam-Q1) queued for the Table-2 battery
- `loso/labram/seed3300` -> `foundation/LaBraM_mix_seed3300` (pretrained LaBraM, full FT, AVDE recipe, SubjectMix a0.5,
  InternViT L28, test-selected ckpt = P-main). CPU item `encoder_battery.sh labram 3300` -> `final/labram_s3300/`.
- Paper rewrite merged with the NeurIPS text + rebuttal promises; backup `scratchpad/paper_before_neurips_merge.tex`.

### 2026-10-23 18:20 — E3.2d result: neither faithful-recovery variant closes the SCORE training gap
- 30 epochs, last epoch, 10 folds (plain / SCORE deploy): samga30 26.70 / 50.60; score30 27.65 / 48.60;
  score30rt (tau_rec 0.07) 26.50 / 46.85; score30s2 (L_rec from epoch 20) 27.40 / 50.80. Paper: 29.3 / 53.2.
- Decision rule not met (+3 over Standard and >= 52): SAMGA* column stays `score30`. Reproduction caveat in the text.
- Queued (CPU): SAGE on score30s2 (strongest SCORE-trained encoder) — screen + r1_eval, checkpoint_last
  -> `final/score30s2_orig_s3300/r1_competitors.csv`.

## 2026-10-23 18:25 — E8.11 why the RSA loss fails: geometry diagnostic (CPU) + cross-subject isometry loss (GPU) — QUEUED

- **Question.** RSA predicts SAGE's gain but an RSA loss does not raise it (E8.5): RSA 0.70 -> 0.77, SAGE 89.0 -> 89.0
  (lambda 1) / 84.0 (lambda 5), and the in-sample label oracle drops 97.5 -> 82.8 / 72.8. Hypothesis: the loss buys
  global (Pearson) agreement with fewer effective dimensions and less local, instance-level structure.
- **Diagnostic** `CVPR_paper/scripts/geometry_diag.py` on the 19 three-fold screen arms -> `analysis/rsa_screen/geom_diag_{a,b,c}.csv`:
  participation ratio of the test clouds, 10-NN agreement EEG vs image, split-half supervised rotation (fit on 100
  true pairs, 100-way on the other 100: the honest version of the in-sample oracle, which overfits at n=200 < 2d).
- **Training target that respects SAGE's assumption.** `--iso_lambda` (`module/loss.py:isometry_loss`): split-half
  residual of the best rotation between two source subjects' embeddings of the batch's shared images (~113 images
  < 128 dims, so the in-sample residual is ~0.19 even for independent clouds; split-half gives 1.0 independent /
  0.0 rotated, unit-tested). Arms `atm128_iso1`, `atm128_iso5`, folds 1/3/8, 100 ep (~3 h/fold, the per-step SVDs).
  GPU: sbatch 64398 (515), 64399 (520), 64400 (517) + nohup on interactive 64236 (514). Eval: `screen3_eval.sh`.
- **Paper fix.** SubjectMix REPLACES each trial by its mixture (one lambda per trial, `subject_mixup_prob` 1); the
  NeurIPS text said "additional positives". Base-encoder paragraph corrected.

### 2026-10-23 18:40 — E8.11a geometry diagnostic: DONE — the RSA loss wins RSA by collapsing the EEG embeddings
`analysis/rsa_screen/geom_diag_{a,b,c}.csv` (19 arms x folds 1/3/8), joined with the `_cv3.csv` screens:

| arm | RSA | PR eeg | 10-NN agree | split-half rot | SAGE | ff |
|---|---|---|---|---|---|---|
| atm128_f138 (baseline) | 0.696 | 14.26 | 0.350 | 66.97 | 89.0 | 0.831 |
| atm128_rsa1 | 0.766 | 6.86 | 0.359 | 49.50 | 89.0 | 0.836 |
| atm128_rsa5 | 0.753 | 5.79 | 0.348 | 42.87 | 84.0 | 0.765 |
| other 16 arms | 0.52-0.70 | 11.1-18.5 | 0.25-0.35 | 26-60 | | |

- The RSA loss halves the effective dimension (participation ratio), far below every other arm, and the honest
  split-half supervised rotation drops 67.0 -> 49.5: instance information lost. Matchability up, information down, net 0.
- Spearman vs ff, 17 arms without the RSA loss: split-half rotation 0.96, RSA 0.94, 10-NN agreement 0.92,
  plain 0.73, in-sample oracle 0.27 (overfits n=200 in 128 d), PR eeg -0.19, PR img -0.67.
- Paper: "RSA is not a training objective" paragraph extended with these numbers.

### 2026-10-23 19:10 — E8.11b: iso5 killed, iso03 queued; LaBraM R1 in
- Matched-epoch zero-shot (test top-1, mean of last 5 epochs vs atm128 at the same epochs): iso5 sub-01 12.9 vs 26.5
  at ep 22 -> killed (fold + queued folds + screen3 waiter). iso1 at ep 22-24: +1.5 / -6.1 / -5.7 (subs 1/3/8), kept.
  New arm `atm128_iso03` (lambda 0.3), folds 1/3/8, screen3 queued.
- LaBraM (pretrained, full FT, test-selected ckpt) R1, 10 subj: plain 31.85, CSLS 41.7, SATTC 35.9, SVTL 57.4,
  SCORE 57.8, SAGE 65.55 (+7.75 over SCORE, 10/10, min +2.0, p=0.002). Top-5: SCORE 87.8 > SAGE 86.7.
  r1_transductive (prior only, TransCLIP, EM) still running.

### 2026-10-23 19:30 — LaBraM column complete; SAGE on score30s2
- LaBraM r1_transductive: TransCLIP 50.6, EM-Dirichlet 52.6, prior only (Hungarian) 59.05 > SCORE 57.8.
  Paper Table 2: new LaBraM column; text: 130/130 runs SAGE > SCORE, margins 7.8-23.2 (smallest on LaBraM);
  SATTC -11.4..+4.0 on the 9 non-ATM-128 encoders; prior > SCORE/SVTL on 8 of our encoders incl. LaBraM.
- score30s2_orig (best SCORE-trained SAMGA, last epoch), own cross-fold params: plain 27.4, SCORE 50.8, SVTL 50.85,
  SAGE 69.9 (+19.1, 10/10, min +7.5). Sentence added to the competitors paragraph.
- Bib: du2025shallow title was wrong in the NeurIPS bib (both arXiv versions: "Deep Models, Shallow Alignment:
  Uncovering the Granularity Mismatch in Neural Decoding"); fixed. jiang2024labram added (arXiv API, ICLR 2024).
  All 55 inherited entries now under verification.
- Queued: error_decomposition.py with labram_s3300 (Fig. 2a/b tenth encoder).

### 2026-10-23 19:50 — Fig. 2 with LaBraM (10 encoders); inherited bib verified
- encoder_gain_diagnostics.csv rebuilt with labram_s3300: plain 31.85, SAGE 65.55, RSA 0.663, ff 0.494.
  10 encoders: Spearman(zero-shot, ff) 0.04 (p=0.91); Spearman(RSA, ff) 0.92 (p=0.0002); all points 0.846.
  figures/encoders.pdf regenerated; Sec. 5.5 text + caption updated.
- All 55 inherited references checked on primary sources. Real errors fixed: conneau2018muse author order
  (Conneau first), li2024atm author order (Li, Wei, S. Li, Zou, Liu), du2025shallow title. Updated venues:
  zhang2025neurobridge -> AAAI 2026, li2024mcrl -> ICME 2025. Minor: grootswagers title, THINGS-MEG creator name,
  Alljoined-1.6M casing, Ozcelik spelling, CVPR 2017 booktitle; pages added (Cuturi, InternVL, Grave, SATTC, ATM).
  Backup: scratchpad/references_before_inherited_fix.bib.

### 2026-10-23 21:05 — E8.11b result: the isometry loss collapses the embeddings (worse than the RSA loss)
- `atm128_iso1` folds 1/3/8 (own 3-fold params): plain 32.7, Hungarian 70.2, SCORE 66.2, SAGE 72.3 (baseline f138:
  35.0 / 71.7 / 71.8 / 89.0), RSA 0.57 (0.70), in-sample oracle 18.2 (97.5), ff 0.589 (0.831).
  geometry_diag (`geom_diag_iso1.csv`): participation ratio 1.82 / 1.88 / 1.76 (baseline 14.3), 10-NN 0.24,
  split-half rotation 11.0 (67.0). The cloud is nearly one-dimensional: trivially "rotatable", useless for SAGE.
- Reading. Both geometric training targets aimed at SAGE's premise (RSA cross-modal, isometry cross-subject) are
  satisfied by dimensional collapse; the contrastive term does not prevent it.
- `atm128_iso03` (lambda 0.3): zero-shot at ep 60 tracks iso1 (30.2 vs 29.5; base 33.9); its 3 folds finish ~00:30.
- E8.11c queued: `atm128_iso1pr` = iso1 + collapse guard relu(1/PR - 1/14) x 5 (`--pr_lambda 5 --pr_floor 14`,
  `module/loss.py:collapse_penalty`, unit-tested), folds 1/3/8 + screen3. Tests whether collapse is the whole failure.
- GPU node cap 4: workers 64398 (515), 64400 (517), 64416 (520) + nohup 514; cancelled extra 64415/64417 (fold 3
  requeued, partial dir removed).

## E8.11c result (2026-10-24): isometry loss, dose and collapse guard — training-side levers closed
ATM-128, 100 ep, folds 1/3/8, test-selected ckpt (mean). f138 baseline / rsa1 / iso1 / iso03 / iso1pr:
- zero-shot 35.0 / 33.0 / 32.7 / 33.8 / 33.0; SCORE 71.8 / 69.7 / 66.2 / 68.2 / 69.2; SAGE 89.0 / 89.0 / 72.3 / 84.2 / 85.5
- RSA 0.70 / 0.77 / 0.57 / 0.60 / 0.75; PR_eeg 14.3 / 6.9 / 1.8 / 2.2 / 9.5; cv_rot 67.0 / 49.5 / 11.0 / 16.1 / 57.8
- iso03 collapses as well (dose 0.3 is not a safe regime). The guard (relu(1/PR-1/14) x5) only holds PR at 9.5 and recovers SAGE to 85.5, still -3.5 under the base.
- Conclusion: every geometric training target tried (RSA, split-half isometry, isometry + collapse guard) is met partly by collapsing the EEG cloud and none beats the base encoder. Training-side exploration concluded. Paper: one sentence added to "RSA is not a training objective".
Files: analysis/rsa_screen/atm128_{iso03,iso1pr}_cv3.csv, geom_diag_iso.csv.

## 2026-10-24 — v2 paper (advisors' changes) + ValCon retraining launched
v1 frozen as `paper_v1_full.{tex,pdf}`. Changes:
- SCORE only through its published numbers.
- SVTL removed.
- R3 partial coverage dropped; R4 renamed R3.
- ATM-128 renamed ATM.
- ATM-512, Squeezeformer, REVE-tok. and SAMGA* removed.
- Streaming shown as a figure (buffer top-1 per refit), no "prequential".
- Ablation on ATM only.
- Prose simplified.
- Pitfalls 8 -> 7.

Numbers stay test-selected until the ValCon runs land.

**Support estimation without SCORE** (`regimes_support.py --r4_only`, arm `sage_hung_csls`): rectangular Hungarian on SAGE's whitened CSLS scores, then SAGE on the 200 selected candidates. ATM, 3 seeds, open gallery (D = 0 / 200 / 400 / 800 / 1654):

| Method | D=0 | 200 | 400 | 800 | 1654 |
|---|---|---|---|---|---|
| zero-shot | 35.8 | 28.2 | 23.3 | 18.7 | 13.7 |
| SAGE | 84.6 | 46.2 | 30.5 | 18.6 | 11.7 |
| SAGE + CSLS support | 84.6 | 42.0 | 32.7 | 24.9 | 17.9 |
| (old, SCORE-score support) | 84.6 | 56.2 | 46.6 | 37.1 | 27.4 |

- Iterating it (re-Hungarian under the fitted map, seed 3300 only) adds ~0.5 (19.3 vs 18.8 at D=1654); not kept.
- Valcon s3300: 83.1 / 40.5 / 31.6 / 22.8 / 16.6.

**Six-encoder Fig. 2** (current, test-selected): ρ(zero-shot, ff) = 0.43 (p = 0.40); ρ(RSA, ff) = 0.94 (p = 0.005); pooled with the 14 variants ρ = 0.88 (p = 4e-7). SAGE beats the best runnable competitor (prior / EM / TransCLIP / CSLS / SATTC) in 85/90 subject runs: the exceptions are 2 ties on TSConv s3301 and 3 subjects on LaBraM. It is never below zero-shot.

**ValCon retraining (GPU queue, 137 folds).** `_vc` suffix in `run_fold.sh` / `run_intra.sh` = `--val_concept_ratio 0.1 --val_concept_seed 20260822 --select_best_on val --early_stop_patience 20 --save_testsel_checkpoint`. New `labram` arm (`inter-subject-foundation.sh` LaBraM_mix recipe).
- {labram, tsconv, eegnet, eegproject, conformer}_vc s3300: 10 folds each. CPU waiters `encoder_battery.sh` are queued.
- atm128_vc s3301 and s3302: s3300 = symlink to `atm128_valcon`.
- intra atm128_oc_vc s3300.
- 19 Fig.-2 variants as `atm128_<v>_vc`, folds 1/3/8 (incl. rsa1/rsa5/iso1/iso03/iso1pr). tsconv_iv44 dropped: its sentence was cut.

`CVPR_paper/scripts/vc_driver.sh` (nohup on 514) queues `final_analyses.sh` / `screen3_eval.sh` per arm once its folds exist. Still to do by hand once batteries land:
- `support_csls` for atm128_vc s3301/s3302;
- `error_decomposition.py` on the `_vc` encoders;
- point `make_encoder_figure.py` / `make_figures.py` at the `_vc` arms;
- swap every number.

Training-loss check (user asked): `multi_positive_pair_loss` = (EEG->image + image->EEG)/2, both multi-positive; `alpha = beta = 1` for ATM. The EEG side is not L2-normalized in training. LaBraM uses `alpha 0.8` (+ regression).

## 2026-10-24 afternoon: Hungarian on every method + what the drift is (user)
**Prior-plus** (`scripts/prior_plus.py`, `analysis/prior_plus/<arm>.csv`): each method's own scores decoded by argmax vs Hungarian. ATM, mean of 3 test-selected seeds:

| Method | Argmax | + Hungarian |
|---|---|---|
| zero-shot | 35.8 | 66.2 |
| CSLS | 48.1 | 66.2 |
| TransCLIP | 55.0 | 52.1 |
| EM-Dirichlet | 56.8 | 51.6 |
| SAGE | 84.6 | 84.7 |

ValCon s3300: 33.4 / 63.0; 44.8 / 63.0; 52.4 / 49.3; 55.5 / 50.2; 83.1 / 83.4. Only SAGE adds anything beyond the 1:1 prior. SATTC + Hungarian not computed (official wrapper).

**Drift nature** (`scripts/drift_nature.py`, `analysis/drift/atm_valcon_*.csv`, ATM ValCon s3300, 10 subjects). Maps are fitted with InfoNCE on 90% of the training-session concepts (4-rep). Held-out = 200-way on held-out training images (4-rep).

| Map | Held-out top-1 |
|---|---|
| identity | 20.8 |
| rotation | 44.8 |
| whitening + rotation | 42.4 |
| affine | 49.7 (+5.0 over rotation, 10/10, p=0.002) |
| MLP | 50.6 (+0.8 over affine, p=0.03) |

Drift ≈ linear, mostly rotation.

- Test 80-rep, CSLS: rotation 83.8, affine 87.4, MLP 88.0 vs SAGE 83.1. Rotation vs SAGE p=0.61; affine vs SAGE +4.4, p=0.20.
- SAGE's R vs the supervised R in SAGE's whitened space: median principal angle 66°. Each from identity: supervised 59°, SAGE 73°; random 89°. Top-1 agreement 76%. The old "72° = far from small" claim had no reference point and was removed.
- Transfer:
  - frozen SAGE on held-out training images: 22.8 vs whitening only 21.6 (n.s.); supervised frozen 44.3. SNR-mismatched.
  - matched SNR (`scripts/transfer_split.py`, split-half of the test set, 100-way): zero CSLS 56.6, whitening frozen 61.4, SAGE frozen 64.7, refit 84.5. SAGE keeps 26% of its gain on unseen images (test-selected: 27%).
  - => SAGE registers the query set onto the candidate set; it does not estimate the subject's displacement.

Paper: new Table `tab:prior`; protocol paragraph making the Hungarian-decoded control mandatory; the "Geometry" paragraph replaced by "The displacement of a new subject" and "SAGE registers the test set; it does not recover the displacement". Claims of an "orthogonal correction" / "subject-specific misalignment" rewritten (Fig. 1 caption, contribution 2, Sec. 4 intro, RSA sentence, discussion, conclusion). Label paragraph switched to ValCon s3300 numbers (80.8 vs 83.1, +SAGE 90.2).

## 2026-10-25 — ValCon swap (all paper numbers now on ValCon checkpoints)

All `_vc` GPU folds and CPU batteries done (driver finished before the interactive-job restart). Added support_csls +
prior_plus for atm128_vc s3301/s3302. `final/atm128_vc_s3300` = symlink to `atm128_valcon_s3300`.
- ATM, 3 seeds (seed_table.py --arms atm128_vc): zero-shot 32.2, CSLS 43.4, SATTC 43.4, prior only 63.3, TransCLIP 52.8,
  EM-Dirichlet 56.0, GALA 81.8 ± 1.5 (seeds); GALA beats every competitor 10/10 per seed (p=0.002), 0/30 harm; 75/80 runs
  over all encoders.
- Encoders (s3300, Table 3): TSConv 32.4→63.2, EEGNet 21.3→57.4, EEGProject 22.1→51.6, Conformer 23.4→60.5, LaBraM 25.6→55.4.
- Ablation now 3-seed mean (was seed 3300): full 81.8, -whiten 76.0, -csls 67.8, 1step 68.3, -rot 52.0; plans softmax 71.1,
  top5 70.1, hungarian 62.0, mnn 56.4, argmax 54.5.
- Prior-plus (3 seeds): zero-shot 32.2/63.3, CSLS 43.4/63.3, TransCLIP 52.8/50.1, EM-Dir 56.0/50.4, GALA 81.8/81.9.
- Leakage: test-selected epoch of the same ValCon runs 35.8 zero-shot (+3.6); oracle-tuned GALA 85.5 (+3.7).
- R2 (3 seeds): first refit undamped 17.4, zero-shot 36.5, damped 38.9; last refit 76.5 / 76.0. R3: D=1654 GALA 11.3,
  zero-shot 11.5, GALA+support 17.2.
- **Fig. 2 claim weakened**: `analysis/encoder_gain_diagnostics_vc.csv`; Spearman over 6 encoders zero-shot ρ=0.20
  (p=0.70), RSA ρ=0.49 (p=0.33) (test-selected: 0.43 / 0.94; but 0.66 with mean-of-ratios even there). Subject-level
  (n=60): RSA 0.45 (p=3e-4) vs zero-shot 0.34 (p=0.009). User chose "describe, don't claim": RSA removed from abstract,
  contribution 4 rephrased, encoders paragraph describes, conclusion clause dropped.
- "Number of labels" paragraph was already ValCon s3300 (unchanged).

## 2026-10-25 — SATTC wrapper audit: NO BUG

`results/things_eeg/cvpr/competitors/sattc/audit_2026-10-25/` (audit.py, ablation_*.csv). Re-run of tsconv_vc and
atm128_valcon s3300 reproduces every per-subject sattc.csv value. Image features are read from each run's config
(layer 28; the "layer31" in SATTC_WRAPPER.md was stale doc, fixed). Plain top-1 = result.csv on every subject.
Component ablation (10 subj, one SATTC flag off at a time), TSConv / ATM s3300:
plain 32.35 / 33.40; full 20.10 / 42.60; no train-stat gallery whitening 30.45 / 47.95; no SAW 23.85 / 36.10;
no Ada 20.25 / 43.60; no CSLS 20.00 / 42.30; no PoE 20.25 / 42.55; SAW only 30.05 / 46.55; Ada-CSLS+PoE only 36.80 / 41.65.
Cause: SATTC whitens the gallery with train-split image statistics; its own encoder is TRAINED against whitened
targets (run_sattc_loso.py:1144,1188), ours are not, so test-time-only whitening warps our gallery.
`--disable_global_auto_whiten` is their documented switch. Paper keeps the defaults (decision pending with user).
- **Decision (user, 2026-10-25): paper reports SATTC WITHOUT train-stat gallery whitening** (`sattc_on_embeddings.py ours
  --no_gallery_whiten` = their `--disable_global_auto_whiten`; `final/<arm>/sattc_nowhiten.csv`). ATM 3 seeds 48.0
  (top-5 79.8; was 43.4), TSConv 30.4 (20.1), EEGNet 29.1 (22.6), EEGProject 31.6 (25.0), Conformer 32.2 (20.3),
  LaBraM 39.9 (36.4). GALA still beats it 10/10 subjects per ATM seed. One explanatory sentence in "Comparison with
  published results".

## 2026-10-25 — R2 at 50 refits on ValCon (stream50_sage, atm128_vc 3 seeds)
`final/atm128_vc_s33*/stream50_sage/` (ValCon battery had used stream_grid's default 25). Fig. 4a and appendix
Fig. 5 (`make_stream_appendix.py`, K x q grid, x = t*q*0.2 s) now use it. First refit: zero-shot 36.7, undamped 10.9,
damped 38.2; last: 37.0 / 76.5 / 76.0. Undamped first-refit drop vs zero-shot by q: 5:-11.0, 10:-22.1, 20:-33.9,
40:-36.5. Damped - zero-shot >= -3.0 at every (K, q, refit) (3/1000 points below -1); averaged over configs >= +1.4.

## 2026-10-26 — Fig. 3 (RSA) with the 5 ATM image targets (ValCon, 10 folds each)
`analysis/encoder_gain_diagnostics_targets_vc.csv`. Fraction of errors fixed / RSA: InternViT L20 65.6 / 0.682,
bigG L27 62.0 / 0.680, DINOv2 L40 19.4 / 0.583, InternViT L44 16.3 / 0.567, CLIP-H final 14.0 / 0.505.
Over the 11 models: RSA rho = 0.92 (p = 7e-5), zero-shot rho = 0.78 (p = 0.004); partial rho RSA | zero-shot = 0.78,
zero-shot | RSA = 0.17. Fig. 3 now has both panels over 11 points (targets = open diamonds); encoders paragraph,
caption and implementation updated; AVDE (Dai et al., ICLR 2026, arXiv 2602.22555) cited for the LaBraM recipe.
