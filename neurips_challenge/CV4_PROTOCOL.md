# CV4: the cross-subject evaluation protocol for the Track-1 sealed phase

Adopted 2026-10-01. Every recipe that is meant for the sealed submission is compared under this protocol and
only under this protocol. Scripts: `neurips_challenge/cv4.sh` (runs), `neurips_challenge/cv4_summary.py`
(tables, decisions, training-length audit). Results: `results/alljoined/track1/cv4/`.
Since the 2026-11-02 cleanup `cv4.sh` holds only the best ensemble's recipes; the recipe tables below are the
historical record (their code: git tag `pre-cleanup-2026-11-02`).

```bash
bash neurips_challenge/cv4.sh --list                         # the 10 kept recipes (best ensemble + its baseline)
bash neurips_challenge/cv4.sh sqf_dbn_optx_things 1 2        # screen a recipe (10 test subjects)
bash neurips_challenge/cv4.sh sqf_dbn_optx_things 3 4        # confirm it (all 20 subjects)
.venv/bin/python neurips_challenge/cv4_summary.py --ref sqf_dbn_things  # tables, paired decisions, audit
.venv/bin/python neurips_challenge/cv4_summary.py --self_test
```

---

## 1. What the protocol has to imitate

Track 1's sealed phase (Oct 28 - Nov 21) ranks submissions on an evaluation cohort that the organisers keep private.

**From the rules and the organisers' Discord answers (2026-09-23 / 2026-09-30):**
- The sealed set contains **both unseen subjects and unseen images**. The user recalls ~11 new participants.
- It is recorded with the **same 32-channel Emotiv FLEX2 montage and channel names** as Alljoined-1.6M (NEMAR nm000134).
- It goes through the **same preprocessing pipeline** as the warm-up. No additional training data will be released.
- **Metric: neuralbench's subject-aggregated retrieval.** Raw predicted embeddings of the repeated presentations of an image are averaged within each subject. They are then ranked by cosine against the unique candidate targets (top-5). Per-subject scores are averaged, so every subject counts equally. `objective.py` will be updated to "the same pooling steps, in the same order, as neuralbench".
- **Unknowns:** the number of repeats per image and the gallery size will not be shared ("models should be agnostic").
- **Batches:** evaluation batches will be shuffled. Batch-level statistics may be used, but the batch size may change.

**Consequences for evaluation:**
- **Test subjects must be absent from training.** A model scored on subjects it was trained on (the warm-up setting, or our pooled runs) measures something else. Scratch LaBraM scored 47 pooled, against TSConv's 28, and ~30 on held-out subjects.
- **Test images must be unseen.** That is automatic here: Alljoined's 200 test images belong to concepts that never appear in the 16,540 training images.
- **Score per subject, then average over subjects**, exactly as the sealed metric does.

## 2. Data

- **Source:** Alljoined-1.6M, NEMAR nm000134 v1.0.3. 20 subjects, 4 sessions each; sub-08's `ses-02old` retake is excluded, as neuralfetch does.
- **Preprocessing** (the grader's pipeline):
  - neuralbench `EegExtractor`: 120 Hz, 0.1-75 Hz band-pass, 50/60 Hz notch, RobustScaler per recording.
  - Epoching by `neurips_challenge/build_alljoined_cache.py`: -0.2 to 0.8 s (120 samples), baseline over the first 0.2 s, clamp at ±20.
  - These are exactly the warm-up THINGS-EEG2 parameters: same extractor config hash `...notch_filter=(50,60)-8c9c9e77`.
- **Files:** `/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/sub-XX/`.
  - `train.npy`: (1654 concepts, 10 images, 4 repeats, 32 ch, 120 t).
  - `test.npy`: (200 images, 1, 80 repeats, 32, 120).
  - The channel order is in `info.json`.
  - The stimuli are byte-identical to THINGS-EEG2's, in the same order.
- **Targets:** DINOv2-giant, 2/3 depth, mean-pooled, 1536-D: `data/things_eeg/image_feature/dinov2-giant_track1`.
- **Single trials:** training uses all 4 repeats as separate trials, never averaged. The grader sends single epochs too.
- **No other dataset is mixed in** (THINGS-EEG2 subjects are on hold; see §11).

## 3. The folds (frozen)

| fold | test subjects | mean pooled score of the fold |
|---|---|---|
| 1 | 2, 4, 5, 11, 15 | 48.1 |
| 2 | 7, 14, 16, 17, 20 | 47.3 |
| 3 | 3, 8, 10, 12, 19 | 46.5 |
| 4 | 1, 6, 9, 13, 18 | 47.2 |

- Each fold trains on the other 15 subjects and tests on its 5. Every subject is a test subject exactly once.
- **How they were drawn:** the 20 subjects were sorted by their sealed-rule top-5 in the 2026-09-29 pooled scratch-LaBraM run, then dealt in snake order (1-2-3-4-4-3-2-1-...). Every fold therefore mixes easy, medium and hard subjects, and the fold means differ by less than 2 points.
- **Hard subjects are included on purpose:** sub-09, 17 and 19 (the weakest decoders) and sub-15 (a quiet recording that electrode-token models collapse on). The sealed cohort will contain such people, and the sealed metric weights them equally.
  - Dropping 9, 17 and 19 from training was tested on 2026-09-29/30 and never helped (`results/alljoined/track1/drop3/`), so they stay in training too.
- The folds are hard-coded in `cv4.sh` (`FOLDS=`). **Never change them**: all cv4 results are comparable only because the folds are fixed.

## 4. The score

`neurips_challenge/score_track1.py`, field `subject_agg_by_subject`:
- for each test subject, the model's raw 1536-D predictions for the 80 repeats of each test image are averaged;
- each average is ranked by cosine against the 200 unique test-image targets;
- the result is the top-5 hit rate over the 200 images (chance is 2.5%).

**Every run scores every test subject at every epoch on a seeded half of its trials, ~40 of the 80 repeats per image** (`--sealed_test_every 1 --sealed_test_fraction 0.5`). The results go to `sealed_per_epoch.jsonl` in the run dir.
- Scoring uses the current (EMA) weights, the official 120 Hz input and the grader's no-subject-id path.
- **A recipe's epoch budget** is the epoch with the best mean over all its test subjects.
- **Its cv4 score** is that epoch's mean: 20 subjects once all folds are done, 10 after the screen.
- **Paired comparisons** use each recipe at its own budget.
- This is test-subject selection of one hyperparameter. That is accepted on purpose: the aim is the best sealed-set model, and the final all-subject model is trained with exactly this budget (§10).

## 5. Training (what every run shares, and the recipes)

**Shared flags** (`alljoined_run.sh`):
- direct 1536-D head; `--img_l2norm`, init temperature 1, asymmetric InfoNCE;
- grad clip 1; one-cycle learning rate (10% warm-up) at 1e-3 unless the recipe says otherwise; weight decay 0.05;
- seed 33; `--eval_unique_gallery`;
- all 32 electrodes, named (LaBraM/REVE use them as tokens);
- `--select_best_on fixed --val_concept_ratio 0`. There is **no validation split**: see §8.

**Recipes** (`cv4.sh recipe()`). The schedule lengths below set the trained range. They come from the per-epoch sealed curves (10-repeat estimate) of the earlier held-out runs. The budget is then picked inside that range (§4), and the audit (§7) catches a range that is too short.

| recipe | model | schedule | evidence for the schedule |
|---|---|---|---|
| `tsc` | TSConv_parameterizable, pool 25/2, batch 1024 | 12 epochs | 16-epoch runs peaked at epoch 6 (18.2) and ended at 17.0; a 40-epoch run peaked at 7 and fell to ~13 |
| `sqf` | OrthoFastTSConvSqueezeformer, pool 25/2, trunk 512, batch 1024 | 30 epochs | still rising at epoch 16 (16.9 → 20.8); averaged-EEG plateaued at ~11 |
| `atm` | ATM, no subject token, trunk 1024, batch 1024 | 20 epochs | plateau at epochs 10-20, slow decline to epoch 40 |
| `lab` | scratch LaBraM, width 400 (16 conv channels, 8 heads), batch 128, bf16, EMA 0.9998, channel dropout | 20-epoch schedule, stop after 6 | 8-epoch full anneal peaked at epochs 4-6 (23.1) and fell at 8 (20.9); the warm-up already found that "the anneal overfits", hence a long schedule with an early stop |
| `plab_gain` | pretrained LaBraM, lr 5e-4, EMA, `--amp_gain_range 2` | 20-epoch schedule, stop after 6 | flat from epoch 4 to 8 |
| `reve_z` | pretrained REVE, lr 2e-4, EMA, `--reve_input_normalization trial_zscore` | 10-epoch schedule, stop after 4 | flat from epoch 2; averaged-EEG declines after 2 |

**Superseded after the audit (2026-10-01):** `lab` and `plab_gain` were still rising at their stop epoch on 15-subject folds. They are replaced by `lab_s10` / `plab_gain_s10`: same 20-epoch schedule, stop after 10, so epochs 1-6 are unchanged.

**Notes on the flags:**
- `--amp_gain_range 2` (added 2026-09-30) multiplies every training trial by one global gain, drawn log-uniformly in [0.5, 2].
- REVE's `trial_zscore` is REVE's own default; the warm-up recipe had turned it off.
- On the 6-subject screen, these two were the best variants (+2.3 / +2.6 over scratch LaBraM). They are the reason those two recipes are in the reference set.

**Why these six:** they are the reference set the user asked for on 2026-10-01 (TSConv, Squeezeformer, scratch LaBraM, pretrained LaBraM + gain, REVE + z-score, ATM).

**Recipes tried earlier and dropped** (pairs screen, 2026-09-30):
- SubjectMix (raw-EEG pairwise, grouped batches): TSConv −0.25, Squeezeformer −1.75.
- EMA on TSConv and Squeezeformer: ±0.2.
- REVE + gain: +0.4.
- ATM + gain was only +1.2 over ATM, so plain ATM is the reference.

**Adding a recipe:** add one line to `recipe()` in `cv4.sh` under a **new name**, and never edit an existing line. Results are keyed by recipe name, so an edit would silently mix two recipes under one name. The new recipe must also pass the training-length audit (§7) before its numbers count.

## 6. Decisions: noise floor and rules

Per-subject noise measured on the earlier held-out runs (sealed top-5):

| source | SD |
|---|---|
| between subjects | ~9 |
| paired difference, two variants of the same model | 1.8-2.4 |
| paired difference, different model families | 2.3-4.1 |
| same run, last vs lowest-test-loss checkpoint (a training-noise floor) | 2.4 |

So with *n* subjects the 2-sigma uncertainty of a mean paired gain is roughly 5/√n for a same-model tweak and 7/√n across families:

| n subjects | same-model tweak | different family |
|---|---|---|
| 10 (screen) | ±1.6 | ±2.2 |
| 20 (full) | ±1.1 | ±1.6 |

Effects under ~1 point cannot be resolved with 20 subjects. Screens over-estimate the winners: pretrained LaBraM was +1.75 on 6 subjects, +0.7 on 12, and tied TSConv.

**Rules, applied mechanically by `cv4_summary.py --ref <baseline>`:**

1. **Screen** on folds 1-2 (10 subjects), paired per subject against the baseline:
   - **drop** if the mean gain is ≤ 0;
   - **promote** to folds 3-4 if the mean gain is ≥ +1 and at least 6 of 10 subjects improved;
   - otherwise unclear: confirm only if the idea matters.
2. **Confirm** on all 4 folds (20 subjects). **Adopt** if:
   - the mean gain is ≥ +1.0 (same arm) or ≥ +1.5 (different arm), **and**
   - the Wilcoxon signed-rank p < 0.1.
3. **One seed per run.** The dominant noise is between subjects, which more seeds do not reduce.
4. **Comparisons are paired on identical subjects.** Never compare a 10-subject mean with a 20-subject mean.

## 7. Training-length audit (every recipe, every time)

`cv4_summary.py` flags a recipe **UNDERTRAINED** when its last trained epoch beats every earlier epoch by more than 0.3 (judged on ≥ 2 folds). Its schedule is too short: rerun it with a longer schedule **under a new recipe name** (e.g. `sqf_e40`). A peak earlier in the range is not an error, because that peak is the budget.

## 8. Why there is no validation split

- **Concept validation does not track new-subject performance.** Holding out 10% of training concepts, from the training subjects, measures a different generalisation axis: it stayed flat at 1-2.5% top-5 over a 1,650-image gallery while held-out-subject scores moved.
- **Lowest test loss picks bad epochs for fine-tuned FMs.** It chose epoch 1 for REVE (−4.6), because contrastive loss rises while ranking still improves.
- **The test-subject curve picks the budget** (§4); there is no separate selection set.
- **If a recipe ever needs early stopping**, the right signal is a held-out *training subject* (one fewer training subject), not held-out concepts.

**No lockbox:** keeping 3-4 subjects untouched would cost 15-20% of the decision data. The sealed phase itself is a partial lockbox: 1 submission/day, best of the last 5.

## 9. Running it

- **Where it runs:** 3 RTX 3080s total, including the interactive session's node. Batch jobs use `--partition=Brain_GPU --account=brain --qos=low --gres=gpu:rtx3080:1 --cpus-per-task=8 --mem=30G`.
- **Worker pattern** (screen folds of every recipe first, then the confirmation folds):
  ```bash
  W='for r in $ORDER; do bash neurips_challenge/cv4.sh $r 1 2; done; for r in $ORDER; do bash neurips_challenge/cv4.sh $r 3 4; done'
  sbatch ... --export=ALL,ORDER="tsc sqf atm lab plab_gain reve_z" --wrap="$W"
  ```
  Give each worker a different `ORDER`. `alljoined_run.sh` claims each run (`<run dir>/../.claim/job`). Another worker skips a claimed run unless the claiming Slurm job is dead.
- **Data staging:** each job copies its subjects' Neurobridge fp16 memmaps to `/tmp/$USER-aj-$SLURM_JOB_ID` once (`stage_eeg_cache.py`; ~1.5 min on batch nodes, up to 30 min on the 1.9 GB interactive slot). Later runs in the same job reuse the copy. If `/tmp` is short of space it reads the NAS directly, which is slower but gives the same numbers.
- **Time per run, rough:**
  - tsc ~35 min; sqf ~1.5 h; atm ~1.2 h; lab ~1.5 h; plab_gain ~1.2 h; reve_z ~1.1 h.
  - Each includes ~10 min of scoring 5 subjects × 16,000 epochs.
  - A full recipe is 4 runs, ≈ 5 GPU-h; a screen is half that.
- **Where things land:**
  - logs: `logs/cv4_*.out`;
  - runs: `results/alljoined/track1/cv4/<recipe>/fold<k>/<arm>/<timestamp>-<arm>/`, holding `train_config.json` (the flags), `epoch_metrics.csv`, `train.log`, `checkpoint_test_best.pth` and `track1_score.json`.

## 10. The final sealed submission

1. **Retrain the adopted recipe on all 20 subjects**, keeping the same schedule but stopping at its cv4 budget (`--stop_after_epoch <budget>`). Use `SPLIT=pooled bash neurips_challenge/alljoined_run.sh <arm> <recipe flags> --stop_after_epoch <budget> --select_best_on fixed --val_concept_ratio 0`. Its "test" is the training subjects' own test images, a sanity check only. The warm-up's full-data refits gained 1-2 points over split-trained models.
2. **The retrained model has no held-out number, so check it against the cv4 fold models:**
   - matching training curves;
   - pooled score at least the fold models';
   - similar prediction-norm statistics.
3. **Alternative:** submit the 4 cv4 fold models as an ensemble. Each one was tested, and the average reduces model noise. It fits the 60 min A100 / 15 GB limits. It is untested here; decide with a cv4-style comparison if it matters.
4. **Robustness checks** before submitting:
   - fewer repeats per image (1-20), since the sealed count is unknown;
   - shuffled, mixed-subject batches of different sizes;
   - grader channel order: permute by name from `chs_info` (see the NeurIPS Track-1 rules memory).

## 11. Known caveats

- **Training-set size:** cv4 trains on 15 subjects and the final model on 20. Transformers may gain more from extra subjects than conv models, so cv4 can slightly undersell them against TSConv. Comparisons within a family are unaffected.
- **Quiet-subject collapse:** electrode-token models (LaBraM, REVE, ATM) score 7-13 on sub-15, against TSConv's 22. The cause is partly input scale: a ×1.8 test-time gain lifts scratch LaBraM from 7 to 18 but hurts normal subjects. Training-time global gain augmentation did not fix it, so the remaining cause is unknown. `neurips_challenge/diag_input_scale.py` reproduces the diagnosis.
- **sub-09:** its grand-average ERP barely correlates with the group's (median r 0.12 vs 0.7-0.8), possibly a timing offset. Not investigated.
- **THINGS-EEG2:** its 10 subjects could be added to training (public; all 32 Emotiv channels exist in its 63; same preprocessing and images). `neurips_challenge/export_things_to_alljoined.py` is ready but has not run; the user put dataset mixing on hold.
- **Unknown sealed repeat count:** cv4 scores with all 80 repeats. With fewer sealed repeats, absolute numbers will be lower and, in principle, the ranking could change. Check before the final choice (§10.4).

## 12. History: protocols this one replaces

| date | protocol | results | outcome |
|---|---|---|---|
| 2026-09-29 | pooled: train and test on all 20 subjects | `results/alljoined/track1/pooled/` | sanity check against ENIGMA (6.00 / 16.25 intra-subject); misleading for the sealed phase (LaBraM 47 vs TSConv 28) |
| 2026-09-29 | single held-out split, subjects 1 4 13 15 (rng 2026) | `results/alljoined/track1/heldout/tsc/` | cancelled; launched without being asked |
| 2026-09-29/30 | drop3: 6 disjoint pairs, with/without 9 17 19 (`drop3_chain.sh`, removed) | `results/alljoined/track1/drop3/` | keep all 20 subjects |
| 2026-09-30/10-01 | pairs screen: 3 of those pairs, confirmations on the other 3 (`heldout_chain.sh`, `heldout_summary.py`, removed) | `results/alljoined/track1/heldout/<fold>/<variant>/<arm>/` | the recipe findings in §5 |

The removed scripts are superseded by `cv4.sh` / `cv4_summary.py`; their results stay where they are. The cancelled pairs confirmations (REVE z-score, pretrained LaBraM + gain, base REVE on pairs 1-6 / 12-13 / 4-5) are partly complete in that tree and are not used for decisions.
