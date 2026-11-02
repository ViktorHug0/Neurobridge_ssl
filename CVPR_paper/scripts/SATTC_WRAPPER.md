# SATTC on our embeddings (E3.1)

`sattc_on_embeddings.py` runs SATTC's own `evaluate_model` (SATTC-CVPR2026 commit 0cdfd1b,
`Retrieval/run_sattc_loso.py`: SAW, then Ada-CSLS, then the structural expert, then PoE). It imports their module
and does not re-port or edit any of it.

## Method
- **Config.** Built by their own argparse and normalisation code. `_run_training_pipeline()` runs with
  `--seed 3300 --data_path ...`, the same flags as our `sattc_repro.sbatch` run, until it reaches `set_seed`; the
  parsed `args` are taken there. The final-stage lines after that point are replayed: subject slots,
  `use_subject_unk`, `deepcopy` and `stage='final'`. All SATTC hyperparameters keep their defaults. We tuned none of
  them.
- **Train statistics.** At test time SATTC uses statistics from the training split. On the first `train_model`
  call, `config._cache_auto_whiten_{text,img}` caches a global ZCA whitening (shrink 0.5) of the train-split
  candidates: 1654 text features and 16,540 image features. `evaluate_model` then whitens the test gallery with it.
  The wrapper fills that cache with their own `_resolve_auto_whitening_state`:
  - **Validation:** from their ViT-H-14 train features.
  - **Our encoder:** from our image projector applied to all 16,540 train images
    (`<run image_feature_dir>/image_train.npy`, read from each run's config; layer 28 for all paper encoders).

  Everything else in SATTC is transductive on the test set: SAW on the 200 queries, plus CSLS, the evidence cases
  and PoE.
- **Pass-through model.** `forward(x, sid)` returns `x`. It also has the two attributes `evaluate_model` reads,
  `logit_scale` and `loss_func`. `logit_scale` cancels exactly under their default automatic logit centring and
  scaling. It has no `.encoder`, so the subject-ID logic does nothing.
- **Loader.** One batch in their test tuple format, `(x, label, text, text_feat, img, img_feat)`.
- **Text features.** These never enter SATTC's ranking. They only feed the reported test loss. For our encoder we
  pass the image gallery in their place.
- **mnn_precheck.** Turned off (`enable_mnn_precheck=False`). It only writes diagnostic files and never changes the
  logits.
- **Metric.** 200-way top-1 and top-5, which has no k-way random sampling. That is the `test_accuracy` / `top5_acc`
  pair in their final CSVs.

## Validation gate (passed)
We ran their saved `sub-XX_final.pth` (epoch 15) on their test set through `evaluate_model`, both directly and
through the pass-through wrapper on the embeddings it produced:

| fold   | logged (their GPU run) | GPU (T1000), model / wrapper | CPU, model / wrapper |
|--------|------------------------|------------------------------|----------------------|
| sub-01 | 12.0 / 31.0            | 12.0 / 31.0                  | 12.0 / 31.0          |
| sub-02 | 12.0 / 30.0            | 12.0 / 30.0                  | 12.0 / 29.5          |

- The wrapper always matches their model exactly.
- On GPU, both folds match the logged numbers exactly.
- The 0.5 pp CPU gap on sub-02 top-5 is a single query at the rank-5 boundary. It comes from CPU-vs-CUDA float
  differences in their EEG forward pass, not from the calibration.
- When this was run, no other folds were finished.

## Results: atm31_valcon seed3300
Setup: 80-repetition queries, `checkpoint_testsel_best.pth`, embeddings encoded on CPU. CSV:
`results/things_eeg/cvpr/competitors/sattc/on_atm31_valcon.csv`.

| subject | plain top-1 / top-5 | SATTC top-1 / top-5 |
|---------|---------------------|---------------------|
| 1  | 42.0 / 78.0 | 48.0 / 83.0 |
| 2  | 43.0 / 73.0 | 47.0 / 78.5 |
| 3  | 27.0 / 51.5 | 44.0 / 73.5 |
| 4  | 25.5 / 62.5 | 34.5 / 65.5 |
| 5  | 47.5 / 77.0 | 49.5 / 76.5 |
| 6  | 29.0 / 53.0 | 42.5 / 69.0 |
| 7  | 27.0 / 61.0 | 47.5 / 75.0 |
| 8  | 33.0 / 68.0 | 36.5 / 67.5 |
| 9  | 39.0 / 69.0 | 47.0 / 78.5 |
| 10 | 47.5 / 81.5 | 57.0 / 88.5 |
| **mean** | **36.05 / 67.45** | **45.35 / 75.55** |

The plain column reproduces the known atm31_valcon numbers (mean 36.05, sub-01 42.0). For reference, on the same
embeddings SAGE reaches 82.30 and SCORE 65.20.

## Caveats
- **Gallery whitening, sensitivity only.** We tried two other sources for the whitening. These are post-hoc
  checks on the test set and are not a tuned result:

  | gallery whitening                   | mean top-1 / top-5 |
  |-------------------------------------|--------------------|
  | none (`global_auto_whiten=False`)   | 50.45 / 82.45      |
  | stats from the 200 test items       | 45.85 / 77.20      |

  SATTC's train-statistic whitening default (45.35) hurts on our 128-d projector space. Even the best of these
  variants stays far below SAGE (82.30).
- The SATTC numbers above are computed on CPU. Rank-5 boundary flips of about ±0.5 pp per subject against a GPU run
  are possible, as the sub-02 CPU row shows.
