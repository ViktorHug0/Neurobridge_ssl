# NeuroBridge / SAGE

This repo is about decoding images from EEG, mainly on THINGS-EEG-2 in the inter-subject setting: train on several people, then test on a held-out one.

At a high level, the pipeline is pretty simple. EEG gets preprocessed, images are represented with frozen visual features, and an EEG encoder is trained to land in the same embedding space so EEG trials can retrieve the right image. The main training recipe is a contrastive loss with grouped multi-positive batches, plus optional variants such as subject mixup.

At test time, the interesting part is the adaptation side. Besides plain cosine retrieval, the repo also has the SAW / CSLS / Sinkhorn / soft-Procrustes tools that reshape the held-out subject's query geometry and give a much stronger transductive evaluation setup.

All accuracies below are **200-way closed-set top-1** on the LOSO held-out subject, averaged over the 10 THINGS-EEG-2 subjects unless stated otherwise. Read the protocol label on every number — `test-selected`, `ValCon` and `LOSO-val` are not interchangeable, and mixing them is the easiest way to overstate a result. `AGENTS.md` is the working map.

---

## SAGE — the paper (NeurIPS submission 28770)

Two operating modes on the same trained encoder.

**SAGE-zero-shot** — cross-subject mixing (SubjectMix) for subject invariance, evaluated inductively with plain cosine:

| | Top-1 |
|---|---|
| prior SOTA cross-subject baseline | 21.8 |
| **SAGE-zero-shot** (TSConv, 10-fold mean) | **35.9** |

**SAGE-TTA** — test-time adaptation as transductive calibration over the full unlabeled query set and the fixed 200-image candidate set. Per-encoder peak after calibration:

| Encoder | Top-1 | Top-5 |
|---|---|---|
| ATM | **77.2** | 92.0 |
| TSConv | 68.5 | 87.8 |
| EEGConformer | 65.0 | 86.8 |

Note the ranking reversal: TSConv is the strongest *inductive* encoder, ATM the strongest *after* calibration.

Evaluated on three datasets — THINGS-EEG-2, AllJoined-1.6M and THINGS-MEG — so the claim covers different acquisition contexts and recording modalities. The AllJoined and MEG results are in the paper appendix; their result trees are listed in `PROTECTED.md`.

**What the headline is and is not.** 77.2 is transductive: it uses all 200 unlabeled test queries jointly. It is not inductive per-trial decoding, and it depends on a closed, known candidate pool.

## Rebuttal / author-response results

Everything here lives in `results/things_eeg/tta_rebuttal/` and `scripts/things_eeg/tta_rebuttal/`, with the narrative in `neurips_author_response_period/`.

**Hyperparameter selection is honest.** The published TTA config was originally the argmax of a grid scored on the held-out test subject. Re-selecting without touching test — freeze the three flat axes at their cross-fold marginal optima and LOSO-select only `tau` — reproduces the published number with no leak: **68.30 blind vs 72.75 test-oracle**. Joint-argmax cross-validation is *pessimistic* (67.30) because it overfits a flat axis. Label-free selection is insufficient, about 10 points short.

**Both contributions transfer to an EEG foundation model.** Full-parameter fine-tuning of LaBraM in place of TSConv, under the AVDE objective:

| | plain | + SAGE-TTA |
|---|---|---|
| no SubjectMix | 27.05 | 49.95 |
| + SubjectMix | 31.85 | **63.10** |

SubjectMix is worth +4.80 plain and +13.15 under TTA; TTA is worth +22.90 / +31.25. They are synergistic — but LaBraM+mix+TTA (63.10) still **loses to TSConv+mix+TTA (68.30)**. A big pretrained foundation model does not beat the lightweight encoder; the contribution is the alignment recipe, which is encoder-agnostic. (CBraMod is far worse, 8.5 on sub-01, and was deferred.) See `scripts/things_eeg/inter-subject-foundation.sh`.

**Where SAGE-TTA breaks, measured rather than asserted.**

- *Too few queries*: break-even at about N=75. At N=25, TTA gives 13.7 against 36.1 for plain cosine.
- *Open / incomplete candidate set*: the bottleneck is **support identification, not alignment**. An oracle restricted to the truly-present items is flat at 43.7–47.6 for every coverage level, including 25%. Relaxing the Sinkhorn marginal (unbalanced OT, `rho`) widens the deployable range — free beats balanced across the whole partial regime and crosses plain cosine at U≈75 instead of U≈100 — but label-free support estimation captures essentially none of the oracle headroom. Below U≈75, plain cosine wins and rotation is the fragile component.
- *Streaming deployment*: with a strictly causal re-fit, damped adaptation crosses plain cosine at ~110 arrivals (about 7 minutes at 0.2 s SOA), though all-subject reliability only from ~400. Cold start is a regression, not a flat spot, so a damping schedule is a safety requirement.
- *Over-rotation*: the soft-Procrustes loop systematically over-rotates. The best mixing coefficient saturates at 0.5–0.7 and never climbs to 1.0 as the buffer grows (10/10 subjects, Wilcoxon p=0.002) — roughly 1.3–1.6 points available from a one-line change to the published config.
- *SAW is not optional*: it buys nothing standalone but everything through the rotation it enables — dropping it costs the best arm up to 11.3 points.

**Deployment framing.** Never compare the calibrate-then-deploy number (49.9) against the 68.3 headline — different query SNR. At matched 40-repetition SNR: plain 31.7, frozen map per-trial 49.9, frozen map + CSLS 54.6, full transductive re-fit 54.4. So the geometric map transfers at no measurable loss; what deployment gives up is a batch hubness correction.

**SubjectMix sensitivity.** `alpha_mix ~ Beta(0.5, 0.5)`. Sweeping two orders of magnitude gives 33.4 / 33.9 / 34.5 / 35.0 / 35.1 for alpha 0.01 → 1.0, so the method is not sharply tuned. Smoothing-style augmentations never beat no-augmentation at any strength; SubjectMix beats the best of them by +5.35 (p=0.0026). Drivers in `scripts/things_eeg/subjectmix_rebuttal.sh`.

## Score ensembling

Nested leave-one-fold-out member selection saturates around 44 (44.45 at k=4); code in tag `pre-cleanup-2026-11-02`.

## Sparse CLIP

Confidence-gated ReLU at `feature_dim` 32768 kept 113 active dimensions (142× sparsity) for −0.2 top-1 on 5 folds; code in tag `pre-cleanup-2026-11-02`.

## Image reconstruction and captioning

ENIGMA reproduced within ~2pp on sub-01 reconstruction metrics, and the best EEG caption arm reached cwBLEU 10.06 against a 29.38 real-image ceiling; code in tag `pre-cleanup-2026-11-02`.

---

## Running things

The venv's `activate` can resolve elsewhere on this machine, so call the interpreter directly. `pytest` is not installed, and there is no test suite.

```bash
.venv/bin/python train.py --help
```

If you just want the main pieces, start here:

- `train.py` — training entrypoint (LOSO loop, loss, per-epoch eval, checkpoints)
- `evaluate.py` — standalone eval from `checkpoint_test_best.pth`, including the full SAGE/SATTC refinement
- `module/dataset.py`, `module/loss.py`, `module/sampler.py`
- `module/util.py` — SAW, CSLS, Sinkhorn, soft Procrustes, retrieval
- `module/eeg_encoder/model.py` — TSConv, EEGNet, EEGConformer, …
- `module/eeg_encoder/foundation.py` — LaBraM / CBraMod full fine-tuning

Main THINGS-EEG sweeps:

- `scripts/things_eeg/inter-subjects.sh` — canonical LOSO sweep
- `scripts/things_eeg/inter-subject-tsconv-val-best.sh` — TSConv + SubjectMix, LOSO-val checkpoint selection
- `scripts/things_eeg/inter-subject-SAATC.sh` — SAGE-TTA
- `scripts/things_eeg/inter-subject-foundation.sh` — foundation-model arms

Main adaptation scripts:

- `scripts/things_eeg/progressive_sattc_candidate_sweep.py`

There are still some older utilities around for augmentation and feature extraction, but they are not the main path anymore.

**Before deleting anything under `results/`, read `PROTECTED.md`.** A September 2026 cleanup removed 227 GB of checkpoints — `.pth` only, so all 48,567 run records and every number above survive — but the SATTC paper track never saved weights in the first place, and several ablation families are now results-only.
