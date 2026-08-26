# Best 2- and 3-model score ensembles (THINGS-EEG2 inductive LOSO, 200-way)

Exhaustive search over every checkpoint in `results/things_eeg/` that is a verified
10-fold LOSO run with `eval_mode=plain_cosine` (300 runs; 42 distinct
encoder x image-backbone configurations dumped and searched).
Members are combined by uniform mean of the L2-normalised cosine score matrix
(`ensemble_experiments/synthetic_subjects/score_npz_ensemble.py` convention). Metric is
`best top1 acc`. Reference single model = 35.90 (pairwise SubjectMix, seed 3300).

| k | members | top1 | top5 | vs 35.90 | Wilcoxon |
|---|---------|------|------|----------|----------|
| 1 | ge100 | 37.05 | - | +1.15 | - |
| 2 | atm_iv + tsconv_eva | **41.65** | 73.40 | +5.75 | p=0.035, 8/10 |
| 3 | atm_iv + tsconv_eva + tsconv_vith | **44.60** | 76.25 | +8.70 | p=0.0098, 8/10 |

Best-3 per fold: 50.0 53.5 36.5 37.5 47.0 34.0 41.0 37.5 51.5 57.5

## Members
| tag | encoder | image features | solo top1 | run |
|-----|---------|----------------|-----------|-----|
| atm_iv | ATM | InternViT-6B_layer28_mean_8bit | 35.20 | inter-subjects/eeg_encoder_sweep_20260426-184949/ATM_seed3300 |
| tsconv_eva | TSConv | EVA02-E-14_layer35_mean | 33.95 | inter-subjects/eeg_encoder_architecture_sweep_mixup_20260429-221100/EVA02-E-14_layer35_mean_TSConv_seed3300_mixup |
| tsconv_vith | TSConv | ViT-H-14_layer10_mean | 32.65 | inter-subjects/eeg_encoder_architecture_sweep_mixup_20260429-221100/ViT-H-14_layer10_mean_TSConv_seed3300_mixup |
| ge100 | TSConv_parameterizable | InternViT-6B_layer28_mean_8bit | 37.05 | synthetic_subjects/promising_allfolds/group_e100/seed3300 |

## Findings
- **Image-backbone diversity beats member accuracy.** Every top combination pairs one
  ATM encoder with TSConv encoders trained against *different* image feature spaces.
  The best single model (ge100, 37.05) is absent from both winners: it is displaced by
  tsconv_vith, which is 4.4 points worse alone but decorrelated.
- Same-backbone ensembling saturates early: the best all-InternViT combination found
  previously (ge100 + 3 pairwise seeds) reaches only 39.45 with four members, below the
  two-member cross-backbone 41.65.
- Adding 15 further encoder x backbone configurations (all <=25 solo) changed neither
  top list: members that weak contribute nothing.

## Caveats
- Taking the argmax over 861 pairs / 11480 triples on the same 10 folds is selection on
  test. Nested check (select on 9 folds, score the held-out fold): 3-model returns
  44.60 -- the same combination wins on every fold subset, so it is stable. The 2-model
  choice is softer: nested gives 39.30 vs 41.65 selected.
- Uniform weights only; no weight search (that would add more test-set selection).
- Excluded as unbuildable: `LaBraM` and the three `multihead/zip6*` runs -- `train.py` no
  longer supports those encoder types, so `evaluate.py` cannot reconstruct them.
  zip6_e50_het scores 43.65 as one checkpoint, but is internally a 6-head model over 3
  image spaces, so it is not a fair single model here -- it independently corroborates
  the diversity finding.

Reproduce: `.venv/bin/python ensemble_experiments/legacy/scratch_claude/ens_search.py` (manifest in
`ensemble_experiments/legacy/scratch_claude/manifest.json`, score dumps in
`results/things_eeg/synthetic_subjects/ensemble_screen/dumps/`).

---

# Shrunk-width rerun: does the ensemble survive a parameter budget?

Same winning k=3 triple, retrained with `eeg_backbone_dim` 1024->128 and
`feature_dim` 512->128. Recipe otherwise identical to the three originals
(50 ep, batch 1024, lr 3e-4, pairwise SubjectMix a=0.5 raw_eeg, spi 9, grouped
batch, multi-positive, seed 3300). `atm_iv` was not retrained: it is already
bb=128/fd=128 on this recipe. Trainable params = encoder + eeg_projector +
img_projector; baseline reference = 4.754M @ 35.90.

| ensemble | top1 | top5 | params | vs 35.90 | paired |
|----------|------|------|--------|----------|--------|
| **DIVERSE-3 shrunk** (atm_iv + eva + vith @128) | **42.25** | 75.80 | **2.34M (0.49x)** | +6.35 | p=0.0039, 9/10 |
| SEEDS-3 control (eva @128, seeds 3300/1/2) | 34.65 | 68.25 | 1.65M (0.35x) | -1.25 | p=0.40, 3/10 |
| DIVERSE-3 full size | 44.60 | 76.25 | 9.19M (1.93x) | +8.70 | p=0.0098, 8/10 |

DIVERSE-3 vs SEEDS-3: **+7.60pp, p=0.0020, 10/10 folds.**

Solo after shrinking: atm_iv 35.20 (unchanged), eva 33.50 (-0.50), vith 31.65
(-1.00), eva-s3301 32.60, eva-s3302 32.85.

## Findings
- Three models at **half the baseline's parameters** beat it by 6.35 points.
- The diversity effect is not a large-model artefact: at matched small scale,
  same-backbone seed ensembling gains only ~1.2pp over its best member and lands
  *below* the baseline, while cross-backbone diversity gains 8.75pp over its best
  member.
- Shrinking 3.9x cost 2.35pp of ensemble accuracy vs ~0.75pp of mean solo
  accuracy, so width reduction slightly erodes complementarity as well as
  individual quality.
- ATM is *better* small (35.20 at bb=128 vs 30.30 at bb=1024); TSConv is
  near-flat (an 11.7x cut costs ~0.7pp with SubjectMix).

## Caveats
- SEEDS-3 is not exactly parameter-matched (1.65M vs 2.34M) because it contains
  no ATM, the heaviest member. The 42% gap cannot plausibly account for 7.6pp
  when an 11.7x cut costs 0.7pp.
- Single seed for the diverse triple; the control supplies the only seed spread
  (eva solo across 3 seeds: 33.50 / 32.60 / 32.85, SD 0.47).
- The `projector_only_bb*` width sweeps in this repo ran with
  `subject_mixup_mode='none'`; their "smaller is monotonically better" trend is an
  unregularised-overfitting effect and does not transfer to the mixup recipe.
