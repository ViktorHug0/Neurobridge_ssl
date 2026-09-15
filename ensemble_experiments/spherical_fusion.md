# Frozen-model spherical fusion

Uses existing seed3300 TSConv-IV33 compact single and ATM-IV28 ValCon
checkpoints (pairwise mixup, alignment128). No training or checkpoint reselection.

For each LOSO fold, fit uncentered orthogonal Procrustes B->A using all unique
source-training images, excluding the 165 held-out validation concepts. Apply
the same map to ATM image and EEG vectors. Reflections are allowed. No centering,
whitening, test alignment, or test-time adaptation.

Select one common query/candidate interpolation weight per method on the original
nine source subjects' concept-validation EEG. Candidate panels are a seed3300
permutation of the 1650 unique validation images, partitioned into 200-image
panels. The final 50-query-image panel is padded with 150 earlier candidates.
Every query is evaluated exactly once, with 200 candidates and no repeated
candidate identities. Weight grid: 0, .25, .5, .75, 1. Ties prefer .5, then the
lower ATM weight. This reuses checkpoint-selection validation, not a new nested
validation split; all outer test labels remain excluded from fitting/selection.

Prespecified outputs: solo retrieval, cross-model retrieval, cross terms only,
four-term unnormalized scores, and equal-weight/raw-row-z/SLERP/normalized-linear
fusion. Each of the four fusion families also reports its validation-selected
weight. Equal-weight SLERP and normalized-linear fusion must coincide. Endpoints
allow selection to fall back to either solo model. Test curves for unselected
weights are not reported. Frozen test embeddings are read from existing archives.

Artifacts: results/things_eeg/spherical_fusion_20260909. Each fold saves alignment,
validation embeddings/identities, locked selection and checkpoint hashes before
loading test archives, test scores, and completion metrics. Fold-idempotent queue
with process lock and source hash guard. GPU only used for validation embedding
extraction and image projection; no mixup is applied at evaluation.

Run from repo root with the project venv activated:

```bash
python -m unittest ensemble_experiments.test_spherical_fusion
python -u -m ensemble_experiments.spherical_fusion
```

Interpretation: this remains two-encoder inference. A gain supports compatibility
of the learned spaces, not proof that one encoder can attain the same accuracy.

## Completed results (2026-09-09)

All ten folds completed on sl-tp-br-513 in 53.6 seconds of measured fold work
(excluding Python import startup). Three geometry tests passed. Recomputed
reference scores reproduce 31.50% TSConv, 33.40% ATM, and 40.20% equal row-z
ensemble exactly. All saved test matrices are finite and 200x200.

| Method | Mean top1 |
|---|---:|
| Equal row-z score average | 40.20% |
| Equal raw cosine score average | 39.90% |
| Aligned equal SLERP / normalized average | 40.00% |
| Four terms without spherical normalization | 39.40% |
| Cross-model terms only | 38.20% |
| TS EEG -> aligned ATM images | 29.10% |
| Aligned ATM EEG -> TS images | 31.20% |
| Validation-selected raw score weights | 38.75% |
| Validation-selected row-z weights | 39.25% |
| Validation-selected SLERP weights | 38.55% |
| Validation-selected normalized-linear weights | 38.55% |

Mean paired image cosine after alignment: .8700 on alignment-training images,
.8616 on held-out validation images. Equal spherical fusion versus equal row-z
wins four folds, loses five, ties one; mean difference -.20 percentage points.
All four weight-selection families selected ATM weight .75 on all ten folds,
except raw averaging selected .5 on fold1. Validation preference did not transfer
to outer-subject retrieval. No outer-test weighting or method reselection is used
to produce the prespecified rows above.

Conclusion: substantial compatibility survives source-only orthogonal alignment,
and a single fused 128D vector preserves nearly all the score ensemble accuracy.
This is not a demonstrated gain or proof of equivalence (one seed), and still
requires both EEG encoders. It supports investigating a common representation
but does not establish that a cheap single encoder can predict it. Broad shared
directions may contribute to the high alignment cosine; cross-model retrieval is
the complementary task-level check. The weaker individual cross-model results
also show alignment is not complete interchangeability.
