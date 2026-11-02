# Saved-score analysis: reference vs G2/C0/C1/C2

Analysis date: 2026-09-16. Five outer subjects, one seed, 200 candidates each.
Run: `.venv/bin/python -m ensemble_experiments.analysis.fusion_c012_scores`.
Machine-readable fold-level results:
`results/things_eeg/fusion_c012_20260915/score_analysis.json`.
No retraining, checkpoint reselection, or fusion-weight tuning was performed.
Candidate object/image identities were checked against the reference in every
fold; saved shared-model top1 values were reproduced.

## Exact error accounting

Counts below aggregate 1,000 subject-image decisions (not 1,000 independent images).
Coverage means at least one branch's top prediction is correct. It is NOT an
upper bound for score fusion, which can succeed when neither branch is top1-correct.

| Pair | Branch A % | Branch B % | Coverage count | Lost covered cases | Both-wrong rescues | Fused correct |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 33.1 | 35.3 | 496 | 109 | 44 | 431 |
| G2 | 33.6 | 35.7 | 496 | 118 | 27 | 405 |
| C0 | 31.9 | 35.8 | 490 | 112 | 25 | 403 |
| C1 | 33.0 | 34.2 | 471 | 96 | 26 | 401 |
| C2 | 31.7 | 35.6 | 475 | 96 | 23 | 402 |

Identity: fused correct = coverage - lost covered cases + both-wrong rescues.
Thus G2's 26-decision deficit decomposes into 17 fewer rescues and 9 additional
lost covered cases, with no aggregate coverage difference. These are aggregate
count differences, not an assertion that the same examples occupy each category.
Reference vs G2 fused predictions lose 99 and gain 73 decisions respectively.

## Distractor agreement

When both branches fail, they choose the same wrong image in 100/504 reference
decisions (19.8%), versus 157/504 G2 decisions (31.2%). G2 has fewer both-wrong
rescues and more identical wrong predictions in each of the five subjects.
Mean within-fold correlation of row-standardized score matrices increases from
0.8245 to 0.8521. This is descriptive, not evidence that correlation alone causes
the accuracy gap.

For standardized branch scores, fused true-vs-best-wrong margin equals the mean
individual margin plus a nonnegative distractor-disagreement bonus:

`bonus = mean_b(max_wrong score_b) - max_wrong(mean_b score_b)`.

Mean individual margins are almost identical: reference -0.263685, G2 -0.263831.
The bonus is lower for G2: 0.108363 vs 0.127291 (~15% reduction), and lower in
every fold. All C variants have similar smaller bonuses (0.1061–0.1121).
This mathematically describes weaker cancellation of competing wrong candidates;
it does not identify which architectural or training choice caused the change.

## Exploratory cross-pairing

Replacing either G2 branch with its standalone reference counterpart gives:

| Pair | Top1 % |
|---|---:|
| Reference TS + reference ATM | 43.1 |
| Reference TS + G2 B | 41.8 |
| G2 A + reference ATM | 41.7 |
| G2 A + G2 B | 40.5 |

This is consistent with a pair-compatibility deficit, not simply one weaker
branch. Cross-pairings are post-hoc test diagnostics, not validated new methods.

## Implications and limits

Restoring private dense readouts (C0), selectively privatizing the residual dense
layer (C1), or increasing correction rank (C2) does not restore the reference's
rescue rate. Prioritize a controlled early-feature-sharing ablation over more
readout capacity: G2 with separate temporal stems, otherwise fixed architecture,
initialization policy, budget, and checkpoint selection. Predict larger
distractor-disagreement bonus and more both-wrong rescues if early sharing is
responsible. A null result would weaken that hypothesis and direct attention to
native ATM vs post-stem attention and training/selection differences.

Do not optimize raw disagreement: making one branch worse can also increase it.
Track branch accuracy, coverage, rescue rate, harmful fusion, and margin bonus
together on source validation; reserve new outer subjects for confirmation.

Reference uses independent validation-loss-selected checkpoints, while new models
use a common fused-validation-top1 checkpoint. Architecture, selection, and joint
training effects are not causally separated here. Five subjects/one seed and
repeated use of their test results do not support significance or unbiased
architecture-selection claims. No claim of biological mechanism is established.
