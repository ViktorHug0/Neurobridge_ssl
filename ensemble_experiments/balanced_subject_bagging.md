# Balanced 7-of-9 subject bagging

This experiment ports the most useful ensemble mechanism from the separate
mental-imagery study: create diversity by assigning each independently initialized
member a fixed subset of source subjects, then average member scores at inference.

The existing THINGS-EEG controls locate the tradeoff:

| Source roster per member | Members | Ensemble top-1 | Mean score correlation |
|---|---:|---:|---:|
| disjoint 3-of-9 | 3 | 31.80% | 0.778 |
| overlapping 6-of-9 | 3 | 35.70% | 0.896 |
| full 9-of-9, cyclic soft weights | 3 | 38.20% | 0.957 |

The new committee uses eight balanced 7-of-9 bags. This matches the approximately
78% subject-bagging rate that worked well in the mental-imagery experiments, while
keeping more training subjects per member than the earlier hard-bagging controls.
For every LOSO target, eight distinct adjacent exclusion pairs are drawn from one
seeded cycle over the nine source subjects. Each source appears in six or seven
members, so exposure is balanced without making the members identical.

The primary, predeclared fusion rule is per-query/member row-z normalization followed
by a uniform mean of the eight 200-way score rows. The analyzer also reports raw-score,
probability, row-z probability, rank, and majority-vote diagnostics copied from the
mental-imagery audit. Those alternatives are diagnostics, not permission to select a
different rule separately for each held-out subject.

An audit of the already completed three-member bags confirms that fusion tricks are not
the missing gain:

| Roster | Row-z | Best of the six fixed rules |
|---|---:|---:|
| disjoint 3-of-9 | 31.80% | 31.85% (raw probability) |
| overlapping 6-of-9 | 35.70% | 35.85% (raw/probability) |
| full-source soft weights | 38.20% | 38.25% (row-z probability) |

The largest change is only 0.15 point. The experiment therefore changes how diversity
is created during training and retains the established, deliberately simple row-z mean
as its primary inference rule.

```bash
sbatch ensemble_experiments/balanced_subject_bagging.sbatch

# Or, from a persistent one-GPU allocation on node 513:
tmux new-session -d -s bag7x8-513 \
  'bash ensemble_experiments/run_balanced_subject_bagging_local.sh'

# Two-node split: even tasks on 513, remaining odd tasks on 522.
TASK_START=0 TASK_STRIDE=2 RUNNER_NAME=runner-513 \
  bash ensemble_experiments/run_balanced_subject_bagging_local.sh
EXPECTED_HOST=sl-tp-br-522 PHYSICAL_CUDA_DEVICE=1 TASK_START=7 TASK_STRIDE=2 \
  RUNNER_NAME=runner-522 bash ensemble_experiments/run_balanced_subject_bagging_local.sh

source .venv/bin/activate
python -m ensemble_experiments.analyze_balanced_subject_bagging \
  --require-complete
```

Besides the per-subject and average CSV tables, the analyzer writes each primary
row-z committee matrix to `fused_scores/balanced7x8-subXX.npz`. This lets the whole
subject-bag committee act as one meta-member in a later, predeclared cross-encoder or
cross-image-target ensemble without pretending that its members share an embedding
space.

This remains an inductive plain-cosine ensemble: every transform reads only one
query's candidate scores. It does not apply SAW, CSLS, Sinkhorn, Procrustes, or any
other test-set adaptation. The launcher retains the repository's historical
test-selected checkpoint convention, so any final number must be labeled accordingly.

## Completed result (2026-09-02)

All 80 folds completed. Averaged over the ten held-out subjects:

| Metric | Top-1 |
|---|---:|
| Mean individual member | 32.60% |
| Best member per subject (post-hoc diagnostic) | 36.80% |
| **Predeclared row-z ensemble** | **37.85%** |
| Raw-score ensemble (diagnostic) | 38.10% |
| Probability ensemble (diagnostic) | 38.05% |
| Member oracle | 62.10% |

The mean pairwise score correlation is 0.921. Balanced 7-of-9 bagging therefore
improves the predeclared row-z ensemble by 5.25 points over its average member and by
2.15 points over the earlier overlapping 6-of-9 committee. It essentially matches,
but does not exceed, the 38.20% full-source soft-weight control. This is a useful
negative result: the mental-imagery diversity recipe transfers, while the remaining
gain appears to require diversity beyond source-subject roster perturbations.
