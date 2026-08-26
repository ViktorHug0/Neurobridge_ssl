# Ensemble experiments

This directory is the canonical home for every ensemble-specific script, launcher,
analysis, and report in the repository.

| Path | Contents |
|---|---|
| `test_selected/` | Test-selected training arms, score dumping, fixed-rule searches, and router controls |
| `validation/` | LOSO-subject validation, concept validation (ValCon), and matched checkpoint controls |
| `architectures/` | EEG encoder candidates, smoke tests, training drivers, and architecture launchers |
| `synthetic_subjects/` | Earlier seed, repetition, shrunk-model, layer-target, and meta-subject ensemble utilities |
| `analysis/` | Current row-z roster sweeps, evidence audit, generated JSON tables, and technical narrative |
| `legacy/scratch_claude/` | Superseded exploratory searches and their small historical artifacts, retained pending deletion review |
| `ensemble_results_report.md` | Concise report-ready summary of the ensemble findings |
| `presentation/` | Beamer source for the ensemble-results presentation; generated build products are ignored |
| `subject_*bagging.sbatch` | Disjoint-triplet and overlapping-six source-cohort ensemble controls |

All commands assume the repository root as the working directory and the project virtual
environment is active.

The old top-level path `new_architectures_for_ensemble` remains as a compatibility symlink.
New commands and imports should use the canonical paths above. The unused
`ensemble50_experiments` and `honest_ensemble_experiments` links were removed.
