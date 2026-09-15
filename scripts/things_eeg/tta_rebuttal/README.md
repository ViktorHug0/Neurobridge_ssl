# TTA Rebuttal Experiments

Clean entrypoints for rebuttal experiments probing practical utility of the
SAGE/SATTC-style transductive alignment stage.

All scripts default to writing under:

```bash
results/things_eeg/tta_rebuttal/
```

The older scripts in `scripts/things_eeg/` are retained for result provenance.
Use this folder for new rebuttal runs.

## Scripts

- `run_progressive_calibration.py`: subset-size curves for fresh TTA on `N`
  unlabeled test queries, evaluated in both `N`-vs-`N` and `N`-vs-200 regimes.
- `run_split_transfer.py`: fit label-free TTA on one calibration subset, freeze
  the map, and evaluate on a disjoint subset. This is the strongest deployment
  realism check.
- `run_repetition_ablation.py`: rebuild the 200 test queries with fewer EEG
  repetitions per image (`80, 70, 60, ...`) and rerun plain cosine, SAW+CSLS,
  and full TTA.
- `run_fewshot_subject_adaptation.py`: few-shot train/val/test split (default
  80/20/100) on held-out subjects; compares train-fit ZCA, orthogonal/ridge/blended
  maps, low-rank alignment, and a compact neural adapter. Reports
  `delta_top1_vs_train_zca` on unseen test stimuli.
- `run_trainset_rotation_transfer.py`: fit subject-adaptation maps on increasing
  numbers of the held-out subject's training images (`100, 500, 1000, 5000, ...,
  all`) and evaluate the frozen map on the fixed 200-way test set. Includes a
  validation-selected Cayley orthogonal adapter as a smarter alternative to scalar
  alpha blending.
- `run_rebuttal_suite.sh`: shell wrapper for the three TTA experiments (and
  optional few-shot FSL via `RUN_FEWSHOT=true`, train-set transfer via
  `RUN_TRAINSET_TRANSFER=true`).

## Example

```bash
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

SOURCE_RUN_DIR="results/things_eeg/inter-subjects/YOUR_RUN/featdim_64" \
SUBJECTS="1 2 3 4 5 6 7 8 9 10" \
bash scripts/things_eeg/tta_rebuttal/run_rebuttal_suite.sh
```

To run only the repetition experiment:

```bash
python scripts/things_eeg/tta_rebuttal/run_repetition_ablation.py \
  --source_run_dir results/things_eeg/inter-subjects/YOUR_RUN/featdim_64 \
  --repetition_counts 80 70 60 50 40 30 20 10
```

Few-shot subject adaptation (default output under the rebuttal suite run):

```bash
python scripts/things_eeg/tta_rebuttal/run_fewshot_subject_adaptation.py \
  --source_run_dir results/things_eeg/inter-subjects/YOUR_RUN/featdim_64 \
  --output_dir results/things_eeg/tta_rebuttal/rebuttal_suite_20260602-134420/fewshot_subject_adaptation \
  --train_size 80 --val_size 20

# Learning curve over train sizes
python scripts/things_eeg/tta_rebuttal/run_fewshot_subject_adaptation.py \
  --train_sizes 5 10 20 40 80 \
  --val_size 20 \
  --output_dir results/things_eeg/tta_rebuttal/rebuttal_suite_20260602-134420/fewshot_subject_adaptation
```

Train-set rotation transfer:

```bash
python scripts/things_eeg/tta_rebuttal/run_trainset_rotation_transfer.py \
  --source_run_dir results/things_eeg/inter-subjects/YOUR_RUN/featdim_64 \
  --output_dir results/things_eeg/tta_rebuttal/rebuttal_suite_20260602-134420/trainset_rotation_transfer \
  --calibration_sizes 100 500 1000 5000 10000 all

# Via suite wrapper
RUN_TRAINSET_TRANSFER=true RUN_TAG=rebuttal_suite_20260602-134420 \
  bash scripts/things_eeg/tta_rebuttal/run_rebuttal_suite.sh
```
