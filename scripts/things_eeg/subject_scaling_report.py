#!/usr/bin/env python
"""Aggregate the subject-count scaling curve: best top1 vs number of training subjects.

Reads run dirs named  <timestamp>-sub-<SS>_n<N>_d<D>/result.csv  produced by run_subject_scaling.sh,
and splices in the N=9 anchor from the current-best LOSO run.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ANCHOR = Path("results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/"
              "param_k30_pool51_do050_featdim512_seed3300")


def best_top1(result_csv):
    with open(result_csv) as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1]["best top1 acc"]), int(rows[-1]["best epoch"])


def main(out_dir):
    # points[N][test_subject] = [(acc, best_epoch) per draw]
    points = defaultdict(lambda: defaultdict(list))

    for csv_path in Path(out_dir).glob("*/result.csv"):
        m = re.search(r"-sub-(\d+)_n(\d+)_d(\d+)$", csv_path.parent.name)
        if m:
            points[int(m.group(2))][int(m.group(1))].append(best_top1(csv_path))

    subjects = sorted({s for per_sub in points.values() for s in per_sub})
    # The N=9 anchor is an EEG2 LOSO run; it says nothing about the EEG1 curve.
    if "things_eeg1" not in str(out_dir):
        for csv_path in ANCHOR.glob("*/result.csv"):
            m = re.search(r"-sub-(\d+)$", csv_path.parent.name)
            if m and int(m.group(1)) in subjects:
                points[9][int(m.group(1))].append(best_top1(csv_path))

    print(f"\nSubject-count scaling curve (best top1 acc, LOSO)  test subjects: {subjects}")
    print("'max ep' is the latest best-epoch over the cell's runs: if it approaches the run's epoch "
          "budget,\nthat cell was compute-starved and the point is a floor, not a measurement.\n")
    print(f"{'N':>3} {'mean':>7} {'sd(draws)':>10} {'max ep':>7} {'runs':>5}   per-subject means")
    for n in sorted(points):
        accs = {s: [a for a, _ in points[n][s]] for s in subjects if points[n][s]}
        per_sub = [statistics.fmean(v) for v in accs.values()]
        draw_sds = [statistics.stdev(v) for v in accs.values() if len(v) > 1]
        n_runs = sum(len(v) for v in points[n].values())
        max_ep = max(e for v in points[n].values() for _, e in v)
        sd = f"{statistics.fmean(draw_sds):7.2f}" if draw_sds else "      -"
        detail = "  ".join(f"s{s:02d}={statistics.fmean(v):.1f}" for s, v in accs.items())
        print(f"{n:>3} {statistics.fmean(per_sub):7.2f} {sd:>10} {max_ep:>7} {n_runs:>5}   {detail}")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "results/things_eeg/inter-subjects/subject_scaling")
