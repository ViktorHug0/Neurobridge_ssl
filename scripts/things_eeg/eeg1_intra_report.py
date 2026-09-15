#!/usr/bin/env python
"""Aggregate the EEG1 intra-subject grid: encoder x preprocessing variant, 150-way top1.

Reports both accuracy columns on purpose. The repo convention ("best top1 acc") is the top1 at
the epoch with the lowest test LOSS, and on EEG1 that epoch lands early (~13) while accuracy is
still climbing -- so on this cohort it undershoots the last-epoch number instead of exceeding it.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main(out_dir):
    cells = defaultdict(dict)  # (variant, encoder) -> {subject: (best, last)}
    for path in Path(out_dir).glob("*/result.csv"):
        m = re.search(r"-(raw|paper)_(\w+?)_sub-(\d+)$", path.parent.name)
        if not m:
            continue
        rows = list(csv.DictReader(open(path)))
        if rows:
            cells[(m.group(1), m.group(2))][int(m.group(3))] = (
                float(rows[-1]["best top1 acc"]), float(rows[-1]["top1 acc"]))

    print("\nTHINGS-EEG1 intra-subject decoding (150-way, chance 0.67%)")
    print(f"{'variant':>7} {'encoder':>22} {'best_top1':>10} {'last_top1':>10} {'sd':>6} "
          f"{'min':>6} {'max':>6} {'n':>3}")
    for key in sorted(cells):
        per_sub = cells[key]
        best = [v[0] for v in per_sub.values()]
        last = [v[1] for v in per_sub.values()]
        sd = statistics.stdev(last) if len(last) > 1 else 0.0
        print(f"{key[0]:>7} {key[1]:>22} {statistics.fmean(best):10.2f} "
              f"{statistics.fmean(last):10.2f} {sd:6.2f} {min(last):6.2f} {max(last):6.2f} "
              f"{len(last):3d}")

    # Paired variant comparison on the subjects that finished both, per encoder.
    print()
    for enc in sorted({e for _, e in cells}):
        a, b = cells.get(("raw", enc), {}), cells.get(("paper", enc), {})
        shared = sorted(set(a) & set(b))
        if shared:
            d = [b[s][1] - a[s][1] for s in shared]
            print(f"{enc}: paper - raw = {statistics.fmean(d):+.2f} pp "
                  f"(last_top1, n={len(shared)} paired, {sum(x > 0 for x in d)} subjects favour paper)")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/things_eeg1/intra-subjects")
