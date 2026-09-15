"""Aggregate completed folds from the inductive covariance pilot grid."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


VARIANT = re.compile(r"alpha(?P<alpha>\d+)_shrink(?P<shrinkage>\d+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for result in sorted(args.result_root.glob("alpha*_shrink*/*-sub-??/result.csv")):
        match = VARIANT.fullmatch(result.parents[1].name)
        if match is None:
            continue
        with result.open(newline="") as handle:
            metric = next(csv.DictReader(handle))
        rows.append(
            {
                "alpha": int(match.group("alpha")) / 100.0,
                "shrinkage": int(match.group("shrinkage")) / 100.0,
                "subject": int(result.parent.name.rsplit("-", 1)[1]),
                "best_top1": float(metric["best top1 acc"]),
                "best_top5": float(metric["best top5 acc"]),
                "best_epoch": int(metric["best epoch"]),
                "run_dir": str(result.parent),
            }
        )
    if not rows:
        print("completed folds=0")
        return

    folds = pd.DataFrame(rows).sort_values(["alpha", "shrinkage", "subject"])
    folds.to_csv(args.result_root / "fold_results.csv", index=False)
    summary = (
        folds.groupby(["alpha", "shrinkage"], as_index=False)
        .agg(
            completed_subjects=("subject", "nunique"),
            mean_best_top1=("best_top1", "mean"),
            mean_best_top5=("best_top5", "mean"),
        )
        .sort_values(["completed_subjects", "mean_best_top1"], ascending=[False, False])
    )
    summary.to_csv(args.result_root / "grid_summary.csv", index=False)
    print(f"completed folds={len(folds)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
