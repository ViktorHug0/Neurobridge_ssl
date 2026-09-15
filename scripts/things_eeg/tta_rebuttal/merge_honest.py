#!/usr/bin/env python3
"""Merge honest-selection grid_scores.csv shards and recompute the protocol summary
over all subjects at once.

Used when the grid was split across machines (e.g. subjects 1-2 on one box, 3-10 on
another) so that no (subject, config) pair is recomputed. Reuses select() from the
experiment script so the selection logic stays in one place.
"""
import argparse
import os

import pandas as pd

from shared import TTAParams
from run_honest_tta_selection import select

CONFIG_COLS = ["saw_shrink", "sinkhorn_tau", "soft_procrustes_steps", "soft_procrustes_power"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="grid_scores.csv files, or dirs containing one, to merge")
    ap.add_argument("--out", required=True, help="output dir for the merged summary")
    args = ap.parse_args()

    paths = [os.path.join(s, "grid_scores.csv") if os.path.isdir(s) else s for s in args.shards]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # if a subject shows up in two shards, keep the later shard's rows
    df = df.drop_duplicates(["subject_id", *CONFIG_COLS], keep="last")

    selected = select(df, None, TTAParams())  # select() ignores its args argument
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "grid_scores.csv"), index=False)
    selected.to_csv(os.path.join(args.out, "selected_per_subject.csv"), index=False)

    summary = (
        selected.groupby("protocol")
        .agg(top1_mean=("test_top1", "mean"), top1_std=("test_top1", "std"),
             top5_mean=("test_top5", "mean"), n_subjects=("test_top1", "size"))
        .sort_values("top1_mean", ascending=False)
    )
    oracle = float(summary.loc["oracle_test", "top1_mean"]) if "oracle_test" in summary.index else float("nan")
    summary["gap_to_oracle"] = (summary["top1_mean"] - oracle).round(2)
    summary.round(2).to_csv(os.path.join(args.out, "protocol_summary.csv"))

    n_sub = df["subject_id"].nunique()
    print(summary.round(2).to_string())
    print(f"\nsubjects={n_sub}  configs/subject={len(df) // max(n_sub, 1)}  rows={len(df)}")
    print(f"Saved merged summary to: {args.out}")

    assert {"oracle_test", "paper_default"} <= set(summary.index), "missing reference protocols after merge"


if __name__ == "__main__":
    main()
