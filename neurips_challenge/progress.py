"""Live status of every Track-1 run: queue state, epoch curve, and final Codabench score.

    .venv/bin/python neurips_challenge/progress.py [--root results/things_eeg/neurips_track1]

test_top5 is train.py's own per-epoch number and is NOT the Codabench metric -- it tracks it
monotonically but is not calibrated to it. The `codabench` column is the real thing, and only
appears once score_track1.py has run.

`sel_ep` is the epoch this run's own --select_best_metric rule retains, so it is comparable
across runs that used different rules. The trailing flag shows how much test top-5 the retained
checkpoint leaves on the table versus the run's best epoch.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess


def curve(path, metric):
    """Last epoch, the epoch this run's own rule retains, and the epoch that peaked."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    if metric == "top5":
        retained = max(rows, key=lambda r: float(r["val_top5_acc"] or 0))
    else:
        retained = min(rows, key=lambda r: float(r["val_loss"] or "inf"))
    peak = max(rows, key=lambda r: float(r["test_top5_acc"] or 0))
    return rows[-1], retained, peak


def select_metric(run_dir):
    try:
        with open(os.path.join(run_dir, "train_config.json")) as handle:
            return json.load(handle).get("select_best_metric", "loss")
    except OSError:
        return "loss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/things_eeg/neurips_track1")
    args = parser.parse_args()

    print(subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-o",
                          "%.8i %.12j %.9T %.10M %.14R"],
                         capture_output=True, text=True).stdout)

    header = f"{'run':44s} {'ep':>4s} {'test_top5':>9s} {'sel_ep':>6s} {'sel_top5':>8s} {'codabench':>9s}"
    print(header)
    print("-" * len(header))
    for metrics in sorted(glob.glob(os.path.join(args.root, "**", "epoch_metrics.csv"),
                                    recursive=True)):
        run_dir = os.path.dirname(metrics)
        got = curve(metrics, select_metric(run_dir))
        if got is None:
            continue
        last, retained, peak = got
        score_path = os.path.join(run_dir, "track1_score.json")
        if os.path.exists(score_path):
            with open(score_path) as handle:
                codabench = f"{json.load(handle)['per_epoch']['top5']:.4f}"
        else:
            codabench = "-"
        name = os.path.relpath(run_dir, args.root)
        missed = float(peak["test_top5_acc"]) - float(retained["test_top5_acc"])
        flag = f"  <- peak ep{peak['epoch']} is +{missed:.2f}" if missed > 0.25 else ""
        print(f"{name[-44:]:44s} {last['epoch']:>4s} {float(last['test_top5_acc']):9.2f} "
              f"{retained['epoch']:>6s} {float(retained['test_top5_acc']):8.2f} "
              f"{codabench:>9s}{flag}")


if __name__ == "__main__":
    main()
