"""Print the Track-1 sweep as one table.

    .venv/bin/python neurips_challenge/collect_results.py [--root results/things_eeg/neurips_track1/sweep]

The headline column is `per_epoch top5` -- the Codabench warm-up leaderboard metric. subject_agg is
shown alongside because it is what NeuralBench reports and what the start-kit's summary.csv rows
use, so the two repos can be compared without re-deriving anything.

Cells that have not finished show what is missing rather than being silently omitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

CELLS = ["250hz_avg", "120hz_avg", "250hz_noavg", "120hz_noavg"]
ARCHS = ["TSConv", "EEGProject", "ATM"]


def load(root: str, cell: str, arch: str):
    hits = sorted(glob.glob(os.path.join(root, cell, arch, "*", "track1_score.json")))
    if not hits:
        return None
    with open(hits[-1]) as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/things_eeg/neurips_track1/sweep")
    args = parser.parse_args()

    print(f"root: {args.root}")
    print(f"metric: per-epoch cosine top-5 vs 200 unique DINOv2 targets (Codabench warm-up protocol)")
    print()
    header = f"{'cell':14s} {'arch':11s} {'top1':>7s} {'top5':>7s} {'subj top5':>10s} {'epoch':>6s} {'valid':>6s}"
    print(header)
    print("-" * len(header))

    done = 0
    for cell in CELLS:
        for arch in ARCHS:
            report = load(args.root, cell, arch)
            if report is None:
                print(f"{cell:14s} {arch:11s} {'-':>7s} {'-':>7s} {'-':>10s} {'-':>6s}   pending")
                continue
            done += 1
            pe, sa = report["per_epoch"], report["subject_agg"]
            print(
                f"{cell:14s} {arch:11s} {pe['top1']:7.4f} {pe['top5']:7.4f} "
                f"{sa['top5']:10.4f} {report['best_epoch']:6d} "
                f"{'yes' if report['leaderboard_valid'] else 'NO':>6s}"
            )
        print()

    print(f"{done}/12 cells complete")
    print()
    print("Reference, start-kit Track1TSConv on its own harness: per-epoch top5 0.1734, subject-agg 0.5815")
    print("Chance on a 200-image gallery: top1 0.005, top5 0.025")
    print()
    print("Read with care:")
    print("  - ATM rows: subject conditioning is inactive at pooled-10 (ids 1-10 vs an embedding")
    print("    table sized 0-9), so every batch falls back to the shared token.")
    print("  - 250hz vs 120hz bundles five changes (rate, window, band-pass, notch, scaler+clamp),")
    print("    and TSConv's flattened width also drops 1440 -> 400 at 120 samples.")
    print("  - avg vs noavg also changes the negative structure: --samples_per_image 10 draws")
    print("    10 of 10 candidates when averaged, 10 of 40 when not.")


if __name__ == "__main__":
    main()
