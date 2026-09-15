"""Report available UltraTinyTSConv + UltraTinyATM subjects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ensemble_experiments.analyze_tiny_compute_ensemble import load_dump, row_z, topk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="results/things_eeg/ultratiny_reference_pair/testselected_internvit28",
    )
    args = parser.parse_args()
    root = Path(args.result_root)
    names = ("ultratiny_tsconv_seed5300", "ultratiny_atm_seed6300")
    rows = []
    for subject in range(1, 11):
        paths = [root / name / f"sub-{subject:02d}" / "embeddings.npz" for name in names]
        if not all(path.exists() for path in paths):
            continue
        scores = np.stack([load_dump(path)[0] for path in paths])
        ts = topk(scores[0])
        atm = topk(scores[1])
        pair = topk(row_z(scores).mean(axis=0))
        rows.append((subject, *ts, *atm, *pair))
        print(
            f"S{subject}: TS={ts[0]:.2f}/{ts[1]:.2f} "
            f"ATM={atm[0]:.2f}/{atm[1]:.2f} pair={pair[0]:.2f}/{pair[1]:.2f}"
        )
    if rows:
        values = np.asarray([row[1:] for row in rows])
        mean = values.mean(axis=0)
        print(
            f"Average n={len(rows)}: TS={mean[0]:.2f}/{mean[1]:.2f} "
            f"ATM={mean[2]:.2f}/{mean[3]:.2f} pair={mean[4]:.2f}/{mean[5]:.2f}"
        )
    else:
        print("complete subjects=0/10")


if __name__ == "__main__":
    main()
