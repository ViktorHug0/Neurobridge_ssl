"""Report TinyTSConv/TinyATM 40-epoch pair results for available subjects."""

import argparse
from pathlib import Path

import numpy as np

from ensemble_experiments.analyze_tiny_compute_ensemble import load_dump, row_z, topk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result-root',
        default='results/things_eeg/tiny_reference_pair_40e/testselected_internvit28',
    )
    parser.add_argument('--run-tag', default='tiny40')
    args = parser.parse_args()
    root = Path(args.result_root)
    names = (
        f"{args.run_tag}_tsconv_seed3300",
        f"{args.run_tag}_atm_seed4300",
    )
    rows = []
    for subject in range(1, 4):
        paths = [root / name / f"sub-{subject:02d}" / "embeddings.npz" for name in names]
        if not all(path.exists() for path in paths):
            continue
        scores = np.stack([load_dump(path)[0] for path in paths])
        ts, atm, pair = topk(scores[0]), topk(scores[1]), topk(row_z(scores).mean(axis=0))
        rows.append((ts[0], atm[0], pair[0]))
        print(f"S{subject}: TS={ts[0]:.2f} ATM={atm[0]:.2f} pair={pair[0]:.2f}")
    if rows:
        mean = np.mean(rows, axis=0)
        print(f"Average n={len(rows)}: TS={mean[0]:.2f} ATM={mean[1]:.2f} pair={mean[2]:.2f}")
    else:
        print("complete subjects=0/3")


if __name__ == "__main__":
    main()
