"""Report the fixed TinyTSConv-3300 + TinyATM-4300 reference pair."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ensemble_experiments.analyze_tiny_compute_ensemble import load_dump, row_z, topk


MEMBERS = ("tiny_tsconv_seed3300", "tiny_atm_seed4300")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="results/things_eeg/tiny_compute_ensemble/testselected_internvit28",
    )
    args = parser.parse_args()
    root = Path(args.result_root)

    rows: list[dict[str, float | int]] = []
    for subject in range(1, 11):
        paths = [root / member / f"sub-{subject:02d}" / "embeddings.npz" for member in MEMBERS]
        if not all(path.exists() for path in paths):
            continue
        scores = np.stack([load_dump(path)[0] for path in paths])
        ts_top1, ts_top5 = topk(scores[0])
        atm_top1, atm_top5 = topk(scores[1])
        pair_top1, pair_top5 = topk(row_z(scores).mean(axis=0))
        rows.append(
            {
                "subject": subject,
                "tiny_tsconv_top1": ts_top1,
                "tiny_tsconv_top5": ts_top5,
                "tiny_atm_top1": atm_top1,
                "tiny_atm_top5": atm_top5,
                "pair_top1": pair_top1,
                "pair_top5": pair_top5,
            }
        )

    print(f"complete subjects={len(rows)}/10")
    if not rows:
        return
    average = {
        "subject": "Average",
        **{
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in rows[0]
            if key != "subject"
        },
    }
    output = root / "tiny_reference_pair.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(average))
        writer.writeheader()
        writer.writerows(rows + [average])
    print(
        f"TinyTSConv={average['tiny_tsconv_top1']:.2f}; "
        f"TinyATM={average['tiny_atm_top1']:.2f}; "
        f"pair={average['pair_top1']:.2f}; pair_top5={average['pair_top5']:.2f}"
    )


if __name__ == "__main__":
    main()
