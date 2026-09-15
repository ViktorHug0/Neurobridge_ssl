"""Plot mean ± std (over seeds) of best top-1 vs alignment dimension from a featdim sweep CSV."""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_csv = os.path.join(
        repo_root,
        "results/things_eeg/inter-subjects/projector_sweep_20260419-221319/sweep_summary.csv",
    )
    default_out = os.path.join(
        repo_root,
        "results/things_eeg/inter-subjects/projector_sweep_20260419-221319/featdim_accuracy_sweep.png",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=["", "projector_only"],
        default="",
        help='Use built-in paths/labels (e.g. "projector_only" for sweep_20260423-112642).',
    )
    p.add_argument(
        "--csv",
        default=default_csv,
        help="Path to sweep_summary.csv (columns: config, best top1 acc, …).",
    )
    p.add_argument(
        "--output",
        default=default_out,
        help="Where to save the figure (PNG).",
    )
    p.add_argument(
        "--xlabel",
        default=r"Embedding Dimension ($d$)",
        help="Matplotlib x-axis label.",
    )
    p.add_argument(
        "--subtitle",
        default="",
        help="Optional subtitle (e.g. projector-only note); empty to omit.",
    )
    args = p.parse_args()

    if args.preset == "projector_only":
        args.csv = os.path.join(
            repo_root,
            "results/things_eeg/inter-subjects/projector_only_sweep_20260423-112642/sweep_summary.csv",
        )
        args.output = os.path.join(
            repo_root,
            "results/things_eeg/inter-subjects/projector_only_sweep_20260423-112642/featdim_accuracy_sweep.png",
        )
        args.xlabel = r"Alignment dimension $d$ (linear projector output)"
        args.subtitle = r"EEG backbone dimension fixed at $128$"

    return args


def main() -> None:
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    df["featdim"] = df["config"].apply(lambda x: int(re.search(r"featdim_(\d+)", x).group(1)))
    df["seed"] = df["config"].apply(lambda x: int(re.search(r"seed_?(\d+)", x).group(1)))

    stats = (
        df.groupby("featdim", as_index=False)
        .agg(mean_top1=("best top1 acc", "mean"), std_top1=("best top1 acc", "std"))
        .sort_values("featdim")
    )
    stats["std_top1"] = stats["std_top1"].fillna(0.0)

    plt.figure(figsize=(8, 5))
    color_top1 = "#1f77b4"

    plt.errorbar(
        stats["featdim"],
        stats["mean_top1"],
        yerr=stats["std_top1"],
        fmt="o-",
        label="Top-1 Accuracy (mean ± std)",
        color=color_top1,
        linewidth=2,
        markersize=8,
        capsize=4,
        elinewidth=1.2,
    )

    plt.xscale("log", base=2)
    plt.xticks(stats["featdim"], [str(d) for d in stats["featdim"]])
    plt.xlabel(args.xlabel, fontsize=12)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    if args.subtitle:
        plt.title(args.subtitle, fontsize=10, pad=8)
    plt.grid(True, which="both", ls="-", alpha=0.3)

    lo = max(0.0, (stats["mean_top1"] - stats["std_top1"]).min() - 1.0)
    hi = (stats["mean_top1"] + stats["std_top1"]).max() + 1.0
    plt.ylim(lo, hi)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
