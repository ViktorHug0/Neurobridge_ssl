#!/usr/bin/env python3
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def _plot_regime_metric(summary_df, regime, metric_prefix, ylabel, output_path):
    df = summary_df[summary_df["regime"] == regime].copy()
    if df.empty:
        return

    methods = ["plain_cosine", "saw_csls_k5", "full_sattc"]
    labels = {
        "plain_cosine": "SAGE-Zero-Shot",
        "saw_csls_k5": "SAGE-Zero-Shot + Whitening",
        "full_sattc": "SAGE-TTA"
    }
    colors = {
        "plain_cosine": "#d62728",
        "saw_csls_k5": "#ff7f0e",
        "full_sattc": "#1f77b4"
    }

    plt.figure(figsize=(21, 13.65))
    mean_col = f"{metric_prefix}_mean"
    std_col = f"{metric_prefix}_std"

    for method in methods:
        m_df = df[df["method"] == method].sort_values("sample_count")
        if m_df.empty:
            continue
        
        x = m_df["sample_count"]
        color = colors.get(method, "gray")
        
        plt.plot(x, m_df[mean_col], color=color, linewidth=2.5, label=labels[method])
        plt.fill_between(x, m_df[mean_col] - m_df[std_col], m_df[mean_col] + m_df[std_col], color=color, alpha=0.1, linewidth=0)

    plt.xlabel("Number of test samples", fontsize=43)
    plt.ylabel(ylabel, fontsize=43)
    plt.tick_params(axis="both", labelsize=36)
    plt.grid(True, alpha=0.3)
    plt.xlim(5, 200)
    plt.ylim(0, 105)
    plt.legend(ncol=1, fontsize=40, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    output_dir = args.output_dir or os.path.dirname(args.csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # 4 plots total: 2 regimes x 2 metrics
    for regime in ["matching", "all"]:
        regime_label = "Small-Scale (N vs N)" if regime == "matching" else "Large-Scale (N vs 200)"
        
        # Top-1
        _plot_regime_metric(
            df, regime, "top1", "Top-1 Accuracy (%)",
            os.path.join(output_dir, f"progressive_top1_{regime}.png")
        )
        # Top-5
        _plot_regime_metric(
            df, regime, "top5", "Top-5 Accuracy (%)",
            os.path.join(output_dir, f"progressive_top5_{regime}.png")
        )


if __name__ == "__main__":
    main()
