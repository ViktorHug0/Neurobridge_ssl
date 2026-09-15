#!/usr/bin/env python3
"""Barplots: 1024d baseline (projector sweep) vs 64d no-mixup vs 64d+mixup (3 seeds, mean ± std)."""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_seed(config: str) -> int:
    m = re.search(r"seed(\d+)", str(config))
    if not m:
        raise ValueError(f"Could not parse seed from config: {config!r}")
    return int(m.group(1))


def _load_sweep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["sub"].astype(str).str.strip().str.lower() == "average"].copy()
    df["seed"] = df["config"].map(_parse_seed)
    for col in ("best top1 acc", "best top5 acc"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_featdim1024_rows(projector_sweep_csv: str) -> pd.DataFrame:
    df = _load_sweep(projector_sweep_csv)
    df = df[df["config"].astype(str).str.startswith("featdim_1024_")].copy()
    df = df.sort_values("seed")
    if len(df) != 3:
        raise ValueError(
            f"Expected 3 featdim_1024_* Average rows in {projector_sweep_csv}, got {len(df)}"
        )
    return df


def _plot_one_metric(
    means,
    stds,
    labels,
    ylabel,
    ylim,
    color,
    output_path,
):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.bar(
        x,
        means,
        width=0.55,
        yerr=stds,
        capsize=5,
        color=color,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_mixup_csv",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260426-133443/dropout_sweep_summary.csv",
        ),
    )
    parser.add_argument(
        "--mixup_csv",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260426-015522/dropout_sweep_summary.csv",
        ),
    )
    parser.add_argument(
        "--projector_sweep_csv",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/projector_sweep_20260419-221319/sweep_summary.csv",
        ),
        help="Sweep summary containing featdim_1024_seed* rows (1024d backbone, no mixup).",
    )
    parser.add_argument(
        "--latex_out",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/mixup_baseline1024_table.tex",
        ),
        help="Write booktabs LaTeX fragment (mean ± std over 3 seeds).",
    )
    parser.add_argument(
        "--output_top1",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/tsconv_dropout_mixup_vs_nomixup_top1.png",
        ),
    )
    parser.add_argument(
        "--output_top5",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/tsconv_dropout_mixup_vs_nomixup_top5.png",
        ),
    )
    args = parser.parse_args()

    df_1024 = _load_featdim1024_rows(args.projector_sweep_csv)
    df_nm = _load_sweep(args.no_mixup_csv)
    df_mx = _load_sweep(args.mixup_csv)

    labels = ["Baseline\n(1024d)", "64d", "64d + mixup"]

    def stats(df):
        return {
            "top1_mean": df["best top1 acc"].mean(),
            "top1_std": df["best top1 acc"].std(ddof=1),
            "top5_mean": df["best top5 acc"].mean(),
            "top5_std": df["best top5 acc"].std(ddof=1),
        }

    s1024 = stats(df_1024)
    s64 = stats(df_nm)
    s64m = stats(df_mx)

    _plot_one_metric(
        means=[s1024["top1_mean"], s64["top1_mean"], s64m["top1_mean"]],
        stds=[s1024["top1_std"], s64["top1_std"], s64m["top1_std"]],
        labels=labels,
        ylabel="Best Top-1 accuracy (%)",
        ylim=(30, 40),
        color="#4c72b0",
        output_path=args.output_top1,
    )
    _plot_one_metric(
        means=[s1024["top5_mean"], s64["top5_mean"], s64m["top5_mean"]],
        stds=[s1024["top5_std"], s64["top5_std"], s64m["top5_std"]],
        labels=labels,
        ylabel="Best Top-5 accuracy (%)",
        ylim=(60, 70),
        color="#dd8452",
        output_path=args.output_top5,
    )

    def _cell(val: float, bold: bool) -> str:
        s = f"{val:.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    t1 = (s1024["top1_mean"], s64["top1_mean"], s64m["top1_mean"])
    t5 = (s1024["top5_mean"], s64["top5_mean"], s64m["top5_mean"])
    i1 = int(np.argmax(t1))
    i5 = int(np.argmax(t5))

    # LaTeX: formal table environment (booktabs); best value per row in bold.
    os.makedirs(os.path.dirname(os.path.abspath(args.latex_out)), exist_ok=True)
    lines = [
        r"% Generated by plot_dropout_mixup_barplot.py",
        f"% Seed std (Top-1 / Top-5): 1024d {s1024['top1_std']:.2f}/{s1024['top5_std']:.2f}, "
        f"64d {s64['top1_std']:.2f}/{s64['top5_std']:.2f}, "
        f"64d+mix {s64m['top1_std']:.2f}/{s64m['top5_std']:.2f}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Zero-shot retrieval accuracy (\%, mean over three random seeds) on THINGS-EEG-2 "
        r"with leave-one-subject-out training. Values use the best test epoch by loss; "
        r"\textbf{Baseline (1024d)} is a wide fixed TSConv with 1024-dimensional embeddings (no mixup); "
        r"\textbf{64d} is TSConv-parameterizable ($k{=}30$, pool $51$, dropout $0.5$, 64-d embedding) without subject mixup; "
        r"\textbf{64d + mixup} is the same backbone with raw-EEG subject mixup ($\alpha{=}0.5$).}",
        r"\label{tab:featdim-mixup-loso}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Baseline (1024d)} & \textbf{64d} & \textbf{64d + mixup} \\",
        r"\midrule",
        "Top-1"
        f" & {_cell(t1[0], i1 == 0)} & {_cell(t1[1], i1 == 1)} & {_cell(t1[2], i1 == 2)} \\\\",
        "Top-5"
        f" & {_cell(t5[0], i5 == 0)} & {_cell(t5[1], i5 == 1)} & {_cell(t5[2], i5 == 2)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.1cm}",
        r"\end{table}",
        "",
    ]
    tex = "\n".join(lines)
    with open(args.latex_out, "w", encoding="utf-8") as f:
        f.write(tex)
    print(tex)
    print(f"Saved: {args.latex_out}")


if __name__ == "__main__":
    main()
