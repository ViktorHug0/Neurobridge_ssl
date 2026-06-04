"""Sparsity vs active-feature coverage vs accuracy for detached-confidence sparse CLIP session."""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVATION_ORDER = ["relu", "relu_gelu"]
ACTIVATION_LABELS = {
    "relu": "ReLU",
    "relu_gelu": "ReLU forward / GELU grad",
}
ACTIVATION_COLORS = {"relu": "#4C72B0", "relu_gelu": "#C44E52"}
ACTIVATION_MARKERS = {"relu": "o", "relu_gelu": "s"}


def parse_args() -> argparse.Namespace:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_session = os.path.join(
        repo,
        "results/things_eeg/inter-subject-sparse/"
        "sparse_clip_confidence_detached_relu_gelu_bb64_seed3300",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session_dir", default=default_session)
    p.add_argument(
        "--output_prefix",
        default=None,
        help="Prefix for CSV/PNG outputs (default: <session_dir>/sparsity_accuracy)",
    )
    return p.parse_args()


def parse_config_name(config_name: str) -> tuple[int, str]:
    m = re.match(r"sparse_fd(\d+)_(relu(?:_gelu)?)_detached_confidence", config_name)
    if not m:
        raise ValueError(f"Unexpected config folder name: {config_name}")
    return int(m.group(1)), m.group(2)


def collect_session(session_dir: str) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(session_dir, "sparse_fd*_detached_confidence_seed*", "inter_subject_summary.csv")
    for summary_path in sorted(glob.glob(pattern)):
        config_name = os.path.basename(os.path.dirname(summary_path))
        feature_dim, activation = parse_config_name(config_name)
        df = pd.read_csv(summary_path)
        df.columns = [c.strip() for c in df.columns]
        avg = df[df["sub"].astype(str).str.strip() == "Average"]
        if avg.empty:
            continue
        r = avg.iloc[0]
        n_subjects = int((df["sub"].astype(str).str.strip() != "Average").sum())
        eeg_l0_mean = float(r["eeg_l0_mean"])
        rows.append(
            {
                "config": config_name,
                "feature_dim": feature_dim,
                "activation": activation,
                "n_subjects": n_subjects,
                "best_top1": float(r["best top1 acc"]),
                "top1_last_epoch": float(r["top1 acc"]),
                "eeg_l0_mean": eeg_l0_mean,
                "eeg_l0_pct": 100.0 * eeg_l0_mean / feature_dim,
                "eeg_l0_frac": float(r["eeg_l0_frac"]),
                "eeg_active_pct": 100.0 * float(r["eeg_active_feat_frac"]),
                "img_l0_pct": 100.0 * float(r["img_l0_mean"]) / feature_dim,
                "img_active_pct": 100.0 * float(r["img_active_feat_frac"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No inter_subject_summary.csv under {session_dir}")
    out = pd.DataFrame(rows)
    out["activation"] = pd.Categorical(out["activation"], categories=ACTIVATION_ORDER, ordered=True)
    return out.sort_values(["feature_dim", "activation"])


def plot_vs_feature_dim(df: pd.DataFrame, output_path: str) -> None:
    metrics = [
        ("best_top1", "Best top-1 accuracy (%)", "Retrieval accuracy"),
        ("eeg_l0_pct", "EEG L0 per sample (% of dim)", "EEG sparsity (L0 / feature_dim)"),
        ("eeg_active_pct", "EEG active dims (% of dim)", "EEG active feature coverage"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    fds = sorted(df["feature_dim"].unique())

    for activation in ACTIVATION_ORDER:
        sub = df[df["activation"] == activation]
        if sub.empty:
            continue
        label = ACTIVATION_LABELS[activation]
        color = ACTIVATION_COLORS[activation]
        marker = ACTIVATION_MARKERS[activation]
        x = sub["feature_dim"].values
        for ax, (col, ylabel, title) in zip(axes, metrics):
            ax.plot(x, sub[col].values, marker=marker, color=color, label=label, linewidth=2, markersize=9)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            if col == "eeg_l0_pct":
                ax.set_yscale("log")
                vals = sub[col].values
                ax.set_ylim(max(0.05, vals.min() * 0.4), vals.max() * 2.5)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(fds)
        ax.set_xticklabels([str(v) for v in fds])
        ax.set_xlabel("Alignment feature dim")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, max(35, df["best_top1"].max() + 4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=9)
    n_complete = int((df["n_subjects"] == 10).sum())
    fig.suptitle(
        f"Detached ||z|| confidence · bb=64 · LOSO ({n_complete}/{len(df)} configs at 10/10 subjects)",
        fontsize=11,
        y=1.14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tradeoff_scatter(df: pd.DataFrame, output_path: str) -> None:
    """Accuracy vs active coverage; point size ~ L0 mass (mean active dims)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    complete = df[df["n_subjects"] == 10]

    for activation in ACTIVATION_ORDER:
        sub = complete[complete["activation"] == activation]
        if sub.empty:
            continue
        ax.scatter(
            sub["eeg_active_pct"],
            sub["best_top1"],
            s=np.clip(sub["eeg_l0_mean"] / 8.0, 30, 400),
            c=ACTIVATION_COLORS[activation],
            marker=ACTIVATION_MARKERS[activation],
            edgecolors="k",
            linewidths=0.6,
            alpha=0.85,
            label=ACTIVATION_LABELS[activation],
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                str(int(row["feature_dim"])),
                (row["eeg_active_pct"], row["best_top1"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=ACTIVATION_COLORS[activation],
            )

    ax.set_xlabel("EEG active feature coverage (% of alignment dims)")
    ax.set_ylabel("Best top-1 accuracy (%)")
    ax.set_title("Sparsity–coverage–accuracy tradeoff (10/10 LOSO configs)\n"
                 "Point size ∝ mean L0 count per sample; labels = feature_dim")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary_table(df: pd.DataFrame) -> None:
    cols = [
        "config",
        "n_subjects",
        "best_top1",
        "eeg_l0_mean",
        "eeg_l0_pct",
        "eeg_active_pct",
    ]
    view = df[cols].copy()
    view["eeg_l0_pct"] = view["eeg_l0_pct"].map(lambda x: f"{x:.2f}")
    view["eeg_active_pct"] = view["eeg_active_pct"].map(lambda x: f"{x:.1f}")
    view["best_top1"] = view["best_top1"].map(lambda x: f"{x:.2f}")
    print("\n--- Sparsity vs active coverage vs accuracy ---\n")
    print(view.to_string(index=False))
    print(
        "\nNote: L0% = mean non-zero dims per test sample / feature_dim; "
        "active% = share of dims ever non-zero on test fold."
    )


def main() -> None:
    args = parse_args()
    session_dir = os.path.abspath(args.session_dir)
    prefix = args.output_prefix or os.path.join(session_dir, "sparsity_accuracy")
    df = collect_session(session_dir)
    csv_path = f"{prefix}_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print_summary_table(df)
    plot_vs_feature_dim(df, f"{prefix}_vs_fd.png")
    complete = df[df["n_subjects"] == 10]
    if len(complete) >= 2:
        plot_tradeoff_scatter(df, f"{prefix}_tradeoff.png")


if __name__ == "__main__":
    main()
