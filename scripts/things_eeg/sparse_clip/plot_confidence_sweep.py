"""Plot best top-1 and EEG/Image sparsity vs feature_dim for sparse CLIP confidence sweep."""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


MODE_ORDER = [
    "no_eeg_norm",
    "detached_confidence",
    "learned_confidence",
]

MODE_LABELS = {
    "no_eeg_norm": "No EEG L2 norm (implicit ||z||)",
    "detached_confidence": "EEG norm + detached ||z||",
    "learned_confidence": "EEG norm + learned confidence",
}

MODE_COLORS = {
    "no_eeg_norm": "#4C72B0",
    "detached_confidence": "#55A868",
    "learned_confidence": "#DD8452",
}

MODE_MARKERS = {
    "no_eeg_norm": "o",
    "detached_confidence": "^",
    "learned_confidence": "D",
}


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_csv = os.path.join(
        repo_root,
        "results/things_eeg/inter-subject-sparse/"
        "sparse_clip_confidence_bb64_seed3300_20260527-172056/sweep_summary.csv",
    )
    default_out = os.path.join(os.path.dirname(default_csv), "confidence_sweep_summary.png")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=default_csv, help="Path to sweep_summary.csv")
    p.add_argument("--output", default=default_out, help="Output PNG path")
    return p.parse_args()


def infer_mode(config: str) -> str:
    if "baseline_noeegnorm" in config:
        return "no_eeg_norm"
    if "detached_confidence" in config:
        return "detached_confidence"
    if "learned_confidence" in config:
        return "learned_confidence"
    raise ValueError(f"Unknown config variant: {config}")


def load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["sub"].astype(str).str.strip().eq("Average")].copy()
    df["feature_dim"] = df["config"].str.extract(r"sparse_fd(\d+)_", expand=False).astype(int)
    df["mode"] = df["config"].map(infer_mode)
    df["best_top1"] = pd.to_numeric(df["best top1 acc"], errors="coerce")
    df["eeg_l0_frac"] = pd.to_numeric(df["eeg_l0_frac"], errors="coerce")
    df["eeg_l0_mean"] = pd.to_numeric(df["eeg_l0_mean"], errors="coerce")
    df["eeg_active_feat_frac"] = pd.to_numeric(df["eeg_active_feat_frac"], errors="coerce")
    if "eeg_active_feat_frac_train" in df.columns:
        df["eeg_active_feat_frac_train"] = pd.to_numeric(
            df["eeg_active_feat_frac_train"], errors="coerce"
        )
    df["img_l0_frac"] = pd.to_numeric(df["img_l0_frac"], errors="coerce")
    df["img_l0_mean"] = pd.to_numeric(df["img_l0_mean"], errors="coerce")
    df["img_active_feat_frac"] = pd.to_numeric(df["img_active_feat_frac"], errors="coerce")
    # Use mean/dim when summary rounds eeg_l0_frac to 0.0
    eeg_l0_frac = df["eeg_l0_mean"] / df["feature_dim"]
    eeg_l0_frac = eeg_l0_frac.fillna(df["eeg_l0_frac"])
    df["eeg_l0_pct"] = (eeg_l0_frac * 100).clip(lower=0.05)
    # Use mean/dim when summary rounds img_l0_frac to 0.0
    img_l0_frac = df["img_l0_mean"] / df["feature_dim"]
    img_l0_frac = img_l0_frac.fillna(df["img_l0_frac"])
    df["img_l0_pct"] = (img_l0_frac * 100).clip(lower=0.05)
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df.sort_values(["feature_dim", "mode"])


def plot_summary(df: pd.DataFrame, output_path: str) -> None:
    has_train_active = "eeg_active_feat_frac_train" in df.columns and df["eeg_active_feat_frac_train"].notna().any()
    ncols = 6 if has_train_active else 5
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.8), sharex=True)
    if has_train_active:
        ax_acc, ax_eeg_sp, ax_eeg_active_test, ax_eeg_active_train, ax_img_sp, ax_img_active = axes
    else:
        ax_acc, ax_eeg_sp, ax_eeg_active_test, ax_img_sp, ax_img_active = axes
        ax_eeg_active_train = None

    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        x = sub["feature_dim"].values
        ax_acc.plot(
            x,
            sub["best_top1"].values,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=2,
            markersize=8,
        )
        ax_eeg_sp.plot(
            x,
            sub["eeg_l0_pct"].values,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=2,
            markersize=8,
        )
        ax_eeg_active_test.plot(
            x,
            (sub["eeg_active_feat_frac"].values * 100.0),
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=2,
            markersize=8,
        )
        if ax_eeg_active_train is not None and "eeg_active_feat_frac_train" in sub.columns:
            ax_eeg_active_train.plot(
                x,
                (sub["eeg_active_feat_frac_train"].values * 100.0),
                marker=MODE_MARKERS[mode],
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
                linewidth=2,
                markersize=8,
            )
        ax_img_sp.plot(
            x,
            sub["img_l0_pct"].values,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=2,
            markersize=8,
        )
        ax_img_active.plot(
            x,
            (sub["img_active_feat_frac"].values * 100.0),
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=2,
            markersize=8,
        )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["feature_dim"].unique()))
        ax.set_xticklabels([str(v) for v in sorted(df["feature_dim"].unique())])
        ax.set_xlabel("Alignment feature dim")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    ax_acc.set_ylabel("Best top-1 accuracy (%)")
    ax_acc.set_title("Retrieval (test-selected checkpoint)")
    ax_acc.set_ylim(0, max(45, df["best_top1"].max() + 5))

    ax_eeg_sp.set_ylabel("EEG L0 fraction (%)")
    ax_eeg_sp.set_title("EEG sparsity (post-ReLU, log scale)")
    eeg_pct = df["eeg_l0_pct"].values
    eeg_y_min = max(0.05, eeg_pct.min() * 0.5)
    eeg_y_max = max(55, eeg_pct.max() * 1.5)
    ax_eeg_sp.set_yscale("log")
    ax_eeg_sp.set_ylim(eeg_y_min, eeg_y_max)

    ax_eeg_active_test.set_ylabel("EEG active feature fraction (%)")
    ax_eeg_active_test.set_title("EEG active feature coverage on test set")
    eeg_active_pct = (df["eeg_active_feat_frac"] * 100.0).dropna()
    if not eeg_active_pct.empty:
        ax_eeg_active_test.set_ylim(max(0, eeg_active_pct.min() - 5), min(100, eeg_active_pct.max() + 5))

    if ax_eeg_active_train is not None:
        ax_eeg_active_train.set_ylabel("EEG active feature fraction (%)")
        ax_eeg_active_train.set_title("EEG active feature coverage on train set")
        train_active_pct = (df["eeg_active_feat_frac_train"] * 100.0).dropna()
        if not train_active_pct.empty:
            ax_eeg_active_train.set_ylim(
                max(0, train_active_pct.min() - 5),
                min(100, train_active_pct.max() + 5),
            )

    ax_img_sp.set_ylabel("Image L0 fraction (%)")
    ax_img_sp.set_title("Image sparsity (post-ReLU, log scale)")
    img_pct = df["img_l0_pct"].values
    img_y_min = max(0.05, img_pct.min() * 0.5)
    img_y_max = max(55, img_pct.max() * 1.5)
    ax_img_sp.set_yscale("log")
    ax_img_sp.set_ylim(img_y_min, img_y_max)

    ax_img_active.set_ylabel("Image active feature fraction (%)")
    ax_img_active.set_title("Image active feature coverage on test set")
    img_active_pct = (df["img_active_feat_frac"] * 100.0).dropna()
    if not img_active_pct.empty:
        ax_img_active.set_ylim(max(0, img_active_pct.min() - 5), min(100, img_active_pct.max() + 5))

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=9)
    fig.suptitle(
        "Sparse CLIP confidence sweep (bb=64, ReLU, avg LOSO subjects, plain cosine)",
        fontsize=11,
        y=1.18,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    df = load_summary(os.path.abspath(args.csv))
    plot_summary(df, os.path.abspath(args.output))


if __name__ == "__main__":
    main()
