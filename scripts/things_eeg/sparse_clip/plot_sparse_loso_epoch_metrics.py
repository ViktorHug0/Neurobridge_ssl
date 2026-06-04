#!/usr/bin/env python3
"""Extract per-epoch accuracy/sparsity from TensorBoard and plot LOSO sparse runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.things_eeg.sparse_clip.measure_session_train_sparsity import collect_train_sparsity

SCALAR_TAGS = [
    "Acc/top1_test",
    "Acc/top5_test",
    "Sparsity/eeg_l0_frac_test",
    "Sparsity/img_l0_frac_test",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Config run dir containing *-sub-* folders")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Where to write CSVs/PNGs (default: <run_dir>/sparse_epoch_plots)",
    )
    p.add_argument("--device", default=None, help="Device for train-set sparsity at best checkpoint")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--skip_train_sparsity", action="store_true")
    return p.parse_args()


def discover_subject_runs(run_dir: str) -> list[tuple[str, str]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*-sub-*"))):
        if not os.path.isdir(path):
            continue
        sub_label = path.rsplit("-", 1)[-1]
        if not re.fullmatch(r"sub-\d{2}", sub_label):
            continue
        runs.append((sub_label, path))
    return runs


def load_tensorboard_scalars(run_path: str, tags: list[str]) -> dict[str, list[tuple[int, float]]]:
    event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
    if not event_files:
        return {tag: [] for tag in tags}

    ea = EventAccumulator(run_path, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        if tag not in available:
            out[tag] = []
            continue
        events = ea.Scalars(tag)
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def scalars_to_frame(scalars: dict[str, list[tuple[int, float]]]) -> pd.DataFrame:
    rows: dict[int, dict[str, float]] = {}
    for tag, series in scalars.items():
        col = tag.split("/", 1)[-1]
        for step, value in series:
            rows.setdefault(step, {})[col] = value
    if not rows:
        return pd.DataFrame(columns=["epoch"])
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "epoch"
    return df.reset_index()


def read_best_epoch(run_path: str) -> int | None:
    result_csv = os.path.join(run_path, "result.csv")
    if not os.path.isfile(result_csv):
        return None
    df = pd.read_csv(result_csv)
    if df.empty or "best epoch" not in df.columns:
        return None
    try:
        return int(df.iloc[0]["best epoch"])
    except (TypeError, ValueError):
        return None


def plot_subject_epoch_curves(
    sub_label: str,
    epoch_df: pd.DataFrame,
    best_epoch: int | None,
    feature_dim: int,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_acc = axes[0]
    if "top1_test" in epoch_df.columns:
        ax_acc.plot(epoch_df["epoch"], epoch_df["top1_test"], color="#1f77b4", linewidth=2, label="Top-1 test")
    if "top5_test" in epoch_df.columns:
        ax_acc.plot(
            epoch_df["epoch"],
            epoch_df["top5_test"],
            color="#1f77b4",
            linewidth=1.5,
            linestyle="--",
            alpha=0.75,
            label="Top-5 test",
        )
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title(f"{sub_label}: retrieval accuracy vs epoch")
    ax_acc.grid(True, linestyle="--", alpha=0.35)
    ax_acc.legend(loc="lower right")

    ax_sp = axes[1]
    if "eeg_l0_frac_test" in epoch_df.columns:
        ax_sp.plot(
            epoch_df["epoch"],
            100.0 * epoch_df["eeg_l0_frac_test"],
            color="#d62728",
            linewidth=2,
            label="EEG L0 / dim (test, per sample)",
        )
    if "img_l0_frac_test" in epoch_df.columns:
        ax_sp.plot(
            epoch_df["epoch"],
            100.0 * epoch_df["img_l0_frac_test"],
            color="#ff7f0e",
            linewidth=2,
            linestyle="--",
            label="Image L0 / dim (test, per sample)",
        )
    ax_sp.set_xlabel("Epoch")
    ax_sp.set_ylabel("Mean active dims (% of alignment dim)")
    ax_sp.set_title("Alignment sparsity on test split (post-ReLU, pre-L2-norm)")
    ax_sp.grid(True, linestyle="--", alpha=0.35)
    ax_sp.legend(loc="upper right")

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, color="black", linestyle=":", linewidth=1.2, alpha=0.8, label="best epoch")
        handles, labels = axes[0].get_legend_handles_labels()
        if "best epoch" not in labels:
            axes[0].legend(handles + axes[0].lines[-1:], labels + ["best epoch"], loc="lower right")

    fig.suptitle(f"Sparse alignment | feature_dim={feature_dim}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_train_active_at_best(summary_df: pd.DataFrame, output_path: str) -> None:
    if summary_df.empty:
        return
    order = sorted(summary_df["sub_label"].unique())
    plot_df = summary_df.set_index("sub_label").loc[order].reset_index()
    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        100.0 * plot_df["eeg_active_feat_frac_train"],
        width,
        label="EEG active dims (train, full split)",
        color="#2ca02c",
    )
    ax.bar(
        x + width / 2,
        100.0 * plot_df["img_active_feat_frac_train"],
        width,
        label="Image active dims (train, full split)",
        color="#9467bd",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["sub_label"], rotation=0)
    ax.set_ylabel("Active feature coverage (% of alignment dim)")
    ax.set_title("Train-set active alignment features at best checkpoint")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def infer_feature_dim(run_dir: str) -> int:
    for run_path in glob.glob(os.path.join(run_dir, "*-sub-*")):
        cfg_path = os.path.join(run_path, "train_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            return int(cfg.get("feature_dim", 0) or 0)
    return 0


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(run_dir, "sparse_epoch_plots"))
    os.makedirs(output_dir, exist_ok=True)

    subject_runs = discover_subject_runs(run_dir)
    if not subject_runs:
        raise FileNotFoundError(f"No *-sub-* run folders under {run_dir}")

    feature_dim = infer_feature_dim(run_dir)
    epoch_rows = []
    train_rows = []

    import torch

    for sub_label, run_path in subject_runs:
        print(f"Processing {sub_label} ...", flush=True)
        scalars = load_tensorboard_scalars(run_path, SCALAR_TAGS)
        epoch_df = scalars_to_frame(scalars)
        if epoch_df.empty:
            print(f"  warning: no TensorBoard scalars in {run_path}", flush=True)
            continue

        epoch_df.insert(0, "sub_label", sub_label)
        epoch_csv = os.path.join(output_dir, f"{sub_label}_epoch_metrics.csv")
        epoch_df.to_csv(epoch_csv, index=False)
        epoch_rows.append(epoch_df)

        best_epoch = read_best_epoch(run_path)
        plot_path = os.path.join(output_dir, f"{sub_label}_accuracy_sparsity_epochs.png")
        plot_subject_epoch_curves(sub_label, epoch_df, best_epoch, feature_dim, plot_path)
        print(f"  saved {epoch_csv}", flush=True)
        print(f"  saved {plot_path}", flush=True)

        if args.skip_train_sparsity:
            continue
        ckpt_path = os.path.join(run_path, "checkpoint_test_best.pth")
        cfg_path = os.path.join(run_path, "train_config.json")
        if not (os.path.isfile(ckpt_path) and os.path.isfile(cfg_path)):
            print(f"  warning: missing checkpoint/config for train sparsity in {run_path}", flush=True)
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        device_str = args.device or cfg.get("device", "cuda:0")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)
        sp = collect_train_sparsity(cfg, ckpt_path, device, args.batch_size)
        train_rows.append({"sub_label": sub_label, "run_dir": run_path, **sp})

    if epoch_rows:
        all_epochs = pd.concat(epoch_rows, ignore_index=True)
        all_epochs_path = os.path.join(output_dir, "all_subjects_epoch_metrics.csv")
        all_epochs.to_csv(all_epochs_path, index=False)
        print(f"Saved combined epoch metrics: {all_epochs_path}", flush=True)

    if train_rows:
        train_df = pd.DataFrame(train_rows)
        train_csv = os.path.join(output_dir, "train_active_features_at_best.csv")
        train_df.to_csv(train_csv, index=False)
        plot_train_active_at_best(
            train_df,
            os.path.join(output_dir, "train_active_features_at_best.png"),
        )
        print(f"Saved train active-feature summary: {train_csv}", flush=True)
        print(f"Saved train active-feature plot: {os.path.join(output_dir, 'train_active_features_at_best.png')}", flush=True)


if __name__ == "__main__":
    main()
