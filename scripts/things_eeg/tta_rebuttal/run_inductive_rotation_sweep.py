#!/usr/bin/env python3
"""Inductive subject adaptation: SAW-on-train + geodesic rotation, cosine-only test retrieval.

Calibration uses unique random subsets of the held-out subject's training split with
ground-truth EEG–image pairs for rotation fitting. No CSLS or other transductive scoring
on the 200-way test set.

Methods (per calibration size, seed, and alpha grid):
  - plain_cosine
  - saw_cosine (fit SAW on cal EEG, apply to test EEG)
  - geodesic_rotation (paired Procrustes on cal, R_alpha = exp(alpha * log R))
  - saw_geodesic_rotation (SAW on cal EEG, then rotation in whitened space)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm

from shared import (
    DEFAULT_OUTPUT_ROOT,
    TTAParams,
    add_common_args,
    aggregate_results,
    apply_saw_transform,
    cosine_scores,
    encode_average_features,
    encode_indexed_average_features,
    ensure_output_dir,
    evaluate_scores,
    fit_paired_orthogonal,
    fit_saw_transform,
    load_subject_train_test_context,
    normalize_rows,
    params_from_args,
    write_config,
)


DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "inductive_rotation_sweep")


def _safe_logm(rotation: np.ndarray) -> np.ndarray:
    L = logm(np.asarray(rotation, dtype=np.float64))
    return np.real(0.5 * (L - L.T))


def geodesic_rotation(rotation: np.ndarray, alpha: float) -> np.ndarray:
    """R_alpha = exp(alpha * log(R)); alpha=0 -> I, alpha=1 -> R."""
    alpha = float(alpha)
    d = rotation.shape[0]
    if alpha <= 0.0:
        return np.eye(d, dtype=np.float32)
    if alpha >= 1.0:
        return np.asarray(rotation, dtype=np.float32)
    A = _safe_logm(rotation)
    return expm(alpha * A).astype(np.float32)


def apply_geodesic_rotation(query_features: np.ndarray, rotation: np.ndarray, alpha: float) -> np.ndarray:
    R_alpha = geodesic_rotation(rotation, alpha)
    return normalize_rows(np.asarray(query_features, dtype=np.float32) @ R_alpha)


def evaluate_cosine_only(query_features, image_features, target_indices=None):
    """Inductive retrieval: L2-normalized cosine, no CSLS."""
    return evaluate_scores(cosine_scores(query_features, image_features), target_indices=target_indices)


def _parse_calibration_sizes(values, total: int) -> list[int]:
    sizes = []
    for value in values:
        if str(value).lower() == "all":
            size = total
        else:
            size = int(value)
        if 1 <= size <= total and size not in sizes:
            sizes.append(size)
    return sizes


def _append_row(rows, subject_id, seed, cal_size, alpha, method, metrics, plain_ref):
    alpha_str = f"{float(alpha):.4g}"
    rows.append(
        {
            "subject_id": int(subject_id),
            "seed": int(seed),
            "calibration_size": int(cal_size),
            "alpha": alpha_str,
            "method": method,
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
            "delta_top1_vs_plain": round(float(metrics["top1_acc"]) - float(plain_ref["top1_acc"]), 4),
            "delta_top5_vs_plain": round(float(metrics["top5_acc"]) - float(plain_ref["top5_acc"]), 4),
        }
    )


def run_subject(source_run_dir: str, subject_id: int, args, params: TTAParams):
    checkpoint_dir, eval_args, train_dataset, test_dataset, modules = load_subject_train_test_context(
        source_run_dir, args, subject_id, average=True
    )
    train_q, train_i, _, _ = encode_indexed_average_features(eval_args, modules, train_dataset)
    test_q, test_i = encode_average_features(eval_args, modules, test_dataset)
    total_train = int(train_q.shape[0])
    cal_sizes = _parse_calibration_sizes(args.calibration_sizes, total_train)
    alphas = [float(a) for a in args.alpha_values]

    plain_metrics = evaluate_cosine_only(test_q, test_i)
    rows = []

    for cal_size in cal_sizes:
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(cal_size))
            cal_idx = np.sort(rng.choice(total_train, size=cal_size, replace=False))
            cal_q = train_q[cal_idx]
            cal_i = train_i[cal_idx]

            _, saw_stats = fit_saw_transform(cal_q, params)
            test_w = apply_saw_transform(test_q, saw_stats)
            saw_metrics = evaluate_cosine_only(test_w, test_i)

            rotation_raw = fit_paired_orthogonal(cal_q, cal_i)
            cal_w = apply_saw_transform(cal_q, saw_stats)
            rotation_whitened = fit_paired_orthogonal(cal_w, cal_i)

            for alpha in alphas:
                _append_row(rows, subject_id, seed, cal_size, alpha, "plain_cosine", plain_metrics, plain_metrics)
                _append_row(rows, subject_id, seed, cal_size, alpha, "saw_cosine", saw_metrics, plain_metrics)

                test_rot = apply_geodesic_rotation(test_q, rotation_raw, alpha)
                _append_row(
                    rows,
                    subject_id,
                    seed,
                    cal_size,
                    alpha,
                    "geodesic_rotation",
                    evaluate_cosine_only(test_rot, test_i),
                    plain_metrics,
                )

                test_w_rot = apply_geodesic_rotation(test_w, rotation_whitened, alpha)
                _append_row(
                    rows,
                    subject_id,
                    seed,
                    cal_size,
                    alpha,
                    "saw_geodesic_rotation",
                    evaluate_cosine_only(test_w_rot, test_i),
                    plain_metrics,
                )

    for row in rows:
        row["total_train_samples"] = total_train
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def _write_summaries(subject_df: pd.DataFrame, output_dir: str):
    subject_df.to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    average_df = aggregate_results(
        subject_df,
        ["method", "calibration_size", "alpha"],
    )
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    delta_df = (
        subject_df.groupby(["method", "calibration_size", "alpha"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top5_mean=("top5_acc", "mean"),
            delta_top1_vs_plain_mean=("delta_top1_vs_plain", "mean"),
            delta_top5_vs_plain_mean=("delta_top5_vs_plain", "mean"),
        )
        .sort_values(["calibration_size", "method", "alpha"])
    )
    delta_df.to_csv(os.path.join(output_dir, "delta_vs_plain.csv"), index=False)

    # Best alpha per (method, calibration_size) averaged over seeds and subjects.
    rot_df = subject_df[subject_df["method"].isin(["geodesic_rotation", "saw_geodesic_rotation"])]
    if len(rot_df):
        best_alpha = (
            rot_df.groupby(["method", "calibration_size", "alpha"], as_index=False)["top1_acc"]
            .mean()
            .sort_values(["method", "calibration_size", "top1_acc"], ascending=[True, True, False])
            .groupby(["method", "calibration_size"], as_index=False)
            .first()
        )
        best_alpha.to_csv(os.path.join(output_dir, "best_alpha_per_calibration_size.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--calibration_sizes",
        nargs="+",
        default=["5", "10", "20", "50", "100", "200", "500", "1000", "2000", "5000", "10000", "all"],
    )
    parser.add_argument(
        "--alpha_values",
        nargs="+",
        type=float,
        default=[round(x, 4) for x in np.linspace(0.0, 1.0, 21)],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[3300],
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "inductive_rotation_sweep")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[inductive_rotation_sweep] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        _write_summaries(pd.DataFrame(all_rows), output_dir)

    subject_df = pd.DataFrame(all_rows)
    _write_summaries(subject_df, output_dir)

    n_cal = len(_parse_calibration_sizes(args.calibration_sizes, 16540))
    n_alpha = len(args.alpha_values)
    n_seed = len(args.seeds)
    print(
        f"\nGrid: {n_cal} calibration sizes × {n_alpha} alphas × {n_seed} seeds "
        f"× {len(args.subjects)} subjects"
    )
    print("Methods per (cal_size, alpha, seed): plain_cosine, saw_cosine, geodesic_rotation, saw_geodesic_rotation")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
