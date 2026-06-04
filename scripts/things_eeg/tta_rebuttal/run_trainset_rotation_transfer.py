#!/usr/bin/env python3
"""Fit target-subject adapters on the subject's train split and test on 200-way retrieval.

The calibration split uses the left-out subject's THINGS-EEG training images
(4 repetitions averaged per image). A paired orthogonal map is fit on the
calibration set and evaluated on the 200-object test set at each alpha level.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from shared import (
    DEFAULT_OUTPUT_ROOT,
    TTAParams,
    add_common_args,
    aggregate_results,
    apply_supervised_orthogonal,
    encode_average_features,
    encode_indexed_average_features,
    ensure_output_dir,
    evaluate_plain,
    evaluate_scores,
    fit_paired_orthogonal,
    load_subject_train_test_context,
    params_from_args,
    score_features,
    write_config,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "rebuttal_suite_featdim64_mixup",
    "trainset_rotation_transfer_featdim64_mixup",
)


def _parse_calibration_sizes(values, total: int):
    sizes = []
    for value in values:
        if str(value).lower() == "all":
            size = total
        else:
            size = int(value)
        if 1 <= size <= total and size not in sizes:
            sizes.append(size)
    return sizes


def _evaluate_test(transformed_test, test_images, params: TTAParams):
    scores = score_features(transformed_test, test_images, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores)


def _append(rows, subject_id, seed, calibration_size, method, hyperparam, metrics, plain_ref):
    rows.append(
        {
            "subject_id": int(subject_id),
            "seed": int(seed),
            "calibration_size": int(calibration_size),
            "fit_size": int(calibration_size),
            "val_size": 0,
            "method": method,
            "hyperparam": hyperparam or "",
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
            "delta_top1_vs_plain": round(float(metrics["top1_acc"]) - float(plain_ref["top1_acc"]), 4),
            "delta_top5_vs_plain": round(float(metrics["top5_acc"]) - float(plain_ref["top5_acc"]), 4),
        }
    )


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, train_dataset, test_dataset, modules = load_subject_train_test_context(
        source_run_dir, args, subject_id, average=True
    )
    train_q, train_i, _, _ = encode_indexed_average_features(eval_args, modules, train_dataset)
    test_q, test_i = encode_average_features(eval_args, modules, test_dataset)
    total_train = train_q.shape[0]
    sizes = _parse_calibration_sizes(args.calibration_sizes, total_train)
    rows = []

    plain_metrics = evaluate_plain(test_q, test_i)

    for calibration_size in sizes:
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(calibration_size))
            calibration_idx = np.sort(rng.choice(total_train, size=calibration_size, replace=False))
            calibration_q = train_q[calibration_idx]
            calibration_i = train_i[calibration_idx]

            _append(rows, subject_id, seed, calibration_size, "plain_cosine", None, plain_metrics, plain_metrics)

            rotation = fit_paired_orthogonal(calibration_q, calibration_i)
            for alpha in args.alpha_values:
                transformed = apply_supervised_orthogonal(test_q, rotation, alpha=float(alpha))
                _append(
                    rows,
                    subject_id,
                    seed,
                    calibration_size,
                    "trainset_orthogonal",
                    f"alpha={float(alpha)}",
                    _evaluate_test(transformed, test_i, params),
                    plain_metrics,
                )

    for row in rows:
        row["total_train_samples"] = int(total_train)
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--calibration_sizes", nargs="+", default=["100", "500", "1000", "5000", "10000", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "trainset_rotation_transfer_featdim64_mixup")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[trainset_rotation_transfer] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["method", "calibration_size", "hyperparam"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    delta_df = (
        subject_df.groupby(["method", "calibration_size", "hyperparam"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top5_mean=("top5_acc", "mean"),
            delta_top1_vs_plain_mean=("delta_top1_vs_plain", "mean"),
            delta_top5_vs_plain_mean=("delta_top5_vs_plain", "mean"),
        )
        .sort_values(["calibration_size", "delta_top1_vs_plain_mean"], ascending=[True, False])
    )
    delta_df.to_csv(os.path.join(output_dir, "delta_vs_plain.csv"), index=False)

    print("\n=== Train-set Rotation Transfer (mean delta vs plain 200-way test) ===")
    print(delta_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
