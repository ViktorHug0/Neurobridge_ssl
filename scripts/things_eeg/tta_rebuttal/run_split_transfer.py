#!/usr/bin/env python3
"""Split-transfer TTA calibration.

Fit the label-free calibration map on one subset of the held-out subject's test
set, freeze it, and evaluate on a disjoint subset. This is the most deployment-
oriented rebuttal experiment because it separates calibration evidence from
evaluation queries.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from shared import (
    add_common_args,
    aggregate_results,
    apply_tta_calibration,
    encode_average_features,
    ensure_output_dir,
    evaluate_plain,
    evaluate_saw_csls,
    fit_tta_calibration,
    load_subject_context,
    params_from_args,
    score_features,
    evaluate_scores,
    write_config,
)


def _fit_sizes(args, total):
    if args.fit_sizes:
        return [int(x) for x in args.fit_sizes if 1 <= int(x) < total]
    return list(range(5, total, 5))


def _append(rows, subject_id, seed, fit_size, scope, method, alpha, metrics):
    rows.append(
        {
            "subject_id": int(subject_id),
            "seed": int(seed),
            "fit_size": int(fit_size),
            "scope": scope,
            "method": method,
            "alpha": np.nan if alpha is None else float(alpha),
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
        }
    )


def _evaluate_transformed(transformed_query, candidates, params, targets):
    scores = score_features(transformed_query, candidates, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores, target_indices=targets)


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(source_run_dir, args, subject_id, average=True)
    query_features, image_features = encode_average_features(eval_args, modules, dataset)
    total = query_features.shape[0]
    rows = []

    for fit_size in _fit_sizes(args, total):
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(fit_size))
            fit_indices = np.sort(rng.choice(total, size=fit_size, replace=False))
            eval_indices = np.setdiff1d(np.arange(total, dtype=np.int64), fit_indices)

            fit_query = query_features[fit_indices]
            fit_images = image_features[fit_indices]
            eval_query = query_features[eval_indices]
            eval_images = image_features[eval_indices]
            calibration = fit_tta_calibration(fit_query, fit_images, params)

            scopes = [
                ("remainder_vs_remainder", eval_images, np.arange(len(eval_indices), dtype=np.int64)),
                ("remainder_vs_all_candidates", image_features, eval_indices),
            ]
            for scope, candidates, targets in scopes:
                _append(
                    rows,
                    subject_id,
                    seed,
                    fit_size,
                    scope,
                    "plain_cosine",
                    None,
                    evaluate_plain(eval_query, candidates, target_indices=targets),
                )
                _append(
                    rows,
                    subject_id,
                    seed,
                    fit_size,
                    scope,
                    "fresh_saw_csls_on_eval",
                    None,
                    evaluate_saw_csls(eval_query, candidates, params, target_indices=targets),
                )
                for alpha in args.alpha_values:
                    transformed = apply_tta_calibration(eval_query, calibration, alpha=float(alpha))
                    _append(
                        rows,
                        subject_id,
                        seed,
                        fit_size,
                        scope,
                        "frozen_tta_from_fit_split",
                        alpha,
                        _evaluate_transformed(transformed, candidates, params, targets),
                    )

    for row in rows:
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--fit_sizes", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175])
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "split_transfer")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[split_transfer] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["scope", "method", "fit_size", "alpha"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    best_df = (
        average_df[average_df["method"] == "frozen_tta_from_fit_split"]
        .sort_values(["scope", "fit_size", "top1_mean"], ascending=[True, True, False])
        .groupby(["scope", "fit_size"], as_index=False)
        .head(1)
    )
    best_df.to_csv(os.path.join(output_dir, "best_alpha_results.csv"), index=False)

    print("\n=== Split Transfer Best Frozen-TTA Results ===")
    print(best_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
