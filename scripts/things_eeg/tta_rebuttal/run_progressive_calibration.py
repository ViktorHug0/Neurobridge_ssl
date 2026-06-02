#!/usr/bin/env python3
"""Progressive full-set TTA calibration.

This is the rebuttal-facing version of the older progressive SATTC candidate
sweep. It asks: as we expose the transductive calibrator to more unlabeled test
queries, how quickly does performance improve?
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from shared import (
    add_common_args,
    aggregate_results,
    encode_average_features,
    ensure_output_dir,
    evaluate_full_tta,
    evaluate_plain,
    evaluate_saw_csls,
    load_subject_context,
    params_from_args,
    write_config,
)


def _sample_counts(args, total):
    if args.sample_counts:
        return [int(x) for x in args.sample_counts if 1 <= int(x) <= total]
    return list(range(max(1, args.min_samples), min(total, args.max_samples) + 1, args.sample_step))


def _append_metrics(rows, subject_id, seed, sample_count, regime, method, metrics):
    rows.append(
        {
            "subject_id": int(subject_id),
            "seed": int(seed),
            "sample_count": int(sample_count),
            "regime": regime,
            "method": method,
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
        }
    )


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(source_run_dir, args, subject_id, average=True)
    query_features, image_features = encode_average_features(eval_args, modules, dataset)
    total = query_features.shape[0]
    rows = []

    for sample_count in _sample_counts(args, total):
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(sample_count))
            selected = np.sort(rng.choice(total, size=sample_count, replace=False))
            query_subset = query_features[selected]

            for regime in ["matching", "all_candidates"]:
                if regime == "matching":
                    candidates = image_features[selected]
                    targets = np.arange(sample_count, dtype=np.int64)
                else:
                    candidates = image_features
                    targets = selected

                _append_metrics(
                    rows,
                    subject_id,
                    seed,
                    sample_count,
                    regime,
                    "plain_cosine",
                    evaluate_plain(query_subset, candidates, target_indices=targets),
                )
                _append_metrics(
                    rows,
                    subject_id,
                    seed,
                    sample_count,
                    regime,
                    "saw_csls",
                    evaluate_saw_csls(query_subset, candidates, params, target_indices=targets),
                )
                _append_metrics(
                    rows,
                    subject_id,
                    seed,
                    sample_count,
                    regime,
                    "full_tta",
                    evaluate_full_tta(query_subset, candidates, params, target_indices=targets),
                )

    for row in rows:
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--sample_step", type=int, default=5)
    parser.add_argument("--sample_counts", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "progressive_calibration")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[progressive_calibration] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["regime", "method", "sample_count"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    print("\n=== Progressive Calibration Average Results ===")
    print(average_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
