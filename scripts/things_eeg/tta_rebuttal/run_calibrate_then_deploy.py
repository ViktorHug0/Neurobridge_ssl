#!/usr/bin/env python3
"""Experiment A: label-free calibration block, then frozen per-trial deployment.

Scenario (S1, calibrated closed-menu BCI): a new user runs a short calibration
block in which each of the K menu items is presented once, in an order the system
never records. We fit the SAGE-TTA calibration on that block -- square K-vs-K, so
the bijective Sinkhorn prior is supplied by the protocol rather than assumed of
the benchmark, and no labels are used. The map is then FROZEN and applied to a
disjoint set of EEG repetitions of the same menu, decoded one trial at a time
with plain cosine (no batch statistics of any kind).

The 80 test repetitions of THINGS-EEG-2 are partitioned into two disjoint blocks,
so the evaluation trials are genuinely fresh measurements.

Two slices through the (fit_reps, eval_reps) grid:
  * fit_reps swept, eval_reps fixed -> how much calibration does the user owe us
    (reported in seconds of recording: K * fit_reps * 0.2 s SOA).
  * eval_reps swept, fit_reps fixed -> how well it holds up as deployment-time
    trial averaging shrinks toward single trials.

Both a strictly per-trial score (plain cosine) and a batch-coupled score (CSLS
over the deployed block) are recorded for every cell.
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
    cosine_scores,
    encode_repetition_blocks,
    ensure_output_dir,
    evaluate_full_tta,
    evaluate_scores,
    fit_tta_calibration,
    load_subject_context,
    params_from_args,
    score_features,
    write_config,
)

SOA_SECONDS = 0.2  # THINGS-EEG-2 RSVP: 100 ms stimulus + 100 ms blank


def _parse_grid(values):
    grid = []
    for item in values:
        fit_reps, eval_reps = item.split(":")
        grid.append((int(fit_reps), int(eval_reps)))
    return grid


def _append(rows, **kwargs):
    metrics = kwargs.pop("metrics")
    rows.append({**kwargs, "top1_acc": round(float(metrics["top1_acc"]), 4), "top5_acc": round(float(metrics["top5_acc"]), 4)})


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(source_run_dir, args, subject_id, average=False)
    grid = _parse_grid(args.rep_grid)
    rows = []

    for seed in args.seeds:
        for fit_reps, eval_reps in grid:
            (fit_queries, eval_queries), image_features, available = encode_repetition_blocks(
                eval_args,
                modules,
                dataset,
                block_sizes=[fit_reps, eval_reps],
                seed=int(seed) * 100000 + int(subject_id) * 1000 + fit_reps * 10 + eval_reps,
            )
            num_objects = image_features.shape[0]
            rng = np.random.default_rng(int(seed) * 7919 + int(subject_id))

            for menu_size in args.menu_sizes:
                menu_size = int(min(menu_size, num_objects))
                idx = np.sort(rng.choice(num_objects, size=menu_size, replace=False)) if menu_size < num_objects \
                    else np.arange(num_objects)
                menu_images = image_features[idx]
                fit_block = fit_queries[idx]
                eval_block = eval_queries[idx]

                # Label-free calibration on the fit block only (square menu_size x menu_size).
                calibration = fit_tta_calibration(fit_block, menu_images, params)
                deployed = apply_tta_calibration(eval_block, calibration, alpha=1.0)

                common = dict(
                    subject_id=int(subject_id),
                    seed=int(seed),
                    fit_reps=int(fit_reps),
                    eval_reps=int(eval_reps),
                    menu_size=int(menu_size),
                    available_repetitions=int(available),
                    calibration_seconds=round(menu_size * fit_reps * SOA_SECONDS, 1),
                )

                # --- deployment on the held-out repetition block ---
                _append(rows, **common, method="no_adapt_percall",
                        metrics=evaluate_scores(cosine_scores(eval_block, menu_images)))
                _append(rows, **common, method="no_adapt_batch_csls",
                        metrics=evaluate_scores(score_features(eval_block, menu_images, use_csls=True, csls_k=params.csls_k)))
                _append(rows, **common, method="calibrated_percall",
                        metrics=evaluate_scores(cosine_scores(deployed, menu_images)))
                _append(rows, **common, method="calibrated_batch_csls",
                        metrics=evaluate_scores(score_features(deployed, menu_images, use_csls=True, csls_k=params.csls_k)))

                # --- references ---
                # what the paper's protocol would report if it refit on the deployment block itself
                _append(rows, **common, method="fresh_tta_on_eval_block",
                        metrics=evaluate_full_tta(eval_block, menu_images, params))
                # what the calibration achieved in-sample, on the block it was fit on
                fitted = apply_tta_calibration(fit_block, calibration, alpha=1.0)
                _append(rows, **common, method="calibration_insample",
                        metrics=evaluate_scores(score_features(fitted, menu_images, use_csls=True, csls_k=params.csls_k)))

    for row in rows:
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument(
        "--rep_grid",
        nargs="+",
        default=["40:40", "5:40", "10:40", "20:40", "40:1", "40:2", "40:5", "40:10", "40:20"],
        help="fit_reps:eval_reps pairs. The two blocks are disjoint, so the sum must not exceed 80.",
    )
    parser.add_argument("--menu_sizes", nargs="+", type=int, default=[10, 25, 50, 100, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "calibrate_then_deploy")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[calibrate_then_deploy] subject={int(subject_id):02d}", flush=True)
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["fit_reps", "eval_reps", "menu_size", "method"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    headline = average_df[
        (average_df["fit_reps"] == 40) & (average_df["eval_reps"] == 40)
    ]
    print("\n=== Calibrate-then-deploy: 40 calibration reps / 40 held-out reps ===")
    print(headline.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
