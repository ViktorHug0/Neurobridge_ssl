#!/usr/bin/env python3
"""Select SAGE-TTA hyperparameters without using the held-out subject's test labels.

The paper's operating point (rho=0.94, k=3, tau=0.1, 12 Sinkhorn iters, 16
Procrustes steps, power 1.2) is the argmax of a grid scored directly on held-out
test accuracy. This script keeps the encoder checkpoint fixed and re-selects the
TTA hyperparameters under protocols that never look at the test subject's labels,
then reports what each protocol costs relative to that oracle.

Selection protocols, in decreasing order of what they assume:

  oracle_test      argmax of the test subject's own top-1. NOT honest; the
                   reference the others are measured against.
  loso_fold_cv     argmax of the MEAN top-1 over the other nine LOSO folds, each
                   scored on its own legitimately held-out subject with its own
                   checkpoint. Uses no information about subject s whatsoever.
                   This is the protocol we recommend.
  val_subject      argmax on a single designated validation fold ((s mod 10)+1).
                   The nested protocol the paper already reports, shown here with
                   tau and k unfrozen.
  kappa_agreement  LABEL-FREE. The 80 test repetitions split into two disjoint
                   blocks give two independent measurements of each stimulus. A
                   good configuration decodes both to the SAME candidate, whatever
                   that candidate is. Scored with Cohen's kappa so that a
                   degenerate map collapsing every query onto one candidate --
                   which would trivially "agree" -- is penalized.
  hubness_uniform  LABEL-FREE. Under a bijective protocol each candidate should be
                   retrieved exactly once; score = 1 - total-variation distance of
                   the retrieval counts from that ideal.
  plan_sharpness   LABEL-FREE. Mean row-max of the row-normalized Sinkhorn plan.
  mutual_nn        LABEL-FREE. Fraction of queries whose top candidate also ranks
                   that query first.
  paper_default    the published configuration applied blind, no selection at all.

Scope note: this isolates the TTA-hyperparameter leak only. The checkpoint is
whatever --source_run_dir provides, and the default run selected its epoch on the
test subject. Point --source_run_dir at a val-selected run to close both leaks.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import os

import numpy as np
import pandas as pd

from shared import (
    add_common_args,
    apply_tta_calibration,
    cosine_scores,
    encode_repetition_blocks,
    ensure_output_dir,
    evaluate_scores,
    fit_tta_calibration,
    load_subject_context,
    params_from_args,
    score_features,
    write_config,
)
from module.util import sinkhorn_normalize

LABEL_FREE = ["kappa_agreement", "hubness_uniform", "plan_sharpness", "mutual_nn"]


def config_grid(args):
    keys = ["saw_shrink", "sinkhorn_tau", "soft_procrustes_steps", "soft_procrustes_power"]
    values = [args.grid_shrink, args.grid_tau, args.grid_steps, args.grid_power]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _kappa(labels_a, labels_b, num_candidates):
    """Chance-corrected agreement between two independent decodes of the same stimuli."""
    observed = float(np.mean(labels_a == labels_b))
    counts_a = np.bincount(labels_a, minlength=num_candidates) / len(labels_a)
    counts_b = np.bincount(labels_b, minlength=num_candidates) / len(labels_b)
    expected = float(np.dot(counts_a, counts_b))
    if expected >= 1.0 - 1e-9:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def _label_free_scores(scores, plan, block_labels):
    num_queries, num_candidates = scores.shape
    top1 = scores.argmax(axis=1)

    counts = np.bincount(top1, minlength=num_candidates).astype(np.float64)
    ideal = num_queries / num_candidates
    hubness = 1.0 - 0.5 * np.abs(counts - ideal).sum() / num_queries

    rows = plan / np.clip(plan.sum(axis=1, keepdims=True), 1e-12, None)
    sharpness = float(rows.max(axis=1).mean())

    best_query_per_candidate = scores.argmax(axis=0)
    mutual = float(np.mean(best_query_per_candidate[top1] == np.arange(num_queries)))

    kappa = _kappa(block_labels[0], block_labels[1], num_candidates) if block_labels else float("nan")
    return {
        "kappa_agreement": kappa,
        "hubness_uniform": float(hubness),
        "plan_sharpness": sharpness,
        "mutual_nn": mutual,
    }


def run_subject(source_run_dir, subject_id, args, base_params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(source_run_dir, args, subject_id, average=False)
    half = int(args.total_repetitions) // 2
    (block_a, block_b), image_features, _ = encode_repetition_blocks(
        eval_args, modules, dataset, block_sizes=[half, half], seed=int(args.seed) * 1000 + int(subject_id)
    )
    # Full-precision queries: the deployed pipeline averages all repetitions.
    (full_queries,), _, _ = encode_repetition_blocks(
        eval_args, modules, dataset, block_sizes=[int(args.total_repetitions)], seed=int(args.seed)
    )

    rows = []
    for config in config_grid(args):
        params = dataclasses.replace(base_params, **config)
        calibration = fit_tta_calibration(full_queries, image_features, params)
        transformed = apply_tta_calibration(full_queries, calibration, alpha=1.0)
        scores = score_features(transformed, image_features, use_csls=params.use_csls, csls_k=params.csls_k)
        metrics = evaluate_scores(scores)

        plan = sinkhorn_normalize(
            scores, tau=params.sinkhorn_tau, num_iters=params.sinkhorn_iters, col_mass=params.sinkhorn_col_mass
        )
        block_labels = [
            cosine_scores(apply_tta_calibration(block, calibration, alpha=1.0), image_features).argmax(axis=1)
            for block in (block_a, block_b)
        ]
        rows.append(
            {
                "subject_id": int(subject_id),
                **config,
                "test_top1": round(float(metrics["top1_acc"]), 4),
                "test_top5": round(float(metrics["top5_acc"]), 4),
                **{k: round(float(v), 6) for k, v in _label_free_scores(scores, plan, block_labels).items()},
                "checkpoint_dir": checkpoint_dir,
            }
        )
    return rows


def select(df: pd.DataFrame, args, base_params):
    """Post-hoc: every protocol is an argmax over the same precomputed table."""
    config_cols = ["saw_shrink", "sinkhorn_tau", "soft_procrustes_steps", "soft_procrustes_power"]
    subjects = sorted(df["subject_id"].unique())
    records = []

    for subject_id in subjects:
        own = df[df["subject_id"] == subject_id].set_index(config_cols)
        others = df[df["subject_id"] != subject_id]
        val_id = (subject_id % max(subjects)) + 1
        val = df[df["subject_id"] == val_id]

        picks = {"oracle_test": own["test_top1"].idxmax()}
        picks["loso_fold_cv"] = others.groupby(config_cols)["test_top1"].mean().idxmax()
        if len(val):
            picks["val_subject"] = val.set_index(config_cols)["test_top1"].idxmax()
        for name in LABEL_FREE:
            if own[name].notna().any():
                picks[name] = own[name].idxmax()
        picks["paper_default"] = (
            base_params.saw_shrink,
            base_params.sinkhorn_tau,
            base_params.soft_procrustes_steps,
            base_params.soft_procrustes_power,
        )

        for protocol, key in picks.items():
            if key not in own.index:
                continue
            row = own.loc[key]
            records.append(
                {
                    "subject_id": int(subject_id),
                    "protocol": protocol,
                    **dict(zip(config_cols, key)),
                    "test_top1": float(row["test_top1"]),
                    "test_top5": float(row["test_top5"]),
                }
            )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--grid_shrink", nargs="+", type=float, default=[0.88, 0.92, 0.94, 0.96, 0.98])
    parser.add_argument("--grid_tau", nargs="+", type=float, default=[0.04, 0.07, 0.10, 0.13, 0.16])
    parser.add_argument("--grid_steps", nargs="+", type=int, default=[16, 20])
    parser.add_argument("--grid_power", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.4])
    parser.add_argument("--total_repetitions", type=int, default=80)
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    base_params = params_from_args(args)
    default_cell = (base_params.saw_shrink, base_params.sinkhorn_tau,
                    base_params.soft_procrustes_steps, base_params.soft_procrustes_power)
    if default_cell not in {tuple(c.values()) for c in config_grid(args)}:
        raise SystemExit(
            f"Published config {default_cell} is not in the grid, so the paper_default "
            "protocol would be silently dropped. Add it to the grid values."
        )
    output_dir = ensure_output_dir(args, "honest_tta_selection")
    write_config(output_dir, args, base_params)
    print(f"[honest_tta_selection] {len(config_grid(args))} configs x {len(args.subjects)} subjects", flush=True)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[honest_tta_selection] subject={int(subject_id):02d}", flush=True)
        all_rows.extend(run_subject(source_run_dir, subject_id, args, base_params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "grid_scores.csv"), index=False)

    grid_df = pd.DataFrame(all_rows)
    selected = select(grid_df, args, base_params)
    selected.to_csv(os.path.join(output_dir, "selected_per_subject.csv"), index=False)

    summary = (
        selected.groupby("protocol")
        .agg(top1_mean=("test_top1", "mean"), top1_std=("test_top1", "std"),
             top5_mean=("test_top5", "mean"), n_subjects=("test_top1", "size"))
        .sort_values("top1_mean", ascending=False)
    )
    oracle = float(summary.loc["oracle_test", "top1_mean"]) if "oracle_test" in summary.index else np.nan
    summary["gap_to_oracle"] = (summary["top1_mean"] - oracle).round(2)
    summary.round(2).to_csv(os.path.join(output_dir, "protocol_summary.csv"))

    print("\n=== TTA hyperparameter selection protocols (test top-1, LOSO mean) ===")
    print(summary.round(2).to_string())
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
