#!/usr/bin/env python3
"""Experiment B: does SAGE-TTA need a bijection, or just known marginals?

Rekw asks whether adaptation requires a one-to-one EEG-image correspondence. It
does not: it requires the candidate marginals to be known, and a bijection is
merely the case "one query per candidate". Here we build the balanced R-to-1
protocol -- each of the 200 menu items is viewed R times -- by partitioning the
80 test repetitions into R disjoint blocks of 80/R repetitions and stacking them
into R*200 queries against the same 200 candidates.

Two controls make the claim falsifiable:

  single_block : one block only (200 queries vs 200 candidates, square) at the
                 SAME per-query trial averaging. Isolates "more queries per
                 candidate" from "less averaging per query".
  col_mass     : the Sinkhorn column target, swept at 1.0 and R. These are
                 provably equivalent here (see sinkhorn_normalize: col_mass is a
                 global rescale, and the Procrustes SVD is scale-invariant), and
                 we record both to show it empirically rather than assert it.

That equivalence is the point. What the pipeline actually assumes is that every
candidate carries EQUAL mass, not that it carries mass exactly one -- so the
bijective case and the balanced R-to-1 case are the same problem, and no code
change is needed to move between them. If all_blocks matches or beats
single_block at equal SNR, the method generalizes beyond the bijective protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
import pandas as pd

from shared import (
    add_common_args,
    aggregate_results,
    cosine_scores,
    encode_repetition_blocks,
    ensure_output_dir,
    evaluate_full_tta,
    evaluate_saw_csls,
    evaluate_scores,
    load_subject_context,
    params_from_args,
    write_config,
)
from module.util import fit_soft_assignment_procrustes, sinkhorn_normalize


def _append(rows, **kwargs):
    metrics = kwargs.pop("metrics")
    rows.append({**kwargs, "top1_acc": round(float(metrics["top1_acc"]), 4), "top5_acc": round(float(metrics["top5_acc"]), 4)})


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(source_run_dir, args, subject_id, average=False)
    rows = []

    for seed in args.seeds:
        for num_blocks in args.block_counts:
            num_blocks = int(num_blocks)
            block_size = int(args.total_repetitions) // num_blocks
            if block_size < 1:
                continue
            query_blocks, image_features, available = encode_repetition_blocks(
                eval_args,
                modules,
                dataset,
                block_sizes=[block_size] * num_blocks,
                seed=int(seed) * 100000 + int(subject_id) * 1000 + num_blocks,
            )
            num_objects = image_features.shape[0]
            common = dict(
                subject_id=int(subject_id),
                seed=int(seed),
                num_blocks=num_blocks,
                reps_per_query=block_size,
                available_repetitions=int(available),
            )

            # --- control: one block, square 200-vs-200 at the same per-query SNR ---
            single = query_blocks[0]
            single_targets = np.arange(num_objects, dtype=np.int64)
            _append(rows, **common, scope="single_block", col_mass=1.0, method="plain_cosine",
                    metrics=evaluate_scores(cosine_scores(single, image_features), single_targets))
            _append(rows, **common, scope="single_block", col_mass=1.0, method="saw_csls",
                    metrics=evaluate_saw_csls(single, image_features, params, single_targets))
            _append(rows, **common, scope="single_block", col_mass=1.0, method="full_tta",
                    metrics=evaluate_full_tta(single, image_features, params, single_targets))

            # --- R-to-1: all blocks stacked, R queries per candidate ---
            stacked = np.concatenate(query_blocks, axis=0)
            stacked_targets = np.tile(np.arange(num_objects, dtype=np.int64), num_blocks)
            _append(rows, **common, scope="all_blocks", col_mass=1.0, method="plain_cosine",
                    metrics=evaluate_scores(cosine_scores(stacked, image_features), stacked_targets))
            _append(rows, **common, scope="all_blocks", col_mass=1.0, method="saw_csls",
                    metrics=evaluate_saw_csls(stacked, image_features, params, stacked_targets))
            for col_mass in sorted({1.0, float(num_blocks)}):
                variant = dataclasses.replace(params, sinkhorn_col_mass=col_mass)
                _append(rows, **common, scope="all_blocks", col_mass=col_mass, method="full_tta",
                        metrics=evaluate_full_tta(stacked, image_features, variant, stacked_targets))

    for row in rows:
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def self_check():
    """Marginal invariants of the generalized Sinkhorn step."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(40, 10)).astype(np.float32)

    plan = sinkhorn_normalize(scores, tau=0.1, num_iters=200, col_mass=4.0)
    assert np.allclose(plan.sum(axis=1), 1.0, atol=1e-3), plan.sum(axis=1)[:5]
    assert np.allclose(plan.sum(axis=0), 4.0, atol=1e-3), plan.sum(axis=0)
    assert abs(plan.sum() - 40.0) < 1e-2, plan.sum()

    square = rng.normal(size=(16, 16)).astype(np.float32)
    default = sinkhorn_normalize(square, tau=0.1, num_iters=200)
    explicit = sinkhorn_normalize(square, tau=0.1, num_iters=200, col_mass=1.0)
    assert np.array_equal(default, explicit), "col_mass=1.0 must reproduce the shipped behaviour"
    assert np.allclose(default.sum(axis=0), 1.0, atol=1e-3)

    # col_mass is a pure global rescale, so it cannot change the orthogonal map.
    # This is why the bijective and balanced R-to-1 protocols are one problem.
    base = sinkhorn_normalize(scores, tau=0.1, num_iters=200, col_mass=1.0)
    assert np.allclose(plan, 4.0 * base, rtol=1e-4, atol=1e-6), "col_mass must act as a global scale"

    # ...and a global scale cannot move the Procrustes solution.
    queries = rng.normal(size=(40, 8)).astype(np.float32)
    candidates = rng.normal(size=(10, 8)).astype(np.float32)
    assert np.allclose(
        fit_soft_assignment_procrustes(queries, candidates, plan),
        fit_soft_assignment_procrustes(queries, candidates, base),
        atol=1e-5,
    ), "the orthogonal map must be invariant to the Sinkhorn column target"
    print("self_check OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--block_counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--total_repetitions", type=int, default=80)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    parser.add_argument("--self_check", action="store_true", help="Run marginal invariants and exit.")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "many_to_one")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[many_to_one] subject={int(subject_id):02d}", flush=True)
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["num_blocks", "reps_per_query", "scope", "method", "col_mass"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    print("\n=== Many-to-one (R queries per candidate) ===")
    print(average_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
