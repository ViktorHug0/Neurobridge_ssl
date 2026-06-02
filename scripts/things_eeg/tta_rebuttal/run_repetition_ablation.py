#!/usr/bin/env python3
"""TTA robustness as test repetitions are reduced.

For each held-out subject, rebuild the 200 test queries by averaging only K EEG
repetitions per image, then run the same 200-way retrieval/TTA pipeline. This
tests whether SAGE-TTA remains useful when the deployment block has less trial
averaging than the standard THINGS-EEG-2 80-repetition test protocol.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from shared import (
    add_common_args,
    aggregate_results,
    encode_repetition_limited_features,
    ensure_output_dir,
    evaluate_full_tta,
    evaluate_plain,
    evaluate_saw_csls,
    load_subject_context,
    params_from_args,
    write_config,
)


def _repetition_counts(args, max_repetitions):
    if args.repetition_counts:
        counts = [int(x) for x in args.repetition_counts]
    else:
        counts = list(range(args.max_repetitions, args.min_repetitions - 1, -args.repetition_step))
    return [count for count in counts if 1 <= count <= max_repetitions]


def _append(rows, subject_id, seed, repetition_count, selection_mode, method, metrics, available_repetitions):
    rows.append(
        {
            "subject_id": int(subject_id),
            "seed": int(seed),
            "repetition_count": int(repetition_count),
            "available_repetitions": int(available_repetitions),
            "selection_mode": selection_mode,
            "method": method,
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
        }
    )


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(source_run_dir, args, subject_id, average=False)
    available_repetitions = int(dataset.num_repetitions)
    counts = _repetition_counts(args, available_repetitions)
    if not counts:
        raise ValueError(f"No valid repetition counts for available_repetitions={available_repetitions}.")

    rows = []
    seeds = args.seeds if args.repetition_selection != "first" else [args.seeds[0]]
    for repetition_count in counts:
        for seed in seeds:
            query_features, image_features, _ = encode_repetition_limited_features(
                eval_args,
                modules,
                dataset,
                repetition_count=repetition_count,
                seed=int(seed) * 100000 + int(subject_id) * 1000 + int(repetition_count),
                selection_mode=args.repetition_selection,
            )
            _append(
                rows,
                subject_id,
                seed,
                repetition_count,
                args.repetition_selection,
                "plain_cosine",
                evaluate_plain(query_features, image_features),
                available_repetitions,
            )
            _append(
                rows,
                subject_id,
                seed,
                repetition_count,
                args.repetition_selection,
                "saw_csls",
                evaluate_saw_csls(query_features, image_features, params),
                available_repetitions,
            )
            _append(
                rows,
                subject_id,
                seed,
                repetition_count,
                args.repetition_selection,
                "full_tta",
                evaluate_full_tta(query_features, image_features, params),
                available_repetitions,
            )

    for row in rows:
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--repetition_counts",
        nargs="+",
        type=int,
        default=None,
        help="Explicit repetition counts. Overrides min/max/step.",
    )
    parser.add_argument("--max_repetitions", type=int, default=80)
    parser.add_argument("--min_repetitions", type=int, default=10)
    parser.add_argument("--repetition_step", type=int, default=10)
    parser.add_argument(
        "--repetition_selection",
        choices=["first", "random_shared", "random_per_sample"],
        default="random_per_sample",
        help=(
            "How to choose K repetitions: first K, one random subset shared by all images, "
            "or a random subset per image."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "repetition_ablation")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[repetition_ablation] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["selection_mode", "method", "repetition_count"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    print("\n=== Repetition Ablation Average Results ===")
    print(average_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
