"""Aggregate the fixed eight-member balanced 7-of-9 subject-bag ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ensemble_experiments.balanced_subject_bagging import NUM_BAGS, manifest
from ensemble_experiments.retrieval_fusion import (
    FUSION_METHODS,
    cosine_scores,
    fuse_scores,
    mean_pairwise_score_correlation,
    oracle_top1,
    retrieval_accuracies,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "results/things_eeg/subject_cohort_bagging/testselected_balanced7x8"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "bag_manifest.json").write_text(json.dumps(manifest(), indent=2) + "\n")

    missing: list[str] = []
    rows: list[dict[str, object]] = []
    for target in range(1, 11):
        matrices = []
        labels = None
        member_top1 = []
        for member in range(NUM_BAGS):
            dump = root / f"target-{target:02d}" / f"member-{member}" / "embeddings.npz"
            if not dump.is_file():
                missing.append(str(dump))
                continue
            with np.load(dump) as data:
                current_labels = np.stack((data["object"], data["image_idx"]), axis=1)
                if labels is None:
                    labels = current_labels
                elif not np.array_equal(labels, current_labels):
                    raise RuntimeError(f"candidate ordering mismatch in {dump}")
                scores = cosine_scores(data["eeg"], data["image"])
            matrices.append(scores)
            member_top1.append(retrieval_accuracies(scores)[0])
        if len(matrices) != NUM_BAGS:
            continue

        scores = np.stack(matrices)
        row: dict[str, object] = {
            "target": target,
            "mean_member_top1": float(np.mean(member_top1)),
            "best_member_top1": float(np.max(member_top1)),
            "oracle_top1": oracle_top1(scores),
            "mean_pair_score_correlation": mean_pairwise_score_correlation(scores),
        }
        row.update(
            {
                f"member_{member}_top1": top1
                for member, top1 in enumerate(member_top1)
            }
        )
        primary_scores = None
        for method in FUSION_METHODS:
            fused = fuse_scores(scores, method)
            if method == "row_z":
                primary_scores = fused
            top1, top5 = retrieval_accuracies(fused)
            row[f"{method}_top1"] = top1
            row[f"{method}_top5"] = top5
        row["row_z_gain_over_mean"] = (
            float(row["row_z_top1"]) - float(row["mean_member_top1"])
        )
        row["row_z_gain_over_best"] = (
            float(row["row_z_top1"]) - float(row["best_member_top1"])
        )
        fused_dir = root / "fused_scores"
        fused_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            fused_dir / f"balanced7x8-sub{target:02d}.npz",
            scores=primary_scores,
            object=labels[:, 0],
            image_idx=labels[:, 1],
            fusion_method="row_z",
        )
        rows.append(row)

    if missing and args.require_complete:
        raise SystemExit(f"waiting for {len(missing)} member dumps")
    if not rows:
        print(f"complete targets=0/10; missing dumps={len(missing)}")
        return

    partial = root / "balanced7x8_ensemble_partial.csv"
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"complete targets={len(rows)}/10; missing dumps={len(missing)}")

    if len(rows) == 10:
        average = {
            "target": "Average",
            **{
                key: float(np.mean([float(row[key]) for row in rows]))
                for key in rows[0]
                if key != "target"
            },
        }
        output = root / "balanced7x8_ensemble.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows + [average])
        print(json.dumps(average, indent=2))


if __name__ == "__main__":
    main()
