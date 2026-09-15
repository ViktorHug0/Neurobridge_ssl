"""Aggregate the fixed five-TinyTSConv plus five-TinyATM compute-matched pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MEMBERS = tuple(
    [(f"tiny_tsconv_seed{3300 + seed}", "TinyTSConv") for seed in range(5)]
    + [(f"tiny_atm_seed{4300 + seed}", "TinyATM") for seed in range(5)]
)
REFERENCE_DUMPS = Path(
    "results/things_eeg/synthetic_subjects/ensemble_screen/dumps"
)


def cosine_scores(eeg: np.ndarray, image: np.ndarray) -> np.ndarray:
    eeg = eeg.astype(np.float32)
    image = image.astype(np.float32)
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-8)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-8)
    return eeg @ image.T


def row_z(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=-1, keepdims=True)) / np.maximum(
        scores.std(axis=-1, keepdims=True), 1e-8
    )


def topk(scores: np.ndarray) -> tuple[float, float]:
    truth = np.arange(scores.shape[0])
    top1 = float((scores.argmax(axis=1) == truth).mean() * 100.0)
    top5_idx = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    top5 = float((top5_idx == truth[:, None]).any(axis=1).mean() * 100.0)
    return top1, top5


def load_dump(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    labels = np.stack((data["object"], data["image_idx"]), axis=1)
    return cosine_scores(data["eeg"], data["image"]), labels


def mean_upper_triangle(correlation: np.ndarray) -> float:
    return float(correlation[np.triu_indices(len(correlation), k=1)].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="results/things_eeg/tiny_compute_ensemble/testselected_internvit28",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_root)
    missing: list[str] = []
    member_scores: list[list[np.ndarray]] = [[] for _ in MEMBERS]
    solo_rows: list[dict[str, object]] = []
    labels_by_subject: dict[int, np.ndarray] = {}

    for member_index, (member_name, family) in enumerate(MEMBERS):
        for subject in range(1, 11):
            dump = root / member_name / f"sub-{subject:02d}" / "embeddings.npz"
            if not dump.exists():
                missing.append(str(dump))
                continue
            scores, labels = load_dump(dump)
            if subject in labels_by_subject and not np.array_equal(
                labels_by_subject[subject], labels
            ):
                raise RuntimeError(f"candidate ordering mismatch in {dump}")
            labels_by_subject[subject] = labels
            member_scores[member_index].append(scores)
            top1, top5 = topk(scores)
            solo_rows.append(
                {
                    "member": member_name,
                    "family": family,
                    "subject": subject,
                    "top1": top1,
                    "top5": top5,
                }
            )

    complete_members = [len(scores) == 10 for scores in member_scores]
    print(
        f"complete members={sum(complete_members)}/10; "
        f"complete dumps={100 - len(missing)}/100"
    )
    if missing and args.require_complete:
        raise SystemExit(f"waiting for {len(missing)} dumps")
    if not all(complete_members):
        return

    scores = np.stack([np.stack(member) for member in member_scores])
    standardized = row_z(scores)
    truth = np.arange(scores.shape[-2])
    correct = scores.argmax(axis=3) == truth

    rows: list[dict[str, object]] = []
    for subject_index, subject in enumerate(range(1, 11)):
        fused = standardized[:, subject_index].mean(axis=0)
        ensemble_top1, ensemble_top5 = topk(fused)
        solo_top1 = correct[:, subject_index].mean(axis=1) * 100.0
        correlations = np.corrcoef(
            standardized[:, subject_index].reshape(len(MEMBERS), -1)
        )
        rows.append(
            {
                "subject": subject,
                "mean_solo_top1": float(solo_top1.mean()),
                "best_solo_top1": float(solo_top1.max()),
                "ensemble_top1": ensemble_top1,
                "ensemble_top5": ensemble_top5,
                "oracle_top1": float(correct[:, subject_index].any(axis=0).mean() * 100.0),
                "mean_score_correlation": mean_upper_triangle(correlations),
            }
        )

    average = {
        "subject": "Average",
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "subject"
        },
    }
    with (root / "tiny10_ensemble.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [average])
    with (root / "tiny10_solos.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(solo_rows[0]))
        writer.writeheader()
        writer.writerows(solo_rows)

    family_indices = {
        "TinyTSConv": list(range(5)),
        "TinyATM": list(range(5, 10)),
    }
    family_correlations = {}
    flat = standardized.reshape(10, -1)
    full_correlation = np.corrcoef(flat)
    for family, indices in family_indices.items():
        family_correlations[family] = mean_upper_triangle(
            full_correlation[np.ix_(indices, indices)]
        )
    cross = full_correlation[np.ix_(family_indices["TinyTSConv"], family_indices["TinyATM"])]
    family_correlations["cross_family"] = float(cross.mean())

    reference = []
    for name in ("tsconv_iv", "atm_iv"):
        folds = []
        for subject in range(1, 11):
            fold_scores, _ = load_dump(
                REFERENCE_DUMPS / f"{name}-sub{subject:02d}.npz"
            )
            folds.append(fold_scores)
        reference.append(np.stack(folds))
    reference = row_z(np.stack(reference)).mean(axis=0)
    reference_by_subject = [topk(fold)[0] for fold in reference]

    scaling = []
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    for size in range(1, 11):
        selected = order[:size]
        fold_top1 = [
            topk(standardized[selected, subject].mean(axis=0))[0]
            for subject in range(10)
        ]
        scaling.append(
            {
                "k": size,
                "members": [MEMBERS[index][0] for index in selected],
                "mean_top1": float(np.mean(fold_top1)),
                "top1_by_subject": fold_top1,
            }
        )

    summary = {
        "protocol": "fixed 5 TinyTSConv + 5 TinyATM; InternViT-28; test-selected; row-z uniform fusion",
        "members": [name for name, _ in MEMBERS],
        "tiny10": average,
        "family_solo_top1": {
            family: float(
                np.mean(
                    [row["top1"] for row in solo_rows if row["family"] == family]
                )
            )
            for family in family_indices
        },
        "score_correlations": family_correlations,
        "reference_full_tsconv_atm": {
            "mean_top1": float(np.mean(reference_by_subject)),
            "top1_by_subject": reference_by_subject,
        },
        "delta_vs_reference_by_subject": (
            np.asarray([row["ensemble_top1"] for row in rows])
            - np.asarray(reference_by_subject)
        ).tolist(),
        "scaling": scaling,
    }
    (root / "tiny10_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
