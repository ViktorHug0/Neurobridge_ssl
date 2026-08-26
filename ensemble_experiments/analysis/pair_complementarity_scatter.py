"""Relate pairwise prediction diversity to row-z ensemble gain."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


SUBJECTS = tuple(range(1, 11))
TRUTH = np.arange(200)


def load_arm(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    for subject in SUBJECTS:
        path = root / f"{name}-sub{subject:02d}.npz"
        data = np.load(path)
        eeg = data["eeg"].astype(np.float32)
        image = data["image"].astype(np.float32)
        eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
        image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
        raw = eeg @ image.T
        scores.append(
            (raw - raw.mean(1, keepdims=True))
            / np.maximum(raw.std(1, keepdims=True), 1e-6)
        )
        labels.append(np.stack([data["object"], data["image_idx"]], axis=1))
    return np.stack(scores), np.stack(labels)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    parser.add_argument(
        "--roster-json",
        default="ensemble_experiments/analysis/target_matrix_test_extended_z.json",
    )
    parser.add_argument("--add", nargs="*", default=[])
    parser.add_argument(
        "--output-prefix",
        default="ensemble_experiments/analysis/pair_complementarity_testselected",
    )
    args = parser.parse_args()

    roster = json.loads(Path(args.roster_json).read_text())["roster"]
    roster = list(dict.fromkeys(roster + args.add))
    root = Path(args.dump_root)
    missing = [
        str(root / f"{name}-sub{subject:02d}.npz")
        for name in roster
        for subject in SUBJECTS
        if not (root / f"{name}-sub{subject:02d}.npz").exists()
    ]
    if missing:
        raise FileNotFoundError("Missing score dumps:\n" + "\n".join(missing))

    arms, reference_labels = {}, None
    for name in roster:
        scores, labels = load_arm(root, name)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"Query order differs for {name}")
        predictions = scores.argmax(2)
        correct = predictions == TRUTH
        wrong_masked = scores.copy()
        wrong_masked[:, TRUTH, TRUTH] = -np.inf
        margin = scores[:, TRUTH, TRUTH] - wrong_masked.max(2)
        arms[name] = {
            "scores": scores,
            "predictions": predictions,
            "correct": correct,
            "margin": margin,
            "top1": float(correct.mean() * 100.0),
        }

    rows = []
    for first, second in itertools.combinations(roster, 2):
        a, b = arms[first], arms[second]
        fused_correct = ((a["scores"] + b["scores"]).argmax(2) == TRUTH)
        pair_top1 = float(fused_correct.mean() * 100.0)
        best_solo = max(a["top1"], b["top1"])
        mean_solo = (a["top1"] + b["top1"]) / 2.0
        both_wrong = ~a["correct"] & ~b["correct"]
        wrong_winner_agreement = (
            float((a["predictions"][both_wrong] == b["predictions"][both_wrong]).mean())
            if both_wrong.any()
            else float("nan")
        )
        oracle_top1 = float((a["correct"] | b["correct"]).mean() * 100.0)
        rows.append({
            "first": first,
            "second": second,
            "first_top1": a["top1"],
            "second_top1": b["top1"],
            "weaker_top1": min(a["top1"], b["top1"]),
            "best_solo_top1": best_solo,
            "mean_solo_top1": mean_solo,
            "pair_top1": pair_top1,
            "gain_over_best": pair_top1 - best_solo,
            "gain_over_mean": pair_top1 - mean_solo,
            "relative_error_reduction_over_best": (
                (pair_top1 - best_solo) / (100.0 - best_solo)
            ),
            "score_correlation": correlation(a["scores"], b["scores"]),
            "margin_correlation": correlation(a["margin"], b["margin"]),
            "correctness_correlation": correlation(
                a["correct"].astype(np.float32), b["correct"].astype(np.float32)
            ),
            "prediction_disagreement": float(
                (a["predictions"] != b["predictions"]).mean()
            ),
            "double_fault": float(both_wrong.mean()),
            "wrong_winner_agreement": wrong_winner_agreement,
            "oracle_top1": oracle_top1,
            "oracle_headroom_over_best": oracle_top1 - best_solo,
            "oracle_headroom_realized": (
                (pair_top1 - best_solo) / (oracle_top1 - best_solo)
                if oracle_top1 > best_solo
                else float("nan")
            ),
        })

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        "score_correlation",
        "margin_correlation",
        "correctness_correlation",
        "prediction_disagreement",
        "double_fault",
        "wrong_winner_agreement",
        "oracle_headroom_over_best",
        "weaker_top1",
        "best_solo_top1",
    ]
    gain_best = np.array([row["gain_over_best"] for row in rows])
    gain_mean = np.array([row["gain_over_mean"] for row in rows])
    associations = {}
    associations_mean = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows])
        associations[metric] = {
            "pearson_r": float(pearsonr(values, gain_best).statistic),
            "spearman_rho": float(spearmanr(values, gain_best).statistic),
        }
        associations_mean[metric] = {
            "pearson_r": float(pearsonr(values, gain_mean).statistic),
            "spearman_rho": float(spearmanr(values, gain_mean).statistic),
        }
    summary = {
        "protocol": "test-selected; per-query row-z; uniform two-member mean",
        "roster": roster,
        "roster_size": len(roster),
        "pairs": len(rows),
        "metric_association_with_gain_over_best": associations,
        "metric_association_with_gain_over_mean": associations_mean,
        "best_pairs_by_gain_over_best": sorted(
            rows, key=lambda row: row["gain_over_best"], reverse=True
        )[:15],
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    x = np.array([row["score_correlation"] for row in rows])
    color = np.array([row["weaker_top1"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True,
                             layout="constrained")
    outcomes = [("gain_over_best", "Gain over stronger member (pp)"),
                ("gain_over_mean", "Gain over pair mean (pp)")]
    scatter = None
    for axis, (key, ylabel) in zip(axes, outcomes):
        y = np.array([row[key] for row in rows])
        scatter = axis.scatter(x, y, c=color, cmap="viridis", s=28, alpha=0.72,
                               linewidths=0)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axis.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.2)
        axis.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Pairwise row-z score correlation")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)
    fig.colorbar(scatter, ax=axes, label="Weaker member solo top-1 (%)", pad=0.02)
    fig.suptitle(
        f"Pair complementarity across {len(roster)} test-selected models ({len(rows)} pairs)"
    )
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))

    mean_color = np.array([row["mean_solo_top1"] for row in rows])
    mean_fig, mean_axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True,
                                       layout="constrained")
    mean_scatter = None
    for axis, (key, ylabel) in zip(mean_axes, outcomes):
        y = np.array([row[key] for row in rows])
        mean_scatter = axis.scatter(
            x, y, c=mean_color, cmap="viridis", s=28, alpha=0.72, linewidths=0
        )
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axis.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.2)
        axis.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Pairwise row-z score correlation")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)
    mean_fig.colorbar(
        mean_scatter, ax=mean_axes, label="Mean solo top-1 of pair (%)", pad=0.02
    )
    mean_fig.suptitle(
        f"Pair complementarity across {len(roster)} test-selected models ({len(rows)} pairs)"
    )
    mean_color_path = prefix.with_name(prefix.name + "_color_pair_mean.png")
    mean_fig.savefig(mean_color_path, dpi=220)

    oracle = np.array([row["oracle_headroom_over_best"] for row in rows])
    predictor_fig, predictor_axis = plt.subplots(figsize=(7.2, 5.8),
                                                  layout="constrained")
    predictor_scatter = predictor_axis.scatter(
        oracle, gain_best, c=x, cmap="magma_r", s=30, alpha=0.72, linewidths=0
    )
    limit = max(float(oracle.max()), float(gain_best.max())) * 1.03
    predictor_axis.plot([0, limit], [0, limit], color="0.45", linestyle="--",
                        linewidth=1.0, label="100% of oracle headroom realized")
    predictor_axis.axhline(0, color="0.6", linewidth=0.8)
    predictor_axis.set_xlim(left=0)
    predictor_axis.set_xlabel("Oracle headroom over stronger member (pp)")
    predictor_axis.set_ylabel("Observed row-z ensemble gain (pp)")
    predictor_axis.grid(alpha=0.18)
    predictor_axis.legend(loc="upper left", fontsize=8)
    predictor_fig.colorbar(
        predictor_scatter, ax=predictor_axis,
        label="Pairwise row-z score correlation",
    )
    predictor_axis.set_title("Available complementarity versus realized gain")
    predictor_path = prefix.with_name(prefix.name + "_oracle_headroom.png")
    predictor_fig.savefig(predictor_path, dpi=220)

    print(f"Wrote {csv_path}, {prefix.with_suffix('.json')}, and plots")
    print("Associations with gain over best:")
    print(json.dumps(associations, indent=2))
    print("Associations with gain over mean:")
    print(json.dumps(associations_mean, indent=2))


if __name__ == "__main__":
    main()
