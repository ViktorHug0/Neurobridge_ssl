"""Compare the matched control and stochastic full-data twin arms."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ARMS = {
    "control": "joint_b05_control",
    "stochastic": "stoch_spec05_cdrop10_keep75_b05",
}
METRICS = (
    "member_a_top1",
    "member_b_top1",
    "pair_top1",
    "pair_top5",
    "score_correlation",
    "oracle_top1",
    "gain_over_best",
)


def load_arm(root: Path, arm: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    arm_root = root / arm / "seed3300_3301"
    for path in sorted(arm_root.glob("*-sub-??/result.csv")):
        subject = int(path.parent.name.rsplit("-", 1)[1])
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        if subject in rows:
            raise RuntimeError(f"duplicate completed subject {subject} under {arm_root}")
        rows[subject] = {metric: float(row[metric]) for metric in METRICS}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "results/things_eeg/decorrelated_models/stochastic_full_data_wave"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    control = load_arm(args.result_root, ARMS["control"])
    stochastic = load_arm(args.result_root, ARMS["stochastic"])
    common = sorted(set(control) & set(stochastic))
    if args.require_complete and common != list(range(1, 11)):
        raise SystemExit(f"waiting for matched folds; complete subjects={common}")
    if not common:
        print("matched folds=0/10")
        return

    rows = []
    for subject in common:
        row: dict[str, object] = {"subject": subject}
        for metric in METRICS:
            row[f"control_{metric}"] = control[subject][metric]
            row[f"stochastic_{metric}"] = stochastic[subject][metric]
            row[f"delta_{metric}"] = stochastic[subject][metric] - control[subject][metric]
        rows.append(row)
    average = {
        "subject": "Average",
        **{
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in rows[0]
            if key != "subject"
        },
    }
    output = args.result_root / "stochastic_vs_control.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [average])
    print(
        f"matched folds={len(common)}/10 "
        f"control_pair={average['control_pair_top1']:.2f} "
        f"stochastic_pair={average['stochastic_pair_top1']:.2f} "
        f"delta={average['delta_pair_top1']:+.2f}"
    )


if __name__ == "__main__":
    main()
