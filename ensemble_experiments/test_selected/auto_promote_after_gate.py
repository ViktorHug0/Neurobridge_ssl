"""Select one Gate-2 promotion on subjects 1/3/5 and optionally submit it.

This is a resource handoff, not a final scientific claim. It uses only the
predeclared gate folds and starts one GPU task after the CPU selector exits, so
the concurrently running ViT-H completion keeps total usage at two GPUs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

import final_ensemble_audit as audit


GATE_SUBJECTS = (1, 3, 5)
NEW_ARMS = (
    "eva35_group_e100",
    "vith10_group_e100",
    "atm_iv_group_e100",
    "multibackbone5_group_e100",
)
GATE2_ARMS = ("atm_iv_group_e100", "multibackbone5_group_e100")
FULL_TASK = {"atm_iv_group_e100": 0, "multibackbone5_group_e100": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--after-job", default=None, help="dependency for submitted GPU job")
    parser.add_argument(
        "--queue-followups",
        action="store_true",
        help="queue the two Gate-3 arms behind the GPU slots released by the primary jobs",
    )
    parser.add_argument(
        "--vith-dump-job",
        default=None,
        help="job after which the ATM--ViT-H gate may use the ViT-H GPU slot",
    )
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    parser.add_argument(
        "--output",
        default="results/things_eeg/ensemble50_testselected/auto_promotion.json",
    )
    args = parser.parse_args()

    names = tuple(dict.fromkeys(audit.DEFAULT_POOL + NEW_ARMS))
    root = Path(args.dump_root)
    raw = {
        name: [audit.load_matrix(root / f"{name}-sub{subject:02d}.npz") for subject in GATE_SUBJECTS]
        for name in names
    }
    z = {name: [audit.row_z(matrix) for matrix in raw[name]] for name in names}
    transformed = {
        "row_z": z,
        "row_rank": {name: [audit.row_rank(matrix) for matrix in raw[name]] for name in names},
        "row_pow05": {
            name: [np.sign(matrix) * np.abs(matrix) ** 0.5 for matrix in z[name]]
            for name in names
        },
        "row_pow125": {
            name: [np.sign(matrix) * np.abs(matrix) ** 1.25 for matrix in z[name]]
            for name in names
        },
        "row_softmax4": {
            name: [
                np.exp(matrix / 4.0 - (matrix / 4.0).max(1, keepdims=True))
                / np.exp(matrix / 4.0 - (matrix / 4.0).max(1, keepdims=True)).sum(
                    1, keepdims=True
                )
                for matrix in z[name]
            ]
            for name in names
        },
    }
    methods = (
        "raw",
        "row_z",
        "row_rank",
        "row_pow05",
        "row_pow125",
        "row_softmax4",
        "gap025",
    )
    rows: list[dict[str, object]] = []
    for size in range(2, 6):
        for combo in itertools.combinations(names, size):
            for method in methods:
                candidate = audit.Candidate(method, combo)
                scores = [
                    audit.top1(audit.fuse(raw, transformed, candidate, fold))
                    for fold in range(len(GATE_SUBJECTS))
                ]
                rows.append(
                    {
                        "mean": float(np.mean(scores)),
                        "scores": scores,
                        "method": method,
                        "members": combo,
                    }
                )

    baseline = json.loads(
        Path("ensemble_experiments/test_selected/frozen_fusion_baseline.json").read_text()
    )["selection_fold_mean_top1"]
    best_overall = max(rows, key=lambda row: row["mean"])
    evidence: dict[str, dict[str, object]] = {}
    for arm in GATE2_ARMS:
        best_with = max((row for row in rows if arm in row["members"]), key=lambda row: row["mean"])
        best_without = max((row for row in rows if arm not in row["members"]), key=lambda row: row["mean"])
        evidence[arm] = {
            "best_with": best_with,
            "best_without_mean": best_without["mean"],
            "marginal_vs_best_without": float(best_with["mean"] - best_without["mean"]),
            "gain_vs_frozen_gate_baseline": float(best_with["mean"] - baseline),
        }

    promotable = [
        arm
        for arm in GATE2_ARMS
        if evidence[arm]["marginal_vs_best_without"] >= (1.0 / 6.0 - 1e-6)
        and evidence[arm]["gain_vs_frozen_gate_baseline"] >= -1e-6
    ]
    if promotable:
        selected = max(
            promotable,
            key=lambda arm: (
                evidence[arm]["marginal_vs_best_without"],
                evidence[arm]["gain_vs_frozen_gate_baseline"],
            ),
        )
        action = "promote_gate2"
    else:
        selected = "bigg27_group_e100"
        action = "launch_bigg_gate"

    report = {
        "gate_subjects": GATE_SUBJECTS,
        "frozen_gate_baseline": baseline,
        "best_overall": best_overall,
        "gate2_evidence": evidence,
        "action": action,
        "selected": selected,
        "submitted_job": None,
        "followup_jobs": {},
    }

    if args.submit:
        dependency = [f"--dependency=afterok:{args.after_job}"] if args.after_job else []
        if action == "promote_gate2":
            command = [
                "sbatch",
                *dependency,
                f"--array={FULL_TASK[selected]}",
                "ensemble_experiments/test_selected/second_gate_full.sbatch",
            ]
        else:
            command = [
                "sbatch",
                *dependency,
                "--array=0",
                "ensemble_experiments/test_selected/third_target_gate.sbatch",
            ]
        output = subprocess.check_output(command, text=True).strip()
        report["submitted_job"] = output

        # Keep useful training flowing after either current GPU releases, while
        # preserving the hard two-GPU limit.  If ATM--InternViT is promoted,
        # BigG waits for that seven-fold completion; otherwise the primary job
        # is already BigG.  ATM--ViT-H waits for the separate ViT-H export job,
        # which itself starts only after the promoted ViT-H completion.
        if args.queue_followups:
            primary_job = output.rsplit(" ", 1)[-1]
            if action == "promote_gate2":
                bigg_output = subprocess.check_output(
                    [
                        "sbatch",
                        f"--dependency=afterany:{primary_job}",
                        "--array=0",
                        "ensemble_experiments/test_selected/third_target_gate.sbatch",
                    ],
                    text=True,
                ).strip()
                report["followup_jobs"]["bigg27_group_e100"] = bigg_output
            if not args.vith_dump_job:
                raise ValueError("--queue-followups requires --vith-dump-job")
            atmvith_output = subprocess.check_output(
                [
                    "sbatch",
                    f"--dependency=afterany:{args.vith_dump_job}",
                    "--array=1",
                    "ensemble_experiments/test_selected/third_target_gate.sbatch",
                ],
                text=True,
            ).strip()
            report["followup_jobs"]["atm_vith_group_e100"] = atmvith_output

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
