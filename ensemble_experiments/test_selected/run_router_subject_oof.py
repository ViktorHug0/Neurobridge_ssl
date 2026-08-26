"""Create one strict source-subject-OOF router-data task.

For outer target ``h``, the other nine subjects are deterministically split into
three groups by striding their sorted IDs.  One task excludes ``h`` and one
three-subject group, trains an expert on the remaining six subjects for the
globally fixed 75 epochs, and exports each excluded source subject separately.
The final checkpoint is selected at the fixed final epoch, so no held-out group
label influences expert checkpoint selection.

This file prepares the expensive R1 protocol.  It is not invoked automatically.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


ALL_SUBJECTS = tuple(range(1, 11))


def source_groups(outer: int) -> tuple[tuple[int, ...], ...]:
    sources = [subject for subject in ALL_SUBJECTS if subject != outer]
    groups = tuple(tuple(sources[offset::3]) for offset in range(3))
    if sorted(value for group in groups for value in group) != sources:
        raise AssertionError("invalid source partition")
    if any(len(group) != 3 for group in groups):
        raise AssertionError("each source group must contain three subjects")
    return groups


def completed_checkpoint(root: Path, output_name: str) -> Path | None:
    matches = [
        Path(path)
        for path in sorted(glob.glob(str(root / f"*-{output_name}")))
        if (Path(path) / "result.csv").is_file()
        and (Path(path) / "checkpoint_test_best.pth").is_file()
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"multiple completed runs for {output_name}: {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", required=True)
    parser.add_argument("--outer", type=int, required=True, choices=ALL_SUBJECTS)
    parser.add_argument("--group", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--recipes", default="ensemble_experiments/test_selected/router_member_recipes.json"
    )
    parser.add_argument(
        "--run-root",
        default="results/things_eeg/ensemble50_testselected/learned_router/subject_oof_runs",
    )
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/ensemble50_testselected/learned_router/subject_oof_dumps",
    )
    parser.add_argument(
        "--eval-root",
        default="results/things_eeg/ensemble50_testselected/learned_router/subject_oof_eval",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    recipes = json.loads(Path(args.recipes).read_text())
    if args.member not in recipes:
        raise ValueError(f"unknown member recipe: {args.member}")
    recipe = recipes[args.member]
    heldout = source_groups(args.outer)[args.group]
    train_subjects = [
        subject
        for subject in ALL_SUBJECTS
        if subject != args.outer and subject not in heldout
    ]
    if len(train_subjects) != 6:
        raise AssertionError(f"expected six inner training subjects, got {train_subjects}")

    run_root = Path(args.run_root) / args.member
    dump_root = Path(args.dump_root)
    eval_root = Path(args.eval_root)
    run_root.mkdir(parents=True, exist_ok=True)
    dump_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    output_name = f"outer{args.outer:02d}-group{args.group}"
    checkpoint = completed_checkpoint(run_root, output_name)
    if checkpoint is None:
        encoder_args = ["--eeg_encoder_type", str(recipe["eeg_encoder_type"])]
        if recipe["eeg_encoder_type"] == "TSConv_parameterizable":
            encoder_args.extend(
                [
                    "--tsconv_temporal_kernel",
                    "30",
                    "--tsconv_pool_kernel",
                    "51",
                    "--tsconv_dropout",
                    "0.50",
                ]
            )
        command = [
            sys.executable,
            "train.py",
            "--batch_size",
            "1024",
            "--num_workers",
            str(args.num_workers),
            "--learning_rate",
            "3e-4",
            "--num_epochs",
            "75",
            "--output_name",
            output_name,
            "--output_dir",
            str(run_root),
            *encoder_args,
            "--train_subject_ids",
            *[str(value) for value in train_subjects],
            "--test_subject_ids",
            *[str(value) for value in heldout],
            "--select_best_on",
            "fixed",
            "--image_feature_dir",
            str(recipe["image_feature_dir"]),
            "--eeg_data_dir",
            "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz",
            "--device",
            args.device,
            "--feature_dim",
            str(recipe["feature_dim"]),
            "--eeg_backbone_dim",
            str(recipe["eeg_backbone_dim"]),
            "--softplus",
            "--img_l2norm",
            "--projector",
            "linear",
            "--save_weights",
            "--text_feature_dir",
            "",
            "--data_average",
            "--grouped_batch_sampler",
            "--samples_per_image",
            str(len(train_subjects)),
            "--multi_positive_loss",
            "--subject_mixup_mode",
            "raw_eeg",
            "--mixup_type",
            str(recipe["mixup_type"]),
            "--subject_mixup_alpha",
            "0.5",
            "--subject_mixup_prob",
            "1.0",
            "--eval_mode",
            "plain_cosine",
            "--seed",
            "3300",
        ]
        subprocess.run(command, check=True)
        checkpoint = completed_checkpoint(run_root, output_name)
        if checkpoint is None:
            raise RuntimeError(f"training completed without checkpoint for {output_name}")

    for source in heldout:
        stem = f"{args.member}-outer{args.outer:02d}-source{source:02d}"
        output = dump_root / f"{stem}.npz"
        sidecar = dump_root / f"{stem}.json"
        metadata = {
            "protocol": "source_subject_oof_three_way",
            "member": args.member,
            "outer_subject": args.outer,
            "scored_source_subject": source,
            "source_group": list(heldout),
            "source_group_index": args.group,
            "source_subject_in_member_training": False,
            "train_subject_ids": train_subjects,
            "checkpoint_selection": "fixed_epoch_75",
            "checkpoint_dir": str(checkpoint),
            "dump": str(output),
        }
        if output.exists() and sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if saved != metadata:
                raise RuntimeError(f"metadata mismatch for existing {sidecar}")
            print(f"[exists] {stem}", flush=True)
            continue
        command = [
            sys.executable,
            "evaluate.py",
            "--checkpoint_dir",
            str(checkpoint),
            "--output_dir",
            str(eval_root),
            "--output_name",
            stem,
            "--eval_mode",
            "plain_cosine",
            "--test_subject_id",
            str(source),
            "--device",
            args.device,
            "--batch_size",
            "32",
            "--num_workers",
            "0",
            "--dump_npz",
            str(output),
        ]
        subprocess.run(command, check=True)
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[wrote] {stem}", flush=True)


if __name__ == "__main__":
    main()
