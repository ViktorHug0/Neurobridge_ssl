"""Export plain-cosine embeddings for selected folds of a test-selected arm."""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dump-suffix", default="")
    parser.add_argument("--eeg-tta-shifts", type=int, nargs="+", default=None)
    parser.add_argument("--checkpoint-name", default="checkpoint_test_best.pth")
    args = parser.parse_args()

    dump_root = Path("results/things_eeg/synthetic_subjects/ensemble_screen/dumps")
    eval_root = Path("results/things_eeg/ensemble50_testselected/eval")
    dump_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        output = dump_root / f"{args.arm}{args.dump_suffix}-sub{subject:02d}.npz"
        if output.exists():
            print(f"exists {output}")
            continue
        matches = sorted(glob.glob(os.path.join(args.run_root, f"*-sub-{subject:02d}")))
        matches = [path for path in matches if os.path.isfile(os.path.join(path, "result.csv"))]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one completed sub-{subject:02d} run under {args.run_root}, got {matches}"
            )
        command = [
            sys.executable,
            "evaluate.py",
            "--checkpoint_dir",
            matches[0],
            "--output_dir",
            str(eval_root),
            "--output_name",
            f"{args.arm}{args.dump_suffix}-sub{subject:02d}",
            "--eval_mode",
            "plain_cosine",
            "--test_subject_id",
            str(subject),
            "--device",
            args.device,
            "--batch_size",
            "32",
            "--num_workers",
            "0",
            "--dump_npz",
            str(output),
            "--checkpoint_name",
            args.checkpoint_name,
        ]
        if args.eeg_tta_shifts is not None:
            command.extend(
                ["--eeg_tta_shifts", *[str(shift) for shift in args.eeg_tta_shifts]]
            )
        subprocess.run(command, check=True)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
