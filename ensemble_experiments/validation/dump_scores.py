"""Export target-test and source-validation embeddings for honest outer folds.

Each checkpoint was selected only on its configured ``val_subject_id``.  For an
outer fold h this script exports scores for both h and that source validation
subject, allowing ensemble membership to be selected without reading h labels.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def checkpoint_for(root: str, outer: int) -> str | None:
    matches = sorted(glob.glob(os.path.join(root, f"*-sub-{outer:02d}")))
    matches = [path for path in matches if os.path.isfile(os.path.join(path, "checkpoint_test_best.pth"))]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for outer={outer} under {root}, got {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="ensemble_experiments/validation/pool.json")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--outers", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    pool = json.load(open(args.pool))
    if args.arms:
        pool = {name: pool[name] for name in args.arms}
    dump_root = Path("results/things_eeg/honest_ensemble/dumps")
    eval_root = Path("results/things_eeg/honest_ensemble/eval")
    dump_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    for arm, roots in pool.items():
        if isinstance(roots, str):
            val_root = test_root = roots
        else:
            val_root = roots["val_root"]
            test_root = roots["test_root"]
        for outer in args.outers:
            val_checkpoint = checkpoint_for(val_root, outer)
            test_checkpoint = checkpoint_for(test_root, outer)
            if val_checkpoint is None or test_checkpoint is None:
                print(
                    f"[skip] {arm} outer={outer}: incomplete val/refit checkpoint pair",
                    flush=True,
                )
                continue
            val_config = json.load(open(os.path.join(val_checkpoint, "train_config.json")))
            test_config = json.load(open(os.path.join(test_checkpoint, "train_config.json")))
            val = val_config.get("val_subject_id")
            if val_config.get("select_best_on") != "val" or val is None:
                raise RuntimeError(f"{val_checkpoint} is not a validation-selected checkpoint")
            if test_root != val_root and test_config.get("select_best_on") != "fixed":
                raise RuntimeError(f"{test_checkpoint} is not a fixed-epoch refit checkpoint")
            for role, subject in (("test", outer), ("val", int(val))):
                checkpoint = test_checkpoint if role == "test" else val_checkpoint
                output = dump_root / f"{arm}-outer{outer:02d}-{role}{subject:02d}.npz"
                if output.exists():
                    continue
                command = [
                    sys.executable,
                    "evaluate.py",
                    "--checkpoint_dir",
                    checkpoint,
                    "--output_dir",
                    str(eval_root),
                    "--output_name",
                    f"{arm}-outer{outer:02d}-{role}{subject:02d}",
                    "--eval_mode",
                    "plain_cosine",
                    "--test_subject_id",
                    str(subject),
                    "--device",
                    args.device,
                    "--batch_size",
                    "32",
                    "--num_workers",
                    str(args.num_workers),
                    "--dump_npz",
                    str(output),
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    detail = result.stderr.strip().splitlines()
                    raise RuntimeError(
                        f"Failed {arm} outer={outer} {role}={subject}: "
                        + (detail[-1] if detail else "unknown error")
                    )
                print(f"[ok] {arm} outer={outer} {role}={subject}", flush=True)


if __name__ == "__main__":
    main()
