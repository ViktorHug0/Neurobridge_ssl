"""Export source-subject score dumps from each outer-fold expert checkpoint.

For outer fold ``h`` the selected checkpoint was trained without subject ``h``.
This script evaluates that same checkpoint on the held-out 200-image test split
of each source subject ``s != h``.  The resulting examples are source-only and
out-of-stimulus, but the source subject was seen during expert training.  They
are therefore a cheap router-development protocol, not strict subject-OOF data.

Every dump receives a JSON sidecar recording the checkpoint and training
subjects.  ``build_router_manifest.py`` uses those sidecars to prevent an
accidental target-subject leak or an overstatement of the OOF protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


def checkpoint_for(run_root: str, outer: int) -> Path:
    matches = [
        Path(path)
        for path in sorted(glob.glob(str(Path(run_root) / f"*-sub-{outer:02d}")))
        if (Path(path) / "checkpoint_test_best.pth").is_file()
        and (Path(path) / "train_config.json").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one completed checkpoint for outer={outer} under "
            f"{run_root}, got {matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool", default="ensemble_experiments/test_selected/router_member_pool.json"
    )
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--outers", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--sources", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--dump-root",
        default="results/things_eeg/ensemble50_testselected/learned_router/source_dumps",
    )
    parser.add_argument(
        "--eval-root",
        default="results/things_eeg/ensemble50_testselected/learned_router/source_eval",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    if args.arms:
        unknown = sorted(set(args.arms) - set(pool))
        if unknown:
            raise ValueError(f"unknown arms: {unknown}")
        pool = {name: pool[name] for name in args.arms}

    dump_root = Path(args.dump_root)
    eval_root = Path(args.eval_root)
    dump_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    for member, run_root in pool.items():
        for outer in args.outers:
            checkpoint = checkpoint_for(run_root, outer)
            train_config = json.loads((checkpoint / "train_config.json").read_text())
            train_subjects = sorted(int(v) for v in train_config["train_subject_ids"])
            if outer in train_subjects:
                raise RuntimeError(
                    f"target leak: outer subject {outer} occurs in {checkpoint}'s training set"
                )
            for source in args.sources:
                if source == outer:
                    continue
                if source not in train_subjects:
                    raise RuntimeError(
                        f"source {source} was not a training subject for {checkpoint}"
                    )
                stem = f"{member}-outer{outer:02d}-source{source:02d}"
                output = dump_root / f"{stem}.npz"
                sidecar = dump_root / f"{stem}.json"
                metadata = {
                    "protocol": "source_seen_heldout_stimulus",
                    "member": member,
                    "outer_subject": outer,
                    "scored_source_subject": source,
                    "source_subject_in_member_training": True,
                    "train_subject_ids": train_subjects,
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
                    str(args.num_workers),
                    "--dump_npz",
                    str(output),
                ]
                subprocess.run(command, check=True)
                sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"[wrote] {stem}", flush=True)


if __name__ == "__main__":
    main()
