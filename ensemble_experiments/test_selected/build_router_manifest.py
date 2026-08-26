"""Build and validate the explicit data manifest used by the learned router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ensemble_experiments/test_selected/router_config.json")
    parser.add_argument(
        "--source-dump-root",
        default="results/things_eeg/ensemble50_testselected/learned_router/source_dumps",
    )
    parser.add_argument(
        "--target-dump-root",
        default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps",
    )
    parser.add_argument(
        "--output",
        default="results/things_eeg/ensemble50_testselected/learned_router/manifest.json",
    )
    parser.add_argument(
        "--require-source-subject-oof",
        action="store_true",
        help="reject source dumps whose subject occurred in the expert training set",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    members = list(config["members"])
    subjects = [int(v) for v in config["outer_subjects"]]
    source_root = Path(args.source_dump_root)
    target_root = Path(args.target_dump_root)
    folds: dict[str, dict] = {}

    for outer in subjects:
        train_records = []
        for source in subjects:
            if source == outer:
                continue
            dumps = {}
            protocols = set()
            for member in members:
                stem = f"{member}-outer{outer:02d}-source{source:02d}"
                dump = source_root / f"{stem}.npz"
                sidecar = source_root / f"{stem}.json"
                if not dump.is_file() or not sidecar.is_file():
                    raise FileNotFoundError(f"missing source dump or sidecar for {stem}")
                metadata = json.loads(sidecar.read_text())
                if int(metadata["outer_subject"]) != outer:
                    raise RuntimeError(f"outer mismatch in {sidecar}")
                if int(metadata["scored_source_subject"]) != source:
                    raise RuntimeError(f"source mismatch in {sidecar}")
                train_subjects = set(int(v) for v in metadata["train_subject_ids"])
                if outer in train_subjects:
                    raise RuntimeError(f"target subject leaked into {sidecar}")
                source_seen = bool(metadata["source_subject_in_member_training"])
                if args.require_source_subject_oof and source_seen:
                    raise RuntimeError(f"source subject is not OOF in {sidecar}")
                protocols.add(str(metadata["protocol"]))
                dumps[member] = str(dump)
            if len(protocols) != 1:
                raise RuntimeError(
                    f"members disagree on source protocol for outer={outer}, source={source}"
                )
            train_records.append(
                {
                    "subject": source,
                    "protocol": protocols.pop(),
                    "dumps": dumps,
                }
            )

        target_dumps = {}
        for member in members:
            path = target_root / f"{member}-sub{outer:02d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            target_dumps[member] = str(path)
        folds[str(outer)] = {
            "outer_subject": outer,
            "router_train": train_records,
            "router_test": {"subject": outer, "dumps": target_dumps},
        }

    manifest = {
        "protocol_version": config["protocol_version"],
        "members": members,
        "config": str(Path(args.config)),
        "require_source_subject_oof": bool(args.require_source_subject_oof),
        "folds": folds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
