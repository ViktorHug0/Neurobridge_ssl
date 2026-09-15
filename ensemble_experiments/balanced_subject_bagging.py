"""Deterministic 7-of-9 subject bags inspired by the MI ensemble study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BAG_SEED = 20260901
NUM_BAGS = 8


def bags_for_target(target: int) -> list[list[int]]:
    """Return eight balanced seven-subject bags for one LOSO target.

    A seeded cycle over the nine source subjects defines eight distinct adjacent
    exclusion pairs.  Seven sources are excluded twice and two are excluded
    once, so source exposure differs by at most one across the committee.
    """
    if target not in range(1, 11):
        raise ValueError("target must be in 1..10")
    sources = np.asarray([subject for subject in range(1, 11) if subject != target])
    cycle = np.random.default_rng(BAG_SEED + target).permutation(sources)
    bags = []
    for member in range(NUM_BAGS):
        excluded = {int(cycle[member]), int(cycle[(member + 1) % len(cycle)])}
        bags.append(sorted(set(sources.tolist()) - excluded))
    return bags


def manifest() -> dict[str, object]:
    return {
        "seed": BAG_SEED,
        "design": "eight deterministic balanced 7-of-9 source-subject bags",
        "bags": {
            f"sub-{target:02d}": bags_for_target(target)
            for target in range(1, 11)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-bag", type=int, nargs=2, metavar=("TARGET", "MEMBER"))
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    if args.print_bag is not None:
        target, member = args.print_bag
        if member not in range(NUM_BAGS):
            raise SystemExit(f"member must be in 0..{NUM_BAGS - 1}")
        print(" ".join(map(str, bags_for_target(target)[member])))
        return
    if args.write_manifest is None:
        parser.error("one of --print-bag or --write-manifest is required")
    args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.write_manifest.write_text(json.dumps(manifest(), indent=2) + "\n")


if __name__ == "__main__":
    main()
