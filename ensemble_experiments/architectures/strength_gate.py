"""Gate a LOSO candidate on its first five subjects against a reference mean."""

import argparse
import csv
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--reference", type=float, required=True)
    parser.add_argument("--max-gap", type=float, required=True)
    return parser.parse_args()


def read_best_top1(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or "best top1 acc" not in rows[0]:
        raise ValueError(f"unexpected result format: {path}")
    return float(rows[0]["best top1 acc"])


def main():
    args = parse_args()
    values = []
    for subject in args.subjects:
        matches = glob.glob(os.path.join(args.run_dir, f"*-sub-{subject:02d}", "result.csv"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one completed fold for subject {subject:02d}, found {len(matches)}"
            )
        values.append(read_best_top1(matches[0]))

    mean = sum(values) / len(values)
    floor = args.reference - args.max_gap
    keep = mean >= floor
    scores = ", ".join(f"{value:.2f}" for value in values)
    print(
        f"[strength-gate] subjects={args.subjects} scores=[{scores}] "
        f"mean={mean:.2f} reference={args.reference:.2f} floor={floor:.2f} "
        f"{'CONTINUE' if keep else 'DROP'}",
        flush=True,
    )
    raise SystemExit(0 if keep else 1)


if __name__ == "__main__":
    main()
