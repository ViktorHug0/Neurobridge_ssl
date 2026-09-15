"""Candidate-set size under strict bijection: N queries against N candidates, N-way retrieval.

80-repetition averages give exactly one query per concept, so restricting to N concepts yields a
bijective transductive problem of size N. This is the paper's main setting with the menu shrunk.
Chance is 1/N, so accuracy is not comparable across N; the question is how the gain over no
adaptation scales as the candidate set shrinks.

CPU-only, post-hoc on cached embeddings.
"""
import argparse
import csv
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from shared import (
    DEFAULT_SOURCE_RUN_DIR, TTAParams, encode_repetition_blocks, evaluate_full_tta,
    evaluate_plain, load_subject_context,
)


def run_subject(source_run_dir, sid, args, params):
    rt = SimpleNamespace(device=args.device, batch_size=1024, num_workers=0)
    _, eval_args, dataset, modules = load_subject_context(source_run_dir, rt, sid, average=False)
    blocks, gallery, _ = encode_repetition_blocks(
        eval_args, modules, dataset, block_sizes=[80], seed=1000 + sid
    )
    queries = blocks[0].astype(np.float32)

    rows = []
    for n in args.menu_sizes:
        for seed in range(args.seeds):
            if n >= len(gallery):
                keep = np.arange(len(gallery))          # full menu, nothing to sample
            else:
                keep = np.random.default_rng(7000 * sid + 13 * seed + n).choice(
                    len(gallery), n, replace=False)
            q, g = queries[keep], gallery[keep]
            plain = evaluate_plain(q, g)
            tta = evaluate_full_tta(q, g, params)
            rows.append((sid, seed, n, plain["top1_acc"], plain["top5_acc"],
                         tta["top1_acc"], tta["top5_acc"]))
            if n >= len(gallery):
                break                                   # full menu has no sampling variance
    print(f"  subject {sid:02d} done", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_run_dir", default=DEFAULT_SOURCE_RUN_DIR)
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--menu_sizes", nargs="+", type=int, default=[25, 50, 100, 150, 200])
    p.add_argument("--seeds", type=int, default=5, help="concept subsets drawn per subject and size")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="results/things_eeg/tta_rebuttal/menu_size_bijective")
    args = p.parse_args()

    params = TTAParams()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for sid in args.subjects:
        rows.extend(run_subject(os.path.abspath(args.source_run_dir), sid, args, params))
        with open(os.path.join(args.output_dir, "subject_results.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "seed", "menu_size", "plain_top1", "plain_top5", "tta_top1", "tta_top5"])
            w.writerows(rows)

    agg = defaultdict(list)
    for _, _, n, p1, _, t1, _ in rows:
        agg[n].append((p1, t1))
    print(f"\nMenu size (bijective, N-way)  ->  {args.output_dir}")
    print(f"  {'N':>6}  {'chance':>7}  {'no adapt':>9}  {'SAGE-TTA':>9}  {'gain':>7}")
    for n in sorted(agg):
        a = np.array(agg[n])
        print(f"  {n:>6}  {100.0 / n:>7.1f}  {a[:, 0].mean():>9.1f}  {a[:, 1].mean():>9.1f}  "
              f"{a[:, 1].mean() - a[:, 0].mean():>+7.1f}")


if __name__ == "__main__":
    main()
