"""Partial-coverage streaming stress test for SAGE-TTA (deployment regime).

Fixed 200-item menu (gallery), but the query stream covers only U of them, with
duplicates and unknown non-uniform multiplicity. Balanced 2-sided Sinkhorn forces
equal column mass onto all 200 -- including the (200-U) absent items -- and mis-fits
the Procrustes rotation. We relax the column constraint to a single knob rho in [0,1]:
rho=1 is balanced (today's method), rho=0 is free-marginal row-softmax, intermediate
is unbalanced OT. Prediction: balanced degrades as U falls, free-marginal stays robust.

Each query is a 20-repetition average; THINGS-EEG-2 has 80 test reps, so every concept
has up to 4 disjoint 20-avg examples. For U selected concepts we draw 1..4 examples each.
CPU-only (post-hoc on cached embeddings); does not touch the GPU.
"""
import argparse
import csv
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from shared import (
    DEFAULT_SOURCE_RUN_DIR, TTAParams, cosine_scores, fit_saw_transform,
    load_subject_context, encode_repetition_blocks, score_features,
)
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize


def damped_assign(scores, tau, num_iters, rho, col_mass=1.0, eps=1e-8):
    """Sinkhorn with a rho-damped column step. rho=1 -> balanced (== sinkhorn_normalize),
    rho=0 -> row-softmax (free column marginal), 0<rho<1 -> unbalanced OT."""
    scores = np.asarray(scores, dtype=np.float32)
    tau = max(float(tau), eps)
    scaled = scores / tau
    scaled -= scaled.max(axis=1, keepdims=True)
    matrix = np.exp(scaled).astype(np.float32, copy=False)
    for _ in range(max(1, int(num_iters))):
        matrix /= np.clip(matrix.sum(axis=1, keepdims=True), eps, None)
        if rho > 0.0:
            matrix *= (float(col_mass) / np.clip(matrix.sum(axis=0, keepdims=True), eps, None)) ** rho
    return matrix


def tta_scores(query, gallery, params, rho):
    """SAGE-TTA (SAW -> CSLS -> Procrustes loop) with a rho-marginal assignment."""
    transformed, _ = fit_saw_transform(query, params)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        assignment = damped_assign(scores, params.sinkhorn_tau, params.sinkhorn_iters, rho)
        step_map = fit_soft_assignment_procrustes(
            transformed, gallery, assignment,
            power=params.soft_procrustes_power,
            normalize_inputs=params.soft_procrustes_normalize_inputs,
        )
        if step_map is None:
            break
        transformed = apply_orthogonal_map(transformed, step_map)
        scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    return scores


def build_stream(blocks, U, rng, max_mult=4):
    """Pick U of 200 concepts; give each 1..max_mult distinct 20-avg examples."""
    n_concepts = blocks[0].shape[0]
    concepts = rng.choice(n_concepts, size=U, replace=False)
    queries, labels = [], []
    for c in concepts:
        m = int(rng.integers(1, min(max_mult, len(blocks)) + 1))
        for b in rng.choice(len(blocks), size=m, replace=False):
            queries.append(blocks[b][c])
            labels.append(int(c))
    return np.stack(queries).astype(np.float32), np.asarray(labels, dtype=np.int64)


def topk_acc(scores, labels, ks=(1, 5)):
    order = np.argsort(-scores, axis=1)
    return {k: 100.0 * np.mean((order[:, :k] == labels[:, None]).any(axis=1)) for k in ks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_run_dir", default=DEFAULT_SOURCE_RUN_DIR)
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--coverage", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200])
    p.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--stream_seeds", type=int, default=5)
    p.add_argument("--block_size", type=int, default=20)
    p.add_argument("--max_mult", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="results/things_eeg/tta_rebuttal/partial_coverage")
    args = p.parse_args()

    params = TTAParams()
    rt = SimpleNamespace(device=args.device, batch_size=1024, num_workers=0)
    n_blocks = 80 // args.block_size
    methods = [("plain", None)] + [(f"rho{r:g}", r) for r in args.rhos]

    rows = []  # (subject, U, seed, method, top1, top5)
    for sid in args.subjects:
        _, eval_args, dataset, modules = load_subject_context(args.source_run_dir, rt, sid, average=False)
        blocks, gallery, _ = encode_repetition_blocks(
            eval_args, modules, dataset, block_sizes=[args.block_size] * n_blocks, seed=1000 + sid
        )
        for U in args.coverage:
            for s in range(args.stream_seeds):
                rng = np.random.default_rng(10_000 * sid + 100 * U + s)
                q, lab = build_stream(blocks, U, rng, args.max_mult)
                for name, rho in methods:
                    sc = cosine_scores(q, gallery) if rho is None else tta_scores(q, gallery, params, rho)
                    acc = topk_acc(sc, lab)
                    rows.append((sid, U, s, name, acc[1], acc[5]))
        print(f"  subject {sid:02d} done ({len(blocks)} blocks, gallery {gallery.shape[0]})")

    # aggregate mean over subjects x seeds
    agg = defaultdict(list)
    for sid, U, s, name, t1, t5 in rows:
        agg[(U, name)].append((t1, t5))
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "partial_coverage_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage_U", "method", "top1_mean", "top5_mean", "n"])
        for (U, name), vals in sorted(agg.items()):
            a = np.array(vals)
            w.writerow([U, name, round(a[:, 0].mean(), 3), round(a[:, 1].mean(), 3), len(vals)])

    print(f"\nTop-1 by coverage (rows=U) x method  ->  {csv_path}")
    header = ["U"] + [m for m, _ in methods]
    print("  " + "  ".join(f"{h:>10}" for h in header))
    for U in args.coverage:
        line = [f"{U:>10}"] + [f"{np.array(agg[(U, m)])[:,0].mean():>10.2f}" for m, _ in methods]
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
