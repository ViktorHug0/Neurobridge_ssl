"""Causal streaming re-fit: does a map estimated on the trials seen so far beat plain cosine?

Deployment regime. The menu (200 images) is fixed and known. Test trials are 20-repetition
averages, so THINGS-EEG-2's 80 test reps give 4 disjoint examples per concept -> a pool of
800 queries. They arrive one at a time in random order. Every `--every` arrivals we re-fit the
calibration on the buffer seen so far, freeze it, and decode the trials that have NOT arrived
yet, one at a time with plain cosine (no batch statistics). Strictly causal: nothing is ever
scored with a map that saw it.

The question is the crossover: how many trials must a new user produce before adaptation pays.

Arms, in increasing order of how much they assume:
  plain      no adaptation (flat reference)
  saw        frozen whitening only, no rotation
  balanced   published pipeline, rho=1 balanced Sinkhorn
  free       rho=0 free column marginal (the buffer covers only part of the menu early on,
             and balanced Sinkhorn forces mass onto the concepts that are absent)
  free_ramp  free, with the rotation damped by alpha = t / (t + alpha_n0) so a badly
             estimated early map is applied weakly instead of at full strength

CPU-only, post-hoc on cached embeddings.
"""
import argparse
import csv
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from dataclasses import replace
from scipy.linalg import schur

from shared import (
    DEFAULT_SOURCE_RUN_DIR, TTAParams, apply_saw_transform, apply_tta_calibration,
    cosine_scores, encode_repetition_blocks, fit_saw_transform, load_subject_context,
    score_features,
)
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes
from run_partial_coverage import damped_assign, topk_acc


def schur_form(orthogonal_map):
    """Real Schur form of a rotation: block-diagonal 2x2 blocks, one per rotation plane.

    Decomposed once per fit and reused for every alpha. logm/expm is not usable here: it
    costs 2.8s at d=512 and its branch cut at theta=pi makes expm(logm(R)) miss R by ~0.08.
    """
    matrix, basis = schur(np.asarray(orthogonal_map, dtype=np.float64), output="real")
    planes, angles, i = [], [], 0
    while i < matrix.shape[0] - 1:
        if abs(matrix[i + 1, i]) > 1e-9:
            planes.append(i)
            angles.append(np.arctan2(matrix[i + 1, i], matrix[i, i]))
            i += 2
        else:  # a 1x1 block is a fixed axis, or the theta=pi case, which cannot be interpolated
            i += 1
    return matrix, basis, planes, angles


def geodesic_blend(form, alpha):
    """Constant angular velocity on SO(d): every plane turns by exactly alpha*theta.

    The chord blend of `blend_orthogonal_map` instead turns each plane by
    atan2(alpha sin t, (1-alpha) + alpha cos t), which equals alpha*t only at
    alpha in {0, 0.5, 1} and deviates more the larger the plane angle.
    """
    matrix, basis, planes, angles = form
    matrix = matrix.copy()
    for i, theta in zip(planes, angles):
        c, s = np.cos(alpha * theta), np.sin(alpha * theta)
        matrix[i:i + 2, i:i + 2] = [[c, -s], [s, c]]
    return (basis @ matrix @ basis.T).astype(np.float32)


def fit_saw_or_center(query, params, no_whiten):
    """SAW as usual, or mean-centering with an identity whitener when whitening is disabled."""
    if not no_whiten:
        return fit_saw_transform(query, params)
    stats = {"mu": query.mean(axis=0, keepdims=True),
             "whitener": np.eye(query.shape[1], dtype=np.float32),
             "renorm": params.saw_renorm}
    return apply_saw_transform(query, stats), stats


def fit_calibration_rho(query, gallery, params, rho, no_whiten=False):
    """fit_tta_calibration with the rho-damped assignment of run_partial_coverage."""
    transformed, saw_stats = fit_saw_or_center(query, params, no_whiten)
    cumulative = np.eye(transformed.shape[1], dtype=np.float32)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        assignment = damped_assign(scores, params.sinkhorn_tau, params.sinkhorn_iters, rho)
        step = fit_soft_assignment_procrustes(
            transformed, gallery, assignment,
            power=params.soft_procrustes_power,
            normalize_inputs=params.soft_procrustes_normalize_inputs,
        )
        if step is None:
            break
        transformed = apply_orthogonal_map(transformed, step)
        cumulative = (cumulative @ np.asarray(step, dtype=np.float32)).astype(np.float32, copy=False)
        scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    return {"saw_stats": saw_stats, "orthogonal_map": cumulative}


def arm_alphas(args, t, n_gallery):
    """(name, alpha) per arm. The ramp sets the rotation strength from the coverage ratio
    u = t/K alone, so it needs no labels and no held-out split: a buffer that covers a small
    fraction of the menu supports only a weak rotation, and cap bounds the strength the
    fully covered buffer is allowed to reach."""
    arms = [(f"alpha{a:g}", a) for a in (args.alphas or [])]
    if args.alpha_ramp:
        cap, const = args.alpha_ramp
        u = t / n_gallery
        arms.append(("ramp", cap * u / (u + const)))
    return arms


def build_pool(blocks):
    """Stack the disjoint 20-avg blocks into one pool of (200 * n_blocks) labelled queries."""
    n_concepts = blocks[0].shape[0]
    queries = np.concatenate(blocks, axis=0).astype(np.float32)
    labels = np.tile(np.arange(n_concepts, dtype=np.int64), len(blocks))
    return queries, labels


def run_subject(source_run_dir, sid, args, params):
    rt = SimpleNamespace(device=args.device, batch_size=1024, num_workers=0)
    _, eval_args, dataset, modules = load_subject_context(source_run_dir, rt, sid, average=False)
    n_blocks = 80 // args.block_size
    blocks, gallery, _ = encode_repetition_blocks(
        eval_args, modules, dataset, block_sizes=[args.block_size] * n_blocks, seed=1000 + sid
    )
    pool, labels = build_pool(blocks)
    if args.menu and args.menu < len(gallery):
        # shrink the menu to `menu` items, fixed per subject: the stream can only contain
        # what the menu offers, so the pool shrinks with it and retrieval becomes menu-way.
        keep = np.sort(np.random.default_rng(500 + sid).choice(len(gallery), args.menu, replace=False))
        remap = np.full(len(gallery), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        mask = np.isin(labels, keep)
        pool, labels, gallery = pool[mask], remap[labels[mask]], gallery[keep]
    checkpoints = args.checkpoints or list(range(args.every, min(args.max_fit, len(pool)) + 1, args.every))
    checkpoints = [t for t in checkpoints if t <= len(pool)]

    rows = []
    for seed in range(args.stream_seeds):
        order = np.random.default_rng(10_000 * sid + seed).permutation(len(pool))
        for t in checkpoints:
            buf = order[:t]
            if args.acausal:
                rest = buf  # transductive: score exactly the samples the map was fitted on
            else:
                rest = order[t:]
                if len(rest) < args.min_eval:
                    break
            q_rest, y_rest = pool[rest], labels[rest]
            coverage = len(np.unique(labels[buf]))
            alpha = t / (t + args.alpha_n0)

            scored = {"plain": cosine_scores(q_rest, gallery)}
            if args.alphas or args.alpha_ramp:
                # alpha enters only at apply time, so one fit serves every damping strength
                cal = fit_calibration_rho(pool[buf], gallery, params, rho=args.rho,
                                          no_whiten=args.no_whiten)
                form = schur_form(cal["orthogonal_map"]) if args.geodesic else None
                whitened = apply_saw_transform(q_rest, cal["saw_stats"])
                for name, a in arm_alphas(args, t, len(gallery)):
                    if args.geodesic:
                        q = apply_orthogonal_map(whitened, geodesic_blend(form, a))
                    else:
                        q = apply_tta_calibration(q_rest, cal, alpha=a)
                    # CSLS is a batch statistic over the scored set, so it only makes sense
                    # transductively; the plain reference stays plain cosine, as in the paper.
                    scored[name] = (
                        score_features(q, gallery, use_csls=True, csls_k=params.csls_k)
                        if args.csls else cosine_scores(q, gallery))
            else:
                # covariance is unestimable from a short buffer, so shrink harder while t is small
                shrunk = replace(params, saw_shrink=1.0 - (1.0 - params.saw_shrink) * min(1.0, t / args.shrink_n_ref))
                saw_stats = fit_saw_transform(pool[buf], params)[1]
                scored["saw"] = cosine_scores(apply_saw_transform(q_rest, saw_stats), gallery)
                cal_bal = fit_calibration_rho(pool[buf], gallery, params, rho=1.0)
                cal_free = fit_calibration_rho(pool[buf], gallery, params, rho=0.0)
                cal_shrunk = fit_calibration_rho(pool[buf], gallery, shrunk, rho=0.0)
                scored["balanced"] = cosine_scores(apply_tta_calibration(q_rest, cal_bal, alpha=1.0), gallery)
                scored["free"] = cosine_scores(apply_tta_calibration(q_rest, cal_free, alpha=1.0), gallery)
                scored["free_ramp"] = cosine_scores(apply_tta_calibration(q_rest, cal_free, alpha=alpha), gallery)
                scored["free_ramp_shrink"] = cosine_scores(apply_tta_calibration(q_rest, cal_shrunk, alpha=alpha), gallery)

            for name, sc in scored.items():
                acc = topk_acc(sc, y_rest)
                rows.append((sid, seed, t, coverage, len(rest), name, acc[1], acc[5]))
        print(f"  subject {sid:02d} seed {seed} done", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_run_dir", default=DEFAULT_SOURCE_RUN_DIR)
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--every", type=int, default=10, help="re-fit interval, in arrivals")
    p.add_argument("--max_fit", type=int, default=800)
    p.add_argument("--checkpoints", nargs="+", type=int, default=None,
                   help="explicit re-fit points; overrides --every/--max_fit")
    p.add_argument("--alphas", nargs="+", type=float, default=None,
                   help="sweep fixed rotation-damping strengths instead of the named arms")
    p.add_argument("--rho", type=float, default=1.0, help="marginal for the alpha sweep (1.0 = published Sinkhorn)")
    p.add_argument("--menu", type=int, default=None, help="restrict the candidate set to this many items")
    p.add_argument("--acausal", action="store_true",
                   help="score the buffer itself instead of the unseen remainder (transductive)")
    p.add_argument("--csls", action="store_true",
                   help="score the calibrated queries with CSLS, as the published pipeline does")
    p.add_argument("--geodesic", action="store_true",
                   help="blend the rotation along the SO(d) geodesic instead of the chord")
    p.add_argument("--no_whiten", action="store_true",
                   help="identity whitener in the alpha sweep: keep centering and renorm, drop SAW")
    p.add_argument("--alpha_ramp", nargs=2, type=float, default=None, metavar=("CAP", "CONST"),
                   help="extra arm with alpha = CAP*u/(u+CONST), u = t/K the coverage ratio")
    p.add_argument("--shrink_n_ref", type=float, default=200.0,
                   help="buffer size at which adaptive shrinkage relaxes to the default")
    p.add_argument("--min_eval", type=int, default=40, help="stop once the unseen remainder is smaller than this")
    p.add_argument("--alpha_n0", type=float, default=200.0, help="ramp midpoint: alpha = t / (t + n0)")
    p.add_argument("--stream_seeds", type=int, default=3)
    p.add_argument("--block_size", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="results/things_eeg/tta_rebuttal/streaming_refit")
    args = p.parse_args()

    params = TTAParams()
    rows = []
    os.makedirs(args.output_dir, exist_ok=True)
    for sid in args.subjects:
        rows.extend(run_subject(os.path.abspath(args.source_run_dir), sid, args, params))
        with open(os.path.join(args.output_dir, "subject_results.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "seed", "t", "coverage_U", "n_eval", "method", "top1", "top5"])
            w.writerows(rows)

    agg = defaultdict(list)
    cov = defaultdict(list)
    for sid, seed, t, coverage, n_eval, name, t1, t5 in rows:
        agg[(t, name)].append((t1, t5))
        cov[t].append(coverage)
    named = ["plain", "saw", "balanced", "free", "free_ramp", "free_ramp_shrink"]
    seen = {m for _, m in agg}
    methods = ([n for n, _ in arm_alphas(args, 1, 1)] if (args.alphas or args.alpha_ramp)
               else [m for m in named if m in seen])
    methods = ["plain"] + [m for m in methods if m != "plain"]
    ts = sorted({t for t, _ in agg})

    summary = os.path.join(args.output_dir, "streaming_summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "coverage_U", "method", "top1_mean", "top5_mean", "n"])
        for t in ts:
            for m in methods:
                if (t, m) in agg:
                    a = np.array(agg[(t, m)])
                    w.writerow([t, round(np.mean(cov[t]), 1), m, round(a[:, 0].mean(), 3), round(a[:, 1].mean(), 3), len(a)])

    print(f"\nTop-1 vs trials seen  ->  {summary}")
    print("  " + "  ".join(f"{h:>10}" for h in ["t", "U", *methods]))
    for t in ts:
        line = [f"{t:>10}", f"{np.mean(cov[t]):>10.0f}"]
        line += [f"{np.array(agg[(t, m)])[:, 0].mean():>10.2f}" if (t, m) in agg else f"{'-':>10}" for m in methods]
        print("  " + "  ".join(line))

    base = {t: np.array(agg[(t, "plain")])[:, 0].mean() for t in ts}
    print("\nCrossover (first t where the arm stays above plain):")
    for m in methods[1:]:
        cross = next((t for i, t in enumerate(ts)
                      if all(np.array(agg[(t2, m)])[:, 0].mean() > base[t2] for t2 in ts[i:] if (t2, m) in agg)), None)
        print(f"  {m:<10} {cross if cross is not None else 'never'}")


if __name__ == "__main__":
    main()
