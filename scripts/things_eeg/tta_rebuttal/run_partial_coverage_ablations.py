"""Partial-coverage ablations motivated by the CLIP/TransCLIP transductive line.

Same regime as run_partial_coverage.py (fixed 200-item menu, query stream covers
only U concepts with 1..4 duplicates), but adds three arms suggested by the
parallel CLIP experiments (transduction-for-vlms / transductive-CLIP):

  Exp 1  support selection  -- 'mask_est': hard-drop the low-estimated-mass gallery
         columns from the Procrustes correspondences (E13: the bottleneck is support
         selection, not the estimator). 'oracle_support': restrict the readout to the
         truly-present items -- the A1-style ceiling (oracle=97.8 on CLIP).
  Exp 2  Procrustes-off     -- 'saw_csls': SAW + CSLS, no rotation (E1/E3: the
         assignment constraint is the engine; the rotation craters when N<d).
  Exp 3  adaptive rho       -- 'adaptive': set the column-marginal relaxation rho per
         batch from the query stream's marginal entropy (E14 entropy gate). Sparse
         batch -> low rho (free); full coverage -> rho->1 (balanced).

References kept in-stream for fair comparison: plain / free (rho0) / balanced (rho1).
CPU-only, post-hoc on cached embeddings.
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
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes
from run_partial_coverage import damped_assign, build_stream, topk_acc


def initial_scores(query, gallery, params):
    """SAW-whiten then CSLS-score, no rotation (== the saw_csls arm)."""
    transformed, _ = fit_saw_transform(query, params)
    return score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)


def tta_generic(query, gallery, params, assign_fn):
    """SAGE-TTA loop (SAW -> CSLS -> Procrustes) with a pluggable assignment."""
    transformed, _ = fit_saw_transform(query, params)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        P = assign_fn(scores)
        step_map = fit_soft_assignment_procrustes(
            transformed, gallery, P,
            power=params.soft_procrustes_power,
            normalize_inputs=params.soft_procrustes_normalize_inputs,
        )
        if step_map is None:
            break
        transformed = apply_orthogonal_map(transformed, step_map)
        scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    return scores


def row_softmax(scores, tau, eps=1e-8):
    scaled = scores / max(float(tau), eps)
    scaled -= scaled.max(axis=1, keepdims=True)
    P = np.exp(scaled).astype(np.float32)
    P /= np.clip(P.sum(axis=1, keepdims=True), eps, None)
    return P


def estimate_keep_mask(scores, tau, thresh, min_keep=20):
    """Estimated support: keep gallery columns whose soft column-mass >= thresh.
    Floor at min_keep heaviest columns so a too-small support can't rank-collapse
    the Procrustes fit (a rotation from <d points in 512-d explodes; cf. E3)."""
    mass = row_softmax(scores, tau).sum(axis=0)  # sums to N
    keep = mass >= float(thresh)
    if keep.sum() < min_keep:
        keep = np.zeros_like(keep)
        keep[np.argsort(-mass)[:min_keep]] = True
    return keep


def masked_softmax_assign(scores, tau, keep_mask, eps=1e-8):
    """Free-marginal row-softmax restricted to kept columns (masked cols get 0)."""
    scaled = np.where(keep_mask[None, :], scores / max(float(tau), eps), -1e30)
    scaled -= scaled.max(axis=1, keepdims=True)
    P = np.exp(scaled).astype(np.float32)
    P[:, ~keep_mask] = 0.0
    P /= np.clip(P.sum(axis=1, keepdims=True), eps, None)
    return P


def adaptive_rho(scores, K, gate=0.9, eps=1e-12):
    """Entropy-gated rho (E15 recipe). H_norm = normalized entropy of the batch's
    HARD-prediction histogram (argmax over the gallery) -- concentrated (sparse
    coverage) -> low H_norm; uniform (full coverage) -> H_norm->1. Gate at `gate`
    so rho stays ~0 (free) until the batch marginal is near-uniform, then ramps to
    1 (balanced). Argmax (not soft mass) is used because the cross-subject base
    scores are too diffuse for soft-mass entropy to track coverage."""
    counts = np.bincount(np.argmax(scores, axis=1), minlength=K).astype(np.float64)
    p = counts / max(counts.sum(), eps)
    p = p[p > 0]
    H_norm = float(-np.sum(p * np.log(p)) / np.log(max(K, 2)))
    return float(np.clip((H_norm - gate) / (1.0 - gate), 0.0, 1.0)), H_norm


def all_method_scores(q, lab, gallery, params, n_concepts, mask_thresh, gate):
    tau, iters = params.sinkhorn_tau, params.sinkhorn_iters
    init = initial_scores(q, gallery, params)  # == saw_csls scores, reused for adaptive rho
    rho_a, h_norm = adaptive_rho(init, n_concepts, gate)

    free = tta_generic(q, gallery, params, lambda s: damped_assign(s, tau, iters, 0.0))
    present = np.zeros(n_concepts, dtype=bool)
    present[np.unique(lab)] = True
    oracle = free.copy()
    oracle[:, ~present] = -np.inf
    # deployable support selection: drop estimated-absent items from the READOUT
    est_keep = estimate_keep_mask(free, tau, mask_thresh)
    mask_readout = free.copy()
    mask_readout[:, ~est_keep] = -np.inf

    scores = {
        "plain": cosine_scores(q, gallery),
        "saw_csls": init,
        "free": free,
        "balanced": tta_generic(q, gallery, params, lambda s: damped_assign(s, tau, iters, 1.0)),
        "mask_est": tta_generic(
            q, gallery, params,
            lambda s: masked_softmax_assign(s, tau, estimate_keep_mask(s, tau, mask_thresh)),
        ),
        "mask_readout": mask_readout,
        "adaptive": tta_generic(q, gallery, params, lambda s: damped_assign(s, tau, iters, rho_a)),
        "oracle_support": oracle,
    }
    return scores, rho_a, h_norm


METHODS = ["plain", "saw_csls", "free", "balanced", "mask_est", "mask_readout",
           "adaptive", "oracle_support"]


def _sanity():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((12, 8)).astype(np.float32)
    P = masked_softmax_assign(s, 0.1, np.array([True, True, False, True, False, True, True, True]))
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-4) and np.all(P[:, [2, 4]] == 0.0)
    keep = estimate_keep_mask(s, 0.1, thresh=1e9, min_keep=2)  # impossible thresh -> fallback
    assert keep.sum() == 2
    r, hn = adaptive_rho(s, 8)
    assert 0.0 <= r <= 1.0 and 0.0 <= hn <= 1.0
    # rho=1 damped column step == balanced marginal; rho=0 == row-softmax
    assert np.allclose(damped_assign(s, 0.1, 12, 0.0), row_softmax(s, 0.1), atol=1e-5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_run_dir", default=DEFAULT_SOURCE_RUN_DIR)
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--coverage", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200])
    p.add_argument("--stream_seeds", type=int, default=5)
    p.add_argument("--block_size", type=int, default=20)
    p.add_argument("--max_mult", type=int, default=4)
    p.add_argument("--mask_thresh", type=float, default=0.5)
    p.add_argument("--adaptive_gate", type=float, default=0.9)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="results/things_eeg/tta_rebuttal/partial_coverage_ablations")
    args = p.parse_args()

    _sanity()
    params = TTAParams()
    rt = SimpleNamespace(device=args.device, batch_size=1024, num_workers=0)
    n_blocks = 80 // args.block_size

    rows = []       # (subject, U, seed, method, top1, top5)
    rho_rows = []   # (subject, U, seed, adaptive_rho, h_norm)
    for sid in args.subjects:
        _, eval_args, dataset, modules = load_subject_context(args.source_run_dir, rt, sid, average=False)
        blocks, gallery, _ = encode_repetition_blocks(
            eval_args, modules, dataset, block_sizes=[args.block_size] * n_blocks, seed=1000 + sid
        )
        n_concepts = blocks[0].shape[0]
        for U in args.coverage:
            for s in range(args.stream_seeds):
                rng = np.random.default_rng(10_000 * sid + 100 * U + s)
                q, lab = build_stream(blocks, U, rng, args.max_mult)
                scores, rho_a, h_norm = all_method_scores(
                    q, lab, gallery, params, n_concepts, args.mask_thresh, args.adaptive_gate)
                rho_rows.append((sid, U, s, rho_a, h_norm))
                for name in METHODS:
                    acc = topk_acc(scores[name], lab)
                    rows.append((sid, U, s, name, acc[1], acc[5]))
        print(f"  subject {sid:02d} done ({len(blocks)} blocks, gallery {gallery.shape[0]})", flush=True)

    agg = defaultdict(list)
    for sid, U, s, name, t1, t5 in rows:
        agg[(U, name)].append((t1, t5))
    rho_agg = defaultdict(list)
    hnorm_agg = defaultdict(list)
    for sid, U, s, r, hn in rho_rows:
        rho_agg[U].append(r)
        hnorm_agg[U].append(hn)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "partial_coverage_ablations_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage_U", "method", "top1_mean", "top5_mean", "n",
                    "adaptive_rho_mean", "h_norm_mean"])
        for (U, name), vals in sorted(agg.items()):
            a = np.array(vals)
            w.writerow([U, name, round(a[:, 0].mean(), 3), round(a[:, 1].mean(), 3), len(vals),
                        round(float(np.mean(rho_agg[U])), 3), round(float(np.mean(hnorm_agg[U])), 3)])

    print(f"\nTop-1 by coverage (rows=U) x method  ->  {csv_path}")
    header = ["U"] + METHODS + ["rho_a", "H_norm"]
    print("  " + "  ".join(f"{h:>13}" for h in header))
    for U in args.coverage:
        line = [f"{U:>13}"] + [f"{np.array(agg[(U, m)])[:, 0].mean():>13.2f}" for m in METHODS]
        line += [f"{np.mean(rho_agg[U]):>13.3f}", f"{np.mean(hnorm_agg[U]):>13.3f}"]
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
