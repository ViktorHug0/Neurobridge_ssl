"""SAGE on NSD using the repository's own calibration code, not a reimplementation.

The earlier NSD attempt was a single-shot approximation: one Procrustes step, plain-cosine
scoring, SAW shrink 0.2. The published pipeline iterates Sinkhorn <-> soft Procrustes 16
times, re-scoring with CSLS(k=3) at every step, at shrink 0.94 and power 1.2. On THINGS-EEG
that loop climbs from a 33.0 Sinkhorn assignment to 42.5 final accuracy; a single step cannot
climb at all. So this calls fit_tta_calibration directly and only supplies the features.

The other fix is the space the rotation lives in. MindEye2 embeddings are 256 tokens x 1664
dims and the published metric is a cosine over the flattened 425,984-dim vector. Fitting one
1664x1664 block shared across token positions constrains the map to I_256 (x) R_token, a tiny
subgroup, and that constraint alone destroyed retrieval. Here the features are projected onto
a shared PCA basis of the flattened space, giving one moderate-dimensional vector per trial --
structurally the same object SAGE consumes on EEG.

Reported per k: baseline, the Sinkhorn assignment's own accuracy, SAGE, and a shuffled-
assignment control. SAGE must beat BOTH the baseline and the assignment to be doing work the
assignment could not do alone.

Usage:  python sage_faithful.py --subj 1 --ks 128 256 512
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL/scripts/things_eeg/tta_rebuttal")

from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize  # noqa: E402
from shared import TTAParams, fit_saw_transform, score_features  # noqa: E402

DATA = "/nasbrain/p20fores/mindeye_data"


def pca_basis(x, k):
    g = (x @ x.T).double()
    evals, evecs = torch.linalg.eigh(g)
    idx = torch.argsort(evals, descending=True)[:k]
    evals, evecs = evals[idx].clamp_min(1e-8), evecs[:, idx]
    return (x.T @ evecs.float()) / evals.sqrt().float()


def top1(scores):
    return float((scores.argmax(axis=1) == np.arange(len(scores))).mean()) * 100


def calibrate(query, gallery, params, shuffle_rng=None):
    """fit_tta_calibration verbatim, with an optional shuffled assignment as the null."""
    transformed, _ = fit_saw_transform(query, params)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    assign_acc = None
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        assignment = sinkhorn_normalize(scores, tau=params.sinkhorn_tau,
                                        num_iters=params.sinkhorn_iters,
                                        col_mass=params.sinkhorn_col_mass)
        if assign_acc is None:
            assign_acc = top1(assignment)
        if shuffle_rng is not None:
            assignment = assignment[:, shuffle_rng.permutation(assignment.shape[1])]
        step = fit_soft_assignment_procrustes(
            transformed, gallery, assignment, power=params.soft_procrustes_power,
            normalize_inputs=params.soft_procrustes_normalize_inputs)
        if step is None:
            break
        transformed = apply_orthogonal_map(transformed, step)
        scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    return scores, assign_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 256, 512, 999])
    ap.add_argument("--no_centre", action="store_true")
    args = ap.parse_args()
    model = args.model_name or f"final_subj{args.subj:02d}_pretrained_1sess_24bs"
    params = TTAParams()

    gal = torch.load(f"{DATA}/gallery_bigG_tokens.pt", weights_only=False).float()
    cv = torch.load(f"{DATA}/clipvoxels/clipvoxels_subj{args.subj}_{model}.pt",
                    weights_only=False)["clipvoxels"].float()
    q = cv.reshape(len(cv), -1)
    g = gal.reshape(len(gal), -1)
    if not args.no_centre:
        q, g = q - q.mean(0, keepdim=True), g - g.mean(0, keepdim=True)

    print(f"subj{args.subj:02d} {model}")
    print(f"{'k':>5} {'baseline':>9} {'assign':>7} {'sage':>7} {'shuffled':>9}")
    for k in args.ks:
        basis = pca_basis(torch.cat([q, g]), k)
        qk = (q @ basis).numpy().astype(np.float32)
        gk = (g @ basis).numpy().astype(np.float32)
        base = top1(score_features(qk, gk, use_csls=False))
        sage_scores, assign_acc = calibrate(qk, gk, params)
        shuf_scores, _ = calibrate(qk, gk, params, shuffle_rng=np.random.RandomState(0))
        print(f"{k:>5} {base:>9.1f} {assign_acc:>7.1f} {top1(sage_scores):>7.1f} {top1(shuf_scores):>9.1f}")

    print(f"\nchance for {len(g)}-way = {100/len(g):.1f}%")
    print("SAGE must exceed baseline AND assignment; shuffled at chance is expected (EEG does the same)")


if __name__ == "__main__":
    main()
