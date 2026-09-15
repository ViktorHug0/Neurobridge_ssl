"""Per-step trace of the SAGE loop on NSD and on EEG, to find what makes one climb and one fall.

EEG starts from a 33% Sinkhorn assignment and ends at 42.5. NSD starts from 87.6 and ends at
59.7. The endpoint alone cannot say whether NSD diverges immediately or decays gradually, nor
whether the assignment SAGE feeds itself is soft (errors average out across many candidates)
or effectively hard (each wrong entry drags the rotation at full weight).

Per step this reports retrieval, the assignment's own accuracy, and the shape of the Sinkhorn
plan: the mean row-max (1.0 = one-hot, 1/n = uniform) and the normalised row entropy.

Usage:  python trace_iterations.py --which nsd
        python trace_iterations.py --which eeg --subject 1
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


def top1(m):
    return float((m.argmax(axis=1) == np.arange(len(m))).mean()) * 100


def plan_shape(plan):
    rows = plan / np.clip(plan.sum(axis=1, keepdims=True), 1e-12, None)
    row_max = float(rows.max(axis=1).mean())
    ent = -(rows * np.log(np.clip(rows, 1e-12, None))).sum(axis=1).mean()
    return row_max, float(ent / np.log(plan.shape[1]))


def trace(query, gallery, params, label):
    transformed, _ = fit_saw_transform(query, params)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    print(f"\n=== {label} ===")
    print(f"plain cosine top1 {top1(score_features(query, gallery, use_csls=False)):.1f}   "
          f"after SAW+CSLS {top1(scores):.1f}")
    print(f"{'step':>4} {'retrieval':>10} {'assign':>7} {'row_max':>8} {'entropy':>8}")
    for step in range(int(params.soft_procrustes_steps)):
        plan = sinkhorn_normalize(scores, tau=params.sinkhorn_tau, num_iters=params.sinkhorn_iters,
                                  col_mass=params.sinkhorn_col_mass)
        rm, ent = plan_shape(plan)
        print(f"{step:>4} {top1(scores):>10.1f} {top1(plan):>7.1f} {rm:>8.4f} {ent:>8.4f}")
        m = fit_soft_assignment_procrustes(transformed, gallery, plan,
                                           power=params.soft_procrustes_power,
                                           normalize_inputs=params.soft_procrustes_normalize_inputs)
        if m is None:
            print("  (procrustes returned None)")
            break
        transformed = apply_orthogonal_map(transformed, m)
        scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    print(f"{'end':>4} {top1(scores):>10.1f}")


def pca_basis(x, k):
    g = (x @ x.T).double()
    evals, evecs = torch.linalg.eigh(g)
    idx = torch.argsort(evals, descending=True)[:k]
    evals, evecs = evals[idx].clamp_min(1e-8), evecs[:, idx]
    return (x.T @ evecs.float()) / evals.sqrt().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["nsd", "eeg"], default="nsd")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    params = TTAParams()

    if args.which == "nsd":
        model = f"final_subj{args.subject:02d}_pretrained_1sess_24bs"
        gal = torch.load(f"{DATA}/gallery_bigG_tokens.pt", weights_only=False).float()
        cv = torch.load(f"{DATA}/clipvoxels/clipvoxels_subj{args.subject}_{model}.pt",
                        weights_only=False)["clipvoxels"].float()
        q = cv.reshape(len(cv), -1)
        g = gal.reshape(len(gal), -1)
        q, g = q - q.mean(0, keepdim=True), g - g.mean(0, keepdim=True)
        basis = pca_basis(torch.cat([q, g]), args.k)
        query = (q @ basis).numpy().astype(np.float32)
        gallery = (g @ basis).numpy().astype(np.float32)
        trace(query, gallery, params, f"NSD subj{args.subject:02d} k={args.k}")
    else:
        from types import SimpleNamespace
        from shared import DEFAULT_SOURCE_RUN_DIR, encode_repetition_blocks, load_subject_context
        rt = SimpleNamespace(device=args.device, batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(DEFAULT_SOURCE_RUN_DIR, rt, args.subject, average=False)
        blocks, gallery, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[20],
                                                      seed=1000 + args.subject)
        trace(blocks[0].astype(np.float32), gallery, params, f"EEG sub{args.subject:02d}")


if __name__ == "__main__":
    main()
