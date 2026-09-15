"""Is the PCA-subspace Procrustes gain real, or is the rotation just memorising its input?

The unconstrained fit lifts subj01 from 65.6 to 96.2 top-1 at k=999 under the ORACLE pairing.
Two things must be ruled out before that counts as SAGE transferring:

  A. Circularity. At k=999 with n=1000 correspondences the rotation has ~k^2/2 free parameters
     against n constraints, so it can encode whatever assignment it was handed. Re-scoring
     afterwards would then just read that assignment back. The test: compare post-rotation
     top-1 against the accuracy of the assignment used to fit it. A rotation that only
     memorises lands AT the assignment's accuracy; a rotation that captures real shared
     geometry lands ABOVE it (and the assignment could otherwise have been used directly,
     with no Procrustes at all).

  B. Degeneracy. A shuffled assignment must NOT improve retrieval. If it does, the metric is
     measuring the fitting procedure rather than the data.

Arms per k: oracle (upper bound) | sinkhorn (label-free, the real method) | shuffled (null).

Usage:  python diagnose_assignment.py --subj 1 --device cpu
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
from module.util import csls_scores, sinkhorn_normalize  # noqa: E402

DATA = "/nasbrain/p20fores/mindeye_data"


def pca_basis(x, k):
    g = (x @ x.T).double()
    evals, evecs = torch.linalg.eigh(g)
    idx = torch.argsort(evals, descending=True)[:k]
    evals, evecs = evals[idx].clamp_min(1e-8), evecs[:, idx]
    return (x.T @ evecs.float()) / evals.sqrt().float()


def top1(q, g):
    qn = torch.nn.functional.normalize(q, dim=1)
    gn = torch.nn.functional.normalize(g, dim=1)
    sim = qn @ gn.T
    return (sim.argmax(1) == torch.arange(len(sim))).float().mean().item() * 100, sim


def procrustes(q, g, assignment):
    cross = q.T @ (assignment @ g)
    u, _, vt = torch.linalg.svd(cross.double(), full_matrices=False)
    return (u @ vt).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 256, 512, 999])
    ap.add_argument("--centre", action="store_true", default=True)
    args = ap.parse_args()
    model = args.model_name or f"final_subj{args.subj:02d}_pretrained_1sess_24bs"

    gal = torch.load(f"{DATA}/gallery_bigG_tokens.pt", weights_only=False).float()
    cv = torch.load(f"{DATA}/clipvoxels/clipvoxels_subj{args.subj}_{model}.pt",
                    weights_only=False)["clipvoxels"].float()
    q = cv.reshape(len(cv), -1)
    g = gal.reshape(len(gal), -1)
    if args.centre:
        q, g = q - q.mean(0, keepdim=True), g - g.mean(0, keepdim=True)
    n = len(q)

    base, sim = top1(q, g)
    # the label-free assignment SAGE actually has access to
    plan = sinkhorn_normalize(csls_scores(sim.numpy(), k=12), tau=0.05, num_iters=20)
    plan_acc = float((plan.argmax(1) == np.arange(n)).mean()) * 100
    print(f"subj{args.subj:02d}  baseline top1 {base:.1f}   sinkhorn assignment top1 {plan_acc:.1f}")
    print("  (a rotation that merely memorises its assignment scores AT the assignment's accuracy)\n")

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    shuffled = np.zeros((n, n), dtype=np.float32)
    shuffled[np.arange(n), perm] = 1.0

    arms = {"oracle": np.eye(n, dtype=np.float32),
            "sinkhorn": plan.astype(np.float32),
            "shuffled": shuffled}

    print(f"{'k':>5}  " + "  ".join(f"{a:>10}" for a in arms))
    for k in args.ks:
        basis = pca_basis(torch.cat([q, g]), k)
        qk, gk = q @ basis, g @ basis
        row = []
        for name, a in arms.items():
            r = procrustes(qk, gk, torch.from_numpy(a))
            q_rot = q - (qk @ basis.T) + ((qk @ r) @ basis.T)
            row.append(top1(q_rot, g)[0])
        print(f"{k:>5}  " + "  ".join(f"{v:10.1f}" for v in row))

    print(f"\nreference: baseline {base:.1f}   sinkhorn assignment {plan_acc:.1f}")
    print("verdict: sinkhorn arm must beat BOTH to be a real Procrustes gain;")
    print("         shuffled arm must stay near or below baseline.")


if __name__ == "__main__":
    main()
