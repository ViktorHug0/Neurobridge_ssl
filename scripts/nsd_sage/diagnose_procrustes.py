"""Why does Procrustes hurt MindEye2 retrieval? Isolate the cause.

Facts to explain (subj01, 1-hour model, 1000-way top-1):
  baseline 65.6 | centred baseline 74.8 | oracle Procrustes under a shared token block 40.2

Candidate causes, and what separates them:

  C1  The shared-token-block factorisation R_flat = I_256 (x) R_token is too restrictive.
      The published metric is a cosine on the FLATTENED 425,984-dim vector, so the rotation
      should live there; a shared 1664x1664 block is a tiny subgroup of O(425984). Test by
      fitting an UNCONSTRAINED rotation inside a PCA subspace of the flattened space, which
      is also the faithful analogue of the EEG setting (one vector per trial, moderate dim).

  C2  Procrustes optimises the wrong thing. It maximises sum_i <q_i R, g_i>, the ABSOLUTE
      matched similarity, whereas retrieval depends on matched similarity RELATIVE to
      unmatched. With a dominant shared component (here the mean vector norm is 23.4 against
      a token norm of 33.6) the rotation can align everything to that component, raising
      matched and unmatched alike and destroying the margin. Test by tracking matched and
      unmatched similarity separately, not just top-1.

Since n=1000 < d, the data spans at most 1000 dimensions, so the PCA basis is exact via the
Gram trick rather than approximate.

Usage:  python diagnose_procrustes.py --subj 1
"""
import argparse

import numpy as np
import torch

DATA = "/nasbrain/p20fores/mindeye_data"


def pca_basis(x, k, device):
    """Orthonormal basis for the top-k directions of x (n, d) with n << d, via the Gram matrix."""
    # only the (n, n) Gram matrix goes to float64; casting the (n, d) data would need 6.8 GB
    g = (x @ x.T).double()                      # (n, n)
    evals, evecs = torch.linalg.eigh(g)
    idx = torch.argsort(evals, descending=True)[:k]
    evals, evecs = evals[idx].clamp_min(1e-8), evecs[:, idx]
    basis = (x.T @ evecs.float()) / evals.sqrt().float()  # (d, k), orthonormal columns
    return basis.to(device)


def retrieval_stats(q, g):
    """top-1 plus the matched/unmatched similarity split that separates C2 from C1."""
    qn = torch.nn.functional.normalize(q, dim=1)
    gn = torch.nn.functional.normalize(g, dim=1)
    sim = qn @ gn.T
    n = len(sim)
    top1 = (sim.argmax(1) == torch.arange(n, device=sim.device)).float().mean().item() * 100
    matched = sim.diagonal().mean().item()
    off = sim.sum().item() - sim.diagonal().sum().item()
    unmatched = off / (n * n - n)
    return top1, matched, unmatched


def procrustes(q, g, assignment=None):
    """Unconstrained orthogonal map taking q toward g (oracle pairing when assignment is None)."""
    cross = q.T @ g if assignment is None else q.T @ (assignment @ g)
    u, _, vt = torch.linalg.svd(cross.double(), full_matrices=False)
    return (u @ vt).float()


def angle_from_identity(r):
    d = len(r)
    return float(np.degrees(np.arccos(np.clip((torch.trace(r) / d).item(), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256, 512, 999])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = args.model_name or f"final_subj{args.subj:02d}_pretrained_1sess_24bs"
    dev = args.device

    gal = torch.load(f"{DATA}/gallery_bigG_tokens.pt", weights_only=False).float()
    cv = torch.load(f"{DATA}/clipvoxels/clipvoxels_subj{args.subj}_{model}.pt",
                    weights_only=False)["clipvoxels"].float()
    q0 = cv.reshape(len(cv), -1).to(dev)
    g0 = gal.reshape(len(gal), -1).to(dev)
    print(f"subj{args.subj:02d} flattened {tuple(q0.shape)}")

    for centred in (False, True):
        q, g = (q0, g0)
        tag = "raw    "
        if centred:
            # centre both on their own mean, the dominant shared component
            q = q0 - q0.mean(0, keepdim=True)
            g = g0 - g0.mean(0, keepdim=True)
            tag = "centred"
        t, m, u = retrieval_stats(q, g)
        print(f"\n{tag} baseline           top1 {t:5.1f}   matched {m:+.4f}  unmatched {u:+.4f}  margin {m-u:+.4f}")

        for k in args.ks:
            basis = pca_basis(torch.cat([q, g]), k, dev)   # shared subspace for both sides
            qk, gk = q @ basis, g @ basis
            r = procrustes(qk, gk)                          # ORACLE pairing: best case for SAGE
            # lift back: rotate inside the subspace, leave the orthogonal complement untouched
            q_rot = q - (qk @ basis.T) + ((qk @ r) @ basis.T)
            t2, m2, u2 = retrieval_stats(q_rot, g)
            var = (qk.pow(2).sum() / q.pow(2).sum()).item()
            print(f"  k={k:<4} var{var*100:5.1f}%  rot {angle_from_identity(r):5.1f}deg  "
                  f"top1 {t2:5.1f} ({t2-t:+5.1f})   matched {m2:+.4f}  unmatched {u2:+.4f}  "
                  f"margin {m2-u2:+.4f}")


if __name__ == "__main__":
    main()
