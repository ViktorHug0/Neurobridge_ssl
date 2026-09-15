"""SAGE-TTA applied post-hoc to MindEye2's NSD retrieval embeddings.

The claim under test is that SAGE's calibration stage transfers to another modality and can be
bolted onto an existing fMRI retrieval pipeline without retraining it. Nothing here touches
MindEye2: it consumes the retrieval-submodule outputs produced by extract_clipvoxels.py and
the same OpenCLIP gallery its own evaluation uses.

Shapes drive the design. MindEye2 retrieval embeddings are 256 tokens x 1664 dims, and the
published metric flattens them to 425,984 before the cosine. A rotation in that flattened
space is a 425,984^2 matrix, which is hopeless. But the token axis is a sample axis, not a
feature axis: fitting one 1664x1664 rotation shared across token positions uses 1000 x 256 =
256,000 token pairs, i.e. 154 pairs per dimension. For comparison SAGE on THINGS-EEG fits a
512x512 rotation from 200 samples, which is underdetermined -- this port is far better
conditioned than the setting the method was designed for.

So the stages split by which axis they live on:
  SAW, Procrustes  -> token space (1664), applied identically at every token position
  CSLS, Sinkhorn   -> the 1000x1000 trial similarity matrix

Sinkhorn needs one shared gallery with a balanced assignment, which the published protocol's
redrawn 300-candidate pools do not provide. The rotation is therefore fitted once on the full
1000-way gallery (one query per image, a permutation) and then evaluated two ways: full
1000-way, and their unmodified 30 x 300 protocol using the already-fitted rotation.

Usage:
  python apply_sage_tta.py --subjects 1 2 5 7 --setting 1sess
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
from module.util import csls_scores, sinkhorn_normalize, subject_adaptive_whiten  # noqa: E402

DATA = "/nasbrain/p20fores/mindeye_data"


def flat_normalize(x):
    """(N,T,D) -> (N,T*D) L2-normalised, matching the published metric."""
    f = x.reshape(len(x), -1)
    return f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-8, None)


def token_procrustes(query, candidate, assignment):
    """One 1664x1664 rotation shared across token positions.

    The weighted cross-covariance that fit_soft_assignment_procrustes builds as
    `query.T @ weights @ candidate` generalises over the token axis to
    `sum_t Q[:,t,:].T @ A @ C[:,t,:]`, which is the same SVD problem in token space.
    """
    moved = np.einsum("ij,jtd->itd", assignment, candidate, optimize=True)
    cross = np.einsum("itd,ite->de", query, moved, optimize=True)
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return (u @ vt).astype(np.float32)


def chord_blend(rotation, alpha):
    """Orthogonal projection of (1-a)I + aR; equals R at a=1 and identity at a=0."""
    if alpha >= 1.0:
        return rotation
    mixed = (1.0 - alpha) * np.eye(len(rotation), dtype=np.float32) + alpha * rotation
    u, _, vt = np.linalg.svd(mixed, full_matrices=False)
    return (u @ vt).astype(np.float32)


def top1_full(query, gallery):
    sim = flat_normalize(query) @ flat_normalize(gallery).T
    return float((sim.argmax(axis=1) == np.arange(len(sim))).mean()) * 100


def top1_300(query, gallery, loops=30, pool=300, seed=42):
    """Their protocol, unchanged, on whatever embeddings we hand it."""
    rng = np.random.RandomState(seed)
    qn, gn = flat_normalize(query), flat_normalize(gallery)
    hits = []
    for _ in range(loops):
        s = rng.choice(len(gn), size=pool, replace=False)
        sim = qn[s] @ gn[s].T
        hits.append((sim.argmax(axis=1) == np.arange(pool)).mean())
    return float(np.mean(hits)) * 100


def rotation_report(rotation):
    """How far the fitted map is from doing nothing.

    If the queries are already aligned to the gallery there is no residual rotation to find,
    a correct fit returns approximately the identity, and alpha should barely matter. A large
    mean plane angle when the baseline retrieval is already good means Procrustes is fitting
    something other than a genuine misalignment.
    """
    d = len(rotation)
    cos_mean = np.clip((np.trace(rotation) / d), -1.0, 1.0)
    return dict(mean_angle_deg=float(np.degrees(np.arccos(cos_mean))),
                frob_from_identity=float(np.linalg.norm(rotation - np.eye(d)) / np.sqrt(2 * d)))


def sage_tta(query, gallery, alpha=1.0, whiten=True, csls_k=12, tau=0.05, iters=20,
             report=None):
    """SAW -> CSLS -> Sinkhorn -> Procrustes, returning the rotated queries."""
    n, t, d = query.shape
    q = query
    if whiten:
        flat = q.reshape(n * t, d)
        # one held-out subject, so SAW degenerates to a single ZCA over its own tokens
        w = subject_adaptive_whiten(flat, np.zeros(len(flat), dtype=np.int64), normalize=False)
        q = w.reshape(n, t, d).astype(np.float32)

    sim = flat_normalize(q) @ flat_normalize(gallery).T
    sim = csls_scores(sim, k=csls_k)
    assignment = sinkhorn_normalize(sim, tau=tau, num_iters=iters)
    rotation = token_procrustes(q, gallery, assignment)
    if report is not None:
        report.update(rotation_report(rotation))
        # how much of the Sinkhorn plan sits on the true pairing
        report["assignment_top1"] = float((assignment.argmax(axis=1) == np.arange(len(assignment))).mean())
    return np.einsum("itd,de->ite", q, chord_blend(rotation, alpha), optimize=True).astype(np.float32)


def _mild_rotation(d, strength, rng):
    """Rotation near the identity: exp of a small skew-symmetric matrix."""
    from scipy.linalg import expm
    a = rng.randn(d, d).astype(np.float64)
    return expm(strength * (a - a.T) / np.linalg.norm(a - a.T)).astype(np.float32)


def _self_check():
    """Two separate properties, because they fail for different reasons.

    1. token_procrustes must recover a planted rotation exactly when handed the true
       assignment. This isolates the einsum generalisation over the token axis -- the only
       piece of maths here that is not already in module/util.py.
    2. The full pipeline must improve retrieval when the baseline is informative. It is a
       refinement of an existing similarity, so it cannot bootstrap from chance: a full
       random rotation leaves CSLS and Sinkhorn nothing to work with, and SAGE correctly
       fails there. The realistic case is a partially-degraded similarity.
    """
    rng = np.random.RandomState(0)
    n, t, d = 200, 4, 16
    gallery = rng.randn(n, t, d).astype(np.float32)

    rot, _ = np.linalg.qr(rng.randn(d, d))
    rot = rot.astype(np.float32)
    query = np.einsum("itd,de->ite", gallery, rot, optimize=True)
    fitted = token_procrustes(query, gallery, np.eye(n, dtype=np.float32))
    recovered = np.einsum("itd,de->ite", query, fitted, optimize=True)
    assert np.allclose(fitted, rot.T, atol=1e-4), "procrustes did not recover the planted rotation"
    assert top1_full(recovered, gallery) > 99.0, "recovered rotation does not restore retrieval"

    # Partial rotation plus noise. A full rotation is not a usable test: it drives the
    # baseline to chance, where CSLS and Sinkhorn have nothing to latch onto and SAGE
    # correctly recovers nothing. The cliff is sharp -- at these settings beta=0.5 leaves a
    # 51% baseline that SAGE lifts to 91%, while beta=0.7 collapses it to 0% and stays there.
    drifted = (np.einsum("itd,de->ite", gallery, chord_blend(rot, 0.5), optimize=True)
               + 1.5 * rng.randn(n, t, d).astype(np.float32))
    before = top1_full(drifted, gallery)
    after = top1_full(sage_tta(drifted, gallery, whiten=False), gallery)
    assert 20.0 < before < 80.0, f"self-check needs an informative baseline, got {before:.1f}%"
    assert after > before + 10.0, f"SAGE barely moved an informative baseline: {before:.1f} -> {after:.1f}"

    assert np.allclose(chord_blend(rot, 0.0), np.eye(d), atol=1e-5), "alpha=0 must be identity"
    assert np.allclose(chord_blend(rot, 1.0), rot, atol=1e-6), "alpha=1 must be the full rotation"
    print(f"self-check ok (exact recovery; drifted baseline {before:.1f}% -> {after:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--setting", default="1sess")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--no_whiten", action="store_true")
    ap.add_argument("--self_check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
        return

    gallery = torch.load(f"{DATA}/gallery_bigG_tokens.pt", weights_only=False).numpy().astype(np.float32)
    rows = {}
    for subj in args.subjects:
        model = f"final_subj{subj:02d}_pretrained_{args.setting}_24bs"
        d = torch.load(f"{DATA}/clipvoxels/clipvoxels_subj{subj}_{model}.pt", weights_only=False)
        query = d["clipvoxels"].float().numpy()
        base = (top1_full(query, gallery), top1_300(query, gallery))
        print(f"subj{subj:02d} baseline   1000-way {base[0]:5.1f}   300-way {base[1]:5.1f}", flush=True)
        rows.setdefault("baseline", []).append(base)
        for a in args.alphas:
            out = sage_tta(query, gallery, alpha=a, whiten=not args.no_whiten)
            got = (top1_full(out, gallery), top1_300(out, gallery))
            print(f"         SAGE a={a:<4} 1000-way {got[0]:5.1f}   300-way {got[1]:5.1f}", flush=True)
            rows.setdefault(f"a={a}", []).append(got)

    print(f"\n===== mean over {len(args.subjects)} subjects ({args.setting}) =====")
    b1k, b300 = np.mean([r[0] for r in rows["baseline"]]), np.mean([r[1] for r in rows["baseline"]])
    print(f"baseline   1000-way {b1k:5.1f}   300-way {b300:5.1f}")
    for k, v in rows.items():
        if k == "baseline":
            continue
        m1k, m300 = np.mean([r[0] for r in v]), np.mean([r[1] for r in v])
        print(f"SAGE {k:<8} 1000-way {m1k:5.1f} ({m1k-b1k:+5.1f})   300-way {m300:5.1f} ({m300-b300:+5.1f})")


if __name__ == "__main__":
    main()
