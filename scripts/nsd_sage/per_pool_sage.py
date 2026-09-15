"""Fit SAGE independently on each 300-candidate pool, as MindEye2's protocol defines the task.

The first port fitted the calibration once on the full 1000-way gallery and evaluated the
frozen transform on the 30 random pools. That was wrong twice over. Each pool is already a
balanced bijective problem (300 queries, 300 candidates, every answer present exactly once),
so nothing prevented fitting on it; what the pools lack is a gallery shared ACROSS loops,
which SAGE never required. And fitting on all 1000 let the transform see the 700 items outside
each pool, a strictly larger transductive set than the 300-way task defines, so that number
was not measuring MindEye2's task.

Here everything is per pool: PCA basis, SAW, CSLS, Sinkhorn and Procrustes see only that
pool's 300 queries and 300 candidates. A pool spans at most 598 dimensions, so the rank has to
be re-selected for this regime; the rank chosen for the 1000-way fit does not carry over.

  --sweep  retrieval against PCA rank
  --diag   plain / Sinkhorn-assignment / SAGE / shuffled-assignment, which distinguishes a
           loop that bootstrapped from a rotation that merely encodes the plan it was given

Usage:  python per_pool_sage.py --sweep --ks 64 128 256 512
        python per_pool_sage.py --diag  --ks 256 512
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL/scripts/things_eeg/tta_rebuttal")
sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL/scripts/nsd_sage")

from module.util import (apply_orthogonal_map, fit_soft_assignment_procrustes,  # noqa: E402
                         sinkhorn_normalize)
from shared import TTAParams, fit_saw_transform  # noqa: E402
from nsd_cv_ablation import load_full, directions, _score, SUBJECTS  # noqa: E402

# refined CV selection; only the rank is re-chosen for the 300-query regime
PARAMS = dict(saw_shrink=0.96, csls_k=1, sinkhorn_tau=0.01, soft_procrustes_power=1.2)


def pool_basis(q, g, k):
    x = torch.cat([q - q.mean(0, keepdim=True), g - g.mean(0, keepdim=True)])
    ev, evec = torch.linalg.eigh((x @ x.T).double())
    idx = torch.argsort(ev, descending=True)[:k]
    ev, evec = ev[idx].clamp_min(1e-8), evec[:, idx]
    return (x.T @ evec.float()) / ev.sqrt().float()


def top1(m):
    return float((m.argmax(axis=1) == np.arange(len(m))).mean()) * 100


def calibrate(q, g, k, p, shuffle_rng=None):
    """Returns (plain, first-assignment accuracy, final score matrix), all within the pool."""
    basis = pool_basis(q, g, k)
    cur = ((q - q.mean(0, keepdim=True)) @ basis).numpy().astype(np.float32)
    gk = ((g - g.mean(0, keepdim=True)) @ basis).numpy().astype(np.float32)
    plain = top1(_score(cur, gk, False, p.csls_k))
    cur, _ = fit_saw_transform(cur, p)
    sc = _score(cur, gk, p.use_csls, p.csls_k)
    assign = None
    for _ in range(int(p.soft_procrustes_steps)):
        pl = sinkhorn_normalize(sc, tau=p.sinkhorn_tau, num_iters=p.sinkhorn_iters,
                                col_mass=p.sinkhorn_col_mass)
        if assign is None:
            assign = top1(pl)
        if shuffle_rng is not None:
            pl = pl[:, shuffle_rng.permutation(pl.shape[1])]
        m = fit_soft_assignment_procrustes(
            cur, gk, pl, power=p.soft_procrustes_power,
            normalize_inputs=p.soft_procrustes_normalize_inputs)
        if m is None:
            break
        cur = apply_orthogonal_map(cur, m)
        sc = _score(cur, gk, p.use_csls, p.csls_k)
    # score in the full space against the untouched gallery, as every other arm does
    gproj = (g @ basis).numpy().astype(np.float32)
    gn = gproj / np.clip(g.norm(dim=1).numpy()[:, None], 1e-8, None)
    qn = cur / np.clip(np.linalg.norm(cur, axis=1, keepdims=True), 1e-8, None)
    return plain, assign, _score(qn, gn, p.use_csls, p.csls_k)


def pools(q, seed=42, loops=30, size=300):
    rng = np.random.RandomState(seed)
    for _ in range(loops):
        yield rng.choice(len(q), size=size, replace=False)


def full_space_arm(q, g, centre, csls, p):
    """Arms that need no subspace: baseline, CSLS only, centring only.

    CSLS operates on the pool's similarity matrix and centring is a single vector subtraction,
    so both run at the full 425,984 dimensions with no basis.
    """
    qq = q - q.mean(0, keepdim=True) if centre else q
    qn = torch.nn.functional.normalize(qq, dim=1)
    gn = torch.nn.functional.normalize(g, dim=1)
    sim = (qn @ gn.T).numpy().astype(np.float32)
    from module.util import csls_scores
    return csls_scores(sim, k=p.csls_k) if csls else sim


def ablation_arm(q, g, k, p, whiten, csls, align, project, basis=None):
    if not project:
        return full_space_arm(q, g, centre=False, csls=csls, p=p)
    if not whiten and not align:
        return full_space_arm(q, g, centre=True, csls=csls, p=p)
    pp = TTAParams(**{**PARAMS, "use_csls": csls})
    if basis is None:
        basis = pool_basis(q, g, k)
    cur = ((q - q.mean(0, keepdim=True)) @ basis).numpy().astype(np.float32)
    gk = ((g - g.mean(0, keepdim=True)) @ basis).numpy().astype(np.float32)
    if whiten:
        cur, _ = fit_saw_transform(cur, pp)
    if align:
        sc = _score(cur, gk, pp.use_csls, pp.csls_k)
        for _ in range(int(pp.soft_procrustes_steps)):
            pl = sinkhorn_normalize(sc, tau=pp.sinkhorn_tau, num_iters=pp.sinkhorn_iters,
                                    col_mass=pp.sinkhorn_col_mass)
            m = fit_soft_assignment_procrustes(
                cur, gk, pl, power=pp.soft_procrustes_power,
                normalize_inputs=pp.soft_procrustes_normalize_inputs)
            if m is None:
                break
            cur = apply_orthogonal_map(cur, m)
            sc = _score(cur, gk, pp.use_csls, pp.csls_k)
    gproj = (g @ basis).numpy().astype(np.float32)
    gn = gproj / np.clip(g.norm(dim=1).numpy()[:, None], 1e-8, None)
    qn = cur / np.clip(np.linalg.norm(cur, axis=1, keepdims=True), 1e-8, None)
    return _score(qn, gn, pp.use_csls, pp.csls_k)


ARMS = [("full SAGE", True, True, True, True),
        ("no whitening", False, True, True, True),
        ("no CSLS", True, False, True, True),
        ("no alignment", True, True, False, True),
        ("CSLS only (no PCA)", False, True, False, False),
        ("centring only (no PCA)", False, False, False, True),
        ("none (published baseline)", False, False, False, False)]


def stage_ablation(k, p, subjects=None, loops=30, out=None):
    """Per-pool ablation. 95% CI over the 30 pools, paired t-test over the 4 subjects.

    Looping arms outermost would recompute each pool's PCA basis once per arm, and that basis
    is the dominant cost (a 600x600 eigendecomposition plus a 425,984 x 600 by 600 x k matmul).
    The basis depends only on (subject, pool), so the loops are ordered subject, pool, arm and
    it is computed once and shared. On a contended 2-CPU host this is the difference between
    hours and tens of minutes.
    """
    from scipy import stats
    print(f"per-pool ablation at PCA rank {k}, 30 pools x 300, {PARAMS}\n", flush=True)
    names = [a[0] for a in ARMS]
    store = {n: [] for n in names}
    sems = {n: [] for n in names}
    subjects = subjects or SUBJECTS
    for s in subjects:
        q, g = load_full(s)
        acc = {n: [] for n in names}
        for i in pools(q, loops=loops):
            qp, gp = q[i], g[i]
            basis = pool_basis(qp, gp, k)          # once per pool, reused by every arm
            for name, wh, cs, al, pr in ARMS:
                acc[name].append(directions(ablation_arm(qp, gp, k, p, wh, cs, al, pr, basis)))
        for n in names:
            store[n].append(np.mean(acc[n], axis=0))
            sems[n].append(np.std(acc[n], axis=0) / np.sqrt(len(acc[n])))
        print(f"  subj{s} done", flush=True)
        del q, g
    print(f"\n{'arm':>28} {'image':>14} {'brain':>14}")
    for n in names:
        m, sem = np.mean(store[n], axis=0), np.mean(sems[n], axis=0)
        store[n] = np.array(store[n])
        print(f"{n:>28} {m[0]:8.1f}+-{1.96*sem[0]:<4.1f} {m[1]:8.1f}+-{1.96*sem[1]:<4.1f}",
              flush=True)
    base = store["none (published baseline)"]
    print(f"\n{'arm':>28} {'d image':>9} {'p':>8} {'d brain':>9} {'p':>8}")
    for name, rows in store.items():
        if name.startswith("none"):
            continue
        out = []
        for j in (0, 1):
            t, pv = stats.ttest_rel(rows[:, j], base[:, j])
            out += [float(rows[:, j].mean() - base[:, j].mean()), float(pv)]
        print(f"{name:>28} {out[0]:>+9.1f} {out[1]:>8.4f} {out[2]:>+9.1f} {out[3]:>8.4f}")
    print("\npaper Table 1, MindEye2 (1 hour): image 79.0  brain 57.4", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--loops", type=int, default=30)
    ap.add_argument("--out", default=None, help="npz of per-subject per-arm means, for chunked runs")
    args = ap.parse_args()
    p = TTAParams(**PARAMS)
    t0 = time.time()

    if args.sweep:
        print(f"per-pool SAGE, 30 pools x 300, {PARAMS}", flush=True)
        print(f"{'k':>5} " + " ".join(f"{'s'+str(s):>12}" for s in SUBJECTS) + f" {'mean':>12}")
        for k in args.ks:
            per_subj = []
            for s in SUBJECTS:
                q, g = load_full(s)
                acc = [directions(calibrate(q[i], g[i], k, p)[2]) for i in pools(q)]
                per_subj.append(np.mean(acc, axis=0))
                del q, g
            m = np.mean(per_subj, axis=0)
            cells = " ".join(f"{a[0]:5.1f}/{a[1]:<6.1f}" for a in per_subj)
            print(f"{k:>5} {cells} {m[0]:5.1f}/{m[1]:<6.1f}  ({(time.time()-t0)/60:.0f} min)", flush=True)

    if args.ablation:
        stage_ablation(args.ks[0], p, args.subjects, args.loops, args.out)

    if args.diag:
        print(f"\nper-pool diagnostic: does the loop bootstrap or read back its assignment?", flush=True)
        for k in args.ks:
            rows = []
            for s in SUBJECTS:
                q, g = load_full(s)
                acc = []
                for i in pools(q):
                    pl, a0, sim = calibrate(q[i], g[i], k, p)
                    _, _, sh = calibrate(q[i], g[i], k, p, shuffle_rng=np.random.RandomState(0))
                    acc.append((pl, a0, top1(sim), top1(sh)))
                rows.append(np.mean(acc, axis=0))
                del q, g
            m = np.mean(rows, axis=0)
            print(f"k={k}: plain {m[0]:5.1f}   assign {m[1]:5.1f}   sage {m[2]:5.1f}   "
                  f"shuffled {m[3]:5.1f}   (sage-assign {m[2]-m[1]:+5.1f})", flush=True)
        print("chance for 300-way = 0.33%", flush=True)
    print("PER_POOL_DONE", flush=True)


if __name__ == "__main__":
    main()
