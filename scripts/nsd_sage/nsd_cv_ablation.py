"""Cross-validated SAGE-TTA on NSD, with a component ablation, in MindEye2's own protocol.

Three requirements drive the design.

1. Baselines must reproduce the published table. Every arm is a transform of the QUERY side
   only, scored in the full flattened 425,984-dim space against an untouched gallery, using
   MindEye2's own metric. The no-TTA row is therefore literally their computation and returns
   79.0 / 57.4 rather than something merely close to it.

2. No hyperparameter may see the subject it is reported on. Parameters are chosen by
   leave-one-subject-out CV over the four subjects the paper evaluates: for held-out subject
   s the grid is scored on the other three, and only the winning configuration is run on s.
   Four held-out numbers, averaged.

3. The ablation mirrors Table 4 of the paper: remove one component at a time from
   {whitening, CSLS, geometric alignment}, at the CV-selected settings.

Retrieval directions follow utils.batchwise_cosine_similarity, whose trailing .T decides which
axis topk ranks over: with sim[i,j] = cos(query_i, gallery_j), image retrieval is the
COLUMN-wise argmax and brain retrieval the ROW-wise argmax. Getting this backwards silently
swaps the two columns, and one of them lands near the published value by coincidence.

Whitening and the rotation need a d x d covariance, impossible at 425,984 dims, so they are
fitted in a shared PCA subspace of the flattened space and lifted back. CSLS needs only the
similarity matrix and runs at full dimension. The 300-candidate protocol is evaluated by
slicing the full 1000 x 1000 cosine matrix, which is exactly equivalent and far cheaper.

Usage:
  python nsd_cv_ablation.py --stage cv        # grid + LOSO selection
  python nsd_cv_ablation.py --stage ablation  # remove-one table at the selected params
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL/scripts/things_eeg/tta_rebuttal")

from module.util import (apply_orthogonal_map, csls_scores, fit_soft_assignment_procrustes,  # noqa: E402
                         sinkhorn_normalize)
from shared import TTAParams, fit_saw_transform  # noqa: E402

DATA = "/nasbrain/p20fores/mindeye_data"
CACHE = f"{DATA}/pca_cache"
SUBJECTS = [1, 2, 5, 7]
SELECTED = f"{DATA}/cv_selected.json"


# --------------------------------------------------------------------------- data

def load_full(subj, setting="1sess"):
    model = f"final_subj{subj:02d}_pretrained_{setting}_24bs"
    gal = torch.load(f"{DATA}/gallery_bigG_tokens.pt", weights_only=False).float()
    cv = torch.load(f"{DATA}/clipvoxels/clipvoxels_subj{subj}_{model}.pt",
                    weights_only=False)["clipvoxels"].float()
    return cv.reshape(len(cv), -1), gal.reshape(len(gal), -1)


_PCA = {}


def pca_parts(subj, k, setting="1sess"):
    """Cached (basis, query mean) for the shared PCA subspace of the flattened space."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/subj{subj}_{setting}_k{k}.pt"
    if path in _PCA:
        return _PCA[path]
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        _PCA[path] = (d["basis"], d["mu_q"], d["mu_g"])
        return _PCA[path]
    q, g = load_full(subj, setting)
    mu_q, mu_g = q.mean(0, keepdim=True), g.mean(0, keepdim=True)
    x = torch.cat([q - mu_q, g - mu_g])
    gram = (x @ x.T).double()
    evals, evecs = torch.linalg.eigh(gram)
    idx = torch.argsort(evals, descending=True)[:k]
    evals, evecs = evals[idx].clamp_min(1e-8), evecs[:, idx]
    basis = (x.T @ evecs.float()) / evals.sqrt().float()
    torch.save({"basis": basis, "mu_q": mu_q, "mu_g": mu_g}, path)
    _PCA[path] = (basis, mu_q, mu_g)
    return _PCA[path]


# --------------------------------------------------------------------------- metric

def cosine_matrix(q, g):
    qn = torch.nn.functional.normalize(q, dim=1)
    gn = torch.nn.functional.normalize(g, dim=1)
    return (qn @ gn.T).numpy().astype(np.float32)


def directions(sim):
    """(image retrieval, brain retrieval) as MindEye2 defines them."""
    n = len(sim)
    brain = float((sim.argmax(axis=1) == np.arange(n)).mean()) * 100   # per query, best gallery
    image = float((sim.argmax(axis=0) == np.arange(n)).mean()) * 100   # per gallery, best query
    return image, brain


def evaluate(sim, use_csls, csls_k, loops=30, pool=300, seed=42):
    """Full-gallery and MindEye2's 30 x 300 protocol, from one precomputed cosine matrix."""
    full = directions(csls_scores(sim, k=csls_k) if use_csls else sim)
    rng = np.random.RandomState(seed)
    acc = []
    for _ in range(loops):
        s = rng.choice(len(sim), size=pool, replace=False)
        sub = sim[np.ix_(s, s)]
        acc.append(directions(csls_scores(sub, k=csls_k) if use_csls else sub))
    m, sd = np.mean(acc, axis=0), np.std(acc, axis=0)
    # MindEye2 reports a 95% CI over the 30 pools; keep the same spread statistic
    return dict(img300=float(m[0]), brain300=float(m[1]), img1k=full[0], brain1k=full[1],
                img300_sem=float(sd[0] / np.sqrt(len(acc))),
                brain300_sem=float(sd[1] / np.sqrt(len(acc))))


# --------------------------------------------------------------------------- method

def subspace_similarity(cur, g, basis):
    """Exact full-space cosine between the lifted query and the untouched gallery, in k dims.

    The lifted query is cur @ basis.T. Because basis has orthonormal columns,
    <cur @ basis.T, g_j> = <cur, g_j @ basis> and ||cur @ basis.T|| = ||cur||, so the whole
    425,984-dim cosine reduces to a k-dim inner product against the gallery's projection,
    divided by the gallery's FULL-space norm. Identical to lifting and is ~1000x cheaper,
    which is what makes a 720-fit grid tractable.
    """
    gn = _gallery_side(g, basis)
    qn = cur / np.clip(np.linalg.norm(cur, axis=1, keepdims=True), 1e-8, None)
    return (qn @ gn.T).astype(np.float32)


_GAL = {}


def _gallery_side(g, basis):
    """(g @ basis) / ||g||_full, cached: it is fixed per subspace and dominates the cost."""
    key = id(basis)
    if key not in _GAL:
        gproj = (g @ basis).numpy().astype(np.float32)
        gnorm = g.norm(dim=1).numpy().astype(np.float32)
        _GAL[key] = gproj / np.clip(gnorm[:, None], 1e-8, None)
    return _GAL[key]


def transform_query(q, g, params, k, whiten, align, project=None):
    """Query-side transform. Returns (cur, basis), or (None, None) for the untouched query.

    `project` isolates the PCA step. Whitening and the rotation cannot run at 425,984 dims, so
    they require it, which means those ablation rows carry a step the baseline row does not.
    On subj01 at k=512 the projection alone moves image retrieval 94.2 -> 89.2 and brain
    retrieval 78.3 -> 91.5, so it is not neutral and gets its own row rather than hiding
    inside the others. The EEG ablation in the paper has no such confound: whitening there is
    a plain d x d operation on a 512-dim embedding.
    """
    project = (whiten or align) if project is None else project
    if not project:
        return None, None
    basis, mu_q, mu_g = params["_pca"]
    qk = ((q - mu_q) @ basis).numpy().astype(np.float32)
    gk = ((g - mu_g) @ basis).numpy().astype(np.float32)

    cur = qk
    if whiten:
        cur, _ = fit_saw_transform(cur, params["tta"])
    if align:
        p = params["tta"]
        scores = _score(cur, gk, p.use_csls, p.csls_k)
        for _ in range(int(p.soft_procrustes_steps)):
            plan = sinkhorn_normalize(scores, tau=p.sinkhorn_tau, num_iters=p.sinkhorn_iters,
                                      col_mass=p.sinkhorn_col_mass)
            step = fit_soft_assignment_procrustes(
                cur, gk, plan, power=p.soft_procrustes_power,
                normalize_inputs=p.soft_procrustes_normalize_inputs)
            if step is None:
                break
            cur = apply_orthogonal_map(cur, step)
            scores = _score(cur, gk, p.use_csls, p.csls_k)
    return cur, basis


def _score(a, b, use_csls, csls_k):
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    s = (an @ bn.T).astype(np.float32)
    return csls_scores(s, k=csls_k) if use_csls else s


def run_arm(subj, cfg, whiten, csls, align, setting="1sess", project=None):
    q, g = load_full(subj, setting)
    params = {"tta": TTAParams(saw_shrink=cfg["shrink"], csls_k=cfg["csls_k"],
                               sinkhorn_tau=cfg["tau"], soft_procrustes_power=cfg["power"],
                               use_csls=csls),
              "_pca": pca_parts(subj, cfg["k"], setting)}
    cur, basis = transform_query(q, g, params, cfg["k"], whiten, align, project)
    sim = _baseline_sim(subj, setting, q, g) if cur is None else subspace_similarity(cur, g, basis)
    return evaluate(sim, use_csls=csls, csls_k=cfg["csls_k"])


_BASE_SIM = {}


def _baseline_sim(subj, setting, q, g):
    """Untouched query vs untouched gallery: the published computation. Cached, it is reused
    by every ablation arm that leaves the query alone."""
    key = (subj, setting)
    if key not in _BASE_SIM:
        _BASE_SIM[key] = cosine_matrix(q, g)
    return _BASE_SIM[key]


# --------------------------------------------------------------------------- stages

GRID = dict(tau=[0.005, 0.01, 0.02, 0.05, 0.1], shrink=[0.2, 0.5, 0.94],
            csls_k=[3, 12, 30], power=[1.0, 1.2], k=[256, 512])


def stage_cv(args):
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]
    print(f"grid: {len(configs)} configs x {len(SUBJECTS)} subjects", flush=True)

    table = {}
    for subj in SUBJECTS:
        for i, cfg in enumerate(configs):
            r = run_arm(subj, cfg, whiten=True, csls=True, align=True)
            table[(subj, i)] = r["img300"] + r["brain300"]
            if i % 20 == 0:
                print(f"  subj{subj} {i}/{len(configs)}", flush=True)
        # bases are ~0.9 GB each; keeping every (subject, k) resident would exhaust RAM
        _PCA.clear(); _GAL.clear(); _BASE_SIM.clear()
    np.save(f"{DATA}/cv_grid.npy", np.array([[table[(s, i)] for i in range(len(configs))]
                                             for s in SUBJECTS]))

    picks = {}
    print(f"\n{'held-out':>9} {'chosen config':>60}")
    for si, subj in enumerate(SUBJECTS):
        others = [s for s in SUBJECTS if s != subj]
        mean = np.mean([[table[(s, i)] for i in range(len(configs))] for s in others], axis=0)
        best = int(np.argmax(mean))
        picks[subj] = configs[best]
        print(f"{subj:>9} {json.dumps(configs[best]):>60}")
    json.dump({str(k): v for k, v in picks.items()}, open(SELECTED, "w"), indent=1)
    print(f"\nwrote {SELECTED}")


def stage_ablation(args):
    picks = {int(k): v for k, v in json.load(open(SELECTED)).items()}
    arms = [("full SAGE", True, True, True, None),
            ("no whitening", False, True, True, None),
            ("no CSLS", True, False, True, None),
            ("no alignment", True, True, False, None),
            ("CSLS only (no PCA)", False, True, False, False),
            ("PCA projection only", False, False, False, True),
            ("none (published baseline)", False, False, False, False)]
    base_rows = None
    all_rows = {}
    print(f"{'arm':>28} {'img300':>14} {'brain300':>14} {'img1k':>7} {'brain1k':>8}")
    for name, wh, cs, al, pr in arms:
        rows = []
        for s in SUBJECTS:
            rows.append(run_arm(s, picks[s], wh, cs, al, project=pr))
            _PCA.clear(); _GAL.clear(); _BASE_SIM.clear()
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        if base_rows is None and name.startswith("none"):
            base_rows = rows
        print(f"{name:>28} {m['img300']:>8.1f}+-{1.96*m['img300_sem']:<4.1f} "
              f"{m['brain300']:>8.1f}+-{1.96*m['brain300_sem']:<4.1f} "
              f"{m['img1k']:>7.1f} {m['brain1k']:>8.1f}", flush=True)
        all_rows[name] = rows
    print("\npaper Table 1, MindEye2 (1 hour): image 79.0  brain 57.4")
    # paired across the 4 subjects, against the untouched-embedding row
    from scipy import stats as _st
    base = all_rows["none (published baseline)"]
    print(f"\n{'arm':>28} {'d img300':>10} {'p':>8} {'d brain300':>11} {'p':>8}")
    for name, rows in all_rows.items():
        if name.startswith("none"):
            continue
        out = []
        for key in ("img300", "brain300"):
            a = [r[key] for r in rows]; b = [r[key] for r in base]
            t, pv = _st.ttest_rel(a, b)
            out += [float(np.mean(a) - np.mean(b)), float(pv)]
        print(f"{name:>28} {out[0]:>+10.1f} {out[1]:>8.4f} {out[2]:>+11.1f} {out[3]:>8.4f}")
    print(f"(paired over n={len(SUBJECTS)} subjects; 4 subjects is what the benchmark offers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "ablation"], required=True)
    a = ap.parse_args()
    (stage_cv if a.stage == "cv" else stage_ablation)(a)
