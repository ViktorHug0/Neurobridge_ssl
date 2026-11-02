"""Candidate fixes for GALA with distractors (R3), on the embedding cache of open_gallery_diag.py.
All label-free: N (queries) and M (candidates) are known at deployment, the support is not.
  gala             published loop (SAW whitening, CSLS, balanced Sinkhorn, Procrustes, 16 steps)
  gala_shr{l}      every Procrustes step shrunk towards the identity: svd(C + l ||C||_2 I)                  (B)
  gala_ratio{c}    shrinkage l = c (1 - N/M): exactly GALA when M = N                                     (B)
  mm_gala*         same, starting from SCORE's moment matching to the gallery instead of SAW               (A)
  prune{b}         self-pruning gallery: column caps w_j = N * mass_j^b / sum(mass^b), refreshed each step (C)
  sup_hung         support = rectangular Hungarian on the initial CSLS scores; GALA fitted on it, scored on all
  sup_mass         support = the N columns with the most plan mass after a pruning run; same
usage: open_gallery_methods.py --seed 3301 [--methods ...]
"""
import argparse, csv, dataclasses, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from module.util import apply_orthogonal_map  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def nrm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mm(q, gal, eps=1e-6):  # SCORE Eq. 3: per-dimension moment matching to the gallery
    return (q - q.mean(0)) / (q.std(0) + eps) * (gal.std(0) + eps) + gal.mean(0)


def capped(s, tau, iters, caps):
    k = np.exp((s - s.max(1, keepdims=True)) / max(tau, 1e-8))
    for _ in range(max(1, int(iters))):
        k /= k.sum(1, keepdims=True)
        k *= np.minimum(1.0, caps / np.clip(k.sum(0, keepdims=True), 1e-12, None))
    return k


def procrustes(t, gal, a, p, lam):
    if p.soft_procrustes_normalize_inputs:
        t, gal = nrm(t), nrm(gal)
    c = t.T @ np.power(np.clip(a, 0, None), p.soft_procrustes_power) @ gal
    u, _, vt = np.linalg.svd(c + lam * np.linalg.norm(c, 2) * np.eye(len(c)))
    return (u @ vt).astype(np.float32)


def loop(t, gal, p, lam=0.0, prune=None):
    """GALA steps (ii)-(v) from start t. prune = sharpening exponent b of the self-pruning column caps (None = balanced)."""
    n, m = len(t), len(gal)
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    caps = np.full((1, m), n / m)
    a = None
    for _ in range(max(1, int(p.soft_procrustes_steps))):
        if prune is None:
            a = damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0)
        else:
            a = capped(s, p.sinkhorn_tau, p.sinkhorn_iters, caps)
            w = a.sum(0) ** prune
            caps = np.minimum(1.0, n * w / w.sum())[None]
        t = apply_orthogonal_map(t, procrustes(t, gal, a, p, lam))
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return s, t, a


def support_fit(t0, gal, p, cols):
    """GALA fitted on the selected columns only, then scored against the whole gallery."""
    _, t, _ = loop(t0, gal[cols], p)
    return score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)


def methods(q, gal, p, which):
    n, m = len(q), len(gal)
    saw = fit_saw_transform(q, p)[0]
    mmq = nrm(mm(q, gal))
    out = {}
    for name in which:
        if name == 'gala':
            out[name] = loop(saw, gal, p)[0]
        elif name.startswith('gala_shr'):
            out[name] = loop(saw, gal, p, lam=float(name[8:]))[0]
        elif name.startswith('gala_ratio'):
            out[name] = loop(saw, gal, p, lam=float(name[10:]) * (1 - n / m))[0]
        elif name == 'mm_gala':
            out[name] = loop(mmq, gal, p)[0]
        elif name.startswith('mm_gala_ratio'):
            out[name] = loop(mmq, gal, p, lam=float(name[13:]) * (1 - n / m))[0]
        elif name.startswith('prune'):
            out[name] = loop(saw, gal, p, prune=float(name[5:]))[0]
        elif name.startswith('sawmm_ratio'):  # whiten (SAW), then match per-dimension moments to the gallery
            out[name] = loop(nrm(mm(saw, gal)), gal, p, lam=float(name[11:]) * (1 - n / m))[0]
        elif name.startswith('blend_ratio'):  # start = SAW at M = N, moving to the gallery moment matching as M grows
            r = 1 - n / m
            out[name] = loop(nrm((1 - r) * nrm(saw) + r * mmq), gal, p, lam=float(name[11:]) * r)[0]
        elif name.startswith('kscale'):  # CSLS neighbourhood grown with the gallery: k = csls_k * M / N
            pk = dataclasses.replace(p, csls_k=int(round(p.csls_k * m / n)))
            out[name] = loop(mmq, gal, pk, lam=float(name[6:]) * (1 - n / m))[0]
        elif name == 'sup_hung':
            s0 = score_features(saw, gal, use_csls=p.use_csls, csls_k=p.csls_k)
            out[name] = support_fit(saw, gal, p, np.sort(linear_sum_assignment(-s0)[1]))
        elif name == 'sup_mass':
            a = loop(saw, gal, p, prune=2.0)[2]
            out[name] = support_fit(saw, gal, p, np.sort(np.argsort(-a.sum(0))[:n]))
    return out


DEFAULT = ['gala', 'gala_shr0.1', 'gala_shr0.3', 'gala_shr1.0', 'gala_ratio0.3', 'gala_ratio1.0', 'gala_ratio3.0',
           'mm_gala', 'mm_gala_ratio1.0', 'prune1.0', 'prune2.0', 'prune4.0', 'sup_hung', 'sup_mass']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--methods', nargs='+', default=DEFAULT)
    ap.add_argument('--tag', default='methods')
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    f = open(f'{C}/{a.tag}_{a.arm}_s{a.seed}.csv', 'w', newline='')
    w = csv.writer(f)
    w.writerow(['subject', 'D', 'method', 'top1', 'top5'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = z['q'].astype(np.float32), z['g'].astype(np.float32), z['d'].astype(np.float32)
        for D in D_LIST:
            gal = np.concatenate([g, dis[:D]], 0)
            for name, s in methods(q, gal, p, a.methods).items():
                acc = topk_acc(s, y)
                w.writerow([sid, D, name, acc[1], acc[5]])
        f.flush()
        print(sid, flush=True)


if __name__ == '__main__':
    main()
