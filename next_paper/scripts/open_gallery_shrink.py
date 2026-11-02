"""Follow-up of open_gallery_diag.py: is SCORE robust to distractors because it barely trusts its correspondences?
Reads the embedding cache of open_gallery_diag.py. Top-1 per (subject, D, method):
  score_mm      SCORE's moment matching + CSLS(10), no rotation
  score_rho0    SCORE without the identity shrinkage of its Procrustes step (rho = 0)
  gala_norot    GALA's whitening + CSLS, no rotation
  gala_1step    GALA with a single Sinkhorn-Procrustes step
  gala_shr{l}   GALA whose every Procrustes step is shrunk towards the identity like SCORE's: svd(C + l ||C||_2 I)
usage: open_gallery_shrink.py --seed 3301
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, D_LIST, R, plan  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from module.util import apply_orthogonal_map  # noqa: E402
from label_free import csls, score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def shrunk_procrustes(t, gal, a, power, lam):
    w = np.power(np.clip(a, 0, None), power)
    c = t.T @ w @ gal
    u, _, vt = np.linalg.svd(c + lam * np.linalg.norm(c, 2) * np.eye(len(c)))
    return (u @ vt).astype(np.float32)


def gala(q, gal, p, lam=0.0, steps=None):
    t, _ = fit_saw_transform(q, p)
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    for _ in range(steps or max(1, int(p.soft_procrustes_steps))):
        a = plan(s, p, 'bal')
        tn = t / np.linalg.norm(t, axis=1, keepdims=True) if p.soft_procrustes_normalize_inputs else t
        gn = gal / np.linalg.norm(gal, axis=1, keepdims=True) if p.soft_procrustes_normalize_inputs else gal
        t = apply_orthogonal_map(t, shrunk_procrustes(tn, gn, a, p.soft_procrustes_power, lam))
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/shrink_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = z['q'], z['g'], z['d']
        for D in (0, 200, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            qm = (q - q.mean(0)) / (q.std(0) + 1e-6) * (gal.std(0) + 1e-6) + gal.mean(0)
            t, _ = fit_saw_transform(q, p)
            res = {'score_mm': csls(qm, gal, 10),
                   'score_rho0': score_deploy(q, gal, rho=0.0),
                   'gala_norot': score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k),
                   'gala_1step': gala(q, gal, p, steps=1)}
            for lam in (0.1, 1.0):
                res[f'gala_shr{lam}'] = gala(q, gal, p, lam=lam)
            for m, s in res.items():
                w.writerow([sid, D, m, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
