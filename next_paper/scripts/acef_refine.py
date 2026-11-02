"""ACEF refinement rounds (D-agnostic): fused scores -> soft plan -> new GALA rotation fitted from that plan -> refit
expert G' -> re-estimate weights -> re-fuse. Reports top-1 and the number of true Hungarian pairs per round.
Plan from fused z-scores: Sinkhorn with the GALA temperature on the standardized fused scores (balanced, as GALA),
or a row-softmax ('row'); the Procrustes fit is GALA's (power gamma), then the usual GALA loop continues from it.
usage: acef_refine.py --seed 3301 [--rounds 3]
"""
import argparse, csv, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm, procrustes  # noqa: E402
from agnostic import zrow  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def fuse(experts):
    w = lsq_acc(list(experts.values()))
    return sum(wk * zrow(s) for wk, s in zip(w, experts.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--rounds', type=int, default=3)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/refine_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'variant', 'round', 'top1', 'true_pairs'])
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = np.load(f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz')
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        x = nrm(fit_saw_transform(q, p)[0])
        y = np.arange(200)
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            base = {'C': loop(nrm(mm(q, gal)), gal, p, lam=1.0)[0], 'S': score_deploy(q, gal),
                    'M': score_features(nrm(mm(q, gal)), gal, use_csls=True, csls_k=10)}
            for variant, keys in (('with_landmark', ('C', 'S')), ('own', ('C', 'M'))):
                gs = loop(x, gal, p)[0]
                for r in range(a.rounds + 1):
                    f = fuse({'G': gs, **{k: base[k] for k in keys}})
                    tp = int((linear_sum_assignment(-f)[1] == y).sum())
                    w.writerow([sid, D, variant, r, topk_acc(f, y)[1], tp])
                    # refit GALA from the fused plan: one Procrustes on the plan, then the usual loop from there
                    plan = damped_assign(zrow(f) * float(gs.std(1).mean()), p.sinkhorn_tau, p.sinkhorn_iters, 1.0)  # fused z on GALA's score scale
                    t = nrm(x @ procrustes(x, gal, plan, p, 0.0))
                    gs = loop(t, gal, p)[0]
        print(sid, flush=True)


if __name__ == '__main__':
    main()
