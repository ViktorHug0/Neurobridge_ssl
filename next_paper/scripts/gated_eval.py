"""Agreement-gated expert fusion (AGF) for partial galleries: robustness evaluation. No knowledge of D, N/M or coverage.
  G     = GALA scores (SAW start, balanced Sinkhorn-Procrustes loop, cross-fold params)
  Cc    = conservative expert: moment matching of the queries to the gallery, every Procrustes step shrunk to the
          identity with lambda = 1
  a     = fraction of queries whose top-1 is the same under G and Cc (label-free)
  AGF_k = a^k z(G) + z(Cc)   (z = per-query z-score); k = 2 chosen on the ATM training-distractor grid
Settings (top-1 over the queries present):
  train   all 200 queries, gallery = 200 test images + D training-concept distractors (paper's R3)
  novel   same with distractors from the 165 held-out (never-target) concepts (ValCon encoders only)
  cover   only U of the 200 test images are queried (U = 200, 100, 50, 25); gallery = the 200 test images + D
Reference: score (our SCORE), robust (r = 1 - N/M, needs N/M).
usage: gated_eval.py --arm atm128_vc --seed 3301 [--settings train novel cover]
"""
import argparse, csv, dataclasses, os, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R, embeddings  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from novel_distractors import pools  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def agf_parts(q, gal, p):
    x = nrm(fit_saw_transform(q, p)[0])
    g = loop(x, gal, p)[0]
    c = loop(nrm(mm(q, gal)), gal, p, lam=1.0)[0]
    return x, g, c, float((g.argmax(1) == c.argmax(1)).mean())


extra = {}


def lsq_acc(scores):
    """Label-free accuracies of K experts from pairwise top-1 agreement: log a_ij = log p_i + log p_j (least squares)."""
    tops = [s.argmax(1) for s in scores]
    rows, rhs = [], []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            r = np.zeros(len(scores))
            r[[i, j]] = 1
            rows.append(r)
            rhs.append(np.log(max(float((tops[i] == tops[j]).mean()), 1e-3)))
    return np.clip(np.exp(np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]), 1e-3, 1.0)


def gal_index_true(y, cols):
    """Queries whose Hungarian-matched gallery column is their own image (gallery rows 0..199 = the test images)."""
    return cols == y


def evaluate(q, y, gal, p):
    x, g, c, a = agf_parts(q, gal, p)
    r = 1 - len(q) / len(gal)
    sc = score_deploy(q, gal)
    c10 = loop(nrm(mm(q, gal)), gal, dataclasses.replace(p, csls_k=10), lam=1.0)[0]
    out = {'plain': score_features(q, gal), 'score': sc, 'gala': g, 'cons': c, 'cons10': c10,
           'robust_NM': loop(nrm((1 - r) * x + r * nrm(mm(q, gal))), gal, p, lam=r)[0]}
    for k in (1, 2, 3):
        out[f'agf{k}'] = a ** k * zrow(g) + zrow(c)
    for name, cc in (('score', sc), ('cons10', c10)):
        ac = float((g.argmax(1) == cc.argmax(1)).mean())
        out[f'agf2_{name}'] = ac ** 2 * zrow(g) + zrow(cc)
    out['agf2_mix'] = a ** 2 * zrow(g) + (zrow(c) + zrow(sc)) / 2
    # label-free expert accuracies from pairwise top-1 agreement (agreement_ij ~ p_i p_j): triplet estimates
    E = {'G': g, 'C': c, 'S': sc}
    top = {k: v.argmax(1) for k, v in E.items()}
    ag = lambda i, j: max(float((top[i] == top[j]).mean()), 1e-3)
    est = {'G': np.sqrt(ag('G', 'C') * ag('G', 'S') / ag('C', 'S')), 'C': np.sqrt(ag('G', 'C') * ag('C', 'S') / ag('G', 'S')),
           'S': np.sqrt(ag('G', 'S') * ag('C', 'S') / ag('G', 'C'))}
    est = {k: float(np.clip(v, 1e-3, 0.999)) for k, v in est.items()}
    for k in est:
        extra[f'phat_{k}'] = est[k]
    for name, f in (('p', lambda v: v), ('p2', lambda v: v ** 2), ('logit', lambda v: np.log(v / (1 - v)) + np.log(len(gal) - 1))):
        out[f'tri_{name}'] = sum(f(est[k]) * zrow(E[k]) for k in E)
    # generalisation to K experts: log a_ij = log p_i + log p_j, least squares over all pairs (no SCORE in 'own*')
    mz = score_features(nrm(mm(q, gal)), gal, use_csls=True, csls_k=10)  # correspondence-free expert
    pool = {'G': g, 'C': c, 'S': sc, 'C10': c10, 'M': mz}
    for name, keys in (('own3', ('G', 'C', 'M')), ('own4', ('G', 'C', 'C10', 'M')), ('own3b', ('G', 'C', 'C10')),
                       ('full4', ('G', 'C', 'S', 'M'))):
        out[f'acef_{name}'] = sum(w_k * zrow(pool[k]) for k, w_k in zip(keys, lsq_acc([pool[k] for k in keys])))
    # bound for the given correspondences: rotation fitted only on the TRUE pairs among ACEF's own Hungarian matches
    f = out['tri_p']
    cols = linear_sum_assignment(-f)[1]
    ok = gal_index_true(y, cols)
    extra['n_true_pairs'] = ok.sum() / 100  # stored x100 below
    if ok.sum() >= 2:
        u, _, vt = np.linalg.svd(x[ok].T @ nrm(gal)[cols[ok]])
        out['bound_true_pairs'] = score_features(nrm(x @ (u @ vt)), gal, use_csls=p.use_csls, csls_k=p.csls_k)
        extra['n_true_pairs'] = ok.sum() / 100  # stored x100 below
    cz = (zrow(c) + zrow(sc)) / 2  # conservative pair, equal weights (their errors are correlated: no estimate)
    pg = est['G']
    out['gateP'] = pg * zrow(g) + cz
    out['gateP2'] = 2 * pg * zrow(g) + cz
    out['gateOdds'] = pg / (1 - pg) * zrow(g) + cz
    return {m: topk_acc(s, y)[1] for m, s in out.items()}, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--settings', nargs='+', default=['train', 'novel', 'cover'])
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    f = open(f'{C}/agf_{a.arm}_s{a.seed}.csv', 'w', newline='')  # v2: + SCORE / k=10 conservative experts
    w = csv.writer(f)
    w.writerow(['subject', 'setting', 'U', 'D', 'method', 'top1', 'agree'])
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        embeddings(a.arm, a.seed, sid)  # builds q, g, d if missing
        z = pools(a.arm, a.seed, sid)  # adds the novel / train pools if missing
        q, g, dis = (z[k].astype(np.float32) for k in ('q', 'g', 'd'))
        for st in a.settings:
            grid = [(200, D) for D in (0, 200, 400, 800, 1654)] if st == 'train' else \
                   [(200, D) for D in (200, 400, 800, 1650)] if st == 'novel' else \
                   [(U, D) for U in (100, 50, 25) for D in (0, 400)]
            for U, D in grid:
                pool = dis if st != 'novel' else z['novel'].astype(np.float32)
                gal = np.concatenate([g, pool[:D]], 0)
                keep = np.sort(np.random.default_rng(100 * sid + U).choice(200, U, replace=False)) if U < 200 else np.arange(200)
                acc, ag = evaluate(q[keep], keep, gal, p)
                for m, v in acc.items():
                    w.writerow([sid, st, U, D, m, v, ag])
                for m, v in extra.items():  # label-free accuracy estimates (in %) under method 'phat_*'
                    w.writerow([sid, st, U, D, m, 100 * v, ag])
        f.flush()
        print(sid, flush=True)


if __name__ == '__main__':
    main()
