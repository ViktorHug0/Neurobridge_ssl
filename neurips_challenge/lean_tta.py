"""cv4 score of an ensemble under the lean test-time adaptation, exactly as a submission's predict() would run it
(score_track1.batch_transform_wknn on each 64-trial batch of the tta_cache.py stream), with the official
subject-aggregated top-5 (official_metrics.py, verbatim from the competition repo), averaged over the folds' people.
    lean_tta.py [--folds 1 2 3 4] [recipe ...]       default: the best ensemble (OPTX5, AUG_LOG: 44.25)
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.official_metrics import group_means, topk_accuracy  # noqa: E402
from neurips_challenge.score_track1 import batch_transform_wknn, fit_fingerprint_wccn  # noqa: E402
from neurips_challenge.tta_cache import DATA, FOLDS, OUT  # noqa: E402

OPTX5 = ['reve_d6_optx_things', 'rmix_reg_optx_things', 'sqf_dbn_optx_things', 'tsc_dbn40_bn64_optx_things', 'reve_scratch_optx_things']
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def official(y_pred, img, subj, G):
    """per-subject official top-5: predictions averaged per (subject, image), cosine ranking against the gallery"""
    keys, pred_sa = group_means(y_pred, np.stack([subj, img], 1))
    n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    return {int(s): topk_accuracy(n(pred_sa[keys[:, 0] == s]) @ n(G).T, keys[keys[:, 0] == s, 1], k=5) for s in np.unique(subj)}


@torch.no_grad()
def fold_scores(k, members, out=OUT):
    c = np.load(f'{out}/fold{k}.npz')
    P = [torch.as_tensor(np.load(f'{out}/{m}_f{k}.npy'), device=DEV) for m in members]
    F = torch.as_tensor(c['F'], device=DEV)
    wccn = {key: v.to(DEV) for key, v in fit_fingerprint_wccn(DATA, [s for s in range(1, 21) if s not in FOLDS[k]]).items()}
    y = torch.cat([batch_transform_wknn([p[b:b + 64] for p in P], F[b:b + 64], wccn) for b in range(0, len(F), 64)])
    return official(y.double().cpu().numpy(), c['img'], c['subj'], c['G'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('members', nargs='*', default=OPTX5)
    ap.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4])
    a = ap.parse_args()
    res = {}
    for k in a.folds:
        sc = fold_scores(k, a.members); res.update(sc)
        print(f'fold {k}: {100 * np.mean(list(sc.values())):.2f}', flush=True)
    print(f'mean over {len(res)} people: {100 * np.mean(list(res.values())):.2f}')
