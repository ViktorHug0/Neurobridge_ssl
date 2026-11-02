"""Match stability across repetition sub-averages (label-free, D-agnostic) to find ACEF's correct pairs.
Each query = average of 80 repetitions; its 4 disjoint 20-repetition blocks are encoded separately (cache key 'blk').
Every expert's FITTED transform (SAW + rotation for GALA and the conservative expert, SCORE's map for the landmark
expert) is applied to each block; block scores are fused with the same ACEF weights.
  stability_i = fraction of the 4 blocks whose fused top-1 is the query's ACEF Hungarian match
Reports: AUC of stability (and of the fused margin) for correct vs wrong pairs, and top-1 of
  acef            ACEF (3 experts)
  R_stab{g}       one Procrustes (SAW space) on ACEF's Hungarian pairs weighted by stability^g, scored on the gallery
  acef4_stab{g}   ACEF re-fused with R_stab{g} as a 4th expert
  acef_blk        ACEF scores + mean of the 4 block fused scores (z-scored)
  bound           rotation on the true pairs among ACEF's Hungarian matches (uses labels)
usage: stability.py --seed 3301 [--arm atm128_vc]
"""
import argparse, csv, os, sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from gated_eval import lsq_acc  # noqa: E402
from shared import apply_saw_transform, encode_repetition_blocks, fit_saw_transform, load_subject_context, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from label_free import csls, score_fit  # noqa: E402
from stream_grid import subject_params  # noqa: E402

EPS = 1e-6


def blocks(arm, seed, sid):
    f = f'{C}/cache/{arm}_s{seed}/sub-{sid:02d}.npz'
    z = dict(np.load(f))
    if 'blk' not in z:
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(f'{R}/results/things_eeg/cvpr/loso/{arm}/seed{seed}', rt, sid, average=False)
        b, _, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[20] * 4, seed=3300)
        z['blk'] = np.stack(b).astype(np.float32)
        np.savez(f, **z)
    return z


def recover(a, b):
    """Orthogonal map m with nrm(a @ m) = b (exact for GALA's loop output)."""
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ vt


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/stab_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'D', 'method', 'value'])
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = blocks(a.arm, a.seed, sid)
        q, g, dis, blk = (z[k].astype(np.float32) for k in ('q', 'g', 'd', 'blk'))
        x, saw = fit_saw_transform(q, p)
        x = nrm(x)
        y = np.arange(200)
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, dis[:D]], 0)
            gn = nrm(gal)
            # experts on the 80-repetition queries, keeping their fitted transforms
            sG, tG, _ = loop(x, gal, p)
            mu, sd, gm, gs = q.mean(0), q.std(0), gal.mean(0), gal.std(0)
            mmf = lambda v: nrm((v - mu) / (sd + EPS) * (gs + EPS) + gm)
            sC, tC, _ = loop(mmf(q), gal, p, lam=1.0)
            lmap = score_fit(q, gal)
            sL = csls(lmap(q), gal, 10)
            wts = lsq_acc([sG, sC, sL])
            f = sum(wk * zrow(s) for wk, s in zip(wts, (sG, sC, sL)))
            rG, rC = recover(x, tG), recover(mmf(q), tC)
            fb = []
            for b in blk:
                bG = score_features(nrm(apply_saw_transform(b, saw) @ rG), gal, use_csls=p.use_csls, csls_k=p.csls_k)
                bC = score_features(nrm(mmf(b) @ rC), gal, use_csls=p.use_csls, csls_k=p.csls_k)
                bL = csls(lmap(b), gal, 10)
                fb.append(sum(wk * zrow(s) for wk, s in zip(wts, (bG, bC, bL))))
            cols = linear_sum_assignment(-f)[1]
            ok = cols == y
            stab = np.mean([fbk.argmax(1) == cols for fbk in fb], 0)
            zf = zrow(f)
            srt = np.sort(zf, 1)
            top = zf[y, cols]
            marg = top - np.where(srt[:, -1] == top, srt[:, -2], srt[:, -1])
            w.writerow([sid, D, 'true_pairs', ok.sum()])
            w.writerow([sid, D, 'auc_stability', auc(stab[ok], stab[~ok])])
            w.writerow([sid, D, 'auc_margin', auc(marg[ok], marg[~ok])])
            w.writerow([sid, D, 'auc_stab_plus_margin', auc((stab + marg / 10)[ok], (stab + marg / 10)[~ok])])
            w.writerow([sid, D, 'prec_stab_ge_0.75', 100 * ok[stab >= 0.75].mean() if (stab >= 0.75).any() else np.nan])
            w.writerow([sid, D, 'n_stab_ge_0.75', (stab >= 0.75).sum()])
            sc = lambda r: score_features(nrm(x @ r), gal, use_csls=p.use_csls, csls_k=p.csls_k)
            res = {'acef': f, 'bound': sc(recover(x[ok], gn[cols][ok])) if ok.sum() > 1 else f,
                   'acef_blk': zrow(f) + zrow(np.mean(fb, 0))}
            for gmm in (1, 2, 4):
                ws = stab ** gmm + 1e-6
                rs = sc(recover(x * ws[:, None], gn[cols]))
                res[f'R_stab{gmm}'] = rs
                res[f'acef4_stab{gmm}'] = sum(wk * zrow(s) for wk, s in zip(lsq_acc([sG, sC, sL, rs]), (sG, sC, sL, rs)))
            for k, s in res.items():
                w.writerow([sid, D, k, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
