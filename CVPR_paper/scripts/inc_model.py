"""Incremental protocol (inc_stream.py), model-level label-free adaptation baselines. They need the encoder and the raw
EEG, so the streams are rebuilt here from the same repetition blocks and the same random draws as inc_stream.py.
At every refit, each method re-adapts on ALL raw queries of the buffer, re-encodes the buffer, and decodes by cosine:
  adabn   AdaBN (Li et al. 2018): BatchNorm statistics re-estimated on the buffer
  tent    TENT (Wang et al. 2021): on the AdaBN statistics, BatchNorm affine parameters minimise the entropy of
          softmax(cos / 0.07) over the gallery, Adam lr 1e-3, 10 full-buffer steps
  ea      Euclidean alignment (He & Wu 2020) of the target to the source reference: X <- R_src^1/2 R_tgt^-1/2 X, with
          R_tgt the arithmetic mean of the spatial covariances of the buffer queries and R_src that of the source
          subjects' averaged test trials (no target label; the encoder is not retrained)
  ra      Riemannian re-centring (Zanini et al. 2018): the same with affine-invariant Riemannian means
--step S (adabn, tent): the causal protocol of inc_causal.py with a refit every S queries (a refit is a pass of the
whole buffer through the encoder): the queries t+1 .. t+S are encoded by the model adapted on the first t queries (the
first S by the unadapted model); one row per block of S queries, n = number of queries.
usage: SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth inc_model.py --source_run_dir R --output_csv O [--subjects ...]
"""
import argparse, os, sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import _test_image_matrix, build_test_dataset, cosine_scores, load_subject_context  # noqa: E402
from bci_baselines import adabn, encode, sqrtm, tent  # noqa: E402
from inc_stream import SCENARIOS, make_stream  # noqa: E402


def raw_blocks(ds, q, seed):
    """(80/q, 200, C, T) q-repetition averages: the blocks of shared.encode_repetition_blocks, before the encoder."""
    eeg = np.asarray(ds.eeg_data_list[0])[:, 0]  # (200, 80, C, T)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(eeg.shape[1]) for _ in range(len(eeg))])
    return np.stack([np.stack([eeg[j, perms[j, b * q:(b + 1) * q]].mean(0, dtype=np.float32) for j in range(len(eeg))])
                     for b in range(eeg.shape[1] // q)])


def covs(x):
    x = x - x.mean(-1, keepdims=True)
    return np.einsum('nct,ndt->ncd', x, x) / x.shape[-1]


def riemann_mean(C, iters=10):
    """Affine-invariant Riemannian (Karcher) mean of SPD matrices, fixed-point iteration from the arithmetic mean."""
    C = C + 1e-6 * np.trace(C, axis1=1, axis2=2).mean() / C.shape[1] * np.eye(C.shape[1])
    M = C.mean(0)
    for _ in range(iters):
        s, si = sqrtm(M), sqrtm(M, inv=True)
        w, v = np.linalg.eigh(si @ C @ si)
        T = np.einsum('nij,nj,nkj->ik', v, np.log(np.clip(w, 1e-12, None)), v) / len(C)
        w, v = np.linalg.eigh(T)
        M = s @ (v * np.exp(w)) @ v.T @ s
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--reps', nargs='+', type=int, default=[10])
    ap.add_argument('--scenarios', nargs='+', default=list(SCENARIOS))
    ap.add_argument('--orders', type=int, default=3)
    ap.add_argument('--n_refits', type=int, default=40)
    ap.add_argument('--step', type=int, default=0)
    ap.add_argument('--methods', nargs='+', default=['adabn', 'tent', 'ea', 'ra'])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.output_csv)), exist_ok=True)
    rows = []
    for sid in a.subjects:
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(a.source_run_dir), rt, sid, average=False)
        model, proj = mods['model'], mods['eeg_projector']
        model.eval()
        with torch.no_grad():
            g = mods['img_projector'](torch.tensor(_test_image_matrix(ds), dtype=torch.float32)).numpy()
        src = np.concatenate([covs(np.asarray(build_test_dataset(ea, s, average=False).eeg_data_list[0])[:, 0].mean(1).astype(np.float32))
                              for s in range(1, 11) if s != sid])
        ref = {'ea': sqrtm(src.mean(0)), 'ra': sqrtm(riemann_mean(src))}
        for q in a.reps:
            blocks = raw_blocks(ds, q, 1000 + sid)
            for sc in a.scenarios:
                for o in range(a.orders):
                    x, y = make_stream(blocks, sc, np.random.default_rng([sid, o, SCENARIOS.index(sc)]))
                    for m in a.methods if a.step else ():
                        for t in range(0, len(x), a.step):
                            mod = model if t == 0 else adabn(model, ea, x[:t], sid) if m == 'adabn' else tent(model, proj, ea, x[:t], g, sid)
                            new = slice(t, t + a.step)
                            hit = np.argsort(-cosine_scores(encode(mod, proj, ea, x[new], sid), g), axis=1)[:, :5] == y[new, None]
                            rows.append((sid, 200, q, sc, o, t + len(hit), m, 100 * hit[:, 0].mean(), 100 * hit.any(1).mean(), len(hit)))
                    if a.step:
                        continue
                    c = covs(x)
                    prev = 0
                    for t in np.unique(np.rint(np.geomspace(5, len(x), a.n_refits)).astype(int)):
                        xb = x[:t]
                        for m in a.methods:
                            if m == 'adabn':
                                e = encode(adabn(model, ea, xb, sid), proj, ea, xb, sid)
                            elif m == 'tent':
                                e = encode(tent(model, proj, ea, xb, g, sid), proj, ea, xb, sid)
                            else:
                                A = (ref[m] @ sqrtm(c[:t].mean(0) if m == 'ea' else riemann_mean(c[:t]), inv=True)).astype(np.float32)
                                e = encode(model, proj, ea, np.einsum('cd,ndt->nct', A, xb), sid)
                            hit = np.argsort(-cosine_scores(e, g), axis=1)[:, :5] == y[:t, None]
                            rows.append((sid, 200, q, sc, o, t, m, 100 * hit[:, 0].mean(), 100 * hit.any(1).mean(), 100 * hit[prev:, 0].mean()))
                        prev = t
                print(f'sub {sid} q {q} {sc} done', flush=True)
        pd.DataFrame(rows, columns=['subject', 'K', 'q', 'scenario', 'order', 't', 'method', 'top1', 'top5', 'n' if a.step else 'new1']).to_csv(a.output_csv, index=False)


if __name__ == '__main__':  # ponytail: self-check of the Karcher mean lives in the first call; run with --subjects 1 --orders 1 to smoke
    main()
