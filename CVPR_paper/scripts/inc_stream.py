"""Incremental transductive calibration: every label-free method re-fitted on the growing unlabeled buffer.

Gallery = the 200 test images (or a random subset of --K). A query is the average of q repetitions (q x 0.2 s of
stimulus). Scenarios set how often each gallery image is queried (B = 80/q queries available per image):
  uniform        every image B times
  skew           long-tailed counts from B down to 1 (exponential profile, imbalance ratio B)
  partial        a random half of the gallery B times, the other half never
Queries arrive in random order; at ~40 geometrically spaced buffer sizes every method is re-fitted on the whole buffer
and the buffer is decoded (top1/top5; new1 = top-1 on the queries that arrived since the previous refit).
Hyper-parameters are the cross-fold ones of the batch test (GALA: --params_csv; TransCLIP / EM-Dirichlet: --trans_csv).
usage: inc_stream.py --cache_dir C --params_csv P --output_csv O [--trans_csv T] [--subjects ...] [--methods ...]
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
import gala  # noqa: E402
from label_free import csls as csls_plain, score_deploy  # noqa: E402
from shared import cosine_scores, fit_saw_transform  # noqa: E402
from stream_grid import subject_params  # noqa: E402

import torch  # noqa: E402

if not torch.cuda.is_available():  # TransCLIP_utils hard-codes .cuda(); run with CUDA_VISIBLE_DEVICES=
    torch.Tensor.cuda = lambda self, *a, **k: self

# partial: half of the gallery never queried; partial25: a quarter; skew25: a quarter never queried, skewed counts on the rest
SCENARIOS = ('uniform', 'skew', 'partial', 'partial25', 'skew25')
_SATTC = {}
# GALA (gala.fit defaults: prior of 2d observations at the zero-shot decoder, cross-fold rho / tau / gamma, 16 steps, CSLS k = 3),
# its ablations, and the earlier post-hoc damping for comparison
GALA = {'gala': {}, 'gala_d': {'prior': 1}, 'gala_noshrink': {'shrink': False}, 'gala_damp': {'shrink': False, 'damp': True}, 'gala_nowhiten': {'whiten_on': False},
        'gala_nocsls': {'use_csls': False}, 'gala_1step': {'steps': 1}, 'gala_norot': {'steps': 0}, 'gala_softmax': {'sinkhorn': False},
        # 2026-11-16 (inc_causal.py only): column masses = query frequencies estimated from the buffer (pseudo-count 1);
        # warm start: one iteration per update from the rotation of the previous update
        'gala_freq1': {'freq': 1.0}, 'gala_warm1': {'steps': 1, 'warm': True},
        # candidate default of 2026-11-16 (inc_causal.py only): no Sinkhorn (row softmax of the CSLS scores), one iteration
        # per update from the rotation of the previous update; its ablations; and the same from scratch with 16 iterations
        'gala_sw': {'sinkhorn': False, 'steps': 1, 'warm': True},
        'gala_sw_nowhiten': {'sinkhorn': False, 'steps': 1, 'warm': True, 'whiten_on': False},
        'gala_sw_nocsls': {'sinkhorn': False, 'steps': 1, 'warm': True, 'use_csls': False},
        'gala_sw_noprior': {'sinkhorn': False, 'steps': 1, 'warm': True, 'shrink': False},
        'gala_sw_sinkhorn': {'steps': 1, 'warm': True}, 'gala_s_norot': {'sinkhorn': False, 'steps': 0}, 'gala_s16': {'sinkhorn': False},
        # relaxed column constraint with warm start: column step to the power 0.5; column masses estimated from the buffer
        'gala_sw_cp50': {'steps': 1, 'warm': True, 'col_power': 0.5}, 'gala_sw_freq1': {'steps': 1, 'warm': True, 'freq': 1.0},
        'gala_sw_freq01': {'steps': 1, 'warm': True, 'freq': 0.1},  # pseudo-count 0.1 instead of 1 (2026-11-19 check)
        # ablations of gala_sw_freq1 (from scratch with 16 iterations = gala_freq1)
        'gala_swf_nowhiten': {'steps': 1, 'warm': True, 'freq': 1.0, 'whiten_on': False},
        'gala_swf_nocsls': {'steps': 1, 'warm': True, 'freq': 1.0, 'use_csls': False},
        'gala_swf_noprior': {'steps': 1, 'warm': True, 'freq': 1.0, 'shrink': False}}


def sattc(x, y, g, logits=False):
    """SATTC's official evaluate_model on the buffer (as sattc_on_embeddings.py ours --no_gallery_whiten, the variant
    the paper reports). Needs PYTHONPATH=/nasbrain/p20fores/SATTC-CVPR2026/.deps. Returns (top1, top5), or with
    logits=True the (t, K) scores its accuracies are computed from (inc_causal.py)."""
    import copy
    if not _SATTC:
        import sattc_on_embeddings as W
        cfg = W.sattc_config()
        cfg.global_auto_whiten = False
        _SATTC.update(W=W, cfg=cfg)
    W = _SATTC['W']
    q, gt = torch.as_tensor(x).float(), torch.as_tensor(g).float()
    yt = torch.as_tensor(y)
    loader = [(q, yt, None, gt[yt], None, gt[yt])]  # per-query aligned features: only their loss reads them
    cfg = copy.deepcopy(_SATTC['cfg'])
    cfg.diag_dump_keep_post = logits  # their switch that keeps the final score matrix on the config
    _, res, _ = W.R.evaluate_model('sub-00', W.PassThrough(), loader, torch.device('cpu'), gt, gt, [len(g)], cfg)
    if logits:
        return cfg._last_eval_diag['csls_logits'].numpy()
    return 100 * res[len(g)]['top1'], 100 * res[len(g)]['top5']


def make_stream(blocks, scenario, rng):
    """blocks (B, K, d) -> queries (N, d), labels (N,), in arrival order."""
    B, K = blocks.shape[:2]
    present = rng.permutation(K)[: {'partial': K // 2, 'partial25': K - K // 4, 'skew25': K - K // 4}.get(scenario, K)]
    n = np.zeros(K, int)
    n[present] = np.rint(B ** (1 - np.arange(len(present)) / (len(present) - 1))) if 'skew' in scenario else B
    idx = np.array([(b, j) for j in range(K) for b in rng.permutation(B)[: n[j]]])
    idx = idx[rng.permutation(len(idx))]
    return blocks[idx[:, 0], idx[:, 1]], idx[:, 1]


def hungarian(x, g):
    """Balanced one-to-one control: every gallery image may receive ceil(t / K) queries."""
    s = cosine_scores(x, g)
    m = -(-len(x) // len(g))
    r, c = linear_sum_assignment(-np.tile(s, (1, m)))
    out = s.copy()
    out[r, c % len(g)] += 10
    return out


def methods(names, x, y, g, p, trans, K):
    """name -> (t, K) scores of the buffer x under that method fitted on (x, g)."""
    t = len(x)
    kw = dict(rho=p.saw_shrink, tau=p.sinkhorn_tau, gamma=p.soft_procrustes_power, steps=p.soft_procrustes_steps, k=p.csls_k)
    out = {}
    for m in names:
        if m == 'plain':
            out[m] = cosine_scores(x, g)
        elif m == 'csls':
            out[m] = csls_plain(x, g, 3)
        elif m == 'hungarian':
            out[m] = hungarian(x, g)
        elif m == 'score':
            out[m] = score_deploy(x, g, k=5, rho=0.05)  # 2026-11-10: the settings of our SCORE reproduction (52.35 in batch on its own encoder; paper values k=10, rho=0.1 give 50.8)
        elif m == 'sattc':
            out[m] = sattc(x, y, g)
        elif m in ('transclip', 'em_dirichlet'):
            from module.transductive import infer
            front, cfg = trans[m]
            xin = fit_saw_transform(x, p)[0] if front == 'saw' else x
            try:
                out[m] = infer(xin, g, m, cfg)
            except (RuntimeError, ValueError, IndexError):  # official solvers need a minimum buffer: zero-shot below it
                out[m] = cosine_scores(x, g)
            if not np.isfinite(out[m]).all():
                out[m] = cosine_scores(x, g)
        elif m in GALA:
            out[m] = gala.fit(x, g, **{**kw, **GALA[m]})(x)
        else:
            raise KeyError(m)
    return out


def trans_configs(path, sid):
    if not path:
        return {}
    r = pd.read_csv(path)
    out = {}
    for m in ('transclip', 'em_dirichlet'):
        row = r[(r.subject == sid) & (r.method == m)].iloc[0]
        cfg = {k: (float(v) if '.' in v else int(v)) for k, v in (kv.split('=') for kv in row.config.split())}
        if m == 'transclip':
            cfg['device'] = 'cpu'
        out[m] = (row.front, cfg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--trans_csv', default='')
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--K', nargs='+', type=int, default=[200])
    ap.add_argument('--reps', nargs='+', type=int, default=[10])
    ap.add_argument('--scenarios', nargs='+', default=list(SCENARIOS))
    ap.add_argument('--orders', type=int, default=3)
    ap.add_argument('--n_refits', type=int, default=40)
    ap.add_argument('--methods', nargs='+', default=['plain', 'score', 'gala_noprior', 'gala'])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.output_csv)), exist_ok=True)
    rows = []
    for sid in a.subjects:
        z = np.load(f'{a.cache_dir}/sub-{sid:02d}.npz')
        p, trans = subject_params(a.params_csv, sid), trans_configs(a.trans_csv, sid)
        for K in a.K:
            keep = np.sort(np.random.default_rng(500 + sid).choice(200, K, replace=False))
            g = z['g'][keep]
            for q in a.reps:
                blocks = z[f'q{q}'][:, keep]
                for sc in a.scenarios:
                    for o in range(a.orders):
                        x, y = make_stream(blocks, sc, np.random.default_rng([sid, o, SCENARIOS.index(sc)]))
                        prev = 0
                        for t in np.unique(np.rint(np.geomspace(5, len(x), a.n_refits)).astype(int)):
                            for m, s in methods(a.methods, x[:t], y[:t], g, p, trans, K).items():
                                if isinstance(s, tuple):  # SATTC's own evaluation returns accuracies only
                                    rows.append((sid, K, q, sc, o, t, m, s[0], s[1], np.nan))
                                    continue
                                top = np.argsort(-s, axis=1)[:, :5]
                                hit = top == y[:t, None]
                                rows.append((sid, K, q, sc, o, t, m, 100 * hit[:, 0].mean(), 100 * hit.any(1).mean(),
                                             100 * hit[prev:, 0].mean()))
                            prev = t
                    print(f'sub {sid} K {K} q {q} {sc} done', flush=True)
        pd.DataFrame(rows, columns=['subject', 'K', 'q', 'scenario', 'order', 't', 'method', 'top1', 'top5', 'new1']).to_csv(a.output_csv, index=False)


if __name__ == '__main__':
    main()
