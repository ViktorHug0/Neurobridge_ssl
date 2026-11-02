"""E3.4 transductive VLM baselines on regime R1 (200 x 200), same 80-repetition embeddings as r1_eval.py.

Heads from module/transductive (EM-Dirichlet / hard EM-Dirichlet / hard k-means via the official transductive-CLIP
solvers, TransCLIP via the official TransCLIP blocks, Sinkhorn, Hungarian). Each head gets a small grid (optionally
after SAW whitening, the shrink being SAGE's cross-fold one for that subject); the reported config is chosen
cross-fold exactly like SAGE's (mean top-1 over the OTHER subjects), never on the target subject.
Embeddings are cached to <cache_dir>/sub-XX.npz (q, g) so later R1 analyses need no re-encoding.
usage: r1_transductive.py --source_run_dir R --params_csv C --output_csv O --cache_dir D [--subjects ...]
"""
import argparse, itertools, os, sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import encode_repetition_blocks, fit_saw_transform, load_subject_context  # noqa: E402
from module.transductive import infer  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
import torch  # noqa: E402

if not torch.cuda.is_available():  # ponytail: TransCLIP_utils.init_sigma hard-codes .cuda(); run with CUDA_VISIBLE_DEVICES=
    torch.Tensor.cuda = lambda self, *a, **k: self

GRIDS = {
    'em_dirichlet': dict(T=[30.0, 70.0, 90.0], lambd=[0.5, 10.0, 100.0], csls_k=[0, 1, 3]),
    'hard_em_dirichlet': dict(T=[30.0, 70.0, 90.0], lambd=[0.5, 10.0, 100.0], csls_k=[0, 1, 3]),
    'hard_kmeans': dict(T=[30.0, 70.0, 90.0], csls_k=[0, 1, 3]),
    'transclip': {'lambda': [0.0, 0.5, 1.0], 'n_neighbors': [3, 10], 'device': ['cpu']},
    'sinkhorn': dict(csls_k=[0, 1, 3], sinkhorn_tau=[0.03, 0.05, 0.08, 0.1], sinkhorn_iters=[10]),
    'hungarian': dict(csls_k=[0, 3]),
}


def embeddings(run_dir, sid, cache_dir):
    path = os.path.join(cache_dir, f'sub-{sid:02d}.npz')
    if not os.path.exists(path):
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(run_dir), rt, sid, average=False)
        (q,), g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
        np.savez(path, q=q, g=g)
    z = np.load(path)
    return z['q'], z['g']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows = []
    for sid in args.subjects:
        q, g = embeddings(args.source_run_dir, sid, args.cache_dir)
        qw = fit_saw_transform(q, subject_params(args.params_csv, sid))[0]
        y = np.arange(len(q))
        for method, grid in GRIDS.items():
            for vals in itertools.product(*grid.values()):
                cfg = dict(zip(grid, vals))
                for saw, x in (('raw', q), ('saw', qw)):
                    a = topk_acc(infer(x, g, method, cfg), y)
                    rows.append({'subject': sid, 'method': method, 'front': saw,
                                 'config': ' '.join(f'{k}={v}' for k, v in cfg.items() if k != 'device'), 'top1': a[1], 'top5': a[5]})
        print(sid, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv.replace('.csv', '_grid.csv'), index=False)
    sel = []  # cross-fold: config with the best mean top-1 over the other subjects
    for method, d in df.groupby('method'):
        for sid in args.subjects:
            means = d[d.subject != sid].groupby(['front', 'config'])['top1'].mean()
            front, config = means.idxmax()
            r = d[(d.subject == sid) & (d.front == front) & (d.config == config)].iloc[0]
            sel.append(r.to_dict())
    sel = pd.DataFrame(sel)
    sel.to_csv(args.output_csv, index=False)
    print(sel.groupby('method')[['top1', 'top5']].mean().round(2).sort_values('top1').to_string())


if __name__ == '__main__':
    main()
