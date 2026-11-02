"""E3.5 BCI-style label-free adaptation baselines on R1 (200 x 200, 80-rep queries), each also followed by SAGE.

  none     the frozen model (reference)
  adabn    AdaBN (Li et al. 2017): BatchNorm statistics re-estimated on the target batch (the 200 averaged queries)
  tent     TENT (Wang et al. 2021): BatchNorm affine parameters (on AdaBN statistics) trained to minimise the entropy
           of softmax(cos(q, g) / 0.07) over the gallery, Adam lr 1e-3, 10 steps, on the target batch
  recenter Euclidean-alignment-style input re-centering without retraining: X <- R_src^{1/2} R_tgt^{-1/2} X, with
           R_tgt the mean spatial covariance of the target batch and R_src that of the SOURCE subjects' averaged
           test-split trials (other people, no target labels). (Full EA re-aligns the source subjects too and
           retrains; that needs GPU and is not run.)
Each adapted embedding is scored plain, CSLS-3 and with SAGE (cross-fold params from --params_csv).
usage: bci_baselines.py --source_run_dir R --params_csv C --output_csv O [--subjects ...]
"""
import argparse, copy, os, sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import _test_image_matrix, apply_tta_calibration, build_test_dataset, fit_tta_calibration, load_subject_context, score_features  # noqa: E402
from module.eeg_encoder.model import run_eeg_backbone  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402


def averaged(ds):
    """The 80-rep averages in the same order/permutation-free form as encode_repetition_blocks([80]) (all reps)."""
    return np.asarray(ds.eeg_data_list[0])[:, 0].mean(1).astype(np.float32)  # (200, ch, t)


def mean_cov(x):
    x = x - x.mean(-1, keepdims=True)
    return np.einsum('nct,ndt->cd', x, x) / (x.shape[0] * x.shape[-1])


def sqrtm(c, inv=False):
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-10 * w.max(), None)
    return (v * (w ** (-0.5 if inv else 0.5))) @ v.T


def encode(model, proj, ea, x, sid):
    with torch.no_grad():
        return proj(run_eeg_backbone(model, ea, torch.tensor(x), torch.full((len(x),), sid, dtype=torch.long))).numpy()


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def adabn(model, ea, x, sid):
    m = copy.deepcopy(model)
    for bn in bn_layers(m):
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over the passes below
    m.train()
    with torch.no_grad():
        run_eeg_backbone(m, ea, torch.tensor(x), torch.full((len(x),), sid, dtype=torch.long))
    m.eval()
    return m


def tent(model, proj, ea, x, g, sid, steps=10, lr=1e-3, temp=0.07):
    m = adabn(model, ea, x, sid)
    params = [p for bn in bn_layers(m) for p in (bn.weight, bn.bias) if p is not None]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    gt = torch.nn.functional.normalize(torch.tensor(g), dim=1)
    xs, ids = torch.tensor(x), torch.full((len(x),), sid, dtype=torch.long)
    for _ in range(steps):  # BN layers stay in eval mode: AdaBN statistics, trainable affine
        q = torch.nn.functional.normalize(proj(run_eeg_backbone(m, ea, xs, ids)), dim=1)
        logp = torch.log_softmax(q @ gt.T / temp, dim=1)
        loss = -(logp.exp() * logp).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows = []
    for sid in args.subjects:
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        model, proj = mods['model'], mods['eeg_projector']
        model.eval()
        x = averaged(ds)
        with torch.no_grad():
            g = mods['img_projector'](torch.tensor(_test_image_matrix(ds), dtype=torch.float32)).numpy()
        src = [averaged(build_test_dataset(ea, s, average=False)) for s in range(1, 11) if s != sid]
        A = (sqrtm(np.mean([mean_cov(z) for z in src], 0)) @ sqrtm(mean_cov(x), inv=True)).astype(np.float32)
        emb = {'none': encode(model, proj, ea, x, sid),
               'adabn': encode(adabn(model, ea, x, sid), proj, ea, x, sid),
               'tent': encode(tent(model, proj, ea, x, g, sid), proj, ea, x, sid),
               'recenter': encode(model, proj, ea, np.einsum('cd,ndt->nct', A, x), sid)}
        y = np.arange(200)
        for name, q in emb.items():
            sc = {'plain': score_features(q, g), 'csls3': score_features(q, g, use_csls=True, csls_k=3),
                  'sage': score_features(apply_tta_calibration(q, fit_tta_calibration(q, g, p)), g, use_csls=True, csls_k=p.csls_k)}
            for k, s in sc.items():
                rows.append({'subject': sid, 'adapt': name, 'score': k, 'top1': topk_acc(s, y)[1]})
        print(sid, {(r['adapt'], r['score']): r['top1'] for r in rows[-12:]}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.groupby(['adapt', 'score'])['top1'].mean().unstack().round(2).to_string())


if __name__ == '__main__':
    main()
