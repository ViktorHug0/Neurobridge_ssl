"""What is the cross-subject drift, and does SAGE's unlabeled correction match it? (v2 paper, user 2026-10-24)

(A) Hypothesis class. On the target subject's own TRAINING session (4-rep averages, 16,540 labeled pairs, concepts
    disjoint from the test set) we fit, with the same InfoNCE loss, maps of increasing capacity from its EEG embeddings
    to the image embeddings: rot (orthogonal), wrot (shrinkage whitening, then orthogonal = SAGE's family), affine,
    mlp (x + MLP(x)). 10% of the training concepts are held out: `heldout` = 200-way top-1 among held-out training
    images (same 4-rep SNR as the fit, 5 draws), `test` = 200-way top-1 on the 80-rep test queries. If affine / mlp
    do not beat rot / wrot with this much data, a rotation (after whitening) describes the drift as far as retrieval
    can tell.
(B) SAGE vs the supervised correction, in SAGE's own whitened space (test-batch whitening applied to both):
    R_sage (unlabeled, test batch) vs R_sup (closed-form Procrustes on all 16,540 labeled training pairs).
    angle_*: median principal angle (deg) of R_sup (vs identity), of R_sage^T R_sup (SAGE vs supervised) and of a
    random rotation (null, ~90 deg). agree: fraction of test queries with the same CSLS top-1 under R_sage and R_sup.
    Transfer: SAGE's correction, fitted on the test batch and FROZEN, applied to held-out training images it never saw
    (heldout_*: whitening only vs whitening + R_sage vs whitening + R_sup, CSLS). *_matched: the same at matched SNR,
    SAGE fitted on one 4-rep held-out batch and applied frozen to another; heldout_sage_refit = SAGE on the batch
    itself; heldout_prior = Hungarian on the zero-shot scores of the batch.
usage: drift_nature.py --source_run_dir R --params_csv C --output_csv O [--subjects ...]
"""
import argparse, os, sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import (apply_saw_transform, apply_tta_calibration, build_test_dataset, encode_indexed_average_features,  # noqa: E402
                    encode_repetition_blocks, fit_paired_orthogonal, fit_saw_transform, fit_tta_calibration,
                    load_subject_train_test_context, normalize_rows, score_features)
from stream_grid import subject_params  # noqa: E402


def top1(x, g, csls=False, k=3):
    return 100 * (score_features(x, g, use_csls=csls, csls_k=k).argmax(1) == np.arange(len(x))).mean()


class Map(torch.nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        if kind in ('rot', 'wrot'):
            self.m = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d, d, bias=False))
        elif kind == 'affine':
            self.m = torch.nn.Linear(d, d)
            torch.nn.init.eye_(self.m.weight); torch.nn.init.zeros_(self.m.bias)
        else:
            self.m = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, d))
            torch.nn.init.zeros_(self.m[2].weight); torch.nn.init.zeros_(self.m[2].bias)

    def forward(self, x):
        return x + self.m(x) if self.kind == 'mlp' else self.m(x)


def fit_map(kind, x, y, seed, epochs=200, tau=0.05):
    """InfoNCE (both directions) from x to y; early stopping on 10% of the pairs."""
    torch.manual_seed(seed)
    x, y = torch.tensor(x), torch.tensor(F.normalize(torch.tensor(y), dim=1).numpy())
    p = torch.randperm(len(x)); va, tr = p[:len(x) // 10], p[len(x) // 10:]
    net = Map(kind, x.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4 if kind in ('affine', 'mlp') else 0)
    best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        for b in tr[torch.randperm(len(tr))].split(1024):
            logits = F.normalize(net(x[b]), dim=1) @ y[b].T / tau
            t = torch.arange(len(b))
            loss = (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lv = sum(F.cross_entropy(F.normalize(net(x[b]), dim=1) @ y[b].T / tau, torch.arange(len(b))).item()
                     for b in va.split(1024))
        if lv < best - 1e-4:
            best, state, bad = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(state)
    return lambda z: net(torch.tensor(z, dtype=torch.float32)).detach().numpy()


def median_angle(m):
    ang = np.degrees(np.abs(np.angle(np.linalg.eigvals(m.astype(np.float64)))))
    return float(np.median(ang))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    rows = []
    for sid in args.subjects:
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, train_ds, _, mods = load_subject_train_test_context(os.path.abspath(args.source_run_dir), rt, sid, average=True)
        tq, ti, obj, _ = encode_indexed_average_features(ea, mods, train_ds)
        (q,), g, _ = encode_repetition_blocks(ea, mods, build_test_dataset(ea, sid, average=False), block_sizes=[80], seed=3300)
        rng = np.random.default_rng(sid)
        held = np.isin(obj, rng.permutation(np.unique(obj))[:len(np.unique(obj)) // 10])
        fq, fi, hq, hi = tq[~held], ti[~held], tq[held], ti[held]
        draws = [rng.choice(len(hq), 200, replace=False) for _ in range(5)]
        ho = lambda f, csls=False: np.mean([top1(f(hq[d]), hi[d], csls) for d in draws])  # noqa: E731
        rec = dict(subject=sid, n_fit=len(fq), n_heldout=len(hq))
        # (A) hypothesis class, all fitted with the same loss on the same labeled pairs
        rec['heldout_identity'], rec['test_identity'] = ho(lambda z: z), top1(q, g)
        _, wstats = fit_saw_transform(fq, p)
        for kind in ('rot', 'wrot', 'affine', 'mlp'):
            pre = (lambda z: apply_saw_transform(z, wstats)) if kind == 'wrot' else normalize_rows
            m = fit_map(kind, pre(fq), fi, seed=sid)
            rec[f'heldout_{kind}'], rec[f'test_{kind}'] = ho(lambda z: m(pre(z))), top1(m(pre(q)), g)
            rec[f'test_{kind}_csls'] = top1(m(pre(q)), g, True)
        R = fit_paired_orthogonal(normalize_rows(fq), normalize_rows(fi))
        rec['heldout_rot_procrustes'], rec['test_rot_procrustes'] = ho(lambda z: normalize_rows(z) @ R), top1(normalize_rows(q) @ R, g)
        # (B) SAGE's unlabeled correction vs the supervised one, in SAGE's whitened space
        cal = fit_tta_calibration(q, g, p)
        R_sage, s = cal['orthogonal_map'], cal['saw_stats']
        R_sup = fit_paired_orthogonal(apply_saw_transform(tq, s), normalize_rows(ti))
        sup = {'saw_stats': s, 'orthogonal_map': R_sup}
        white = {'saw_stats': s, 'orthogonal_map': np.eye(len(R_sup), dtype=np.float32)}
        rec.update(test_sage=top1(apply_tta_calibration(q, cal), g, True), test_sup_in_sage_space=top1(apply_tta_calibration(q, sup), g, True),
                   agree=100 * (score_features(apply_tta_calibration(q, cal), g, True, p.csls_k).argmax(1)
                                == score_features(apply_tta_calibration(q, sup), g, True, p.csls_k).argmax(1)).mean(),
                   angle_sup=median_angle(R_sup), angle_sage_vs_sup=median_angle(R_sage.T @ R_sup),
                   angle_sage=median_angle(R_sage), angle_random=median_angle(ortho_group.rvs(len(R_sup), random_state=sid)),
                   heldout_white_only=ho(lambda z: apply_tta_calibration(z, white), True),
                   heldout_sage_frozen=ho(lambda z: apply_tta_calibration(z, cal), True),
                   heldout_sup_frozen=ho(lambda z: apply_tta_calibration(z, sup), True),
                   heldout_identity_csls=ho(lambda z: z, True))
        # matched SNR (4-rep held-out batches of 200): SAGE refitted on the batch itself (transductive) vs SAGE fitted
        # on ANOTHER held-out batch and applied frozen (does the correction transfer to images it was not fitted on?)
        fits = [fit_tta_calibration(hq[d], hi[d], p) for d in draws]
        rec['heldout_sage_refit'] = np.mean([top1(apply_tta_calibration(hq[d], c), hi[d], True) for d, c in zip(draws, fits)])
        rec['heldout_sage_frozen_matched'] = np.mean([top1(apply_tta_calibration(hq[d], fits[(i + 1) % 5]), hi[d], True)
                                                      for i, d in enumerate(draws)])
        rec['heldout_white_only_matched'] = np.mean([top1(apply_tta_calibration(hq[d], dict(fits[(i + 1) % 5], orthogonal_map=np.eye(len(R_sup), dtype=np.float32))), hi[d], True)
                                                     for i, d in enumerate(draws)])
        rec['heldout_prior'] = np.mean([100 * (linear_sum_assignment(-score_features(hq[d], hi[d]))[1] == np.arange(200)).mean() for d in draws])
        rows.append(rec)
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df.mean(numeric_only=True).round(1).to_string())


if __name__ == '__main__':
    main()
