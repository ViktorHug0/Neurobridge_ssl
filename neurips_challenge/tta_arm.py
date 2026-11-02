"""ARM-lite (TTA_LOG T26): learn the in-batch person affinity END TO END through the centring operator.
Affinity S_ij = cos(A z_i, A z_j), z = WCCN-whitened, batch-centred fingerprint (fixed, fold-k training people), A =
I + U V^T (rank r, U = 0 at init: starts exactly at the wknn space). Soft neighbours W = softmax(S / tau) (no self),
C = q - mu - c (W q - mu), then the fixed top-3 shrink, + .5 mu (= batch_grouping). Loss: per-trial InfoNCE of each
member's corrected output against the 200 gallery targets. To evaluate fold k it trains ONLY on the cached streams of
folds j != k (their people are unseen by their own models and are not fold k's people). The learned part sees only
fingerprints, so it cannot memorise the shared gallery. Saves A per fold for tta_lab 'arm:K:c' (hard top-K, as wknn).
    tta_arm.py [--rank 32] [--steps 3000]
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.score_track1 import batch_transform  # noqa: E402
from neurips_challenge.tta_lab import DEV, OUT, REF, load, mn, wccn_affinity  # noqa: E402

FOLDS = (1, 2, 3, 4)


def prep(k_eval):
    """per fold: fingerprints projected with fold k_eval's WCCN (z, not batch-centred yet), mean-normed member outputs,
    targets; batch boundaries = consecutive 64 rows"""
    ctx_eval = load(k_eval, REF[:1]); aff, _ = wccn_affinity(ctx_eval); del ctx_eval
    data = {}
    for k in FOLDS:
        ctx = load(k, REF)
        Z = torch.cat([((ctx['F'][b:b + 4096].double() - aff.m0) / aff.sd @ aff.T).float() for b in range(0, len(ctx['F']), 4096)])
        Q = [torch.cat([torch.cat(mn([P[b:b + 64]])) for b in range(0, len(P), 64)]).half() for P in ctx['P']]
        data[k] = dict(Z=Z, Q=Q, img=torch.as_tensor(ctx['img'], device=DEV), G=Fn.normalize(torch.as_tensor(ctx['G'], device=DEV).float(), dim=1))
        del ctx
    return data


def correct(q, W, c):
    mu = q.mean(0, keepdim=True)
    return batch_transform(q - mu - c * (W @ q - mu), a=0, k=3, detach_dir=True) + .5 * mu


def train(k_eval, data, rank, steps, seed=0):
    torch.manual_seed(seed)
    D = data[k_eval]['Z'].shape[1]
    U = torch.zeros(D, rank, device=DEV, requires_grad=True); V = (torch.randn(D, rank, device=DEV) / D ** .5).requires_grad_()
    log_tau = torch.tensor(np.log(.05), device=DEV, requires_grad=True); c_ = torch.tensor(.7, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([U, V, log_tau, c_], lr=1e-3)
    src = [k for k in FOLDS if k != k_eval]
    for step in range(steps):
        loss = 0
        for _ in range(8):
            k = src[np.random.randint(len(src))]; d = data[k]; b = 64 * np.random.randint(len(d['img']) // 64)
            z = d['Z'][b:b + 64]; z = z - z.mean(0); z = z + (z @ V) @ U.T
            zn = Fn.normalize(z, dim=1); S = zn @ zn.T
            W = torch.softmax(S.masked_fill(torch.eye(64, device=DEV, dtype=torch.bool), -1e4) / log_tau.exp(), 1)
            for q in d['Q']:
                y = Fn.normalize(correct(q[b:b + 64].float(), W, c_), dim=1)
                loss = loss + Fn.cross_entropy(y @ d['G'].T / .05, d['img'][b:b + 64])
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            U.mul_(1 - 1e-4)  # mild pull back toward the identity map
        if step % 500 == 0:
            print(f'fold {k_eval} step {step} loss {loss.item() / 24:.4f} tau {log_tau.exp().item():.3f} c {c_.item():.3f}', flush=True)
    return dict(U=U.detach().cpu(), V=V.detach().cpu(), tau=log_tau.exp().item(), c=c_.item())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rank', type=int, default=32); ap.add_argument('--steps', type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(f'{OUT}/arm', exist_ok=True)
    for k in FOLDS:
        data = prep(k)
        torch.save(train(k, data, a.rank, a.steps), f'{OUT}/arm/fold{k}_r{a.rank}.pt')
        del data; torch.cuda.empty_cache()
