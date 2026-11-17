"""Key tuning (oracle). Frozen model; for the held-out subject learn ONLY an additive input pattern (a new key)
with the contrastive loss on that subject's labelled train-image trials, and test on the 200 test images.
  full   : free 63x250 pattern added to the input
  span   : own key replaced by  sum_t w_t m_t  over the 9 training keys (9 numbers, softmax or free)
  span+  : own key kept, plus a free combination of the (training key - own key) directions (9 numbers)
Tells how much of the gap is the choice of operating point alone."""
import pandas as pd
from mi_lib import *
from e03_stitch import clip_loss  # noqa

RUN = sys.argv[1] if len(sys.argv) > 1 else 'mix'


def fit(m, make, params, X, Tg, is_va, Xte, g, epochs=15, lr=1e-2):
    opt = torch.optim.Adam(params, lr=lr)
    Xtr, Ttr, Xva, Tva = X[~is_va], Tg[~is_va], X[is_va], Tg[is_va]
    best = (-1, None)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm) - 511, 512):
            idx = perm[i:i + 512]
            loss = clip_loss(grad_forward(m, make(Xtr[idx])), Ttr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            va = top1(torch.cat([grad_forward(m, make(Xva[i:i + 1024])) for i in range(0, len(Xva), 1024)]).cpu(), Tva.cpu())
            te = top1(grad_forward(m, make(Xte)).cpu(), g)
        if va > best[0]:
            best = (va, te)
    return best[1]


def grad_forward(m, x):
    e = m.enc
    return m.eeg_proj(e.proj_eeg(e.projection(e.tsconv(x.unsqueeze(1))).flatten(1)))


if __name__ == '__main__':
    rows = []
    for fold in range(1, 11):
        torch.manual_seed(0)
        m = M(RUN, fold)
        g = img_emb(m)
        X, Tg, Xte = T(eeg(fold, 'train')).to(DEV), img_emb(m, 'train').to(DEV), T(eeg(fold)).to(DEV)
        is_va = torch.tensor(np.isin(np.arange(16540) // 10, np.random.RandomState(0).permutation(1654)[:165]), device=DEV)
        keys = torch.stack([T(eeg(t, 'train')).mean(0) for t in m.train_subs]).to(DEV)  # (9,63,250), train-image means
        own = X.mean(0)
        r = dict(fold=fold, base=top1(grad_forward(m, Xte).detach().cpu(), g))
        b = torch.zeros(63, 250, device=DEV, requires_grad=True)
        r['full'] = fit(m, lambda x: x + b, [b], X, Tg, is_va, Xte, g, lr=3e-3)
        w = torch.zeros(9, device=DEV, requires_grad=True)
        r['span'] = fit(m, lambda x: x - own + torch.einsum('k,kct->ct', torch.softmax(w, 0), keys), [w], X, Tg, is_va, Xte, g, lr=5e-2)
        v = torch.zeros(9, device=DEV, requires_grad=True)
        r['span+'] = fit(m, lambda x: x + torch.einsum('k,kct->ct', v, keys - own), [v], X, Tg, is_va, Xte, g, lr=2e-2)
        r['w'] = torch.softmax(w, 0).detach().cpu().numpy().round(2).tolist()
        r['v'] = v.detach().cpu().numpy().round(2).tolist()
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e08_keytune_{RUN}.csv', index=False)
    print(df[['base', 'full', 'span', 'span+']].mean().round(2))
