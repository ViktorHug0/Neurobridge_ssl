"""A2b. Trained low-rank stitching. At ONE stage insert a' = a + a A B + b (rank r, identity at init), train only
(A, B, b) with the contrastive loss on the held-out subject's own train-image trials (ORACLE: uses that subject's
labelled data), everything else frozen. Reports held-out top-1 on the 200 test images.
Conv stages mix channels (input/temporal: 63 electrodes, spatial: 40 filters) identically at every time step.
Answers: at which depth is the held-out subject linearly correctable, and at what rank."""
import pandas as pd
from mi_lib import *
from mi_lib import _steps

RUN = sys.argv[1] if len(sys.argv) > 1 else 'mix'
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(1, 11)
RANKS = [int(r) for r in os.environ['RANKS'].split(',')] if os.environ.get('RANKS') else [0, 1, 2, 4, 8, 16, 64, -1]  # -1 = full
CH_DIM = {'input': 1, 'temporal': 2, 'spatial': 1, 'backbone': 1, 'trunk': 1, 'align': 1}
EPOCHS, BS, SCALE = 15, 512, float(torch.nn.functional.softplus(torch.tensor(np.log(1 / 0.07))))


class LowRank(torch.nn.Module):
    def __init__(self, d, r, dim):
        super().__init__()
        self.dim, r = dim, d if r < 0 else min(r, d)
        self.A = torch.nn.Parameter(torch.randn(d, r) / d ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(r, d))
        self.b = torch.nn.Parameter(torch.zeros(d))

    def forward(self, a):
        a = a.transpose(self.dim, -1)
        return (a + a @ self.A @ self.B + self.b).transpose(self.dim, -1)


def run_with(m, x, stage, f):
    steps = _steps(m)
    if stage == 'input':
        x = f(x)
    for name in STAGES[1:]:
        x = steps[name](x)
        if name == stage:
            x = f(x)
    return x


def clip_loss(e, t):
    logits = e @ t.T * SCALE
    y = torch.arange(len(e), device=e.device)
    return (torch.nn.functional.cross_entropy(logits, y) + torch.nn.functional.cross_entropy(logits.T, y)) / 2


def fit(m, stage, r, Xtr, Ttr, Xva, Tva, Xte, g, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        d = run_with(m, Xte[:2], stage, lambda a: a) if stage == 'align' else None
        probe = {}
        run_with(m, Xte[:2], stage, lambda a: probe.setdefault('s', a.shape) and a)
    f = LowRank(probe['s'][CH_DIM[stage]], r, CH_DIM[stage]).to(DEV)
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    best = (1e9, None, None)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            loss = clip_loss(run_with(m, Xtr[idx], stage, f), Ttr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ev = torch.cat([run_with(m, Xva[i:i + 1024], stage, f) for i in range(0, len(Xva), 1024)])
            va = top1(ev.cpu(), Tva.cpu())
            te = top1(run_with(m, Xte, stage, f).cpu(), g)
        if -va < best[0]:
            best = (-va, te, ep)
    return dict(acc=best[1], best_ep=best[2], val=-best[0], last=te)


if __name__ == '__main__':
    rows = []
    for fold in FOLDS:
        m = M(RUN, fold)
        g = img_emb(m)
        subj = fold % 10 + 1 if os.environ.get('SEEN') else fold  # SEEN=1: control, adapt a subject the model trained on
        X = T(eeg(subj, 'train')).to(DEV)
        Tg = img_emb(m, 'train').to(DEV)
        Xte = T(eeg(subj)).to(DEV)
        rng = np.random.RandomState(0)
        va_obj = rng.permutation(1654)[:165]
        is_va = torch.tensor(np.isin(np.arange(16540) // 10, va_obj), device=DEV)
        for stage in STAGES:
            for r in RANKS:
                if r > 0 and stage in ('input', 'temporal', 'spatial') and r >= 63:
                    continue  # 63 electrodes / 40 filters: 'full' covers it
                res = fit(m, stage, r, X[~is_va], Tg[~is_va], X[is_va], Tg[is_va], Xte, g)
                rows.append(dict(fold=fold, subj=subj, stage=stage, rank=r, **res))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/e03_stitch_{RUN}{os.environ.get("TAG", "")}.csv', index=False)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index='stage', columns='rank', values='acc', sort=False).round(2))
