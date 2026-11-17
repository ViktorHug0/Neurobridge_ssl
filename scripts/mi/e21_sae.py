"""A4. Sparse dictionary on the TSConv trunk (1024-d, the layer read linearly by the projector) and cross-subject
consistency of its latents.
  1. TopK SAE (4096 latents, k = 32) on the 9 training subjects' train-image activations.
  2. consistency_j = mean pairwise correlation between training subjects of latent j's tuning curve over the 16540
     training images (training subjects only: available inductively).
  3. held-out subject, test images: remove (zero in the SAE code, keep the SAE residual) the latents with the LOWEST
     consistency, the HIGHEST, or a RANDOM set of the same size; top-1 after the projector.
Also: fidelity (accuracy through the SAE reconstruction) and firing statistics seen vs held-out."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO

if DEV == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.13)
ARCH = sys.argv[1] if len(sys.argv) > 1 else 'tsconv'
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(1, 11)
NL, K = 4096, 32


class SAE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = torch.nn.Linear(d, NL)
        self.dec = torch.nn.Linear(NL, d)
        self.dec.weight.data = torch.nn.functional.normalize(self.enc.weight.data.T.clone(), dim=0)

    def code(self, x):
        z = torch.relu(self.enc(x - self.dec.bias))
        v, i = z.topk(K, dim=1)
        return torch.zeros_like(z).scatter_(1, i, v)

    def forward(self, x):
        z = self.code(x)
        return self.dec(z), z


@torch.no_grad()
def trunk(m, x, s):
    out = []
    for i in range(0, len(x), 128):
        xb = x[i:i + 128].to(DEV)
        out.append(m.enc(xb).cpu())
    return torch.cat(out)


rows = []
for fold in FOLDS:
    torch.manual_seed(fold)
    m = GM(glob.glob(f'{LOSO}/{ARCH}/seed3300/*-sub-{fold:02d}')[0])
    g = img_emb(m)
    H = {s: trunk(m, T(eeg(s, 'train')), s) for s in m.train_subs}  # (16540, 1024) per training subject
    X = torch.cat(list(H.values()))
    sae = SAE(X.shape[1]).to(DEV)
    sae.dec.bias.data = X.mean(0).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(3000):
        xb = X[torch.randint(len(X), (4096,))].to(DEV)
        rec, _ = sae(xb)
        loss = ((rec - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sae.dec.weight.data = torch.nn.functional.normalize(sae.dec.weight.data, dim=0)
    with torch.no_grad():
        code = lambda h: torch.cat([sae.code(h[i:i + 4096].to(DEV)).cpu() for i in range(0, len(h), 4096)])
        Z = torch.stack([code(H[s]) for s in m.train_subs])  # (9, 16540, NL)
        Zc = Z - Z.mean(1, keepdim=True)
        Zn = Zc / (Zc.norm(dim=1, keepdim=True) + 1e-8)
        C = torch.einsum('sil,til->lst', Zn, Zn)  # (NL, 9, 9)
        cons = (C.sum((1, 2)) - C.diagonal(dim1=1, dim2=2).sum(1)) / 72
        alive = Z.abs().sum((0, 1)) > 0
        fvu = (((sae(X[:20000].to(DEV))[0].cpu() - X[:20000]) ** 2).sum() / ((X[:20000] - X[:20000].mean(0)) ** 2).sum()).item()
        order = torch.argsort(torch.where(alive, cons, torch.full_like(cons, 9.0)))  # alive latents, least consistent first
        n_alive = int(alive.sum())
        for name, sub in (('heldout', fold), ('seen', m.train_subs[fold % 9])):
            h = trunk(m, T(eeg(sub)), sub)
            z = code(h)
            rec = sae.dec(z.to(DEV)).cpu()
            resid = h - rec
            acc = lambda hh: top1(m.eeg_proj(hh.to(DEV)).cpu(), g)
            r = dict(fold=fold, who=name, n_alive=n_alive, fvu=fvu, base=acc(h), through_sae=acc(rec),
                     l0=(z > 0).float().sum(1).mean().item(), mean_cons_of_active=(cons[None] * (z > 0)).sum().item() / (z > 0).sum().item())
            for frac in (0.1, 0.25, 0.5):
                n = int(frac * n_alive)
                gen = torch.Generator().manual_seed(0)
                sets = {'low': order[:n], 'high': order[n_alive - n:n_alive], 'rand': order[:n_alive][torch.randperm(n_alive, generator=gen)[:n]]}
                for sn, idx in sets.items():
                    z2 = z.clone(); z2[:, idx] = 0
                    r[f'drop_{sn}_{frac}'] = acc(sae.dec(z2.to(DEV)).cpu() + resid)
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    del H, X, Z, Zc, Zn
    pd.DataFrame(rows).to_csv(f'{OUT}/e21_sae_{ARCH}.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).groupby('who').mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
