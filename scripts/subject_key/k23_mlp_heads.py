"""K23: nonlinear distilled heads on the frozen trunk (CPU, from the caches of k21_heads.py). Per teacher space
(members 0, 2, 3 of t6) a head is trained by gradient to reproduce the member's EEG embedding (1 - cosine):
  linear        Linear(1440, d)                       (the gradient-trained twin of the ridge head)
  mlp           Linear(1440, 512) - GELU - Linear(512, d)
  res           the shape of the encoders' own head: Linear(1440, d), then x + Linear(GELU(x)), LayerNorm
dropout p on the trunk feature while training. Scored as in K21 (calibrated, mean of the three cosine matrices).
Usage: k23_mlp_heads.py [folds]"""
import sys
import pandas as pd, torch, torch.nn as nn

SK = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/subject_key'
FOLDS = [int(f) for f in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 5, 7]
nz = torch.nn.functional.normalize
acc = lambda S: 100 * (S.argmax(1) == torch.arange(len(S))).float().mean().item()
rank = lambda S: (S > S.diag()[:, None]).sum(1).float().mean().item()


class Res(nn.Module):
    def __init__(self, d):
        super().__init__(); self.a, self.b, self.n = nn.Linear(1440, d), nn.Linear(d, d), nn.LayerNorm(d)

    def forward(self, x):
        h = self.a(x)
        return self.n(h + self.b(nn.functional.gelu(h)))


def make(kind, d):
    return {'linear': lambda: nn.Linear(1440, d), 'mlp': lambda: nn.Sequential(nn.Linear(1440, 512), nn.GELU(), nn.Linear(512, d)), 'res': lambda: Res(d)}[kind]()


rows = []
for fold in FOLDS:
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    c = torch.load(f'{SK}/cache/trunk_atm128_vc_seed3300_f{fold}.pt')
    X, Xte = c['tr'].float(), c['te'].float()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xte = (X - mu) / sd, (Xte - mu) / sd
    r = dict(fold=fold)
    for kind in ('linear', 'mlp', 'res'):
        for p in (0.0, 0.5):
            torch.manual_seed(0)
            S = []
            for k in (0, 2, 3):
                Y = t['eeg'][k].float()
                head = make(kind, Y.shape[1])
                opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=8 * (len(X) // 2048 + 1))
                for ep in range(8):
                    perm = torch.randperm(len(X))
                    for i in range(0, len(X), 2048):
                        idx = perm[i:i + 2048]
                        loss = (1 - nn.functional.cosine_similarity(head(nn.functional.dropout(X[idx], p)), Y[idx])).mean()
                        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                head.eval()
                with torch.no_grad():
                    e = nz(head(Xte), dim=1)
                S.append(nz(e - e.mean(0) + t['img'][k].float().mean(0), dim=1) @ Gte[k].float().T)
            r[f'{kind}_p{p}'], r[f'{kind}_p{p}_rank'], r[f'{kind}_p{p}_loss'] = acc(sum(S)), rank(sum(S)), loss.item()
            print(fold, kind, p, round(r[f'{kind}_p{p}'], 1), round(r[f'{kind}_p{p}_rank'], 2), round(loss.item(), 3), flush=True)
    rows.append(r)
print(pd.DataFrame(rows).mean().round(2).to_string())
