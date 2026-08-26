"""Learned MoE router over the frozen best-4 experts (inductive LOSO).

For each held-out subject k, a router is trained ONLY on other subjects. It reads
per-expert confidence features (and optionally the experts' EEG embeddings) for a
query and emits a gate per expert; the fused score is the gated mean of the experts'
200-way cosine score vectors. Trained end-to-end with a retrieval cross-entropy
(objective A), with a straight-through estimator so the gates commit to a discrete
subset {1..4 experts} at inference.

Architecture / hyperparameters are selected on an INNER validation split of the
router's own training subjects -- never on subject k.

usage: router_moe.py [--quick]
"""
import argparse, itertools, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DUMP = 'results/things_eeg/synthetic_subjects/router/dumps'
E = ['atm_iv', 'ge100', 'tsconv_eva', 'tsconv_vith']
K, Q = len(E), 200
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load():
    """S[k,f,s] -> (200,200) cosine scores ; Z[k,f,s] -> (200,dim) eeg embeddings."""
    S = np.zeros((K, 10, 10, Q, Q), np.float32)
    Z = [[[None] * 10 for _ in range(10)] for _ in range(K)]
    for ki, n in enumerate(E):
        for f in range(10):
            for s in range(10):
                d = np.load(f'{DUMP}/{n}-f{f+1:02d}-s{s+1:02d}.npz')
                e = d['eeg'].astype(np.float64); i = d['image'].astype(np.float64)
                en = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
                i /= np.maximum(np.linalg.norm(i, axis=1, keepdims=True), 1e-12)
                S[ki, f, s] = (en @ i.T).astype(np.float32)
                Z[ki][f][s] = np.concatenate([en, np.linalg.norm(e, axis=1, keepdims=True)], 1).astype(np.float32)
    return S, Z


def conf_feats(s):
    """s: (Q,Q) scores -> (Q,7) confidence descriptors available at inference."""
    srt = np.sort(s, 1)
    mx, mean, std = s.max(1), s.mean(1), s.std(1) + 1e-12
    p = np.exp((s - mx[:, None]) / 0.05); p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return np.stack([mx, srt[:, -1] - srt[:, -2], srt[:, -1] - srt[:, -5],
                     mean, std, (mx - mean) / std, ent], 1).astype(np.float32)


class Router(nn.Module):
    """arch in {lin_conf, mlp_conf, mlp_emb, mlp_both, siamese}"""
    def __init__(self, arch, dc, de, hid=64):
        super().__init__(); self.arch = arch
        if arch == 'lin_conf':  self.net = nn.Linear(dc * K, K)
        elif arch == 'mlp_conf': self.net = nn.Sequential(nn.Linear(dc * K, hid), nn.GELU(), nn.Linear(hid, K))
        elif arch == 'mlp_emb':  self.net = nn.Sequential(nn.Linear(de, hid), nn.GELU(), nn.Linear(hid, K))
        elif arch == 'mlp_both': self.net = nn.Sequential(nn.Linear(dc * K + de, hid), nn.GELU(), nn.Linear(hid, K))
        elif arch == 'siamese':  # shared scorer applied per expert -> cannot memorise expert identity
            self.net = nn.Sequential(nn.Linear(dc, hid), nn.GELU(), nn.Linear(hid, 1))
        self.logit_bias = nn.Parameter(torch.ones(K) * 2.0)   # start near "use everything"

    def forward(self, c, z):          # c:(B,K,dc)  z:(B,de)
        if self.arch == 'siamese':
            g = self.net(c).squeeze(-1)
        elif self.arch == 'mlp_emb':
            g = self.net(z)
        elif self.arch == 'mlp_both':
            g = self.net(torch.cat([c.flatten(1), z], 1))
        else:
            g = self.net(c.flatten(1))
        return g + self.logit_bias


def fuse(gate, S):                    # gate:(B,K) in [0,1], S:(B,K,Q)
    w = gate / gate.sum(1, keepdim=True).clamp_min(1e-6)
    return (w.unsqueeze(-1) * S).sum(1)


def run(arch, tr, va, te, hp, seed=0):
    torch.manual_seed(seed)
    dc, de = tr['c'].shape[-1], tr['z'].shape[-1]
    m = Router(arch, dc, de, hp['hid']).to(dev)
    logT = nn.Parameter(torch.tensor(np.log(1 / 0.05), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(list(m.parameters()) + [logT], lr=hp['lr'], weight_decay=hp['wd'])
    y = torch.arange(Q, device=dev)
    best, best_state = -1, None
    for ep in range(hp['epochs']):
        m.train()
        perm = torch.randperm(len(tr['c']), device=dev)
        for b in perm.split(256):
            g = torch.sigmoid(m(tr['c'][b], tr['z'][b]))
            gh = ((g > 0.5).float() - g).detach() + g          # straight-through
            gate = gh if hp['hard'] else g
            logits = fuse(gate, tr['S'][b]) * logT.exp()
            loss = F.cross_entropy(logits, tr['y'][b]) + hp['lam'] * g.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            g = torch.sigmoid(m(va['c'], va['z']))
            gate = (g > 0.5).float()
            gate = torch.where(gate.sum(1, keepdim=True) == 0, torch.ones_like(gate), gate)
            acc = (fuse(gate, va['S']).argmax(1) == va['y']).float().mean().item()
        if acc > best: best, best_state = acc, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        g = torch.sigmoid(m(te['c'], te['z']))
        gate = (g > 0.5).float()
        gate = torch.where(gate.sum(1, keepdim=True) == 0, torch.ones_like(gate), gate)
        acc = (fuse(gate, te['S']).argmax(1) == te['y']).float().mean().item()
        nsel = gate.sum(1).mean().item()
    return best, acc, nsel


def pack(items, S, Z, feats):
    """items: list of (fold, subject). Returns tensors on device."""
    c = np.concatenate([feats[(f, s)] for f, s in items])
    z = np.concatenate([np.concatenate([Z[k][f][s] for k in range(K)], 1) for f, s in items])
    sc = np.concatenate([np.stack([S[k, f, s] for k in range(K)], 1) for f, s in items])
    return {'c': torch.tensor(c, device=dev), 'z': torch.tensor(z, device=dev),
            'S': torch.tensor(sc, device=dev),
            'y': torch.arange(Q, device=dev).repeat(len(items))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    S, Z = load()
    feats = {(f, s): np.stack([conf_feats(S[k, f, s]) for k in range(K)], 1) for f in range(10) for s in range(10)}
    uni = np.mean([(np.mean([S[k, f, f] for k in range(K)], 0).argmax(1) == np.arange(Q)).mean() for f in range(10)]) * 100
    print(f'sanity: uniform mean over the 10 held-out folds = {uni:.2f}\n')

    archs = ['lin_conf', 'mlp_conf', 'siamese', 'mlp_emb', 'mlp_both']
    grid = [dict(lr=lr, wd=wd, lam=lam, hid=64, epochs=(20 if a.quick else 60), hard=hard)
            for lr in (1e-3, 1e-2) for wd in (1e-4, 1e-2) for lam in (0.0, 0.05) for hard in (True, False)]
    if a.quick: grid = grid[:4]

    results = {}
    for src in ('A_seen', 'B_unseen', 'C_union'):
        per_fold, picks = [], []
        for k in range(10):
            others = [j for j in range(10) if j != k]
            va_subs, tr_subs = others[:2], others[2:]
            def items(subs):
                if src == 'A_seen':   return [(k, s) for s in subs]
                if src == 'B_unseen': return [(s, s) for s in subs]
                return [(k, s) for s in subs] + [(s, s) for s in subs]
            tr, va, te = pack(items(tr_subs), S, Z, feats), pack(items(va_subs), S, Z, feats), pack([(k, k)], S, Z, feats)
            best = (-1, None, None)
            for arch in archs:
                for hp in grid:
                    v, t, n = run(arch, tr, va, te, hp)
                    if v > best[0]: best = (v, t, (arch, hp, n))
            per_fold.append(best[1] * 100); picks.append(best[2])
            print(f'  [{src}] fold {k+1:2d}: val {best[0]*100:5.2f}  TEST {best[1]*100:5.2f}  '
                  f'arch={best[2][0]:9s} experts_used={best[2][2]:.2f}', flush=True)
        results[src] = per_fold
        print(f'  [{src}] MEAN over 10 folds = {np.mean(per_fold):.2f}   (uniform {uni:.2f})\n', flush=True)
    json.dump({'uniform': uni, 'results': results}, open('ensemble_experiments/legacy/scratch_claude/router_moe_results.json', 'w'), indent=1)
    print('=== summary ===')
    for src, v in results.items():
        print(f'  {src:9s} {np.mean(v):6.2f}  vs uniform {uni:.2f}  ({np.mean(v)-uni:+.2f})')


if __name__ == '__main__':
    main()
