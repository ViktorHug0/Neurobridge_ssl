"""Learned participant embedding for in-batch routing (ROUTING_LOG E48, user idea 2026-10-15). A small network maps one
raw trial to a unit vector such that trials of the same participant are close, trained with a cosine-margin softmax
over the TRAINING participants (identities known offline; on the s11 testbed: the other split's 9 Alljoined subjects +
the 10 THINGS-EEG2 subjects, = fit_fingerprint_lda's subjects). It must generalise to unseen participants, so it is
scored on the split's 11 test participants by in-batch neighbour purity (fraction of a trial's K=4 nearest batch-mates
in a shuffled 64-batch that are the same person; chance ~0.08 at 11 participants), next to the deployed LDA fingerprint.
Inputs: the 656-d fingerprint (score_track1.eeg_fingerprint) + learned log-variance features (temporal filters ->
per-filter spatial filters -> log power over the trial, a learned filter-bank CSP).
    subject_embed.py [--arch fp|cnn|both] [--loss am|supcon] [--tag T]
--loss supcon: deployment-like episodes (64 trials from 11 random training participants), supervised contrastive.  -> results/alljoined/track1/routing/emb_<tag>_split{k}.npy
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.score_track1 import eeg_fingerprint  # noqa: E402
from neurips_challenge.route_study import DATA, SPLITS  # noqa: E402

OUT = 'results/alljoined/track1/routing'
EEG1 = list(range(1, 51))  # THINGS-EEG1 ds003825 subjects (eeg1_identity_cache.py)


class Embedder(nn.Module):
    def __init__(self, n_fp=656, arch='both', filters=16, spatial=4, dim=64, n_classes=19):
        super().__init__()
        self.arch = arch
        n_in = (n_fp if arch in ('fp', 'both') else 0) + (filters * spatial if arch in ('cnn', 'both') else 0)
        if arch in ('cnn', 'both'):
            self.temporal = nn.Conv2d(1, filters, (1, 13), padding=(0, 6))
            self.spatial = nn.Conv2d(filters, filters * spatial, (32, 1), groups=filters)
        self.fp_norm = nn.BatchNorm1d(n_fp) if arch in ('fp', 'both') else None
        self.mlp = nn.Sequential(nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, dim))
        self.classes = nn.Parameter(torch.randn(n_classes, dim) * .1)

    def forward(self, x, fp):
        f = []
        if self.fp_norm is not None:
            f.append(self.fp_norm(fp))
        if self.arch in ('cnn', 'both'):
            z = self.spatial(self.temporal(x[:, None])).squeeze(2)            # (B, F*S, T)
            f.append(torch.log(z.var(-1) + 1e-4))
        return nn.functional.normalize(self.mlp(torch.cat(f, 1)), dim=-1)

    def loss(self, e, y, s=20.0, m=.2):  # AM-softmax
        cos = e @ nn.functional.normalize(self.classes, dim=-1).T
        return nn.functional.cross_entropy(s * (cos - m * nn.functional.one_hot(y, cos.shape[1])), y)


def episodes(by_class, n_ep=8, n=64, people=11):
    """n_ep deployment-like batches: n trials drawn from `people` random training participants (~6 each)"""
    out = []
    for _ in range(n_ep):
        who = torch.randperm(len(by_class))[:people][torch.randint(people, (n,))]
        out.append(torch.stack([by_class[c][torch.randint(len(by_class[c]), (1,))][0] for c in who.tolist()]))
    return torch.cat(out)


def supcon(e, y, n=64, tau=.1):
    """supervised contrastive loss inside each consecutive n-trial episode: a trial's same-person batch-mates are
    its positives, every other batch-mate a negative (= how the embedding is used: nearest batch-mates)"""
    e, y = e.view(-1, n, e.shape[-1]), y.view(-1, n)
    sim = e @ e.transpose(1, 2) / tau
    eye = torch.eye(n, dtype=torch.bool, device=e.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - sim.logsumexp(-1, keepdim=True)
    pos = (y[:, :, None] == y[:, None, :]) & ~eye
    has = pos.any(-1)
    return -((logp * pos).sum(-1)[has] / pos.sum(-1)[has]).mean()


def train_trials(s, n_img=200):
    """first n_img training images x all repeats, contiguous read (random reads saturate the NAS); padding dropped"""
    x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r')[:n_img], np.float32).reshape(-1, 32, 120)
    return x[np.abs(x).sum((1, 2)) > 0]


def purity(E, sid, rng, K=4, n_batches=400):
    out = []
    for _ in range(n_batches):
        i = rng.choice(len(E), 64, replace=False)
        D = torch.cdist(E[i], E[i]); D.fill_diagonal_(1e9)
        nb = D.topk(K, largest=False).indices
        out.append((sid[i][nb] == sid[i][:, None]).float().mean().item())
    return np.mean(out)


def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k in (1, 2):
        t0 = time.time(); torch.manual_seed(k)
        subs = SPLITS[k]; train = [s for s in SPLITS[3 - k] if s not in subs] + list(range(21, 31))
        xs = [train_trials(s) for s in train]
        if a.eeg1:  # + THINGS-EEG1 identities (eeg1_identity_cache.py)
            # ponytail: the 8 longest recordings OOM'd in the 3.9 GB CPU step (8 18 20 33 45 47 49 50); use the 42 extracted
            fs = [f for f in (f'{DATA}/.cache/eeg1id_s{s:02d}.npy' for s in EEG1) if os.path.exists(f)]
            print(f'+ {len(fs)} THINGS-EEG1 identities', flush=True)
            xs += [np.load(f) for f in fs]
        n_cls = len(xs)
        X = torch.tensor(np.concatenate(xs)); y = torch.tensor(np.repeat(np.arange(n_cls), [len(x) for x in xs]))
        del xs
        scale = X[:5000].std().item()
        Fp = torch.cat([eeg_fingerprint(X[i:i + 4000]) for i in range(0, len(X), 4000)])
        model = Embedder(arch=a.arch, n_classes=n_cls).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
        steps = a.epochs * (len(X) // 512); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        model.train(); it = 0
        by_class = [torch.nonzero(y == c).squeeze(1) for c in range(n_cls)]
        for ep in range(a.epochs):
            perm = torch.randperm(len(X)); tot = 0
            for j in range(0, len(X) - 511, 512):
                b = perm[j:j + 512] if a.loss == 'am' else episodes(by_class)
                xb = (X[b] / scale).to(dev); xb = xb - xb.mean(-1, keepdim=True)
                e = model(xb, Fp[b].to(dev))
                loss = model.loss(e, y[b].to(dev)) if a.loss == 'am' else supcon(e, y[b].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); it += 1
            print(f'split {k} ep {ep} loss {tot / (len(X) // 512):.3f}', flush=True)
        from neurips_challenge.score_track1 import fit_fingerprint_lda
        L = fit_fingerprint_lda(DATA, train)
        model.eval(); E, sid, EL = [], [], []
        with torch.no_grad():
            for i, s in enumerate(subs):  # trial order = test.npy order = trial_dots / neighbour_lda.fingerprint
                xt = np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)
                for j in range(0, len(xt), 4000):
                    xb = torch.tensor(np.asarray(xt[j:j + 4000], np.float32))
                    fb = eeg_fingerprint(xb); xb = xb / scale; xb = xb - xb.mean(-1, keepdim=True)
                    E.append(model(xb.to(dev), fb.to(dev)).cpu()); EL.append((fb - L['m0']) / L['sd'] @ L['W'])
                sid.append(torch.full((len(xt),), i))
        E, sid, EL = torch.cat(E), torch.cat(sid), torch.cat(EL)
        os.makedirs(OUT, exist_ok=True); np.save(f'{OUT}/emb_{a.tag}_split{k}.npy', E.numpy().astype(np.float16))
        print(f'RESULT split {k} {a.tag}: in-batch K=4 neighbour purity {purity(E, sid, np.random.default_rng(0)):.3f} '
              f'vs deployed LDA fingerprint {purity(EL, sid, np.random.default_rng(0)):.3f} '
              f'(11 unseen participants, chance ~0.08) ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arch', default='both'); ap.add_argument('--tag', default=None)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--loss', default='am', choices=['am', 'supcon'])
    ap.add_argument('--eeg1', action='store_true')
    a = ap.parse_args(); a.tag = a.tag or a.arch
    m = Embedder(n_classes=3); x = torch.randn(8, 32, 120); e = m(x, eeg_fingerprint(x))
    assert e.shape == (8, 64) and torch.allclose(e.norm(dim=-1), torch.ones(8), atol=1e-5)
    m.loss(e, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])).backward()
    yy = torch.arange(64) % 11; ee = nn.functional.one_hot(yy, 64).float()
    assert supcon(ee, yy) < supcon(nn.functional.normalize(torch.randn(64, 64), dim=-1), yy), 'supcon must prefer clustered'
    main(a)
