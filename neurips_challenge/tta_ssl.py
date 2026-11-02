"""Temporal-contrastive identity embedder for in-batch person grouping (TTA_LOG T27; Banville et al. 2021 style).
Positives: two training-image trials of the same person AND session recorded within DT seconds; negatives: trials of
OTHER people (same-person, other-time pairs are masked out of the denominator). Raw 32 x 120 trial -> small conv
encoder with mean / log-std statistics pooling -> 64-d. Trained on a fold's 15 Alljoined training people; scored by
in-batch purity@4 on the fold's grader stream (5 unseen people per 64-batch), vs the WCCN fingerprint.
    tta_ssl.py <fold> [--steps 4000] [--dt 60]
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.build_alljoined_cache import PARQUET_GLOB  # noqa: E402
from neurips_challenge.grader_sim import DATA, FOLDS  # noqa: E402
from neurips_challenge.tta_cache import OUT  # noqa: E402

DEV = torch.device('cuda')


def train_times(s):
    """train.npy row -> (session, absolute onset s) for the REAL trials (= build_alljoined_cache slot order)"""
    ev = pd.concat([pd.read_parquet(p, columns=['type', 'start', 'label', 'stim_file', 'session', 'run'])
                    for p in glob.glob(PARQUET_GLOB) if f'subject={s}-' in os.path.basename(p)])
    ev = ev[(ev.type == 'Image') & (ev.label == 'stim_train')].sort_values(['session', 'run', 'start'])
    counts = np.load(f'{DATA}/sub-{s:02d}/train_counts.npy'); seen = {}
    rows, sess, t = [], [], []
    run_off = {r: i * 1e5 for i, r in enumerate(sorted(set(zip(ev.session, ev.run))))}  # runs never mix in time
    for stim, se, ru, st in zip(ev.stim_file, ev.session, ev.run, ev.start):
        k = int(stim.split('.')[0]); o, i = k // 10, k % 10; r = seen.get((o, i), 0); seen[(o, i)] = r + 1
        if r < min(int(counts[o, i]), 4):
            rows.append((o * 10 + i) * 4 + r); sess.append(int(se)); t.append(run_off[(se, ru)] + float(st))
    return np.array(rows), np.array(sess), np.array(t)


class Enc(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(32, 96, 7, padding=3), nn.GELU(), nn.Conv1d(96, 96, 7, padding=3, stride=2), nn.GELU(),
                                  nn.Conv1d(96, 96, 5, padding=2, stride=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, x):
        x = (x - x.mean(-1, keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-6)  # trial-level amplitude only
        h = self.conv(x)
        return self.head(torch.cat([h.mean(-1), torch.log(h.std(-1) + 1e-4)], 1))


def load_people(people, n=4000, seed=0):
    X, P, S, T = [], [], [], []
    for p in people:
        rows, sess, t = train_times(p)
        pick = np.sort(np.random.default_rng([seed, p]).choice(len(rows), min(n, len(rows)), replace=False))
        x = np.load(f'{DATA}/sub-{p:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)
        X.append(torch.as_tensor(np.asarray(x[rows[pick]], np.float32)).half()); P.append(np.full(len(pick), p)); S.append(sess[pick]); T.append(t[pick])
        print(f'person {p}: {len(pick)} trials, {len(set(sess[pick]))} sessions', flush=True)
    return torch.cat(X), np.concatenate(P), np.concatenate(S), np.concatenate(T)


def main(k, steps, dt):
    people = [s for s in range(1, 21) if s not in FOLDS[k]]
    X, P, S, T = load_people(people)
    enc = Enc().to(DEV); opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-2)
    groups = {}  # (person, session) -> indices sorted by time
    for i in np.lexsort((T, S, P)):
        groups.setdefault((P[i], S[i]), []).append(i)
    keys = list(groups); rng = np.random.default_rng(0)
    for step in range(steps):
        a, b = [], []
        for g in rng.choice(len(keys), 128):  # anchor + a positive within dt seconds of the same session
            idx = groups[keys[g]]; j = rng.integers(len(idx)); tt = T[idx]
            near = [m for m in idx if abs(T[m] - T[idx[j]]) <= dt and m != idx[j]]
            if near:
                a.append(idx[j]); b.append(near[rng.integers(len(near))])
        ia, ib = np.array(a), np.array(b)
        za = Fn.normalize(enc(X[ia].to(DEV).float()), dim=1); zb = Fn.normalize(enc(X[ib].to(DEV).float()), dim=1)
        L = za @ zb.T / .1
        same_person = torch.as_tensor(P[ia][:, None] == P[ib][None, :], device=DEV)
        L = L.masked_fill(same_person & ~torch.eye(len(ia), dtype=torch.bool, device=DEV), -1e4)  # not negatives
        loss = Fn.cross_entropy(L, torch.arange(len(ia), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f'step {step} loss {loss.item():.3f}', flush=True)
    # purity on the fold's grader stream (raw trials by dataset idx), batch-centred cosine
    c = np.load(f'{OUT}/fold{k}.npz'); enc.eval(); Z = []
    with torch.no_grad():
        for s in np.unique(c['subj']):
            r = c['subj'] == s
            x = np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)
            Zs = torch.cat([enc(torch.as_tensor(np.asarray(x[c['idx'][r][b:b + 1024]], np.float32), device=DEV)) for b in range(0, r.sum(), 1024)])
            Z.append((np.where(r)[0], Zs))
        E = torch.empty(len(c['subj']), Z[0][1].shape[1], device=DEV)
        for rows, Zs in Z:
            E[torch.as_tensor(rows, device=DEV)] = Zs
        subj = torch.as_tensor(c['subj'], device=DEV); pur = []
        for b in range(0, len(E), 64):
            e = E[b:b + 64]; e = Fn.normalize(e - e.mean(0), dim=1); Sm = e @ e.T; Sm.fill_diagonal_(-9); s_ = subj[b:b + 64]
            pur.append((s_[Sm.topk(min(4, len(e) - 1), 1).indices] == s_[:, None]).float().mean(1))
    print(f'RESULT fold {k} ssl dt{dt} purity@4 {torch.cat(pur).mean():.3f}', flush=True)
    np.save(f'{OUT}/ssl_f{k}_dt{dt}.npy', E.cpu().numpy().astype(np.float16))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('fold', type=int); ap.add_argument('--steps', type=int, default=4000); ap.add_argument('--dt', type=float, default=60)
    a = ap.parse_args()
    main(a.fold, a.steps, a.dt)
