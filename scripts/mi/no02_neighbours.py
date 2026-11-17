"""O02 the RSVP neighbours inside each epoch: do they leak into the output, and what do they cost?
Alljoined shows 20 images in a row at ~4.7 per second (SOA ~0.215 s), then a 1.4 s gap. The epoch (-0.2 .. 0.8 s) of image
j therefore also holds the responses to image j-1 (onset -0.215 s) and to j+1, j+2, j+3 (+0.215, +0.43, +0.645 s). At 80
repeats these average out (the neighbours differ across repeats); at ~3 repeats they do not.
New people of fold k, never-trained images, single trials, lag ensemble (cached member outputs):
  (a) leak: cosine of the output to the embedding of each neighbour image, against a random image, as a share of the
      cosine to the target; and how often the NEXT / PREVIOUS image would be retrieved from this epoch (5k gallery, when
      that neighbour is itself a never-trained image);
  (b) natural knock-out: target retrieval (r=1 top-5, exact ranks, 5k) by position in the 20-image sequence: the first
      image has no predecessor, the last ones have fewer or no successors.
    no02_neighbours.py <fold>"""
import pandas as pd
from nc_lib import *

k = int(sys.argv[1]); BIDS = '/nasbrain/p20fores/alljoined/nm000134'
F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz')
# the stream's repeat index (holdout_eval.py order: people, then np.nonzero over (image, repeat), then one permutation)
subj, img, rep = [], [], []
for s in FOLDS[k]:
    x = np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)
    gi, r = np.nonzero(np.stack([np.abs(np.asarray(x[i])).sum((1, 2)) > 0 for i in HELD]))
    subj += [s] * len(gi); img += gi.tolist(); rep += r.tolist()
o = np.random.default_rng(k * 100 + 7).permutation(len(img)); subj, img, rep = np.array(subj)[o], np.array(img)[o], np.array(rep)[o]
assert (subj == st['subj']).all() and (img == st['img']).all()
# events: per person, (image id, chronological occurrence) -> neighbours and position in the sequence
NB = {}
for s in FOLDS[k]:
    seen = {}
    for f in sorted(glob.glob(f'{BIDS}/sub-{s:02d}/ses-*/eeg/*_events.tsv')):
        d = pd.read_csv(f, sep='\t'); d = d[d.trial_type.str.startswith('stim')]
        v = d['value'].to_numpy(); on = d['onset'].to_numpy(); tr = (d['trial_type'] == 'stim_train').to_numpy()
        start = np.r_[True, np.diff(on) > .6]; seq = np.cumsum(start) - 1; pos = np.arange(len(v)) - np.maximum.accumulate(np.where(start, np.arange(len(v)), 0))
        n = np.bincount(seq)[seq]
        for i in range(len(v)):
            if not tr[i]:
                continue
            r = seen.get(v[i], 0); seen[v[i]] = r + 1
            nb = [v[i + d] if 0 <= i + d < len(v) and seq[i + d] == seq[i] and tr[i + d] else -1 for d in (-2, -1, 1, 2, 3)]
            NB[(s, v[i], r)] = (pos[i], n[i] - 1 - pos[i], *nb)
T = np.array([NB.get((s, HELD[i], r), (-1, -1, -1, -1, -1, -1, -1)) for s, i, r in zip(subj, img, rep)])
print('trials', len(T), 'with events', (T[:, 0] >= 0).mean().round(4), flush=True)
Y = lambda r, e: scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')).float())
y = nz(sum(Y(r, e) for r, e in (('d6', 12), ('rmix', 10), ('sqfx', 40), ('scr', 12), ('sqf', 32)))).to(DEV)
E = nz(torch.tensor(np.asarray(TARGETS)).float()).to(DEV); gal = gallery(); held_pos = torch.full((16540,), -1, dtype=torch.long); held_pos[HELD] = torch.arange(5000)
cos = lambda ids: (y * E[torch.as_tensor(ids).to(DEV)]).sum(1).cpu().numpy()
g = np.random.default_rng(0); c_t = cos(HELD[img]); c_r = cos(g.integers(0, 16540, len(img)))
print('(a) cosine to the target %.4f, to a random image %.4f' % (c_t.mean(), c_r.mean()))
S = torch.cat([(y[b:b + 8192] @ gal.T).cpu() for b in range(0, len(y), 8192)])  # CPU: the GPU is shared
rank_of = lambda pos_in_gal: (S > S.gather(1, pos_in_gal[:, None])).sum(1).numpy()
r_t = rank_of(torch.as_tensor(img))
print('    target top-5 (r=1): %.3f %%' % (100 * (r_t < 5).mean()))
for name, col in (('j-2', 2), ('j-1', 3), ('j+1', 4), ('j+2', 5), ('j+3', 6)):
    ok = T[:, col] >= 0; ids = T[ok, col]
    c = cos(np.where(ok, T[:, col], 0))[ok]
    hp = held_pos[torch.as_tensor(ids)]; h = (hp >= 0).numpy()  # the neighbour is itself a never-trained image: retrievable in the same gallery
    Sh = S[torch.as_tensor(np.nonzero(ok)[0][h])]; rk = (Sh > Sh.gather(1, hp[h][:, None])).sum(1).numpy()
    print('    neighbour %s: cosine %.4f = %+.1f %% of the target excess (trained neighbours %+.1f %%, never-trained %+.1f %%) | retrieved from this epoch top-5 %.3f %% (chance 0.10, n %d)'
          % (name, c.mean(), 100 * (c.mean() - c_r.mean()) / (c_t.mean() - c_r.mean()), 100 * (c[~h].mean() - c_r.mean()) / (c_t.mean() - c_r.mean()),
             100 * (c[h].mean() - c_r.mean()) / (c_t.mean() - c_r.mean()), 100 * (rk < 5).mean(), h.sum()), flush=True)
print('(b) target retrieval by position in the sequence (r=1 top-5 %, cosine excess over random, n)')
ex = c_t - c_r.mean()
for name, m in [('first image (no predecessor)', T[:, 0] == 0), ('second', T[:, 0] == 1), ('middle (>= 3 before, >= 3 after)', (T[:, 0] >= 3) & (T[:, 1] >= 3)),
                ('third from last (2 successors)', T[:, 1] == 2), ('second from last (1 successor)', T[:, 1] == 1), ('last image (no successor)', T[:, 1] == 0)]:
    hit = r_t[m] < 5; se = 100 * hit.std() / max(len(hit), 1) ** .5
    print('    %-34s %.3f +- %.3f   excess %.4f   n %d' % (name, 100 * hit.mean(), se, ex[m].mean(), m.sum()), flush=True)
np.save(f'{OUT}/o02_seq_f{k}.npy', T)
