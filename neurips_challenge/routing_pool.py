"""Routing over an expert pool: the 40 affinity models (5-subject REVE-6) as experts, each target j served by the 30
that never saw it. Compares uniform mixing, oracle weights (fit on half the images, scored on the other half) and
label-free weights (ERP similarity between j and each expert's training subjects). Uses the saved per-(subject,
image) mean predictions, so it is CPU-only. Usage: routing_pool.py"""
import glob, json, re, sys
import numpy as np
sys.path.insert(0, 'neurips_challenge')
from affinity_summary import ROOT

subsets = [list(map(int, l.split())) for l in open('neurips_challenge/affinity_subsets.txt') if not l.startswith('#')]


def load_pool():
    preds, gallery = {}, None
    for f in glob.glob(f'{ROOT}/s*/reve/2026*/sealed_per_epoch.jsonl'):
        m = int(re.search(r'/s(\d+)/', f).group(1)) - 1
        ev = [json.loads(l) for l in open(f)]
        ep = ev[int(np.argmax([np.mean(list(e['by_subject'].values())) for e in ev]))]['epoch']
        z = np.load(f.replace('sealed_per_epoch.jsonl', f'sealed_preds_ep{ep:03d}.npz'))
        preds[m] = {int(k[1:]): z[k].astype(np.float32) / float(z['mean_norm']) for k in z.files if k != 'mean_norm'}
        if gallery is None:
            gallery = np.load(f.replace('sealed_per_epoch.jsonl', 'sealed_gallery.npy'))
    return preds, gallery


def top5(P, G, rows=None):
    rows = np.arange(len(P)) if rows is None else rows
    S = P[rows] @ G.T
    return 100 * np.mean([(S[k] > S[k, r]).sum() < 5 for k, r in enumerate(rows)])


if __name__ == '__main__':
    preds, G = load_pool()
    rng = np.random.default_rng(0)
    halves = np.split(rng.permutation(200), 2)
    out = {k: [] for k in ('single', 'uniform', 'oracle', 'oracle_top5')}
    for j in range(1, 21):
        ms = [m for m in preds if j in preds[m]]
        P = np.stack([preds[m][j] for m in ms])                    # (experts, 200, D)
        acc = np.array([top5(p, G) for p in P])
        out['single'].append(acc.mean()); out['uniform'].append(top5(P.sum(0), G))
        o, o5 = [], []
        for a, b in (halves, halves[::-1]):
            fit = np.array([top5(p, G, a) for p in P])             # expert quality on images a only
            w = np.exp((fit - fit.max()) / 2.0)
            o.append(top5(np.tensordot(w, P, 1), G, b))
            o5.append(top5(P[np.argsort(-fit)[:5]].sum(0), G, b))
        out['oracle'].append(np.mean(o)); out['oracle_top5'].append(np.mean(o5))
    base = np.array(out['uniform'])
    for k, v in out.items():
        v = np.array(v); print(f'{k:12s} {v.mean():6.2f}   vs uniform {v.mean() - base.mean():+.2f}, better on {(v > base).sum()}/20')
