"""Clean-proxy read of a trained arm against its baseline, fold k, new people x never-trained images, 5k and 14k
galleries, top-5 % r=3 / r=4 with the paired bootstrap gain over the baseline at the same epoch.
    nc05_eval_arm.py <fold> <arm recipe> <baseline recipe> <epochs e.g. 8,10,12> [test-time inputs: raw,K,batch,own]
test-time input (for key-free arms): raw | K (x - mean training key) | batch (x - 64-batch mean ERP) | own (x - oracle key)
Also scores 5 seen people on the same images (r=3, 5k) to show what the arm does to the person-specific part."""
import pandas as pd
from nc_lib import *

k, arm, base, epochs = int(sys.argv[1]), sys.argv[2], sys.argv[3], [int(e) for e in sys.argv[4].split(',')]
modes = sys.argv[5].split(',') if len(sys.argv) > 5 else ['raw']
NEW, SEENP = FOLDS[k], FOLDS[k % 4 + 1]
TRAIN = [s for s in range(1, 21) if s not in NEW]
keys = {s: key(s).to(DEV) for s in range(1, 21)}
K = torch.stack([keys[s] for s in TRAIN]).mean(0)
own = torch.stack([keys.get(s, K) for s in range(21)])
cells = {'new': cell(NEW), 'seen': cell(SEENP)}
gals = {'5k': gallery(), '14k': gallery(big=True)}


def run(recipe, ep, mode):
    f = load(recipe, k, f'checkpoint_ep{ep:02d}.pth' if glob.glob(f'{ROOT}/{recipe}/fold{k}/*/*/checkpoint_ep{ep:02d}.pth') else 'checkpoint_test_best.pth')
    out = {}
    for pn, (X, subj, img) in cells.items():
        sid = torch.as_tensor(subj).to(DEV)
        batch = lambda a, b: -torch.cat([torch.as_tensor(X[c:c + 64]).mean(0, keepdim=True).expand(len(X[c:c + 64]), -1, -1) for c in range(a, min(b, len(X)), 64)]).to(DEV)
        delta = {'raw': None, 'K': lambda a, b: -K, 'own': lambda a, b: -own[sid[a:b]], 'batch': batch}[mode]  # batch: 64-trial mean ERP, as predict() could compute it
        y = f(X, delta)
        out[pn] = {gn: score(y, subj, img, g, hits=True) for gn, g in (gals.items() if pn == 'new' else [('5k', gals['5k'])])}
    return out


rows = []
for ep in epochs:
    b = run(base, ep, 'raw')
    for mode in modes:
        a = run(arm, ep, mode)
        r = dict(epoch=ep, input=mode, seen_base=b['seen']['5k'][0]['r3'], seen_arm=a['seen']['5k'][0]['r3'])
        for gn in gals:
            for rr in ('r3', 'r4'):
                g, se = paired(a['new'][gn][1][rr], b['new'][gn][1][rr])
                r[f'{gn} {rr}'] = f"{b['new'][gn][0][rr]:.3f} -> {a['new'][gn][0][rr]:.3f} ({g:+.1f} +- {se:.1f} %)"
        rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/nc05_{arm}_f{k}.csv', index=False)
pd.set_option('display.width', 250); print(df.round(3).to_string())
