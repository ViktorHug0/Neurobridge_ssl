"""N04 epoch stitching: where does the late-epoch loss on new images sit? REVE-d6 (ho5k, fold 1), parts = embedding,
   6 blocks, readout (final_layer + head). hybrid(j) = the first j parts from checkpoint A, the rest from B.
N05 key amplification (ep10): x + a (m - K) for the person's own key m: does a stronger person cue help?
top-5 % r=3, 5k gallery of never-trained images, new and seen people."""
import pandas as pd
from nc_lib import *

k = 1
NEW, SEENP = FOLDS[k], FOLDS[2]
TRAIN = [s for s in range(1, 21) if s not in NEW]
cells = {'new': cell(NEW), 'seen': cell(SEENP)}
gal = gallery()
f = load('d6', k, 'checkpoint_ep10.pth')
rows = []
if 'amp' in sys.argv[1:] or len(sys.argv) == 1:
    keys = {s: key(s).to(DEV) for s in range(1, 21)}
    K = torch.stack([keys[s] for s in TRAIN]).mean(0)
    dev = torch.stack([keys.get(s, K) - K for s in range(21)])
    for pn, (X, subj, img) in cells.items():
        sid = torch.as_tensor(subj).to(DEV)
        for a in (-0.5, 0.25, 0.5, 1.0, 2.0):
            rows.append(dict(exp='amp', person=pn, a=a, r3=score(f(X, lambda i, j: a * dev[sid[i:j]]), subj, img, gal)['r3']))
            print(rows[-1], flush=True)
if 'stitch' in sys.argv[1:] or len(sys.argv) == 1:
    d = os.path.dirname(sorted(glob.glob(f'{ROOT}/{MEMBERS["d6"][0]}/fold{k}/*/*/track1_score.json'))[-1])
    sd = {e: torch.load(f'{d}/checkpoint_ep{e:02d}.pth', map_location=DEV, weights_only=False)['model_state_dict'] for e in {int(e) for e in os.environ.get('PAIRS', '16:10,10:16').replace(',', ':').split(':')}}
    part = lambda n: 0 if 'transformer.layers' not in n and not n.startswith(('fm.final_layer', 'head')) else \
        7 if n.startswith(('fm.final_layer', 'head')) else 1 + int(n.split('.')[3])
    for A, B in (tuple(int(e) for e in pr.split(':')) for pr in os.environ.get('PAIRS', '16:10,10:16').split(',')):
        for j in (int(j) for j in os.environ.get('JS', '0,1,2,3,4,5,6,7,8').split(',')):
            f.model.load_state_dict({n: (sd[A] if part(n) < j else sd[B])[n] for n in sd[A]})
            r = dict(exp='stitch', first=A, rest=B, j=j)
            for pn, (X, subj, img) in cells.items():
                r[pn] = score(f(X), subj, img, gal)['r3']
            rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/nc03_{"_".join(sys.argv[1:]) or "all"}{os.environ.get("TAG", "")}.csv', index=False)
print(pd.DataFrame(rows).round(3).to_string())
