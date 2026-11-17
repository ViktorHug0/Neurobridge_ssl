"""Ensemble-level clean read of a new arm (fold 1; the stream and member caches of logs/holdout/f1, scaled sum as in
submission.py): ens5 @ sub-6 budgets (d6 10, rmix 10, sqfX, scr 10, sqf 32) vs the arm swapped for the member it derives from (ROLE=d6|rmix|sqfx) / added,
and the pair rmix + 0.5 x SQF-X with the arm swapped in. Gains are against the row's own reference.
top-5 % r=3 / r=4, 5k and 14k galleries, paired bootstrap vs ens5.
    nc06_ens.py <arm recipe> <epoch> [input: raw | K | batch]"""
import pandas as pd
from nc_lib import *

k, arm, ep, mode = int(os.environ.get('K', 1)), sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'raw'
F = f'{R}/logs/holdout/f{k}'
st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
X = np.load(f'{F}/X.npy', mmap_mode='r')
Y = lambda m: scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float())
NAMES = {1: ('reve_d6_ho5k_long_optx_things__ep10', 'rmix_reg_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         3: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_mid_optx_things', 'sqf_dbn_ho5k_long_things__ep32')}
D6, RM, SX, SC, SP = (Y(m) for m in NAMES[k])
p = f'{OUT}/hold_f{k}_{arm}_ep{ep:02d}_{mode}.npy'
if not os.path.exists(p):
    f = load(arm, k, f'checkpoint_ep{ep:02d}.pth' if glob.glob(f'{ROOT}/{arm}/fold{k}/*/*/checkpoint_ep{ep:02d}.pth') else 'checkpoint_test_best.pth')
    Kk = torch.stack([key(s) for s in range(1, 21) if s not in FOLDS[k]]).mean(0).to(DEV)
    delta = {'raw': None, 'K': lambda a, b: -Kk,
             'batch': lambda a, b: -torch.cat([torch.as_tensor(np.asarray(X[c:c + 64])).mean(0, keepdim=True).expand(len(X[c:c + 64]), -1, -1) for c in range(a, min(b, len(X)), 64)]).to(DEV)}[mode]
    np.save(p, f(X, delta, bs=512).cpu().numpy().astype(np.float16))  # bs is a multiple of 64: batch means follow the 64-grid
A = scaled(torch.as_tensor(np.load(p)).float())
ROLE = os.environ.get('ROLE', 'd6')  # which member the arm is a variant of: d6 | rmix | sqfx
mem = {'d6': D6, 'rmix': RM, 'sqfx': SX, 'scr': SC, 'sqf': SP}
ens = lambda **sw: sum({**mem, **sw}.values())
rows = {f'{ROLE} alone': mem[ROLE], 'arm alone': A, 'ens5': ens(), f'ens5, arm for {ROLE}': ens(**{ROLE: A}), 'ens5 + arm': ens() + A,
        'rmix + .5 sqfx': RM + .5 * SX, 'pair, arm in': (A + .5 * SX) if ROLE != 'sqfx' else (RM + .5 * A)}
REF = {'arm alone': f'{ROLE} alone', f'ens5, arm for {ROLE}': 'ens5', 'ens5 + arm': 'ens5', 'pair, arm in': 'rmix + .5 sqfx'}
out = []
for gn, g in (('5k', gallery()), ('14k', gallery(big=True))):
    res = {n: score(y, subj, img, g, hits=True) for n, y in rows.items()}
    for n, (s, h) in res.items():
        ref = res[REF.get(n, n)][1]
        out.append(dict(gallery=gn, row=n, r3=s['r3'], r4=s['r4'], **{f'gain {rr} %': '%+.1f +- %.1f' % paired(h[rr], ref[rr]) for rr in ('r3', 'r4')}))
df = pd.DataFrame(out); df.to_csv(f'{OUT}/nc06_f{k}_{arm}_ep{ep:02d}_{mode}.csv', index=False)
pd.set_option('display.width', 200); print(df.round(3).to_string(index=False))
