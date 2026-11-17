"""D11 both lag-trained members together (fold 1, logs/holdout/f1 stream, cached outputs): ens5 and the pair
rmix + 0.5 SQF-X with rmix (ep10) and SQF-X (ep40) replaced by their --things_lag 5 versions. Paired bootstrap."""
import pandas as pd
from nc_lib import *

F = f'{R}/logs/holdout/f1'
st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float())
D6, RM, SX, SC, SP = (Y(f'{F}/{m}.npy') for m in ('reve_d6_ho5k_long_optx_things__ep10', 'rmix_reg_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things',
                                                 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'))
RL, SL = Y(f'{OUT}/hold_f1_mi_rmix_ho5k_lag5_ep10_raw.npy'), Y(f'{OUT}/hold_f1_mi_sqfx_ho5k_lag5_ep40_raw.npy')
rows = {'ens5': (D6 + RM + SX + SC + SP, None), 'ens5, both lag': (D6 + RL + SL + SC + SP, 'ens5'),
        'pair': (RM + .5 * SX, None), 'pair, both lag': (RL + .5 * SL, 'pair'), 'pair, both lag vs ens5': (RL + .5 * SL, 'ens5')}
out = []
for gn, g in (('5k', gallery()), ('14k', gallery(big=True))):
    res = {n: score(y, subj, img, g, hits=True) for n, (y, _) in rows.items()}
    for n, (_, ref) in rows.items():
        s, h = res[n]
        out.append(dict(gallery=gn, row=n, r3=s['r3'], r4=s['r4'], **({} if ref is None else {f'gain {rr} %': '%+.1f +- %.1f' % paired(h[rr], res[ref][1][rr]) for rr in ('r3', 'r4')})))
print(pd.DataFrame(out).fillna('').round(3).to_string(index=False))
