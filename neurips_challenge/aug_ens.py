"""AUG_LOG: the number that counts -- the lean-TTA 5-member ensemble (K4 c1, one d = p/(p+n) shrink of the summed
person-centred outputs, tta_greedy.py) with ONE member replaced by an augmented variant, on the folds both are cached for.
    aug_ens.py <fold,...> <base member>=<variant> [...]     e.g. aug_ens.py 1 reve_d6_things=reve_d6_xmix_things
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge import tta_greedy as g  # noqa: E402

BASE = ['reve_d6_things', 'sqf_dbn_things', 'tsc_dbn40_bn64_things', 'rmix_reg_things', 'reve_scratch_things']
if __name__ == '__main__':
    folds = [int(f) for f in sys.argv[1].split(',')]
    swaps = dict(a.split('=') for a in sys.argv[2:])
    g.POOL = BASE + list(swaps.values())
    data = [g.centred(k) for k in folds]
    alt = [g.POOL.index(swaps.get(m, m)) for m in BASE]
    print(f'folds {folds}: base {g.score(data, list(range(5))):.2f} -> swapped {g.score(data, alt):.2f}  ({swaps})')
    for m, v in swaps.items():  # each swap alone
        one = [g.POOL.index(v if b == m else b) for b in BASE]
        print(f'  {m} -> {v} alone: {g.score(data, one):.2f}')
