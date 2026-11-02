"""AUG_LOG 2026-10-29 goal metric: the 6-member ensemble (lean TTA, as aug_ens.py) = a 5-member set + one candidate.
    add6.py <fold,...> <candidate> [...]      sets: base 5 and all-optx 5"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge import tta_greedy as g  # noqa: E402
from neurips_challenge.aug_ens import BASE  # noqa: E402

if __name__ == '__main__':
    folds = [int(f) for f in sys.argv[1].split(',')]
    cands = sys.argv[2:]
    OPTX = [m.replace('_things', '_optx_things') for m in BASE]
    g.POOL = BASE + OPTX + cands
    data = [g.centred(k) for k in folds]
    b5, o5 = list(range(5)), list(range(5, 10))
    print(f'folds {folds}: base5 {g.score(data, b5):.2f}  optx5 {g.score(data, o5):.2f}')
    for i, c in enumerate(cands, 10):
        print(f'  + {c:24s} base5+1 {g.score(data, b5 + [i]):.2f}  optx5+1 {g.score(data, o5 + [i]):.2f}', flush=True)
