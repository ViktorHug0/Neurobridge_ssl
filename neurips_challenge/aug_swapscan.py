"""AUG_LOG: every candidate member in every slot of the base 5-set (lean TTA), on the given folds; sorted by score.
    aug_swapscan.py <fold,...> <candidate> [...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge import tta_greedy as g  # noqa: E402
from neurips_challenge.aug_ens import BASE  # noqa: E402

folds = [int(f) for f in sys.argv[1].split(',')]
cands = sys.argv[2:]
g.POOL = BASE + cands
data = [g.centred(k) for k in folds]
base = g.score(data, list(range(5)))
rows = [(g.score(data, [5 + i if j == s else j for j in range(5)]), c, BASE[s]) for i, c in enumerate(cands) for s in range(5)]
print(f'folds {folds}: base {base:.2f}')
for sc, c, slot in sorted(rows, reverse=True)[:25]:
    print(f'  {sc:6.2f} ({sc - base:+.2f})  {c:28s} in {slot} slot')
