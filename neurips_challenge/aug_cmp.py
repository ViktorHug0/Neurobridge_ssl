"""AUG_LOG screens: per-epoch fold mean of a recipe vs its base recipe (same fold, sealed rule, half trials).
    aug_cmp.py <base> <recipe> [recipe ...] [--fold k]"""
import glob, json, sys
import numpy as np
R = '/nasbrain/p20fores/Neurobridge_SSL/results/alljoined/track1/cv4'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
fold = int(sys.argv[sys.argv.index('--fold') + 1]) if '--fold' in sys.argv else 1
args = [a for a in args if a != str(fold)]


def curve(r):
    f = sorted(glob.glob(f'{R}/{r}/fold{fold}/*/*/sealed_per_epoch.jsonl'))
    return [np.mean(list(json.loads(l)['by_subject'].values())) for l in open(f[-1])] if f else []


base = curve(args[0])
print(f'{args[0]:28s} ' + ' '.join(f'{v:5.1f}' for v in base) + f' | best {max(base):.1f}')
for r in args[1:]:
    c = curve(r)
    if not c:
        print(f'{r:28s} (no epochs yet)'); continue
    d = [a - b for a, b in zip(c, base)]
    print(f'{r:28s} ' + ' '.join(f'{v:5.1f}' for v in c) + f' | best {max(c):.1f} ({max(c) - max(base):+.1f}) matched mean {np.mean(d):+.2f}')
