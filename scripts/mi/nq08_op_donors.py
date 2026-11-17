"""Q08 other cohorts as DONORS of recording operators (no image label needed from them).
Operator transfer (module/augment.py) re-renders a trial of person s through person t's channel mixing,
x -> R_t^(1/2) R_s^(-1/2) x, with R the 32 x 32 channel covariance; t is drawn among the people of the batch (25 here).
A donor is a person of another cohort, exported through the grader chain (nq02 MSS: 32 people, 9 electrodes interpolated;
nq06 ds007964: 19 people, all 32 recorded): only R_t^(1/2) is kept, from all the person's trials, shrink .05 as in training.
    nq08_op_donors.py build                 -> <OUT>/op_donors.npz: S (K, 32, 32) float64, name (K,); op_donors_ds.npz, op_donors_mss.npz: S of one cohort
    nq08_op_donors.py check <recipe> <fold> <checkpoint>
check = does a frozen member already cope with these operators? Held-out people, never-trained images (5k, r=3): intact /
each person re-rendered through another HELD-OUT Alljoined person / through a ds007964 donor / through an MSS donor.
A drop under donors that is larger than under Alljoined people = operators the training set does not cover."""
import glob
from nc_lib import *

SRC = {'ds': '/nasbrain/p20fores/ds007964/export/sub-*.npz', 'mss': '/nasbrain/p20fores/mss_ds005589/export/sub-*.npz'}


def roots(X, shrink=0.05):
    """R^(1/2), R^(-1/2) of the channel covariance of trials X (n, 32, T), as cross_subject_operator_transfer."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(DEV).double(); c = torch.einsum('nct,ndt->cd', X, X) / (X.shape[0] * X.shape[-1])
    c = (1 - shrink) * c + shrink * torch.eye(len(c), device=DEV, dtype=torch.float64) * torch.diagonal(c).mean(); e, v = torch.linalg.eigh(c); e = e.clamp_min(1e-8)
    return (v * e.sqrt()) @ v.T, (v * e.rsqrt()) @ v.T


if sys.argv[1] == 'build':
    S, name = [], []
    for tag, pat in SRC.items():
        people = {}
        for p in sorted(glob.glob(pat)):
            people.setdefault(os.path.basename(p)[:6], []).append(p)  # MSS: the sessions of a person pooled
        for who, files in people.items():
            S.append(roots(np.concatenate([np.load(p)['X'] for p in files]))[0].cpu().numpy()); name.append(f'{tag}:{who}')
    S, name = np.stack(S), np.array(name); np.savez(f'{OUT}/op_donors.npz', S=S, name=name)
    for tag in SRC:  # one file per cohort for --op_donors
        np.savez(f'{OUT}/op_donors_{tag}.npz', S=S[[n.startswith(tag) for n in name]])
    print('donors:', len(name), {t: sum(n.startswith(t) for n in name) for t in SRC}, 'mean diagonal %.3f' % np.mean([np.diag(s).mean() for s in S]))
else:
    recipe, k, ck = sys.argv[2], int(sys.argv[3]), sys.argv[4]; f = load(recipe, k, ck); gal = gallery(); X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X)
    d = np.load(f'{OUT}/op_donors.npz'); D = {t: torch.as_tensor(d['S'][[n.startswith(t) for n in d['name']]]).to(DEV) for t in SRC}; rng = np.random.default_rng(0)
    own = {s: roots(X[subj == s]) for s in FOLDS[k]}; base = None
    for cond in ('intact', 'alljoined', 'ds', 'mss'):
        Y = X.clone()
        for s in FOLDS[k] if cond != 'intact' else []:
            T = own[rng.choice([j for j in FOLDS[k] if j != s])][0] if cond == 'alljoined' else D[cond][rng.integers(len(D[cond]))]
            m = torch.as_tensor(subj == s); Y[m] = torch.einsum('cd,ndt->nct', (T @ own[s][1]).float().cpu(), X[m])
        y = f(Y, bs=1000); sc, h = score(y, subj, img, gal, hits=True); P = 1 - rank(scaled(y), img, gal).float().mean().item() / 4999; base = base or (h['r3'], P)
        print('%-10s top-5 r=3 %.3f (%+.1f +- %.1f %%) | single-trial P(target > random) %.4f (%+.4f) | mean |x| %.3f' % (cond, sc['r3'], *paired(h['r3'], base[0]), P, P - base[1], Y.abs().mean()), flush=True)
