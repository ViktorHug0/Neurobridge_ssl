"""H01 is the noise level what still separates THINGS-EEG2 people from Alljoined people after the lag?
THINGS-EEG2 trials (shifted 5 samples later) are scaled by g and mixed with real Alljoined noise: the half-difference of
two repeats of one (person, image), (a - b) / sqrt 2, which has the single-trial noise variance and no evoked response
or key. g = 0.31 brings the THINGS evoked amplitude (key norm 6.8) to Alljoined's (2.1).
(a) instrument = Alljoined-ONLY REVE-d6 (fold 3, ep10): five THINGS people it never saw, against its own held-out
    Alljoined people. If noise-matched THINGS people land at the held-out level, they are Alljoined-like people.
(b) the joint lag model (mi_d6_ho5k_lag5, fold 1): THINGS training people, against Alljoined training people.
Never-trained images, 5k gallery, exact ranks, top-5 % r=1 / r=3 / r=4."""
from nc_lib import *
from nd03_cross_dataset_lib import shift

gal = gallery()


def noise_pool(people, n_img=2500, seed=3):
    im = np.sort(np.random.default_rng(seed).choice(16540, n_img, replace=False)); out = []
    for s in people:
        x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)[im])
        ok = np.abs(x).sum((2, 3)).min(1) > 0
        out += [(x[ok, 0] - x[ok, 1]) / 2 ** .5, (x[ok, 2] - x[ok, 3]) / 2 ** .5]
    return torch.as_tensor(np.concatenate(out))


def run(f, name, X, subj, img, g=None, N=None):
    if g is None:
        d = lambda a, b: shift(torch.as_tensor(X[a:b]).to(DEV), 5) - torch.as_tensor(X[a:b]).to(DEV) if N == 'lag' else None
        d = d if N == 'lag' else None
    else:
        idx = torch.as_tensor(np.random.default_rng(0).integers(0, len(N), len(X)))
        d = lambda a, b: g * shift(torch.as_tensor(X[a:b]).to(DEV), 5) + N[idx[a:b]].to(DEV) - torch.as_tensor(X[a:b]).to(DEV)
    sc = score(f(X, d), subj, img, gal); print('%-58s %.3f / %.3f / %.3f' % (name, sc['r1'], sc['r3'], sc['r4']), flush=True)


for title, f, ref_people, noise_people in (('(a) Alljoined-only REVE-d6, fold 3', load('reve_d6_ho5k_mid_optx_aj', 3, 'checkpoint_ep10.pth'), FOLDS[3], FOLDS[3]),
                                             ('(b) joint lag REVE-d6, fold 1', load('mi_d6_ho5k_lag5', 1, 'checkpoint_test_best.pth'), FOLDS[2], FOLDS[2])):
    print(title, flush=True)
    X, subj, img = cell(ref_people); run(f, 'Alljoined reference people', X, subj, img); del X
    N = noise_pool(noise_people); print('  noise pool', tuple(N.shape), 'rms %.2f' % N.pow(2).mean().sqrt(), flush=True)
    X, subj, img = cell([21, 23, 25, 27, 29]); print('  THINGS trial rms %.2f' % float(np.sqrt((X[:20000] ** 2).mean())), flush=True)
    run(f, 'THINGS people, lag only', X, subj, img, N='lag')
    for g in (1., .5, .31, .2):
        run(f, f'THINGS people, lag, x{g} + Alljoined noise', X, subj, img, g, N)
    del X, N, f; torch.cuda.empty_cache()
