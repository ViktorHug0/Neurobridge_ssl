"""E14 are the test runs harder because their mean response (key) differs from the training runs'?
Per held-out person of fold k: k_train = mean ERP of its training-split trials, k_test = mean ERP of its test-run trials.
1. how the two keys differ (norm, correlation, best lag).
2. causal test through the frozen members (d6 + rmix + SQF-X): test-run trials with the test-run key replaced by the
   person's training-run key (x - k_test + k_train), and training-run trials (never-trained images) given the test-run
   key; top-5 % r=1 / r=3, exact ranks."""
from nc_lib import *
from nd03_cross_dataset_lib import shift
k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
L = ('long', 'mid')[k != 1]
P = FOLDS[k]
Xt = np.concatenate([np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r')[:, 0].reshape(-1, 32, 120) for s in P])
st = np.repeat(P, 16000); it = np.tile(np.repeat(np.arange(200), 80), 5)
o = np.random.default_rng(11).permutation(len(Xt)); Xt, st, it = Xt[o], st[o], it[o]
ktr = {s: key(s).to(DEV) for s in P}; kte = {s: torch.tensor(Xt[st == s].mean(0)).to(DEV) for s in P}
c = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
print('person: |k_train| |k_test| corr | corr by shift of k_test (-3..+3 samples) | |k_test - k_train|')
for s in P:
    print('  sub-%02d  %.2f  %.2f  %.3f | %s | %.2f' % (s, ktr[s].norm(), kte[s].norm(), c(ktr[s], kte[s]), ' '.join('%.2f' % c(ktr[s][:, 12:108], shift(kte[s], n)[:, 12:108]) for n in range(-3, 4)), (kte[s] - ktr[s]).norm()))
# split-half reliability of each key for scale: two halves of the test trials, two halves of the training trials are not cached; use test halves
for s in P[:2]:
    x = Xt[st == s]; print('  sub-%02d test-run key, two halves of the trials: corr %.3f' % (s, c(torch.tensor(x[::2].mean(0)), torch.tensor(x[1::2].mean(0)))))
T = nz(torch.tensor(np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/image_test.npy').reshape(200, 1536)).float()).to(DEV)
galt = torch.cat([T, gallery()]); gal = gallery()
F = f'{R}/logs/holdout/f{k}'; sh_ = np.load(f'{F}/stream.npz'); Xh, sh, ih = np.load(f'{F}/X.npy', mmap_mode='r'), sh_['subj'], sh_['img']
fs = [load(f'reve_d6_ho5k_{L}_optx_things', k, 'checkpoint_ep10.pth'), load(f'rmix_reg_ho5k_{L}_optx_things', k, 'checkpoint_ep10.pth'), load('sqf_dbn_ho5k_long_optx_things', k)]
groups = {}
for i, kk in enumerate(zip(st.tolist(), it.tolist())):
    groups.setdefault(kk, []).append(i)
g3 = torch.as_tensor(np.array([v[j:j + 3] for v in groups.values() for j in range(0, 78, 3)])); tt = torch.as_tensor(it)
D = lambda sign: torch.stack([sign * (ktr[s] - kte[s]) if s in ktr else torch.zeros(32, 120, device=DEV) for s in range(21)])
def run_test(name, delta):
    sid = torch.as_tensor(st).to(DEV)
    y = sum(scaled(f(Xt, None if delta is None else (lambda a, b: delta[sid[a:b]]))) for f in fs)
    h1, h3 = (rank(y, tt, galt) < 5).numpy(), (rank(y[g3].mean(1), tt[g3[:, 0]], galt) < 5).numpy()
    print('test runs, %-46s r=1 %.3f  r=3 %.3f' % (name, 100 * h1.mean(), 100 * h3.mean()), flush=True); return h1
def run_held(name, delta):
    sid = torch.as_tensor(sh).to(DEV)
    y = sum(scaled(f(Xh, None if delta is None else (lambda a, b: delta[sid[a:b]]))) for f in fs)
    s = score(y, sh, ih, gal); print('training runs (never-trained images), %-26s r=1 %.3f  r=3 %.3f' % (name, s['r1'], s['r3']), flush=True)
b = run_test('as recorded', None)
for name, d in (('test key -> training key', D(1)), ('half-way', .5 * D(1)), ('test key removed', torch.stack([-kte[s] if s in kte else torch.zeros(32, 120, device=DEV) for s in range(21)]))):
    h = run_test(name, d); print('    vs as recorded (r=1, trials resampled): %+.1f +- %.1f %%' % paired(h, b))
run_held('as recorded', None); run_held('training key -> test key', D(-1))
