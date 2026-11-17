"""U04 the auxiliary-target members STACKED in the six: reference six (d6 jitter, rmix lag, SQF-X jitter, REVE-scratch, SQF,
LaBraM; each at its packaged checkpoint) against the same six with every available `<recipe><suffix>` twin swapped in, one
at a time and all together. Held-out people x 5,000 never-trained images, 5k gallery, r=3; top-5 / top-100 / ranking error,
paired. Outputs cached as in nu01 (<OUT>/u01_f<k>_<recipe>_<checkpoint>.npy).
    [TWINS=slot:suffix,...] [REF=slot:recipe[:ckpt],...] [SWAP=slot:recipe[:ckpt],...] nu04_stack.py <fold> [suffix, default _auxlow]   (TWINS picks the slots and a suffix per slot: 0 d6, 1 rmix, 2 SQF-X, 3 REVE-scratch, 4 SQF, 5 LaBraM)"""
from nc_lib import *

k = int(sys.argv[1]); suf = sys.argv[2] if len(sys.argv) > 2 else '_auxlow'
SIX = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5' + os.environ.get('RMIX', ''), 'checkpoint_ep10.pth'),  # RMIX=_jit3: the packaged rmix recipe (sub16)
       ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
       ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]
X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery(); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g3[:, 0]]; rk = lambda y: rank(y[g3].mean(1), tg, gal).numpy()


def out(r, c):
    p = f'{OUT}/u01_f{k}_{r}_{c}.npy'
    if not os.path.exists(p):
        f = load(r, k, c); np.save(p, f(X, bs=int(os.environ.get('BS', 1000))).numpy().astype(np.float16)); del f; torch.cuda.empty_cache()
    return scaled(torch.as_tensor(np.load(p).astype(np.float32)))


def versus(r, b):
    bi = np.random.default_rng(0).integers(0, len(b), (1000, len(b))); o = []
    for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 4999.):
        a, c = fn(r).astype(float), fn(b).astype(float); o.append('%+5.1f +- %.1f' % (100 * (a.mean() / c.mean() - 1), 100 * (a[bi].mean(1) / c[bi].mean(1) - 1).std()))
    return 'top-5 %.3f %% (%s) | top-100 %s | ranking error %s' % (100 * (r < 5).mean(), *o)


SIX = [(r if glob.glob(f'{ROOT}/{r}/fold{k}/*/*/{c}') else r.replace('_jit3', ''), c) for r, c in SIX]  # fold 2: SQF-X exists without jitter only
for a in filter(None, os.environ.get('REF', '').split(',')):  # REF=slot:recipe[:checkpoint]: another reference in that slot
    j, r, *c = a.split(':'); SIX[int(j)] = (r, c[0] if c else SIX[int(j)][1])
ref = [out(r, c) for r, c in SIX]; base = rk(sum(ref)); have = {}
TW = {int(a.split(':')[0]): a.split(':')[1] for a in os.environ['TWINS'].split(',')} if os.environ.get('TWINS') else {j: suf for j in range(6)}  # slot -> suffix
for j, (r, c) in enumerate(SIX):
    if j in TW and glob.glob(f'{ROOT}/{r}{TW[j]}/fold{k}/*/*/{c}'):
        have[j] = out(r + TW[j], c)
for a in filter(None, os.environ.get('SWAP', '').split(',')):  # SWAP=slot:recipe[:checkpoint]: an explicit replacement (TWINS=9:x switches the suffix twins off)
    j, r, *c = a.split(':'); have[int(j)] = out(r, c[0] if c else SIX[int(j)][1]); TW[int(j)] = ' -> ' + r + (':' + c[0] if c else '')
print('fold %d reference six: top-5 %.3f %%, top-100 %.2f %%, ranking error %.4f | twins available: %s' % (k, 100 * (base < 5).mean(), 100 * (base < 100).mean(), (base / 4999.).mean(), [SIX[j][0] + TW[j] for j in have]))
for j, y in have.items():
    print('  swap %-34s alone vs its reference: %s' % (SIX[j][0], versus(rk(y), rk(ref[j]))), flush=True)
    print('  %-39s in the six: %s' % ('', versus(rk(sum(ref) - ref[j] + y), base)), flush=True)
if len(have) > 1:
    print('  ALL %d twins swapped in:                  %s' % (len(have), versus(rk(sum(have.get(j, ref[j]) for j in range(6))), base)), flush=True)
    if len(have) < 6:
        sub = sorted(have); print('  only the %d twinned members, twins vs references: %s' % (len(sub), versus(rk(sum(have[j] for j in sub)), rk(sum(ref[j] for j in sub)))), flush=True)
if os.environ.get('SAVE'):  # ranks of the reference six and of the six with every swap, 5k and 14k galleries, r=3 -> pooled over folds by nu06_pool.py
    big = gallery(big=True); new = sum(have.get(j, ref[j]) for j in range(6)); r14 = lambda y: rank(y[g3].mean(1), tg, big).numpy()
    np.savez(f"{OUT}/stack_{os.environ['SAVE']}_f{k}.npz", base=base, new=rk(new), base14=r14(sum(ref)), new14=r14(new))

