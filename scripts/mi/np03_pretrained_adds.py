"""P03 what the pretrained member adds to the lag five (cached outputs, new people, never-trained images, 5k, r=3, 3 folds).
(a) independence of the evidence: per query, each member's z-score of the target among the gallery scores; correlation
    of the z-scores between members (the ensemble gains from members whose evidence is less correlated);
(b) where the gain of six over five falls: by person strength, by image typicality, by band of gallery principal axes
    (correlation decoded vs true coordinate, five / LaBraM / six).
    np03_pretrained_adds.py [extra recipe:epoch, e.g. mi_cbm_ho5k_lag5:10 (fold 1 only)]"""
from nc_lib import *
from nc06_names import NAMES

Yp = lambda p: scaled(torch.as_tensor(np.load(p)).float()); Y = lambda k, r, e: Yp(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')
G = gallery(); typ = (G @ nz(G.mean(0, keepdim=True)).T)[:, 0].cpu(); tq = torch.bucketize(typ, torch.quantile(typ, torch.tensor([.2, .4, .6, .8])))
gc = G.cpu() - G.cpu().mean(0); PC = torch.linalg.svd(gc, full_matrices=False)[2].T; Zg = gc @ PC; BANDS = [(0, 4), (4, 16), (16, 64), (64, 256), (256, 1536)]
extra = [a.split(':') for a in sys.argv[1:]]; names = ['d6', 'rmix', 'sqfx', 'scr', 'sqf', 'LaBraM'] + [r for r, _ in extra]
Z, HIT, SUBJ, TT, DEC = [], {}, [], [], {}
for k in ((1,) if extra else (1, 2, 3)):
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], torch.as_tensor(st['img'])
    mem = [Y(k, 'd6', 12), Y(k, 'rmix', 10), Y(k, 'sqfx', 40)] + ([Y(k, 'scr', 12), Y(k, 'sqf', 32)] if k in (1, 2) else [Yp(f'{R}/logs/holdout/f{k}/{m}.npy') for m in NAMES[k][3:]])
    mem += [Yp(f'{OUT}/hold_f{k}_mi_plab_ho5k_lag5_long_ep10_raw.npy')] + [Yp(f'{OUT}/hold_f{k}_{r}_ep{int(e):02d}_raw.npy') for r, e in extra]
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); t = img[g4[:, 0]]; SUBJ.append(subj[g4[:, 0].numpy()]); TT.append(t)
    sets = {n: m for n, m in zip(names, mem)}; sets['five'] = sum(mem[:5]); sets['six'] = sum(mem[:6])
    zs = []
    for n, y in sets.items():
        q = nz(y[g4[:, :3]].mean(1)); z, h = [], []
        for b in range(0, len(q), 4096):
            Sc = q[b:b + 4096].to(DEV) @ G.T; tg = Sc.gather(1, t[b:b + 4096].to(DEV)[:, None])
            z.append(((tg[:, 0] - Sc.mean(1)) / Sc.std(1)).cpu()); h.append(((Sc > tg).sum(1) < 5).cpu())
        HIT.setdefault(n, []).append(torch.cat(h).numpy()); zs.append(torch.cat(z))
        mean_q = torch.zeros(5000, 1536).index_add_(0, t, q) / torch.bincount(t, minlength=5000).clamp(min=1)[:, None]  # r=3 queries averaged per image over people: the decoded coordinate
        DEC.setdefault(n, []).append((mean_q - mean_q.mean(0)) @ PC)
    Z.append(torch.stack(zs, 1))
Z = torch.cat(Z); cols = list(sets); C = np.corrcoef(Z.T.numpy()); H = {n: np.concatenate(v) for n, v in HIT.items()}; subj = np.concatenate(SUBJ); tt = torch.cat(TT)
print('(a) mean z of the target (r=3):', ' '.join('%s %.3f' % (n, Z[:, i].mean()) for i, n in enumerate(cols)))
print('    correlation of the z-scores between single members:'); m = len(names)
for i in range(m):
    print('      %-8s' % names[i], ' '.join('%.2f' % C[i, j] for j in range(m)), '| mean with the others of the five %.3f' % np.mean([C[i, j] for j in range(5) if j != i]))
print('    top-5 %%:', ' '.join('%s %.3f' % (n, 100 * H[n].mean()) for n in cols), '| six vs five %+.1f +- %.1f %%' % paired(H['six'], H['five']))
ps = sorted(set(subj.tolist()), key=lambda s: H['five'][subj == s].mean())
print('(b) by person (weak -> strong), five -> six top-5 %:', ' | '.join('%d: %.2f -> %.2f' % (s, 100 * H['five'][subj == s].mean(), 100 * H['six'][subj == s].mean()) for s in ps))
third = np.array_split(np.array(ps), 3); print('    thirds of people, weak / mid / strong, gain of six: ' + ' / '.join('%+.1f +- %.1f %%' % paired(H['six'][np.isin(subj, g)], H['five'][np.isin(subj, g)]) for g in third))
print('    by typicality quintile of the target (atypical -> typical), five -> six: ' + ' | '.join('%.2f -> %.2f' % (100 * H['five'][(tq[tt] == i).numpy()].mean(), 100 * H['six'][(tq[tt] == i).numpy()].mean()) for i in range(5)))
cc = lambda a, b: ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / ((a - a.mean(0)).norm(dim=0) * (b - b.mean(0)).norm(dim=0)).clamp_min(1e-9)
for n in ['five', 'LaBraM', 'six'] + names[6:] + ['rmix', 'sqfx']:
    r = torch.stack([cc(d, Zg) for d in DEC[n]]).mean(0); print('    %-8s r(decoded, true) by band of gallery principal axes: ' % n + ' '.join('%d-%d %.3f' % (a, b, r[a:b].mean()) for a, b in BANDS), flush=True)
