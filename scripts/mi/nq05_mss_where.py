"""Q05 where the MSS cohort changes a member (cached outputs, new people, never-trained images, 5k gallery, r=3).
Per held-out image: cosine to its nearest MSS stimulus (10,000 images of 20 classes, same target space). If MSS teaches
its own content, the member with MSS gains on the held-out images close to MSS images and not elsewhere.
Top-5 % of the member with MSS vs its reference by quintile of that cosine (far -> near), and per held-out person.
    nq05_mss_where.py [arm:fold:epoch_mss:epoch_ref ..., default rmix:1:12:10 rmix:3:12:10 d6:1:12:10]"""
from nc_lib import *

G = gallery().to(DEV); F = nz(torch.tensor(np.load('/nasbrain/p20fores/mss_ds005589/targets.npz')['F'])).to(DEV)
near = (G @ F.T).max(1).values.cpu().numpy(); q5 = np.searchsorted(np.quantile(near, [.2, .4, .6, .8]), near)
print('cosine of a held-out image to its nearest MSS image: quintile edges', np.quantile(near, [.2, .4, .6, .8]).round(3))
for a in sys.argv[1:] or ['rmix:1:12:10', 'rmix:3:12:10', 'd6:1:12:10']:
    arm, k, e1, e0 = a.split(':'); st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], torch.as_tensor(st['img'])
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); t = img[g4[:, 0]].numpy(); s = subj[g4[:, 0].numpy()]
    H = []
    for name, e in ((f'mi_{arm}_ho5k_lag5_mssS', e1), (f'mi_{arm}_ho5k_lag5', e0)):
        q = nz(scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_{name}_ep{int(e):02d}_raw.npy')).float())[g4[:, :3]].mean(1)).to(DEV)
        Sc = q @ G.T; H.append(((Sc > Sc.gather(1, torch.as_tensor(t, device=DEV)[:, None])).sum(1) < 5).cpu().numpy())
    print('%s f%s, with MSS @%s vs reference @%s: %.3f vs %.3f (%+.1f +- %.1f %%)' % (arm, k, e1, e0, 100 * H[0].mean(), 100 * H[1].mean(), *paired(H[0], H[1])))
    print('    by quintile, far -> near: ' + ' | '.join('%.2f vs %.2f' % (100 * H[0][q5[t] == i].mean(), 100 * H[1][q5[t] == i].mean()) for i in range(5)))
    print('    by person: ' + ' | '.join('%d: %.2f vs %.2f' % (p, 100 * H[0][s == p].mean(), 100 * H[1][s == p].mean()) for p in sorted(set(s.tolist()))), flush=True)
