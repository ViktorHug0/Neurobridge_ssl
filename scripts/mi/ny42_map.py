"""Y42 what makes a model add to E2? (user, 19 Nov: hybrids; is pretrained / scratch a meaningful family label?) Every model cached on the
fold (one checkpoint per run: test_best, else the last), against E2's nine members: r=3 top-5 alone (four triplets), the correlation of its
target z-score with the conv members (SQF-X, SQF), the hard-label token members and the soft-label token members, the share of its z-score
NOT explained by the nine (1 - R2, linear), and what it adds to E2 at weight 1.
    ny42_map.py <fold>"""
import re
from nc_lib import *

k = int(sys.argv[1])
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.),
      ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5), ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
CONV, HARD, SOFT = [0, 1], [2, 3, 4], [5, 6, 7, 8]
gal = gallery(); X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); g3 = torch.as_tensor(np.concatenate([g4[:, [a for a in range(4) if a != o]] for o in range(4)])); tg = torch.as_tensor(img)[g3[:, 0]]
u = lambda f: scaled(torch.as_tensor(np.load(f).astype(np.float32)))[g3].mean(1)  # r=3 queries, four triplets per group


def zs(Q):  # target z-score among the gallery
    q = nz(Q); out = []
    for b in range(0, len(q), 4096):
        S = q[b:b + 4096].to(DEV) @ gal.T; t = S.gather(1, tg[b:b + 4096].to(DEV)[:, None]); out.append(((t - S.mean(1, keepdim=True)) / S.std(1, keepdim=True)).squeeze(1).cpu())
    return torch.cat(out)


top5 = lambda Q: (rank(Q, tg, gal) < 5).float().mean().item()
P = [u(f'{OUT}/u01_f{k}_{r}_{c}.npy') for r, c, _ in E2]; ENS = sum(w * p for (_, _, w), p in zip(E2, P)); base = top5(ENS)
ZE = torch.stack([zs(p) for p in P], 1); A = torch.cat([ZE, torch.ones(len(ZE), 1)], 1); pinv = torch.linalg.pinv(A)
runs = {}
for f in sorted(glob.glob(f'{OUT}/u01_f{k}_*.npy')):
    r, c = re.match(rf'u01_f{k}_(.*)_(checkpoint_.*)\.npy', os.path.basename(f)).groups()
    if (r, c) not in [(a, b) for a, b, _ in E2] and (r not in runs or c == 'checkpoint_test_best.pth') and re.search(os.environ.get('ONLY', ''), r):
        runs[r] = f
print('fold %d, E2 r=3 top-5 %.3f %%, %d candidates. Columns: alone %% | corr with conv / hard tokens / soft tokens | unexplained by the nine | added to E2 (w 1)' % (k, 100 * base, len(runs)), flush=True)
rows = []
for r, f in runs.items():
    q = u(f); z = zs(q); cc = lambda I: np.mean([np.corrcoef(z.numpy(), ZE[:, i].numpy())[0, 1] for i in I])
    res = z - A @ (pinv @ z); rows.append((r.replace('mi_', '').replace('_ho5k', ''), 100 * top5(q), cc(CONV), cc(HARD), cc(SOFT), (res.var() / z.var()).item(), 100 * (top5(ENS + q) / base - 1)))
for r in sorted(rows, key=lambda r: -r[5]):
    print('%-34s %5.2f | %.2f %.2f %.2f | %.2f | %+5.1f' % r, flush=True)
R = np.array([r[1:] for r in rows]); ok = R[:, 0] > 0.5  # models that trained
for lab, x in (('alone', R[ok, 0]), ('unexplained', R[ok, 4]), ('corr conv', R[ok, 1]), ('corr hard', R[ok, 2]), ('corr soft', R[ok, 3])):
    print('correlation of the added value with %s: %+.2f' % (lab, np.corrcoef(x, R[ok, 5])[0, 1]))
B = np.c_[R[ok, 0], R[ok, 4], np.ones(ok.sum())]; co = np.linalg.lstsq(B, R[ok, 5], rcond=None)[0]; pred = B @ co
print('added value ~ %.2f x alone + %.1f x unexplained %+.1f (R2 %.2f, n %d)' % (*co, 1 - ((R[ok, 5] - pred) ** 2).sum() / ((R[ok, 5] - R[ok, 5].mean()) ** 2).sum(), ok.sum()))
