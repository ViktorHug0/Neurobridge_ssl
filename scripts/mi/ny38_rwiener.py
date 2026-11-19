"""Y38 a repeat-aware output filter. A member trained on single trials shrinks each output axis for the noise of ONE trial; the grader then
averages ~3 outputs, where the noise is 3x lower and the best shrinkage per axis is different. Axes = generalised eigen-axes of the E2
output: reproducible covariance R (between (person, image) groups, noise removed) against the within-group noise covariance N, fitted on
the FIT folds. Per-axis gain c_k, map q -> sum_k c_k d_k (p_k . q) (c = 1 is the identity; one 1536 x 1536 matrix, stateless).
 (a) closed forms: c = ((rho + 1) / (rho + 1/3))^g (un-shrink for r = 3); c = 0 below a rho threshold (drop noise axes);
 (b) fitted: 32 bands of axes by rho, gains fitted on the FIT folds with the r=3 retrieval loss (and, for comparison, the r=1 loss);
read on the TEST folds: per-trial and r=3 top-5 (four triplets), paired against the identity.
    ny38_rwiener.py <fit folds> <test folds>"""
from nc_lib import *

FIT, TEST = ([int(a) for a in s.split(',')] for s in sys.argv[1:3])
E2 = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_ep24.pth', 1.5), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth', .5), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth', 1.),
      ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth', 1.), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth', .5),
      ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth', 1.), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth', 1.)]
gal = gallery(); D = {}
for k in FIT + TEST:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; del X; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); g3 = torch.cat([g4[:, [a for a in range(4) if a != o]] for o in range(4)])
    u = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{(r if k != 2 or 'long_soft3' not in r else 'mi_plab_ho5k_lag5_jit3_soft3')}_{(c if k != 2 or 'long_soft3' not in r else 'checkpoint_test_best.pth')}.npy").astype(np.float32)))
    Y = sum(w * u(r, c) for r, c, w in E2); D[k] = (Y, g4, g3, torch.as_tensor(img)); print('fold', k, 'loaded', flush=True)
R, N, ng = torch.zeros(1536, 1536, dtype=torch.float64), torch.zeros(1536, 1536, dtype=torch.float64), 0
for k in FIT:  # covariances on the GPU in chunks
    Y, g4 = D[k][:2]; M = Y[g4].mean(1); mu = M.mean(0)
    for b in range(0, len(g4), 8192):
        m = (M[b:b + 8192] - mu).to(DEV).double(); e = (Y[g4[b:b + 8192]].to(DEV).double() - M[b:b + 8192, None].to(DEV).double()).reshape(-1, 1536)
        R += (m.T @ m).cpu(); N += (e.T @ e).cpu()
    ng += len(g4)
N = N / (3 * ng); R = R / ng - N / 4  # N: single-trial noise covariance; R: reproducible covariance of a group
w, V = torch.linalg.eigh(N + 1e-4 * N.diagonal().mean() * torch.eye(1536, dtype=torch.float64)); Wm, Wp = (V * w.rsqrt()) @ V.T, (V * w.sqrt()) @ V.T
rho, U = torch.linalg.eigh(Wm @ R @ Wm); o = torch.argsort(rho, descending=True); rho, U = rho[o], U[:, o]; P, Dk = (Wm @ U).float(), (Wp @ U).float()  # p_k (filters), d_k (directions)
print('rho (reproducible / single-trial noise variance) of axes 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1536: ' + ' '.join('%.3f' % rho[i - 1] for i in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1536)), flush=True)
rp = rho.clamp_min(0).float(); G = (gal.cpu() @ Dk).to(DEV)  # gallery in axis coordinates: score = sum_k c_k (p_k . q)(d_k . g)


def hits(k, c):
    Y, g4, g3, img = D[k]; Z = Y @ P; q1, q3 = Z[g4.ravel()] * c, Z[g3].mean(1) * c; out = []
    for q, t in ((q1, img[g4.ravel()]), (q3, img[g3[:, 0]])):
        r = torch.cat([((q[b:b + 4096].to(DEV) @ G.T) > (q[b:b + 4096].to(DEV) * G[t[b:b + 4096].to(DEV)]).sum(1, keepdim=True) + 0).sum(1).cpu() for b in range(0, len(q), 4096)])
        out.append((r < 5).numpy().astype(float))
    return out[0], out[1].reshape(4, -1).mean(0)


def report(name, c):
    h1 = np.concatenate([hits(k, c)[0] for k in TEST]); h3 = np.concatenate([hits(k, c)[1] for k in TEST])
    print('%-44s per-trial %.3f %% (%+.1f) | r=3 %.3f %%: %+.2f ± %.2f | gain %.3f' % (name, 100 * h1.mean(), 100 * (h1.mean() / B1.mean() - 1), 100 * h3.mean(), *paired(h3, B3, 1000), h3.mean() / h1.mean()), flush=True)


one = torch.ones(1536); B1 = np.concatenate([hits(k, one)[0] for k in TEST]); B3 = np.concatenate([hits(k, one)[1] for k in TEST])
print('test folds %s, identity: per-trial %.3f %% | r=3 %.3f %% | gain %.3f' % (TEST, 100 * B1.mean(), 100 * B3.mean(), B3.mean() / B1.mean()))
for g in (.25, .5, 1.):
    report('un-shrink ((rho+1)/(rho+1/3))^%.2f' % g, ((rp + 1) / (rp + 1 / 3)) ** g)
for g in (.25, .5):
    report('shrink more ((rho+1/3)/(rho+1))^%.2f' % g, ((rp + 1 / 3) / (rp + 1)) ** g)
for kk in (64, 256, 768):
    report('keep the %d most reproducible axes' % kk, (torch.arange(1536) < kk).float())
band = torch.bucketize(torch.arange(1536), torch.tensor([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408]), right=True)
for rr in (3, 1):  # fitted band gains, r=3 loss and r=1 loss
    torch.manual_seed(0); lg = torch.zeros(band.max() + 1, device=DEV, requires_grad=True); ls = torch.zeros((), device=DEV, requires_grad=True); opt = torch.optim.Adam([lg, ls], lr=.05)
    Q, T = [], []
    for k in FIT:
        Y, g4, g3, img = D[k]; Z = Y @ P; Q.append(Z[g3].mean(1) if rr == 3 else Z[g4.ravel()]); T.append(img[g3[:, 0]] if rr == 3 else img[g4.ravel()])
    Q, T = torch.cat(Q), torch.cat(T); Q = Q / Q.norm(dim=1).mean()
    for step in range(600):
        i = torch.randint(0, len(Q), (4096,)); c = lg.exp()[band.to(DEV)]
        loss = torch.nn.functional.cross_entropy((Q[i].to(DEV) * c) @ G.T * ls.exp() * 10, T[i].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
    c = (lg.detach().exp() / lg.detach().exp()[0])[band.to(DEV)].cpu()
    print('fitted with the r=%d loss (loss %.4f); gains by band (most -> least reproducible): ' % (rr, loss.item()) + ' '.join('%.2f' % v for v in (lg.detach().exp() / lg.detach().exp()[0]).cpu()), flush=True)
    report('fitted band gains, r=%d loss' % rr, c)
