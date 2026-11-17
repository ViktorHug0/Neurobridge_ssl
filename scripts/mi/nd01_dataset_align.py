"""D01 how do the two training datasets differ at the input, and how much of it is a fixable linear mismatch?
Alljoined (sub-1..20) and THINGS-EEG2 (sub-21..30) saw the same training images, so per-image mean responses are paired.
Fold-1 training people only (15 Alljoined, 10 THINGS). 4,000 random training images: 3,000 to fit, 1,000 to test.
  1. first-order: dataset keys (mean ERP), their lag cross-correlation, per-electrode gain.
  2. content: image-specific response = per-image mean minus the dataset key. Agreement between two groups of people =
     1,000-way image identification (top-1 %, z-scored features, 0-0.6 s) and mean feature correlation.
     within Alljoined (two halves), within THINGS (two halves), across datasets, and across after: a lag, a per-electrode
     gain, a 32x32 electrode re-mix (ridge), a spatio-temporal filter (7 taps).
Saves the fitted maps to results/things_eeg/mi/nc/dataset_align_f1.npz."""
from nc_lib import *

k = 1
AJ = [s for s in range(1, 21) if s not in FOLDS[k]]
TH = list(range(21, 31))
rng = np.random.default_rng(0)
IMG = np.sort(rng.choice(16540, 4000, replace=False)); fit = rng.permutation(4000) < 3000  # boolean mask over IMG
p = f'{OUT}/d01_means.npz'
if not os.path.exists(p):
    out = {}
    for s in AJ + TH:
        x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)[IMG])
        n = (np.abs(x).sum((2, 3)) > 0).sum(1)
        out[f'sum{s}'], out[f'n{s}'], out[f'rms{s}'] = x.sum(1), n, np.sqrt((x ** 2).sum((0, 1, 3)) / (n.sum() * 120))
        print(s, flush=True)
    np.savez(p, **out)
d = np.load(p)
mean = lambda people: torch.tensor(sum(d[f'sum{s}'] for s in people) / np.maximum(sum(d[f'n{s}'] for s in people), 1)[:, None, None]).float().to(DEV)
groups = {'AJ': AJ, 'TH': TH, 'AJa': AJ[::2], 'AJb': AJ[1::2], 'THa': TH[::2], 'THb': TH[1::2]}
M = {g: mean(pp) for g, pp in groups.items()}
KEY = {g: m.mean(0) for g, m in M.items()}
C = {g: M[g] - KEY[g] for g in M}  # image-specific content
W0, W1 = 24, 96  # 0..0.6 s
F, Tt = torch.tensor(fit).to(DEV), torch.tensor(~fit).to(DEV)

print('== 1. first order')
print('key norm  AJ %.2f  TH %.2f   corr(K_AJ, K_TH) %.2f' % (KEY['AJ'].norm(), KEY['TH'].norm(), torch.corrcoef(torch.stack([KEY['AJ'].flatten(), KEY['TH'].flatten()]))[0, 1]))
for lag in range(-6, 7):
    a, b = KEY['AJ'][:, 12:108], torch.roll(KEY['TH'], lag, 1)[:, 12:108]
    print('  TH key shifted by %+d samples (%+.0f ms): corr %.3f' % (lag, lag * 1000 / 120, torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]))
rms = {g: np.mean([d[f'rms{s}'] for s in pp], 0) for g, pp in (('AJ', AJ), ('TH', TH))}
print('single-trial RMS per electrode, TH / AJ:', np.round(rms['TH'] / rms['AJ'], 2).tolist())
print('content norm per image (mean): AJ %.2f  TH %.2f' % (C['AJ'][:, :, W0:W1].flatten(1).norm(dim=1).mean(), C['TH'][:, :, W0:W1].flatten(1).norm(dim=1).mean()))


def agree(a, b):
    """1,000 test images: top-1 identification (%) of a's image among b's, and mean feature-wise correlation."""
    a, b = a[Tt][:, :, W0:W1].flatten(1), b[Tt][:, :, W0:W1].flatten(1)
    za, zb = (a - a.mean(0)) / a.std(0), (b - b.mean(0)) / b.std(0)
    S = za @ zb.T
    return 100 * (S.argmax(1) == torch.arange(len(S), device=DEV)).float().mean().item(), (za * zb).mean().item()


def remix(src, dst, taps=1, lam=1e-2):
    """ridge map (32 x 32*taps) from src's content to dst's, fitted on the fit images; returns apply(x)."""
    half = taps // 2
    stack = lambda x: torch.cat([torch.roll(x, l, -1) for l in range(-half, half + 1)], -2)  # (n, 32*taps, T)
    X, Y = stack(src[F])[:, :, 12:108], dst[F][:, :, 12:108]
    G = torch.einsum('nct,ndt->cd', X, X); B = torch.einsum('nct,ndt->cd', Y, X)
    Wm = B @ torch.linalg.inv(G + lam * G.diagonal().mean() * torch.eye(len(G), device=DEV))
    return (lambda x: torch.einsum('cd,...dt->...ct', Wm, stack(x))), Wm


print('== 2. content agreement on 1,000 held-out images: identification top-1 % | feature corr')
rows = {'within AJ (7 vs 8 people)': (C['AJa'], C['AJb']), 'within TH (5 vs 5 people)': (C['THa'], C['THb']),
        'across  AJ half vs TH half': (C['AJa'], C['THa']), 'across  AJ (15) vs TH (10)': (C['AJ'], C['TH'])}
for n, (a, b) in rows.items():
    print('  %-34s %5.1f | %.3f' % (n, *agree(a, b)))
print('  across AJ vs TH after transforming TH:')
for lag in range(-4, 5):
    print('    lag %+d samples: %5.1f | %.3f' % (lag, *agree(C['AJ'], torch.roll(C['TH'], lag, -1))))
g = (C['AJ'][F][:, :, W0:W1] * C['TH'][F][:, :, W0:W1]).sum((0, 2)) / (C['TH'][F][:, :, W0:W1] ** 2).sum((0, 2))
print('    per-electrode gain (LS): %5.1f | %.3f   gains %s' % (*agree(C['AJ'], C['TH'] * g[:, None]), np.round(g.cpu().numpy(), 2).tolist()))
save = dict(K_AJ=KEY['AJ'].cpu().numpy(), K_TH=KEY['TH'].cpu().numpy())
for taps in (1, 3, 7):
    f, Wm = remix(C['TH'], C['AJ'], taps)
    print('    electrode re-mix, %d tap(s): %5.1f | %.3f' % (taps, *agree(C['AJ'], f(C['TH']))))
    save[f'W{taps}'] = Wm.cpu().numpy()
    # control: the same map fitted and applied within Alljoined (what a re-mix buys by denoising alone)
    f2, _ = remix(C['AJb'], C['AJa'], taps)
    print('      control within AJ halves, same fit: %5.1f | %.3f' % agree(C['AJa'], f2(C['AJb'])))
# reverse direction (AJ -> TH) and what the map looks like
f, W1m = remix(C['TH'], C['AJ'], 1)
print('re-mix W1: diagonal mean %.2f, off-diagonal abs mean %.3f, identity-fit cos %.2f' % (W1m.diagonal().mean(), (W1m - torch.diag(W1m.diagonal())).abs().mean(), (W1m.diagonal().sum() / (W1m.norm() * 32 ** .5))))
np.savez(f'{OUT}/dataset_align_f{k}.npz', **save)
