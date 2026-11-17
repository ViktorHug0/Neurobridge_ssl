"""E07 is there a linear mismatch between THINGS-EEG2 and Alljoined beyond the 5-sample lag?
The frozen Alljoined-ONLY REVE-d6 (fold 3, ep10) is the measuring instrument (as in D03): a linear input adapter for the
THINGS dataset, x' = sum_tau W_tau x(t - tau) (W_tau 32x32, taps -5..+5 around the +5 lag), is fitted by gradient
through the frozen model on labelled THINGS trials (five people, 5,000 images the model trained on, its own contrastive
loss) and tested on five OTHER THINGS people x the 5,000 never-trained images (5k, exact ranks).
Variants: lag only (the initialisation), per-electrode gain, temporal filter shared by all electrodes, spatial re-mix
at the lag, full spatio-temporal. A gain here = more to align before joint training (a --things_adapter).
Saves the fitted adapters to results/things_eeg/mi/nc/e07_adapters.pt."""
import torch.nn.functional as Fn
from nc_lib import *

k = 3
f = load('reve_d6_ho5k_mid_optx_aj', k, 'checkpoint_ep10.pth')
for p in list(f.model.parameters()) + list(f.eproj.parameters()):
    p.requires_grad_(False)
Xf, sf, imf = cell([22, 24, 26, 28, 30], SEEN)
Xe, se, ime = cell([21, 23, 25, 27, 29], HELD)
Tf = nz(torch.tensor(np.asarray(TARGETS[SEEN])).float()).to(DEV)
gal = gallery()
TAPS = list(range(0, 11))  # shift later by 0..10 samples; the lag is tap 5


def apply(W, x):  # W (taps, 32, 32), x (n, 32, 120): replicate-padded shifts
    xs = torch.stack([torch.cat([x[..., :1].expand(*x.shape[:-1], n), x[..., :x.shape[-1] - n]], -1) if n else x for n in TAPS])
    return torch.einsum('kcd,kndt->nct', W, xs)


def build(kind, P):
    eye = torch.eye(32, device=DEV)
    if kind == 'lag only':
        return torch.stack([eye * (n == 5) for n in TAPS])
    if kind == 'per-electrode gain':
        return torch.stack([torch.diag(P) * (n == 5) for n in TAPS])
    if kind == 'shared temporal filter':
        return P[:, None, None] * eye
    if kind == 'per-electrode temporal filter':
        return torch.stack([torch.diag(P[i]) for i in range(len(TAPS))])
    if kind == 'spatial re-mix at the lag':
        return torch.stack([P * (n == 5) for n in TAPS])
    return P  # full spatio-temporal


INIT = {'per-electrode gain': lambda: torch.ones(32), 'shared temporal filter': lambda: torch.tensor([float(n == 5) for n in TAPS]),
        'per-electrode temporal filter': lambda: torch.tensor([float(n == 5) for n in TAPS])[:, None].repeat(1, 32),
        'spatial re-mix at the lag': lambda: torch.eye(32), 'full spatio-temporal': lambda: torch.stack([torch.eye(32) * (n == 5) for n in TAPS])}


def fwd(x):
    return f.eproj(run_eeg_backbone(f.model, f.cfg, x, torch.full((len(x),), 10 ** 6, device=DEV))).float()


@torch.no_grad()
def test(W):
    out = torch.cat([fwd(apply(W, torch.as_tensor(Xe[b:b + 512]).to(DEV))).cpu() for b in range(0, len(Xe), 512)])
    return score(out, se, ime, gal, hits=True)


base = test(build('lag only', None)); print('lag only:', {a: round(b, 3) for a, b in base[0].items()}, flush=True)
saved = {}
for kind in INIT:
    P = INIT[kind]().to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([P], lr=float(os.environ.get('LR', 3e-3)))
    g = torch.Generator().manual_seed(0)
    for ep in range(int(os.environ.get('EPOCHS', 2))):
        perm = torch.randperm(len(Xf), generator=g); tot = 0
        for i, b in enumerate(range(0, len(perm) - 255, 256)):
            idx = perm[b:b + 256].numpy(); idx.sort()
            x = torch.as_tensor(Xf[idx]).to(DEV); t = torch.as_tensor(imf[idx]).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = fwd(apply(build(kind, P), x))
            logits = y @ Tf.T  # the model's own loss form (EEG -> image, temperature 1), against all 5,000 fit images
            loss = Fn.cross_entropy(logits, t)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sc, h = test(build(kind, P.detach()))
        print(f'{kind}, epoch {ep + 1}: fit loss {tot / (i + 1):.4f} | test', {a: round(b, 3) for a, b in sc.items()},
              '| vs lag only r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (*paired(h['r3'], base[1]['r3']), *paired(h['r4'], base[1]['r4'])), flush=True)
    saved[kind] = build(kind, P.detach()).cpu()
    torch.save(saved, f'{OUT}/e07_adapters.pt')
