"""E08 control for E07 / a static input filter for new people. A per-electrode temporal FIR (taps -5..+5 around lag 0) is
fitted by gradient through a frozen JOINT member on its held-out Alljoined people (fold A, 5,000 images the model
trained on, the member's contrastive loss), then tested (a) on the same people x never-trained images through the same
model and (b) on fold B's held-out people x never-trained images through fold B's model (other people, other model =
what a deployed static filter would face). 5k, exact ranks, paired against no filter.
    ne08_newperson_filter.py <member> <fold A> <fold B> [checkpoint]"""
import torch.nn.functional as Fn
from nc_lib import *
from nd03_cross_dataset_lib import shift

m, ka, kb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); ck = sys.argv[4] if len(sys.argv) > 4 else None
REC = {'d6': {1: 'reve_d6_ho5k_long_optx_things', 3: 'reve_d6_ho5k_mid_optx_things'}, 'rmix': {1: 'rmix_reg_ho5k_long_optx_things', 3: 'rmix_reg_ho5k_mid_optx_things'},
       'sqfx': {1: 'sqf_dbn_ho5k_long_optx_things', 3: 'sqf_dbn_ho5k_long_optx_things'}}[m]
CK = ck or ('checkpoint_test_best.pth' if m == 'sqfx' else 'checkpoint_ep10.pth')
TAPS = list(range(-5, 6))
gal = gallery(); Tf = nz(torch.tensor(np.asarray(TARGETS[SEEN])).float()).to(DEV)


def frozen(k):
    f = load(REC[k], k, CK)
    for p in list(f.model.parameters()) + list(f.eproj.parameters()):
        p.requires_grad_(False)
    return lambda x: f.eproj(run_eeg_backbone(f.model, f.cfg, x, torch.full((len(x),), 10 ** 6, device=DEV))).float()


def apply(P, x):  # P (taps, 32) per-electrode or (taps,) shared
    return sum((P[i][:, None] if P.dim() == 2 else P[i]) * shift(x, n) for i, n in enumerate(TAPS))


@torch.no_grad()
def test(fwd, P, k):
    F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); X = np.load(f'{F}/X.npy', mmap_mode='r')
    out = torch.cat([fwd(torch.as_tensor(np.asarray(X[b:b + 512])).to(DEV) if P is None else apply(P, torch.as_tensor(np.asarray(X[b:b + 512])).to(DEV))).cpu() for b in range(0, len(X), 512)])
    return score(out, st['subj'], st['img'], gal, hits=True)


fa = frozen(ka)
Xf, sf, imf = cell(FOLDS[ka], SEEN)
base_a = test(fa, None, ka); print(f'fold {ka} no filter:', {a: round(b, 3) for a, b in base_a[0].items()}, flush=True)
fitted = {}
for kind, init in (('shared', torch.tensor([float(n == 0) for n in TAPS])), ('per-electrode', torch.tensor([float(n == 0) for n in TAPS])[:, None].repeat(1, 32))):
    P = init.to(DEV).requires_grad_(True); opt = torch.optim.Adam([P], lr=3e-3); g = torch.Generator().manual_seed(0)
    for ep in range(2):
        perm = torch.randperm(len(Xf), generator=g); tot = 0
        for i, b in enumerate(range(0, len(perm) - 255, 256)):
            idx = np.sort(perm[b:b + 256].numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = fa(apply(P, torch.as_tensor(Xf[idx]).to(DEV)))
            loss = Fn.cross_entropy(y @ Tf.T, torch.as_tensor(imf[idx]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sc, h = test(fa, P.detach(), ka)
        print(f'{kind} filter, epoch {ep + 1}: fit loss {tot / (i + 1):.4f} | fold {ka} same people, new images', {a: round(b, 3) for a, b in sc.items()},
              '| r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (*paired(h['r3'], base_a[1]['r3']), *paired(h['r4'], base_a[1]['r4'])), flush=True)
    fitted[kind] = P.detach()
    torch.save({a: b.cpu() for a, b in fitted.items()}, f'{OUT}/e08_filter_{m}_f{ka}.pt')
    print(f'  {kind} filter: mean over electrodes', np.round((P.detach().mean(1) if P.dim() == 2 else P.detach()).cpu().numpy(), 2).tolist(), flush=True)
del fa; torch.cuda.empty_cache()
fb = frozen(kb)
base_b = test(fb, None, kb); print(f'fold {kb} no filter:', {a: round(b, 3) for a, b in base_b[0].items()}, flush=True)
for kind, P in fitted.items():
    sc, h = test(fb, P, kb)
    print(f'{kind} filter of fold {ka} on fold {kb} (other people, other model):', {a: round(b, 3) for a, b in sc.items()},
          '| r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (*paired(h['r3'], base_b[1]['r3']), *paired(h['r4'], base_b[1]['r4'])), flush=True)
