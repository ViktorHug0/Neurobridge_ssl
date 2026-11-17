"""E21 per-person functional operators for the THINGS-EEG2 people, in Alljoined coordinates: a 32x32 spatial re-mix per
THINGS person fitted through the frozen Alljoined-ONLY REVE-d6 (fold 3, ep10; it never saw THINGS) on that person's
trials of 5,000 trained images, AFTER the dataset filter of E07 (lag + per-electrode FIR); tested on the same person x
never-trained images (5k, exact ranks) against the filter alone. Saved to results/things_eeg/mi/nc/e21_ops_things.pt.
With these, M_t^-1 M_s also re-renders a THINGS person as an Alljoined one (and back) in --func_transfer_ops."""
import torch.nn.functional as Fn
from nc_lib import *
from nd03_cross_dataset_lib import shift
f = load('reve_d6_ho5k_mid_optx_aj', 3, 'checkpoint_ep10.pth')
for p in list(f.model.parameters()) + list(f.eproj.parameters()):
    p.requires_grad_(False)
fwd = lambda x: f.eproj(run_eeg_backbone(f.model, f.cfg, x, torch.full((len(x),), 10 ** 6, device=DEV))).float()
Fl = torch.tensor(np.load(f'{OUT}/things_filter_e07.npy')).to(DEV)
filt = lambda x: sum(Fl[i][:, None] * shift(x, i) for i in range(len(Fl)))
gal = gallery(); Tf = nz(torch.tensor(np.asarray(TARGETS[SEEN])).float()).to(DEV)
ops, H = {}, {'none': [], 'own': []}
for s in range(21, 31):
    Xf, _, imf = cell([s], SEEN); Xe, se, ie = cell([s], HELD)
    M = torch.eye(32, device=DEV).requires_grad_(True); opt = torch.optim.Adam([M], lr=3e-3); g = torch.Generator().manual_seed(0)
    for ep in range(3):
        perm = torch.randperm(len(Xf), generator=g)
        for b in range(0, len(perm) - 255, 256):
            idx = np.sort(perm[b:b + 256].numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = fwd(torch.einsum('cd,ndt->nct', M, filt(torch.as_tensor(Xf[idx]).to(DEV))))
            loss = Fn.cross_entropy(y @ Tf.T, torch.as_tensor(imf[idx]).to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
    ops[s] = M.detach().cpu(); torch.save(ops, f'{OUT}/e21_ops_things.pt')
    with torch.no_grad():
        out = {n: torch.cat([fwd(torch.einsum('cd,ndt->nct', A, filt(torch.as_tensor(Xe[b:b + 512]).to(DEV)))).cpu() for b in range(0, len(Xe), 512)]) for n, A in (('none', torch.eye(32, device=DEV)), ('own', M.detach()))}
    sc = {n: score(y, se, ie, gal, hits=True) for n, y in out.items()}
    for n in H:
        H[n].append(sc[n][1])
    print('sub-%d: |M - I| %.2f, cond %.1f | filter only %s -> + own operator %s' % (s, (M.detach() - torch.eye(32, device=DEV)).norm(), torch.linalg.cond(M.detach()), ' / '.join('%.2f' % v for v in sc['none'][0].values()), ' / '.join('%.2f' % v for v in sc['own'][0].values())), flush=True)
for r in ('r3', 'r4'):
    a, b = np.concatenate([h[r] for h in H['own']]), np.concatenate([h[r] for h in H['none']]); print('pooled %s: %.3f -> %.3f, %+.1f +- %.1f %%' % (r, 100 * b.mean(), 100 * a.mean(), *paired(a, b)))
