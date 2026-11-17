"""E19 per-person functional operators for training: a 32x32 spatial re-mix M_s per held-out person of fold k, fitted
through that fold's frozen REVE-d6 (ep10) on the person's trials of 5,000 trained images (as E17, spatial only so that
M_t^-1 M_s is defined), tested on the same person x never-trained images. Saved to results/things_eeg/mi/nc/e19_ops_f<k>.pt
(person -> M). Used by train.py --func_transfer_ops: x_s -> M_t^-1 M_s x_s re-renders person s as person t.
    ne19_fit_operators.py <fold> [epochs=3]"""
import torch.nn.functional as Fn
from nc_lib import *
k = int(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
L = ('long', 'mid')[k != 1]
f = load(f'reve_d6_ho5k_{L}_optx_things', k, 'checkpoint_ep10.pth')
for p in list(f.model.parameters()) + list(f.eproj.parameters()):
    p.requires_grad_(False)
fwd = lambda x: f.eproj(run_eeg_backbone(f.model, f.cfg, x, torch.full((len(x),), 10 ** 6, device=DEV))).float()
gal = gallery(); Tf = nz(torch.tensor(np.asarray(TARGETS[SEEN])).float()).to(DEV)
F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); Xh = np.load(f'{F}/X.npy', mmap_mode='r'); sid = torch.as_tensor(st['subj'])
ops = {}
for s in FOLDS[k]:
    Xf, _, imf = cell([s], SEEN)
    M = torch.eye(32, device=DEV).requires_grad_(True); opt = torch.optim.Adam([M], lr=3e-3); g = torch.Generator().manual_seed(0)
    for ep in range(EP):
        perm = torch.randperm(len(Xf), generator=g); tot = 0
        for i, b in enumerate(range(0, len(perm) - 255, 256)):
            idx = np.sort(perm[b:b + 256].numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = fwd(torch.einsum('cd,ndt->nct', M, torch.as_tensor(Xf[idx]).to(DEV)))
            loss = Fn.cross_entropy(y @ Tf.T, torch.as_tensor(imf[idx]).to(DEV)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    ops[s] = M.detach().cpu(); print(f'sub-{s:02d}: fit loss {tot / (i + 1):.4f}, |M - I| {(M.detach() - torch.eye(32, device=DEV)).norm():.2f}, cond {torch.linalg.cond(M.detach()):.1f}', flush=True)
    torch.save(ops, f'{OUT}/e19_ops_f{k}.pt')
T = torch.stack([ops.get(s, torch.eye(32)) for s in range(21)]).to(DEV)
base = score(f(Xh), st['subj'], st['img'], gal, hits=True)
own = score(f(Xh, lambda a, b: torch.einsum('ncd,ndt->nct', T[sid[a:b]], torch.as_tensor(np.asarray(Xh[a:b])).to(DEV)) - torch.as_tensor(np.asarray(Xh[a:b])).to(DEV)), st['subj'], st['img'], gal, hits=True)
print('none', {a: round(b, 3) for a, b in base[0].items()}, '| own operator', {a: round(b, 3) for a, b in own[0].items()}, '| r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (*paired(own[1]['r3'], base[1]['r3']), *paired(own[1]['r4'], base[1]['r4'])),
      '| per person r3: ' + ' '.join('%d: %.2f->%.2f' % (s, 100 * base[1]['r3'][base[1]['subj'] == s].mean(), 100 * own[1]['r3'][own[1]['subj'] == s].mean()) for s in FOLDS[k]))
