"""E17 oracle input adapter PER new person: how much of the person gap is a linear input misalignment?
For each held-out person of fold k, a per-electrode temporal FIR (taps -5..+5), then also a 32x32 spatial re-mix, is
fitted through the frozen joint member on that person's trials of 5,000 trained images (labels used: an oracle, not
deployable) and tested on the same person x never-trained images (5k, exact ranks, r=1 / r=3 / r=4).
Reference: new 1.37 -> seen 2.37 (E06). If a 352-parameter filter recovers much of that, people could be canonicalised
in TRAINING (each fitted through the fold that holds it out).
    ne17_person_adapter.py <fold> <member d6|rmix>"""
import torch.nn.functional as Fn
from nc_lib import *
from nd03_cross_dataset_lib import shift
k, m = int(sys.argv[1]), sys.argv[2]
L = ('long', 'mid')[k != 1]
f = load({'d6': f'reve_d6_ho5k_{L}_optx_things', 'rmix': f'rmix_reg_ho5k_{L}_optx_things'}[m], k, 'checkpoint_ep10.pth')
for p in list(f.model.parameters()) + list(f.eproj.parameters()):
    p.requires_grad_(False)
fwd = lambda x: f.eproj(run_eeg_backbone(f.model, f.cfg, x, torch.full((len(x),), 10 ** 6, device=DEV))).float()
TAPS = list(range(-5, 6)); gal = gallery(); Tf = nz(torch.tensor(np.asarray(TARGETS[SEEN])).float()).to(DEV)
F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); Xh = np.load(f'{F}/X.npy', mmap_mode='r')
apply = lambda P, M, x: torch.einsum('cd,ndt->nct', M, sum(P[i][:, None] * shift(x, n) for i, n in enumerate(TAPS)))


@torch.no_grad()
def outputs(adapt):  # the whole held-out stream (64-batches as served), each trial through its person's adapter
    out = []
    for b in range(0, len(Xh), 512):
        x = torch.as_tensor(np.asarray(Xh[b:b + 512])).to(DEV); s = st['subj'][b:b + 512]
        if adapt:
            x = torch.cat([apply(*adapt[int(p)], x[i:i + 1]) for i, p in enumerate(s)]) if len(set(s.tolist())) > 1 else apply(*adapt[int(s[0])], x)
        out.append(fwd(x).cpu())
    return torch.cat(out)


base = score(outputs(None), st['subj'], st['img'], gal, hits=True)
ad_t, ad_ts = {}, {}
for s in FOLDS[k]:
    Xf, _, imf = cell([s], SEEN)
    P = torch.tensor([float(n == 0) for n in TAPS])[:, None].repeat(1, 32).to(DEV).requires_grad_(True); M = torch.eye(32, device=DEV).requires_grad_(True)
    for stage, params in (('t', [P]), ('ts', [P, M])):
        opt = torch.optim.Adam(params, lr=3e-3); g = torch.Generator().manual_seed(0)
        for ep in range(3):
            perm = torch.randperm(len(Xf), generator=g); tot = 0
            for i, b in enumerate(range(0, len(perm) - 255, 256)):
                idx = np.sort(perm[b:b + 256].numpy())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    y = fwd(apply(P, M, torch.as_tensor(Xf[idx]).to(DEV)))
                loss = Fn.cross_entropy(y @ Tf.T, torch.as_tensor(imf[idx]).to(DEV)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        (ad_t if stage == 't' else ad_ts)[s] = (P.detach().clone(), M.detach().clone())
        print(f'sub-{s:02d} {stage}: fit loss {tot / (i + 1):.4f}; filter mean over electrodes', np.round(P.detach().mean(1).cpu().numpy(), 2).tolist(), '| re-mix off-diagonal norm %.2f' % (M.detach() - torch.eye(32, device=DEV)).norm(), flush=True)
torch.save({'t': {s: [x.cpu() for x in v] for s, v in ad_t.items()}, 'ts': {s: [x.cpu() for x in v] for s, v in ad_ts.items()}}, f'{OUT}/e17_person_adapters_{m}_f{k}.pt')
for name, ad in (('temporal filter per person', ad_t), ('temporal + spatial per person', ad_ts)):
    sc, h = score(outputs(ad), st['subj'], st['img'], gal, hits=True)
    print('%-32s' % name, {a: round(b, 3) for a, b in sc.items()}, 'vs none', {a: round(b, 3) for a, b in base[0].items()}, '| r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (*paired(h['r3'], base[1]['r3']), *paired(h['r4'], base[1]['r4'])),
          '| per person r3: ' + ' '.join('%d: %.2f->%.2f' % (s, 100 * base[1]['r3'][base[1]['subj'] == s].mean(), 100 * h['r3'][h['subj'] == s].mean()) for s in FOLDS[k]), flush=True)
