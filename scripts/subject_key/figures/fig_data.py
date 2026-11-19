"""Real data behind the presentation figures (frozen ATM-128, validation-selected, seed 3300, fold 1: subject 1 held out).
Run from scripts/subject_key:  MI_DEV=cpu ../../.venv/bin/python figures/fig_data.py   -> figures/fig_data.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sk_lib import *

FOLD = 1
m = load('atm128_vc/seed3300', FOLD)
g = img_emb(m)
x = {s: T(eeg(s)) for s in range(1, 11)}                    # (200, 63, 250) 80-repetition averages
mu = {s: x[s].mean(0, keepdim=True) for s in x}             # the key: image-independent mean ERP
amap = lambda inp, s: atm_emb(m, inp, s)[1].mean(1)         # head-averaged attention, (n, 64, 64); token 0 = shared token
A_own = {s: amap(x[s], s) for s in x}
A_key = {s: amap(mu[s], s)[0] for s in x}
E_own = {s: nz(atm_emb(m, x[s], s)[0], dim=1) for s in x}
flat = lambda A: A[..., 1:, 1:].flatten(-2)
corr = lambda a, b: torch.nn.functional.cosine_similarity(a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True), dim=-1)

# pairwise statistics over the seen subjects (a = owner of the content, b = owner of the key)
seen = m.train_subs
stats = {k: [] for k in ('same_key_other_image', 'key_alone', 'same_image_other_subject', 'swap_vs_key_owner', 'swap_vs_content_owner')}
perm = torch.roll(torch.arange(200), 1)
swaps = {}
for a in seen:
    fa = flat(A_own[a])
    stats['same_key_other_image'].append(corr(fa, fa[perm]).mean().item())
    stats['key_alone'].append(corr(fa, flat(A_key[a])[None]).mean().item())
    for b in seen:
        if a == b:
            continue
        stats['same_image_other_subject'].append(corr(fa, flat(A_own[b])).mean().item())
        S = amap(x[a] - mu[a] + mu[b], a)                   # content of a under the key of b
        stats['swap_vs_key_owner'].append(corr(flat(S), flat(A_own[b])).mean().item())
        stats['swap_vs_content_owner'].append(corr(flat(S), fa).mean().item())
        swaps[(a, b)] = S[:8].numpy().astype(np.float16)
        print(a, b, {k: round(v[-1], 3) for k, v in stats.items()}, flush=True)
np.savez_compressed('figures/fig_data.npz', fold=FOLD, seen=np.array(seen), g=g.numpy(), anchor=img_emb(m, 'train').mean(0).numpy(),
                    x=torch.stack([x[s][:8] for s in range(1, 11)]).numpy(), mu=torch.cat([mu[s] for s in range(1, 11)]).numpy(),
                    A_own=torch.stack([A_own[s] for s in range(1, 11)]).numpy().astype(np.float16), A_key=torch.stack([A_key[s] for s in range(1, 11)]).numpy(),
                    E_own=torch.stack([E_own[s] for s in range(1, 11)]).numpy(),
                    swap_pairs=np.array(list(swaps)), swap_maps=np.stack(list(swaps.values())), **{'stat_' + k: np.array(v) for k, v in stats.items()})
print({k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) for k, v in stats.items()})
