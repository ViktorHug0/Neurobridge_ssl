"""R03 is a new person's decoding limited by how well the member can read WHO the trial is from? (frozen member)
The members decode conditionally on the person's image-independent mean ERP (the key, N09). A single trial holds the key
exactly but mixed with the image response and the noise. Test: x -> x + beta * key_s, with the person's key (a) from all
their trials (oracle) or (b) from the other trials of that person in the same 64-batch (what a test batch can give).
beta > 0 makes the identity louder; a gain means identity read-out is a bottleneck and batch-mates can supply it.
Held-out people, never-trained images, 5k, r=3: top-5 %, top-100 %, 1 - AUC.
    nr03_key_gain.py <recipe> <fold> <checkpoint>"""
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]; f = load(recipe, k, ck); gal = gallery(); X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X)
groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]
K = torch.zeros_like(X); Kb = torch.zeros_like(X); s_ = torch.as_tensor(subj)
for p in FOLDS[k]:
    K[s_ == p] = X[s_ == p].mean(0)
for b in range(0, len(X), 64):  # leave-one-out mean of the same person's trials in the batch (the proxy batch holds 5 people, ~13 trials each; the sealed batch ~6)
    sb = s_[b:b + 64]; xb = X[b:b + 64]
    for p in sb.unique():
        m = sb == p; n = int(m.sum()); Kb[b:b + 64][m] = (xb[m].sum(0, keepdim=True) - xb[m]) / max(n - 1, 1)
print(recipe, ck, 'fold', k, '| key norm / trial norm %.3f | batch-mate key vs oracle key: cosine %.3f' % (K.flatten(1).norm(dim=1).mean() / X.flatten(1).norm(dim=1).mean(), torch.nn.functional.cosine_similarity(K.flatten(1), Kb.flatten(1)).mean()))
for name, Y in [('intact', X)] + [('oracle %+.2f' % b, X + b * K) for b in (-1, -0.5, 0.5, 1, 2)] + [('mates %+.2f' % b, X + b * Kb) for b in (-1, 0.5, 1, 2)]:
    r = rank(scaled(f(Y, bs=1000).cpu())[g4[:, :3]].mean(1), tg, gal).cpu().numpy()
    print('%-13s top-5 %.3f | top-100 %.2f | 1 - AUC %.4f' % (name, 100 * (r < 5).mean(), 100 * (r < 100).mean(), r.mean() / 4999), flush=True)
