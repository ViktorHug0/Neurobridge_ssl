"""W01 outputs of a candidate composition at input shifts -5..+5, for the alignment replay (`E2E=<tag> ns06_align_batching.py <fold>`):
the same cache as nr07 builds for the sub16 composition, for any member list -> <OUT>/autolag_e2e_<tag>_f<k>.npy (11, n, 1536) float16.
    nw01_e2e_cache.py <fold> <tag> <recipe:checkpoint> [...]"""
from nc_lib import *

k, tag = int(sys.argv[1]), sys.argv[2]; S = list(range(-5, 6)); tt = torch.arange(120)
X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X); n = len(X) // 64 * 64; X = X[:n]; E = np.zeros((len(S), n, 1536), np.float16)
for spec in sys.argv[3:]:
    recipe, ck = spec.split(':'); f = load(recipe, k, ck)
    for j, s in enumerate(S):
        E[j] += scaled(f(X[:, :, (tt - s).clamp(0, 119)], bs=1000).cpu().float()).numpy().astype(np.float16)
    print('cached', recipe, flush=True); del f; torch.cuda.empty_cache()
np.save(f'{OUT}/autolag_e2e_{tag}_f{k}.npy', E)
