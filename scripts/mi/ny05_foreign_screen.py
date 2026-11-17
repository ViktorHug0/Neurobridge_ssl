"""Y05 which members matter on people from ANOTHER LAB? The composition was chosen on held-out Alljoined people (cv4 folds); Y04 shows the
packages behave differently on a foreign cohort (ST-EEGFormer: +1-4 % next to B on held-out Alljoined people, +9 % on ds007964). The sealed
people are a new recording campaign and decode about twice worse than held-out Alljoined people (E00): somewhere between the two.
Fold-k models (none trained on these cohorts), cohort trials time-aligned (nc_lib.cohort), the cohort's own gallery, r=3, per-64-batch
scaled sum. Base = sub24's ten of the fold. Tables: each base member dropped; each candidate added; each candidate swapped for rmix / d6 soft.
Change in % against the ten, paired bootstrap over (person, image) groups.
    ny05_foreign_screen.py <cohort> <fold>"""
from nc_lib import *
from nv02_union_lists import TEN, CANDS

name, k = sys.argv[1], int(sys.argv[2]); X, subj, img, G, g3 = cohort(name); n = len(X) // 64 * 64; g3 = g3[(g3 < n).all(1)]; t = torch.as_tensor(img)[g3[:, 0]]


def out(c):
    p = f'{OUT}/y05_{name}_f{k}_{c}.npy'
    if not os.path.exists(p):
        r, ck = c.split('_checkpoint_'); f = load(r, k, 'checkpoint_' + ck); np.save(p, f(X[:n], bs=int(os.environ.get('BS', 256))).numpy().astype(np.float16)); del f; torch.cuda.empty_cache()
    return scaled(torch.as_tensor(np.load(p).astype(np.float32)))


have = lambda c: bool(glob.glob(f"{ROOT}/{c.split('_checkpoint_')[0]}/fold{k}/*/*/checkpoint_{c.split('_checkpoint_')[1]}"))
Y = {c: out(c) for c in TEN + [c for c in CANDS if have(c)]}; S = sum(Y[c] for c in TEN); bi = np.random.default_rng(0).integers(0, len(g3), (1000, len(g3)))
rk = lambda s: rank(s[g3].mean(1), t, G).numpy(); b = rk(S)
def line(lab, s):
    a = rk(s); print('%-58s' % lab + ' | '.join('%s %+5.1f +- %.1f' % (m, 100 * ((a < q).mean() / (b < q).mean() - 1), 100 * ((a[bi] < q).mean(1) / (b[bi] < q).mean(1) - 1).std()) for m, q in (('top-5', 5), ('top-20', 20), ('top-100', 100))), flush=True)
print(f'{name}, fold-{k} models: {len(g3)} groups, gallery {len(G)}; the ten: top-5 %.3f %%, top-20 %.3f %%, top-100 %.2f %%' % tuple(100 * (b < q).mean() for q in (5, 20, 100)))
print('-- each member of the ten DROPPED'); [line(c, S - Y[c]) for c in TEN]
print('-- each candidate ADDED'); [line(c, S + Y[c]) for c in Y if c not in TEN]
for slot in (TEN[1], TEN[6]):
    print('-- each candidate SWAPPED for', slot); [line(c, S - Y[slot] + Y[c]) for c in Y if c not in TEN]
