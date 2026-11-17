"""E13 is the E07 filter a property of the datasets or of the instrument? The per-electrode filter fitted through the
Alljoined-only REVE-d6 is applied to THINGS trials decoded by ANOTHER Alljoined-only model (rmix, fold 3, ep10):
five test THINGS people x never-trained images, 5k, exact ranks, against the plain 5-sample lag."""
from nc_lib import *
from nd03_cross_dataset_lib import shift
F = torch.tensor(np.load(f'{OUT}/things_filter_e07.npy')).to(DEV)  # (11, 32), tap i = later by i samples
X, subj, img = cell([21, 23, 25, 27, 29], HELD); Xt = torch.as_tensor(X); gal = gallery()
for m in sys.argv[1:] or ['rmix_reg_ho5k_mid_optx_aj']:
    f = load(m, 3, 'checkpoint_ep10.pth')
    lag = f(X, lambda a, b: shift(Xt[a:b].to(DEV), 5) - Xt[a:b].to(DEV))
    fil = f(X, lambda a, b: sum(F[i][:, None] * shift(Xt[a:b].to(DEV), i) for i in range(11)) - Xt[a:b].to(DEV))
    (sl, hl), (sf, hf) = score(lag, subj, img, gal, hits=True), score(fil, subj, img, gal, hits=True)
    print(m, '| lag only', {a: round(b, 3) for a, b in sl.items()}, '| filter', {a: round(b, 3) for a, b in sf.items()}, '| r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (*paired(hf['r3'], hl['r3']), *paired(hf['r4'], hl['r4'])), flush=True)
