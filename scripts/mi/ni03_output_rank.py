"""I05 why is TSConv weak on the clean proxy? Its bottleneck head makes the output rank 64. Effective rank of each
member's outputs (fold 1, held-out people x never-trained images), and the score of each member alone when its output is
cut to its top-k principal directions (5k gallery, r=3 / r=4, exact ranks)."""
from nc_lib import *
st = np.load(f'{R}/logs/holdout/f1/stream.npz'); subj, img = st['subj'], st['img']; gal = gallery()
M = {'TSConv lag @36': 'mi_tsc_ho5k_lag5_ep36', 'd6 lag @12': 'mi_d6_ho5k_lag5_ep12', 'rmix lag @10': 'mi_rmix_ho5k_lag5_ep10', 'SQF-X lag': 'mi_sqfx_ho5k_lag5_ep40', 'LaBraM lag @8': 'mi_plab_ho5k_lag5_ep08'}
for n, p in M.items():
    y = torch.as_tensor(np.load(f'{OUT}/hold_f1_{p}_raw.npy')).float()
    U, S, V = torch.pca_lowrank(y[::5].to(DEV), q=600, center=False, niter=2); S = S.cpu(); V = V.cpu()
    e = S ** 2 / (S ** 2).sum()
    rank = int((S > 1e-3 * S[0]).sum()); pr = float(1 / (e ** 2).sum())
    row = ['full %.3f / %.3f' % (score(y, subj, img, gal)['r3'], score(y, subj, img, gal)['r4'])]
    for k in (256, 64, 32):
        yk = (y @ V[:, :k]) @ V[:, :k].T
        s = score(yk, subj, img, gal); row.append('top %d: %.3f / %.3f' % (k, s['r3'], s['r4']))
    print('%-16s directions above 0.1 %% of the first: %3d of 600 measured | participation ratio %.1f | %s' % (n, rank, pr, ' | '.join(row)), flush=True)
