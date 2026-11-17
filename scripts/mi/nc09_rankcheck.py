from nc_lib import *  # how often does the legacy target-rank (separate dot product) count the target above itself?
from nc_lib import *
F = f'{R}/logs/holdout/f1'; st = np.load(f'{F}/stream.npz')
y = scaled(torch.as_tensor(np.load(f'{F}/rmix_reg_ho5k_long_optx_things__ep10.npy')).float())
t = torch.as_tensor(st['img'])
for big in (False, True):
    g = gallery(big=big)
    a, b = rank(y, t, g), rank(y, t, g, legacy=True)
    print('gallery', len(g), '| queries where legacy rank is one too high: %.1f %%' % (100 * (b > a).float().mean()),
          '| top-1 exact %.4f legacy %.4f | top-5 exact %.4f legacy %.4f' % (100 * (a < 1).float().mean(), 100 * (b < 1).float().mean(), 100 * (a < 5).float().mean(), 100 * (b < 5).float().mean()))
