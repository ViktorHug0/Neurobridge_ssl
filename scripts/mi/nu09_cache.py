"""nu09_cache.py <fold> <recipe:checkpoint[:shift]> ...: cache fold-k outputs (held-out people x 5,000 never-trained images, cell rows) as
u01_f<k>_<recipe>_<checkpoint>.npy (shift 0, the file every nv01 / ny10 read uses) or u01s<shift>_f<k>_... for an input moved by <shift>
samples (x[t - shift], edges repeated), for snapshot and shift-averaging reads."""
from nc_lib import *

k = int(sys.argv[1]); X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X = torch.as_tensor(X[:n]); tt = torch.arange(120)
for spec in sys.argv[2:]:
    r, c, *s = spec.split(':'); s = int(s[0]) if s else 0; p = f"{OUT}/u01{'s%d' % s if s else ''}_f{k}_{r}_{c}.npy"
    if os.path.exists(p) or not glob.glob(f'{ROOT}/{r}/fold{k}/*/*/{c}'):
        continue
    f = load(r, k, c); x = X[:, :, (tt - s).clamp(0, 119)] if s else X
    np.save(p, f(x, bs=int(os.environ.get('BS', 256))).numpy().astype(np.float16)); del f; torch.cuda.empty_cache(); print('cached', p.split('/')[-1], flush=True)
