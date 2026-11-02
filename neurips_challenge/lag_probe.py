"""AUG_LOG: does a subject's image-specific response sit at a time lag from the group's? (sub-09 / sub-17 show no zero-lag
image-locked correlation.) Per-image repeat means of the first N training concepts, each subject's grand ERP removed;
correlation of subject s at time t + lag with the mean of the other Alljoined subjects at time t, over channels/images."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from module.dataset import _eeg_cache_key, _eeg_cache_path
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
CH = json.load(open(f'{D}/info.json'))['ch_names']
N = 300  # concepts (x 10 images)


def means(s):
    X = np.asarray(np.load(_eeg_cache_path(D, _eeg_cache_key(subject_id=s, train=True, average=False, selected_channels=CH,
                                                             time_window=[0, 120]))[:-4] + '_fp16.npy', mmap_mode='r')[:N], np.float32)
    n = (np.abs(X[..., :8]).sum((-1, -2)) > 0).sum(2)
    A = (X.sum(2) / np.maximum(n, 1)[..., None, None]).reshape(-1, 32, 120)
    return A - A.mean(0)  # image-specific part


A = {s: means(s) for s in range(1, 21)}
lags = range(-24, 25, 2)  # samples at 120 Hz (+-200 ms)
for s in [int(a) for a in sys.argv[1:]] or [9, 17, 3]:
    g = np.mean([A[t] for t in A if t != s], 0)
    r = []
    for L in lags:
        a, b = (A[s][..., L:], g[..., :120 - L]) if L >= 0 else (A[s][..., :120 + L], g[..., -L:])
        r.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    best = int(np.argmax(r))
    print(f'sub-{s:02d} r(lag 0) {r[len(r) // 2]:.4f}  best lag {list(lags)[best]:+d} samples ({list(lags)[best] / 120 * 1000:+.0f} ms) r {r[best]:.4f}  |  ' + ' '.join(f'{v:.3f}' for v in r))
