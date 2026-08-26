"""Is subject 3 a scale/variance outlier? Every ortho arm collapses on that fold.

Reads each subject's test EEG one at a time and reports amplitude scale and the
spread of per-channel standard deviations.
"""
import glob
import numpy as np

D = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
print(f"{'sub':>3s} {'std':>8s} {'p99|x|':>8s} {'ch-std min':>10s} {'ch-std max':>10s} {'max/med':>8s}")
for s in range(1, 11):
    f = glob.glob(f'{D}/sub-{s:02d}/*test*.npy')
    if not f:
        print(f'{s:>3d} missing'); continue
    a = np.load(f[0], allow_pickle=True)
    if isinstance(a, np.ndarray) and a.dtype == object:
        a = a.item()
    if isinstance(a, dict):
        a = a['preprocessed_eeg_data']
    a = np.asarray(a, dtype=np.float32).reshape(-1, np.shape(a)[-2], np.shape(a)[-1])
    sub = a[::7]                                   # subsample trials to stay in memory
    cs = sub.std(axis=(0, 2))
    print(f'{s:>3d} {sub.std():8.3f} {np.percentile(np.abs(sub), 99):8.3f} '
          f'{cs.min():10.3f} {cs.max():10.3f} {cs.max()/np.median(cs):8.2f}')
    del a, sub
