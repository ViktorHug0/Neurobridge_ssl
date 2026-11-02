"""Label-alignment sanity check: does the same image evoke more similar responses across independent subject groups
than different images do? r_same - r_diff on per-image averages (window 0.05-0.40 s, all channels)."""
import numpy as np
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
win = slice(24 + 6, 24 + 48)
def img_mean(s, reps=None):
    X = np.load(f'{D}/sub-{s:02d}/train.npy', mmap_mode='r')
    X = np.asarray(X[:, :, reps if reps is not None else slice(None)][..., win], np.float32).mean(2)  # (1654,10,32,T)
    X = X.reshape(16540, -1); X -= X.mean(0); X /= X.std(0) + 1e-6
    return X
def score(A, B, name):
    same = (A * B).mean(1)                                 # per-image correlation-like score
    perm = np.random.default_rng(0).permutation(len(B))
    diff = (A * B[perm]).mean(1)
    se = np.sqrt(same.var() / len(same) + diff.var() / len(diff))
    print(f'{name:40s} same {same.mean():+.4f}  diff {diff.mean():+.4f}  z {(same.mean() - diff.mean()) / se:+.1f}', flush=True)
e2a, e2b = img_mean(21), img_mean(22)
score(e2a, e2b, 'EEG2 sub-21 vs sub-22 (4 reps each)')
e1a, e1b = img_mean(31), img_mean(32)
score(e1a, e1b, 'EEG1 subj 1-4 vs 5-8 (4 subj each)')
score(e1a, e2a, 'EEG1 subj 1-4 vs EEG2 sub-21')
o1 = np.load('/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Preprocessed_data_250Hz/sub-02/preprocessed_eeg_training.npy', allow_pickle=True).item()
print('old 250 Hz export loaded', o1['preprocessed_eeg_data'].shape, flush=True)
aj = img_mean(1)
score(e2a, aj, 'EEG2 sub-21 vs Alljoined sub-01')
score(e1a, aj, 'EEG1 subj 1-4 vs Alljoined sub-01')
