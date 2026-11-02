"""AUG_LOG: pairwise hyperalignment rotations between training subjects, for the --hyperalign_prob augmentation.
For every ordered pair (s, t) of the same dataset, R_st = argmin_{R orthogonal} ||A_t - R A_s|| over the repeat-averaged
training responses A (32 ch x [images x time]) to the SAME images (Procrustes: R = U V^T, U S V^T = A_t A_s^T).
Noise is independent across people, so the cross-covariance A_t A_s^T only carries the image-locked signal.
A map s->t only uses s and t's TRAINING-image data, so one file serves every cv4 fold (pairs are drawn among the
fold's training subjects).
    hyperalign_maps.py [out.npy]   -> (31, 31, 32, 32) float32, index = subject id (0 unused), identity off-dataset
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.dataset import _eeg_cache_key, _eeg_cache_path  # noqa: E402

D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{D}/hyperalign_R.npy'
CH = json.load(open(f'{D}/info.json'))['ch_names']
S = list(range(1, 31))


X = {s: np.load(_eeg_cache_path(D, _eeg_cache_key(subject_id=s, train=True, average=False, selected_channels=CH,
                                                 time_window=[0, 120]))[:-4] + '_fp16.npy', mmap_mode='r') for s in S}
C = np.zeros((len(S) * 32, len(S) * 32))  # [s ch, t ch] cross-products of the repeat means (baseline-corrected data)
for i in range(0, 1654, 100):
    a = []
    for s in S:
        x = np.asarray(X[s][i:i + 100], np.float32)  # (100, 10, 4, 32, 120)
        n = (np.abs(x[..., :8]).sum((-1, -2)) > 0).sum(2)  # valid repeats per image (missing ones are zeros)
        a.append((x.sum(2) / np.maximum(n, 1)[..., None, None]).reshape(-1, 32, 120).transpose(1, 0, 2).reshape(32, -1))
    a = np.concatenate(a); C += a @ a.T; print('images', i, flush=True)
np.save(OUT.replace('_R.npy', '_C.npy'), C)
blk = lambda s, t: C[(s - 1) * 32:s * 32, (t - 1) * 32:t * 32]  # A_s A_t^T
R = np.tile(np.eye(32, dtype=np.float32), (31, 31, 1, 1))
for s in S:
    for t in S:
        if s != t and (s >= 21) == (t >= 21):
            U, _, Vt = np.linalg.svd(blk(t, s))
            R[s, t] = U @ Vt
# ||A_t - R A_s||^2 = tr(A_t A_t^T) + tr(A_s A_s^T) - 2 tr(R A_s A_t^T)
err = lambda s, t, M: np.trace(blk(t, t)) + np.trace(blk(s, s)) - 2 * np.trace(M @ blk(s, t))
fit = {(s, t): 1 - err(s, t, R[s, t]) / err(s, t, np.eye(32)) for s, t in ((1, 3), (21, 22), (6, 9))}
print('residual reduction vs identity', {k: round(float(v), 3) for k, v in fit.items()})
np.save(OUT, R); print('saved', OUT)
