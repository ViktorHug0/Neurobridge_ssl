"""AUG_LOG: per-subject grand-average ERP of the repeat-averaged training responses (mean over the 16540 images), the
quantity hyperalign_maps.py's cross-products need removed to keep only image-SPECIFIC signal.  -> (31, 32, 120) .npy"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.dataset import _eeg_cache_key, _eeg_cache_path  # noqa: E402
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
CH = json.load(open(f'{D}/info.json'))['ch_names']
M = np.zeros((31, 32, 120))
for s in range(1, 31):
    X = np.load(_eeg_cache_path(D, _eeg_cache_key(subject_id=s, train=True, average=False, selected_channels=CH, time_window=[0, 120]))[:-4] + '_fp16.npy', mmap_mode='r')
    for i in range(0, 1654, 100):
        x = np.asarray(X[i:i + 100], np.float32)
        n = (np.abs(x[..., :8]).sum((-1, -2)) > 0).sum(2)
        M[s] += (x.sum(2) / np.maximum(n, 1)[..., None, None]).reshape(-1, 32, 120).sum(0)
    M[s] /= 16540; print('subject', s, flush=True)
np.save(sys.argv[1], M)
