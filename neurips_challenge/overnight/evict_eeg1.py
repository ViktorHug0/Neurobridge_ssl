"""Drop the THINGS-EEG1 files (pseudo-subjects 31..90: raw arrays + fp16/f32 caches) from this node's page cache, so
they stop competing with a run's warmed working set inside the job's memory limit (posix_fadvise DONTNEED)."""
import json, os, sys
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from module.dataset import _eeg_cache_key, _eeg_cache_path
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
ch = json.load(open(f'{D}/info.json'))['ch_names']
paths = []
for s in range(31, 91):
    p = _eeg_cache_path(D, _eeg_cache_key(subject_id=s, train=True, average=False, selected_channels=ch, time_window=[0, 120]))
    paths += [p, p[:-4] + '_fp16.npy', f'{D}/sub-{s:02d}/train.npy']
n = 0
for p in paths:
    if os.path.exists(p):
        fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
print('evicted', n, 'files', flush=True)
