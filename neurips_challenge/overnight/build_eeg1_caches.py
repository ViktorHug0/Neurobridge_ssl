"""Build the fp16 training caches of the EEG1 pseudo-subjects (sub-31..42) on the NAS, then mark eeg1_caches_ok
(cv4.sh *_e1_things refuses to run without it: otherwise every job rebuilds them into its node's /tmp)."""
import json, os, sys
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from module.dataset import EEGPreImageDataset
DATA = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
ch = json.load(open(f'{DATA}/info.json'))['ch_names']
import sys
IDS = [int(a) for a in sys.argv[1:]] or list(range(31, 43))
for s in IDS:
    d = EEGPreImageDataset([s], DATA, ch, [0, 120], '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1',
                           '', False, [], False, False)
    print(s, len(d), flush=True); del d
open(f'{DATA}/.cache/' + {31: 'eeg1', 43: 'eeg1m', 55: 'eeg1w', 67: 'eeg1s', 79: 'eeg1r'}[IDS[0]] + '_caches_ok', 'w').write(f'{IDS}\n')
