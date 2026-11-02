"""Identity set for the participant embedder (ROUTING_LOG E48c): 4000 single trials per THINGS-EEG1 subject (48 extra
identities next to the 19 Alljoined + THINGS-EEG2 training participants), preprocessed exactly like the deleted
sub-31..42 export (export_things_eeg1_to_alljoined.load = the grader's neuralbench chain). Training images only.
    eeg1_identity_cache.py [subject ...]  -> <AJ>/.cache/eeg1id_sXX.npy (4000, 32, 120) float32
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.export_things_eeg1_to_alljoined import AJ, IMAGES, SUBJECTS, load  # noqa: E402
from preprocess_eeg1 import eeg2_image_index  # noqa: E402

if __name__ == '__main__':
    image_index, _ = eeg2_image_index(IMAGES)
    for s in [int(a) for a in sys.argv[1:]] or SUBJECTS:  # one subject per process keeps memory flat
        out = f'{AJ}/.cache/eeg1id_s{s:02d}.npy'
        if os.path.exists(out):
            continue
        x, present = load(s, image_index)
        idx = np.sort(np.random.default_rng(s).choice(np.where(present)[0], min(4000, present.sum()), replace=False))
        np.save(out + '.tmp.npy', x[idx]); os.replace(out + '.tmp.npy', out)
        print(f'EEG1 sub-{s:02d}: {len(idx)} trials', flush=True)
