"""Adds decoy embeddings to the open_gallery cache: image 1 of every training concept (distractors use image 0),
pushed through the subject's frozen image projector. Decoys are never put in the gallery; they are known-absent
candidates from the same pool as the distractors.
usage: add_decoys.py --seed 3301
"""
import argparse, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from shared import load_subject_context  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='atm128_vc')
ap.add_argument('--seed', type=int, required=True)
a = ap.parse_args()
for sid in range(1, 11):
    f = f'{C}/cache/{a.arm}_s{a.seed}/sub-{sid:02d}.npz'
    z = dict(np.load(f))
    if 'dec' in z:
        continue
    rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
    _, ea, _, mods = load_subject_context(f'{R}/results/things_eeg/cvpr/loso/{a.arm}/seed{a.seed}', rt, sid, average=False)
    raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy'), mmap_mode='r')
    z['dec'] = project_raw_images(ea, mods, np.asarray(raw[np.random.default_rng(1000 + sid).permutation(1654), 1], np.float32))
    np.savez(f, **z)
    print(sid, z['dec'].shape, flush=True)
