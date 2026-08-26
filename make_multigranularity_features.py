"""Build a multi-granularity image target by concatenating L2-normalized InternViT layers.

SAMGA fuses several intermediate layers with a learned router; concatenating the layers and
letting the (already present) linear image projector learn the mixing is the same hypothesis
with no changes to train.py -- point --image_feature_dir at the output directory.

Each layer is L2-normalized before concatenation so no single layer's scale dominates.
"""
import argparse
import os

import numpy as np

BASE = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature'


def l2norm(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', nargs='+', type=int, default=[25, 28, 31])
    ap.add_argument('--dirs', nargs='+', default=None,
                    help='concatenate these feature dirs instead of InternViT layers '
                         '(use to fuse different vision backbones into one target)')
    ap.add_argument('--out_name', default=None)
    args = ap.parse_args()

    sources = args.dirs or [f'InternViT-6B_layer{l}_mean_8bit' for l in args.layers]
    out_name = args.out_name or 'InternViT-6B_concat_' + '-'.join(str(l) for l in args.layers)
    out_dir = os.path.join(BASE, out_name)
    os.makedirs(out_dir, exist_ok=True)

    for split in ('image_train.npy', 'image_test.npy'):
        parts = []
        for src_dir in sources:
            src = os.path.join(BASE, src_dir, split)
            parts.append(l2norm(np.load(src).astype(np.float32)))
        merged = np.concatenate(parts, axis=-1).astype(np.float16)
        np.save(os.path.join(out_dir, split), merged)
        print(f'{split}: {merged.shape} -> {out_dir}')


if __name__ == '__main__':
    main()
