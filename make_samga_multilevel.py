"""Build SAMGA's stacked multilevel image target: [n_obj, n_img, K, D].

SAMGA's router expects a batch of shape [B, K, D] -- one candidate representation per visual
layer, NOT a concatenation. Their release uses InternViT layers 20/24/28/32/36; this repo has
23/25/27/28/29/31/33 extracted, so the default here is the symmetric 23/25/28/31/33 around the
same depth prior centre (28).

Written layer-by-layer into a preallocated fp16 array so peak RSS stays near one layer.
"""
import argparse
import os

import numpy as np

BASE = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', nargs='+', type=int, default=[23, 25, 28, 31, 33])
    ap.add_argument('--out_name', default=None)
    args = ap.parse_args()

    out_name = args.out_name or 'internvit_multilevel_' + '_'.join(str(l) for l in args.layers)
    out_dir = os.path.join(BASE, out_name)
    os.makedirs(out_dir, exist_ok=True)

    for split in ('image_train.npy', 'image_test.npy'):
        srcs = [os.path.join(BASE, f'InternViT-6B_layer{l}_mean_8bit', split) for l in args.layers]
        head = np.load(srcs[0], mmap_mode='r')
        out = np.empty(head.shape[:-1] + (len(args.layers), head.shape[-1]), dtype=np.float16)
        for k, src in enumerate(srcs):
            x = np.load(src).astype(np.float32)
            x /= np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
            out[..., k, :] = x.astype(np.float16)
            del x
        np.save(os.path.join(out_dir, split), out)
        print(f'{split}: {out.shape} -> {out_dir}')


if __name__ == '__main__':
    main()
