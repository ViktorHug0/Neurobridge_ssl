#!/bin/bash
# Y01 from a queue line: open-image targets (COCO + ImageNet val), then the two bank files of Y02 (seeded halves: the other halves are the proxy's out-of-THINGS gallery)
cd /nasbrain/p20fores/Neurobridge_SSL
.venv/bin/python scripts/mi/ny01_open_images.py > logs/mi/y01_open_images.log 2>&1
.venv/bin/python - >> logs/mi/y01_open_images.log 2>&1 <<'PY'
import numpy as np
D = 'data/things_eeg/image_feature/dinov2-giant_track1'; c, i = np.load(f'{D}/open_coco.npy'), np.load(f'{D}/open_imagenet.npy')
pc, pi = np.random.default_rng(0).permutation(len(c)), np.random.default_rng(0).permutation(len(i))
np.save(f'{D}/open_bank45k.npy', np.concatenate([c[pc[:20000]], i[pi[:25000]]]))          # loss bank (Y02)
np.save(f'{D}/open_gallery45k.npy', np.concatenate([c[pc[20000:]], i[pi[25000:]]]))      # proxy distractors (Y03), disjoint from the bank
T = np.load(f'{D}/image_train.npy').reshape(16540, 1536); held = np.random.default_rng(0).choice(16540, 5000, replace=False)  # = train.py held_out()
np.save(f'{D}/things_trained_bank.npy', T[np.setdiff1d(np.arange(16540), held)].astype(np.float16))  # the 11,540 images a ho5k run trains on
print('banks saved', flush=True)
PY
