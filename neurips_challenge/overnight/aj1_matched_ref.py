"""Matched in-domain vs Alljoined-1 zero-shot: fold-1 models, 200-way gallery, 8 repeats per image, top-5.
  in-domain: fold-1 test subjects (unseen), test.npy 200 images, 8 random repeats each
  Alljoined-1: 8 subjects, 20 random 200-image galleries out of the 960, all (<=8) repeats
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from neurips_challenge.score_track1 import build_modules, build_test_dataset
from neurips_challenge.cv4_ensemble import DATA, run_dirs
from neurips_challenge.trial_routing import FOLDS
from neurips_challenge.zeroshot_alljoined1 import D, n
from train import run_eeg_backbone

dev = torch.device('cuda'); rng = np.random.default_rng(0)
z = np.load(f'{D}/dinov2_track1.npz'); row = {i: r for r, i in enumerate(z['ids'])}; G1 = z['F']
GT = np.load('/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/image_test.npy').reshape(200, -1)


def top5(P, G):  # P[i] belongs to G[i]
    s = n(P) @ n(G).T
    return float(np.mean([(s[i] > s[i, i]).sum() < 5 for i in range(len(P))]))


@torch.no_grad()
def predict(model, cfg, eproj, X):
    return np.concatenate([eproj(run_eeg_backbone(model, cfg, torch.as_tensor(X[b:b + 256].astype(np.float32), device=dev),
                                                   torch.full((len(X[b:b + 256]),), 10 ** 6, device=dev))).float().cpu().numpy()
                           for b in range(0, len(X), 256)])


for r in sys.argv[1:]:
    d = run_dirs(r)[1]
    cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
    ck = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=dev, weights_only=False)
    model, eproj, _ = build_modules(cfg, ck, build_test_dataset(cfg, [FOLDS[1][0]]), dev, 120)
    ind = []
    for s in FOLDS[1]:
        X = np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(200, 80, 32, 120)
        cnt = np.load(f'{DATA}/sub-{s:02d}/test_counts.npy').ravel()
        idx = [(i, j) for i in range(200) for j in rng.choice(max(1, min(cnt[i], 80)), 8, replace=min(cnt[i], 80) < 8)]
        p = predict(model, cfg, eproj, np.stack([X[i, j] for i, j in idx])).reshape(200, 8, -1).mean(1)
        ind.append(top5(p, GT))
    aj = []
    for s in range(1, 9):
        zz = np.load(f'{D}/sub-{s:02d}.npz'); p = predict(model, cfg, eproj, zz['X']); y = zz['nsd']
        u = np.unique(y); P = np.stack([p[y == i].mean(0) for i in u]); g = G1[[row[i] for i in u]]
        for _ in range(20):
            k = rng.choice(len(u), 200, replace=False); aj.append(top5(P[k], g[k]))
    print(f'RESULT {r}: 200-way top-5, 8 reps | in-domain unseen subj {100 * np.mean(ind):.2f} | Alljoined-1 {100 * np.mean(aj):.2f} | chance 2.50', flush=True)
    del model, eproj; torch.cuda.empty_cache()
