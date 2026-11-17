"""cv4 fold teacher for ensemble distillation (--distill_targets): mean L2-normalised prediction of the fold's members
(final-epoch weights, trained on exactly this fold's training subjects) on every training trial, in train.py's
(subject, object, image, rep) row order, written straight to an fp16 memmap (1.65M x 1536). Real subject ids are passed
(per-dataset BatchNorm: THINGS trials take the path they were trained on).

    export_teacher_cv4.py <fold> <recipe> ...   -> results/alljoined/track1/cv4/teachers/fold<k>_<recipes joined by +>.npy
"""
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import run_eeg_backbone  # noqa: E402
from neurips_challenge.score_track1 import build_modules  # noqa: E402
from neurips_challenge.tta_cache import DATA, ROOT  # noqa: E402

K, RECIPES = sys.argv[1], sys.argv[2:]
OUT = f'{ROOT}/teachers/fold{K}_{"+".join(RECIPES)}.npy'
runs = [sorted(glob.glob(f'{ROOT}/{r}/fold{K}/*/2026*/checkpoint_test_best.pth'))[0] for r in RECIPES]
device = torch.device('cuda')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = None
with torch.no_grad():
    for n, ck in enumerate(runs):
        cfg = SimpleNamespace(**json.load(open(os.path.join(os.path.dirname(ck), 'train_config.json'))))
        cfg.eeg_data_dir = DATA
        if out is None:
            dataset = EEGPreImageDataset(cfg.train_subject_ids, DATA, cfg.selected_channels, cfg.time_window,
                                         cfg.image_feature_dir, '', average=False, train=True)
            loader = DataLoader(dataset, batch_size=1024, num_workers=6)
            out = np.lib.format.open_memmap(OUT + '.tmp.npy', mode='w+', dtype=np.float16, shape=(len(dataset), 1536))
            ids = cfg.train_subject_ids
        assert cfg.train_subject_ids == ids, 'teachers must share the training set (row order)'
        model, eeg_projector, _ = build_modules(cfg, torch.load(ck, map_location=device, weights_only=False), dataset, device, 120)
        i = 0
        for eeg, _, _, subject_batch, _, _, _ in loader:
            p = torch.nn.functional.normalize(eeg_projector(run_eeg_backbone(model, cfg, eeg.to(device), subject_batch.to(device))).float(), dim=1)
            p = p.cpu().numpy() / len(runs)
            out[i:i + len(p)] = p if n == 0 else out[i:i + len(p)].astype(np.float32) + p
            i += len(p)
        assert i == len(out)
        print(f'{ck}: {i} rows', flush=True)
        del model, eeg_projector
out.flush(); del out
os.replace(OUT + '.tmp.npy', OUT)
print('saved', OUT, flush=True)
