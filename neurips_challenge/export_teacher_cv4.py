"""cv4 fold teacher for distillation (A18): mean L2-normalised prediction of the fold's three base models (TSC bn64,
SQF dbn, REVE d6; final-epoch weights, trained on exactly this fold's training subjects) on every training trial, in
train.py's (subject, object, image, rep) order -> --distill_targets. Differs from export_teacher.py in three ways the
cv4 scale needs: rows go straight to an fp16 memmap (1.65M x 1536 would not fit in RAM twice), the real subject ids are
passed (per-dataset BatchNorm: THINGS trials take the THINGS path they were trained on), and the NAS data dir is read.

    export_teacher_cv4.py <fold>      -> results/alljoined/track1/cv4/teachers/fold<k>.npy
    export_teacher_cv4.py <fold> <recipe> ...   -> teachers/fold<k>_<n>m.npy (n-member teacher)
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
from neurips_challenge.score_track1 import build_modules  # noqa: E402
from train import run_eeg_backbone  # noqa: E402

K = sys.argv[1]
R = 'results/alljoined/track1/cv4'
RECIPES = sys.argv[2:] or ('tsc_dbn40_bn64_things', 'sqf_dbn_things', 'reve_d6_things')  # TTA_LOG T36: optional bigger teacher
OUT = f'{R}/teachers/fold{K}' + (f'_{len(RECIPES)}m' if sys.argv[2:] else '') + '.npy'
runs = [sorted(glob.glob(f'{R}/{r}/fold{K}/*/2026*/checkpoint_test_best.pth'))[0] for r in RECIPES]
device = torch.device('cuda')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = None
with torch.no_grad():
    for n, ck in enumerate(runs):
        cfg = SimpleNamespace(**json.load(open(os.path.join(os.path.dirname(ck), 'train_config.json'))))
        if out is None:
            dataset = EEGPreImageDataset(cfg.train_subject_ids, cfg.eeg_data_dir, cfg.selected_channels, cfg.time_window,
                                         cfg.image_feature_dir, '', False, [], False, False, None, True, False, False, False)
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
