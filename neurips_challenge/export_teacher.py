"""Per-trial ensemble teacher for distillation: mean of the runs' L2-normalised predictions on
every TRAINING trial, in train.py's (subject, object, image, rep) flat order. train.py reads it
with --distill_targets (row = dataset index).

    export_teacher.py <out.npy> [--average] <run_dir> [<run_dir> ...]

--average: the teachers see each subject's 4-repetition mean (privileged, cleaner input) and every
repetition row gets that prediction.
"""

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

out, run_dirs = sys.argv[1], [a for a in sys.argv[2:] if a != "--average"]
average = "--average" in sys.argv
device = torch.device("cuda")
total, dataset = None, None
with torch.no_grad():
    for run_dir in run_dirs:
        cfg = SimpleNamespace(**json.load(open(os.path.join(run_dir, "train_config.json"))))
        checkpoint = torch.load(os.path.join(run_dir, "checkpoint_test_best.pth"), map_location=device,
                                weights_only=False)
        if dataset is None:  # training trials, train.py's own data dir (no channel permutation)
            dataset = EEGPreImageDataset(cfg.train_subject_ids, cfg.eeg_data_dir, cfg.selected_channels,
                                         cfg.time_window, cfg.image_feature_dir, "", False, [], average, False,
                                         None, True, False, False, False)  # train split, no augmentation
            loader = DataLoader(dataset, batch_size=1024, num_workers=6)
        model, eeg_projector, _ = build_modules(cfg, checkpoint, dataset, device, 120)
        preds = []
        for eeg, _, _, subject_batch, _, _, _ in loader:
            unknown = torch.full_like(subject_batch, 10**6).to(device)
            p = eeg_projector(run_eeg_backbone(model, cfg, eeg.to(device), unknown)).float()
            preds.append(torch.nn.functional.normalize(p, dim=1).half().cpu())
        preds = torch.cat(preds).float()
        if average:  # (s, o, i) rows -> (s, o, i, r) rows
            preds = preds.repeat_interleave(4, dim=0)
        total = preds if total is None else total + preds
        print(f"{run_dir}: {len(preds)} rows", flush=True)
np.save(out, (total / len(run_dirs)).half().numpy())
print("saved", out, tuple(total.shape))
