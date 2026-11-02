"""Probe per-trial output post-processing on one run's grader predictions (no cross-trial info).

    postproc_probe.py <run_dir>

kNN prior: p' = a * p/|p| + (1 - a) * sum_j softmax(beta * cos(p, t_j)) t_j/|t_j| over the
16,540 TRAINING image embeddings t_j, which a submission can carry. Also tries mean shifts.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.score_track1 import OFFICIAL_EEG_DIR, build_modules, build_test_dataset  # noqa: E402

run_dir = sys.argv[1]
SHIFTS = [0, -1, 1, -2, 2, -4, 4]
MC = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--mc=')), 0))
TTA_ONLY = '--tta' in sys.argv
device = torch.device("cuda")
cfg = SimpleNamespace(**json.load(open(os.path.join(run_dir, "train_config.json"))))
cfg.eeg_data_dir = OFFICIAL_EEG_DIR
checkpoint = torch.load(os.path.join(run_dir, "checkpoint_test_best.pth"), map_location=device, weights_only=False)
n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])

from train import run_eeg_backbone  # noqa: E402

preds, labels, gallery = [], [], None
with torch.no_grad():
    for subject_id in cfg.test_subject_ids:
        dataset = build_test_dataset(cfg, [subject_id])
        if gallery is None:
            model, eeg_projector, _ = build_modules(cfg, checkpoint, dataset, device, n_times)
            gallery = torch.nn.functional.normalize(
                torch.tensor(dataset.image_features[:, 0, :], dtype=torch.float32, device=device), dim=1)
        for eeg, _, _, subject_batch, object_idx, _, _ in DataLoader(dataset, batch_size=512):
            eeg = eeg.to(device)
            if eeg.shape[-1] != n_times:
                eeg = torch.nn.functional.interpolate(eeg, size=n_times, mode="linear", align_corners=False)
            unknown = torch.full_like(subject_batch, 10**6).to(device)
            views = []
            if MC:  # MC dropout: view 0 deterministic, the rest with dropout active
                views.append(torch.nn.functional.normalize(eeg_projector(run_eeg_backbone(model, cfg, eeg, unknown)).float(), dim=1))
                for m in model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.train()
                views += [torch.nn.functional.normalize(eeg_projector(run_eeg_backbone(model, cfg, eeg, unknown)).float(), dim=1)
                          for _ in range(MC)]
                model.eval()
                preds.append(torch.stack(views, 1).half().cpu())
                labels.append(object_idx.to(device))
                continue
            for shift in SHIFTS:  # edge-padded time shift of the same trial
                x = eeg if shift == 0 else torch.cat([eeg[..., :1].expand(-1, -1, shift), eeg[..., :-shift]], -1) \
                    if shift > 0 else torch.cat([eeg[..., -shift:], eeg[..., -1:].expand(-1, -1, -shift)], -1)
                views.append(torch.nn.functional.normalize(
                    eeg_projector(run_eeg_backbone(model, cfg, x, unknown)).float(), dim=1))
            preds.append(torch.stack(views, 1).half().cpu())
            labels.append(object_idx.to(device))
    views = torch.cat(preds)  # (N, len(SHIFTS), D)
    y = torch.cat(labels)
    p = views[:, 0].to(device).float()
    train = np.load(os.path.join(cfg.image_feature_dir, "image_train.npy"))
    train = torch.nn.functional.normalize(
        torch.tensor(train.reshape(-1, train.shape[-1]), dtype=torch.float32, device=device), dim=1)

    def top5(q):
        hits = 0
        for chunk, labels_chunk in zip(q.split(16384), y.split(16384)):
            s = torch.nn.functional.normalize(chunk.to(device).float(), dim=1) @ gallery.T
            hits += (s.topk(5, dim=1).indices == labels_chunk[:, None]).any(1).sum().item()
        return hits / len(y)

    if MC:
        print(f"deterministic {top5(p):.4f}; MC mean of {MC}: {top5(views[:, 1:].float().mean(1)):.4f}; "
              f"both: {top5(views.float().mean(1)):.4f}")
        sys.exit()
    print(f"raw {top5(p):.4f}", flush=True)
    for k in sorted({abs(v) for v in SHIFTS}):
        keep = [i for i, v in enumerate(SHIFTS) if abs(v) <= k]
        print(f"TTA shifts |s|<={k}: {top5(views[:, keep].mean(1)):.4f}", flush=True)
    if TTA_ONLY:
        sys.exit()
    mu = train.mean(0)
    for lam in (-0.5, -0.2, 0.2, 0.5):
        print(f"p + {lam} * mean_train  {top5(p + lam * mu):.4f}", flush=True)
    for beta in (10, 30, 100):
        prior = torch.cat([torch.softmax(beta * (chunk @ train.T), dim=1) @ train for chunk in p.split(8192)])
        prior = torch.nn.functional.normalize(prior, dim=1)
        print(f"beta {beta}: " + " ".join(f"a={a}:{top5(a * p + (1 - a) * prior):.4f}"
                                          for a in (0.0, 0.3, 0.5, 0.7, 0.85)), flush=True)
