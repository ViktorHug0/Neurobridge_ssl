"""Per-member official-loader scores for ens5, plus the loader's channel order and input scale."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurips_challenge/src/track1_eeg_image")
from score_submission import _load_solver, _normalize, _official_loaders  # noqa: E402

SUB = Path(__file__).resolve().parent / "ens5"
OURS = json.load(open("/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench/info.json"))["ch_names"]

loader = _official_loaders(128)
channels = loader.dataset.extractors["neuro"]._channels
ch_names = [name for name, _ in sorted(channels.items(), key=lambda x: x[1])]
print("official order == our cache order:", ch_names == OURS)
print("official ch_names:", ch_names)
model = _load_solver(SUB).load_model({
    "submission_dir": SUB, "device": "cuda", "n_chans": 63, "n_times": 120, "n_outputs": 1536,
    "chs_info": [{"ch_name": n} for n in ch_names]})
preds = {name: [] for name in model.members}
targets = []
with torch.inference_mode():
    for i, batch in enumerate(loader):
        x = batch.data["neuro"].cuda().float()
        if i == 0:
            print("input: shape", tuple(x.shape), "mean", x.mean().item(), "std", x.std().item(), flush=True)
        for name, member in model.members.items():
            preds[name].append(member(x).float().cpu().numpy())
        y = batch.data["target"]
        targets.append((y[:, 0] if y.ndim == 3 else y).float().numpy())
y_true = np.concatenate(targets)
candidates, idx = np.unique(y_true, axis=0, return_inverse=True)
g = _normalize(candidates)
total = 0
for name, chunks in preds.items():
    s = _normalize(np.concatenate(chunks)) @ g.T
    total = total + s
    top5 = np.mean(np.any(np.argsort(s, axis=1)[:, -5:] == idx[:, None], axis=1))
    print(f"{name}: official top5 {top5:.4f}", flush=True)
top5 = np.mean(np.any(np.argsort(total, axis=1)[:, -5:] == idx[:, None], axis=1))
print(f"ensemble: official top5 {top5:.4f}")
