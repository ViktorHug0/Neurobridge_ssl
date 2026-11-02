"""Run in the challenge kit venv: what order does the OFFICIAL warm-up test loader serve trials in?
Reports, per batch of 64, the number of distinct images and subjects, and the longest same-image run."""
import sys
import numpy as np
sys.path.insert(0, "/nasbrain/p20fores/Neurips_challenge/src/track1_eeg_image")
from score_submission import _official_loaders  # noqa: E402

loader = _official_loaders(64)
ys, subs = [], []
for b, batch in enumerate(loader):
    data = batch.data  # neuralset Batch, as score_submission.py reads it
    y = data["target"]
    ys.append(np.asarray(y.reshape(len(y), -1)[:, :8].float().cpu().numpy()))
    sid = data.get("subject_id")
    subs.append(np.asarray(sid.cpu() if hasattr(sid, "cpu") else sid).reshape(-1) if sid is not None else np.full(len(y), -1))
    if b == 0:
        print("batch keys", list(data.keys()), "neuro", tuple(data["neuro"].shape))
y = np.concatenate(ys); s = np.concatenate(subs)
_, img = np.unique(y.round(5), axis=0, return_inverse=True); img = img.reshape(-1)
print("n trials", len(img), "n images", img.max() + 1, "subjects in order of appearance", list(dict.fromkeys(s.tolist())))
d = [len(np.unique(img[i:i + 64])) for i in range(0, len(img), 64)]
print("distinct images per 64-batch: median", int(np.median(d)), "min", min(d), "max", max(d))
runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, img, -1]) != 0)); print("longest same-image run", runs.max(), "mean run", runs.mean().round(2))
print("first 40 image ids:", img[:40].tolist())
