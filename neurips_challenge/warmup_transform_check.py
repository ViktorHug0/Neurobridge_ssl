"""Run in the challenge kit venv, on a GPU: the packaged ens5 warm-up submission through the OFFICIAL test loader
(batch 64, as the harness), with and without the batch transform on each predict() output.
Reports the official per-trial top5_acc and top5_acc_subject_agg for both.
    warmup_transform_check.py [submission_dir]"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/Neurips_challenge/src/track1_eeg_image")
sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
from score_submission import _load_solver, _official_loaders  # noqa: E402
from neurips_challenge.official_metrics import group_means, topk_accuracy  # noqa: E402


def batch_transform(P, a=0.5, k=1, b=0.5):  # = neurips_challenge/score_track1.batch_transform
    mu = P.mean(0, keepdim=True); C = P - mu
    if k and len(P) > k:
        V = torch.linalg.svd(C.float(), full_matrices=False)[2][:k].to(C.dtype)
        C = C - b * (C @ V.T) @ V
    return C + (1 - a) * mu


sub = Path(sys.argv[1] if len(sys.argv) > 1 else "/nasbrain/p20fores/Neurobridge_SSL/neurips_challenge/submissions/ens5").resolve()
loader = _official_loaders(64)
channels = loader.dataset.extractors["neuro"]._channels
ch_names = [name for name, _ in sorted(channels.items(), key=lambda x: x[1])]
first = loader.dataset[0].data
model = _load_solver(sub).load_model({"submission_dir": sub, "device": "cuda", "n_chans": int(first["neuro"].shape[-2]),
                                      "n_times": int(first["neuro"].shape[-1]), "n_outputs": int(first["target"].shape[-1]),
                                      "chs_info": [{"ch_name": n} for n in ch_names]})
plain, tf, ys, subj = [], [], [], []
with torch.inference_mode():
    for i, batch in enumerate(loader):
        p = model.predict(batch.data["neuro"].cuda(non_blocking=True)).float()
        plain.append(p.cpu().numpy()); tf.append(batch_transform(p).cpu().numpy())
        y = batch.data["target"]; y = y[:, 0] if y.ndim == 3 else y
        ys.append(y.float().numpy()); subj.append(np.asarray(batch.data["subject_id"]).reshape(-1))
        if i % 500 == 0:
            print(f"batch {i}/{len(loader)}", flush=True)
y_true, subj = np.concatenate(ys), np.concatenate(subj)
cand, tidx = np.unique(y_true, axis=0, return_inverse=True); tidx = tidx.reshape(-1)
n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
for tag, P in (("plain", np.concatenate(plain)), ("transformed", np.concatenate(tf))):
    keys, sa = group_means(P, np.stack([subj, tidx], 1))
    print(f"{tag}: top5_acc {topk_accuracy(n(P) @ n(cand).T, tidx, 5):.4f}  "
          f"top5_acc_subject_agg {topk_accuracy(n(sa) @ n(cand).T, keys[:, 1], 5):.4f}", flush=True)
