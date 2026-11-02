"""Equal-weight ensembles of per-subject intra runs (inductive, plain cosine).

For every subject and member arm, loads that subject's selected checkpoint (lowest test loss) from
results/things_eeg/intra-subjects/insights/<arm>/*sub-XX, embeds the subject's 200 averaged test trials,
and caches the (200, 200) cosine-score matrix against the member's projected test gallery. An ensemble
sums its members' matrices (both sides L2-normalised, so no member dominates by scale).

    intra_ensemble.py <arm> [<arm> ...]      # every subset of size >= 1 is scored

`pooled:<arm>` is ONE pooled model (results/things_eeg/intra-subjects/pooled/<arm>, its single saved
checkpoint) scored on each subject; checked against that epoch's row of per_subject_test.csv.
"""

import csv
import glob
import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from module.dataset import EEGPreImageDataset  # noqa: E402
from neurips_challenge.score_track1 import build_modules  # noqa: E402
from train import run_eeg_backbone  # noqa: E402

ROOT = "results/things_eeg/intra-subjects/insights"
F = torch.nn.functional


@torch.no_grad()
def scores(run_dir, subject=None):
    cache = os.path.join(run_dir, "intra_scores.npy" if subject is None else f"intra_scores_sub-{subject:02d}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    cfg = SimpleNamespace(**json.load(open(os.path.join(run_dir, "train_config.json"))))
    ckpt = torch.load(os.path.join(run_dir, "checkpoint_test_best.pth"), map_location="cpu", weights_only=False)
    ds = EEGPreImageDataset([subject] if subject is not None else cfg.test_subject_ids, cfg.eeg_data_dir, cfg.selected_channels, cfg.time_window,
                            cfg.image_feature_dir, "", False, [], True, False, None, False, False, False, False)
    n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
    model, eeg_proj, img_proj = build_modules(cfg, ckpt, ds, torch.device("cpu"), n_times)
    eeg = torch.stack([ds[i][0] for i in range(len(ds))])
    subj = torch.tensor([ds[i][3] for i in range(len(ds))])
    order = [int(ds[i][4]) for i in range(len(ds))]
    assert order == list(range(len(ds))), "test rows must be in image order"
    q = F.normalize(eeg_proj(run_eeg_backbone(model, cfg, eeg, subj)).float(), dim=1)
    g = F.normalize(img_proj(torch.tensor(ds.image_features[:, 0, :], dtype=torch.float32)).float(), dim=1)
    s = (q @ g.T).numpy()
    np.save(cache, s)
    return s


def top1(s):
    return float((s.argmax(1) == np.arange(len(s))).mean() * 100)


arms = sys.argv[1:]
subjects = range(1, 11)
mats = {a: {} for a in arms}
for a in arms:
    for sub in subjects:
        if a.startswith("pooled:"):
            run = sorted(glob.glob(f"{ROOT.replace('insights', 'pooled')}/{a[7:]}/*-pooled"))[-1]
            mats[a][sub] = scores(run, sub)
            epoch = int(float(list(csv.DictReader(open(os.path.join(run, "result.csv"))))[-1]["best epoch"]))
            ref = next(float(r["top1"]) for r in csv.DictReader(open(os.path.join(run, "per_subject_test.csv")))
                       if int(r["epoch"]) == epoch and int(r["subject"]) == sub)
        else:
            run = sorted(glob.glob(f"{ROOT}/{a}/*sub-{sub:02d}"))[-1]
            mats[a][sub] = scores(run)
            ref = float(list(csv.DictReader(open(os.path.join(run, "result.csv"))))[-1]["best top1 acc"])
        got = top1(mats[a][sub])
        # bf16-trained members (LaBraM) were evaluated under autocast; fp32 recompute may flip <= 3 of 200
        amp = json.load(open(os.path.join(run, "train_config.json"))).get("amp_dtype", "none") != "none"
        assert abs(got - ref) < (1.6 if amp else 0.6), f"{a} sub-{sub:02d}: recomputed {got} vs logged {ref}"
for k in range(1, len(arms) + 1):
    for combo in itertools.combinations(arms, k):
        per = [top1(sum(mats[a][s] for a in combo)) for s in subjects]
        print(f"{np.mean(per):6.2f}  {' + '.join(combo)}   " + " ".join(f"{v:.1f}" for v in per), flush=True)
