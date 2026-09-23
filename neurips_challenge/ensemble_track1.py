"""Equal-weight ensembles under the Codabench per-trial protocol.

A submission would output the mean of the members' L2-normalised predictions. For one query
that mean's norm is a positive constant, so its cosine ranking over the gallery equals the
ranking of the SUM of the members' cosine-score matrices. Each model's (160000, 200) score
matrix is therefore computed once (same data, order and unknown subject ids as
score_track1.py) and cached next to its checkpoint; any subset is then scored in seconds.

    ensemble_track1.py --min_top5 0.34 --sizes 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from neurips_challenge.score_track1 import (  # noqa: E402
    OFFICIAL_EEG_DIR, _normalize, build_modules, build_test_dataset,
)

RESULTS = os.path.join(REPO, "results/things_eeg/neurips_track1")
CACHE = "track1_scores.npy"


@torch.no_grad()
def score_matrix(run_dir: str, device) -> tuple[np.ndarray, np.ndarray]:
    """(queries, 200) cosine scores and labels on the official input, as the grader sees it."""
    from train import run_eeg_backbone

    with open(os.path.join(run_dir, "train_config.json")) as handle:
        cfg = SimpleNamespace(**json.load(handle))
    cfg.eeg_data_dir = OFFICIAL_EEG_DIR
    checkpoint = torch.load(os.path.join(run_dir, "checkpoint_test_best.pth"),
                            map_location=device, weights_only=False)
    n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
    scores, labels, gallery = [], [], None
    for subject_id in cfg.test_subject_ids:
        dataset = build_test_dataset(cfg, [subject_id])
        if gallery is None:
            model, eeg_projector, img_projector = build_modules(cfg, checkpoint, dataset, device, n_times)
            targets = torch.tensor(dataset.image_features[:, 0, :], dtype=torch.float32, device=device)
            gallery = torch.tensor(_normalize(img_projector(targets).float().cpu().numpy()), device=device)
        for eeg, _, _, subject_batch, object_idx, _, _ in DataLoader(dataset, batch_size=512):
            eeg = eeg.to(device)
            if eeg.shape[-1] != n_times:
                eeg = torch.nn.functional.interpolate(eeg, size=n_times, mode="linear", align_corners=False)
            unknown = torch.full_like(subject_batch, 10**6).to(device)
            preds = eeg_projector(run_eeg_backbone(model, cfg, eeg, unknown)).float()
            scores.append((torch.nn.functional.normalize(preds, dim=1) @ gallery.T).cpu().numpy())
            labels.append(object_idx.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def top_k(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argpartition(-scores, 5, axis=1)[:, :5]
    top5 = (order == labels[:, None]).any(1).mean()
    top1 = (scores.argmax(1) == labels).mean()
    return float(top1), float(top5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_top5", type=float, default=0.34, help="members need a full grader top-5 above this")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=os.path.join(RESULTS, "ensembles/sweep.csv"))
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    members = []  # (name, run_dir, single top5)
    for path in glob.glob(os.path.join(RESULTS, "**/track1_score.json"), recursive=True):
        with open(path) as handle:
            top5 = json.load(handle)["per_epoch"]["top5"]
        if top5 > args.min_top5:
            run_dir = os.path.dirname(path)
            members.append((os.path.relpath(run_dir, RESULTS).split(os.sep)[1], run_dir, top5))  # the run tag
    members.sort(key=lambda m: -m[2])
    print(f"{len(members)} members above {args.min_top5}:")
    for name, _, top5 in members:
        print(f"  {top5:.4f}  {name}")

    scores, labels = {}, None
    for name, run_dir, top5 in members:
        cache = os.path.join(run_dir, CACHE)
        if os.path.exists(cache):
            scores[name] = np.load(cache)
            labels = np.load(os.path.join(RESULTS, "ensembles/labels.npy"))
        else:
            scores[name], got = score_matrix(run_dir, device)
            np.save(cache, scores[name])
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            np.save(os.path.join(RESULTS, "ensembles/labels.npy"), got)
            assert labels is None or np.array_equal(labels, got), "members disagree on query order"
            labels = got
        # The cached matrix must reproduce the recorded grader score, or the sweep is meaningless.
        _, check = top_k(scores[name], labels)
        assert abs(check - top5) < 2e-4, f"{name}: {check:.4f} != recorded {top5:.4f}"

    rows = []
    for size in args.sizes:
        for combo in itertools.combinations([m[0] for m in members], size):
            top1, top5 = top_k(sum(scores[n] for n in combo), labels)
            rows.append((size, top5, top1, "+".join(combo)))
    rows.sort(key=lambda r: (r[0], -r[1]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("size,top5,top1,members\n")
        handle.writelines(f"{s},{t5:.5f},{t1:.5f},{m}\n" for s, t5, t1, m in rows)
    print(f"\n{len(rows)} ensembles -> {args.out}\nbest per size:")
    for size in args.sizes:
        s, t5, t1, m = next(r for r in rows if r[0] == size)
        print(f"  {s}: top5 {t5:.4f} top1 {t1:.4f}  {m}")


if __name__ == "__main__":
    main()
