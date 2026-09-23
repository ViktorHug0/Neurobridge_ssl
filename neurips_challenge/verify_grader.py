"""Check score_track1.py's metrics against NeuralBench's own metric stack.

Run with the CHALLENGE venv, which has neuraltrain/neuralbench installed:

    /nasbrain/p20fores/Neurips_challenge/.venv/bin/python neurips_challenge/verify_grader.py

Our scorer and theirs get identical synthetic predictions, and must agree at all three
aggregation levels. This is the check that does not need a trained model: if it passes, any
disagreement between this repo and the start-kit is a modelling difference, not a metric bug.

What is being compared:
  per_epoch    <- neuraltrain TopkAcc with subjects = arange(N)   (callbacks.py:313, no averaging)
  subject_agg  <- agg_retrieval_preds(..., subjects_pred)         (callbacks.py:309)
  instance_agg <- agg_retrieval_preds(..., subjects=None)         (callbacks.py:311)
and the gallery in both cases is agg_per_group(y_true, groups, "mean") (callbacks.py:295).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_our_metrics():
    """Import score_track1 without executing its Neurobridge-only imports."""
    spec = importlib.util.spec_from_file_location(
        "score_track1", os.path.join(HERE, "score_track1.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def neuralbench_metrics(y_pred, y_true, groups_pred, subjects_pred, level: str):
    """Reproduce neuralbench/callbacks.py:_get_test_full_metrics for one aggregation level."""
    from neuraltrain.metrics.metrics import TopkAcc
    from neuraltrain.metrics.utils import agg_per_group, agg_retrieval_preds

    agg_y_true, agg_groups_true = agg_per_group(
        y_true, groups=groups_pred, agg_func="mean"
    )
    if level == "subject_agg":
        subjects = subjects_pred
    elif level == "instance_agg":
        subjects = None
    else:
        subjects = torch.arange(y_pred.shape[0]).tolist()

    agg_y_pred, agg_groups_pred = agg_retrieval_preds(
        y_pred, groups_pred=groups_pred, subjects_pred=subjects
    )
    out = {}
    for name, k in (("top1", 1), ("top5", 5)):
        metric = TopkAcc(topk=k)
        metric.reset()
        metric.update(agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true)
        out[name] = float(metric.compute())
    out["n_queries"] = int(agg_y_pred.shape[0])
    return out


def main() -> None:
    ours = load_our_metrics()
    rng = np.random.default_rng(33)

    n_objects, n_subjects, n_reps, dim = 200, 10, 8, 64
    targets = rng.normal(size=(n_objects, dim)).astype(np.float32)

    # Predictions: the right target plus noise, so accuracy lands strictly between chance and 1
    # at every level and a mis-wired aggregation cannot pass by accident.
    labels = np.tile(np.repeat(np.arange(n_objects), n_reps), n_subjects)
    subjects = np.repeat(np.arange(n_subjects), n_objects * n_reps)
    # scale tuned so no level saturates: a level pinned at 0 or 1 would agree even if mis-wired.
    preds = targets[labels] + rng.normal(scale=45.0, size=(len(labels), dim)).astype(np.float32)

    gallery_normed = ours._normalize(targets)
    per_epoch = ours.topk_acc(preds, labels, gallery_normed)

    sums = np.zeros((n_subjects, n_objects, dim), dtype=np.float64)
    counts = np.zeros((n_subjects, n_objects), dtype=np.int64)
    np.add.at(sums, (subjects, labels), preds)
    np.add.at(counts, (subjects, labels), 1)
    seen = counts > 0
    subject_agg = ours.topk_acc(
        (sums[seen] / counts[seen][:, None]).astype(np.float32),
        np.broadcast_to(np.arange(n_objects), counts.shape)[seen],
        gallery_normed,
    )
    instance_agg = ours.topk_acc(
        (sums.sum(0) / counts.sum(0)[:, None]).astype(np.float32),
        np.arange(n_objects),
        gallery_normed,
    )

    # NeuralBench side: groups are the stimulus identity, exactly as `filepath` is used there.
    y_pred = torch.tensor(preds)
    y_true = torch.tensor(targets[labels])
    groups_pred = [f"image_{i:04d}" for i in labels]
    subjects_pred = [f"sub-{s:02d}" for s in subjects]

    failures = []
    for level, mine in (
        ("per_epoch", per_epoch),
        ("subject_agg", subject_agg),
        ("instance_agg", instance_agg),
    ):
        theirs = neuralbench_metrics(y_pred, y_true, groups_pred, subjects_pred, level)
        ok = (
            mine["n_queries"] == theirs["n_queries"]
            and abs(mine["top1"] - theirs["top1"]) < 1e-6
            and abs(mine["top5"] - theirs["top5"]) < 1e-6
        )
        status = "ok " if ok else "MISMATCH"
        print(
            f"{status} {level:13s} ours top1={mine['top1']:.6f} top5={mine['top5']:.6f} "
            f"n={mine['n_queries']:6d} | neuralbench top1={theirs['top1']:.6f} "
            f"top5={theirs['top5']:.6f} n={theirs['n_queries']:6d}"
        )
        if not ok:
            failures.append(level)

    if failures:
        sys.exit(f"grader disagrees with NeuralBench at: {', '.join(failures)}")
    print("\nAll three aggregation levels agree with NeuralBench's own metric code.")


if __name__ == "__main__":
    main()
