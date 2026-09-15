#!/usr/bin/env python3
"""Open-gallery and streaming stress tests for SAGE-TTA (reviewer concern B).

Two regimes that deliberately break the closed-set N=K bijection the reviewers
suspect the transductive stage exploits:

  A. open_gallery  -- 200 real EEG queries retrieved against a gallery of
     200 + D images, where the D distractors (drawn from the held-out subject's
     *training* concepts) match no query. Sweeps D. Methods: plain cosine,
     SAW+CSLS, and fresh full TTA (all batch/transductive).

  B. streaming     -- fit the label-free SAW + orthogonal calibration map on a
     warmup buffer of W queries, freeze it, then decode every remaining query
     *independently* with plain cosine against the fixed 200-way gallery (no
     CSLS/Sinkhorn coupling across queries -> genuine per-trial decoding).
     Sweeps W. Baselines: plain cosine (no adaptation) and oracle fresh batch
     TTA that sees all 200 queries at once (the closed-set number).

Everything is built on the shared rebuttal primitives; no new TTA math.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from shared import (
    TTAParams,
    add_common_args,
    aggregate_results,
    apply_tta_calibration,
    cosine_scores,
    encode_average_features,
    ensure_output_dir,
    evaluate_full_tta,
    evaluate_plain,
    evaluate_saw_csls,
    evaluate_scores,
    fit_tta_calibration,
    load_subject_train_test_context,
    params_from_args,
    score_features,
    write_config,
    _test_image_matrix,
)


def project_raw_images(eval_args, modules, raw_images, batch_size=1024):
    """Push raw image features through the frozen image projector."""
    raw_images = np.asarray(raw_images, dtype=np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, raw_images.shape[0], batch_size):
            batch = torch.tensor(raw_images[start : start + batch_size], device=modules["device"])
            out.append(modules["img_projector"](batch).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def build_distractor_pool(eval_args, modules, train_dataset, seed):
    """Unique training-concept image embeddings (disjoint from the 200 test concepts)."""
    raw = _test_image_matrix(train_dataset)
    raw = np.unique(raw, axis=0)  # dedup repeated per-trial copies
    rng = np.random.default_rng(seed)
    rng.shuffle(raw)
    return project_raw_images(eval_args, modules, raw)


# --- Experiment A: open / non-bijective gallery -----------------------------

def run_open_gallery(subject_id, query, gallery200, distractors, params, args):
    n = gallery200.shape[0]
    targets = np.arange(n, dtype=np.int64)
    rows = []
    for d in args.distractor_counts:
        d = int(min(d, distractors.shape[0]))
        gallery = gallery200 if d == 0 else np.concatenate([gallery200, distractors[:d]], axis=0)
        for name, metrics in (
            ("plain_cosine", evaluate_plain(query, gallery, target_indices=targets)),
            ("saw_csls", evaluate_saw_csls(query, gallery, params, target_indices=targets)),
            ("full_tta", evaluate_full_tta(query, gallery, params, target_indices=targets)),
        ):
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "distractors": d,
                    "gallery_size": int(gallery.shape[0]),
                    "method": name,
                    "top1_acc": round(metrics["top1_acc"], 4),
                    "top5_acc": round(metrics["top5_acc"], 4),
                }
            )
    return rows


# --- Experiment B: streaming / per-trial decode -----------------------------

def evaluate_frozen_per_trial(eval_query, calibration, gallery, targets):
    """Frozen map applied per row, then plain cosine -> no cross-query coupling."""
    transformed = apply_tta_calibration(eval_query, calibration, alpha=1.0)
    return evaluate_scores(cosine_scores(transformed, gallery), target_indices=targets)


def evaluate_frozen_batch_csls(eval_query, calibration, gallery, targets, params):
    """Frozen map + CSLS over the streamed batch -> isolates the CSLS coupling effect."""
    transformed = apply_tta_calibration(eval_query, calibration, alpha=1.0)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores, target_indices=targets)


def run_streaming(subject_id, query, gallery200, params, args):
    total = gallery200.shape[0]
    rows = []
    for warmup in args.warmup_sizes:
        warmup = int(min(warmup, total - 1))
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100003 + int(subject_id) * 101 + warmup)
            buf = np.sort(rng.choice(total, size=warmup, replace=False))
            rest = np.setdiff1d(np.arange(total, dtype=np.int64), buf)
            calibration = fit_tta_calibration(query[buf], gallery200[buf], params)

            for name, metrics in (
                # per-trial deployable decode of streamed queries vs full gallery
                (
                    "stream_frozen_percall",
                    evaluate_frozen_per_trial(query[rest], calibration, gallery200, rest),
                ),
                # frozen map but CSLS re-coupled over the streamed batch (less deployable)
                (
                    "stream_frozen_csls",
                    evaluate_frozen_batch_csls(query[rest], calibration, gallery200, rest, params),
                ),
                # no-adaptation lower bound on the same streamed queries
                ("stream_cosine", evaluate_plain(query[rest], gallery200, target_indices=rest)),
            ):
                rows.append(_stream_row(subject_id, warmup, seed, name, metrics))
    # closed-set oracle (all 200 queries at once) -- warmup-independent reference
    oracle = evaluate_full_tta(query, gallery200, params, target_indices=np.arange(total, dtype=np.int64))
    for seed in args.seeds:
        rows.append(_stream_row(subject_id, total, seed, "oracle_batch_tta", oracle))
    return rows


def _stream_row(subject_id, warmup, seed, method, metrics):
    return {
        "subject_id": int(subject_id),
        "warmup": int(warmup),
        "seed": int(seed),
        "method": method,
        "top1_acc": round(metrics["top1_acc"], 4),
        "top5_acc": round(metrics["top5_acc"], 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--experiment", choices=["open_gallery", "streaming", "both"], default="both")
    parser.add_argument("--distractor_counts", nargs="+", type=int, default=[0, 100, 200, 400, 800, 1454])
    parser.add_argument("--warmup_sizes", nargs="+", type=int, default=[10, 25, 50, 100, 150])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "openset_streaming")
    write_config(output_dir, args, params)

    open_rows, stream_rows = [], []
    for subject_id in args.subjects:
        print(f"[openset_streaming] subject={int(subject_id):02d}")
        _, eval_args, train_dataset, test_dataset, modules = load_subject_train_test_context(
            source_run_dir, args, subject_id, average=True
        )
        query, gallery200 = encode_average_features(eval_args, modules, test_dataset)

        if args.experiment in ("open_gallery", "both"):
            distractors = build_distractor_pool(eval_args, modules, train_dataset, seed=args.seed)
            open_rows.extend(run_open_gallery(subject_id, query, gallery200, distractors, params, args))
            pd.DataFrame(open_rows).to_csv(os.path.join(output_dir, "open_gallery_subject.csv"), index=False)

        if args.experiment in ("streaming", "both"):
            stream_rows.extend(run_streaming(subject_id, query, gallery200, params, args))
            pd.DataFrame(stream_rows).to_csv(os.path.join(output_dir, "streaming_subject.csv"), index=False)

    if open_rows:
        agg = aggregate_results(pd.DataFrame(open_rows), ["method", "distractors", "gallery_size"])
        agg.to_csv(os.path.join(output_dir, "open_gallery_average.csv"), index=False)
        print("\n=== Open-gallery (200 queries vs 200+D) ===")
        print(agg.to_string(index=False))

    if stream_rows:
        agg = aggregate_results(pd.DataFrame(stream_rows), ["method", "warmup"])
        agg.to_csv(os.path.join(output_dir, "streaming_average.csv"), index=False)
        print("\n=== Streaming per-trial decode ===")
        print(agg.to_string(index=False))

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
