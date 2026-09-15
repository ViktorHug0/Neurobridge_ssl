#!/usr/bin/env python3
"""Benchmark transductive inference heads on the 200-way LOSO test set.

Encodes each held-out subject's EEG (TSConv) and InternViT image prototypes
*once* (reusing the progressive-SATTC encoding pipeline), then evaluates every
selected inference head on the same cached ``(query, image)`` pair. This keeps the
comparison fair and guarantees the incumbent ``full_sattc`` path is identical to
the existing evaluation code.

Example
-------
    .venv/bin/python scripts/things_eeg/transductive_benchmark.py \
        --source_run_dir results/things_eeg/intra-subjects \
        --held_out_subjects 1 2 3 \
        --methods full_sattc plain_cosine em_dirichlet hard_em_dirichlet \
        --output_dir results/transductive_benchmark
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.transductive import infer, list_methods  # noqa: E402
from module.util import topk  # noqa: E402
from train import seed_everything  # noqa: E402

# Methods whose class-balance hyper-parameter (lambd) is swept.
EM_LAMBDA_METHODS = {"em_dirichlet", "hard_em_dirichlet", "em_gaussian", "em_gaussian_cov"}
# Methods built on softmax(T*cosine) simplex features, so the temperature T matters.
SIMPLEX_METHODS = EM_LAMBDA_METHODS | {"kl_kmeans", "soft_kmeans", "hard_kmeans"}
# TransCLIP (group B): embedding-regime GMM + Laplacian + text-KL, with ablations.
TRANSCLIP_METHODS = {"transclip", "transclip_no_kl", "transclip_no_lap",
                     "transclip_mu_only", "transclip_sigma_only", "transclip_anchor"}

# Best per-method config from the sweeps (used with --use_best_configs). Each entry is
# (use_saw_front_end, param_overrides). Empirically, SAW whitening helps ONLY the SATTC
# refinement; for every other head a raw (no-SAW) CSLS front-end is better — including all
# EM/clustering/simplex heads and sinkhorn (verified on 10 LOSO subjects, 2026-06-23).
# Some simplex T-optima sit near the low edge of the swept grid (~70), so those are mild
# lower bounds. See papers/transductive_benchmark_plan.md.
BEST_CONFIGS = {
    "full_sattc":        (True,  {}),                                              # 68.45 (SAW 0.94 + tuned refine)
    "sattc_hungarian":   (True,  {}),                                              # 66.95
    "hungarian_sattc":   (True,  {}),                                              # Hungarian assign first, then SATTC
    "hungarian":         (False, {"csls_k": 0}),                                   # 51.40 (raw cosine; CSLS no-op)
    "sinkhorn":          (False, {"csls_k": 1, "sinkhorn_tau": 0.08, "sinkhorn_iters": 10}),  # 49.00
    "hard_em_dirichlet": (False, {"T": 70.0, "lambd": 0.5, "csls_k": 1}),          # 47.15
    "em_dirichlet":      (False, {"T": 90.0, "lambd": 10.0, "csls_k": 1}),         # 46.90
    "hard_kmeans":       (False, {"T": 70.0, "csls_k": 1}),                        # 46.55
    "em_gaussian":       (False, {"T": 70.0, "lambd": 10.0, "csls_k": 1}),         # 46.50
    "soft_kmeans":       (False, {"T": 70.0, "csls_k": 1}),                        # 46.50
    "transclip":         (False, {"lambda": 0.5, "n_neighbors": 3}),              # 46.35
    "csls":              (False, {"csls_k": 1}),                                   # 42.00
    "kl_kmeans":         (False, {"T": 90.0, "csls_k": 1}),                        # 41.60
    "plain_cosine":      (False, {}),                                              # 35.90 (raw control)
    "em_gaussian_cov":   (False, {"T": 110.0, "lambd": 0.5, "csls_k": 1}),         # 36.00
}


def _load_sweep_module():
    """Import the progressive-SATTC sweep by file path to reuse its encoders."""
    path = os.path.join(os.path.dirname(__file__), "progressive_sattc_candidate_sweep.py")
    spec = importlib.util.spec_from_file_location("progressive_sattc_candidate_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SWEEP = _load_sweep_module()


def _saw_params(args):
    return {
        "saw_shrink": args.sattc_saw_shrink,
        "saw_diag": args.sattc_saw_diag,
        "saw_renorm": not args.sattc_saw_no_renorm,
        "csls_k": args.sattc_csls_k,
    }


def _method_params(method, args, lam=None, temp=None, tc_lambda=None, tc_nn=None):
    """Per-method hyper-parameters. EM/Dirichlet share T/iter/lambd; full_sattc
    carries the incumbent CSLS+Sinkhorn+Procrustes settings.

    ``lam``/``temp`` override the EM class-balance and simplex temperature; ``tc_lambda``/
    ``tc_nn`` override the TransCLIP text-KL weight and k-NN count (all used by the sweeps).
    """
    if method in {"full_sattc", "sattc_hungarian", "hungarian_sattc"}:
        return {
            "use_csls": True,
            "csls_k": args.sattc_csls_k,
            "sinkhorn_enabled": True,
            "sinkhorn_tau": args.sattc_sinkhorn_tau,
            "sinkhorn_iters": args.sattc_sinkhorn_iters,
            "soft_procrustes_enabled": True,
            "soft_procrustes_steps": args.sattc_soft_procrustes_steps,
            "soft_procrustes_power": args.sattc_soft_procrustes_power,
            "soft_procrustes_normalize_inputs": args.sattc_soft_procrustes_normalize_inputs,
        }
    if method in {"csls"}:
        return {"csls_k": args.sattc_csls_k}
    if method in {"sinkhorn"}:
        # Front-end CSLS via sattc_csls_k; its own tau/iters stay fixed (fix 3 not applied).
        return {"csls_k": args.sattc_csls_k, "sinkhorn_tau": 0.08, "sinkhorn_iters": 10}
    if method in TRANSCLIP_METHODS:
        return {
            "lambda": tc_lambda if tc_lambda is not None else args.transclip_lambda,
            "n_neighbors": int(tc_nn) if tc_nn is not None else args.transclip_neighbors,
            "max_iter": args.transclip_max_iter,
            "clip_scale": args.transclip_clip_scale,
            "device": args.device,
        }
    # EM / clustering / Dirichlet solvers (also plain_cosine / hungarian, which only read csls_k)
    effective_T = temp if temp is not None else args.feature_T
    params = {"T": float(effective_T), "iter": args.em_iter, "iter_mm": args.em_iter_mm,
              "csls_k": args.sattc_csls_k, "device": "cpu"}
    effective_lambda = lam if lam is not None else args.em_lambda
    if effective_lambda is not None:
        params["lambd"] = float(effective_lambda)
    return params


def _needs_saw(method, args):
    # plain_cosine is the raw control: never SAW/CSLS.
    if method == "plain_cosine":
        return False
    # With --use_best_configs, SAW is decided per method by the BEST_CONFIGS table.
    if args.use_best_configs and method in BEST_CONFIGS:
        return BEST_CONFIGS[method][0]
    # SATTC heads always use SAW (matches incumbent); everything else only with --saw_all.
    return method in {"full_sattc", "sattc_hungarian", "hungarian_sattc"} or args.saw_all


def _build_train_dataset(eval_args, subject_id):
    """Left-out subject's TRAIN recordings, EEG averaged over the 4 repetitions/image."""
    return EEGPreImageDataset(
        subject_ids=[int(subject_id)],
        eeg_data_dir=eval_args.eeg_data_dir,
        selected_channels=eval_args.selected_channels,
        time_window=eval_args.time_window,
        image_feature_dir=eval_args.image_feature_dir,
        text_feature_dir=getattr(eval_args, "text_feature_dir", ""),
        image_aug=False,
        aug_image_feature_dirs=[],
        average=True,            # average across the 4 train repetitions per image
        _random=False,
        eeg_transform=None,
        train=True,
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=_to_bool_safe(getattr(eval_args, "frozen_eeg_prior", False)),
    )


def _to_bool_safe(v):
    return SWEEP._to_bool(v, False)


def _encode_subject(subject_id, args):
    checkpoint_dir = SWEEP._find_checkpoint_dir(args.source_run_dir, subject_id)
    train_cfg = SWEEP._load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = SWEEP._load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = SWEEP._build_eval_args(train_cfg, eval_cfg, args, subject_id)
    if args.eval_split == "train":
        dataset = _build_train_dataset(eval_args, subject_id)
    else:
        dataset = SWEEP._build_test_dataset(eval_args, subject_id)
    modules = SWEEP._load_modules(eval_args, checkpoint_dir, dataset)

    # Subsample the candidate pool to n_images (encode only the selected items).
    total = len(dataset)
    k = total if args.n_images is None else min(int(args.n_images), total)
    if k < total:
        from torch.utils.data import Subset
        rng = np.random.default_rng(args.eval_seed + int(subject_id))
        sel = np.sort(rng.choice(total, size=k, replace=False))
        dataset = Subset(dataset, sel.tolist())
        print(f"[transductive_benchmark]   subsampled {k}/{total} {args.eval_split} items")

    encoded = SWEEP._encode_subject_features(eval_args, modules, dataset)
    return eval_args, encoded


def _iter_method_configs(method, args):
    """Yield ``_method_params`` kwargs for each swept config of ``method``.

    EM heads sweep lambda, simplex heads sweep T, TransCLIP heads sweep its own
    (lambda, n_neighbors); everything else runs a single config.
    """
    if args.use_best_configs:
        yield {}  # single config; params come from BEST_CONFIGS overrides
        return
    if method in TRANSCLIP_METHODS:
        tc_lams = args.transclip_lambda_grid or [None]
        tc_nns = args.transclip_neighbors_grid or [None]
        for nn in tc_nns:
            for lam in tc_lams:
                yield {"tc_lambda": lam, "tc_nn": nn}
        return
    lam_list = args.em_lambda_grid if (args.em_lambda_grid and method in EM_LAMBDA_METHODS) else [None]
    temp_list = args.feature_T_grid if (args.feature_T_grid and method in SIMPLEX_METHODS) else [None]
    for temp in temp_list:
        for lam in lam_list:
            yield {"lam": lam, "temp": temp}


def _evaluate_subject(subject_id, args):
    eval_args, encoded = _encode_subject(subject_id, args)
    query_all = np.asarray(encoded["query_features"], dtype=np.float32)
    image_all = np.asarray(encoded["image_features"], dtype=np.float32)
    k_images = image_all.shape[0]                       # number of class prototypes
    # n_queries queries drawn from the first N (bijective) classes; rest are distractors.
    n = k_images if args.n_queries is None else min(int(args.n_queries), k_images)
    targets = np.arange(n, dtype=np.int64)
    saw_params = _saw_params(args)

    rows = []
    for method in args.methods:
        query_src = query_all
        if _needs_saw(method, args):
            query_src = SWEEP._process_query_features_safe(query_all, "saw_csls", saw_params)
        query = query_src[:n]                            # N queries; image_all keeps all K prototypes
        for cfg in _iter_method_configs(method, args):
            params = _method_params(method, args, **cfg)
            if args.use_best_configs and method in BEST_CONFIGS:
                params.update(BEST_CONFIGS[method][1])   # apply per-method best overrides
            try:
                t0 = time.perf_counter()
                scores = infer(query, image_all, method, params)
                duration_s = time.perf_counter() - t0
                top5, top1 = topk(scores, 5, target_indices=targets)
                top1_acc = round(100.0 * top1 / n, 4)
                top5_acc = round(100.0 * top5 / n, 4)
            except Exception as exc:  # one bad method must not kill the run
                print(f"[transductive_benchmark] method={method} subject={subject_id} FAILED: {exc}")
                top1_acc, top5_acc, duration_s = float("nan"), float("nan"), float("nan")
            if method in TRANSCLIP_METHODS:
                lam_value, temp_value, nn_value = params.get("lambda"), None, params.get("n_neighbors")
            else:
                lam_value = params.get("lambd") if method in EM_LAMBDA_METHODS else None
                temp_value = params.get("T") if method in SIMPLEX_METHODS else None
                nn_value = None
            rows.append({
                "subject_id": int(subject_id),
                "method": method,
                "lambda": lam_value,
                "feature_T": temp_value,
                "n_neighbors": nn_value,
                "n_queries": int(n),
                "n_images": int(k_images),
                "sample_count": int(n),
                "top1_acc": top1_acc,
                "top5_acc": top5_acc,
                "duration_s": round(float(duration_s), 5),
                "params": json.dumps(params, sort_keys=True),
            })
            tags = "".join([
                "" if temp_value is None else f" T={temp_value:g}",
                "" if lam_value is None else f" lambda={lam_value:g}",
                "" if nn_value is None else f" nn={nn_value:g}",
            ])
            print(f"[transductive_benchmark] subject={subject_id:02d} method={method:<20}"
                  f"{tags} top1={top1_acc} top5={top5_acc} ({duration_s:.3f}s)")
    return rows


def _summarize(subject_df):
    summary = (
        subject_df.groupby(["method", "lambda", "feature_T", "n_neighbors"], as_index=False, dropna=False).agg(
            top1_mean=("top1_acc", "mean"),
            top1_std=("top1_acc", "std"),
            top5_mean=("top5_acc", "mean"),
            top5_std=("top5_acc", "std"),
            duration_mean=("duration_s", "mean"),
            n_subjects=("subject_id", "nunique"),
        )
        .sort_values("top1_mean", ascending=False)
    )
    summary["top1_std"] = summary["top1_std"].fillna(0.0)
    summary["top5_std"] = summary["top5_std"].fillna(0.0)
    summary["duration_mean"] = summary["duration_mean"].round(4)
    return summary


def _make_barplot(summary_df, out_path, title):
    """Top-1 barplot per method (best config per method), sorted by top-1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # one (best top-1) row per method; drop methods whose every config is NaN
    # (e.g. transclip can degenerate at very small batch sizes)
    valid = summary_df.dropna(subset=["top1_mean"])
    best = valid.loc[valid.groupby("method")["top1_mean"].idxmax()]
    best = best.sort_values("top1_mean", ascending=False)
    methods = best["method"].tolist()
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(methods)), 6))
    bars = ax.bar(x, best["top1_mean"], 0.65, color="tab:blue")
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=40, ha="right")
    ax.set_ylabel("top-1 accuracy (%)"); ax.set_ylim(0, 100); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved barplot:          {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_run_dir", type=str, required=True)
    parser.add_argument("--held_out_subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=["full_sattc", "plain_cosine", "em_dirichlet"],
                        help=f"Any of: {list_methods()}")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for the EEG encoder")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2099)
    # Suite mode: per-method best configs, auto barplot
    parser.add_argument("--use_best_configs", action="store_true",
                        help="run each method with its tuned best config (BEST_CONFIGS) + per-method SAW")
    parser.add_argument("--plot", action="store_true", help="write a comparison barplot when finished")
    # Evaluation set selection / subsampling
    parser.add_argument("--eval_split", choices=["test", "train"], default="test",
                        help="'train' uses the left-out subject's train recordings (EEG averaged over 4 reps)")
    parser.add_argument("--n_images", type=int, default=None,
                        help="number of image-class prototypes K in the eval batch (subsample; default = all)")
    parser.add_argument("--n_queries", type=int, default=None,
                        help="number of EEG queries N (<= K; default = K, i.e. bijective)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="shortcut: set both n_images and n_queries to this (bijective N-way task)")
    parser.add_argument("--eval_seed", type=int, default=12345, help="seed for subsampling the eval pool")
    # SAW preprocessing
    parser.add_argument("--saw_all", action="store_true", help="apply SAW whitening to every method, not just full_sattc")
    parser.add_argument("--sattc_saw_shrink", type=float, default=0.94)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_saw_no_renorm", action="store_true")
    parser.add_argument("--sattc_csls_k", type=int, default=1)
    parser.add_argument("--sattc_sinkhorn_tau", type=float, default=0.08)
    parser.add_argument("--sattc_sinkhorn_iters", type=int, default=10)
    parser.add_argument("--sattc_soft_procrustes_steps", type=int, default=6)
    parser.add_argument("--sattc_soft_procrustes_power", type=float, default=1.0)
    parser.add_argument("--sattc_soft_procrustes_normalize_inputs", action="store_true")
    # EM / Dirichlet hyper-parameters
    parser.add_argument("--feature_T", type=float, default=30.0, help="temperature for simplex features when no grid is given")
    parser.add_argument("--feature_Ts", nargs="+", type=float, default=None,
                        help="explicit list of temperatures to sweep for simplex heads")
    parser.add_argument("--feature_T_start", type=float, default=None, help="grid sweep: first temperature")
    parser.add_argument("--feature_T_stop", type=float, default=None, help="grid sweep: last temperature (inclusive)")
    parser.add_argument("--feature_T_step", type=float, default=None, help="grid sweep: temperature step")
    parser.add_argument("--em_iter", type=int, default=20)
    parser.add_argument("--em_iter_mm", type=int, default=50)
    # TransCLIP (group B) hyper-parameters
    parser.add_argument("--transclip_lambda", type=float, default=1.0, help="text-KL anchor weight")
    parser.add_argument("--transclip_neighbors", type=int, default=3, help="k for the k-NN Laplacian")
    parser.add_argument("--transclip_max_iter", type=int, default=10, help="outer block-MM iterations")
    parser.add_argument("--transclip_clip_scale", type=float, default=100.0, help="logit scale for y_hat")
    parser.add_argument("--transclip_lambdas", nargs="+", type=float, default=None,
                        help="grid sweep: text-KL weights for TransCLIP heads")
    parser.add_argument("--transclip_neighbors_list", nargs="+", type=int, default=None,
                        help="grid sweep: k-NN counts for TransCLIP heads")
    # NB: the solver's built-in default (int(K/5)*N = 8000 at N=K=200) collapses the
    # clustering in this one-sample-per-class regime; ~[0, 200] works far better. Sweep this.
    parser.add_argument("--em_lambda", type=float, default=1.0,
                        help="class-balance value for EM heads when no grid is given. Sweep via the grid args below.")
    parser.add_argument("--em_lambdas", nargs="+", type=float, default=None,
                        help="explicit list of lambda values to sweep for EM heads")
    parser.add_argument("--em_lambda_start", type=float, default=None, help="grid sweep: first lambda")
    parser.add_argument("--em_lambda_stop", type=float, default=None, help="grid sweep: last lambda (inclusive)")
    parser.add_argument("--em_lambda_step", type=float, default=None, help="grid sweep: lambda step")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # --subsample is a shortcut for a bijective N-way task.
    if args.subsample is not None:
        args.n_images = args.subsample
        args.n_queries = args.subsample
    if args.n_queries is not None and args.n_images is not None and args.n_queries > args.n_images:
        parser.error("--n_queries must be <= --n_images")
    print(f"[transductive_benchmark] eval_split={args.eval_split} "
          f"n_images={args.n_images or 'all'} n_queries={args.n_queries or 'all'}"
          f"{' (use_best_configs)' if args.use_best_configs else ''}")

    # Build the lambda grid (explicit list takes precedence over start/stop/step).
    if args.em_lambdas is not None:
        args.em_lambda_grid = [float(x) for x in args.em_lambdas]
    elif args.em_lambda_step is not None and args.em_lambda_start is not None and args.em_lambda_stop is not None:
        grid = np.arange(args.em_lambda_start, args.em_lambda_stop + args.em_lambda_step / 2.0, args.em_lambda_step)
        # Keep the endpoint inclusive when aligned, but never exceed the requested stop.
        args.em_lambda_grid = [float(round(x, 6)) for x in grid if x <= args.em_lambda_stop + 1e-9]
    else:
        args.em_lambda_grid = None
    if args.em_lambda_grid:
        print(f"[transductive_benchmark] lambda grid: {len(args.em_lambda_grid)} values "
              f"[{args.em_lambda_grid[0]:g} .. {args.em_lambda_grid[-1]:g}]")

    # Build the temperature grid (explicit list takes precedence over start/stop/step).
    if args.feature_Ts is not None:
        args.feature_T_grid = [float(x) for x in args.feature_Ts]
    elif args.feature_T_step is not None and args.feature_T_start is not None and args.feature_T_stop is not None:
        grid = np.arange(args.feature_T_start, args.feature_T_stop + args.feature_T_step / 2.0, args.feature_T_step)
        args.feature_T_grid = [float(round(x, 6)) for x in grid if x <= args.feature_T_stop + 1e-9]
    else:
        args.feature_T_grid = None
    if args.feature_T_grid:
        print(f"[transductive_benchmark] temperature grid: {len(args.feature_T_grid)} values "
              f"[{args.feature_T_grid[0]:g} .. {args.feature_T_grid[-1]:g}]")

    # TransCLIP grids (explicit lists only).
    args.transclip_lambda_grid = [float(x) for x in args.transclip_lambdas] if args.transclip_lambdas else None
    args.transclip_neighbors_grid = [int(x) for x in args.transclip_neighbors_list] if args.transclip_neighbors_list else None
    if args.transclip_lambda_grid or args.transclip_neighbors_grid:
        print(f"[transductive_benchmark] transclip grid: lambdas={args.transclip_lambda_grid} "
              f"neighbors={args.transclip_neighbors_grid}")

    all_rows = []
    for subject_id in args.held_out_subjects:
        print(f"[transductive_benchmark] === subject {subject_id:02d} ===")
        all_rows.extend(_evaluate_subject(subject_id, args))

    subject_df = pd.DataFrame(all_rows)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    subject_csv = os.path.join(args.output_dir, f"transductive_benchmark_subjects_{stamp}.csv")
    summary_csv = os.path.join(args.output_dir, f"transductive_benchmark_summary_{stamp}.csv")
    subject_df.to_csv(subject_csv, index=False)
    summary_df = _summarize(subject_df)
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== summary (mean over subjects) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved per-subject rows: {subject_csv}")
    print(f"Saved summary:          {summary_csv}")

    if args.plot:
        title = (f"Transductive methods — {args.eval_split} split, "
                 f"{args.n_queries or 'all'}q/{args.n_images or 'all'}K, "
                 f"{len(args.held_out_subjects)} subj (best config/method)")
        _make_barplot(summary_df, os.path.join(args.output_dir, f"comparison_barplot_{stamp}.png"), title)


if __name__ == "__main__":
    main()
