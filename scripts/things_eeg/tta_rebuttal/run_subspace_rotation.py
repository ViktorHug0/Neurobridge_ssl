#!/usr/bin/env python3
"""Subspace-constrained rotation estimation for sample-efficient subject adaptation.

Methods compared:
1. Unconstrained Procrustes (SVD-based, needs n >> d)
2. Subspace-constrained Procrustes (rotation lives in learned low-dim subspace)
3. Geodesic interpolation at fixed alpha (expm(alpha * logm(R)))
4. Cross-validated geodesic alpha (select alpha on val split)
5. Analytic geodesic alpha heuristic
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from scipy.optimize import minimize

from shared import (
    DEFAULT_OUTPUT_ROOT,
    TTAParams,
    add_common_args,
    aggregate_results,
    encode_average_features,
    encode_indexed_average_features,
    ensure_output_dir,
    evaluate_plain,
    evaluate_scores,
    fit_paired_orthogonal,
    load_subject_train_test_context,
    normalize_rows,
    params_from_args,
    score_features,
    write_config,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "rebuttal_suite_featdim64_mixup",
    "subspace_rotation",
)


# --- Lie-algebra utilities ---

def _skew_to_vec(A):
    """Extract upper-triangle of skew-symmetric matrix as a flat vector."""
    d = A.shape[0]
    return A[np.triu_indices(d, k=1)].astype(np.float64)


def _vec_to_skew(v, d):
    """Reconstruct skew-symmetric matrix from upper-triangle vector."""
    A = np.zeros((d, d), dtype=np.float64)
    A[np.triu_indices(d, k=1)] = v
    A -= A.T
    return A


def _safe_logm(R):
    """Matrix logarithm of an orthogonal matrix, forced skew-symmetric."""
    L = logm(R.astype(np.float64))
    return np.real(0.5 * (L - L.T))


def _project_to_orthogonal(M):
    """Project a matrix onto O(d) via SVD."""
    u, _, vt = np.linalg.svd(M, full_matrices=False)
    return (u @ vt).astype(np.float64)


# --- Subspace PCA ---

def compute_rotation_subspace(rotations, explained_variance_target=0.95):
    """PCA on vectorized log-maps of per-subject rotations."""
    d = rotations[0].shape[0]
    vecs = np.stack([_skew_to_vec(_safe_logm(R)) for R in rotations], axis=0)
    mean_vec = vecs.mean(axis=0)
    centered = vecs - mean_vec

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = (S**2).sum()
    if total_var < 1e-12:
        explained = np.ones(1)
        k = 1
    else:
        explained = np.cumsum(S**2) / total_var
        k = int(np.searchsorted(explained, explained_variance_target)) + 1
        k = max(1, min(k, len(S)))

    basis_vecs = Vt[:k]
    basis = np.stack([_vec_to_skew(v, d) for v in basis_vecs], axis=0)
    return basis, mean_vec, explained, vecs


# --- Method: Subspace-constrained Procrustes ---

def fit_subspace_rotation_scipy(calibration_q, calibration_i, basis, mean_vec, d):
    """Optimize R = expm(mean + sum c_i * B_i) over k coefficients."""
    k = basis.shape[0]
    Q = calibration_q.astype(np.float64)
    I_target = calibration_i.astype(np.float64)
    basis_vecs = np.stack([_skew_to_vec(basis[i]) for i in range(k)], axis=0)

    def _objective(c):
        A = _vec_to_skew(mean_vec + basis_vecs.T @ c, d)
        R = expm(A)
        residual = (Q @ R) - I_target
        return 0.5 * np.sum(residual**2)

    c0 = np.zeros(k, dtype=np.float64)
    result = minimize(_objective, c0, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    A = _vec_to_skew(mean_vec + basis_vecs.T @ result.x, d)
    R = expm(A)
    return R.astype(np.float32), result.x


# --- Method: Cross-validated geodesic alpha ---

def fit_geodesic_cv_alpha(cal_q, cal_i, test_q, test_i, params, alpha_grid, val_fraction=0.3, seed=0):
    """Split calibration into fit/val, select best geodesic alpha on val retrieval."""
    n = len(cal_q)
    val_size = max(4, int(round(n * val_fraction)))
    val_size = min(val_size, n // 2)
    if val_size < 4:
        # Too few samples to split; use analytic heuristic
        alpha_hat = _analytic_alpha(n, cal_q.shape[1])
        R = fit_paired_orthogonal(cal_q, cal_i)
        A = _safe_logm(R)
        R_out = expm(alpha_hat * A).astype(np.float32)
        return R_out, alpha_hat, "analytic"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    fit_idx = perm[val_size:]

    fit_q, fit_i = cal_q[fit_idx], cal_i[fit_idx]
    val_q, val_i = cal_q[val_idx], cal_i[val_idx]

    R = fit_paired_orthogonal(fit_q, fit_i)
    A = _safe_logm(R)

    best_alpha = 0.0
    best_val_top1 = -1.0
    for alpha in alpha_grid:
        R_alpha = expm(float(alpha) * A).astype(np.float32)
        transformed_val = normalize_rows(val_q @ R_alpha)
        scores = score_features(transformed_val, val_i, use_csls=params.use_csls, csls_k=params.csls_k)
        metrics = evaluate_scores(scores)
        if metrics["top1_acc"] > best_val_top1:
            best_val_top1 = metrics["top1_acc"]
            best_alpha = float(alpha)

    # Refit on full calibration with selected alpha
    R_full = fit_paired_orthogonal(cal_q, cal_i)
    A_full = _safe_logm(R_full)
    R_out = expm(best_alpha * A_full).astype(np.float32)
    return R_out, best_alpha, "cv"


def _analytic_alpha(n, d):
    """Heuristic: alpha(n) = n / (n + n0) where n0 ~ d."""
    n0 = float(d)
    return float(n) / (float(n) + n0)


# --- Evaluation ---

def _evaluate_test(transformed_test, test_images, params: TTAParams):
    scores = score_features(transformed_test, test_images, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores)


def _append(rows, subject_id, seed, calibration_size, method, hyperparam, metrics, plain_ref, extra=None):
    row = {
        "subject_id": int(subject_id),
        "seed": int(seed),
        "calibration_size": int(calibration_size),
        "method": method,
        "hyperparam": hyperparam or "",
        "top1_acc": round(float(metrics["top1_acc"]), 4),
        "top5_acc": round(float(metrics["top5_acc"]), 4),
        "delta_top1_vs_plain": round(float(metrics["top1_acc"]) - float(plain_ref["top1_acc"]), 4),
        "delta_top5_vs_plain": round(float(metrics["top5_acc"]) - float(plain_ref["top5_acc"]), 4),
    }
    if extra:
        row.update(extra)
    rows.append(row)


# --- Main per-subject routine ---

def run_subject(source_run_dir, subject_id, all_subject_ids, args, params):
    other_subjects = [s for s in all_subject_ids if s != subject_id]

    # Step 1: fit full rotations for reference subjects
    print(f"  [step 1] Fitting full rotations for {len(other_subjects)} reference subjects...")
    reference_rotations = []
    for ref_id in other_subjects:
        _, ref_eval_args, ref_train_dataset, _, ref_modules = load_subject_train_test_context(
            source_run_dir, args, ref_id, average=True
        )
        ref_q, ref_i, _, _ = encode_indexed_average_features(ref_eval_args, ref_modules, ref_train_dataset)
        R = fit_paired_orthogonal(ref_q, ref_i)
        reference_rotations.append(R)

    d = reference_rotations[0].shape[0]

    # Step 2: compute subspace
    print(f"  [step 2] Computing rotation subspace (d={d})...")
    basis, mean_vec, explained_var, _ = compute_rotation_subspace(reference_rotations)
    k_95 = int(np.searchsorted(explained_var, 0.95)) + 1
    k_99 = int(np.searchsorted(explained_var, 0.99)) + 1
    print(f"    dims for 95% variance: {k_95}, 99%: {k_99}, total components: {len(explained_var)}")

    # Load held-out subject
    checkpoint_dir, eval_args, train_dataset, test_dataset, modules = load_subject_train_test_context(
        source_run_dir, args, subject_id, average=True
    )
    train_q, train_i, _, _ = encode_indexed_average_features(eval_args, modules, train_dataset)
    test_q, test_i = encode_average_features(eval_args, modules, test_dataset)
    total_train = train_q.shape[0]

    plain_metrics = evaluate_plain(test_q, test_i)
    rows = []

    sizes = []
    for v in args.calibration_sizes:
        s = total_train if str(v).lower() == "all" else int(v)
        if 1 <= s <= total_train and s not in sizes:
            sizes.append(s)

    alpha_cv_grid = np.linspace(0.0, 1.0, 21).tolist()

    for calibration_size in sizes:
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(calibration_size))
            cal_idx = np.sort(rng.choice(total_train, size=calibration_size, replace=False))
            cal_q = train_q[cal_idx]
            cal_i = train_i[cal_idx]

            _append(rows, subject_id, seed, calibration_size, "plain_cosine", None, plain_metrics, plain_metrics)

            # 1. Unconstrained Procrustes
            rotation_full = fit_paired_orthogonal(cal_q, cal_i)
            transformed = normalize_rows(test_q @ rotation_full)
            _append(rows, subject_id, seed, calibration_size, "unconstrained_procrustes", None,
                    _evaluate_test(transformed, test_i, params), plain_metrics)

            # 2. Subspace-constrained
            for k in args.subspace_dims:
                if k > basis.shape[0]:
                    continue
                R_sub, _ = fit_subspace_rotation_scipy(cal_q, cal_i, basis[:k], mean_vec, d)
                transformed = normalize_rows(test_q @ R_sub)
                _append(rows, subject_id, seed, calibration_size, "subspace_procrustes",
                        f"k={k}", _evaluate_test(transformed, test_i, params), plain_metrics)

            # 3. Fixed geodesic alphas
            A_full = _safe_logm(rotation_full)
            for alpha in args.alpha_values:
                R_alpha = expm(float(alpha) * A_full).astype(np.float32)
                transformed = normalize_rows(test_q @ R_alpha)
                _append(rows, subject_id, seed, calibration_size, "geodesic_fixed",
                        f"alpha={float(alpha)}", _evaluate_test(transformed, test_i, params), plain_metrics)

            # 4. Cross-validated geodesic alpha
            R_cv, best_alpha, mode = fit_geodesic_cv_alpha(
                cal_q, cal_i, test_q, test_i, params, alpha_cv_grid,
                val_fraction=args.val_fraction, seed=int(seed) + int(subject_id))
            transformed = normalize_rows(test_q @ R_cv)
            _append(rows, subject_id, seed, calibration_size, "geodesic_cv_alpha",
                    f"alpha={best_alpha:.2f}({mode})",
                    _evaluate_test(transformed, test_i, params), plain_metrics)

            # 5. Analytic alpha heuristic (no val split needed)
            alpha_hat = _analytic_alpha(calibration_size, d)
            R_hat = expm(alpha_hat * A_full).astype(np.float32)
            transformed = normalize_rows(test_q @ R_hat)
            _append(rows, subject_id, seed, calibration_size, "geodesic_analytic_alpha",
                    f"alpha={alpha_hat:.3f}",
                    _evaluate_test(transformed, test_i, params), plain_metrics)

    for row in rows:
        row["total_train_samples"] = int(total_train)
        row["checkpoint_dir"] = checkpoint_dir
    return rows, {"k_95": k_95, "k_99": k_99, "explained_variance": explained_var.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--calibration_sizes", nargs="+", default=["10", "20", "50", "100", "500", "1000", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    parser.add_argument("--subspace_dims", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--val_fraction", type=float, default=0.3)
    parser.add_argument("--all_subject_ids", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "subspace_rotation")
    write_config(output_dir, args, params)

    all_rows = []
    subspace_info = {}
    all_subject_ids = args.all_subject_ids
    for subject_id in args.subjects:
        print(f"[subspace_rotation] subject={int(subject_id):02d}")
        subject_rows, info = run_subject(source_run_dir, subject_id, all_subject_ids, args, params)
        all_rows.extend(subject_rows)
        subspace_info[f"sub-{int(subject_id):02d}"] = info
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    import json
    with open(os.path.join(output_dir, "subspace_analysis.json"), "w") as f:
        json.dump(subspace_info, f, indent=2)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["method", "calibration_size", "hyperparam"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    delta_df = (
        subject_df.groupby(["method", "calibration_size", "hyperparam"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top5_mean=("top5_acc", "mean"),
            delta_top1_vs_plain_mean=("delta_top1_vs_plain", "mean"),
            delta_top5_vs_plain_mean=("delta_top5_vs_plain", "mean"),
        )
        .sort_values(["calibration_size", "delta_top1_vs_plain_mean"], ascending=[True, False])
    )
    delta_df.to_csv(os.path.join(output_dir, "delta_vs_plain.csv"), index=False)

    # Summary: best method per calibration size
    best_per_size = delta_df.loc[delta_df.groupby("calibration_size")["top1_mean"].idxmax()]
    print("\n=== Best method per calibration size ===")
    print(best_per_size[["calibration_size", "method", "hyperparam", "top1_mean", "delta_top1_vs_plain_mean"]].to_string(index=False))
    print(f"\nSubspace dims (k for 95%): {[v['k_95'] for v in subspace_info.values()]}")
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
