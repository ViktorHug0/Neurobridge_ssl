#!/usr/bin/env python3
"""Few-shot subject adaptation on held-out test objects.

Split each subject's 200 test objects into train / val / test (default 80/20/100),
fit lightweight subject adapters on labeled train pairs, select hyperparameters on
val only, and evaluate transfer to unseen test stimuli.

Results include explicit deltas vs train-fit ZCA whitening (mandatory baseline).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from shared import (
    DEFAULT_FEWSHOT_OUTPUT,
    NeuralAdapterConfig,
    TTAParams,
    add_common_args,
    aggregate_results,
    apply_linear_map,
    apply_supervised_orthogonal,
    apply_torch_adapter,
    apply_train_zca,
    apply_tta_calibration,
    encode_average_features,
    ensure_output_dir,
    evaluate_plain,
    evaluate_saw_csls,
    evaluate_scores,
    fit_low_rank_adapter,
    fit_neural_adapter,
    fit_paired_orthogonal,
    fit_paired_ridge,
    fit_tta_calibration,
    fit_train_zca_stats,
    load_subject_context,
    params_from_args,
    score_features,
    write_config,
)


def _train_sizes(args, total: int):
    if args.train_sizes:
        sizes = []
        for s in args.train_sizes:
            s = int(s)
            if s + int(args.val_size) < total:
                sizes.append(s)
        return sizes
    return [int(args.train_size)]


def _split_indices(total: int, train_size: int, val_size: int, seed: int, subject_id: int):
    rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(train_size))
    perm = rng.permutation(total)
    train_indices = np.sort(perm[:train_size])
    val_indices = np.sort(perm[train_size : train_size + val_size])
    test_indices = np.sort(perm[train_size + val_size :])
    return train_indices, val_indices, test_indices


def _scopes(test_indices, all_images):
    return [
        ("test_vs_test", all_images[test_indices], np.arange(len(test_indices), dtype=np.int64)),
        ("test_vs_all_candidates", all_images, test_indices),
    ]


def _eval_transformed(transformed, candidates, targets, params: TTAParams):
    scores = score_features(transformed, candidates, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores, target_indices=targets)


def _row(
    subject_id,
    seed,
    train_size,
    scope,
    method,
    fit_protocol,
    hyperparam,
    metrics,
    train_zca_ref,
):
    delta1 = round(float(metrics["top1_acc"]) - float(train_zca_ref["top1_acc"]), 4)
    delta5 = round(float(metrics["top5_acc"]) - float(train_zca_ref["top5_acc"]), 4)
    return {
        "subject_id": int(subject_id),
        "seed": int(seed),
        "train_size": int(train_size),
        "scope": scope,
        "method": method,
        "fit_protocol": fit_protocol,
        "hyperparam": hyperparam if hyperparam is not None else "",
        "top1_acc": round(float(metrics["top1_acc"]), 4),
        "top5_acc": round(float(metrics["top5_acc"]), 4),
        "delta_top1_vs_train_zca": delta1,
        "delta_top5_vs_train_zca": delta5,
    }


def _select_alpha_on_val(
    train_q,
    train_i,
    val_q,
    val_i,
    candidates,
    val_targets,
    saw_stats,
    use_zca: bool,
    map_kind: str,
    alpha_values,
    params: TTAParams,
):
    if use_zca:
        train_q_w = apply_train_zca(train_q, saw_stats)
        val_q_w = apply_train_zca(val_q, saw_stats)
    else:
        train_q_w, val_q_w = train_q, val_q

    if map_kind == "orthogonal":
        base_map = fit_paired_orthogonal(train_q_w, train_i)
    else:
        raise ValueError(map_kind)

    best = None
    for alpha in alpha_values:
        transformed = apply_supervised_orthogonal(val_q_w, base_map, alpha=float(alpha))
        metrics = _eval_transformed(transformed, candidates, val_targets, params)
        cand = {"alpha": float(alpha), "map": base_map, "metrics": metrics}
        if best is None or metrics["top1_acc"] > best["metrics"]["top1_acc"]:
            best = cand
    return best


def _select_ridge_on_val(
    train_q,
    train_i,
    val_q,
    candidates,
    val_targets,
    saw_stats,
    use_zca: bool,
    ridge_values,
    params: TTAParams,
):
    if use_zca:
        train_q_w = apply_train_zca(train_q, saw_stats)
        val_q_w = apply_train_zca(val_q, saw_stats)
    else:
        train_q_w, val_q_w = train_q, val_q

    best = None
    for ridge in ridge_values:
        weights = fit_paired_ridge(train_q_w, train_i, ridge=float(ridge))
        transformed = apply_linear_map(val_q_w, weights)
        metrics = _eval_transformed(transformed, candidates, val_targets, params)
        cand = {"ridge": float(ridge), "weights": weights, "metrics": metrics}
        if best is None or metrics["top1_acc"] > best["metrics"]["top1_acc"]:
            best = cand
    return best


def _select_rank_on_val(
    train_q,
    train_i,
    val_q,
    val_i,
    candidates,
    val_targets,
    rank_values,
    neural_config: NeuralAdapterConfig,
    params: TTAParams,
):
    best = None
    for rank in rank_values:
        bundle = fit_low_rank_adapter(
            train_q,
            train_i,
            val_q,
            val_i,
            candidates,
            val_targets,
            rank=int(rank),
            config=neural_config,
            params=params,
        )
        transformed = apply_torch_adapter(bundle, val_q)
        metrics = _eval_transformed(transformed, candidates, val_targets, params)
        cand = {"rank": int(rank), "bundle": bundle, "metrics": metrics}
        if best is None or metrics["top1_acc"] > best["metrics"]["top1_acc"]:
            best = cand
    return best


def _fit_and_apply_orthogonal(train_q, train_i, test_q, alpha: float, saw_stats=None):
    if saw_stats is not None:
        train_q = apply_train_zca(train_q, saw_stats)
        test_q = apply_train_zca(test_q, saw_stats)
    rotation = fit_paired_orthogonal(train_q, train_i)
    return apply_supervised_orthogonal(test_q, rotation, alpha=alpha), rotation


def _fit_and_apply_ridge(train_q, train_i, test_q, ridge: float, saw_stats=None):
    if saw_stats is not None:
        train_q = apply_train_zca(train_q, saw_stats)
        test_q = apply_train_zca(test_q, saw_stats)
    weights = fit_paired_ridge(train_q, train_i, ridge=ridge)
    return apply_linear_map(test_q, weights), weights


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, dataset, modules = load_subject_context(
        source_run_dir, args, subject_id, average=True
    )
    query_features, image_features = encode_average_features(eval_args, modules, dataset)
    total = query_features.shape[0]
    val_size = int(args.val_size)
    rows = []

    import torch

    device_str = args.device if torch.cuda.is_available() else "cpu"
    neural_config = NeuralAdapterConfig(
        hidden_dim=int(args.neural_hidden_dim) if args.neural_hidden_dim > 0 else 0,
        num_layers=int(args.neural_num_layers),
        dropout=float(args.neural_dropout),
        residual_scale=float(args.neural_residual_scale),
        learning_rate=float(args.neural_lr),
        weight_decay=float(args.neural_weight_decay),
        max_epochs=int(args.neural_max_epochs),
        patience=int(args.neural_patience),
        mse_weight=float(args.neural_mse_weight),
        device=device_str,
    )

    for train_size in _train_sizes(args, total):
        if train_size + val_size >= total:
            continue
        for seed in args.seeds:
            train_idx, val_idx, test_idx = _split_indices(total, train_size, val_size, seed, subject_id)
            train_q = query_features[train_idx]
            train_i = image_features[train_idx]
            val_q = query_features[val_idx]
            val_i = image_features[val_idx]
            test_q = query_features[test_idx]

            val_candidates = image_features
            val_targets = val_idx
            train_zca_stats = fit_train_zca_stats(train_q, params)

            scope_metrics_train_zca = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                zca_test = apply_train_zca(test_q, train_zca_stats)
                scope_metrics_train_zca[scope] = _eval_transformed(zca_test, candidates, targets, params)

            def append_all(method, fit_protocol, hyperparam, per_scope_metrics):
                for scope, metrics in per_scope_metrics.items():
                    rows.append(
                        _row(
                            subject_id,
                            seed,
                            train_size,
                            scope,
                            method,
                            fit_protocol,
                            hyperparam,
                            metrics,
                            scope_metrics_train_zca[scope],
                        )
                    )

            # --- Baselines ---
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                per_scope[scope] = evaluate_plain(test_q, candidates, target_indices=targets)
            append_all("plain_cosine", "n/a", None, per_scope)

            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                per_scope[scope] = scope_metrics_train_zca[scope]
            append_all("train_zca", "train_only", None, per_scope)

            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                per_scope[scope] = evaluate_saw_csls(test_q, candidates, params, target_indices=targets)
            append_all("fresh_saw_csls_on_test", "test_only", None, per_scope)

            calibration = fit_tta_calibration(train_q, train_i, params)
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed = apply_tta_calibration(test_q, calibration, alpha=1.0)
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all("unsupervised_fit_split_tta", "train_only", None, per_scope)

            # --- Supervised orthogonal ---
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed, _ = _fit_and_apply_orthogonal(train_q, train_i, test_q, alpha=1.0)
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all("supervised_orthogonal", "train_only", None, per_scope)

            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed, _ = _fit_and_apply_orthogonal(
                    train_q, train_i, test_q, alpha=1.0, saw_stats=train_zca_stats
                )
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all("supervised_zca_orthogonal", "train_only", None, per_scope)

            best_alpha = _select_alpha_on_val(
                train_q,
                train_i,
                val_q,
                val_i,
                val_candidates,
                val_targets,
                train_zca_stats,
                use_zca=False,
                map_kind="orthogonal",
                alpha_values=args.alpha_values,
                params=params,
            )
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed = apply_supervised_orthogonal(
                    test_q, best_alpha["map"], alpha=best_alpha["alpha"]
                )
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all(
                "supervised_blended_orthogonal",
                "train_only",
                f"alpha={best_alpha['alpha']}",
                per_scope,
            )

            best_ridge = _select_ridge_on_val(
                train_q,
                train_i,
                val_q,
                val_candidates,
                val_targets,
                saw_stats=None,
                use_zca=False,
                ridge_values=args.ridge_values,
                params=params,
            )
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed = apply_linear_map(test_q, best_ridge["weights"])
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all(
                "supervised_ridge_linear",
                "train_only",
                f"ridge={best_ridge['ridge']}",
                per_scope,
            )

            best_zca_ridge = _select_ridge_on_val(
                train_q,
                train_i,
                val_q,
                val_candidates,
                val_targets,
                saw_stats=train_zca_stats,
                use_zca=True,
                ridge_values=args.ridge_values,
                params=params,
            )
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                test_w = apply_train_zca(test_q, train_zca_stats)
                transformed = apply_linear_map(test_w, best_zca_ridge["weights"])
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all(
                "supervised_zca_ridge_linear",
                "train_only",
                f"ridge={best_zca_ridge['ridge']}",
                per_scope,
            )

            best_rank = _select_rank_on_val(
                train_q,
                train_i,
                val_q,
                val_i,
                val_candidates,
                val_targets,
                args.rank_values,
                neural_config,
                params,
            )
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed = apply_torch_adapter(best_rank["bundle"], test_q)
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all(
                "supervised_subject_alignment_layer",
                "train_only",
                f"rank={best_rank['rank']}",
                per_scope,
            )

            neural_bundle = fit_neural_adapter(
                train_q,
                train_i,
                val_q,
                val_i,
                val_candidates,
                val_targets,
                neural_config,
                params,
            )
            per_scope = {}
            for scope, candidates, targets in _scopes(test_idx, image_features):
                transformed = apply_torch_adapter(neural_bundle, test_q)
                per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
            append_all("supervised_neural_adapter", "train_only", "val_early_stop", per_scope)

            # --- Optional refit on train+val ---
            if args.refit_train_val_after_selection:
                combined_idx = np.concatenate([train_idx, val_idx])
                comb_q = query_features[combined_idx]
                comb_i = image_features[combined_idx]
                comb_zca = fit_train_zca_stats(comb_q, params)

                per_scope = {}
                for scope, candidates, targets in _scopes(test_idx, image_features):
                    transformed, _ = _fit_and_apply_orthogonal(
                        comb_q, comb_i, test_q, alpha=1.0, saw_stats=comb_zca
                    )
                    per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
                append_all("supervised_zca_orthogonal", "train_val_refit", None, per_scope)

                best_alpha_refit = _select_alpha_on_val(
                    comb_q,
                    comb_i,
                    val_q,
                    val_i,
                    val_candidates,
                    val_targets,
                    comb_zca,
                    use_zca=True,
                    map_kind="orthogonal",
                    alpha_values=args.alpha_values,
                    params=params,
                )
                per_scope = {}
                for scope, candidates, targets in _scopes(test_idx, image_features):
                    test_w = apply_train_zca(test_q, comb_zca)
                    transformed = apply_supervised_orthogonal(
                        test_w, best_alpha_refit["map"], alpha=best_alpha_refit["alpha"]
                    )
                    per_scope[scope] = _eval_transformed(transformed, candidates, targets, params)
                append_all(
                    "supervised_blended_orthogonal",
                    "train_val_refit",
                    f"alpha={best_alpha_refit['alpha']}",
                    per_scope,
                )

    for row in rows:
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--train_size", type=int, default=80)
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--train_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--ridge_values", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
    parser.add_argument("--rank_values", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--refit_train_val_after_selection", action="store_true")
    parser.add_argument("--neural_hidden_dim", type=int, default=0)
    parser.add_argument("--neural_num_layers", type=int, default=1)
    parser.add_argument("--neural_dropout", type=float, default=0.1)
    parser.add_argument("--neural_residual_scale", type=float, default=0.1)
    parser.add_argument("--neural_lr", type=float, default=1e-3)
    parser.add_argument("--neural_weight_decay", type=float, default=1e-2)
    parser.add_argument("--neural_max_epochs", type=int, default=200)
    parser.add_argument("--neural_patience", type=int, default=20)
    parser.add_argument("--neural_mse_weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_FEWSHOT_OUTPUT

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "fewshot_subject_adaptation")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[fewshot] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(
        subject_df,
        ["scope", "method", "fit_protocol", "train_size", "hyperparam"],
    )
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    delta_df = (
        subject_df.groupby(["scope", "method", "fit_protocol", "train_size"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            delta_top1_mean=("delta_top1_vs_train_zca", "mean"),
            delta_top5_mean=("delta_top5_vs_train_zca", "mean"),
        )
        .sort_values(["scope", "train_size", "delta_top1_mean"], ascending=[True, True, False])
    )
    delta_df.to_csv(os.path.join(output_dir, "delta_vs_train_zca.csv"), index=False)

    best_df = (
        subject_df[subject_df["fit_protocol"] == "train_only"]
        .groupby(["scope", "train_size", "method"], as_index=False)
        .agg(top1_mean=("top1_acc", "mean"), delta_top1_mean=("delta_top1_vs_train_zca", "mean"))
        .sort_values(["scope", "train_size", "delta_top1_mean"], ascending=[True, True, False])
    )
    best_df.to_csv(os.path.join(output_dir, "best_val_selected_results.csv"), index=False)

    print("\n=== Few-Shot Adaptation (mean delta vs train_zca) ===")
    summary = (
        delta_df[delta_df["fit_protocol"] == "train_only"]
        .groupby(["scope", "method"], as_index=False)["delta_top1_mean"]
        .mean()
        .sort_values("delta_top1_mean", ascending=False)
    )
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
