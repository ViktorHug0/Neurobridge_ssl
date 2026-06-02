#!/usr/bin/env python3
"""Fit target-subject adapters on the subject's train split and test on 200-way retrieval.

The calibration split uses the left-out subject's THINGS-EEG training images
(4 repetitions averaged per image). Each fitted map is frozen and evaluated on
the fixed 200-object test set.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from shared import (
    CayleyOrthogonalAdapter,
    DEFAULT_OUTPUT_ROOT,
    EmbeddingResidualAdapter,
    TTAParams,
    add_common_args,
    aggregate_results,
    apply_linear_map,
    apply_supervised_orthogonal,
    apply_torch_adapter,
    apply_train_zca,
    encode_average_features,
    encode_indexed_average_features,
    ensure_output_dir,
    evaluate_plain,
    evaluate_saw_csls,
    evaluate_scores,
    fit_neural_adapter,
    fit_paired_orthogonal,
    fit_paired_ridge,
    fit_train_zca_stats,
    load_subject_train_test_context,
    normalize_rows,
    params_from_args,
    score_features,
    write_config,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "rebuttal_suite_20260602-134420",
    "trainset_rotation_transfer",
)


def _parse_calibration_sizes(values, total: int):
    sizes = []
    for value in values:
        if str(value).lower() == "all":
            size = total
        else:
            size = int(value)
        if 1 <= size <= total and size not in sizes:
            sizes.append(size)
    return sizes


def _split_calibration_indices(indices, val_fraction: float, min_val_size: int, seed: int):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 4 or val_fraction <= 0:
        return indices, np.array([], dtype=np.int64)
    val_size = int(round(len(indices) * float(val_fraction)))
    val_size = max(int(min_val_size), val_size)
    val_size = min(val_size, len(indices) // 2)
    if val_size <= 0:
        return indices, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    return np.sort(perm[val_size:]), np.sort(perm[:val_size])


def _evaluate_test(transformed_test, test_images, params: TTAParams):
    scores = score_features(transformed_test, test_images, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores)


def _evaluate_val(transformed_val, candidate_images, val_positions, params: TTAParams):
    scores = score_features(transformed_val, candidate_images, use_csls=params.use_csls, csls_k=params.csls_k)
    return evaluate_scores(scores, target_indices=val_positions)


def _append(rows, subject_id, seed, calibration_size, fit_size, val_size, method, hyperparam, metrics, plain_ref):
    rows.append(
        {
            "subject_id": int(subject_id),
            "seed": int(seed),
            "calibration_size": int(calibration_size),
            "fit_size": int(fit_size),
            "val_size": int(val_size),
            "method": method,
            "hyperparam": hyperparam or "",
            "top1_acc": round(float(metrics["top1_acc"]), 4),
            "top5_acc": round(float(metrics["top5_acc"]), 4),
            "delta_top1_vs_plain": round(float(metrics["top1_acc"]) - float(plain_ref["top1_acc"]), 4),
            "delta_top5_vs_plain": round(float(metrics["top5_acc"]) - float(plain_ref["top5_acc"]), 4),
        }
    )


def _select_alpha(train_q, train_i, val_q, val_i, alpha_values, params):
    rotation = fit_paired_orthogonal(train_q, train_i)
    if len(val_q) == 0:
        return {"alpha": 1.0, "rotation": rotation}
    best = None
    for alpha in alpha_values:
        transformed = apply_supervised_orthogonal(val_q, rotation, alpha=float(alpha))
        metrics = _evaluate_val(transformed, val_i, np.arange(len(val_i), dtype=np.int64), params)
        cand = {"alpha": float(alpha), "rotation": rotation, "metrics": metrics}
        if best is None or metrics["top1_acc"] > best["metrics"]["top1_acc"]:
            best = cand
    return best


def _select_ridge(train_q, train_i, val_q, val_i, ridge_values, params):
    best = None
    for ridge in ridge_values:
        weights = fit_paired_ridge(train_q, train_i, ridge=float(ridge))
        if len(val_q) == 0:
            metrics = {"top1_acc": 0.0, "top5_acc": 0.0}
        else:
            metrics = _evaluate_val(apply_linear_map(val_q, weights), val_i, np.arange(len(val_i), dtype=np.int64), params)
        cand = {"ridge": float(ridge), "weights": weights, "metrics": metrics}
        if best is None or metrics["top1_acc"] > best["metrics"]["top1_acc"]:
            best = cand
    return best


def _batch_indices(n_items: int, batch_size: int, rng):
    order = rng.permutation(n_items)
    for start in range(0, n_items, batch_size):
        yield order[start : start + batch_size]


def _fit_cayley_rotation(train_q, train_i, val_q, val_i, args, params):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dim = train_q.shape[1]
    model = CayleyOrthogonalAdapter(dim, init_scale=args.cayley_init_scale).to(device)
    train_q_t = torch.tensor(train_q, dtype=torch.float32, device=device)
    train_i_t = torch.tensor(train_i, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cayley_lr, weight_decay=args.cayley_weight_decay)
    rng = np.random.default_rng(int(args.seed))
    best_state = None
    best_val = -1.0
    stale = 0
    tau = max(float(args.cayley_temperature), 1e-6)

    for _epoch in range(int(args.cayley_epochs)):
        model.train()
        for batch_idx in _batch_indices(len(train_q), int(args.cayley_batch_size), rng):
            q = train_q_t[batch_idx]
            img = train_i_t[batch_idx]
            optimizer.zero_grad()
            out = torch.nn.functional.normalize(model(q), dim=1)
            img = torch.nn.functional.normalize(img, dim=1)
            logits = (out @ img.T) / tau
            labels = torch.arange(len(batch_idx), device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = loss + float(args.cayley_mse_weight) * torch.nn.functional.mse_loss(out, img)
            loss.backward()
            optimizer.step()

        if len(val_q) == 0:
            continue
        transformed = _apply_cayley(model, val_q)
        metrics = _evaluate_val(transformed, val_i, np.arange(len(val_i), dtype=np.int64), params)
        if metrics["top1_acc"] > best_val:
            best_val = metrics["top1_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.cayley_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _apply_cayley(model, features):
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device)
        return torch.nn.functional.normalize(model(x), dim=1).cpu().numpy().astype(np.float32, copy=False)


def _fit_residual_mlp(train_q, train_i, val_q, val_i, args, params):
    from shared import NeuralAdapterConfig

    config = NeuralAdapterConfig(
        hidden_dim=int(args.neural_hidden_dim) if args.neural_hidden_dim > 0 else 0,
        num_layers=int(args.neural_num_layers),
        dropout=float(args.neural_dropout),
        residual_scale=float(args.neural_residual_scale),
        learning_rate=float(args.neural_lr),
        weight_decay=float(args.neural_weight_decay),
        max_epochs=int(args.neural_epochs),
        patience=int(args.neural_patience),
        mse_weight=float(args.neural_mse_weight),
        temperature=float(args.neural_temperature),
        device=args.device if torch.cuda.is_available() else "cpu",
    )
    return fit_neural_adapter(
        train_q,
        train_i,
        val_q,
        val_i,
        val_i,
        np.arange(len(val_i), dtype=np.int64),
        config,
        params,
    )


def run_subject(source_run_dir, subject_id, args, params):
    checkpoint_dir, eval_args, train_dataset, test_dataset, modules = load_subject_train_test_context(
        source_run_dir, args, subject_id, average=True
    )
    train_q, train_i, _, _ = encode_indexed_average_features(eval_args, modules, train_dataset)
    test_q, test_i = encode_average_features(eval_args, modules, test_dataset)
    total_train = train_q.shape[0]
    sizes = _parse_calibration_sizes(args.calibration_sizes, total_train)
    rows = []

    plain_metrics = evaluate_plain(test_q, test_i)
    saw_test_metrics = evaluate_saw_csls(test_q, test_i, params)

    for calibration_size in sizes:
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(calibration_size))
            calibration_idx = np.sort(rng.choice(total_train, size=calibration_size, replace=False))
            calibration_q = train_q[calibration_idx]
            calibration_i = train_i[calibration_idx]
            fit_idx, val_idx = _split_calibration_indices(
                calibration_idx,
                val_fraction=args.val_fraction,
                min_val_size=args.min_val_size,
                seed=int(seed) + int(subject_id) + int(calibration_size),
            )
            fit_q = train_q[fit_idx]
            fit_i = train_i[fit_idx]
            val_q = train_q[val_idx]
            val_i = train_i[val_idx]
            fit_size = len(fit_idx)
            val_size = len(val_idx)

            _append(rows, subject_id, seed, calibration_size, fit_size, val_size, "plain_cosine", None, plain_metrics, plain_metrics)
            _append(rows, subject_id, seed, calibration_size, fit_size, val_size, "fresh_saw_csls_on_test", None, saw_test_metrics, plain_metrics)

            zca_stats = fit_train_zca_stats(calibration_q, params)
            transformed = apply_train_zca(test_q, zca_stats)
            _append(
                rows,
                subject_id,
                seed,
                calibration_size,
                calibration_size,
                0,
                "trainset_zca",
                None,
                _evaluate_test(transformed, test_i, params),
                plain_metrics,
            )

            rotation = fit_paired_orthogonal(calibration_q, calibration_i)
            transformed = apply_supervised_orthogonal(test_q, rotation, alpha=1.0)
            _append(
                rows,
                subject_id,
                seed,
                calibration_size,
                calibration_size,
                0,
                "trainset_orthogonal",
                None,
                _evaluate_test(transformed, test_i, params),
                plain_metrics,
            )

            calibration_q_zca = apply_train_zca(calibration_q, zca_stats)
            rotation_zca = fit_paired_orthogonal(calibration_q_zca, calibration_i)
            transformed = apply_supervised_orthogonal(apply_train_zca(test_q, zca_stats), rotation_zca, alpha=1.0)
            _append(
                rows,
                subject_id,
                seed,
                calibration_size,
                calibration_size,
                0,
                "trainset_zca_orthogonal",
                None,
                _evaluate_test(transformed, test_i, params),
                plain_metrics,
            )

            alpha_fit = _select_alpha(fit_q, fit_i, val_q, val_i, args.alpha_values, params)
            transformed = apply_supervised_orthogonal(test_q, alpha_fit["rotation"], alpha=alpha_fit["alpha"])
            _append(
                rows,
                subject_id,
                seed,
                calibration_size,
                fit_size,
                val_size,
                "trainset_blended_orthogonal",
                f"alpha={alpha_fit['alpha']}",
                _evaluate_test(transformed, test_i, params),
                plain_metrics,
            )

            ridge_fit = _select_ridge(fit_q, fit_i, val_q, val_i, args.ridge_values, params)
            transformed = apply_linear_map(test_q, ridge_fit["weights"])
            _append(
                rows,
                subject_id,
                seed,
                calibration_size,
                fit_size,
                val_size,
                "trainset_ridge_linear",
                f"ridge={ridge_fit['ridge']}",
                _evaluate_test(transformed, test_i, params),
                plain_metrics,
            )

            if args.run_cayley:
                cayley = _fit_cayley_rotation(fit_q, fit_i, val_q, val_i, args, params)
                transformed = _apply_cayley(cayley, test_q)
                _append(
                    rows,
                    subject_id,
                    seed,
                    calibration_size,
                    fit_size,
                    val_size,
                    "trainset_cayley_orthogonal",
                    "val_early_stop",
                    _evaluate_test(transformed, test_i, params),
                    plain_metrics,
                )

            if args.run_neural_adapter and calibration_size <= int(args.max_neural_calibration_size):
                neural = _fit_residual_mlp(fit_q, fit_i, val_q, val_i, args, params)
                transformed = apply_torch_adapter(neural, test_q)
                _append(
                    rows,
                    subject_id,
                    seed,
                    calibration_size,
                    fit_size,
                    val_size,
                    "trainset_residual_mlp",
                    "val_early_stop",
                    _evaluate_test(transformed, test_i, params),
                    plain_metrics,
                )

    for row in rows:
        row["total_train_samples"] = int(total_train)
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--calibration_sizes", nargs="+", default=["100", "500", "1000", "5000", "10000", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3300, 3301, 3302])
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--min_val_size", type=int, default=20)
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--ridge_values", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--run_cayley", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cayley_epochs", type=int, default=200)
    parser.add_argument("--cayley_patience", type=int, default=25)
    parser.add_argument("--cayley_batch_size", type=int, default=512)
    parser.add_argument("--cayley_lr", type=float, default=1e-3)
    parser.add_argument("--cayley_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cayley_mse_weight", type=float, default=0.5)
    parser.add_argument("--cayley_temperature", type=float, default=0.07)
    parser.add_argument("--cayley_init_scale", type=float, default=1e-3)
    parser.add_argument("--run_neural_adapter", action="store_true")
    parser.add_argument("--max_neural_calibration_size", type=int, default=5000)
    parser.add_argument("--neural_hidden_dim", type=int, default=0)
    parser.add_argument("--neural_num_layers", type=int, default=1)
    parser.add_argument("--neural_dropout", type=float, default=0.1)
    parser.add_argument("--neural_residual_scale", type=float, default=0.1)
    parser.add_argument("--neural_lr", type=float, default=1e-3)
    parser.add_argument("--neural_weight_decay", type=float, default=1e-2)
    parser.add_argument("--neural_epochs", type=int, default=100)
    parser.add_argument("--neural_patience", type=int, default=15)
    parser.add_argument("--neural_mse_weight", type=float, default=0.5)
    parser.add_argument("--neural_temperature", type=float, default=0.07)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "trainset_rotation_transfer")
    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[trainset_rotation_transfer] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, params))
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)

    subject_df = pd.DataFrame(all_rows)
    average_df = aggregate_results(subject_df, ["method", "calibration_size", "hyperparam"])
    average_df.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    delta_df = (
        subject_df.groupby(["method", "calibration_size"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top5_mean=("top5_acc", "mean"),
            delta_top1_vs_plain_mean=("delta_top1_vs_plain", "mean"),
            delta_top5_vs_plain_mean=("delta_top5_vs_plain", "mean"),
        )
        .sort_values(["calibration_size", "delta_top1_vs_plain_mean"], ascending=[True, False])
    )
    delta_df.to_csv(os.path.join(output_dir, "delta_vs_plain.csv"), index=False)

    print("\n=== Train-set Rotation Transfer (mean delta vs plain 200-way test) ===")
    print(delta_df.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
