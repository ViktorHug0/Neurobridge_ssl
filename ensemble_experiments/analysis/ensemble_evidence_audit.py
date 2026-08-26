"""Read-only audit of the stored THINGS-EEG2 ensemble score dumps.

This script does not train or evaluate a model.  It reconstructs cosine score
matrices from existing NPZ dumps and prints the core evidence used in
``ensemble_results_analysis_20260823.md``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t, wilcoxon


ROOT = Path("results/things_eeg/synthetic_subjects/ensemble_screen/dumps")
SUBJECTS = tuple(range(1, 11))
TRUTH = np.arange(200)
_CACHE: dict[str, np.ndarray] = {}


def load_scores(name: str) -> np.ndarray:
    """Return ten independently L2-normalized 200 x 200 cosine matrices."""
    if name not in _CACHE:
        matrices = []
        for subject in SUBJECTS:
            data = np.load(ROOT / f"{name}-sub{subject:02d}.npz")
            eeg = data["eeg"].astype(np.float32)
            image = data["image"].astype(np.float32)
            eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
            image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
            matrices.append(eeg @ image.T)
        _CACHE[name] = np.stack(matrices)
    return _CACHE[name]


def row_z(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(2, keepdims=True)) / np.maximum(
        scores.std(2, keepdims=True), 1e-6
    )


def transform(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return scores
    standardized = row_z(scores)
    if method == "row_z":
        return standardized
    if method == "signed_power_1.25":
        return np.sign(standardized) * np.abs(standardized) ** 1.25
    raise ValueError(method)


def predictions(name: str) -> np.ndarray:
    return load_scores(name).argmax(2)


def accuracies(predicted: np.ndarray) -> np.ndarray:
    return (predicted == TRUTH).mean(1) * 100.0


def fused_predictions(members: list[str], method: str) -> np.ndarray:
    fused = sum(transform(load_scores(name), method) for name in members)
    return fused.argmax(2)


def t_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    half_width = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return [float(values.mean() - half_width), float(values.mean() + half_width)]


def committee_report(members: list[str], method: str, baseline: str) -> dict:
    member_correct = np.stack([predictions(name) == TRUTH for name in members])
    fused_correct = fused_predictions(members, method) == TRUTH
    baseline_fold = accuracies(predictions(baseline))
    fused_fold = fused_correct.mean(1) * 100.0
    delta = fused_fold - baseline_fold
    oracle = member_correct.any(0)
    correct_count = member_correct.sum(0)
    leave_one_out = {}
    for removed in members:
        subset = [name for name in members if name != removed]
        leave_one_out[removed] = float(
            fused_correct.mean() * 100.0
            - (fused_predictions(subset, method) == TRUTH).mean() * 100.0
        )
    return {
        "members": members,
        "transform": method,
        "solo_mean": {
            name: float((predictions(name) == TRUTH).mean() * 100.0) for name in members
        },
        "top1_by_subject": fused_fold.tolist(),
        "mean_top1": float(fused_fold.mean()),
        "oracle_top1": float(oracle.mean() * 100.0),
        "leave_one_out_drop": leave_one_out,
        "unique_correct_trials": {
            name: int(
                np.sum(
                    member_correct[index]
                    & ~np.delete(member_correct, index, axis=0).any(0)
                )
            )
            for index, name in enumerate(members)
        },
        "accuracy_given_number_of_correct_members": {
            str(count): {
                "trials": int(np.sum(correct_count == count)),
                "ensemble_top1": float(fused_correct[correct_count == count].mean() * 100.0),
            }
            for count in range(len(members) + 1)
            if np.any(correct_count == count)
        },
        "versus_baseline": {
            "baseline": baseline,
            "fold_delta": delta.tolist(),
            "mean_delta": float(delta.mean()),
            "mean_delta_t95": t_interval(delta),
            "subjects_improved": int(np.sum(delta > 0)),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
        },
    }


def pair_report(first: str, second: str) -> dict:
    a = load_scores(first)
    b = load_scores(second)
    a_correct = predictions(first) == TRUTH
    b_correct = predictions(second) == TRUTH
    pair_top1 = accuracies((a + b).argmax(2)).mean()
    best_solo = max(a_correct.mean(), b_correct.mean()) * 100.0
    return {
        "score_correlation": float(np.corrcoef(row_z(a).ravel(), row_z(b).ravel())[0, 1]),
        "correctness_correlation": float(
            np.corrcoef(a_correct.ravel(), b_correct.ravel())[0, 1]
        ),
        "raw_pair_top1": float(pair_top1),
        "gain_over_better_solo": float(pair_top1 - best_solo),
        "oracle_top1": float((a_correct | b_correct).mean() * 100.0),
        "double_fault": float((~a_correct & ~b_correct).mean() * 100.0),
    }


def category_report(pairs: list[tuple[str, str]]) -> dict:
    rows = [pair_report(*pair) for pair in pairs]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def binary_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Mann-Whitney form of ROC AUC, with average ranks for ties."""
    labels = labels.astype(bool).ravel()
    values = values.ravel()
    positive = int(labels.sum())
    negative = len(labels) - positive
    positive_rank_sum = rankdata(values, method="average")[labels].sum()
    return float(
        (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    )


def confidence_routing_report(members: list[str]) -> dict:
    """Can within-row confidence identify the correct expert for a query?"""
    raw = np.stack([load_scores(name) for name in members])
    standardized = np.stack([row_z(scores) for scores in raw])
    correct = raw.argmax(3) == TRUTH
    sorted_scores = np.sort(standardized, axis=3)
    margin = sorted_scores[:, :, :, -1] - sorted_scores[:, :, :, -2]
    gap5 = sorted_scores[:, :, :, -1] - sorted_scores[:, :, :, -5:].mean(3)
    margin_member = margin.argmax(0)
    margin_scores = np.take_along_axis(
        raw, margin_member[None, :, :, None], axis=0
    )[0]
    gap5_member = gap5.argmax(0)
    gap5_scores = np.take_along_axis(
        raw, gap5_member[None, :, :, None], axis=0
    )[0]
    return {
        "margin_correctness_auc_by_member": {
            name: binary_auc(correct[index], margin[index])
            for index, name in enumerate(members)
        },
        "margin_correctness_auc_mean": float(
            np.mean(
                [
                    binary_auc(correct[index], margin[index])
                    for index in range(len(members))
                ]
            )
        ),
        "gap5_correctness_auc_by_member": {
            name: binary_auc(correct[index], gap5[index])
            for index, name in enumerate(members)
        },
        "gap5_correctness_auc_mean": float(
            np.mean(
                [binary_auc(correct[index], gap5[index]) for index in range(len(members))]
            )
        ),
        "max_margin_expert_top1": float(
            (margin_scores.argmax(2) == TRUTH).mean() * 100.0
        ),
        "max_gap5_expert_top1": float(
            (gap5_scores.argmax(2) == TRUTH).mean() * 100.0
        ),
        "raw_uniform_top1": float((raw.sum(0).argmax(2) == TRUTH).mean() * 100.0),
        "row_z_uniform_top1": float(
            (standardized.sum(0).argmax(2) == TRUTH).mean() * 100.0
        ),
        "oracle_top1": float(correct.any(0).mean() * 100.0),
    }


def fixed_k_selection_report(members: list[str], size: int, method: str) -> dict:
    """All-fold and leave-one-subject-out selection within one declared pool."""
    rows = []
    for combination in itertools.combinations(members, size):
        fold_accuracy = accuracies(fused_predictions(list(combination), method))
        rows.append((combination, fold_accuracy))
    selected, selected_fold = max(rows, key=lambda row: float(row[1].mean()))
    held_out, selected_per_held_out = [], []
    for held in range(len(SUBJECTS)):
        chosen, chosen_fold = max(
            rows, key=lambda row: float(np.delete(row[1], held).mean())
        )
        held_out.append(float(chosen_fold[held]))
        selected_per_held_out.append(list(chosen))
    return {
        "pool": members,
        "size": size,
        "transform": method,
        "all_fold_selected_members": list(selected),
        "all_fold_selected_top1": float(selected_fold.mean()),
        "all_fold_selected_by_subject": selected_fold.tolist(),
        "nested_lofo_top1": float(np.mean(held_out)),
        "nested_lofo_by_subject": held_out,
        "distinct_nested_member_sets": len(
            {tuple(combination) for combination in selected_per_held_out}
        ),
        "nested_member_sets": selected_per_held_out,
    }


def main() -> None:
    seeds = ["p3300", "p3301", "p3302"]
    internvit_encoders = ["atm_iv", "tsconv_iv", "conf_iv", "eegnet_iv", "eegproj_iv"]
    atm_targets = ["atm_iv", "atm_eva", "atm_bigg", "atm_vith"]
    tsconv_targets = [
        "tsconv_iv",
        "tsconv_eva",
        "tsconv_bigg",
        "tsconv_dino",
        "tsconv_vitb",
        "tsconv_vith",
    ]
    cross_axis_pairs = [
        (atm, tsconv)
        for atm in atm_targets
        for tsconv in tsconv_targets
        if (atm, tsconv) != ("atm_iv", "tsconv_iv")
    ]
    val_atm = [
        "atm25_valcon",
        "atm_iv_valcon",
        "atm31_valcon",
        "atm33_valcon",
        "atm35_valcon",
    ]
    val_tsconv = [
        "iv25_valcon",
        "iv28_valcon",
        "iv31_valcon",
        "iv33_valcon",
        "iv35_valcon",
    ]
    concept_val_pool = val_atm + val_tsconv
    test_selected_control_pool = [name + "_ctl" for name in concept_val_pool]

    val_control_deltas = {}
    for name in val_atm + val_tsconv:
        val_accuracy = accuracies(predictions(name))
        control_accuracy = accuracies(predictions(name + "_ctl"))
        val_control_deltas[name] = {
            "concept_val_mean": float(val_accuracy.mean()),
            "test_selected_control_mean": float(control_accuracy.mean()),
            "mean_delta": float((control_accuracy - val_accuracy).mean()),
        }

    report = {
        "scope": "read-only analysis of existing score dumps; no model execution",
        "committees": {
            "seed_only_three": committee_report(seeds, "signed_power_1.25", "p3300"),
            "reference_four": committee_report(
                ["atm_iv", "ge100", "tsconv_eva", "tsconv_vith"],
                "raw",
                "ge100",
            ),
            "frozen_five": committee_report(
                ["atm_iv", "ge100", "iv33", "tsconv_bigg", "atm_vith"],
                "signed_power_1.25",
                "ge100",
            ),
            "squeezeformer_four": committee_report(
                ["atm_vith", "atm_iv_group_e75", "iv33_group_e75", "sqf28"],
                "row_z",
                "ge100",
            ),
            "concept_validation_pair": committee_report(
                ["atm33_valcon", "iv28_valcon"],
                "signed_power_1.25",
                "atm_iv_valcon",
            ),
        },
        "focused_pool_selection": fixed_k_selection_report(
            [
                "atm_iv",
                "ge100",
                "tsconv_eva",
                "tsconv_vith",
                "iv33",
                "tsconv_bigg",
                "atm_vith",
                "atm_iv_group_e75",
                "iv33_group_e75",
                "sqf28",
            ],
            size=4,
            method="row_z",
        ),
        "concept_validation_selection": {
            "k2_signed_power": fixed_k_selection_report(
                concept_val_pool, size=2, method="signed_power_1.25"
            ),
            "k4_raw": fixed_k_selection_report(
                concept_val_pool, size=4, method="raw"
            ),
            "k5_row_z": fixed_k_selection_report(
                concept_val_pool, size=5, method="row_z"
            ),
        },
        "matched_test_selected_control_selection": {
            "k4_raw": fixed_k_selection_report(
                test_selected_control_pool, size=4, method="raw"
            ),
            "k5_signed_power": fixed_k_selection_report(
                test_selected_control_pool, size=5, method="signed_power_1.25"
            ),
        },
        "confidence_routing": {
            "pooled_router_members": confidence_routing_report(
                [
                    "atm_iv",
                    "iv33",
                    "atm_vith_group_e75",
                    "bigg27_group_e75",
                    "iv_vith_dino_aux025_group_e75",
                ]
            ),
            "squeezeformer_four": confidence_routing_report(
                ["atm_vith", "atm_iv_group_e75", "iv33_group_e75", "sqf28"]
            ),
        },
        "pair_categories": {
            "seed_only": category_report(list(itertools.combinations(seeds, 2))),
            "same_target_different_encoder": category_report(
                list(itertools.combinations(internvit_encoders, 2))
            ),
            "same_atm_encoder_different_target": category_report(
                list(itertools.combinations(atm_targets, 2))
            ),
            "same_tsconv_encoder_different_target": category_report(
                list(itertools.combinations(tsconv_targets, 2))
            ),
            "different_encoder_and_target": category_report(cross_axis_pairs),
            "atm_different_internvit_depth": category_report(
                list(itertools.combinations(val_atm, 2))
            ),
            "tsconv_different_internvit_depth": category_report(
                list(itertools.combinations(val_tsconv, 2))
            ),
        },
        "checkpoint_selection_controls": val_control_deltas,
        "checkpoint_selection_mean_delta": float(
            np.mean([row["mean_delta"] for row in val_control_deltas.values()])
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
