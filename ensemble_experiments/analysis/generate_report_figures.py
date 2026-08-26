"""Generate the presentation figures used by ensemble_results_report.md.

Historical aggregate values are copied from the audited tables in the report. Pairwise
plots are regenerated from the frozen 45-model CSV, while intervention plots aggregate
the completed per-subject result.csv files directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "ensemble_experiments" / "analysis"
OUT = ANALYSIS / "report_figures"
RESULTS = ROOT / "results" / "things_eeg" / "decorrelated_models"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def label_bars(axis: plt.Axes, fmt: str = "{:.2f}") -> None:
    for container in axis.containers:
        axis.bar_label(container, fmt=fmt, padding=3, fontsize=9)


def load_completed_arm(relative: str) -> pd.DataFrame:
    arm_root = RESULTS / relative / "seed3300_3301"
    paths = sorted(arm_root.glob("*-sub-*/result.csv"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected 10 completed subjects for {relative}, found {len(paths)}")
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise RuntimeError(f"Expected one result row in {path}, found {len(frame)}")
        row = frame.iloc[0].copy()
        row["subject"] = int(path.parent.name.rsplit("-", 1)[-1])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def sem95(values: pd.Series) -> float:
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def seed_ensembles() -> None:
    labels = ["Seed 3300", "Seed 3301", "Seed 3302", "Ensemble"]
    tsconv = [35.05, 34.85, 34.60, 37.10]
    atm = [35.20, 34.40, 35.15, 37.45]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 5.2), layout="constrained")
    axis.bar(x - width / 2, tsconv, width, label="TSConv", color="#4C78A8")
    axis.bar(x + width / 2, atm, width, label="ATM", color="#F58518")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_ylim(30, 39)
    axis.set_title("Seed ensembles")
    axis.legend()
    label_bars(axis)
    save(fig, "seed_ensembles")


def depth_scaling() -> None:
    k = np.arange(1, 6)
    fig, axis = plt.subplots(figsize=(7.4, 5.2), layout="constrained")
    axis.plot(k, [36.50, 38.65, 39.70, 39.90, 40.15], "o-", lw=2,
              label="ATM", color="#F58518")
    axis.plot(k, [36.10, 39.75, 41.15, 41.35, 41.40], "o-", lw=2,
              label="TSConv", color="#4C78A8")
    axis.set_xticks(k)
    axis.set_xlabel("Number of members")
    axis.set_ylabel("Best top-1 accuracy (%)")
    axis.set_title("Depth ensemble scaling")
    axis.legend()
    save(fig, "depth_ensemble_scaling")


def diversity_progression() -> None:
    labels = [
        "Seeds",
        "ATM\nlayers",
        "TSConv\nlayers",
        "Same target\n28 + 28",
        "Best targets\n25 + 33",
    ]
    correlations = [0.956, 0.915, 0.918, 0.851, 0.810]
    gains = [1.58, 2.60, 3.20, 5.00, 8.90]
    colors = ["#9C9C9C", "#F58518", "#4C78A8", "#76B7B2", "#54A24B"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout="constrained")
    axes[0].bar(labels, correlations, color=colors)
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_ylabel("Mean score correlation")
    axes[0].set_title("Score correlation")
    axes[0].tick_params(axis="x", labelsize=8)
    label_bars(axes[0], "%.3f")
    axes[1].bar(labels, gains, color=colors)
    axes[1].set_ylim(0, 10)
    axes[1].set_ylabel("Mean pair gain (pp)")
    axes[1].set_title("Pair gain")
    axes[1].tick_params(axis="x", labelsize=8)
    label_bars(axes[1], "%.2f")
    fig.suptitle("Diversity progression", fontsize=15)
    save(fig, "diversity_progression")


def cross_encoder_pair() -> None:
    labels = [
        "ATM-28",
        "TSConv-28",
        "ATM-28 +\nTSConv-28",
        "ATM-25 +\nTSConv-33",
    ]
    values = [35.20, 36.10, 40.65, 43.55]
    fig, axis = plt.subplots(figsize=(8.1, 5.1), layout="constrained")
    axis.bar(
        labels,
        values,
        color=["#F58518", "#4C78A8", "#76B7B2", "#54A24B"],
        width=0.65,
    )
    axis.set_ylim(30, 45)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_title("ATM and TSConv")
    label_bars(axis)
    save(fig, "atm_tsconv_pair")


def pairwise_complementarity() -> None:
    frame = pd.read_csv(ANALYSIS / "pair_complementarity_testselected.csv")
    x = frame["score_correlation"].to_numpy()
    color = frame["mean_solo_top1"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained", sharex=True)
    points = None
    for axis, column, title, ylabel in [
        (axes[0], "gain_over_mean", "Gain over pair mean", "Gain over pair mean (pp)"),
        (axes[1], "gain_over_best", "Gain over stronger member", "Gain over pair max (pp)"),
    ]:
        y = frame[column].to_numpy()
        points = axis.scatter(x, y, c=color, cmap="viridis", s=24, alpha=0.70,
                              linewidths=0)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axis.plot(line_x, slope * line_x + intercept, color="black", lw=1.4)
        axis.axhline(0, color="0.5", ls="--", lw=0.9)
        axis.set_xlabel("Pairwise row-z score correlation")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    fig.colorbar(points, ax=axes, label="Mean solo top-1 of pair (%)")
    fig.suptitle("Pairwise complementarity", fontsize=15)
    save(fig, "pairwise_complementarity")


def sota_comparison() -> None:
    labels = [
        "Ensembling",
        "SAGE (ours, 2026)",
        "SUP-MCRL (2026)",
        "ViEEG (2026)",
        "Shallow Alignment (2026)",
        "SIMON (2026)",
        "NeuroBridge (2025)",
        "NeuroCLIP (2025)",
        "Neural-MCRL (2024)",
        "NICE (2024)",
    ]
    # Top-5 was recomputed from the same four row-z score dumps that yield 48.20 top-1.
    top1 = [48.20, 35.90, 24.00, 22.90, 21.80, 19.60, 19.00, 17.00, 14.00, 6.20]
    top5 = [77.60, 69.20, 52.90, 51.40, 49.40, 49.90, 45.90, 40.30, 34.30, 21.40]
    colors = ["#029E73", "#4C78A8"] + ["#B8B8B8"] * (len(labels) - 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.7), sharey=True,
                             layout="constrained")
    y = np.arange(len(labels))
    for axis, values, title in zip(axes, [top1, top5], ["Top-1", "Top-5"]):
        bars = axis.barh(y, values, color=colors, height=0.68)
        axis.set_xlim(0, 85)
        axis.set_xlabel("Accuracy (%)")
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=9)
    save(fig, "sota_comparison")


def predictive_metrics() -> None:
    summary = json.loads((ANALYSIS / "pair_complementarity_testselected.json").read_text())
    associations = summary["metric_association_with_gain_over_mean"]
    entries = [
        ("Score correlation", associations["score_correlation"]["spearman_rho"]),
        ("Margin correlation", associations["margin_correlation"]["spearman_rho"]),
        ("Correctness correlation", associations["correctness_correlation"]["spearman_rho"]),
        ("Wrong-winner agreement", associations["wrong_winner_agreement"]["spearman_rho"]),
        ("Prediction disagreement", associations["prediction_disagreement"]["spearman_rho"]),
        ("Oracle headroom", associations["oracle_headroom_over_best"]["spearman_rho"]),
    ]
    entries.sort(key=lambda item: item[1])
    labels, values = zip(*entries)
    colors = ["#4C78A8" if value < 0 else "#54A24B" for value in values]
    fig, axis = plt.subplots(figsize=(8.0, 5.3), layout="constrained")
    axis.barh(labels, values, color=colors)
    axis.axvline(0, color="0.35", lw=0.9)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Spearman correlation with gain over pair mean")
    axis.set_title("Pairwise complementarity metrics")
    for index, value in enumerate(values):
        axis.text(value + (0.025 if value >= 0 else -0.025), index, f"{value:.3f}",
                  va="center", ha="left" if value >= 0 else "right", fontsize=9)
    save(fig, "pairwise_complementarity_metrics")


def ensemble_scaling() -> None:
    k = np.arange(1, 7)
    values = np.array([37.05, 44.10, 46.55, 48.20, 48.85, 49.50])
    fig, axis = plt.subplots(figsize=(7.4, 5.2), layout="constrained")
    axis.plot(k, values, "o-", color="#4C78A8", lw=2.3, ms=7)
    for x, value in zip(k, values):
        axis.annotate(f"{value:.2f}", (x, value), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=9)
    axis.set_xticks(k)
    axis.set_ylim(35.5, 51)
    axis.set_xlabel("Number of members")
    axis.set_ylabel("Best top-1 accuracy (%)")
    axis.set_title("Ensemble scaling")
    save(fig, "ensemble_scaling")


def committee_ablation() -> None:
    labels = ["Full", "Without\nATM/ViT-H", "Without\nATM/IV-28", "Without\nTSConv/IV-33", "Without\nSqueezeformer"]
    values = [48.20, 45.30, 45.65, 44.75, 44.95]
    colors = ["#54A24B"] + ["#B8B8B8"] * 4
    fig, axis = plt.subplots(figsize=(9.1, 5.2), layout="constrained")
    axis.bar(labels, values, color=colors)
    axis.set_ylim(42, 50)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_title("Committee ablation")
    label_bars(axis)
    save(fig, "committee_ablation")


def validation_protocol() -> None:
    labels = ["ATM/IV-28", "TSConv/IV-33", "TSConv/BigG", "Pair", "Triple"]
    loso = [29.15, 30.75, 30.10, 37.70, 40.20]
    valcon = [33.40, 33.30, 30.10, 41.40, 43.15]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    axis.bar(x - width / 2, loso, width, label="LOSO validation", color="#9C9C9C")
    axis.bar(x + width / 2, valcon, width, label="ValCon", color="#4C78A8")
    axis.set_xticks(x, labels)
    axis.set_ylim(25, 46)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_title("Validation protocol")
    axis.legend()
    label_bars(axis)
    save(fig, "validation_protocol")


def validation_scaling() -> None:
    k = np.arange(1, 7)
    valcon = [33.40, 41.40, 43.75, 44.90, 45.15, 45.65]
    test = [36.35, 44.60, 46.00, 47.75, 48.40, 48.90]
    fig, axis = plt.subplots(figsize=(7.6, 5.2), layout="constrained")
    axis.plot(k, valcon, "o-", color="#4C78A8", lw=2, label="ValCon")
    axis.plot(k, test, "o-", color="#F58518", lw=2,
              label="Matched test-selected")
    axis.axhline(31.5, color="#B00020", lw=1.8, ls=":",
                 label="SAGE LOSO-val accuracy")
    axis.set_xticks(k)
    axis.set_xlabel("Number of members")
    axis.set_ylabel("Best top-1 accuracy (%)")
    axis.set_title("Validation ensemble scaling")
    axis.set_ylim(30, 50)
    axis.legend()
    save(fig, "validation_ensemble_scaling")


def intervention_data() -> tuple[dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    direct = {
        0.00: load_completed_arm("heterogeneous_wave/decor_atm_tsconv_l000_b0_bs512"),
        0.10: load_completed_arm("heterogeneous_wave/decor_atm_tsconv_l010_b0_bs512"),
        0.25: load_completed_arm("heterogeneous_wave/decor_atm_tsconv_l025_b0_bs512"),
    }
    rescue = {
        0.00: direct[0.00],
        0.10: load_completed_arm("heterogeneous_wave/rescue_atm_tsconv_r1_g010_b0_bs512"),
        0.30: load_completed_arm("heterogeneous_wave/rescue_atm_tsconv_r2_g030_b0_bs512"),
        0.50: load_completed_arm("heterogeneous_wave/rescue_atm_tsconv_r3_g050_b0_bs512"),
    }
    return direct, rescue


def intervention_accuracy(direct: dict[float, pd.DataFrame],
                          rescue: dict[float, pd.DataFrame]) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 5.3), layout="constrained")
    for frames, label, color in [
        (direct, "Direct decorrelation", "#4C78A8"),
        (rescue, "Rescue", "#E45756"),
    ]:
        coefficients = np.array(list(frames))
        means = np.array([frame["pair_top1"].mean() for frame in frames.values()])
        errors = np.array([sem95(frame["pair_top1"]) for frame in frames.values()])
        axis.errorbar(coefficients, means, yerr=errors, marker="o", ms=7, lw=2,
                      capsize=4, label=label, color=color)
        for coefficient, mean in zip(coefficients, means):
            axis.annotate(f"{mean:.2f}", (coefficient, mean), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=9)
    axis.set_xlabel("Loss coefficient")
    axis.set_ylabel("Pair top-1 accuracy (%)")
    axis.set_title("Intervention pair accuracy")
    axis.legend()
    save(fig, "intervention_pair_accuracy")


def intervention_trajectories(direct: dict[float, pd.DataFrame],
                              rescue: dict[float, pd.DataFrame]) -> None:
    natural = pd.read_csv(ANALYSIS / "pair_complementarity_testselected.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), layout="constrained",
                             sharex=True)
    for axis, gain_column, title, ylabel in [
        (axes[0], "gain_over_mean", "Gain over pair mean", "Gain over pair mean (pp)"),
        (axes[1], "gain_over_best", "Gain over pair max", "Gain over pair max (pp)"),
    ]:
        axis.scatter(natural["score_correlation"], natural[gain_column],
                     color="0.78", s=15, alpha=0.32, linewidths=0,
                     label="Natural pairs")
        for frames, label, color, symbol in [
            (direct, "Direct decorrelation", "#4C78A8", "λ"),
            (rescue, "Rescue", "#E45756", "γ"),
        ]:
            x = [frame["score_correlation"].mean() for frame in frames.values()]
            y = [frame[gain_column].mean() for frame in frames.values()]
            axis.plot(x, y, "o-", color=color, lw=2.0, ms=6, label=label)
            for coefficient, px, py in zip(frames, x, y):
                if symbol == "γ" and coefficient == 0:
                    continue
                if symbol == "λ":
                    offsets = {0.00: (6, -12), 0.10: (-27, 6), 0.25: (5, 5)}
                    text_label = "control" if coefficient == 0 else f"λ={coefficient:g}"
                else:
                    offsets = {0.10: (7, -14), 0.30: (7, 5), 0.50: (7, 0)}
                    text_label = f"γ={coefficient:g}"
                axis.annotate(
                    text_label,
                    (px, py),
                    xytext=offsets[coefficient],
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75,
                          "pad": 0.4},
                )
        axis.set_xlabel("Pairwise row-z score correlation")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle("Intervention trajectories", fontsize=15)
    save(fig, "intervention_trajectories")


def complementarity_utilization(direct: dict[float, pd.DataFrame],
                                rescue: dict[float, pd.DataFrame]) -> None:
    settings = {
        "Control": direct[0.00],
        "Direct λ=.25": direct[0.25],
        "Rescue γ=.30": rescue[0.30],
    }
    best = np.array([
        frame[["member_a_top1", "member_b_top1"]].max(axis=1).mean()
        for frame in settings.values()
    ])
    pair = np.array([frame["pair_top1"].mean() for frame in settings.values()])
    oracle = np.array([frame["oracle_top1"].mean() for frame in settings.values()])
    realized = pair - best
    unused = oracle - pair
    x = np.arange(len(settings))
    fig, axis = plt.subplots(figsize=(7.8, 5.4), layout="constrained")
    axis.bar(x, best, label="Stronger member", color="#9C9C9C")
    axis.bar(x, realized, bottom=best, label="Realized headroom", color="#54A24B")
    axis.bar(x, unused, bottom=pair, label="Unrealized headroom", color="#E0E0E0",
             edgecolor="#A0A0A0")
    for index, (pair_value, oracle_value) in enumerate(zip(pair, oracle)):
        axis.text(index, pair_value - 0.3, f"Pair {pair_value:.2f}", ha="center",
                  va="top", fontsize=9)
        axis.text(index, oracle_value + 0.3, f"Oracle {oracle_value:.2f}", ha="center",
                  va="bottom", fontsize=9)
    axis.set_xticks(x, settings)
    axis.set_ylim(0, 55)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_title("Complementarity utilization")
    axis.legend(loc="lower right")
    save(fig, "complementarity_utilization")


def write_intervention_summary(direct: dict[float, pd.DataFrame],
                               rescue: dict[float, pd.DataFrame]) -> None:
    rows = []
    for family, frames in [("direct_decorrelation", direct), ("rescue", rescue)]:
        for coefficient, frame in frames.items():
            row = {"family": family, "coefficient": coefficient, "subjects": len(frame)}
            for column in [
                "member_a_top1", "member_b_top1", "pair_top1", "gain_over_best",
                "gain_over_mean", "score_correlation", "oracle_top1",
                "oracle_headroom_realized",
            ]:
                row[column] = float(frame[column].mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "intervention_summary.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    seed_ensembles()
    depth_scaling()
    diversity_progression()
    cross_encoder_pair()
    pairwise_complementarity()
    sota_comparison()
    predictive_metrics()
    ensemble_scaling()
    committee_ablation()
    validation_protocol()
    validation_scaling()
    direct, rescue = intervention_data()
    intervention_accuracy(direct, rescue)
    intervention_trajectories(direct, rescue)
    complementarity_utilization(direct, rescue)
    write_intervention_summary(direct, rescue)
    print(f"Wrote report figures to {OUT}")


if __name__ == "__main__":
    main()
