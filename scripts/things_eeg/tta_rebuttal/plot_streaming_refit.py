"""Plot the causal streaming re-fit curve (run_streaming_refit.py)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# categorical slots 1,2,3,7 of the validated default palette; plain is a neutral reference
COLORS = {"plain": "#52514e", "saw": "#1baf7a", "balanced": "#eb6834",
          "free": "#4a3aa7", "free_ramp": "#2a78d6"}
LABELS = {"plain": "no adaptation", "saw": "whitening only",
          "balanced": "published config ($\\rho{=}1$)", "free": "free marginal ($\\rho{=}0$)",
          "free_ramp": "free + damped rotation"}
ORDER = ["plain", "saw", "balanced", "free", "free_ramp"]
REPS, SOA = 20, 0.2  # one query = 20 repetitions at the benchmark's 0.2 s SOA.
# NB: this converts to stimulus-presentation time only. It excludes breaks and setup, and it
# assumes a deployment paradigm whose per-query cost matches the benchmark's.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/things_eeg/tta_rebuttal/streaming_refit/subject_results.csv")
    p.add_argument("--out", default="results/things_eeg/tta_rebuttal/streaming_refit/streaming_refit.png")
    p.add_argument("--t_max", type=int, default=700, help="drop the tail where the held-out remainder is tiny")
    args = p.parse_args()

    d = pd.read_csv(args.results)
    d = d[d.t <= args.t_max]
    # average seeds within a subject first, so the band is the spread across subjects
    per_subj = d.groupby(["method", "t", "subject"]).top1.mean().reset_index()
    mean = per_subj.groupby(["method", "t"]).top1.mean().unstack(0)
    std = per_subj.groupby(["method", "t"]).top1.std().unstack(0)
    delta = mean[ORDER[1:]].sub(mean["plain"], axis=0)

    # per-subject deltas for the winner: mean crossover vs all-subject crossover
    w = per_subj.pivot_table(index=["subject", "t"], columns="method", values="top1")
    w_delta = (w["free_ramp"] - w["plain"]).unstack(0)
    ts = list(mean.index)
    cross_mean = next(t for i, t in enumerate(ts) if all(delta.loc[t2, "free_ramp"] > 0 for t2 in ts[i:]))
    # first checkpoint at which every subject is positive (one subject dips -0.33 at t=500,
    # so the strictly-sustained criterion would report a much later t on a noise blip)
    cross_all = next(t for t in ts if (w_delta.loc[t] > 0).all())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlim(10, args.t_max)
        ax.set_xticks([10, 20, 50, 100, 200, 500])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("trials seen by the calibration (log scale)")
        top = ax.secondary_xaxis("top", functions=(lambda x: x * REPS * SOA / 60, lambda m: m * 60 / (REPS * SOA)))
        top.set_xticks([1, 2, 5, 10, 20, 40])
        top.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        top.minorticks_off()
        top.set_xlabel("minutes of stimulus presentation", labelpad=6)

    for m in ORDER:
        ax1.plot(mean.index, mean[m], color=COLORS[m], lw=2,
                 ls="--" if m == "plain" else "-", marker="o", ms=3, label=LABELS[m])
    ax1.set_ylabel("top-1 accuracy (%), per-trial, held-out")
    ax1.set_title("Decoding the trials that have not arrived yet", fontsize=10.5, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax2.axhline(0, color=COLORS["plain"], lw=1.2, ls="--")
    ax2.fill_between(mean.index, delta["free_ramp"] - std["free_ramp"], delta["free_ramp"] + std["free_ramp"],
                     color=COLORS["free_ramp"], alpha=0.13, lw=0)
    for m in ORDER[1:]:
        ax2.plot(mean.index, delta[m], color=COLORS[m], lw=2, marker="o", ms=3, label=LABELS[m])
    # stagger the two callouts (one low, one high) so they clear the curves and each other
    for t, txt, y, va in [(cross_mean, f"mean crossover\n{t_min(cross_mean)}", -13, "top"),
                          (cross_all, f"every subject positive\n{t_min(cross_all)}", 21, "top")]:
        ax2.axvline(t, color=COLORS["plain"], lw=0.9, ls=":", alpha=0.8)
        ax2.annotate(txt, xy=(t * 1.06, y), fontsize=8, color="#0b0b0b", va=va, ha="left")
    ax2.set_ylabel("top-1 gain over no adaptation (pts)")
    ax2.set_title("Warm-up cost, then payoff", fontsize=10.5, loc="left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}\n  mean crossover t={cross_mean} ({t_min(cross_mean)}), all-subject t={cross_all} ({t_min(cross_all)})")


def t_min(t):
    return f"{t} trials, {t * REPS * SOA / 60:.0f} min presented"


if __name__ == "__main__":
    main()
