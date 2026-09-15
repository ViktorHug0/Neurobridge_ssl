"""Four repetition levels side by side, one panel each (run_streaming_refit.py --geodesic).

Shared y-axis so the accuracy levels compare directly across panels. Each panel keeps its own
trial range, because the pools differ (3200/1600/800/400), but the minutes axis on top ends at
roughly 50 in every panel: pool x repetitions is always 200 x 80 = 16000 trials, so all four
conditions spend the SAME recording budget and differ only in how they spend it.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
PLAIN = "#52514e"
SOA = 0.2
RUNS = [(5, 3100), (10, 1550), (20, 750), (40, 350)]


def panel_data(results_dir, reps, t_max):
    d = pd.read_csv(os.path.join(results_dir, f"reps{reps}", "subject_results.csv"))
    d = d[d.t <= t_max]
    per_subj = d.groupby(["method", "t", "subject"]).top1.mean().reset_index()
    return per_subj.groupby(["method", "t"]).top1.mean().unstack(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/things_eeg/tta_rebuttal/streaming_geodesic_reps")
    p.add_argument("--out", default="results/things_eeg/tta_rebuttal/streaming_geodesic_reps/streaming_reps_grid.png")
    p.add_argument("--reps", nargs="+", type=int, default=[r for r, _ in RUNS])
    p.add_argument("--t_max", nargs="+", type=int, default=[t for _, t in RUNS],
                   help="last checkpoint per panel, in the same order as --reps")
    p.add_argument("--budget", default="~50 min", help="recording budget quoted in the header")
    args = p.parse_args()

    runs = list(zip(args.reps, args.t_max))
    means = [panel_data(args.results, r, t) for r, t in runs]
    alphas = sorted(float(c[5:]) for c in means[0].columns if c.startswith("alpha"))
    cols = [f"alpha{a:g}" for a in alphas]
    ramp = [LinearSegmentedColormap.from_list("r", RAMP)(x) for x in np.linspace(0, 1, len(alphas))]
    lo = min(m[cols + ["plain"]].values.min() for m in means)
    hi = max(m[cols + ["plain"]].values.max() for m in means)

    fig, axes = plt.subplots(1, len(runs), figsize=(3.85 * len(runs), 4.2), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.90, top=0.78, bottom=0.145, wspace=0.09)

    for ax, m, (reps, t_max) in zip(axes, means, runs):
        ts = list(m.index)
        ax.set_xscale("log")
        ax.set_xlim(min(ts), t_max)
        ax.set_ylim(lo - 2, hi + 2)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(ts, m["plain"], color=PLAIN, lw=2, ls="--", zorder=3)
        for i, c in enumerate(cols):
            ax.plot(ts, m[c], color=ramp[i], lw=2.4 if c == cols[-1] else 1.5)
        ax.set_title(f"{reps} repetitions per query", fontsize=10.5, loc="left")
        ax.set_xlabel("trials seen")
        top = ax.secondary_xaxis("top", functions=(lambda x, r=reps: x * r * SOA / 60,
                                                   lambda mn, r=reps: mn * 60 / (r * SOA)))
        top.set_xticks([2, 5, 10, 20, 50])
        top.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        top.minorticks_off()
        top.tick_params(labelsize=8)

    axes[0].set_ylabel("top-1 accuracy (%), per-trial, held-out")
    axes[0].plot([], [], color=PLAIN, lw=2, ls="--", label="no adaptation")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.478, 0.955,
             f"minutes of stimulus presentation (every panel spends the same {args.budget} budget)",
             ha="center", fontsize=9.5, color="#3d3d3a")

    cax = fig.add_axes([0.915, 0.17, 0.011, 0.58])
    cmap = ListedColormap(ramp)
    edges = np.arange(alphas[0] - 0.05, alphas[-1] + 0.06, 0.1)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(edges, cmap.N), cmap=cmap),
                      cax=cax, ticks=alphas)
    cb.set_label("rotation damping $\\alpha$ (geodesic)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
