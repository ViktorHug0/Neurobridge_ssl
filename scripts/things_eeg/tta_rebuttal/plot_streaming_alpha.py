"""Plot the causal streaming alpha sweep (run_streaming_refit.py --alphas ...).

alpha is ordinal, so the ten lines take a single-hue light->dark ramp rather than ten
categorical hues. Ten steps is more than one hue can separate (the ordinal validator fails
adjacent dL at 0.047), so colour carries ORDER only: the colourbar and the selective end
labels carry identity. If you need every alpha individually readable, use the heatmap form.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

# blue ordinal ramp, steps 250..700 (light end clears 2:1 on the light surface).
# Resampled to however many alphas the run swept, so the endpoints (and their contrast)
# hold whether the sweep has 10 steps or 11.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
PLAIN = "#52514e"
INK = "#0b0b0b"
SOA = 0.2  # stimulus-presentation time only; see run_streaming_refit.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/things_eeg/tta_rebuttal/streaming_alpha_sweep/subject_results.csv")
    p.add_argument("--out", default="results/things_eeg/tta_rebuttal/streaming_alpha_sweep/streaming_alpha_sweep.png")
    p.add_argument("--t_max", type=int, default=700)
    p.add_argument("--reps", type=int, default=20, help="repetitions per query, for the time axis")
    p.add_argument("--single", action="store_true",
                   help="one panel: accuracy only, no adaptation reference plus one line per alpha")
    p.add_argument("--title", default=None, help="override the panel title")
    args = p.parse_args()

    d = pd.read_csv(args.results)
    d = d[d.t <= args.t_max]
    per_subj = d.groupby(["method", "t", "subject"]).top1.mean().reset_index()
    mean = per_subj.groupby(["method", "t"]).top1.mean().unstack(0)
    alphas = sorted((float(m[5:]) for m in mean.columns if m.startswith("alpha")))
    cols = [f"alpha{a:g}" for a in alphas]
    delta = mean[cols].sub(mean["plain"], axis=0)
    ts = list(mean.index)

    ramp = [LinearSegmentedColormap.from_list("ramp", RAMP)(x) for x in np.linspace(0, 1, len(alphas))]
    cmap = ListedColormap(ramp)
    if args.single:
        fig, ax1 = plt.subplots(1, 1, figsize=(7.4, 4.5))
        fig.subplots_adjust(left=0.105, right=0.80, top=0.80, bottom=0.14)
        axes = (ax1,)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3))
        fig.subplots_adjust(left=0.07, right=0.865, top=0.80, bottom=0.145, wspace=0.24)
        axes = (ax1, ax2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(min(ts), args.t_max)
        ax.set_xticks([20, 50, 100, 200, 500])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("trials seen by the calibration (log scale)")
        top = ax.secondary_xaxis("top", functions=(lambda x: x * args.reps * SOA / 60,
                                                   lambda m: m * 60 / (args.reps * SOA)))
        top.set_xticks([1, 2, 5, 10, 20, 40])
        top.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        top.minorticks_off()
        top.set_xlabel("minutes of stimulus presentation", labelpad=6)

    ax1.plot(ts, mean["plain"], color=PLAIN, lw=2, ls="--", label="no adaptation")
    for i, c in enumerate(cols):
        ax1.plot(ts, mean[c], color=ramp[i], lw=2.6 if c == cols[-1] else 1.6)
    ax1.set_ylabel("top-1 accuracy (%), per-trial, held-out")
    ax1.set_title(args.title or "Damping strength vs evidence available", fontsize=10.5, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    if args.single:
        cax = fig.add_axes([0.825, 0.17, 0.019, 0.6])
        edges = np.arange(alphas[0] - 0.05, alphas[-1] + 0.06, 0.1)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(edges, cmap.N), cmap=cmap),
                          cax=cax, ticks=alphas)
        cb.set_label("rotation damping $\\alpha$", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=200, facecolor="white")
        print(f"wrote {args.out}")
        report(ts, cols, alphas, delta)
        return

    ax2.axhline(0, color=PLAIN, lw=1.4, ls="--")
    for i, c in enumerate(cols):
        ax2.plot(ts, delta[c], color=ramp[i], lw=2.6 if c == cols[-1] else 1.6)
    # ten one-hue steps cannot be told apart by colour, so identity rides on the colourbar
    # plus these labels; the published config is also drawn thicker.
    best = cols[int(np.argmax(delta.loc[ts[-1]].values))]
    labels = {cols[0]: f"$\\alpha$={alphas[0]:g}", best: f"$\\alpha$={float(best[5:]):g}  best",
              cols[-1]: f"$\\alpha$={alphas[-1]:g}  published"}
    placed = []
    for c, lbl in labels.items():
        y = delta.loc[ts[-1], c]
        while any(abs(y - q) < 1.4 for q in placed):
            y += 1.4
        placed.append(y)
        ax2.annotate(lbl, xy=(ts[-1] * 1.03, y), fontsize=8.5, color=INK, va="center", ha="left")
    ax2.set_ylabel("top-1 gain over no adaptation (pts)")
    ax2.set_title("The damping strength that pays moves with the evidence", fontsize=10.5, loc="left")
    ax2.set_xlim(min(ts), args.t_max * 2.1)

    cax = fig.add_axes([0.888, 0.17, 0.013, 0.6])
    edges = np.arange(alphas[0] - 0.05, alphas[-1] + 0.06, 0.1)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(edges, cmap.N), cmap=cmap),
                      cax=cax, ticks=alphas)
    cb.set_label("rotation damping $\\alpha$  (1.0 = published config)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")
    report(ts, cols, alphas, delta)


def report(ts, cols, alphas, delta):
    print("\ncrossover (first t where the arm stays above no-adaptation):")
    for i, c in enumerate(cols):
        cross = next((t for j, t in enumerate(ts) if all(delta.loc[t2, c] > 0 for t2 in ts[j:])), None)
        print(f"  alpha={alphas[i]:.1f}  {'t=' + str(cross) if cross else 'never':>8}   gain@{ts[-1]}={delta.loc[ts[-1], c]:+6.2f}")
    print("\nbest alpha by buffer size:")
    for t in ts[:: max(1, len(ts) // 12)]:
        row = delta.loc[t]
        print(f"  t={t:<4} best alpha={float(row.idxmax()[5:]):.1f}  gain={row.max():+6.2f}")


if __name__ == "__main__":
    main()
