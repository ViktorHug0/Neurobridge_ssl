"""Subject-scaling curve from neurips_challenge/scaling.sh runs (+ the cv4 runs as N = 15).

    scaling_summary.py [--out results/alljoined/track1/scaling/scaling_curve.png]
    scaling_summary.py --things      # THINGS-EEG2 subjects first / last (scaling.sh THINGS=...): scaling_things.png

A point = one run's best per-epoch sealed-rule top-5 (seeded half of the trials, ~40 repeats/image), mean over
the fold's 5 test subjects; the curve averages folds (and orders). Also printed without sub-15, on which
electrode-token models collapse.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/alljoined/track1")
LABELS = {"tsc": "TSConv", "plab": "pretrained LaBraM + gain", "reve": "scratch REVE (6 layers)"}
CV4 = {"tsc": "tsc", "plab": "plab_gain_s10"}  # reve (depth 6) runs its own N = 15  # the N = 15 point of each curve


def best(run_dir, drop=()):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "sealed_per_epoch.jsonl"))]
    return max(np.mean([v for s, v in r["by_subject"].items() if s not in drop]) for r in rows if r["fraction"] == 0.5)


def collect(model, drop=(), things=None):
    pts = defaultdict(list)  # N -> [run scores]
    if things:  # its own runs, plus the shared 25-subject run; THINGS last reuses the plain curve below N = 17
        for f in glob.glob(f"{R}/scaling/{model}/fold*/o*_things*/n*/*/*/track1_score.json"):
            n = int(f.split("/")[-4][1:])
            if f"_things{things}/" in f or n == 25:
                pts[n].append(best(os.path.dirname(f), drop))
        if things == "first":
            return dict(sorted(pts.items()))
    for f in glob.glob(f"{R}/scaling/{model}/fold*/o[0-9]/n*/*/*/track1_score.json"):
        pts[int(f.split("/")[-4][1:])].append(best(os.path.dirname(f), drop))
    for f in glob.glob(f"{R}/cv4/{CV4[model]}/fold[12]/*/*/track1_score.json") if model in CV4 else []:
        pts[15].append(best(os.path.dirname(f), drop))
    return dict(sorted(pts.items()))


def peaked_last(run_dir):
    """The run's best evaluation is its last one: still rising, so the point is a floor."""
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "sealed_per_epoch.jsonl"))]
    s = [np.mean(list(r["by_subject"].values())) for r in rows if r["fraction"] == 0.5]
    return len(s) > 1 and s[-1] > max(s[:-1])  # a tie at the end is a plateau, not a rise


def plot_things(out):
    import matplotlib.pyplot as plt
    for f in sorted(glob.glob(f"{R}/scaling/*/fold*/o*/n*/*/*/track1_score.json")):  # finished runs only
        if peaked_last(os.path.dirname(f)):
            print("UNDERTRAINED (best = last evaluation):", f.split("/scaling/")[1].rsplit("/", 3)[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, things in zip(axes, ("first", "last")):
        for model, color in (("tsc", "tab:blue"), ("reve", "tab:red")):
            pts = collect(model, things=things)
            if not pts:
                continue
            n = np.array(list(pts)); m = np.array([np.mean(v) for v in pts.values()])
            print(f"{model:5s} THINGS {things:5s} " + "  ".join(f"N={k}: {np.mean(v):.1f} ({len(v)})" for k, v in pts.items()))
            ax.plot(n, m, "o-", color=color, label=LABELS[model])
            ax.fill_between(n, [min(v) for v in pts.values()], [max(v) for v in pts.values()], color=color, alpha=0.15)
        ax.axvspan(0.5, 10.5, color="tab:green", alpha=0.07) if things == "first" else ax.axvspan(15.5, 25.5, color="tab:green", alpha=0.07)
        ax.set_xticks(range(1, 26, 2)); ax.set_xlim(0, 26); ax.grid(alpha=0.3)
        ax.set_xlabel("training subjects (green: THINGS-EEG2 subjects being added)")
        ax.set_title(f"THINGS-EEG2 subjects {things}")
    axes[0].set_ylabel("sealed-rule top-5, new Alljoined subjects (%)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("plot:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/scaling/scaling_curve.png")
    ap.add_argument("--things", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.things:
        return plot_things(a.out.replace("scaling_curve.png", "scaling_things.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, drop, title in zip(axes, [(), ("15",)], ["all test subjects", "without sub-15"]):
        for model, color in (("tsc", "tab:blue"), ("plab", "tab:orange"), ("reve", "tab:red")):
            pts = collect(model, drop)
            if not pts:
                continue
            n = np.array(list(pts)); m = np.array([np.mean(v) for v in pts.values()])
            lo = np.array([np.min(v) for v in pts.values()]); hi = np.array([np.max(v) for v in pts.values()])
            print(f"{model:5s} {title:18s} " + "  ".join(f"N={k}: {np.mean(v):.1f} ({len(v)} runs)" for k, v in pts.items()))
            ax.plot(n, m, "o-", color=color, label=LABELS[model])
            ax.fill_between(n, lo, hi, color=color, alpha=0.15)
            if len(n) >= 3:  # log-linear fit, extrapolated to the final 20-subject model and to 30 subjects
                b, c = np.polyfit(np.log(n), m, 1)
                xs = np.array([1, 30]); ax.plot(xs, b * np.log(xs) + c, ":", color=color, alpha=0.6)
                print(f"{'':24s} fit: {b:+.2f} per doubling x ln2 = {b * np.log(2):+.2f}; "
                      f"N=20 -> {b * np.log(20) + c:.1f}, N=30 -> {b * np.log(30) + c:.1f}")
        ax.set_xscale("log"); ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 20, 30]); ax.set_xticklabels([1, 3, 5, 7, 9, 11, 13, 15, 20, 30])
        ax.set_xlabel("training subjects"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel("sealed-rule top-5, new subjects (%)"); axes[0].legend()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print("plot:", a.out)


if __name__ == "__main__":
    main()
