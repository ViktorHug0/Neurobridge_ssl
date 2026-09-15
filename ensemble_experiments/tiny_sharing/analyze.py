"""Deliverables of tiny_sharing_plan.md sec. 9, from the Phase B screen.

    python -m ensemble_experiments.tiny_sharing.analyze [--result-root DIR]

Emits the 10-row accuracy/resource table, the contrast graph annotated with
paired accuracy deltas, the two interaction contrasts, the correctness-overlap
and margin decomposition, and the C0 checkpoint-coupling cost.
"""

import argparse
import json
import os

import numpy as np

from ensemble_experiments.tiny_sharing.data import REPO
from ensemble_experiments.tiny_sharing.models import CONFIG_IDS

SLOTS = [1, 2, 3]
DEFAULT_ROOT = os.path.join(REPO, "results/things_eeg/tiny_sharing/screen")


def load(root):
    runs = {}
    for slot in SLOTS:
        for config in CONFIG_IDS:
            d = os.path.join(root, f"slot{slot}", config)
            m = os.path.join(d, "metrics_best_mean.json")
            s = os.path.join(d, "summary.json")
            if os.path.isfile(m) and os.path.isfile(s):
                runs[(slot, config)] = {
                    "metrics": json.load(open(m)),
                    "summary": json.load(open(s)),
                    "dir": d,
                }
    return runs


def _mean(runs, config, path, scale=100.0):
    vals = [runs[(s, config)]["metrics"][path] for s in SLOTS if (s, config) in runs]
    return (float(np.mean(vals)) * scale, len(vals)) if vals else (float("nan"), 0)


def accuracy_table(runs):
    print("\n=== 10-row accuracy / resource table (inner validation, mean of "
          "available screening folds, common-epoch checkpoint) ===")
    head = (f"{'ID':<4}{'n':>2}  {'TS':>6}{'ATM':>7}{'FUSED':>7}{'ORACLE':>8}"
            f"{'gain>best':>10}{'agree':>7}  {'EEGuniq':>8}{'shared':>8}"
            f"{'step_ms':>8}{'ep_s':>7}{'GPU_MB':>8}{'sel_ep':>7}")
    print(head)
    print("-" * len(head))
    rows = {}
    for config in CONFIG_IDS:
        present = [s for s in SLOTS if (s, config) in runs]
        if not present:
            print(f"{config:<4}{0:>2}  (not run)")
            continue
        sm = [runs[(s, config)]["summary"] for s in present]
        row = {
            "n": len(present),
            "top1_ts": _mean(runs, config, "top1_ts")[0],
            "top1_atm": _mean(runs, config, "top1_atm")[0],
            "top1_fused": _mean(runs, config, "top1_fused")[0],
            "oracle": _mean(runs, config, "oracle_top1")[0],
            "gain_best": _mean(runs, config, "gain_over_best_branch")[0],
            "agree": _mean(runs, config, "prediction_agreement")[0],
            "eeg_unique": sm[0]["parameters"]["eeg_unique"],
            "shared": sm[0]["parameters"]["shared_parameters"],
            "step_ms": 1000 * float(np.mean([x["warm_step_seconds"] for x in sm])),
            "epoch_s": float(np.mean([x["mean_epoch_seconds"] for x in sm])),
            "gpu_mb": float(np.mean([x["peak_gpu_bytes"] for x in sm])) / 2 ** 20,
            "sel_ep": float(np.mean([x["selected_epoch_common"] for x in sm])),
        }
        rows[config] = row
        print(f"{config:<4}{row['n']:>2}  {row['top1_ts']:>6.2f}{row['top1_atm']:>7.2f}"
              f"{row['top1_fused']:>7.2f}{row['oracle']:>8.2f}{row['gain_best']:>10.2f}"
              f"{row['agree']:>7.2f}  {row['eeg_unique']:>8d}{row['shared']:>8d}"
              f"{row['step_ms']:>8.1f}{row['epoch_s']:>7.1f}{row['gpu_mb']:>8.0f}"
              f"{row['sel_ep']:>7.1f}")
    return rows


def paired_delta(runs, a, b, key="top1_fused"):
    """Mean paired difference A(b) - A(a) over folds where both ran, in pp."""
    d = [
        runs[(s, b)]["metrics"][key] - runs[(s, a)]["metrics"][key]
        for s in SLOTS if (s, a) in runs and (s, b) in runs
    ]
    return (100 * float(np.mean(d)), len(d)) if d else (float("nan"), 0)


def contrast_graph(runs):
    print("\n=== contrast graph, annotated with paired fused-top1 deltas (pp) ===")
    edges = [
        ("C0", "C1", "bridge (NOT a sharing penalty)"),
        ("C1", "C2", "tie T"),
        ("C1", "C3", "tie S"),
        ("C1", "C4", "tie R"),
        ("C2", "C5", "+S"),
        ("C3", "C5", "+T"),
        ("C5", "C6", "+R"),
        ("C4", "C6", "+T,S"),
        ("C6", "C7", "+N"),
        ("C1", "C8", "new ordering (topology cost)"),
        ("C8", "C9", "shared stem activation/computation"),
    ]
    for a, b, label in edges:
        for key, tag in (("top1_fused", "fused"), ("oracle_top1", "oracle"),
                         ("gain_over_best_branch", "gain>best")):
            d, n = paired_delta(runs, a, b, key)
            if key == "top1_fused":
                print(f"  {a} -> {b:<3} {label:<38} {tag}{d:+7.2f}  (n={n})", end="")
            else:
                print(f"  {tag}{d:+7.2f}", end="")
        print()


def interactions(runs):
    print("\n=== interaction contrasts (pp, per fold; negative = combined tie "
          "loses more than the sum of the parts on this metric scale) ===")
    for name, (p, q, r, base) in {
        "A(C5)-A(C2)-A(C3)+A(C1)": ("C5", "C2", "C3", "C1"),
        "A(C6)-A(C5)-A(C4)+A(C1)": ("C6", "C5", "C4", "C1"),
    }.items():
        per_fold = []
        for s in SLOTS:
            if not all((s, c) in runs for c in (p, q, r, base)):
                continue
            v = (runs[(s, p)]["metrics"]["top1_fused"]
                 - runs[(s, q)]["metrics"]["top1_fused"]
                 - runs[(s, r)]["metrics"]["top1_fused"]
                 + runs[(s, base)]["metrics"]["top1_fused"])
            per_fold.append(100 * v)
        if per_fold:
            fmt = ", ".join(f"slot{s}={v:+.2f}" for s, v in zip(SLOTS, per_fold))
            print(f"  {name}: mean {np.mean(per_fold):+.2f}   [{fmt}]")


def overlap_and_margins(runs):
    print("\n=== correctness-overlap and margin decomposition (mean over folds) ===")
    head = (f"{'ID':<4}{'both':>7}{'TSonly':>8}{'ATMonly':>9}{'neither':>9}"
            f"{'rescue':>8}{'floss':>7}  {'m_TS':>7}{'m_ATM':>7}{'m_fus':>7}"
            f"{'bonus':>7}{'CKA':>6}")
    print(head)
    print("-" * len(head))
    for config in CONFIG_IDS:
        present = [s for s in SLOTS if (s, config) in runs]
        if not present:
            continue
        m = [runs[(s, config)]["metrics"] for s in present]
        o = {k: np.mean([x["overlap"][k] for x in m])
             for k in ("both", "ts_only", "atm_only", "neither")}
        g = lambda k: np.mean([x[k] for x in m])
        print(f"{config:<4}{o['both']:>7.0f}{o['ts_only']:>8.0f}{o['atm_only']:>9.0f}"
              f"{o['neither']:>9.0f}{g('fusion_rescues'):>8.0f}"
              f"{g('fusion_losses'):>7.0f}  {g('margin_ts'):>7.3f}"
              f"{g('margin_atm'):>7.3f}{g('margin_fused'):>7.3f}"
              f"{g('distractor_complementarity_bonus'):>7.3f}{g('cka_readout'):>6.2f}")


def _rowz(x):
    return (x - x.mean(1, keepdims=True)) / np.clip(x.std(1, keepdims=True), 1e-8, None)


def checkpoint_coupling(runs):
    """C0/C1 independent-checkpoint vs common-checkpoint, same trajectory."""
    print("\n=== checkpoint coupling cost (independent branch-best vs one common "
          "epoch, same observed trajectory) ===")
    for config in ("C0", "C1"):
        deltas = []
        for slot in SLOTS:
            run = runs.get((slot, config))
            if run is None:
                continue
            paths = [os.path.join(run["dir"], f"scores_best_{k}.npz") for k in ("ts", "atm")]
            if not all(os.path.isfile(p) for p in paths):
                continue
            a, b = (np.load(p) for p in paths)
            correct = a["correct"]
            fused = _rowz(a["scores"]) + _rowz(b["scores"])
            indep = float((fused.argmax(1) == correct).mean())
            common = run["metrics"]["top1_fused"]
            deltas.append((100 * indep, 100 * common, 100 * (indep - common),
                           int(a["epoch"]), int(b["epoch"]),
                           run["metrics"]["selected_epoch"]))
        for slot, d in zip(SLOTS, deltas):
            print(f"  {config} slot{slot}: independent {d[0]:.2f} (ep {d[3]}/{d[4]}) "
                  f"vs common {d[1]:.2f} (ep {d[5]}) -> {d[2]:+.2f} pp")
        if deltas:
            print(f"  {config} mean coupling cost: "
                  f"{np.mean([d[2] for d in deltas]):+.2f} pp")


def bridge_gate(runs):
    """Feasibility flag from plan sec. 7 Phase A, not an equivalence criterion."""
    print("\n=== bridge gate (feasibility flag, NOT statistical equivalence) ===")
    for slot in SLOTS:
        if (slot, "C0") not in runs or (slot, "C1") not in runs:
            continue
        c0, c1 = runs[(slot, "C0")]["metrics"], runs[(slot, "C1")]["metrics"]
        drop = 100 * (c0["top1_fused"] - c1["top1_fused"])
        g0, g1 = c0["gain_over_best_branch"], c1["gain_over_best_branch"]
        lost_half = g0 > 0 and g1 < 0.5 * g0
        flag = drop > 2.0 or lost_half
        print(f"  slot{slot}: fused drop {drop:+.2f} pp; gain over best branch "
              f"{100*g0:.2f} -> {100*g1:.2f} pp; "
              f"{'WARNING - revise the bridge' if flag else 'within threshold'}")


def stability(runs):
    print("\n=== ranking stability: best within 20 epochs vs within 40 ===")
    for config in CONFIG_IDS:
        vals = []
        for slot in SLOTS:
            run = runs.get((slot, config))
            if run is None:
                continue
            p = os.path.join(run["dir"], "metrics_best_mean_20.json")
            if os.path.isfile(p):
                vals.append((100 * json.load(open(p))["top1_fused"],
                             100 * run["metrics"]["top1_fused"]))
        if vals:
            e20 = np.mean([v[0] for v in vals])
            e40 = np.mean([v[1] for v in vals])
            print(f"  {config}: fused {e20:.2f} (<=20 ep) -> {e40:.2f} (<=40 ep) "
                  f"[{e40 - e20:+.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    runs = load(args.result_root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.result_root}")
    print(f"loaded {len(runs)} of {len(SLOTS) * len(CONFIG_IDS)} paired jobs "
          f"from {args.result_root}")
    accuracy_table(runs)
    bridge_gate(runs)
    contrast_graph(runs)
    interactions(runs)
    overlap_and_margins(runs)
    checkpoint_coupling(runs)
    stability(runs)
    print("\nScreening priorities are the accuracy/compute trade-off and the "
          "interpretable contrasts, not the highest of ten inner-validation numbers.\n"
          "Source validation is reused for selection, so these are optimistic "
          "development estimates, not generalization estimates.")


if __name__ == "__main__":
    main()
