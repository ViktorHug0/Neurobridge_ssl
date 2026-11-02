"""Read the cv4 protocol results (neurips_challenge/cv4.sh) and apply its rules (neurips_challenge/README.md, protocol).

    cv4_summary.py                     per-recipe table: chosen epoch budget, score, training-length audit
    cv4_summary.py --ref tsc           + every recipe paired against `tsc` on the subjects both have
    cv4_summary.py --self_test

Every run logs, each epoch, the sealed-rule score of every test subject on a seeded half of its trials
(`sealed_per_epoch.jsonl`). A recipe's epoch budget is the epoch with the best mean over all its test
subjects; its cv4 score is that epoch's mean, and paired comparisons use each recipe at its own budget.
The final all-subject model is trained with that budget.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/alljoined/track1/cv4")
SCREEN_FOLDS, ALL_FOLDS = {1, 2}, {1, 2, 3, 4}
FRACTION = 0.5  # cv4.sh --sealed_test_fraction: per-epoch scores use a seeded half of each subject's trials


def load(root=ROOT):
    """{recipe: {"arm": arm, "folds": {k: {epoch: {subject: top5}}}}} for finished (scored) runs."""
    out = {}
    for f in sorted(glob.glob(f"{root}/*/fold*/*/*/track1_score.json")):
        recipe, fold, arm = f.split("/")[-5:-2]
        curve = os.path.join(os.path.dirname(f), "sealed_per_epoch.jsonl")
        if not os.path.exists(curve):
            continue
        rows = [json.loads(line) for line in open(curve)]
        r = out.setdefault(recipe, {"arm": arm, "folds": {}})
        r["folds"][int(fold[4:])] = {row["epoch"]: row["by_subject"] for row in rows if row["fraction"] == FRACTION}
    return out


def curve(r):
    """Mean over all the recipe's test subjects, per epoch (epochs every fold reached)."""
    epochs = sorted(set.intersection(*(set(f) for f in r["folds"].values())))
    return epochs, np.array([np.mean([v for f in r["folds"].values() for v in f[e].values()]) for e in epochs])


def audit(epochs, c):
    """Is the chosen budget inside the trained range? (curve noise at 20 subjects x 80 repeats: ~+-0.5)"""
    if len(c) >= 4 and c[-1] > np.max(c[:-1]) + 0.3:  # a plateau that dips and recovers is not "rising"
        return "UNDERTRAINED (the last epoch beats every earlier one: train longer)"
    return "ok"


def decide(diffs, folds, same_arm):
    """Protocol §6: screen on folds 1-2 (10 subjects), confirm on all 4 folds (20 subjects)."""
    d = np.asarray(diffs, float)
    mean, better = d.mean(), int((d > 0).sum())
    p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d) else float("nan")
    if folds >= ALL_FOLDS:
        need = 1.0 if same_arm else 1.5
        verdict = "ADOPT" if mean >= need and p < 0.1 else "reject" if mean <= 0 else "no clear gain"
    elif folds >= SCREEN_FOLDS:
        verdict = "drop" if mean <= 0 else "promote to folds 3-4" if mean >= 1 and better >= 6 else "unclear: confirm or drop"
    else:
        verdict = "incomplete screen"
    return mean, better, len(d), p, verdict


def at_budget(r):
    epochs, c = curve(r)
    e = epochs[int(np.argmax(c))]
    return e, {(k, s): v for k, f in r["folds"].items() for s, v in f[e].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="reference recipe for paired comparisons")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--verdict", help="print only this recipe's decision against --ref (for scripts)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    res = load()
    if a.verdict:
        if a.verdict not in res or a.ref not in res:
            print("missing")
            return
        _, new = at_budget(res[a.verdict]); _, ref = at_budget(res[a.ref])
        keys = sorted(set(new) & set(ref))
        print(decide([new[k] - ref[k] for k in keys], {k for k, _ in keys}, res[a.verdict]["arm"] == res[a.ref]["arm"])[-1])
        return
    if not res:
        print(f"no finished cv4 runs with per-epoch logs under {ROOT}")
        return
    print(f"{'recipe':11s} {'arm':5s} folds  n  budget  score  last-epoch | audit")
    for name, r in res.items():
        epochs, c = curve(r)
        e, scores = at_budget(r)
        print(f"{name:11s} {r['arm']:5s} {''.join(map(str, sorted(r['folds']))):5s} {len(scores):2d}  ep {e:3d}  {np.mean(list(scores.values())):5.2f}"
              f"  {c[-1]:5.2f} (ep {epochs[-1]}) | {audit(epochs, c)}{' (provisional: 1 fold)' if len(r['folds']) < 2 else ''}")
        print(f"{'':11s} curve: " + " ".join(f"{x:.1f}" for x in c))
    if a.ref:
        _, ref = at_budget(res[a.ref])
        print(f"\npaired vs {a.ref}, each recipe at its own budget (B - A per test subject):")
        for name, r in res.items():
            if name == a.ref:
                continue
            _, new = at_budget(r)
            keys = sorted(set(new) & set(ref))
            if not keys:
                continue
            folds = {k for k, _ in keys}
            mean, better, n, p, verdict = decide([new[k] - ref[k] for k in keys], folds, r["arm"] == res[a.ref]["arm"])
            print(f"  {name:11s} {mean:+5.2f} over {n:2d} subjects, better on {better:2d}, Wilcoxon p {p:.3f} -> {verdict}")


def self_test():
    rising, peaked = np.array([10, 12, 14, 16, 18, 20, 22, 24.0]), np.array([10, 20, 22, 21, 19, 18, 17, 16.0])
    assert audit(range(8), rising).startswith("UNDERTRAINED")
    assert audit(range(8), peaked) == "ok"
    assert audit(range(10), np.array([16.7, 19.6, 24.4, 25.4, 25.5, 26.8, 26.0, 25.9, 26.4, 26.9])) == "ok"  # lab_s10 fold 1
    r = {"arm": "x", "folds": {1: {1: {"a": 10.0}, 2: {"a": 30.0}, 3: {"a": 20.0}}, 2: {1: {"b": 10.0}, 2: {"b": 20.0}, 3: {"b": 24.0}}}}
    assert at_budget(r)[0] == 2 and at_budget(r)[1] == {(1, "a"): 30.0, (2, "b"): 20.0}
    assert decide([2.0] * 7 + [-1.0] * 3, {1, 2}, True)[-1] == "promote to folds 3-4"
    assert decide([-0.5] * 10, {1, 2}, True)[-1] == "drop"
    assert decide([2.0] * 17 + [-1.0] * 3, ALL_FOLDS, False)[-1] == "ADOPT"  # +1.55 >= 1.5
    assert decide([1.2] * 12 + [-0.3] * 8, ALL_FOLDS, False)[-1] == "no clear gain"  # +0.6 < 1.5
    print("cv4_summary self-test ok")


if __name__ == "__main__":
    main()
