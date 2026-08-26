"""Add one global per-candidate bias to a fixed fusion rule.

The ledger's candidate scorer was explicitly denied candidate identity, so the
subject-independent part of candidate hubness has never been corrected.  A
single 200-vector `b`, fitted once on pooled labeled queries and then frozen,
is still one global rule: it has no subject input, no per-fold refit, and reads
no other query at inference.

    fused'(q, c) = fused(q, c) / T + b(c)

`--nested` refits `b` and reselects (T, lambda) on nine folds and scores the
tenth, which is the number that says whether the correction generalizes to a
subject it was not fitted on.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from global_rule_search import (
    RAW,
    SUBJECTS,
    TRANSFORMS,
    TRUTH,
    acc_by_fold,
    confidences,
    stack,
)

TEMPERATURES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


def build_fused(rule: dict) -> np.ndarray:
    """Rebuild the (10, 200, 200) fused score tensor for one frozen rule."""
    fused = None
    for name, transform, weight in zip(
        rule["members"], rule["transforms"], rule["weights"]
    ):
        term = TRANSFORMS[transform](RAW[name])
        if rule["conf_stat"] != "const":
            conf = confidences(RAW[name])[rule["conf_stat"]] ** rule["conf_alpha"]
            term = term * conf[:, :, None]
        term = term * weight
        fused = term.copy() if fused is None else fused + term
    return fused


def fit_bias(
    fused: np.ndarray, folds: np.ndarray, temperature: float, lam: float, steps: int = 600
) -> np.ndarray:
    """Full-batch Adam on pooled cross-entropy; only `b` is free."""
    logits = (fused[folds] / temperature).reshape(-1, 200)
    onehot = np.zeros_like(logits)
    onehot[np.arange(len(logits)), np.tile(TRUTH, len(folds))] = 1.0
    b = np.zeros(200, dtype=np.float32)
    m = np.zeros(200, dtype=np.float32)
    v = np.zeros(200, dtype=np.float32)
    for step in range(1, steps + 1):
        x = logits + b
        x -= x.max(1, keepdims=True)
        p = np.exp(x)
        p /= p.sum(1, keepdims=True)
        grad = (p - onehot).mean(0) + 2.0 * lam * b
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        b -= 0.05 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return b


def evaluate(fused: np.ndarray, b: np.ndarray, temperature: float) -> np.ndarray:
    return acc_by_fold(fused / temperature + b)


FUSED: np.ndarray | None = None


def _fit_one(args: tuple[np.ndarray, float, float]) -> tuple[np.ndarray, float, float, float]:
    folds, temperature, lam = args
    b = fit_bias(FUSED, folds, temperature, lam)
    return b, temperature, lam, float(evaluate(FUSED, b, temperature)[folds].mean())


def select(folds: np.ndarray, pool) -> tuple[np.ndarray, float, float, float]:
    """Choose (T, lambda) and fit b, using only `folds`."""
    jobs = [(folds, t, lam) for t in TEMPERATURES for lam in LAMBDAS]
    return max(pool.map(_fit_one, jobs, chunksize=1), key=lambda r: r[3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="results/things_eeg/ensemble50_testselected/global_rule_search.json")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--output", default="results/things_eeg/ensemble50_testselected/candidate_bias.json")
    args = parser.parse_args()

    rules = json.loads(Path(args.rules).read_text())["all_ten"]
    needed = {n for rule in rules.values() for n in rule["members"]}
    for name in sorted(needed):
        RAW[name] = stack(name)

    global FUSED
    all_folds = np.arange(len(SUBJECTS))
    report = {}
    for k, rule in sorted(rules.items(), key=lambda kv: int(kv[0])):
        fused = FUSED = build_fused(rule)
        base = acc_by_fold(fused)
        pool = Pool(args.processes)
        b, temperature, lam, _ = select(all_folds, pool)
        pooled_folds = evaluate(fused, b, temperature)

        nested = []
        for held in range(len(SUBJECTS)):
            train = np.array([f for f in range(len(SUBJECTS)) if f != held])
            bh, th, _, _ = select(train, pool)
            nested.append(float(evaluate(fused, bh, th)[held]))
        pool.close()

        report[k] = {
            "members": rule["members"],
            "no_bias_mean": float(base.mean()),
            "no_bias_per_fold": base.tolist(),
            "pooled_bias_mean": float(pooled_folds.mean()),
            "pooled_bias_per_fold": pooled_folds.tolist(),
            "nested_bias_mean": float(np.mean(nested)),
            "nested_bias_per_fold": nested,
            "temperature": temperature,
            "lambda": lam,
            "bias": [round(float(x), 5) for x in b],
        }
        print(
            f"k={k}: no-bias {base.mean():.2f} -> pooled-fit {pooled_folds.mean():.2f} "
            f"(T={temperature}, lam={lam}) | nested {np.mean(nested):.2f}",
            flush=True,
        )
        print("   pooled folds: " + " ".join(f"{v:.1f}" for v in pooled_folds), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
