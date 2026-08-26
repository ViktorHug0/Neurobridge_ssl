"""Search one globally fixed, query-local fusion rule for k = 2, 3, 4 members.

Rule family (every parameter is shared by all ten folds; nothing reads another
query, so this stays inside the no-transductive-adaptation contract):

    fused(q, c) = sum_m  w_m * conf(s_m(q, :)) ** alpha * T_m(s_m(q, :))[c]

  * s_m  member m's 200 cosine scores for query q (L2-normalized EEG/image)
  * T_m  one query-local transform per member, from a fixed menu
  * conf one query-local confidence statistic of those same 200 scores
  * w_m  global per-member weights

Stage 1 screens member sets by beam search under a single shared transform and
uniform weights.  Stage 2 refines the survivors over per-member transforms,
global weights, and the confidence exponent.  Stage 3 repeats the whole
procedure leave-one-fold-out to report what actually generalizes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("results/things_eeg/synthetic_subjects/ensemble_screen/dumps")
SUBJECTS = tuple(range(1, 11))
TRUTH = np.arange(200)

# --- query-local transform menu -------------------------------------------


def _z(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.mean(2, keepdims=True)) / np.maximum(
        matrix.std(2, keepdims=True), 1e-6
    )


def _rank(matrix: np.ndarray) -> np.ndarray:
    order = matrix.argsort(2)
    ranks = np.empty(order.shape, dtype=np.float32)
    np.put_along_axis(
        ranks, order, np.broadcast_to(np.arange(200, dtype=np.float32), order.shape), 2
    )
    return ranks / 199.0


def _rrf(matrix: np.ndarray, c: float) -> np.ndarray:
    return (1.0 / (c + (199.0 - _rank(matrix) * 199.0) + 1.0)).astype(np.float32)


def _topm(matrix: np.ndarray, m: int) -> np.ndarray:
    z = _z(matrix)
    thr = np.partition(z, -m, axis=2)[..., -m][..., None]
    return np.where(z >= thr, z, 0.0).astype(np.float32)


def _softmax(matrix: np.ndarray, tau: float) -> np.ndarray:
    x = _z(matrix) / tau
    x = x - x.max(2, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(2, keepdims=True)).astype(np.float32)


def _pow(matrix: np.ndarray, p: float) -> np.ndarray:
    z = _z(matrix)
    return (np.sign(z) * np.abs(z) ** p).astype(np.float32)


TRANSFORMS: dict[str, callable] = {
    "raw": lambda m: m,
    "rank": _rank,
    **{f"pow{p}": (lambda m, p=p: _pow(m, p)) for p in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)},
    **{f"sm{t}": (lambda m, t=t: _softmax(m, t)) for t in (0.5, 1.0, 2.0, 4.0)},
    **{f"rrf{c}": (lambda m, c=c: _rrf(m, c)) for c in (10.0, 30.0, 60.0)},
    **{f"top{m}": (lambda x, m=m: _topm(x, m)) for m in (5, 10, 25, 50)},
}

# --- query-local confidence statistics -------------------------------------


def confidences(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Per (fold, query) positive scalars derived only from that query's row."""
    z = _z(matrix)
    top = np.sort(z, axis=2)[:, :, ::-1]
    probs = _softmax(matrix, 1.0)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(2) / np.log(200.0)
    return {
        "const": np.ones(z.shape[:2], dtype=np.float32),
        "margin": np.maximum(top[:, :, 0] - top[:, :, 1], 1e-6),
        "zmax": np.maximum(top[:, :, 0], 1e-6),
        "gap5": np.maximum(top[:, :, 0] - top[:, :, 1:5].mean(2), 1e-6),
        "negent": np.maximum(1.0 - entropy, 1e-6),
    }


CONF_STATS = ("const", "margin", "zmax", "gap5", "negent")
ALPHAS = (0.25, 0.5, 1.0, 2.0)

# --- stage 3: continuous per-member transform ------------------------------
#
# The discrete menu above forces one shape on every member.  Stage 3 gives each
# member its own (family, parameter, top-m truncation, confidence statistic),
# and coordinate-ascends on measured top-1 rather than on a surrogate loss.

FAMILIES = (
    ("raw", 0.0),
    ("rank", 0.0),
    ("vote", 0.0),
    *[("pow", p) for p in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)],
    *[("sm", t) for t in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)],
    *[("rrf", c) for c in (1.0, 5.0, 20.0, 60.0)],
)
TRUNCATIONS = (5, 10, 15, 25, 40, 60, 100, 200)

MENU_SEED = {
    "raw": ("raw", 0.0, 200),
    "rank": ("rank", 0.0, 200),
    **{f"pow{p}": ("pow", p, 200) for p in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)},
    **{f"sm{t}": ("sm", t, 200) for t in (0.5, 1.0, 2.0, 4.0)},
    **{f"rrf{c}": ("rank", 0.0, 200) for c in (10.0, 30.0, 60.0)},
    **{f"top{m}": ("pow", 1.0, m) for m in (5, 10, 25, 50)},
}


def apply_family(raw: np.ndarray, z: np.ndarray, family: str, param: float, m: int) -> np.ndarray:
    """One member's query-local transform: shape, then optional top-m truncation."""
    if family == "raw":
        out = raw
    elif family == "rank":
        out = _rank(raw)
    elif family == "vote":
        # pure plurality vote: every kept candidate contributes the same weight,
        # so the member votes for a set instead of grading it
        out = np.ones_like(z)
    elif family == "rrf":
        out = 1.0 / (param + (199.0 - _rank(raw) * 199.0) + 1.0)
    elif family == "pow":
        out = np.sign(z) * np.abs(z) ** param
    else:
        x = z / param
        x = x - x.max(2, keepdims=True)
        e = np.exp(x)
        out = e / e.sum(2, keepdims=True)
    if m < 200:
        thr = np.partition(z, -m, axis=2)[..., -m][..., None]
        out = np.where(z >= thr, out, 0.0)
    return out.astype(np.float32)

# --- data ------------------------------------------------------------------


def arms_with_all_folds() -> list[str]:
    names = {p.name.rsplit("-sub", 1)[0] for p in ROOT.glob("*.npz")}
    return sorted(
        n
        for n in names
        if all((ROOT / f"{n}-sub{s:02d}.npz").exists() for s in SUBJECTS)
    )


def stack(name: str) -> np.ndarray:
    out = np.empty((len(SUBJECTS), 200, 200), dtype=np.float32)
    for i, subject in enumerate(SUBJECTS):
        data = np.load(ROOT / f"{name}-sub{subject:02d}.npz")
        eeg = data["eeg"].astype(np.float32)
        image = data["image"].astype(np.float32)
        eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
        image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
        out[i] = eeg @ image.T
    return out


def acc_by_fold(fused: np.ndarray) -> np.ndarray:
    return (fused.argmax(2) == TRUTH).mean(1) * 100.0


# --- stage 1: member-set screening under one shared transform --------------

RAW: dict[str, np.ndarray] = {}


def _screen(args: tuple[str, int, int]) -> tuple[str, dict]:
    """Beam-search member sets for one transform, keeping per-fold accuracies.

    The beam is grown on the all-ten mean; the cached per-fold rows let the
    nested pass re-rank the same candidate pool on nine folds cheaply.
    """
    transform, width, kmax = args
    names = sorted(RAW)
    tensors = {n: TRANSFORMS[transform](RAW[n]) for n in names}

    def evaluate(combo: tuple[str, ...]) -> np.ndarray:
        fused = tensors[combo[0]].copy()
        for n in combo[1:]:
            fused += tensors[n]
        return acc_by_fold(fused)

    out: dict[int, list] = {}
    scored = [
        (float(acc.mean()), combo, acc.tolist())
        for combo in itertools.combinations(names, 2)
        for acc in (evaluate(combo),)
    ]
    scored.sort(key=lambda x: -x[0])
    out[2] = scored[:width]
    level = scored[:width]
    for k in range(3, kmax + 1):
        seen, scored = set(), []
        for _, combo, _ in level:
            for n in names:
                if n in combo:
                    continue
                nxt = tuple(sorted(combo + (n,)))
                if nxt in seen:
                    continue
                seen.add(nxt)
                acc = evaluate(nxt)
                scored.append((float(acc.mean()), nxt, acc.tolist()))
        scored.sort(key=lambda x: -x[0])
        out[k] = scored[:width]
        level = scored[:width]
    return transform, out


# --- stage 2: refine one member set ----------------------------------------


class Refiner:
    """Evaluates the full rule family for one fixed member set."""

    def __init__(self, members: tuple[str, ...], menu: tuple[str, ...]):
        self.members = members
        self.menu = menu
        self.tensors = {
            (n, t): TRANSFORMS[t](RAW[n]) for n in set(members) for t in menu
        }
        self.conf = {n: confidences(RAW[n]) for n in set(members)}
        self.z = {n: _z(RAW[n]) for n in set(members)}

    def accuracy(
        self,
        transforms: tuple[str, ...],
        weights: np.ndarray,
        stat: str,
        alpha: float,
    ) -> np.ndarray:
        fused = None
        for n, t, w in zip(self.members, transforms, weights):
            term = self.tensors[(n, t)]
            if stat != "const":
                term = term * (self.conf[n][stat] ** alpha)[:, :, None]
            contribution = term * w if w != 1.0 else term
            fused = contribution.copy() if fused is None else fused + contribution
        return acc_by_fold(fused)

    def refine(self, folds: tuple[int, ...], seed_transform: str) -> dict:
        """Coordinate ascent on per-member transform, weights, then confidence."""
        sel = np.asarray(folds)
        k = len(self.members)
        transforms = tuple([seed_transform] * k)
        weights = np.ones(k, dtype=np.float32)
        stat, alpha = "const", 0.0

        def objective(tr, w, st, al) -> float:
            return float(self.accuracy(tr, w, st, al)[sel].mean())

        best = objective(transforms, weights, stat, alpha)
        grid = np.geomspace(0.15, 6.0, 21).astype(np.float32)
        for _ in range(3):
            improved = False
            for i in range(k):  # per-member transform
                for t in self.menu:
                    if t == transforms[i]:
                        continue
                    trial = transforms[:i] + (t,) + transforms[i + 1 :]
                    value = objective(trial, weights, stat, alpha)
                    if value > best + 1e-9:
                        best, transforms, improved = value, trial, True
            for i in range(1, k):  # global weights (w[0] fixed: scale-invariant)
                for candidate in grid:
                    if candidate == weights[i]:
                        continue
                    trial = weights.copy()
                    trial[i] = candidate
                    value = objective(transforms, trial, stat, alpha)
                    if value > best + 1e-9:
                        best, weights, improved = value, trial, True
            for st in CONF_STATS:  # query-local confidence weighting
                for al in (ALPHAS if st != "const" else (0.0,)):
                    value = objective(transforms, weights, st, al)
                    if value > best + 1e-9:
                        best, stat, alpha, improved = value, st, al, True
            if not improved:
                break
        return self.refine_parametric(sel, transforms, weights, stat, alpha, best)

    def accuracy_p(self, cfg: list[dict], weights: np.ndarray) -> np.ndarray:
        """Top-1 for a per-member (family, param, truncation, confidence) config."""
        fused = None
        for name, c, w in zip(self.members, cfg, weights):
            term = apply_family(RAW[name], self.z[name], c["family"], c["param"], c["m"])
            if c["stat"] != "const":
                term = term * (self.conf[name][c["stat"]] ** c["alpha"])[:, :, None]
            term = term * w
            fused = term.copy() if fused is None else fused + term
        return acc_by_fold(fused)

    def refine_parametric(self, sel, transforms, weights, stat, alpha, best) -> dict:
        """Coordinate ascent on per-member shape, truncation, and confidence."""
        cfg = []
        for t in transforms:
            family, param, m = MENU_SEED[t]
            cfg.append({"family": family, "param": param, "m": m, "stat": stat, "alpha": alpha})
        weights = weights.copy()
        k = len(self.members)

        def objective(c, w) -> float:
            return float(self.accuracy_p(c, w)[sel].mean())

        best = max(best, objective(cfg, weights))
        grid = np.geomspace(0.15, 6.0, 21).astype(np.float32)
        for _ in range(3):
            improved = False
            for i in range(k):
                for field, options in (
                    ("shape", FAMILIES),
                    ("m", TRUNCATIONS),
                    ("conf", [(s, a) for s in CONF_STATS for a in (ALPHAS if s != "const" else (0.0,))]),
                ):
                    for option in options:
                        trial = [dict(c) for c in cfg]
                        if field == "shape":
                            trial[i]["family"], trial[i]["param"] = option
                        elif field == "m":
                            trial[i]["m"] = option
                        else:
                            trial[i]["stat"], trial[i]["alpha"] = option
                        value = objective(trial, weights)
                        if value > best + 1e-9:
                            best, cfg, improved = value, trial, True
            for i in range(1, k):
                for candidate in grid:
                    trial = weights.copy()
                    trial[i] = candidate
                    value = objective(cfg, trial)
                    if value > best + 1e-9:
                        best, weights, improved = value, trial, True
            if not improved:
                break
        return {
            "members": list(self.members),
            "config": cfg,
            "weights": [round(float(w), 4) for w in weights],
            "selection_mean": best,
            "per_fold": self.accuracy_p(cfg, weights).tolist(),
        }


def _refine_one(args: tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], str]) -> dict:
    members, menu, folds, seed_transform = args
    return Refiner(members, menu).refine(folds, seed_transform)


def search(
    screened: dict[str, dict],
    folds: tuple[int, ...],
    ks: tuple[int, ...],
    keep: int,
    menu: tuple[str, ...],
    pool: Pool,
) -> dict[int, dict]:
    """Stage 2 over the cached screen, ranking candidates on `folds` only."""
    sel = np.asarray(folds)
    results: dict[int, dict] = {}
    for k in ks:
        ranked: dict[tuple[str, ...], tuple[float, str]] = {}
        for transform, per_k in screened.items():
            for _, combo, per_fold in per_k[k]:
                mean = float(np.asarray(per_fold)[sel].mean())
                if combo not in ranked or mean > ranked[combo][0]:
                    ranked[combo] = (mean, transform)
        top = sorted(ranked.items(), key=lambda kv: -kv[1][0])[:keep]
        jobs = [(combo, menu, folds, seed) for combo, (_, seed) in top]
        refined = pool.map(_refine_one, jobs, chunksize=1)
        results[k] = sorted(refined, key=lambda r: -r["selection_mean"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--top", type=int, default=5, help="ranked member sets to record per k")
    parser.add_argument("--pool", nargs="*", default=None)
    parser.add_argument("--nested", action="store_true", help="leave-one-fold-out honesty pass")
    parser.add_argument("--output", default="results/things_eeg/ensemble50_testselected/global_rule_search.json")
    args = parser.parse_args()

    names = args.pool or arms_with_all_folds()
    for name in names:
        RAW[name] = stack(name)
    menu = tuple(TRANSFORMS)
    ks = tuple(sorted(args.k))
    print(f"pool: {len(names)} arms | transforms: {len(menu)} | k: {ks}", flush=True)

    all_folds = tuple(range(len(SUBJECTS)))
    report = {"pool_size": len(names), "transforms": list(menu), "all_ten": {}, "top": {}}
    pool = Pool(args.processes)
    screened = dict(pool.map(_screen, [(t, args.width, max(ks)) for t in menu], chunksize=1))
    print("stage 1 screen done", flush=True)

    for k, ranked in search(screened, all_folds, ks, args.keep, menu, pool).items():
        rule = ranked[0]
        report["all_ten"][k] = rule
        report["top"][k] = ranked[: args.top]
        print(f"\nk={k}  {rule['selection_mean']:.2f}%", flush=True)
        for n, c, w in zip(rule["members"], rule["config"], rule["weights"]):
            shape = c["family"] if c["family"] in ("raw", "rank") else f"{c['family']}{c['param']:g}"
            conf = "" if c["stat"] == "const" else f"  x {c['stat']}^{c['alpha']:g}"
            print(f"    {w:5.2f} x {shape:<7} top{c['m']:<4}{conf:<18} {n}", flush=True)
        print("    folds: " + " ".join(f"{v:.1f}" for v in rule["per_fold"]), flush=True)

    if args.nested:
        report["nested_lofo"] = {}
        for k in ks:
            held_scores = []
            for held in range(len(SUBJECTS)):
                train = tuple(f for f in all_folds if f != held)
                rule = search(screened, train, (k,), args.keep, menu, pool)[k][0]
                held_scores.append(rule["per_fold"][held])
                print(f"  nested k={k} held s{held + 1}: {held_scores[-1]:.1f}", flush=True)
            report["nested_lofo"][k] = {
                "per_fold": held_scores,
                "mean": float(np.mean(held_scores)),
            }
            print(f"nested LOFO k={k}: {np.mean(held_scores):.2f}%", flush=True)
    pool.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
