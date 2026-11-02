"""Ensembles of cv4 recipes, scored under the sealed rule on all 20 cv4 test subjects (80 repeats).

    cv4_ensemble.py cache tsc lab_s10 ...     per fold run: per-(subject, image) mean predictions -> ensemble_preds.npz
    cv4_ensemble.py search [--max 5] [--pool r1 r2 ...]   greedy + exhaustive search (pool: one version per recipe)

Combination = what a submission can do per trial: each member's output divided by that model's mean prediction
norm, members summed. The grader's repeat averaging is linear, so the per-(subject, image) means of the members
combine the same way. Uses each run's best-epoch checkpoint (checkpoint_sealed_best.pth) when saved, else its last epoch. Member choice is
made on the same 20 subjects it is scored on (optimistic for the chosen set, mildly for 6 candidates).
"""
import glob
import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.cv4_summary import ROOT, decide  # noqa: E402

DATA = "/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench"


def run_dirs(recipe):
    return {int(f.split("/")[-4][4:]): os.path.dirname(f) for f in glob.glob(f"{ROOT}/{recipe}/fold*/*/*/track1_score.json")}


def cache(recipes):
    import torch
    from neurips_challenge.score_track1 import build_modules, build_test_dataset, score_modules
    device = torch.device("cuda:0")
    for recipe in recipes:
        for k, d in sorted(run_dirs(recipe).items()):
            out = os.path.join(d, "ensemble_preds.npz")
            if os.path.exists(out):
                continue
            cfg = SimpleNamespace(**json.load(open(os.path.join(d, "train_config.json"))))
            cfg.eeg_data_dir = DATA  # the run's staged /tmp copy is gone
            # the run's best epoch when it saved one (runs since 2026-10-02), else its last epoch
            best = os.path.exists(os.path.join(d, "checkpoint_sealed_best.pth"))
            name, score_file = (("checkpoint_sealed_best.pth", "track1_score_sealed.json") if best
                                else ("checkpoint_test_best.pth", "track1_score.json"))
            ckpt = torch.load(os.path.join(d, name), map_location=device, weights_only=False)
            n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
            mods = build_modules(cfg, ckpt, build_test_dataset(cfg, [cfg.test_subject_ids[0]]), device, n_times)
            with torch.no_grad():
                r = score_modules(*mods, cfg, cfg.test_subject_ids, device, 512, 0, return_preds=True)
            ref = json.load(open(os.path.join(d, score_file)))["subject_agg_by_subject"]
            for s, v in r["subject_agg_by_subject"].items():  # same checkpoint, same data: must reproduce the run's score
                assert abs(v["top5"] - ref[s]["top5"]) < 0.011, (recipe, k, s, v["top5"], ref[s]["top5"])
            np.savez(out + ".tmp.npz", gallery=r["gallery_normed"], mean_norm=r["mean_norm"],
                     **{f"s{s}": p for s, p in r["subject_mean_preds"].items()})
            os.replace(out + ".tmp.npz", out)
            print(f"cached {recipe} fold{k}", flush=True)


def load(recipe):
    """{subject: (200, D) member-scaled mean predictions}, gallery; None unless all 4 folds are cached."""
    dirs = run_dirs(recipe)
    if sorted(dirs) != [1, 2, 3, 4] or not all(os.path.exists(f"{d}/ensemble_preds.npz") for d in dirs.values()):
        return None
    preds, gallery = {}, None
    for d in dirs.values():
        z = np.load(f"{d}/ensemble_preds.npz")
        gallery = z["gallery"] if gallery is None else gallery
        assert np.allclose(gallery, z["gallery"], atol=1e-4), "members must rank against the same targets"
        preds.update({key[1:]: z[key] / z["mean_norm"] for key in z.files if key.startswith("s")})
    return preds, gallery


def top5_per_subject(members, gallery):
    out = {}
    for s in members[0]:
        p = sum(m[s] for m in members)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        top = np.argsort(p @ gallery.T, axis=1)[:, -5:]
        out[s] = 100 * float(np.mean(np.any(top == np.arange(len(p))[:, None], axis=1)))
    return out


def search(max_size, only=None):
    pool = {}
    for r in only or sorted({p.split("/")[-5] for p in glob.glob(f"{ROOT}/*/fold*/*/*/ensemble_preds.npz")}):
        loaded = load(r)
        if loaded:
            pool[r] = loaded
    gallery = next(iter(pool.values()))[1]
    score = lambda names: top5_per_subject([pool[n][0] for n in names], gallery)  # noqa: E731
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    singles = {n: score([n]) for n in pool}
    print("singles (best-epoch checkpoints where saved, 80 repeats, 20 subjects):")
    for n in sorted(singles, key=lambda n: -mean(singles[n])):
        print(f"  {n:14s} {mean(singles[n]):.2f}")
    ref = singles.get("tsc", singles.get("tsc_b11"))
    chosen = [max(singles, key=lambda n: mean(singles[n]))]
    print("\ngreedy:")
    while len(chosen) < min(max_size, len(pool)):
        best = max((n for n in pool if n not in chosen), key=lambda n: mean(score(chosen + [n])))
        chosen.append(best)
        s = score(chosen)
        vs = decide([s[k] - ref[k] for k in ref], {1, 2, 3, 4}, False) if ref else None
        print(f"  {len(chosen)}: {' + '.join(chosen):60s} {mean(s):.2f}" + (f"  vs tsc {vs[0]:+.2f}, better on {vs[1]}/20, p {vs[3]:.3f}" if vs else ""))
    print("\nexhaustive, best per size:")
    for size in range(2, min(max_size, len(pool)) + 1):
        best = max(itertools.combinations(pool, size), key=lambda c: mean(score(list(c))))
        print(f"  {size}: {' + '.join(best):60s} {mean(score(list(best))):.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache(sys.argv[2:])
    else:
        pool = sys.argv[sys.argv.index("--pool") + 1:] if "--pool" in sys.argv else None
        search(int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 5, pool)
