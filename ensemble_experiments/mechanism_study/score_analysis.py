"""Label-informed interventions and geometry diagnostics; never deployable fusion."""
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import rankdata
from .common import COMMITTEES, OUTPUT, load_member, margin_metrics, row_z, write_json


def rearrange(scores, mode, rng, geometry=None):
    out = scores.copy()
    members, queries, candidates = scores.shape
    for q in range(queries):
        negatives = np.delete(np.arange(candidates), q)
        groups = [negatives]
        if mode == 'stratified':
            groups = np.array_split(negatives[np.argsort(geometry[q, negatives])], 10)
        for group in groups:
            for m in range(members):
                values = scores[m, q, group]
                out[m, q, group] = np.sort(values) if mode == 'aligned' else rng.permutation(values)
    # True score and full multiset are preserved, hence true rank is unchanged.
    np.testing.assert_array_equal(np.sort(out, axis=-1), np.sort(scores, axis=-1))
    return out


def convex_oracle(scores):
    z = row_z(scores)
    members, queries, candidates = z.shape
    q = np.arange(queries)
    feasible = (z.argmax(-1) == q).any(0)
    for query in np.flatnonzero(~feasible):
        differences = z[:, query, query, None] - z[:, query, :]
        differences = np.delete(differences, query, axis=1)
        result = linprog(
            np.r_[np.zeros(members), -1.],
            A_ub=np.column_stack((-differences.T, np.ones(candidates - 1))),
            b_ub=np.zeros(candidates - 1),
            A_eq=[np.r_[np.ones(members), 0.]], b_eq=[1.],
            bounds=[(0, None)] * members + [(None, None)], method='highs',
        )
        if not result.success:
            raise RuntimeError(result.message)
        feasible[query] = result.x[-1] > 1e-7
    return float(feasible.mean() * 100)


def geometry_rows(scores, kernels, subject, names):
    """Matched wrong-candidate rank percentiles, plus rescue/damage counts."""
    rows = []
    q = np.arange(scores.shape[1])
    ranks = []
    for kernel in kernels:
        masked = kernel.copy()
        masked[q, q] = -np.inf
        ranks.append((rankdata(masked, axis=1) - 2) / (len(q) - 2))
    for a, b in itertools.permutations(range(len(names)), 2):
        pa, pb = scores[a].argmax(-1), scores[b].argmax(-1)
        wrong = pa != q
        fused = row_z(scores[[a, b]]).mean(0).argmax(-1) == q
        own = ranks[a][q, pa]
        alternate = ranks[b][q, pa]
        rows.append({
            'subject': subject, 'member': names[a], 'partner': names[b],
            'n_errors': int(wrong.sum()),
            'wrong_winner_own_geometry_percentile': float(own[wrong].mean()),
            'wrong_winner_other_geometry_percentile': float(alternate[wrong].mean()),
            'geometry_rank_gap': float((own - alternate)[wrong].mean()),
            'rescues': int((wrong & fused).sum()),
            'damage': int((~wrong & ~fused).sum()),
            'net_gain_pp': float((fused.mean() - (~wrong).mean()) * 100),
            'partner_solo_correct_on_errors': float((pb[wrong] == q[wrong]).mean()),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, default=str(OUTPUT / 'score_analysis'))
    p.add_argument('--repeats', type=int, default=100)
    args = p.parse_args()
    from pathlib import Path
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    rows, geometry = [], []
    for committee, names in COMMITTEES.items():
        for subject in range(1, 11):
            loaded = [load_member(n, subject) for n in names]
            for item in loaded[1:]:
                np.testing.assert_array_equal(item[2], loaded[0][2])
            scores = np.stack([x[0] for x in loaded])
            kernels = np.stack([x[1] for x in loaded])
            original = margin_metrics(scores)
            original['convex_oracle_top1'] = convex_oracle(scores)
            rows.append(dict(committee=committee, subject=subject, mode='original', repeat=0, **original))
            for mode in ['aligned', 'permuted', 'stratified']:
                for repeat in range(1 if mode == 'aligned' else args.repeats):
                    changed = rearrange(scores, mode, np.random.default_rng(20260907 + 1000 * subject + repeat), kernels.mean(0))
                    rows.append(dict(committee=committee, subject=subject, mode=mode, repeat=repeat, **margin_metrics(changed)))
            geometry.extend(dict(committee=committee, **r) for r in geometry_rows(scores, kernels, subject, names))
            print(committee, subject, original, flush=True)
            pd.DataFrame(rows).to_csv(out / 'interventions.csv', index=False)
            pd.DataFrame(geometry).to_csv(out / 'geometry.csv', index=False)
    frame = pd.DataFrame(rows)
    # Average randomizations inside folds before averaging subjects.
    folds = frame.groupby(['committee', 'mode', 'subject']).mean(numeric_only=True).reset_index()
    folds.groupby(['committee', 'mode']).mean(numeric_only=True).to_csv(out / 'summary.csv')
    write_json(out / 'complete.json', {'protocol': 'historical checkpoint diagnostics; labels used only for analysis', 'repeats': args.repeats})


if __name__ == '__main__':
    main()
