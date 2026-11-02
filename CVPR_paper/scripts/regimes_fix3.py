"""E4.3/E4.4 fix search v3: estimate the support with SCORE, then run SAGE on it.

regimes_fix.py: SAGE restricted to the TRUE present concepts ('oracle_support') reaches 82.3 in R4 even with 1654
distractors (SCORE 26.4), so support estimation is the bottleneck. v1's shortlist came from raw CSLS and failed; here
the shortlist is the union of each query's top-k under SCORE's ALIGNED scores (k in 1/2/3), then SAGE (whitening saw
or mm, plan bal or free, 4 or 16 steps) is fitted and decoded on that sub-gallery only.
Same problems and seeds as regimes_fix.py. Writes fix3_rows.csv (+ the shortlist's recall of the true support).
usage: regimes_fix3.py --source_run_dir R --params_csv C --output_dir O [--subjects ...] [--draws 2]
"""
import argparse, csv, itertools, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from regimes_fix import sage_variant  # noqa: E402  (also puts tta_rebuttal on sys.path)
from shared import encode_repetition_blocks, load_subject_context  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from regimes import r3_stream  # noqa: E402

GRID = list(itertools.product([1, 2, 3], ['saw', 'mm'], ['bal', 'free'], [4, 16]))


def evaluate(q, y, g, p):
    sc = score_deploy(q, g)
    acc = {'score': topk_acc(sc, y)[1]}
    for k in (1, 2, 3):
        cols = np.unique(np.argsort(-sc, axis=1)[:, :k])
        acc[f'stat_recall_k{k}'] = float(np.isin(np.unique(y), cols).mean())  # true support kept
        acc[f'stat_size_k{k}'] = len(cols) / len(g)
    for k, white, plan, steps in GRID:
        cols = np.unique(np.argsort(-sc, axis=1)[:, :k])
        acc[f'sc{k}_{white}_{plan}_s{steps}'] = topk_acc(sage_variant(q, g, p, white, plan, 0.0, steps, keep=cols), y)[1]
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--draws', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'fix3_rows.csv')
    done = {int(r['subject']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    f = open(out, 'a', newline='')
    w = csv.writer(f)
    if not done:
        w.writerow(['subject', 'regime', 'law', 'level', 'draw', 'method', 'top1'])
    for sid in args.subjects:
        if sid in done:
            continue
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        blocks, g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[10] * 8, seed=2000 + sid)
        (q80,), _, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
        rows = []
        for law in ('uniform', 'dirichlet'):
            for U in (25, 50, 100, 200):
                for d in range(args.draws):
                    q, y = r3_stream(blocks, U, law, np.random.default_rng(100_000 * sid + 1000 * U + d + (0 if law == 'uniform' else 7)))
                    rows += [(sid, 'R3', law, U, d, m, a) for m, a in evaluate(q, y, g, p).items()]
        raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy')).astype(np.float32)
        raw = raw.reshape(1654, -1, raw.shape[-1])[:, 0]
        distractors = project_raw_images(ea, mods, raw[np.random.default_rng(sid).permutation(1654)])
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, distractors[:D]], axis=0)
            rows += [(sid, 'R4', 'open', D, 0, m, a) for m, a in evaluate(q80, np.arange(200), gal, p).items()]
        w.writerows(rows)
        f.flush()
        print(f'subject {sid:02d} done', flush=True)
    f.close()


if __name__ == '__main__':
    main()
