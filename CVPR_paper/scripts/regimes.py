"""E4.3 / E4.4: coverage-multiplicity (R3) and open-gallery (R4) regimes, every label-free method on one model.

R3  fixed 200-item gallery; the unlabeled batch covers only U concepts. Queries are 10-repetition averages (8 disjoint
    examples per concept). Two multiplicity laws: 'uniform' (1..4 examples per present concept, the rebuttal's) and
    'dirichlet' (Veilleux et al. 2021: N = 3U queries, proportions ~ Dir(1) over the U concepts, >= 1 and <= 8 each).
R4  80-repetition queries of all 200 concepts; the gallery adds D distractor images (one per TRAINING concept,
    disjoint from the test concepts) pushed through the same frozen image projector.
Methods: plain, csls3, score, svtl, sage (balanced Sinkhorn, the published loop), sage_free (rho = 0, free column
marginal) and sage_adapt (entropy-gated rho, run_partial_coverage_ablations.adaptive_rho). SAGE hyper-parameters per
subject are the cross-fold ones from --params_csv (chosen on the 200x200 regime of the other folds).
usage: regimes.py --source_run_dir R --params_csv C --output_dir O [--subjects ...]
"""
import argparse, csv, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import encode_repetition_blocks, load_subject_context, score_features  # noqa: E402
from run_partial_coverage import tta_scores, topk_acc  # noqa: E402
from run_partial_coverage_ablations import adaptive_rho  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from label_free import score_deploy, svtl_refine  # noqa: E402
from stream_grid import subject_params  # noqa: E402

U_LIST = [25, 50, 75, 100, 150, 200]
D_LIST = [0, 200, 400, 800, 1654]


def methods(q, g, params):
    init = score_features(q, g, use_csls=True, csls_k=params.csls_k)
    rho_a, _ = adaptive_rho(init, g.shape[0])
    return {
        'plain': score_features(q, g), 'csls3': init, 'score': score_deploy(q, g), 'svtl': svtl_refine(q, g),
        'sage': tta_scores(q, g, params, rho=1.0), 'sage_free': tta_scores(q, g, params, rho=0.0),
        'sage_adapt': tta_scores(q, g, params, rho=rho_a),
    }


def r3_stream(blocks, U, law, rng):
    concepts = rng.choice(blocks[0].shape[0], size=U, replace=False)
    if law == 'uniform':
        counts = rng.integers(1, 5, size=U)
    else:
        counts = 1 + rng.multinomial(2 * U, rng.dirichlet(np.ones(U)))
        counts = np.minimum(counts, len(blocks))
    q, y = [], []
    for c, m in zip(concepts, counts):
        for b in rng.choice(len(blocks), size=int(m), replace=False):
            q.append(blocks[b][c])
            y.append(int(c))
    return np.stack(q).astype(np.float32), np.asarray(y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_run_dir', required=True)
    p.add_argument('--params_csv', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--draws', type=int, default=5)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'regime_rows.csv')
    done = {int(r['subject']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    f = open(out, 'a', newline='')
    w = csv.writer(f)
    if not done:
        w.writerow(['subject', 'regime', 'law', 'level', 'draw', 'n_queries', 'method', 'top1', 'top5'])
    for sid in args.subjects:
        if sid in done:
            continue
        params = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, eval_args, dataset, modules = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        blocks, g, _ = encode_repetition_blocks(eval_args, modules, dataset, block_sizes=[10] * 8, seed=2000 + sid)
        (q80,), _, _ = encode_repetition_blocks(eval_args, modules, dataset, block_sizes=[80], seed=3300)
        rows = []
        for law in ('uniform', 'dirichlet'):
            for U in U_LIST:
                for d in range(args.draws):
                    q, y = r3_stream(blocks, U, law, np.random.default_rng(100_000 * sid + 1000 * U + d + (0 if law == 'uniform' else 7)))
                    for m, s in methods(q, g, params).items():
                        a = topk_acc(s, y)
                        rows.append((sid, 'R3', law, U, d, len(q), m, a[1], a[5]))
        raw = np.load(os.path.join(eval_args.image_feature_dir, 'image_train.npy')).astype(np.float32)
        raw = raw[:, 0] if raw.ndim == 4 else raw.reshape(1654, -1, raw.shape[-1])[:, 0]  # one image per training concept (4-D: SAMGA stacked layers)
        distractors = project_raw_images(eval_args, modules, raw[np.random.default_rng(sid).permutation(1654)])
        y = np.arange(200)
        for D in D_LIST:
            gal = np.concatenate([g, distractors[:D]], axis=0)
            for m, s in methods(q80, gal, params).items():
                a = topk_acc(s, y)
                rows.append((sid, 'R4', 'open', D, 0, 200, m, a[1], a[5]))
        w.writerows(rows)
        f.flush()
        print(f'subject {sid:02d} done', flush=True)
    f.close()


if __name__ == '__main__':
    main()
