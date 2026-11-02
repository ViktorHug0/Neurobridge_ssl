"""E5.1/E5.2 label-equivalence with matched scoring: how many LABELED target trials does unlabeled SAGE equal?

run_inductive_rotation_sweep.py scores its supervised maps with plain cosine, while SAGE ends with CSLS (a transductive
re-ranking), so the comparison mixed two things. Here every supervised map is also scored with CSLS, and E5.2
(labels + the unlabeled test batch) is added:
  sup_rot_{cos,csls}    orthogonal Procrustes fitted on n labeled TRAIN-split pairs of the target subject
                        (4-rep averages, concepts disjoint from the test set), applied to the 80-rep test queries
  sup_ridge_{cos,csls}  ridge map Q W ~ I on the same pairs (ridge 1.0)
  sup_rot_sage          the supervised rotation, then SAGE (cross-fold params) on the unlabeled test batch
references: plain, csls3, sage (unlabeled only). n in --sizes, 3 seeds per n.
usage: label_equiv2.py --source_run_dir R --params_csv C --output_csv O [--subjects ...]
"""
import argparse, os, sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import (apply_linear_map, build_test_dataset, apply_tta_calibration, encode_indexed_average_features, encode_repetition_blocks,  # noqa: E402
                    fit_paired_orthogonal, fit_paired_ridge, fit_tta_calibration, load_subject_train_test_context,
                    normalize_rows, score_features)
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--sizes', nargs='+', type=int, default=[50, 200, 1000, 5000, 16540])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows = []
    for sid in args.subjects:
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, train_ds, _, mods = load_subject_train_test_context(os.path.abspath(args.source_run_dir), rt, sid, average=True)
        tq, ti, _, _ = encode_indexed_average_features(ea, mods, train_ds)
        test_ds = build_test_dataset(ea, sid, average=False)  # single repetitions, for the 80-rep block encoding
        (q,), g, _ = encode_repetition_blocks(ea, mods, test_ds, block_sizes=[80], seed=3300)
        y = np.arange(len(q))
        acc = lambda x, csls: topk_acc(score_features(x, g, use_csls=csls, csls_k=p.csls_k), y)[1]  # noqa: E731
        sage = lambda x: acc(apply_tta_calibration(x, fit_tta_calibration(x, g, p)), True)  # noqa: E731
        ref = {'plain': acc(q, False), 'csls3': acc(q, True), 'sage': sage(q)}
        for n in args.sizes:
            for seed in range(3):
                idx = np.random.default_rng(100_000 * seed + 1000 * sid + n).choice(len(tq), min(n, len(tq)), replace=False)
                R = fit_paired_orthogonal(normalize_rows(tq[idx]), normalize_rows(ti[idx]))
                Wr = fit_paired_ridge(tq[idx], ti[idx], ridge=1.0)
                qr, qw = normalize_rows(q @ R), apply_linear_map(q, Wr)
                res = dict(ref, sup_rot_cos=acc(qr, False), sup_rot_csls=acc(qr, True), sup_ridge_cos=acc(qw, False),
                           sup_ridge_csls=acc(qw, True), sup_rot_sage=sage(qr))
                rows += [{'subject': sid, 'n': n, 'seed': seed, 'method': m, 'top1': v} for m, v in res.items()]
        print(sid, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.groupby(['n', 'method'])['top1'].mean().unstack().round(1).to_string())


if __name__ == '__main__':
    main()
