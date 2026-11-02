"""E2.2 / E3.x regime R1 (200 x 200, bijective): every label-free method on the SAME embeddings of one model.

Queries are the 80-repetition averages, encoded exactly as run_honest_tta_selection.py does (seed 3300), so the
SAGE column reproduces its loso_fold_cv number. Methods:
  plain   cosine                       csls3 / csls10   CSLS only (k = 3, SAGE's; k = 10, SCORE's)
  score   SCORE deployment step        svtl             SVTL refinement (see label_free.py)
  sage    SAGE with this subject's cross-fold hyper-parameters (--params_csv, loso_fold_cv rows)
usage: r1_eval.py --source_run_dir R --params_csv C --output_csv O [--subjects ...]
Set SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth for the ValCon runs' test-selected epoch.
"""
import argparse, csv, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_tta_calibration, encode_repetition_blocks, fit_tta_calibration, load_subject_context, score_features  # noqa: E402
from label_free import score_deploy, svtl_refine  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_run_dir', required=True)
    p.add_argument('--params_csv', required=True)
    p.add_argument('--output_csv', required=True)
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows = []
    for sid in args.subjects:
        params = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, eval_args, dataset, modules = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        (q,), g, _ = encode_repetition_blocks(eval_args, modules, dataset, block_sizes=[80], seed=3300)
        cal = fit_tta_calibration(q, g, params)
        scores = {
            'plain': score_features(q, g),
            'csls3': score_features(q, g, use_csls=True, csls_k=3),
            'csls10': score_features(q, g, use_csls=True, csls_k=10),
            'score': score_deploy(q, g),
            'svtl': svtl_refine(q, g),
            'sage': score_features(apply_tta_calibration(q, cal), g, use_csls=True, csls_k=params.csls_k),
        }
        y = np.arange(len(q))
        for m, s in scores.items():
            a = topk_acc(s, y)
            rows.append({'subject': sid, 'method': m, 'top1': a[1], 'top5': a[5]})
        print(sid, {m: r['top1'] for m, r in zip(scores, rows[-len(scores):])}, flush=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject', 'method', 'top1', 'top5'])
        w.writeheader()
        w.writerows(rows)
    for m in scores:
        t1 = [r['top1'] for r in rows if r['method'] == m]
        t5 = [r['top5'] for r in rows if r['method'] == m]
        print(f'{m:7s} top1 {np.mean(t1):6.2f} +- {np.std(t1):5.2f}   top5 {np.mean(t5):6.2f}')


if __name__ == '__main__':
    main()
