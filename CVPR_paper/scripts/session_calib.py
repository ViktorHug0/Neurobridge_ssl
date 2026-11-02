"""E6.2 session-level calibration on the test split (recording sessions recovered from the preprocessing shuffle).

The released THINGS-EEG2 test arrays concatenate 4 sessions x 20 repetitions and then permute the 80 repetitions
with sklearn.utils.shuffle(random_state=20200220) (Gifford et al. preprocessing_utils.save_prepr), so
session(rep i) = shuffle(arange(80), 20200220)[i] // 20 (verified: per-repetition channel-covariance fingerprints
correlate 0.37 within vs -0.10 across recovered sessions, permutation null 99th pct 0.03).
Per subject and session s: queries = the 20 repetitions of session s averaged. Methods:
  plain | sage_own (fit on session s itself) | sage_cross (fit on another session s', applied frozen to s; mean over s')
  | plain_mix / sage_mix: 20 repetitions drawn across sessions (same SNR, no session coherence).
SAGE hyper-parameters: cross-fold ones from --params_csv (chosen at 80 repetitions on the other folds).
usage: session_calib.py --source_run_dir R --params_csv C --output_csv O [--subjects ...]
"""
import argparse, os, sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import _test_image_matrix, apply_tta_calibration, fit_tta_calibration, load_subject_context, score_features  # noqa: E402
from train import run_eeg_backbone  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402

SESSION = shuffle(np.arange(80), random_state=20200220) // 20


def encode(ea, mods, ds, reps_per_object):
    """reps_per_object: (200, r) repetition indices to average per object -> (queries, gallery)."""
    eeg = np.asarray(ds.eeg_data_list[0])[:, 0]  # (200, 80, ch, t)
    avg = np.stack([eeg[o, r].mean(0) for o, r in enumerate(reps_per_object)]).astype(np.float32)
    sid = torch.full((len(avg),), int(ea.test_subject_id), dtype=torch.long)
    with torch.no_grad():
        q = mods['eeg_projector'](run_eeg_backbone(mods['model'], ea, torch.tensor(avg), sid)).numpy()
        g = mods['img_projector'](torch.tensor(_test_image_matrix(ds), dtype=torch.float32)).numpy()
    return q, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows, y = [], np.arange(200)
    sage = lambda x, cal, p: score_features(apply_tta_calibration(x, cal), g, use_csls=True, csls_k=p.csls_k)  # noqa: E731
    for sid in args.subjects:
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        qs = []
        for s in range(4):
            q, g = encode(ea, mods, ds, [np.flatnonzero(SESSION == s)] * 200)
            qs.append(q)
        cals = [fit_tta_calibration(q, g, p) for q in qs]
        for s in range(4):
            acc = lambda sc: topk_acc(sc, y)[1]  # noqa: E731
            rows.append({'subject': sid, 'session': s, 'method': 'plain', 'top1': acc(score_features(qs[s], g))})
            rows.append({'subject': sid, 'session': s, 'method': 'sage_own', 'top1': acc(sage(qs[s], cals[s], p))})
            rows.append({'subject': sid, 'session': s, 'method': 'sage_cross',
                         'top1': np.mean([acc(sage(qs[s], cals[t], p)) for t in range(4) if t != s])})
            rng = np.random.default_rng(1000 * sid + s)
            qm, _ = encode(ea, mods, ds, [rng.choice(80, 20, replace=False) for _ in range(200)])
            rows.append({'subject': sid, 'session': s, 'method': 'plain_mix', 'top1': acc(score_features(qm, g))})
            rows.append({'subject': sid, 'session': s, 'method': 'sage_mix', 'top1': acc(sage(qm, fit_tta_calibration(qm, g, p), p))})
        print(sid, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.groupby('method')['top1'].agg(['mean', 'std']).round(2).to_string())


if __name__ == '__main__':
    main()
