"""E8.5 RSA-loss screen: per subject of one LOSO run dir, R1 (80 reps) zero-shot, Hungarian, SCORE deployment, SAGE
(hyper-parameters frozen from a reference arm's cross-fold screen, e.g. atm128_s3300, so arms differ only by the
encoder), plus the E8.4 diagnostics RSA and oracle in-sample Procrustes.
usage: rsa_screen_eval.py <run_dir> <params_csv> <out_csv> [subjects...]
"""
import csv, os, sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_tta_calibration, encode_repetition_blocks, fit_tta_calibration, load_subject_context, score_features  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def nrm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    run, params_csv, out = sys.argv[1:4]
    subs = [int(s) for s in sys.argv[4:]] or list(range(1, 11))
    rows = []
    for sid in subs:
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(run), rt, sid, average=False)
        (q,), g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
        qn, gn = nrm(np.asarray(q, float)), nrm(np.asarray(g, float))
        y = np.arange(len(qn))
        p = subject_params(params_csv, sid)
        s = qn @ gn.T
        r, c = linear_sum_assignment(-s)
        qq, gg = nrm(qn - qn.mean(0)), nrm(gn - gn.mean(0))
        iu = np.triu_indices(len(qn), 1)
        u, _, vt = np.linalg.svd(qn.T @ gn)
        uc, _, vtc = np.linalg.svd(qq.T @ gg)  # centred clouds, as SAW / the RSA loss see them
        rec = dict(subject=sid,
                   plain=100 * (s.argmax(1) == y).mean(),
                   hungarian=100 * (c[np.argsort(r)] == y).mean(),
                   score=100 * (score_deploy(q, g).argmax(1) == y).mean(),
                   sage=100 * (score_features(apply_tta_calibration(q, fit_tta_calibration(q, g, p)), g, use_csls=True,
                                              csls_k=p.csls_k).argmax(1) == y).mean(),
                   rsa=np.corrcoef((qq @ qq.T)[iu], (gg @ gg.T)[iu])[0, 1],
                   oracle_in=100 * ((qn @ u @ vt @ gn.T).argmax(1) == y).mean(),
                   oracle_c=100 * ((qq @ uc @ vtc @ gg.T).argmax(1) == y).mean())
        rows.append(rec)
        print(rec, flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
