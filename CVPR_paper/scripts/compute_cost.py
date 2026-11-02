"""E8.4 wall-clock of one label-free calibration on CPU (R1, 200 x 200, cached embeddings of one subject).
Methods: plain cosine, CSLS-3, SCORE deployment step, SAGE (that subject's cross-fold params; 16 steps).
Median of --repeats runs; thread count from OMP_NUM_THREADS.
usage: OMP_NUM_THREADS=1 compute_cost.py --cache_npz E --params_csv C --subject S [--repeats 20]
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_tta_calibration, fit_tta_calibration, score_features  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--cache_npz', required=True)
ap.add_argument('--params_csv', required=True)
ap.add_argument('--subject', type=int, required=True)
ap.add_argument('--repeats', type=int, default=20)
a = ap.parse_args()
z = np.load(a.cache_npz)
q, g = z['q'], z['g']
p = subject_params(a.params_csv, a.subject)
methods = {'plain': lambda: score_features(q, g), 'csls3': lambda: score_features(q, g, use_csls=True, csls_k=3),
           'score': lambda: score_deploy(q, g),
           'sage': lambda: score_features(apply_tta_calibration(q, fit_tta_calibration(q, g, p)), g, use_csls=True, csls_k=p.csls_k)}
print(f'q {q.shape} g {g.shape} threads {os.environ.get("OMP_NUM_THREADS")} steps {p.soft_procrustes_steps}')
for name, f in methods.items():
    f()  # warm-up
    t = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    print(f'{name:6s} median {1000 * np.median(t):8.1f} ms')
