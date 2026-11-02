"""Cache track1_scores.npy for the given run dirs (the matrices ensemble_track1.py sweeps)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.ensemble_track1 import CACHE, RESULTS, score_matrix  # noqa: E402

for run_dir in sys.argv[1:]:
    scores, labels = score_matrix(run_dir, torch.device("cuda"))
    assert np.array_equal(labels, np.load(os.path.join(RESULTS, "ensembles/labels.npy")))
    np.save(os.path.join(run_dir, CACHE), scores)
    print("cached", run_dir, flush=True)
