"""plain + SCORE deployment (R1, 80 reps) on one SAMGA/SCORE variant run dir; ckpt via SAGE_CHECKPOINT_NAME."""
import os, sys, csv
from types import SimpleNamespace
import numpy as np
os.chdir('/nasbrain/p20fores/Neurobridge_SSL')
sys.path.insert(0, 'scripts/things_eeg/tta_rebuttal'); sys.path.insert(0, 'CVPR_paper/scripts')
from shared import encode_repetition_blocks, load_subject_context, score_features
from label_free import score_deploy
src, out = sys.argv[1], sys.argv[2]
subs = sorted(int(d.split('sub-')[1][:2]) for d in os.listdir(src) if 'sub-' in d and os.path.exists(f'{src}/{d}/result.csv'))
rows = []
for sid in subs:
    rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
    _, ea, ds, mods = load_subject_context(os.path.abspath(src), rt, sid, average=False)
    (q,), g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
    y = np.arange(len(q))
    for m, s in (('plain', score_features(q, g)), ('score', score_deploy(q, g))):
        rows.append(dict(subject=sid, method=m, top1=100 * (s.argmax(1) == y).mean()))
    print(sid, rows[-2:], flush=True)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['subject', 'method', 'top1']); w.writeheader(); w.writerows(rows)
