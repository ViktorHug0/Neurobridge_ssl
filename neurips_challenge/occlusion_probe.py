"""Which part of the -0.2..0.8 s epoch do the trained encoders use? Sealed rule (subject_agg top-5, fraction 0.5) on
cv4 fold-1 test subjects, with parts of every input replaced by an image-agnostic template: the mean epoch of a
fold-1 TRAINING subject (sub-01), so the replaced span keeps a realistic ERP shape but carries no trial identity.
  occ<t>   : samples [t, t+12) (100 ms) replaced           -> drop = what that window contributes
  keep<a-b>: only samples [a, b) kept, the rest replaced  -> what a crop to [a, b) would keep
Sample 24 = stimulus onset, 120 Hz.
    occlusion_probe.py <recipe> [<recipe> ...]
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402
from neurips_challenge import score_track1 as st  # noqa: E402
from neurips_challenge.cv4_ensemble import DATA, run_dirs  # noqa: E402
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

orig = train.run_eeg_backbone
VIEWS = {'none': None, **{f'occ{t}': (t, t + 12) for t in range(0, 109, 6)},
         **{f'keep{a}-{b}': (a, b) for a, b in [(24, 60), (24, 84), (30, 96), (30, 72), (0, 84), (24, 120)]}}

for r in sys.argv[1:]:
    d = run_dirs(r)[1]
    cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
    ck = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location='cuda', weights_only=False)
    ds = st.build_test_dataset(cfg, [1])
    tpl = torch.stack([torch.as_tensor(ds[int(i)][0]) for i in np.random.default_rng(0).choice(len(ds), 3000, replace=False)]).mean(0).cuda()
    mods = st.build_modules(cfg, ck, ds, 'cuda', int(cfg.time_window[1]) - int(cfg.time_window[0]))
    out = {}
    for name, (a, b) in ((n, v or (0, 0)) for n, v in VIEWS.items()):
        mask = torch.zeros(tpl.shape[-1], device='cuda', dtype=torch.bool); mask[a:b] = True
        if name.startswith('keep'):
            mask = ~mask                                        # replace everything outside [a, b)
        train.run_eeg_backbone = lambda m, c, x, s, mask=mask: orig(m, c, torch.where(mask, tpl, x), s)
        res = st.score_modules(*mods, cfg, FOLDS[1], 'cuda', 128, 2, fraction=0.5)
        out[name] = round(100 * res['subject_agg']['top5'], 2)
        print(r, name, out[name], flush=True)
    train.run_eeg_backbone = orig
    print('RESULT', r, json.dumps(out), flush=True)
