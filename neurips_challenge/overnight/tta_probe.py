"""Test-time augmentation probe: average each trial's prediction over augmented copies, sealed rule, fraction 0.5.
    tta.py <ckpt_dir> [<ckpt_dir> ...]"""
import json, os, sys
from types import SimpleNamespace
import torch
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL'); sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL/neurips_challenge')
import train, score_track1 as st

orig = train.run_eeg_backbone


def shift(x, s):  # edge-replicated time shift by s samples
    if s == 0: return x
    return torch.cat([x[..., :1].expand(*x.shape[:-1], s), x[..., :-s]], -1) if s > 0 else \
        torch.cat([x[..., -s:], x[..., -1:].expand(*x.shape[:-1], -s)], -1)


VIEWS = {
    'none': [lambda x: x],
    'shift1': [lambda x, s=s: shift(x, s) for s in (-1, 0, 1)],
    'shift2': [lambda x, s=s: shift(x, s) for s in (-2, -1, 0, 1, 2)],
    'shift4': [lambda x, s=s: shift(x, s) for s in (-4, -2, 0, 2, 4)],
    'scale': [lambda x, g=g: g * x for g in (0.8, 1.0, 1.25)],
    'chdrop': [lambda x, i=i: x * (torch.rand(x.shape[:2] + (1,), generator=torch.Generator().manual_seed(i)).to(x.device) > 0.1)
               for i in range(5)],
}

for d in sys.argv[1:]:
    cfg = SimpleNamespace(**json.load(open(os.path.join(d, 'train_config.json'))))
    ck = torch.load(os.path.join(d, 'checkpoint_sealed_best.pth'), map_location='cuda', weights_only=False)
    probe = st.build_test_dataset(cfg, [cfg.test_subject_ids[0]])
    mods = st.build_modules(cfg, ck, probe, 'cuda', int(cfg.time_window[1]) - int(cfg.time_window[0]))
    out = {}
    for name, views in VIEWS.items():
        train.run_eeg_backbone = lambda m, c, x, sub, views=views: torch.stack([orig(m, c, v(x), sub) for v in views]).mean(0)
        r = st.score_modules(*mods, cfg, cfg.test_subject_ids, 'cuda', 128, 2, fraction=0.5)
        out[name] = round(100 * r['subject_agg']['top5'], 2)
        print(d.split('/cv4/')[1][:40], name, out[name], flush=True)
    train.run_eeg_backbone = orig
    print('RESULT', d.split('/cv4/')[1], json.dumps(out), flush=True)
