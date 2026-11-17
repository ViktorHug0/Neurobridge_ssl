"""Shared helpers for the mechanistic-interpretability experiments (TSConv LOSO checkpoints).

Stages of TSConv_parameterizable, in order (eval mode):
  input (63,250) -> temporal (40,63,T) -> spatial (40,T) -> backbone (40*T) -> trunk (1024) -> align (512)
`forward_from(m, stage, act)` continues the frozen network from any stage, so an intervention is
"edit the activation at one stage, run the rest".
"""
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import TSConv_parameterizable  # noqa: E402
from module.projector import build_projector  # noqa: E402

RB = os.path.join(REPO, 'results/things_eeg/subjectmix_rebuttal')
RUNS = {'mix': f'{RB}/method_subjectmix/seed3300', 'nomix': f'{RB}/base_no_aug/seed3300'}
OUT = os.path.join(REPO, 'results/things_eeg/mi')
STAGES = ['input', 'temporal', 'spatial', 'backbone', 'trunk', 'align']
DEV = os.environ.get('MI_DEV') or ('cuda' if torch.cuda.is_available() else 'cpu')


def run_dir(run, sub):
    return glob.glob(f'{RUNS.get(run, run)}/*-sub-{sub:02d}')[0]


class M:
    """A loaded checkpoint: encoder + both projectors, all frozen, eval mode."""

    def __init__(self, run, sub):
        d = run_dir(run, sub)
        self.cfg = c = json.load(open(f'{d}/train_config.json'))
        ck = torch.load(f'{d}/checkpoint_test_best.pth', map_location=DEV)
        img_dim = ck['img_projector_state_dict']['linear.weight'].shape[1] if 'linear.weight' in ck['img_projector_state_dict'] \
            else next(iter(ck['img_projector_state_dict'].values())).shape[1]
        bb = c.get('eeg_backbone_dim', 0) or img_dim
        hd = c.get('tsconv_head_dropout', -1.0)
        self.enc = TSConv_parameterizable(
            feature_dim=bb, temporal_filters=c['tsconv_temporal_filters'], temporal_kernel=c['tsconv_temporal_kernel'],
            pool_kernel=c['tsconv_pool_kernel'], pool_stride=c['tsconv_pool_stride'],
            spatial_filters=c['tsconv_spatial_filters'], projection_filters=c['tsconv_projection_filters'],
            activation=c['tsconv_activation'], dropout=c['tsconv_dropout'], head_dropout=None if hd < 0 else hd)
        self.eeg_proj = build_projector(c['projector'], bb, c['feature_dim'])
        self.img_proj = build_projector(c['projector'], img_dim, c['feature_dim'])
        self.enc.load_state_dict(ck['model_state_dict'])
        self.eeg_proj.load_state_dict(ck['eeg_projector_state_dict'])
        self.img_proj.load_state_dict(ck['img_projector_state_dict'])
        for mod in (self.enc, self.eeg_proj, self.img_proj):
            mod.to(DEV).eval().requires_grad_(False)
        self.test_sub = c['test_subject_ids'][0]
        self.train_subs = c['train_subject_ids']


def _steps(m):
    e = m.enc
    return {
        'temporal': lambda x: e.tsconv[:4](x.unsqueeze(1)),
        'spatial': lambda x: e.tsconv[4:](x).squeeze(2),
        'backbone': lambda x: e.projection(x.unsqueeze(2)).flatten(1),
        'trunk': e.proj_eeg,
        'align': m.eeg_proj,
    }


@torch.no_grad()
def forward_from(m, stage, act, to='align', bs=512, keep=None):
    """Run the frozen network from `stage` (activation `act`) up to `to`. keep: list of stages to also return."""
    steps = _steps(m)
    i0, i1 = STAGES.index(stage), STAGES.index(to)
    outs = {k: [] for k in (keep or [])}
    res = []
    for s in range(0, len(act), bs):
        x = act[s:s + bs].to(DEV).float()
        for name in STAGES[i0 + 1:i1 + 1]:
            x = steps[name](x)
            if name in outs:
                outs[name].append(x.cpu())
        res.append(x.cpu())
    res = torch.cat(res)
    return (res, {k: torch.cat(v) for k, v in outs.items()}) if keep else res


_DS = {}


def eeg(sub, split='test', average=True):
    """EEG of one subject as a (memmapped) array. test avg (200,63,250); test reps (200,80,63,250); train avg (16540,63,250)."""
    key = (sub, split, average)
    if key not in _DS:
        cfg = json.load(open(f'{run_dir("mix", 1)}/train_config.json'))
        ds = EEGPreImageDataset([sub], cfg['eeg_data_dir'], [], [0, 250], cfg['image_feature_dir'], '', average, False,
                                None, split == 'train', False, False)
        x = ds.eeg_data_list[0]
        x = x.reshape(-1, *x.shape[2:])  # merge (object, image)
        _DS[key] = x
        _DS[('img', split)] = ds.image_features.reshape(-1, ds.image_features.shape[-1])
    return _DS[key]


def img_raw(split='test'):
    if ('img', split) not in _DS:
        eeg(1, split)
    return _DS[('img', split)]


@torch.no_grad()
def img_emb(m, split='test'):
    """Projected image targets (L2-normalised) for the 200 test or 16540 train images."""
    x = torch.tensor(img_raw(split), dtype=torch.float32)
    out = torch.cat([m.img_proj(x[s:s + 4096].to(DEV)).cpu() for s in range(0, len(x), 4096)])
    return torch.nn.functional.normalize(out, dim=1)


def top1(emb, gallery, k=1):
    """Top-k retrieval accuracy (%) of query i against gallery row i, cosine."""
    s = torch.nn.functional.normalize(emb.float(), dim=1) @ gallery.T
    hit = (s.topk(k, dim=1).indices == torch.arange(len(s))[:, None]).any(1)
    return 100 * hit.float().mean().item()


def T(x):
    return torch.tensor(np.asarray(x, dtype=np.float32))
