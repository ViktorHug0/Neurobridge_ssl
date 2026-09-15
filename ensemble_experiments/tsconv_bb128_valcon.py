"""Full-width TSConv with 128D backbone/alignment, matching the IV33 solo recipe.

Reuses the verified compact ValCon trainer in-process, substituting only the
model constructor. Existing experiment source and results are not modified.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from ensemble_experiments.compact_valcon.models import CompactDecoder
from ensemble_experiments.compact_valcon import train as trainer
from ensemble_experiments.mechanism_study.common import write_json, load_member, row_z
from module.eeg_encoder.model import TSConv_parameterizable
from module.projector import ProjectorLinear
from module.loss import ContrastiveLoss

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/things_eeg/tsconv_bb128_valcon_20260911'


class BB128Decoder(CompactDecoder):
    def __init__(self, arm='single', channels=63):
        nn.Module.__init__(self)
        if arm != 'single':
            raise ValueError(arm)
        self.arm, self.targets, self.align_dims, self.width = arm, [33], [128], 128
        self.backbone = TSConv_parameterizable(
            feature_dim=128, channels_num=channels, temporal_kernel=30,
            pool_kernel=51, pool_stride=5)
        self.eeg_heads = nn.ModuleList([ProjectorLinear(128, 128)])
        self.image_heads = nn.ModuleList([ProjectorLinear(3200, 128)])
        self.criteria = nn.ModuleList([
            ContrastiveLoss(0.07, 1., 1., False, True, False, False, True)])


def sources():
    paths = [Path(__file__), ROOT/'train.py']
    paths += list((ROOT/'module').rglob('*.py'))
    paths += list((ROOT/'ensemble_experiments/compact_valcon').glob('*.py'))
    paths += [ROOT/'ensemble_experiments/mechanism_study/common.py',
              ROOT/'ensemble_experiments/mechanism_study/train_pair.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def verify_manifest():
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    if manifest['sources'] != sources():
        raise RuntimeError('Source changed since preparation; refusing mixed experiment')
    return manifest


def summarize():
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for subject in range(1,6):
            folder = OUTPUT/f'runs/single/seed3300/sub-{subject:02d}'
            if not (folder/'complete.json').exists():
                continue
            row = json.loads((folder/'complete.json').read_text())
            atm, _, keys = load_member('atm_iv_valcon', subject)
            with np.load(folder/'test_scores.npz') as dump:
                np.testing.assert_array_equal(keys, np.stack([dump['object'], dump['image_idx']], axis=1))
                ts = dump['scores'][0]
            fused = .5 * (row_z(ts) + row_z(atm))
            row['atm_reference_top1'] = float(100*(atm.argmax(1)==np.arange(200)).mean())
            row['ensemble_with_atm_top1'] = float(100*(fused.argmax(1)==np.arange(200)).mean())
            rows.append(row)
        write_json(OUTPUT/'summary.json', dict(completed=len(rows), expected=5, subjects=rows,
            means={k:float(np.mean([r[k] for r in rows])) for k in
                   ('top1','atm_reference_top1','ensemble_with_atm_top1','training_seconds')}
                   if rows else {}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', type=int, choices=[0,1])
    p.add_argument('--train', type=int, choices=range(1,6))
    args = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if (OUTPUT/'manifest.json').exists():
            verify_manifest(); print('ALREADY_PREPARED', flush=True); return
        from ensemble_experiments.compact_valcon import benchmark
        benchmark.CompactDecoder = BB128Decoder
        result = benchmark.measure('single', 'fp32', steps=3)
        assert result['backbone_width'] == 128 and result['alignment_dims'] == [128]
        write_json(OUTPUT/'benchmark.json', result)
        write_json(OUTPUT/'manifest.json', dict(sources=sources(), created=time.time(),
            model='TSConv_parameterizable', temporal_filters=40, spatial_filters=40,
            projection_filters=40, temporal_kernel=30, backbone_dim=128, alignment_dim=128,
            image_target='InternViT-6B_layer33_mean_8bit', seed=3300, precision='fp32',
            epochs=100, early_stop_patience=20, mixup='pairwise', mixup_alpha=.5,
            val_concept_seed=20260822, val_concept_ratio=.1,
            tasks=[dict(subject=s,worker=(s-1)%2) for s in range(1,6)], benchmark=result))
        print('PREPARED', json.dumps(result), flush=True); return
    manifest = verify_manifest()
    if args.train is not None:
        trainer.CompactDecoder = BB128Decoder
        sys.argv = [sys.argv[0], '--arm','single','--subject',str(args.train),
                    '--seed','3300','--epochs','100','--patience','20','--workers','2',
                    '--precision','fp32','--output',str(OUTPUT/'runs')]
        trainer.main()
        summarize()
        return
    if args.worker is None:
        p.error('Specify --prepare, --worker or --train')
    with (OUTPUT/f'worker{args.worker}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = [t for t in manifest['tasks'] if t['worker']==args.worker]
        for task in tasks:
            verify_manifest()
            subject = task['subject']
            write_json(OUTPUT/f'status_worker{args.worker}.json',
                       dict(state='running', subject=subject, host=socket.gethostname(), updated=time.time()))
            with (OUTPUT/f'sub-{subject:02d}.log').open('a', buffering=1) as log:
                result = subprocess.run([sys.executable,'-u','-m',
                    'ensemble_experiments.tsconv_bb128_valcon','--train',str(subject)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(OUTPUT/f'status_worker{args.worker}.json',
                    dict(state='failed',subject=subject,exit_code=result.returncode))
                raise RuntimeError(f'Failed subject {subject}; see per-subject log')
        write_json(OUTPUT/f'status_worker{args.worker}.json', dict(state='complete',tasks=tasks))
        print('WORKER_COMPLETE', args.worker, flush=True)


if __name__ == '__main__':
    main()
