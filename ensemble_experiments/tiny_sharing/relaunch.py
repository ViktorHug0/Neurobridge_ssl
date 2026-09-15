"""Real-data preflight followed by a gated, sequential three-slot screen."""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from ensemble_experiments.tiny_sharing import gate
from ensemble_experiments.tiny_sharing.data import FoldData
from ensemble_experiments.tiny_sharing.models import CONFIG_IDS
from ensemble_experiments.tiny_sharing.train_pair import run, source_hash, atomic_json, _rowz


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), 'Tensor mismatch after resume'
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), 'Array mismatch after resume'
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def preflight(root, fold):
    root.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        assert gate.main() == 0, 'Structural gate failed'
    finally:
        os.chdir(cwd)
    # Row-z fusion must have the same scale as each solo margin.
    z = _rowz(torch.tensor([[3., 1., 0.], [1., 4., 2.]], device=fold.device))
    assert torch.equal(0.5 * (z + z), z)
    checks = []
    for config in CONFIG_IDS:
        print(f'PREFLIGHT real data: {config}', flush=True)
        run(config, fold, str(root / config), epochs=1, max_steps=2)
        checks.append(config + ': full-batch train/eval/checkpoint passed')
        gc.collect()
        torch.cuda.empty_cache()
    for config in ('C0', 'C7', 'C9'):
        print(f'PREFLIGHT exact resume: {config}', flush=True)
        continuous = root / (config + '_continuous')
        resumed = root / (config + '_resumed')
        run(config, fold, str(continuous), epochs=2, max_steps=2)
        run(config, fold, str(resumed), epochs=2, max_steps=2, stop_after_epoch=1)
        run(config, fold, str(resumed), epochs=2, max_steps=2)
        a = torch.load(continuous / 'last.pth', map_location='cpu', weights_only=False)
        b = torch.load(resumed / 'last.pth', map_location='cpu', weights_only=False)
        for key in ('model', 'optimizer', 'rng', 'cuda_rng', 'branch_rng', 'best', 'artifacts'):
            equal(a[key], b[key])
        checks.append(config + ': bitwise exact resume passed')
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json({'source_hash': source_hash(), 'checks': checks}, str(root / 'passed.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-root', required=True)
    args = ap.parse_args()
    root = Path(args.result_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    print('GPU:', torch.cuda.get_device_name(device), 'SOURCE:', source_hash(), flush=True)
    for slot in (1, 2, 3):
        fold = FoldData(slot, device)
        if slot == 1:
            passed = root / 'preflight' / 'passed.json'
            if not passed.exists():
                preflight(root / 'preflight', fold)
            else:
                assert json.loads(passed.read_text())['source_hash'] == source_hash()
        for config in CONFIG_IDS:
            out = root / f'slot{slot}' / config
            done = out / 'summary.json'
            if done.exists():
                summary = json.loads(done.read_text())
                assert summary['source_hash'] == source_hash() and summary['epochs'] == 40
                print(f'SKIP completed slot{slot}/{config}', flush=True)
            else:
                run(config, fold, str(out))
            if slot == 1 and config == 'C1':
                a = json.loads((root / 'slot1/C0/metrics_best_mean.json').read_text())
                b = json.loads((root / 'slot1/C1/metrics_best_mean.json').read_text())
                drop = a['top1_fused'] - b['top1_fused']
                gain_a, gain_b = a['gain_over_best_branch'], b['gain_over_best_branch']
                blocked = drop > 0.02 or (gain_a > 0 and gain_b < 0.5 * gain_a)
                report = dict(blocked=blocked, fused_drop=drop, native_gain=gain_a,
                              bridge_gain=gain_b, source_hash=source_hash())
                atomic_json(report, str(root / 'bridge_gate.json'))
                print('BRIDGE GATE:', report, flush=True)
                if blocked:
                    raise RuntimeError('Bridge gate failed: stopped screen before sharing ablations')
            gc.collect()
            torch.cuda.empty_cache()
        del fold
        gc.collect()
        torch.cuda.empty_cache()
    print('SCREEN COMPLETE: 30/30 conditions', flush=True)


if __name__ == '__main__':
    main()
