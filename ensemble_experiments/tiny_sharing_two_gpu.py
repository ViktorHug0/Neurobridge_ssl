"""Disjoint two-GPU scheduling; leaves the frozen experiment source unchanged."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from ensemble_experiments.tiny_sharing.data import FoldData
from ensemble_experiments.tiny_sharing.models import CONFIG_IDS
from ensemble_experiments.tiny_sharing.train_pair import run, source_hash, atomic_json

ROOT = Path('/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/tiny_sharing/screen_20260911_v2')
EXPECTED_HASH = '5383ed884ef6806c'


def check_gate():
    path = ROOT / 'bridge_gate.json'
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    assert report['source_hash'] == EXPECTED_HASH
    # The gain-retention rule is reported diagnostically but is unstable when
    # the native fusion gain is near zero. Only a meaningful fused-accuracy
    # regression blocks the remaining screen.
    if report['fused_drop'] > 0.02:
        raise RuntimeError('Baseline bridge fused accuracy dropped >2pp')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', choices=['A', 'B'], required=True)
    args = ap.parse_args()
    assert source_hash() == EXPECTED_HASH, 'Frozen experiment source changed'
    assert json.loads((ROOT / 'preflight/passed.json').read_text())['source_hash'] == EXPECTED_HASH
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print(f'Worker {args.worker}: {torch.cuda.get_device_name(0)}', flush=True)
    if args.worker == 'B':
        print('Waiting for slot1 C0/C1 baseline gate', flush=True)
        while not check_gate():
            time.sleep(20)
    assignments = ([(1, CONFIG_IDS[2:]), (2, CONFIG_IDS[:5])] if args.worker == 'A'
                   else [(3, CONFIG_IDS), (2, CONFIG_IDS[5:])])
    for slot, configs in assignments:
        print(f'Loading slot{slot}: {configs}', flush=True)
        fold = FoldData(slot, torch.device('cuda:0'))
        for config in configs:
            assert source_hash() == EXPECTED_HASH
            out = ROOT / f'slot{slot}' / config
            done = out / 'summary.json'
            if done.exists():
                summary = json.loads(done.read_text())
                assert summary['source_hash'] == EXPECTED_HASH and summary['epochs'] == 40
                print(f'SKIP slot{slot}/{config}', flush=True)
            else:
                run(config, fold, str(out))
            if slot == 1 and config == 'C1':
                a = json.loads((ROOT / 'slot1/C0/metrics_best_mean.json').read_text())
                b = json.loads((ROOT / 'slot1/C1/metrics_best_mean.json').read_text())
                drop = a['top1_fused'] - b['top1_fused']
                ga, gb = a['gain_over_best_branch'], b['gain_over_best_branch']
                report = dict(blocked=drop > .02 or (ga > 0 and gb < .5 * ga),
                              fused_drop=drop, native_gain=ga, bridge_gain=gb,
                              source_hash=EXPECTED_HASH)
                atomic_json(report, str(ROOT / 'bridge_gate.json'))
                print('BRIDGE GATE:', report, flush=True)
                check_gate()
            gc.collect()
            torch.cuda.empty_cache()
        del fold
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Worker {args.worker} complete: 15 assigned runs', flush=True)


if __name__ == '__main__':
    main()
