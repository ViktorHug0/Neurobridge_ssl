"""Real-data interruption test in a disposable, isolated smoke directory."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from .train import OUTPUT, ROOT


def main():
    root = Path(tempfile.mkdtemp(prefix='resume-', dir=OUTPUT))
    base_args = ['--arm','hybrid','--subject','1','--epochs','2','--max-batches','2']
    interrupt_code = '''
from ensemble_experiments.compact_valcon import train
original = train.atomic_save
def save(path, payload):
    original(path, payload)
    if path.name == 'last.pth':
        raise SystemExit(75)
train.atomic_save = save
train.main()
'''
    commands = [
        ([sys.executable,'-m','ensemble_experiments.compact_valcon.train',*base_args,'--output',str(root/'full')], 0),
        ([sys.executable,'-c',interrupt_code,*base_args,'--output',str(root/'resumed')], 75),
        ([sys.executable,'-m','ensemble_experiments.compact_valcon.train',*base_args,'--output',str(root/'resumed')], 0),
    ]
    for i, (command, expected) in enumerate(commands):
        with (root/f'process{i}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != expected:
            raise RuntimeError(f'Resume test process{i}: exit{result.returncode}; see {root}')
    states = [torch.load(root/p/'hybrid/seed3300/sub-01/last.pth',map_location='cpu',weights_only=False)
              for p in ['full','resumed']]
    for k in states[0]['model']:
        torch.testing.assert_close(states[0]['model'][k], states[1]['model'][k], rtol=0, atol=0)
        torch.testing.assert_close(states[0]['best_model'][k], states[1]['best_model'][k], rtol=0, atol=0)
    assert states[0]['best'] == states[1]['best']
    assert [r['epoch'] for r in states[1]['records']] == [1,2]
    report = dict(passed=True, exact_model_and_selected_weights=True, output=str(root))
    print(json.dumps(report), flush=True)
    from ensemble_experiments.mechanism_study.common import write_json
    write_json(OUTPUT/'resume_test.json', report)


if __name__ == '__main__': main()
