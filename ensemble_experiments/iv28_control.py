"""Matched IV28-only control; never edits the active Nash trainer or model."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ensemble_experiments.compact_valcon import train
from ensemble_experiments.compact_valcon.models import CompactDecoder
from ensemble_experiments.mechanism_study.common import write_json

OUTPUT = train.ROOT/'results/things_eeg/iv28_control_20260909'


class IV28Control(CompactDecoder):
    def __init__(self, arm='dual_head', channels=63):
        assert arm=='dual_head'
        # Instantiate in the original order so the surviving tensors and RNG
        # consumption exactly match the joint model before removing IV33.
        super().__init__('dual_head',channels)
        self.targets=[28]
        self.align_dims=[128]
        self.eeg_heads=nn.ModuleList([self.eeg_heads[1]])
        self.image_heads=nn.ModuleList([self.image_heads[1]])
        self.criteria=nn.ModuleList([self.criteria[1]])


def tests():
    train.seed_everything(3300); joint=CompactDecoder('dual_head').eval()
    train.seed_everything(3300); solo=IV28Control().eval()
    for a,b in zip(joint.backbone.parameters(),solo.backbone.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(joint.eeg_heads[1].parameters(),solo.eeg_heads[0].parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(joint.image_heads[1].parameters(),solo.image_heads[0].parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    eeg=torch.randn(8,63,250); images=torch.randn(8,6400)
    sf=solo(eeg,images); jf=joint(eeg,images)[1]
    for a,b in zip(sf[0],jf): torch.testing.assert_close(a,b,atol=0,rtol=0)
    changed=images.clone(); changed[:,:3200]=torch.randn_like(changed[:,:3200])*100
    for a,b in zip(sf[0],solo(eeg,changed)[0]): torch.testing.assert_close(a,b,atol=0,rtol=0)
    positive=torch.eye(8,dtype=torch.bool)
    expected=joint.criteria[1].multi_positive_pair_loss(*jf,positive)
    torch.testing.assert_close(solo.loss(sf,positive),expected,atol=0,rtol=0)
    solo.loss(sf,positive).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in solo.parameters() if p.requires_grad)
    return solo.size()


def resume_test():
    size=tests()
    root=Path(tempfile.mkdtemp(prefix='resume-',dir=OUTPUT))
    for index,(folder,interrupt,expected) in enumerate([
        ('full',False,0),('resumed',True,75),('resumed',False,0)]):
        command=[sys.executable,'-m','ensemble_experiments.iv28_control','--subject','1',
                 '--output',str(root/folder),'--epochs','2','--max-batches','2']
        if interrupt: command.append('--interrupt')
        with (root/f'process{index}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode!=expected: raise RuntimeError(f'Failed audit: {root}/process{index}.log')
    states=[torch.load(root/p/'dual_head/seed3300/sub-01/last.pth',map_location='cpu',weights_only=False)
            for p in ['full','resumed']]
    for field in ['model','best_model']:
        for key in states[0][field]:
            torch.testing.assert_close(states[0][field][key],states[1][field][key],atol=0,rtol=0)
    assert states[0]['best']==states[1]['best']
    assert [r['epoch'] for r in states[1]['records']]==[1,2]
    write_json(OUTPUT/'audit.json',dict(passed=True,bitwise_resume=True,matched_initialization=True,
                                      iv33_input_invariance=True,model_size=size,path=str(root)))
    print('AUDIT_PASSED',flush=True)


def queue():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=['ensemble_experiments/iv28_control.py','ensemble_experiments/compact_valcon/train.py',
           'ensemble_experiments/compact_valcon/models.py','train.py','module/loss.py',
           'module/dataset.py','module/eeg_encoder/model.py']
    hashes={p:hashlib.sha256((train.ROOT/p).read_bytes()).hexdigest() for p in paths}
    if (OUTPUT/'manifest.json').exists():
        assert json.loads((OUTPUT/'manifest.json').read_text())['hashes']==hashes
    write_json(OUTPUT/'manifest.json',dict(experiment='IV28 only, matched backbone896 and IV28 initialization',
               hashes=hashes,loss='full single-task IV28 loss',seed=3300,
               note='dual_head directory label is retained only for original trainer compatibility'))
    if not (OUTPUT/'audit.json').exists(): resume_test()
    assert json.loads((OUTPUT/'audit.json').read_text())['passed']
    for subject in range(1,11):
        for p,h in hashes.items(): assert hashlib.sha256((train.ROOT/p).read_bytes()).hexdigest()==h
        if (OUTPUT/f'runs/dual_head/seed3300/sub-{subject:02d}/complete.json').exists(): continue
        with (OUTPUT/f'sub-{subject:02d}.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','ensemble_experiments.iv28_control',
                            '--subject',str(subject)],stdout=log,stderr=subprocess.STDOUT,check=True)
        results=[json.loads(f.read_text()) for f in sorted((OUTPUT/'runs').glob('dual_head/seed3300/sub-*/complete.json'))]
        write_json(OUTPUT/'summary.json',dict(folds=len(results),mean_top1=float(np.mean([r['top1'] for r in results])),results=results))
        print('FOLD_COMPLETE',subject,flush=True)
    print('ALL_FOLDS_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--subject',type=int,choices=range(1,11))
    p.add_argument('--output',type=Path,default=OUTPUT/'runs')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--interrupt',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.subject is None: queue(); return
    if args.max_batches and args.output==OUTPUT/'runs': raise ValueError('Isolate smoke outputs')
    if args.output==train.OUTPUT/'runs': raise ValueError('Never overwrite historical runs')
    train.CompactDecoder=IV28Control
    if args.interrupt:
        original=train.atomic_save
        def save(path,payload):
            original(path,payload)
            if path.name=='last.pth': raise SystemExit(75)
        train.atomic_save=save
    sys.argv=[sys.argv[0],'--arm','dual_head','--subject',str(args.subject),
              '--output',str(args.output),'--epochs',str(args.epochs),'--max-batches',str(args.max_batches)]
    train.main()


if __name__=='__main__': main()
