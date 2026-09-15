"""Preflight, exact-resume audit, and persistent sequential LOSO queue."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train import seed_everything, _GroupedSubset, build_image_positive_mask, cross_subject_stimulus_mix
from module.sampler import GroupedImageBatchSampler
from ensemble_experiments.compact_valcon.models import CompactDecoder
from ensemble_experiments.compact_valcon.train import ROOT, dataset, split_indices
from ensemble_experiments.mechanism_study.common import write_json
from ensemble_experiments.nash_direction import combine, backward, actual_update

OUTPUT = ROOT/'results/things_eeg/nash_direction_20260909'


def preflight():
    a = torch.tensor([3.,0.],device='cuda')
    b = torch.tensor([-1.,1.],device='cuda')
    d, _ = combine(a,b)
    torch.testing.assert_close(d.norm(),((a+b)*.5).norm())
    assert torch.dot(a,d)>0 and torch.dot(b,d)>0
    scaled,_ = combine(a*7,b*.3)
    torch.testing.assert_close(d/d.norm(),scaled/scaled.norm())
    for other in [-a,torch.zeros_like(a)]:
        value,metrics = combine(a,other)
        assert metrics['fallback']==1 and torch.isfinite(value).all()

    data = dataset(list(range(2,11)),True)
    tr,_,_ = split_indices(data)
    subset = _GroupedSubset(data,tr)
    sampler = GroupedImageBatchSampler(subset,1024,samples_per_image=9,seed=3300)
    batch = next(iter(DataLoader(subset,batch_sampler=sampler,num_workers=0)))
    eeg,images,subjects,objects,ids = [batch[i].cuda() for i in [0,1,3,4,5]]
    positive = build_image_positive_mask(objects,ids)
    seed_everything(3300)
    model = CompactDecoder('dual_head').cuda()
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    # Two backward passes must recover ordinary mean gradients, including heads.
    seed_everything(11)
    model.loss(model(eeg,images),positive).backward()
    reference = {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    for rule in ['mean','nash']:
        model.load_state_dict(initial); model.zero_grad(set_to_none=True); seed_everything(11)
        _,_,_ = backward(model,model(eeg,images),positive,rule=rule)
        for name,p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None
                continue
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            if rule=='mean' or not name.startswith('backbone.'):
                torch.testing.assert_close(p.grad,reference[name],atol=2e-5,rtol=2e-4)
        if rule=='nash':
            old = torch.cat([reference[k].flatten() for k in reference if k.startswith('backbone.')])
            new = torch.cat([p.grad.flatten() for p in model.backbone.parameters()])
            torch.testing.assert_close(old.norm(),new.norm(),rtol=1e-5,atol=1e-6)
    del reference, initial, model
    torch.cuda.empty_cache()
    results=[]
    for rule in ['mean','nash']:
        seed_everything(3300)
        model=CompactDecoder('dual_head').cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        times=[]; diagnostics=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(12):
            torch.cuda.synchronize(); tick=time.monotonic()
            x=cross_subject_stimulus_mix(eeg,objects,ids,subjects,alpha=.5,mixup_type='pairwise')
            optimizer.zero_grad(set_to_none=True)
            features=model(x,images)
            if rule=='mean':
                loss=model.loss(features,positive); loss.backward()
            else:
                loss,metrics,grads=backward(model,features,positive)
                before=torch.cat([p.detach().flatten() for p in model.backbone.parameters()])
            optimizer.step()
            if rule=='nash':
                metrics.update(actual_update(model,before,grads))
                diagnostics.append({k:float(v) for k,v in metrics.items()})
            torch.cuda.synchronize()
            if step>=3: times.append(time.monotonic()-tick)
        results.append(dict(rule=rule,seconds_per_step=float(np.mean(times)),
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            diagnostics=diagnostics,**model.size()))
        del model,optimizer
        torch.cuda.empty_cache()
    write_json(OUTPUT/'preflight.json',dict(passed=True,gpu=torch.cuda.get_device_name(),results=results))
    print('PREFLIGHT',json.dumps(results),flush=True)


def verify_resume():
    root=Path(tempfile.mkdtemp(prefix='resume-',dir=OUTPUT))
    args=['--arm','dual_head','--gradient-rule','nash','--subject','1','--epochs','2','--max-batches','2']
    code='''
from ensemble_experiments.compact_valcon import train
original=train.atomic_save
def save(path,payload):
    original(path,payload)
    if path.name=='last.pth': raise SystemExit(75)
train.atomic_save=save
train.main()
'''
    for index,(prefix,directory,expected) in enumerate([
        (['-m','ensemble_experiments.compact_valcon.train'],'full',0),
        (['-c',code],'resumed',75),
        (['-m','ensemble_experiments.compact_valcon.train'],'resumed',0)]):
        with (root/f'process{index}.log').open('w') as log:
            result=subprocess.run([sys.executable,*prefix,*args,'--output',str(root/directory)],
                                  cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode!=expected: raise RuntimeError(f'Resume audit failed: {root}, process {index}')
    states=[torch.load(root/p/'dual_head/seed3300/sub-01/last.pth',map_location='cpu',weights_only=False)
            for p in ['full','resumed']]
    for field in ['model','best_model']:
        for key in states[0][field]:
            torch.testing.assert_close(states[0][field][key],states[1][field][key],rtol=0,atol=0)
    assert states[0]['best']==states[1]['best']
    assert [r['epoch'] for r in states[1]['records']]==[1,2]
    write_json(OUTPUT/'resume_audit.json',dict(passed=True,bitwise_weights=True,path=str(root)))
    print('RESUME_AUDIT_PASSED',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['preflight','resume','queue'],default='queue')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.mode=='preflight': preflight(); return
    if args.mode=='resume': verify_resume(); return
    lock=(OUTPUT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=['ensemble_experiments/nash_direction.py','ensemble_experiments/nash_experiment.py',
             'ensemble_experiments/compact_valcon/train.py','ensemble_experiments/compact_valcon/models.py',
             'train.py','module/loss.py','module/dataset.py','module/eeg_encoder/model.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    if (OUTPUT/'source_hashes.json').exists():
        assert json.loads((OUTPUT/'source_hashes.json').read_text())==hashes
    write_json(OUTPUT/'source_hashes.json',hashes)
    for mode,file in [('preflight','preflight.json'),('resume','resume_audit.json')]:
        if not (OUTPUT/file).exists():
            subprocess.run([sys.executable,'-m','ensemble_experiments.nash_experiment','--mode',mode],check=True)
        assert json.loads((OUTPUT/file).read_text())['passed']
    for subject in range(1,11):
        for p,h in hashes.items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
        path=OUTPUT/f'runs/dual_head/seed3300/sub-{subject:02d}/complete.json'
        if path.exists(): continue
        with (OUTPUT/f'sub-{subject:02d}.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','ensemble_experiments.compact_valcon.train',
                            '--arm','dual_head','--gradient-rule','nash','--subject',str(subject),
                            '--output',str(OUTPUT/'runs')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        completed=[json.loads(f.read_text()) for f in sorted((OUTPUT/'runs').glob('dual_head/seed3300/sub-*/complete.json'))]
        write_json(OUTPUT/'summary.json',dict(folds=len(completed),mean_top1=float(np.mean([r['top1'] for r in completed])),
                                            results=completed))
        print('FOLD_COMPLETE',subject,flush=True)
    print('ALL_FOLDS_COMPLETE',flush=True)


if __name__=='__main__': main()
