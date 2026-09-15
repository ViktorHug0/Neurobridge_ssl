"""Two disjoint persistent queues; every model/subject has its own process/log."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import numpy as np
import torch
from .models import CONFIGS,DESCRIPTIONS
from .train import OUTPUT,REPO,sources
from ensemble_experiments.mechanism_study.common import write_json


def baseline_complete():
    root=REPO/'results/things_eeg/tsconv_bb128_valcon_20260911'
    assert all((root/f'runs/single/seed3300/sub-{s:02d}/complete.json').exists() for s in range(1,6))
    for worker in (0,1):
        assert json.loads((root/f'status_worker{worker}.json').read_text())['state']=='complete'


def summary():
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[json.loads(p.read_text()) for p in sorted((OUTPUT/'runs').glob('F*/seed3300/sub-*/complete.json'))]
        means={}
        for config in CONFIGS:
            subset=[r for r in rows if r['config']==config]
            if subset:
                means[config]=dict(n=len(subset),**{k:float(np.mean([r[k] for r in subset]))
                    for k in ('top1','top1_calibrated','parameters','eeg_parameters','train_seconds')})
        write_json(OUTPUT/'summary.json',dict(completed=len(rows),expected=30,means=means,folds=rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--worker',type=int,choices=[0,1])
    a=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    baseline_complete()
    passed=json.loads((OUTPUT/'preflight/passed.json').read_text())
    assert passed['passed'] and passed['sources']==sources(),'Preflight missing or source changed'
    path=OUTPUT/'manifest.json'
    if a.prepare:
        if path.exists():
            assert json.loads(path.read_text())['sources']==sources()
            return
        tasks=[dict(config=c,subject=s,worker=(j+s-1)%2) for s in range(1,6) for j,c in enumerate(CONFIGS)]
        write_json(path,dict(sources=sources(),configs=DESCRIPTIONS,tasks=tasks,subjects=list(range(1,6)),
                            seed=3300,backbone_dim=128,alignment_dim=128,precision='fp32',
                            max_epochs=100,patience=20,mixup='pairwise',baseline_top1=43.1))
        print('PREPARED 30 TASKS',flush=True)
        return
    assert a.worker is not None
    plan=json.loads(path.read_text())
    assert plan['sources']==sources()
    # Hold one device lock for the whole queue, including all child processes.
    device=str(torch.cuda.get_device_properties(0).uuid)
    with (OUTPUT/f'device-{device}.lock').open('a') as gpu_lock, (OUTPUT/f'worker{a.worker}.lock').open('a') as lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=[t for t in plan['tasks'] if t['worker']==a.worker]
        for n,task in enumerate(tasks):
            assert sources()==plan['sources'],'Sources changed; refusing mixed experiment'
            c,s=task['config'],task['subject']
            status=dict(state='running',task=task,completed_in_queue=n,total=len(tasks),
                        host=socket.gethostname(),pid=os.getpid(),updated=time.time())
            write_json(OUTPUT/f'status_worker{a.worker}.json',status)
            with (OUTPUT/f'{c}-sub-{s:02d}.log').open('a',buffering=1) as log:
                result=subprocess.run([sys.executable,'-u','-m','ensemble_experiments.full_sharing_v2.train',
                                       '--config',c,'--subject',str(s)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(OUTPUT/f'status_worker{a.worker}.json',dict(**{k:v for k,v in status.items() if k!='state'},
                           state='failed',exit_code=result.returncode))
                raise RuntimeError(f'{c} subject{s} failed; see its log')
            summary()
        write_json(OUTPUT/f'status_worker{a.worker}.json',dict(state='complete',completed=15,tasks=tasks))
        print('WORKER COMPLETE',a.worker,flush=True)


if __name__=='__main__':
    main()
