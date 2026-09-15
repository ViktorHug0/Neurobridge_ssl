"""Real-data smoke and exact interruption/resume checks for both new arms."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from .train import OUTPUT, ROOT
from .models import ARMS
from ensemble_experiments.mechanism_study.common import write_json


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='resume-',dir=OUTPUT));reports=[]
    interrupt='''
from ensemble_experiments.compact_followup import train
original=train.atomic_save
def save(path,payload):
    original(path,payload)
    if path.name=='last.pth':raise SystemExit(75)
train.atomic_save=save
train.main()
'''
    for arm in ARMS:
        base=['--arm',arm,'--subject','1','--epochs','2','--max-batches','2']
        commands=[([sys.executable,'-m','ensemble_experiments.compact_followup.train',*base,'--output',str(root/'full')],0),
            ([sys.executable,'-c',interrupt,*base,'--output',str(root/'resumed')],75),
            ([sys.executable,'-m','ensemble_experiments.compact_followup.train',*base,'--output',str(root/'resumed')],0)]
        for i,(command,expected) in enumerate(commands):
            log_path=root/f'{arm}-{i}.log'
            with log_path.open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode!=expected:raise RuntimeError(f'Unexpected exit{result.returncode}: {log_path}')
        a,b=[torch.load(root/p/arm/'seed3300/sub-01/last.pth',map_location='cpu',weights_only=False)
             for p in ['full','resumed']]
        for group in ['model','best_model']:
            for k in a[group]:torch.testing.assert_close(a[group][k],b[group][k],rtol=0,atol=0)
        assert a['best']==b['best']
        assert [r['epoch'] for r in b['records']]==[1,2]
        reports.append(dict(arm=arm,passed=True,exact_weights=True))
        print('RESUME_OK',arm,flush=True)
    write_json(OUTPUT/'resume_test.json',dict(passed=True,arms=reports,output=str(root)))
    print('ALL_SMOKE_AND_RESUME_TESTS_PASSED',root,flush=True)


if __name__=='__main__':main()
