"""Structural, full-batch real-data and exact interrupted-resume checks."""
import copy
import gc
import json
from pathlib import Path
import torch

from .models import FullPair, CONFIGS, seeded
from .train import Fold, run, deterministic, OUTPUT, sources
from ensemble_experiments.mechanism_study.common import write_json
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import TSConv_parameterizable


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b), 'Resume tensor mismatch'
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:
        assert a==b,(a,b)


def structural():
    x=torch.randn(4,63,250)
    image=torch.randn(4,6400)
    ids=torch.tensor([1,2,3,10])
    checks=[]
    for c in CONFIGS:
        m=FullPair(c).eval()
        y=m(x,image,ids)
        assert all(e.shape==i.shape==(4,128) for e,i in y)
        total=sum(p.numel() for p in m.parameters())
        assert total==m.size()['parameters']
        opt=torch.optim.AdamW(m.parameters())
        assert len(opt.param_groups[0]['params'])==len({id(p) for p in m.parameters()})
        restored=FullPair(c).eval()
        restored.load_state_dict(m.state_dict())
        assert restored.size()['aliases']==m.size()['aliases']
        for (a,b),(aa,bb) in zip(y,restored(x,image,ids)):
            torch.testing.assert_close(a,aa,rtol=0,atol=0)
            torch.testing.assert_close(b,bb,rtol=0,atol=0)
        checks.append(c+': shapes, unique parameter accounting, alias roundtrip')
    for a,b in [('F1','F2'),('F3','F4'),('F4','F5')]:
        ma,mb=FullPair(a).eval(),FullPair(b).eval()
        for (ea,ia),(eb,ib) in zip(ma(x,image,ids),mb(x,image,ids)):
            torch.testing.assert_close(ea,eb,rtol=0,atol=0)
            torch.testing.assert_close(ia,ib,rtol=0,atol=0)
        checks.append(a+'/'+b+': identical initial eval outputs')
    for c,expected in [('F3',2),('F4',1),('F5',1),('F6',1)]:
        m=FullPair(c).eval();calls=[]
        stems={id(m.stem_a):m.stem_a}
        if hasattr(m,'stem_b'):stems[id(m.stem_b)]=m.stem_b
        handles=[s.register_forward_hook(lambda *args:calls.append(1)) for s in stems.values()]
        m(x,image,ids)
        for h in handles:h.remove()
        assert len(calls)==expected,(c,len(calls))
        checks.append(c+': verified stem invocation count '+str(expected))
    m=FullPair('F1').eval()
    native_ts=seeded(3300,lambda:TSConv_parameterizable(feature_dim=128,temporal_kernel=30)).eval()
    torch.testing.assert_close(m.eeg_heads[0](native_ts(x)),m.encode(x,ids)[0],rtol=0,atol=0)
    native_atm=seeded(3300,lambda:ATMS(feature_dim=128,temporal_kernel=30)).eval()
    native_atm.enc_eeg[0].tsconv[0]=copy.deepcopy(m.stem_b.conv)
    for subject_ids in (ids,torch.ones(4,dtype=torch.long)):
        torch.testing.assert_close(m.eeg_heads[1](native_atm(x,subject_ids)),
                                   m.encode(x,subject_ids)[1],rtol=1e-5,atol=1e-6)
    checks.append('F1 native TS equivalence; ATM flatten-layout equivalence for known/unknown IDs')
    return checks


def main():
    deterministic()
    out=OUTPUT/'preflight'
    out.mkdir(parents=True,exist_ok=True)
    print('STRUCTURAL PREFLIGHT',flush=True)
    checks=structural()
    print('\n'.join(checks),flush=True)
    write_json(out/'structural.json',dict(checks=checks,sources=sources()))
    fold=Fold(1)
    for c in CONFIGS:
        print('REAL DATA FULL BATCH',c,flush=True)
        run(c,fold,out/c,epochs=1,max_steps=2,val_limit=2,smoke=True)
        checks.append(c+': real full-batch train, gradients, validation and checkpoint')
        gc.collect();torch.cuda.empty_cache()
    for c in ('F1','F4','F5','F6'):
        print('EXACT RESUME',c,flush=True)
        base,resumed=out/(c+'_continuous'),out/(c+'_resumed')
        run(c,fold,base,epochs=2,max_steps=2,val_limit=2,smoke=True)
        run(c,fold,resumed,epochs=2,max_steps=2,val_limit=2,smoke=True,stop_after_epoch=1)
        run(c,fold,resumed,epochs=2,max_steps=2,val_limit=2,smoke=True)
        a=torch.load(base/'last.pth',map_location='cpu',weights_only=False)
        b=torch.load(resumed/'last.pth',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','best','bad','selected','best_model'):
            equal(a[key],b[key])
        equal(a['rng']['torch'],b['rng']['torch']);equal(a['rng']['cuda'],b['rng']['cuda'])
        checks.append(c+': bitwise exact model/optimizer/selection/RNG resume')
        del a,b
        gc.collect();torch.cuda.empty_cache()
    write_json(out/'passed.json',dict(passed=True,checks=checks,sources=sources(),gpu=torch.cuda.get_device_name(0)))
    print('PREFLIGHT PASSED',len(checks),'checks',flush=True)


if __name__=='__main__':
    main()
