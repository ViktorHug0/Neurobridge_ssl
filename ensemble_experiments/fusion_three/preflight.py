import copy
import gc
import json
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from .models import Decoder, ARMS
from . import train
from .recipe import Fold, pairwise_mix, positive_mask
from ensemble_experiments.full_sharing_v2.models import FullPair, seeded
from module.eeg_encoder.model import TSConv_parameterizable


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),'Tensor mismatch'
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


def structural():
    x=torch.randn(3,63,250);images=torch.randn(3,6400);ids=torch.tensor([1,2,10])
    for arm in ARMS:
        m=Decoder(arm).eval();calls=[]
        stem=m.pair.stem_a if arm=='G2' else m.stem
        hook=stem.register_forward_hook(lambda *args:calls.append(1))
        out=m(x,images,ids);hook.remove()
        assert len(calls)==1
        assert all(e.shape==i.shape==(3,128) for e,i in out)
        restored=Decoder(arm).eval();restored.load_state_dict(m.state_dict())
        equal(out,restored(x,images,ids))
        assert m.size()['parameters']==sum(p.numel() for p in m.parameters())
        assert m.size()['parameters']==m.size()['eeg_parameters']+m.size()['image_parameters']+2
        print('STRUCTURAL',arm,m.size(),flush=True)
    m=Decoder('G2').eval();ref=FullPair('F4').eval()
    equal(m(x,images,ids),ref(x,images,ids))
    assert m.pair.readout_a[0].shared is m.pair.readout_b[0].shared
    assert m.pair.readout_a[1].f[1].shared is m.pair.readout_b[1].f[1].shared
    assert m.pair.readout_a[0].up is not m.pair.readout_b[0].up
    before=m.encode(x,ids)
    with torch.no_grad():m.pair.readout_a[0].up.weight.fill_(.05)
    after=m.encode(x,ids)
    assert not torch.equal(before[0],after[0]);equal(before[1],after[1])
    m=Decoder('G3').eval()
    with torch.no_grad():m.side.gain.zero_()
    ts=seeded(3300,lambda:TSConv_parameterizable(feature_dim=128,temporal_kernel=30)).eval()
    equal(m.encode(x,ids)[0],m.eeg_heads[0](ts(x)))


def panels():
    n=450;keys=np.stack([np.arange(n)//10,np.arange(n)%10],1)
    order=np.random.default_rng(3300).permutation(n);panels=[]
    for start in range(0,n,200):
        q=order[start:start+200];c=q if len(q)==200 else np.concatenate([q,order[:200-len(q)]])
        panels.append((q,c))
    fold=SimpleNamespace(keys=keys,panels=panels,sources=[1,2])
    x=np.random.default_rng(1).normal(size=(n,128));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    ii=np.concatenate([x,x]);ee=np.concatenate([x,-x]);kk=np.concatenate([keys,keys]);ss=np.repeat([1,2],n)
    shuffle=np.random.default_rng(4).permutation(2*n)
    r=train.panel_metrics([ee[shuffle]]*2,[ii[shuffle]]*2,kk[shuffle],ss[shuffle],fold)
    assert r['top1']==50 and r['subject_top1']=={'1':100.,'2':0.} and r['queries']==900
    print('PANEL IDENTITIES / NO SUBJECT AVERAGING PASSED',flush=True)


def mixup_check():
    from train import cross_subject_stimulus_mix
    for device in ('cpu','cuda'):
        x=torch.randn(27,63,250,device=device);o=torch.arange(3,device=device).repeat_interleave(9)
        i=torch.zeros(27,dtype=torch.long,device=device);s=torch.arange(1,10,device=device).repeat(3)
        for ids in (s,torch.ones_like(s)):
            torch.manual_seed(91);random.seed(91)
            a=cross_subject_stimulus_mix(x,o,i,ids,alpha=.5,mixup_type='pairwise')
            torch.manual_seed(91);random.seed(91)
            b=pairwise_mix(x,o,i,ids)
            equal(a,b)
    print('HISTORICAL PAIRWISE MIXUP EXACT MATCH',flush=True)


def benchmark():
    fold=Fold(1);b=next(iter(fold.loader))
    x,img,s,o,i=[b[j].cuda() for j in (0,1,3,4,5)];pos=positive_mask(o,i)
    results={}
    for arm in ARMS:
        m=Decoder(arm).cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
        times=[];torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
            losses=m.losses(m(x,img,s),pos)
            if step==0:train.gradients(m,losses)
            torch.stack(losses).mean().backward();opt.step();torch.cuda.synchronize()
            times.append(time.monotonic()-t)
        peak=torch.cuda.max_memory_allocated()/2**30
        m.eval();infer=[]
        with torch.inference_mode():
            for k in range(8):
                torch.cuda.synchronize();t=time.monotonic();m.encode(x[:200],s[:200]);torch.cuda.synchronize()
                if k>=3:infer.append(time.monotonic()-t)
        results[arm]=dict(**m.size(),full_batch_step_seconds=float(np.median(times[1:])),
                          peak_gpu_gib=peak,eeg_inference_200_ms=1000*float(np.median(infer)))
        print('BENCHMARK',arm,results[arm],flush=True)
        del m,opt,losses;gc.collect();torch.cuda.empty_cache()
    return results


def main():
    train.deterministic();out=train.OUTPUT/'preflight';out.mkdir(parents=True,exist_ok=True)
    structural();panels();mixup_check();gc.collect();torch.cuda.empty_cache()
    bench=benchmark();gc.collect();torch.cuda.empty_cache()
    sources=train.source_hashes()
    for arm in ARMS:
        continuous=out/f'{arm}_continuous';resumed=out/f'{arm}_resumed'
        print('REAL TRAIN / FULL VALIDATION / RESUME',arm,flush=True)
        train.run(arm,1,sources,continuous,epochs=2,max_steps=2,smoke=True)
        gc.collect();torch.cuda.empty_cache()
        train.run(arm,1,sources,resumed,epochs=2,max_steps=2,stop_after=1,smoke=True)
        gc.collect();torch.cuda.empty_cache()
        train.run(arm,1,sources,resumed,epochs=2,max_steps=2,smoke=True)
        a=torch.load(continuous/'last.pth',map_location='cpu',weights_only=False)
        b=torch.load(resumed/'last.pth',map_location='cpu',weights_only=False)
        for k in ('model','optimizer','rng','best','bad','selected','best_model'):equal(a[k],b[k])
        for x,y in zip(a['history'],b['history']):
            for k in ('train_loss','validation','selected_epoch','patience_count'):equal(x[k],y[k])
        assert a['history'][0]['validation']['queries']==14850
        print('EXACT RESUME PASSED',arm,flush=True)
        del a,b;gc.collect();torch.cuda.empty_cache()
    train.write_json(out/'passed.json',dict(passed=True,sources=sources,benchmark=bench,
        checks=['shapes; one stem invocation; unique params; checkpoint aliases',
                'G2 initially exact F4; private low-rank correction isolation; G3 bypass exact TS',
                'per-subject panel identities and padding; historical CPU/CUDA pairwise equivalence',
                'all three full contrastive batch forward/backward and memory benchmarks',
                'all three bitwise resume model/optimizer/RNG/selection; all14850 validation queries']))
    print('ALL PREFLIGHT CHECKS PASSED',flush=True)


if __name__=='__main__':main()
