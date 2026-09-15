"""Nine-source LOSO, reference-matched concept validation and resumable training."""
import argparse
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ensemble_experiments.compact_valcon.train import dataset, split_indices, capture_rng, restore_rng
from ensemble_experiments.mechanism_study.common import write_json, row_z, margin_metrics
from ensemble_experiments.mechanism_study.train_pair import atomic_save
from module.sampler import GroupedImageBatchSampler
from train import _GroupedSubset, cross_subject_stimulus_mix, build_image_positive_mask
from .models import FullPair, CONFIGS, DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUTPUT = REPO/'results/things_eeg/full_sharing_20260914'


def sources():
    files = list(Path(__file__).parent.glob('*.py')) + list((REPO/'module').rglob('*.py'))
    files += [REPO/'train.py',ROOT/'compact_valcon/train.py',ROOT/'compact_valcon/models.py',
              ROOT/'mechanism_study/common.py',ROOT/'mechanism_study/train_pair.py']
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Fold:
    def __init__(self, subject):
        self.subject = subject
        self.source_subjects = [s for s in range(1,11) if s != subject]
        self.data = dataset(self.source_subjects, True)
        tr,va,held = split_indices(self.data)
        self.split = dict(training_subjects=self.source_subjects,outer_subject=subject,
                          held_concepts=held,train_items=len(tr),validation_items=len(va))
        ref = REPO/f'results/things_eeg/tsconv_bb128_valcon_20260911/runs/single/seed3300/sub-{subject:02d}/split.json'
        assert json.loads(ref.read_text()) == self.split, 'Reference data split mismatch'
        self.train_data = _GroupedSubset(self.data,tr)
        self.sampler = GroupedImageBatchSampler(self.train_data,1024,samples_per_image=9,seed=3300)
        self.generator = torch.Generator().manual_seed(3300)
        self.loader = DataLoader(self.train_data,batch_sampler=self.sampler,num_workers=0,
                                 pin_memory=True,generator=self.generator)
        self.val_loader = DataLoader(Subset(self.data,va),batch_size=200,shuffle=False,
                                     num_workers=0,generator=torch.Generator().manual_seed(3300))


def batch_features(model,batch):
    return model(batch[0].cuda(non_blocking=True),batch[1].cuda(non_blocking=True),
                 batch[3].cuda(non_blocking=True))


@torch.inference_mode()
def validation(model,fold,limit=0):
    model.eval()
    totals = np.zeros(2)
    steps = 0
    for batch in fold.val_loader:
        f = batch_features(model,batch)
        positive = build_image_positive_mask(batch[4].cuda(),batch[5].cuda())
        totals += [float(x) for x in model.losses(f,positive)]
        steps += 1
        if limit and steps >= limit:
            break
    return (totals/steps).tolist()


@torch.inference_mode()
def collect(model,loader):
    model.eval()
    eegs,images = [[],[]],[[],[]]
    keys = []
    for batch in loader:
        for j,(e,i) in enumerate(batch_features(model,batch)):
            eegs[j].append(F.normalize(e,dim=-1).cpu().numpy())
            images[j].append(F.normalize(i,dim=-1).cpu().numpy())
        keys.append(np.stack([batch[4].numpy(),batch[5].numpy()],axis=1))
    return [np.concatenate(x) for x in eegs],[np.concatenate(x) for x in images],np.concatenate(keys)


def unit(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-8)


def calibration(model,fold):
    eeg,image,keys = collect(model,fold.val_loader)
    unique,inverse,counts = np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    # One query per held image, EEG representations averaged over the nine sources.
    aggregated = []
    for arrays in (eeg,image):
        parts=[]
        for x in arrays:
            y=np.zeros((len(unique),128),dtype=np.float64)
            np.add.at(y,inverse,x)
            parts.append(unit(y/counts[:,None]))
        aggregated.append(parts)
    eeg,image = aggregated
    order=np.random.default_rng(3300).permutation(len(unique))
    a,b,truth=[],[],[]
    for start in range(0,len(order),200):
        q=order[start:start+200]
        cand=q if len(q)==200 else np.concatenate([q,order[:200-len(q)]])
        a.append(eeg[0][q]@image[0][cand].T)
        b.append(eeg[1][q]@image[1][cand].T)
        truth.append(np.arange(len(q)))
    a,b,truth=row_z(np.concatenate(a)),row_z(np.concatenate(b)),np.concatenate(truth)
    grid=np.linspace(0,1,21)
    acc=[float(100*((w*a+(1-w)*b).argmax(1)==truth).mean()) for w in grid]
    idx=min(range(21),key=lambda j:(-acc[j],abs(grid[j]-.5),j))
    return dict(ts_weight=float(grid[idx]),validation_top1=acc[idx],grid=grid.tolist(),
                grid_top1=acc,queries=len(truth),selection='source-concept validation after checkpoint selection')


def gradient_diagnostics(model,losses):
    out={}
    for name,params in model.shared_blocks().items():
        vectors=[]
        for loss in losses:
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                      for g,p in zip(grads,params)]))
        a,b=vectors
        out[name]=dict(cosine=float(F.cosine_similarity(a,b,dim=0)),
                       norm_ts=float(a.norm()),norm_atm=float(b.norm()))
    return out


def run(config,fold,out,epochs=100,max_steps=0,val_limit=0,stop_after_epoch=None,smoke=False):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (max_steps or val_limit) and not smoke:
        raise ValueError('Batch limits require smoke=True')
    torch.manual_seed(3300)
    model=FullPair(config).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    manifest=dict(config=config,description=DESCRIPTIONS[config],subject=fold.subject,
        split=fold.split,epochs=epochs,patience=20,seed=3300,alignment_dim=128,backbone_dim=128,
        temporal_kernel=30,filters=40,precision='fp32',batch_size=1024,effective_batch=1017,
        mixup='pairwise',mixup_alpha=.5,learning_rate=3e-4,weight_decay=1e-4,
        validation_selection='minimum mean branch loss',image_targets=[33,28],
        raw_image_inputs=True,subject_token_policy='native, actual subject IDs including evaluation',
        sources=sources(),max_steps=max_steps,val_limit=val_limit,smoke=smoke,size=model.size())
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise RuntimeError(f'Manifest mismatch at {out}')
    if (out/'complete.json').exists():
        print('SKIP',out,flush=True)
        return json.loads((out/'complete.json').read_text())
    write_json(mf,manifest)
    best,bad,selected,start=float('inf'),0,0,1
    history,best_model=[],None
    if (out/'last.pth').exists():
        state=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        best,bad,selected,start=state['best'],state['bad'],state['selected'],state['epoch']+1
        history,best_model=state['history'],state['best_model']
        print('RESUME',config,fold.subject,start,flush=True)
    else:
        torch.manual_seed(13300)
    print('START',config,fold.subject,model.size()['parameters'],len(fold.loader),'steps',flush=True)
    for epoch in range(start,epochs+1):
        if bad>=20:
            break
        fold.sampler.epoch=epoch-1
        fold.generator.manual_seed(3300+epoch)
        random.seed(3300+epoch)
        mix_rng=torch.Generator(device='cuda').manual_seed(3300+epoch)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        total=np.zeros(2);step_times=[];diags=[];steps=0
        torch.cuda.synchronize();tick=time.monotonic()
        for batch in fold.loader:
            eeg,images,subjects,objects,ids=[batch[i].cuda(non_blocking=True) for i in (0,1,3,4,5)]
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.set_rng_state(mix_rng.get_state())
                eeg=cross_subject_stimulus_mix(eeg,objects,ids,subjects,alpha=.5,mixup_type='pairwise')
                mix_rng.set_state(torch.cuda.get_rng_state())
            positives=build_image_positive_mask(objects,ids)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize();t0=time.monotonic()
            losses=model.losses(model(eeg,images,subjects),positives)
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite train loss')
            diagnostic=steps%25==0
            if diagnostic:
                diags.append(gradient_diagnostics(model,losses))
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if not diagnostic:step_times.append(time.monotonic()-t0)
            total += [float(x) for x in losses]
            steps+=1
            if max_steps and steps>=max_steps:
                break
        train_seconds=time.monotonic()-tick
        t0=time.monotonic()
        val=validation(model,fold,val_limit)
        if not np.isfinite(val).all():
            raise FloatingPointError('Nonfinite validation loss')
        mean=float(np.mean(val))
        if mean<best:
            best,bad,selected=mean,0,epoch
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
        row=dict(epoch=epoch,train_loss=(total/steps).tolist(),validation_loss=val,
                 validation_mean=mean,steps=steps,train_seconds=train_seconds,
                 validation_seconds=time.monotonic()-t0,
                 step_seconds=float(np.median(step_times)) if step_times else None,
                 peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                 gradient_diagnostics=diags,best_epoch=selected,patience_count=bad)
        history.append(row)
        atomic_save(out/'last.pth',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(),
                    best=best,bad=bad,selected=selected,epoch=epoch,history=history,best_model=best_model))
        atomic_save(out/'best.pth',dict(model=best_model,epoch=selected,validation_loss=best))
        write_json(out/'epochs.json',history)
        print(json.dumps({k:v for k,v in row.items() if k!='gradient_diagnostics'}),flush=True)
        if stop_after_epoch is not None and epoch>=stop_after_epoch:
            return None
    model.load_state_dict(best_model)
    if smoke:
        write_json(out/'complete.json',dict(smoke=True,config=config,selected_epoch=selected))
        return
    cal=calibration(model,fold)
    write_json(out/'calibration.json',cal)  # committed before loading any outer test data
    test=dataset([fold.subject],False)
    ee,ii,keys=collect(model,DataLoader(test,batch_size=200,shuffle=False,num_workers=0))
    assert keys.shape==(200,2) and len(np.unique(keys,axis=0))==200
    scores=np.stack([e@i.T for e,i in zip(ee,ii)])
    metrics=margin_metrics(scores)
    z=row_z(scores.astype(np.float64))
    weighted=cal['ts_weight']*z[0]+(1-cal['ts_weight'])*z[1]
    metrics.update(top1_calibrated=float(100*(weighted.argmax(1)==np.arange(200)).mean()),
        solo_top1=[float(100*(s.argmax(1)==np.arange(200)).mean()) for s in scores],
        top5=float(100*(np.argsort(z.mean(0),axis=1)[:,-5:]==np.arange(200)[:,None]).any(1).mean()))
    with (out/'test_scores.tmp').open('wb') as handle:
        np.savez_compressed(handle,scores=scores,object=keys[:,0],image_idx=keys[:,1],
                            eeg_0=ee[0],eeg_1=ee[1],image_0=ii[0],image_1=ii[1])
    (out/'test_scores.tmp').replace(out/'test_scores.npz')
    result=dict(config=config,subject=fold.subject,selected_epoch=selected,epochs_completed=len(history),
                validation_loss=best,**metrics,**model.size(),
                train_seconds=sum(r['train_seconds'] for r in history),
                validation_seconds=sum(r['validation_seconds'] for r in history))
    write_json(out/'complete.json',result)
    print('COMPLETE',json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',choices=CONFIGS,required=True)
    p.add_argument('--subject',type=int,choices=range(1,6),required=True)
    a=p.parse_args()
    deterministic()
    run(a.config,Fold(a.subject),OUTPUT/f'runs/{a.config}/seed3300/sub-{a.subject:02d}')


if __name__=='__main__':
    main()
