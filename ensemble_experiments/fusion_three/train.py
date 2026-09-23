import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ensemble_experiments.mechanism_study.common import write_json, row_z, margin_metrics
from .models import Decoder, ARMS, DESCRIPTIONS
from .recipe import Fold, REPO, dataset, positive_mask, pairwise_mix, capture_rng, restore_rng

OUTPUT=REPO/'results/things_eeg/fusion_three_20260915'


def source_hashes():
    paths=set(Path(__file__).parent.glob('*.py'))
    paths.update((REPO/'module').rglob('*.py'))
    paths.add(REPO/'ensemble_experiments/full_sharing_v2/models.py')
    paths.add(REPO/'ensemble_experiments/mechanism_study/common.py')
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def deterministic():
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def atomic_save(path, state):
    path=Path(path);temp=path.with_suffix('.tmp')
    torch.save(state,temp);temp.replace(path)


@torch.inference_mode()
def collect(model, loader):
    model.eval();ee=[[],[]];ii=[[],[]];keys=[];subjects=[]
    for batch in loader:
        features=model(batch[0].cuda(non_blocking=True),batch[1].cuda(non_blocking=True),batch[3].cuda())
        for j,(e,i) in enumerate(features):
            ee[j].append(F.normalize(e,dim=-1).cpu().numpy())
            ii[j].append(F.normalize(i,dim=-1).cpu().numpy())
        keys.append(np.stack([batch[4].numpy(),batch[5].numpy()],axis=1));subjects.append(batch[3].numpy())
    return [np.concatenate(x) for x in ee],[np.concatenate(x) for x in ii],np.concatenate(keys),np.concatenate(subjects)


def panel_metrics(ee,ii,keys,subjects,fold):
    index={tuple(k):i for i,k in enumerate(fold.keys)}
    mapping=np.array([index[tuple(k)] for k in keys])
    by_subject={};solo_correct=np.zeros(2,dtype=int);total=0;correct=0
    # Candidate images are independent of EEG subject. Use the first source's rows.
    first=np.flatnonzero(subjects==fold.sources[0]);order=np.argsort(mapping[first])
    first=first[order];np.testing.assert_array_equal(mapping[first],np.arange(len(index)))
    candidates=[x[first] for x in ii]
    for s in fold.sources:
        rows=np.flatnonzero(subjects==s);rows=rows[np.argsort(mapping[rows])]
        np.testing.assert_array_equal(mapping[rows],np.arange(len(index)))
        for j in range(2):np.testing.assert_allclose(ii[j][rows],candidates[j],rtol=1e-5,atol=1e-6)
        subject_correct=0;queries=0
        for q,c in fold.panels:
            scores=np.stack([ee[j][rows[q]].astype(np.float64)@candidates[j][c].astype(np.float64).T for j in range(2)])
            truth=np.arange(len(q))  # panel construction puts query keys first
            assert np.array_equal(fold.keys[q],fold.keys[c[:len(q)]])
            fused=row_z(scores).mean(0)
            n=int((fused.argmax(-1)==truth).sum())
            subject_correct+=n;queries+=len(q)
            solo_correct+=(scores.argmax(-1)==truth).sum(-1)
        by_subject[str(s)]=100*subject_correct/queries
        correct+=subject_correct;total+=queries
    return dict(correct=correct,queries=total,top1=100*correct/total,
                subject_top1=by_subject,solo_top1=(100*solo_correct/total).tolist())


def validate(model,fold):
    return panel_metrics(*collect(model,fold.val_loader),fold)


def monitor_test(model, subject):
    """Diagnostic only: preserve training RNG and mode; never drive selection."""
    rng = capture_rng()
    training = model.training
    try:
        loader = DataLoader(dataset([subject],False),batch_size=200,shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(3300))
        ee,ii,keys,sids = collect(model,loader)
        assert len(keys)==200 and len(np.unique(keys,axis=0))==200 and np.all(sids==subject)
        scores = np.stack([e@i.T for e,i in zip(ee,ii)])
        fused = row_z(scores).mean(0)
        truth = np.arange(200)
        return dict(top1=float(100*(fused.argmax(1)==truth).mean()),
                    top5=float(100*(np.argsort(fused,axis=1)[:,-5:]==truth[:,None]).any(1).mean()),
                    solo_top1=[float(100*(s.argmax(1)==truth).mean()) for s in scores],
                    selection_role='monitor_only')
    finally:
        restore_rng(rng)
        model.train(training)


def gradients(model, losses):
    out={}
    for name,params in model.shared_blocks().items():
        vectors=[]
        for loss in losses:
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(grads,params)]))
        out[name]=dict(cosine=float(F.cosine_similarity(*vectors,dim=0)),norm0=float(vectors[0].norm()),norm1=float(vectors[1].norm()))
    return out


def run(arm,subject,source_manifest,out=None,epochs=100,max_steps=0,stop_after=None,smoke=False):
    if (epochs!=100 or max_steps) and not smoke:raise ValueError('Limits are smoke-only')
    out=Path(out) if out else OUTPUT/f'runs/{arm}/sub-{subject:02d}'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(arm=arm,subject=subject,sources=source_manifest,epochs=epochs,max_steps=max_steps,smoke=smoke,
            seed=3300,patience=20,batch_size=1024,effective_batch=1017,mixup='pairwise',alpha=.5,
            learning_rate=3e-4,weight_decay=1e-4,precision='fp32',alignment_dim=128,
            selection='strict maximum fixed-panel source-subject fused top1; ties keep earliest',
            image_targets=[33,28],fusion='equal row-z',attention_description=DESCRIPTIONS[arm],
            test_monitor='each epoch; diagnostic only; RNG preserved')
        if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
        if (out/'complete.json').exists():print('SKIP',arm,subject,flush=True);return
        write_json(out/'manifest.json',manifest)
        fold=Fold(subject)
        write_json(out/'split.json',fold.split)
        write_json(out/'panels.json',dict(keys=fold.keys.tolist(),panels=[dict(query=q.tolist(),candidate=c.tolist()) for q,c in fold.panels],
            query_unit='one EEG per source subject and held image; no cross-subject averaging',panel_seed=3300))
        torch.manual_seed(3300);model=Decoder(arm).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        write_json(out/'model_size.json',model.size())
        best=-1;bad=0;selected=0;start=1;best_model=None;history=[]
        if (out/'last.pth').exists():
            state=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            best,bad,selected,start=state['best'],state['bad'],state['selected'],state['epoch']+1
            best_model,history=state['best_model'],state['history'];restore_rng(state['rng'])
            print('RESUME',arm,subject,start,flush=True)
        else:torch.manual_seed(13300)
        print('START',arm,subject,model.size(),len(fold.loader),'steps',flush=True)
        for epoch in range(start,epochs+1):
            if bad>=20:break
            fold.sampler.epoch=epoch-1;fold.generator.manual_seed(3300+epoch);random.seed(3300+epoch)
            mix_rng=torch.Generator(device='cuda').manual_seed(3300+epoch)
            model.train();torch.cuda.reset_peak_memory_stats();total=np.zeros(2);steps=0;times=[];diags=[]
            torch.cuda.synchronize();tick=time.monotonic()
            for batch in fold.loader:
                eeg,images,subjects,objects,ids=[batch[i].cuda(non_blocking=True) for i in (0,1,3,4,5)]
                with torch.random.fork_rng(devices=[0]):
                    torch.cuda.set_rng_state(mix_rng.get_state())
                    eeg=pairwise_mix(eeg,objects,ids,subjects)
                    mix_rng.set_state(torch.cuda.get_rng_state())
                opt.zero_grad(set_to_none=True);positive=positive_mask(objects,ids)
                torch.cuda.synchronize();t0=time.monotonic()
                losses=model.losses(model(eeg,images,subjects),positive);loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                diagnostic=steps%25==0
                if diagnostic:diags.append(gradients(model,losses))
                loss.backward();opt.step();torch.cuda.synchronize()
                if not diagnostic:times.append(time.monotonic()-t0)
                total+=[float(x) for x in losses];steps+=1
                if max_steps and steps>=max_steps:break
            train_seconds=time.monotonic()-tick
            tick=time.monotonic();val=validate(model,fold)
            if val['correct']>best:
                best,bad,selected=val['correct'],0,epoch
                best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:bad+=1
            row=dict(epoch=epoch,train_loss=(total/steps).tolist(),validation=val,selected_epoch=selected,
                     patience_count=bad,steps=steps,train_seconds=train_seconds,
                     validation_seconds=time.monotonic()-tick,median_step_seconds=float(np.median(times)) if times else None,
                     peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,gradient_diagnostics=diags)
            test_tick=time.monotonic()
            row['test']=monitor_test(model,subject)
            row['test_seconds']=time.monotonic()-test_tick
            history.append(row)
            atomic_save(out/'last.pth',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(),epoch=epoch,
                best=best,bad=bad,selected=selected,best_model=best_model,history=history))
            atomic_save(out/'best.pth',dict(model=best_model,epoch=selected,validation_correct=best))
            write_json(out/'epochs.json',history)
            print(json.dumps({k:v for k,v in row.items() if k!='gradient_diagnostics'}),flush=True)
            if stop_after is not None and epoch>=stop_after:return
        if smoke:
            write_json(out/'complete.json',dict(smoke=True,arm=arm,subject=subject));return
        write_json(out/'selection.json',dict(epoch=selected,validation_correct=best,
                   queries=history[selected-1]['validation']['queries'],rule=manifest['selection']))
        model.load_state_dict(best_model)
        # Final reported scores use the validation-selected model, never test selection.
        test=dataset([subject],False)
        ee,ii,keys,sids=collect(model,DataLoader(test,batch_size=200,shuffle=False,num_workers=0))
        assert len(keys)==200 and len(np.unique(keys,axis=0))==200 and np.all(sids==subject)
        scores=np.stack([e@i.T for e,i in zip(ee,ii)])
        with (out/'test_scores.tmp').open('wb') as handle:
            np.savez_compressed(handle,scores=scores,object=keys[:,0],image_idx=keys[:,1],subject=sids,
                                eeg_0=ee[0],eeg_1=ee[1],image_0=ii[0],image_1=ii[1])
        (out/'test_scores.tmp').replace(out/'test_scores.npz')
        metrics=margin_metrics(scores)
        metrics['solo_top1']=[float(100*(s.argmax(1)==np.arange(200)).mean()) for s in scores]
        metrics['top5']=float(100*(np.argsort(row_z(scores).mean(0),axis=1)[:,-5:]==np.arange(200)[:,None]).any(1).mean())
        result=dict(arm=arm,subject=subject,**metrics,**model.size(),selected_epoch=selected,epochs_completed=len(history),
                    train_seconds=sum(x['train_seconds'] for x in history),validation_seconds=sum(x['validation_seconds'] for x in history),
                    validation_top1=history[selected-1]['validation']['top1'])
        write_json(out/'complete.json',result);print('COMPLETE',json.dumps(result),flush=True)


def summarize():
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[json.loads(p.read_text()) for p in sorted((OUTPUT/'runs').glob('G*/sub-*/complete.json'))]
        means={}
        for arm in ARMS:
            subset=[r for r in rows if r['arm']==arm]
            if subset:means[arm]=dict(n=len(subset),top1=float(np.mean([r['top1'] for r in subset])))
        write_json(OUTPUT/'summary.json',dict(completed=len(rows),expected=15,means=means,folds=rows))


def worker(arm):
    # Each persistent worker imports training/model code once for all five folds.
    # Hash checks occur at startup/resume; unrelated edits cannot halt a live queue.
    provenance=source_hashes()
    approved=json.loads((OUTPUT/'preflight/passed.json').read_text())
    assert approved['passed'] and approved['sources']==provenance,'Source changed since preflight'
    device=str(torch.cuda.get_device_properties(0).uuid)
    with (OUTPUT/f'device-{device}.lock').open('a') as gpu,(OUTPUT/f'worker-{arm}.lock').open('a') as lock:
        for h in (gpu,lock):fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for subject in range(1,6):
            status=dict(state='running',arm=arm,subject=subject,host=socket.gethostname())
            write_json(OUTPUT/f'status-{arm}.json',status)
            try:
                run(arm,subject,provenance)
                summarize();gc.collect();torch.cuda.empty_cache()
            except BaseException as exc:
                write_json(OUTPUT/f'status-{arm}.json',dict(status,state='failed',error=repr(exc)))
                raise
        write_json(OUTPUT/f'status-{arm}.json',dict(state='complete',arm=arm,subjects=list(range(1,6))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args()
    deterministic();OUTPUT.mkdir(parents=True,exist_ok=True);worker(a.arm)
