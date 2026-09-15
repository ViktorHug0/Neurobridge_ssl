"""Two follow-up arms using the established compact-ValCon data/selection code."""
import argparse
import fcntl
import json
import random
import os
import socket
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from module.sampler import GroupedImageBatchSampler
from train import _GroupedSubset, seed_everything, cross_subject_stimulus_mix, build_image_positive_mask
from ensemble_experiments.compact_valcon.train import (
    ROOT, dataset, split_indices, validation_loss, test_scores, capture_rng, restore_rng, atomic_save)
from ensemble_experiments.mechanism_study.common import write_json
from .models import ARMS, make_model, distillation_loss
from .teachers import Teachers

OUTPUT = ROOT/'results/things_eeg/compact_followup_20260908'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--subject',type=int,choices=range(1,11),required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--output',type=Path,default=OUTPUT/'runs')
    args = p.parse_args()
    if args.max_batches and args.output==OUTPUT/'runs': raise ValueError('Smoke needs separate output')
    out = args.output/args.arm/'seed3300'/f'sub-{args.subject:02d}'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists(): print('ALREADY_COMPLETE',out,flush=True); return
    config = {**vars(args),'output':str(args.output),'seed':3300,'mixup_type':'pairwise',
        'mixup_alpha':.5,'alignment_dims':[128,128],'batch_size':1024,'effective_batch_size':1017,
        'learning_rate':3e-4,'weight_decay':1e-4,'precision':'fp32','patience':20,
        'val_concept_seed':20260822,'val_concept_ratio':.1,
        'selection':'minimum mean-head source-concept validation loss',
        'kd_weight':1.0 if args.arm=='distill' else 0.,'kd_temperature':1.0,
        'teacher_chunk_size':128,'teacher_policy':'frozen, online on exact mixed training EEG only'}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Config changed in existing directory')
    write_json(out/'config.json',config)
    write_json(out/'runtime.json',dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        slurm_job=os.environ.get('SLURM_JOB_ID'),slurm_memory_mb=os.environ.get('SLURM_MEM_PER_NODE'),
        torch_version=torch.__version__,cuda_version=torch.version.cuda))
    data = dataset([s for s in range(1,11) if s!=args.subject],True)
    tr, va, held = split_indices(data)
    split = dict(training_subjects=[s for s in range(1,11) if s!=args.subject],
        outer_subject=args.subject,held_concepts=held,train_items=len(tr),validation_items=len(va))
    write_json(out/'split.json',split)
    tr_data = _GroupedSubset(data,tr)
    sampler = GroupedImageBatchSampler(tr_data,1024,samples_per_image=9,seed=3300)
    loader = DataLoader(tr_data,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,generator=torch.Generator().manual_seed(3300))
    val_loader = DataLoader(Subset(data,va),batch_size=200,shuffle=False,num_workers=0,pin_memory=True,
        generator=torch.Generator().manual_seed(3300))
    teacher = Teachers(args.subject,split).cuda() if args.arm=='distill' else None
    if teacher is not None: write_json(out/'teacher_provenance.json',teacher.provenance)
    seed_everything(3300)
    model = make_model(args.arm,data.channels_num).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    write_json(out/'model_size.json',model.size())
    start,best,bad,selected_epoch = 1,float('inf'),0,0
    best_weights,records = None,[]
    last = out/'last.pth'
    if last.exists():
        state = torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for value in optimizer.state.values():
            for k,v in value.items():
                if isinstance(v,torch.Tensor): value[k]=v.cuda()
        restore_rng(state['rng'])
        start,best,bad = state['epoch']+1,state['best'],state['bad']
        selected_epoch,best_weights,records = state['selected_epoch'],state['best_model'],state['records']
        atomic_save(out/'best.pth',dict(model=best_weights,epoch=selected_epoch,val_loss=best))
    else: seed_everything(13300)
    print('START',args.arm,args.subject,model.size(),'steps',len(loader),flush=True)
    for epoch in range(start,args.epochs+1):
        if bad>=20: break
        sampler.epoch=epoch-1;random.seed(3300+epoch)
        mix_rng=torch.Generator(device='cuda').manual_seed(3300+epoch)
        torch.cuda.reset_peak_memory_stats();model.train()
        totals=dict(supervised=0.,kd=0.,teacher_seconds=0.)
        steps=0;torch.cuda.synchronize();tick=time.monotonic()
        for batch in loader:
            eeg,images,subjects,objects,ids=[batch[i].cuda(non_blocking=True) for i in [0,1,3,4,5]]
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.set_rng_state(mix_rng.get_state())
                eeg=cross_subject_stimulus_mix(eeg,objects,ids,subjects,alpha=.5,mixup_type='pairwise')
                mix_rng.set_state(torch.cuda.get_rng_state())
            targets=None
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None:
                torch.cuda.synchronize();tt=time.monotonic()
                targets=teacher(eeg,images,subjects)
                torch.cuda.synchronize();totals['teacher_seconds']+=time.monotonic()-tt
            features=model(eeg,images)
            supervised=model.loss(features,build_image_positive_mask(objects,ids))
            kd=distillation_loss(features,targets,objects,ids) if targets is not None else supervised.new_zeros(())
            loss=supervised+config['kd_weight']*kd
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward();optimizer.step()
            totals['supervised']+=supervised.item();totals['kd']+=kd.item();steps+=1
            if args.max_batches and steps>=args.max_batches: break
        torch.cuda.synchronize();elapsed=time.monotonic()-tick
        vt=time.monotonic();val=validation_loss(model,val_loader,'cuda:0')
        if not np.isfinite(val): raise FloatingPointError('Non-finite validation loss')
        if val<best:
            best,bad,selected_epoch=val,0,epoch
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_save(out/'best.pth',dict(model=best_weights,epoch=epoch,val_loss=val))
        else: bad+=1
        row=dict(epoch=epoch,steps=steps,train_loss=(totals['supervised']+totals['kd'])/steps,
            supervised_loss=totals['supervised']/steps,kd_loss=totals['kd']/steps,
            validation_loss=val,train_seconds=elapsed,teacher_seconds=totals['teacher_seconds'],
            validation_seconds=time.monotonic()-vt,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records.append(row)
        atomic_save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),
            epoch=epoch,best=best,bad=bad,selected_epoch=selected_epoch,best_model=best_weights,records=records))
        write_json(out/'epochs.json',records);print(json.dumps(row),flush=True)
    selected=torch.load(out/'best.pth',map_location='cuda:0',weights_only=False)
    model.load_state_dict(selected['model'])
    test=dataset([args.subject],False)
    metrics,dump=test_scores(model,DataLoader(test,batch_size=200,shuffle=False),'cuda:0')
    np.savez_compressed(out/'test_scores.npz',**dump)
    result=dict(arm=args.arm,subject=args.subject,seed=3300,selected_epoch=selected['epoch'],
        epochs_completed=len(records),validation_loss=selected['val_loss'],**model.size(),**metrics,
        training_seconds=sum(r['train_seconds'] for r in records),
        validation_seconds=sum(r['validation_seconds'] for r in records),
        teacher_seconds=sum(r['teacher_seconds'] for r in records),
        teacher_parameters=sum(p.numel() for p in teacher.parameters()) if teacher is not None else 0)
    write_json(out/'complete.json',result);print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
