"""F1 selection-policy replay. Frozen architecture; no test-driven selection."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from ensemble_experiments.full_sharing_v2 import train as base
from ensemble_experiments.full_sharing_v2.models import FullPair

OUTPUT = base.REPO / 'results/things_eeg/f1_checkpoint_replay_20260915'
POLICIES = ('common_original', 'independent_same_horizon', 'independent_early',
            'common_100', 'independent_100')


def sources():
    result = base.sources()
    result[str(Path(__file__).relative_to(base.REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def slot():
    return dict(best=float('inf'), epoch=0, bad=0, stopped=None, model=None)


def update(item, loss, epoch, weights, freeze):
    if freeze and item['stopped'] is not None:
        return
    if loss < item['best']:
        item.update(best=float(loss), epoch=epoch, bad=0, model=weights())
    else:
        item['bad'] += 1
    if freeze and item['bad'] >= 20:
        item['stopped'] = epoch


def track_epoch(tracks, model, val, epoch):
    # CPU snapshots do not consume training RNG or modify live model tensors.
    snapshot = None
    def weights():
        nonlocal snapshot
        if snapshot is None:
            snapshot = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        return snapshot
    for j in range(2):
        update(tracks['branch'][j], val[j], epoch, weights, True)
        update(tracks['full'][j], val[j], epoch, weights, False)
    update(tracks['common'], np.mean(val), epoch, weights, True)
    update(tracks['common_full'], np.mean(val), epoch, weights, False)
    if tracks['common']['stopped'] == epoch:
        # These snapshots remain immutable; subsequent slot updates replace them.
        tracks['at_common_stop'] = [dict(x) for x in tracks['full']]


def selected(tracks):
    return dict(common_original=[tracks['common']]*2,
                independent_same_horizon=tracks['at_common_stop'] or tracks['full'],
                independent_early=tracks['branch'],
                common_100=[tracks['common_full']]*2,
                independent_100=tracks['full'])


def metadata(tracks, epochs):
    return dict(policies={name:[dict(epoch=x['epoch'], validation_loss=x['best']) for x in pair]
                          for name,pair in selected(tracks).items()},
                common_stop=tracks['common']['stopped'],
                independent_stops=[x['stopped'] for x in tracks['branch']],
                epochs_completed=epochs, selection='validation loss only',
                fusion='fixed equal row-z',
                note='Null stop means patience not exhausted before the budget cap.')


def evaluate(model, fold, out, tracks, epochs):
    plan = metadata(tracks, epochs)
    base.write_json(out/'selection.json', plan)  # before any outer test loading
    # Preserve every policy checkpoint independently of the final resumable state.
    for name,pair in selected(tracks).items():
        base.atomic_save(out/f'{name}.pth', dict(epochs=[x['epoch'] for x in pair],
                                              branch_source_models=[x['model'] for x in pair]))
    test = base.dataset([fold.subject], False)
    loader = DataLoader(test, batch_size=200, shuffle=False, num_workers=0)
    cache = {}
    keys = None
    results = {}
    for name,pair in selected(tracks).items():
        scores = []
        for j,item in enumerate(pair):
            epoch = item['epoch']
            if epoch not in cache:
                model.load_state_dict(item['model'])
                ee,ii,kk = base.collect(model,loader)
                assert kk.shape == (200,2) and len(np.unique(kk,axis=0)) == 200
                if keys is None: keys = kk
                np.testing.assert_array_equal(keys,kk)
                cache[epoch] = np.stack([e@i.T for e,i in zip(ee,ii)])
            scores.append(cache[epoch][j])
        scores = np.stack(scores)
        results[name] = dict(base.margin_metrics(scores), epochs=[x['epoch'] for x in pair],
                             solo_top1=[float(100*(x.argmax(1)==np.arange(200)).mean()) for x in scores])
        with (out/f'{name}.tmp').open('wb') as handle:
            np.savez_compressed(handle,scores=scores,object=keys[:,0],image_idx=keys[:,1])
        (out/f'{name}.tmp').replace(out/f'{name}_scores.npz')
    result = dict(subject=fold.subject, **plan, results=results)
    base.write_json(out/'complete.json',result)
    print('COMPLETE',json.dumps(result),flush=True)


def run(subject, out=None, epochs=100, max_steps=0, val_limit=0, stop_after=None, smoke=False):
    out = Path(out) if out else OUTPUT/f'sub-{subject:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (max_steps or val_limit or epochs!=100) and not smoke:
        raise ValueError('Restricted runs are preflight only')
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provenance = dict(subject=subject, sources=sources(), epochs=epochs,
                          max_steps=max_steps,val_limit=val_limit,smoke=smoke,
                          seed=3300,alignment_dim=128,mixup='pairwise',policies=POLICIES)
        # JSON converts tuples to lists; normalize for equality on resume.
        provenance = json.loads(json.dumps(provenance))
        if (out/'manifest.json').exists():
            assert json.loads((out/'manifest.json').read_text()) == provenance
        if (out/'complete.json').exists():
            print('SKIP',subject,flush=True); return
        base.write_json(out/'manifest.json',provenance)
        fold = base.Fold(subject)
        torch.manual_seed(3300)
        model = FullPair('F1').cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        tracks = dict(common=slot(), common_full=slot(), branch=[slot(),slot()],
                      full=[slot(),slot()], at_common_stop=None)
        history=[]; start=1
        if (out/'last.pth').exists():
            state=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            tracks,history,start=state['tracks'],state['history'],state['epoch']+1
            base.restore_rng(state['rng'])
            print('RESUME',subject,start,flush=True)
        else:
            torch.manual_seed(13300)
        print('START F1 REPLAY',subject,len(fold.loader),'steps',flush=True)
        for epoch in range(start,epochs+1):
            # Deliberately continue through 100; stopping decisions are virtual.
            fold.sampler.epoch=epoch-1
            fold.generator.manual_seed(3300+epoch)
            random.seed(3300+epoch)
            mix_rng=torch.Generator(device='cuda').manual_seed(3300+epoch)
            model.train(); torch.cuda.reset_peak_memory_stats()
            total=np.zeros(2); steps=0
            torch.cuda.synchronize(); tick=time.monotonic()
            for batch in fold.loader:
                eeg,images,subjects,objects,ids=[batch[i].cuda(non_blocking=True) for i in (0,1,3,4,5)]
                with torch.random.fork_rng(devices=[0]):
                    torch.cuda.set_rng_state(mix_rng.get_state())
                    eeg=base.cross_subject_stimulus_mix(eeg,objects,ids,subjects,alpha=.5,mixup_type='pairwise')
                    mix_rng.set_state(torch.cuda.get_rng_state())
                positives=base.build_image_positive_mask(objects,ids)
                opt.zero_grad(set_to_none=True)
                losses=model.losses(model(eeg,images,subjects),positives)
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                if steps%25==0:
                    base.gradient_diagnostics(model,losses)  # empty for F1, as in original
                loss.backward(); opt.step(); torch.cuda.synchronize()
                total += [float(x) for x in losses]; steps+=1
                if max_steps and steps>=max_steps:break
            train_seconds=time.monotonic()-tick
            tick=time.monotonic(); val=base.validation(model,fold,val_limit)
            if not np.isfinite(val).all():raise FloatingPointError('Nonfinite validation')
            track_epoch(tracks,model,val,epoch)
            row=dict(epoch=epoch,train_loss=(total/steps).tolist(),validation_loss=val,
                     validation_mean=float(np.mean(val)),steps=steps,
                     train_seconds=train_seconds,validation_seconds=time.monotonic()-tick,
                     common_best=tracks['common']['epoch'],common_stop=tracks['common']['stopped'],
                     branch_best=[x['epoch'] for x in tracks['branch']],
                     branch_stops=[x['stopped'] for x in tracks['branch']],
                     peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            history.append(row)
            base.atomic_save(out/'last.pth',dict(model=model.state_dict(),optimizer=opt.state_dict(),
                             rng=base.capture_rng(),tracks=tracks,history=history,epoch=epoch))
            base.write_json(out/'epochs.json',history)
            print(json.dumps(row),flush=True)
            if stop_after is not None and epoch>=stop_after:return
        if smoke:
            base.write_json(out/'complete.json',dict(smoke=True));return
        evaluate(model,fold,out,tracks,len(history))


def summary():
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[json.loads(p.read_text()) for p in sorted(OUTPUT.glob('sub-*/complete.json'))]
        base.write_json(OUTPUT/'summary.json',dict(completed=len(rows),expected=5,folds=rows,
            means={name:float(np.mean([r['results'][name]['top1'] for r in rows])) for name in POLICIES} if rows else {}))


def worker(index):
    expected=json.loads((OUTPUT/'preflight_passed.json').read_text())['sources']
    assert sources()==expected
    device=str(torch.cuda.get_device_properties(0).uuid)
    with (OUTPUT/f'device-{device}.lock').open('a') as gpu, (OUTPUT/f'worker{index}.lock').open('a') as lock:
        for handle in (gpu,lock):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=[1,3,5] if index==0 else [2,4]
        for s in tasks:
            assert sources()==expected
            status=dict(state='running',subject=s,host=socket.gethostname(),tasks=tasks)
            base.write_json(OUTPUT/f'status_worker{index}.json',status)
            with (OUTPUT/f'sub-{s:02d}.log').open('a',buffering=1) as log:
                child=subprocess.run([sys.executable,'-u','-m','ensemble_experiments.f1_checkpoint_replay',
                                      '--subject',str(s)],cwd=base.REPO,stdout=log,stderr=subprocess.STDOUT)
            if child.returncode:
                base.write_json(OUTPUT/f'status_worker{index}.json',dict(status,state='failed',exit_code=child.returncode))
                raise RuntimeError(f'Subject{s} failed')
            summary()
        base.write_json(OUTPUT/f'status_worker{index}.json',dict(state='complete',tasks=tasks))


def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,choices=range(1,6))
    p.add_argument('--worker',type=int,choices=(0,1)); a=p.parse_args()
    base.deterministic();OUTPUT.mkdir(parents=True,exist_ok=True)
    historical=json.loads((base.OUTPUT/'manifest.json').read_text())['sources']
    assert historical==base.sources(),'Frozen F1 source changed'
    if a.worker is not None:worker(a.worker)
    elif a.subject is not None:run(a.subject)
    else:p.error('Specify worker or subject')


if __name__=='__main__':main()
