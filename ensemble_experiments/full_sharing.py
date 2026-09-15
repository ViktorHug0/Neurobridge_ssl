"""Full-size ATM/TSConv sharing grid, matched at 128D backbone/alignment.

F1 is the independent matched control. F2--F6 tie temporal, spatial/projection,
convolutional, readout, and convolutional+readout blocks respectively. ATM keeps
its native attention front; TSConv keeps its raw-EEG front. Shared convolution
weights therefore test coefficient compatibility while the two branches still
execute their own front ends.
"""
import argparse, fcntl, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from module.eeg_encoder.model import TSConv_parameterizable
from module.eeg_encoder.atm.atm import ATMS
from module.projector import ProjectorLinear
from module.loss import ContrastiveLoss
from module.sampler import GroupedImageBatchSampler
from train import _GroupedSubset, cross_subject_stimulus_mix, build_image_positive_mask, seed_everything
from ensemble_experiments.compact_valcon.train import dataset, split_indices
from ensemble_experiments.mechanism_study.common import write_json, row_z

ROOT = Path('/nasbrain/p20fores/Neurobridge_SSL')
OUTPUT = ROOT/'results/things_eeg/full_sharing_20260911'
CONFIGS = {
    'F1': (), 'F2': ('T',), 'F3': ('S',), 'F4': ('T','S'),
    'F5': ('R',), 'F6': ('T','S','R'),
}
DESCRIPTIONS = {
    'F1':'matched independent ATM+TSConv', 'F2':'shared temporal convolution',
    'F3':'shared spatial convolution and projection', 'F4':'shared convolution block',
    'F5':'shared late EEG readout', 'F6':'shared convolution and readout',
}

def source_hash():
    h=hashlib.sha256()
    for p in [Path(__file__), ROOT/'module/eeg_encoder/model.py', ROOT/'module/eeg_encoder/atm/atm.py', ROOT/'ensemble_experiments/compact_valcon/train.py']:
        h.update(p.read_bytes())
    return h.hexdigest()[:20]

class FullPair(nn.Module):
    def __init__(self, config, channels=63):
        super().__init__(); self.config=config; self.ties=CONFIGS[config]
        seed_everything(3300)
        atm=ATMS(feature_dim=128,channels_num=channels,d_model=250,n_heads=4,e_layers=1,d_ff=256,
                 temporal_kernel=30,pool_kernel=51,pool_stride=5,temporal_filters=40,
                 spatial_filters=40,projection_filters=40)
        seed_everything(3300)
        ts=TSConv_parameterizable(feature_dim=128,channels_num=channels,temporal_kernel=30,
                                  pool_kernel=51,pool_stride=5,temporal_filters=40,
                                  spatial_filters=40,projection_filters=40)
        if 'T' in self.ties: atm.enc_eeg.tsconv[0]=ts.tsconv[0]
        if 'S' in self.ties:
            atm.enc_eeg.tsconv[4]=ts.tsconv[4]; atm.enc_eeg.projection[0]=ts.projection
        if 'R' in self.ties: atm.proj_eeg=ts.proj_eeg
        self.atm,self.ts=atm,ts
        self.eeg_heads=nn.ModuleList([ProjectorLinear(128,128),ProjectorLinear(128,128)])
        self.image_heads=nn.ModuleList([ProjectorLinear(3200,128),ProjectorLinear(3200,128)])
        self.criteria=nn.ModuleList([ContrastiveLoss(0.07,1.,1.,False,True,False,False,True) for _ in range(2)])

    def forward(self,eeg,images,subjects):
        b33,b28=images.split(3200,dim=-1)
        # ATM receives layer-28 images; TS receives layer-33 images.
        za=self.eeg_heads[0](self.atm(eeg,subjects)); zt=self.eeg_heads[1](self.ts(eeg))
        return [(za,self.image_heads[0](F.normalize(b28,dim=-1))),
                (zt,self.image_heads[1](F.normalize(b33,dim=-1)))]
    def loss(self,features,positive):
        return torch.stack([c.multi_positive_pair_loss(e.float(),i.float(),positive)
                            for c,(e,i) in zip(self.criteria,features)]).mean()
    def size(self):
        return {'parameters':sum(p.numel() for p in self.parameters()),
                'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'eeg_parameters':sum(p.numel() for p in self.atm.parameters())+sum(p.numel() for p in self.ts.parameters())+sum(p.numel() for p in self.eeg_heads.parameters()),
                'shared_ties':self.ties}

@torch.inference_mode()
def validate(model,loader,device,return_scores=False):
    model.eval(); losses=[]; es=[[],[]]; ims=[[],[]]; keys=[]
    for batch in loader:
        eeg,img,obj,ids=[batch[i].to(device) for i in (0,1,4,5)]
        f=model(eeg,img,torch.ones(len(eeg),dtype=torch.long,device=device)*10)
        losses.append(float(model.loss(f,build_image_positive_mask(obj,ids))))
        for j,(e,i) in enumerate(f): es[j].append(F.normalize(e,dim=-1).cpu().numpy()); ims[j].append(F.normalize(i,dim=-1).cpu().numpy())
        keys.extend(zip(obj.cpu().numpy(),ids.cpu().numpy()))
    if not return_scores:return float(np.mean(losses))
    e=[np.concatenate(x) for x in es]; im=[np.concatenate(x) for x in ims]; scores=np.stack([e[j]@im[j].T for j in range(2)])
    z=row_z(scores.astype(np.float64)).mean(0); truth=np.arange(len(z)); fused=z.argmax(1)==truth
    return {'top1':float(fused.mean()*100),'solo_top1':[float((s.argmax(1)==truth).mean()*100) for s in scores],
            'oracle_top1':float(((scores.argmax(2)==truth).any(0)).mean()*100),'scores':scores,'keys':np.array(keys)}

def train_one(config,subject,epochs=100,device='cuda:0'):
    out=OUTPUT/'runs'/config/'seed3300'/f'sub-{subject:02d}'; out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists(): return
    sources=[s for s in range(1,11) if s!=subject]; data=dataset(sources,True); tr,va,held=split_indices(data)
    train_data=_GroupedSubset(data,tr); sampler=GroupedImageBatchSampler(train_data,1024,samples_per_image=9,seed=3300)
    loader=DataLoader(train_data,batch_sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(Subset(data,va),batch_size=200,shuffle=False,num_workers=0)
    model=FullPair(config,data.channels_num).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    manifest=dict(config=config,description=DESCRIPTIONS[config],subject=subject,seed=3300,epochs=epochs,alignment_dim=128,backbone_dim=128,mixup='pairwise',mixup_alpha=.5,val_concept_seed=20260822,held_concepts=held,source_hash=source_hash(),model_size=model.size(),train_subjects=sources)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest: raise RuntimeError('manifest mismatch')
    write_json(out/'manifest.json',manifest)
    best=float('inf'); bad=0; records=[]; best_state=None; start=1
    if (out/'last.pth').exists():
        st=torch.load(out/'last.pth',map_location='cpu',weights_only=False); model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);
        for v in opt.state.values():
            for k,x in v.items():
                if isinstance(x,torch.Tensor):v[k]=x.to(device)
        best,bad,records,start=st['best'],st['bad'],st['records'],st['epoch']+1;best_state=st['best_state'];torch.set_rng_state(st['rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
    for epoch in range(start,epochs+1):
        model.train();sampler.epoch=epoch-1; tick=time.monotonic(); total=0;steps=0
        for batch in loader:
            eeg,img,subjects,obj,ids=[batch[i].to(device,non_blocking=True) for i in (0,1,3,4,5)]
            eeg=cross_subject_stimulus_mix(eeg,obj,ids,subjects,alpha=.5,mixup_type='pairwise'); positive=build_image_positive_mask(obj,ids)
            opt.zero_grad(set_to_none=True); loss=model.loss(model(eeg,img,subjects),positive)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();opt.step();total+=float(loss);steps+=1
        val=validate(model,val_loader,device); row=dict(epoch=epoch,train_loss=total/steps,val_loss=val,seconds=time.monotonic()-tick);records.append(row)
        if val<best:best,bad=val,0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,bad=bad,best_state=best_state,records=records,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'last.pth.tmp');os.replace(out/'last.pth.tmp',out/'last.pth')
        if bad>=20:break
    model.load_state_dict(best_state); test=dataset([subject],False); test_loader=DataLoader(test,batch_size=200,shuffle=False,num_workers=0); result=validate(model,test_loader,device,True); scores=result.pop('scores');keys=result.pop('keys');np.savez_compressed(out/'test_scores.npz',scores=scores,object=keys[:,0],image_idx=keys[:,1]); result.update(config=config,subject=subject,seed=3300,selected_epoch=min(records,key=lambda x:x['val_loss'])['epoch'],epochs_completed=len(records),**model.size(),training_seconds=sum(x['seconds'] for x in records));write_json(out/'complete.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS);p.add_argument('--subject',type=int);p.add_argument('--worker',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--prepare',action='store_true');a=p.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if a.prepare:
        write_json(OUTPUT/'manifest.json',dict(source_hash=source_hash(),seed=3300,configs=CONFIGS,subjects=list(range(1,6)),alignment_dim=128,backbone_dim=128));print('PREPARED',OUTPUT);return
    if a.config and a.subject: train_one(a.config,a.subject,device=a.device);return
    if a.worker is None:raise SystemExit('worker required')
    tasks=[(c,s) for s in range(1,6) for c in CONFIGS if ((s-1)*6+list(CONFIGS).index(c))%2==a.worker]
    for c,s in tasks: train_one(c,s,device=a.device)
    write_json(OUTPUT/f'status_worker{a.worker}.json',dict(state='complete',tasks=tasks))
if __name__=='__main__':main()
