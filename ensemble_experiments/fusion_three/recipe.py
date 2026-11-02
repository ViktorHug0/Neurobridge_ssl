"""Fixed pairwise-only recipe, isolated from unrelated main train.py changes."""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from module.dataset import EEGPreImageDataset
from module.sampler import GroupedImageBatchSampler

REPO=Path(__file__).resolve().parents[2]
EEG_DIR='/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
FEATURES=REPO/'data/things_eeg/image_feature'


def dataset(subjects, training):
    data=EEGPreImageDataset(subjects,train=training,eeg_data_dir=EEG_DIR,selected_channels=[],
        time_window=[0,250],image_feature_dir=str(FEATURES/'InternViT-6B_layer33_mean_8bit'),
        text_feature_dir='',image_aug=False,aug_image_feature_dirs=[],average=True,_random=False)
    extra=np.load(FEATURES/'InternViT-6B_layer28_mean_8bit'/('image_train.npy' if training else 'image_test.npy'))
    data.image_features=np.concatenate([data.image_features,extra],axis=-1).astype(np.float32)
    return data


def positive_mask(objects, images):
    return (objects[:,None]==objects[None,:]) & (images[:,None]==images[None,:])


def pairwise_mix(features, objects, images, subjects, alpha=.5):
    # Equivalent to the historical scalar pairwise branch, including RNG order.
    obj,img,sid=[x.detach().cpu().tolist() for x in (objects,images,subjects)]
    groups={}
    for i,key in enumerate(zip(obj,img)):groups.setdefault(key,[]).append(i)
    mixed=features.clone()
    for indices in groups.values():
        n=len(indices)
        if n<2:continue
        idx=torch.tensor(indices,device=features.device,dtype=torch.long)
        x=features[idx]; ss=[sid[i] for i in indices]
        if len(set(ss))==n:
            order=torch.randperm(n,device=features.device)
            offset=random.randint(1,n-1)
            partner=torch.empty(n,device=features.device,dtype=torch.long)
            partner[order]=torch.roll(order,shifts=offset)
        else:
            positions=[]
            for i,s in enumerate(ss):
                candidates=[j for j,t in enumerate(ss) if j!=i and t!=s] or [j for j in range(n) if j!=i]
                positions.append(random.choice(candidates))
            partner=torch.tensor(positions,device=features.device,dtype=torch.long)
        concentration=torch.full((n,),max(float(alpha),1e-3),device=features.device,dtype=torch.float32)
        lam=torch.distributions.Beta(concentration,concentration).sample().to(x.dtype)
        lam=lam.view(n,*([1]*(x.dim()-1)))
        mixed[idx]=lam*x+(1-lam)*x[partner]
    return mixed


class GroupedSubset(Subset):
    def get_image_group_indices(self):
        mapping={int(src):i for i,src in enumerate(self.indices)}
        return {key:[mapping[i] for i in rows if i in mapping]
                for key,rows in self.dataset.get_image_group_indices().items()
                if any(i in mapping for i in rows)}


class Fold:
    def __init__(self, subject):
        import json
        self.subject=subject;self.sources=[s for s in range(1,11) if s!=subject]
        self.data=dataset(self.sources,True)
        groups=self.data.get_image_group_indices();concepts=sorted({key[0] for key in groups})
        held=set(np.random.default_rng(20260822).permutation(concepts)[:int(np.floor(len(concepts)*.1))].tolist())
        tr,va=[],[]
        for (obj,_),rows in groups.items():(va if obj in held else tr).extend(rows)
        self.split=dict(training_subjects=self.sources,outer_subject=subject,held_concepts=sorted(held),
                        train_items=len(tr),validation_items=len(va))
        ref=REPO/f'results/things_eeg/tsconv_bb128_valcon_20260911/runs/single/seed3300/sub-{subject:02d}/split.json'
        assert self.split==json.loads(ref.read_text())
        train=GroupedSubset(self.data,tr)
        self.sampler=GroupedImageBatchSampler(train,1024,samples_per_image=9,seed=3300)
        self.generator=torch.Generator().manual_seed(3300)
        self.loader=DataLoader(train,batch_sampler=self.sampler,num_workers=0,pin_memory=True,generator=self.generator)
        self.val_loader=DataLoader(Subset(self.data,va),batch_size=200,shuffle=False,num_workers=0,
                                  generator=torch.Generator().manual_seed(3300))
        self.keys=np.array(sorted(key for key in groups if key[0] in held),dtype=np.int64)
        order=np.random.default_rng(3300).permutation(len(self.keys))
        self.panels=[]
        for start in range(0,len(order),200):
            query=order[start:start+200]
            candidate=query if len(query)==200 else np.concatenate([query,order[:200-len(query)]])
            assert len(np.unique(candidate))==200
            self.panels.append((query,candidate))


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
