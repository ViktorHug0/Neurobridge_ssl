"""Frozen existing ValCon teachers; never optimize, reselect, or refit them."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from module.eeg_encoder.atm.atm import ATMS
from module.projector import ProjectorLinear
from ensemble_experiments.compact_valcon.models import CompactDecoder
from ensemble_experiments.compact_valcon.train import ROOT, OUTPUT as PREVIOUS


def teacher_paths(subject):
    paths = list((ROOT/'results/things_eeg/honest_ensemble/atm_iv_valcon/seed3300').glob(
        f'*-sub-{subject:02d}/checkpoint_test_best.pth'))
    if len(paths)!=1: raise ValueError(f'Expected one ATM teacher for {subject}: {paths}')
    ts = PREVIOUS/'runs/single/seed3300'/f'sub-{subject:02d}'
    if not (ts/'complete.json').exists(): raise ValueError(f'Incomplete TS teacher: {ts}')
    return paths[0], ts/'best.pth'


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


class Teachers(nn.Module):
    def __init__(self, subject, split):
        super().__init__()
        atm_path, ts_path = teacher_paths(subject)
        cfg = json.loads((atm_path.parent/'train_config.json').read_text())
        ts_split = json.loads((ts_path.parent/'split.json').read_text())
        assert ts_split == split
        assert cfg['select_best_on']=='val' and cfg['val_concept_seed']==20260822
        assert cfg['val_concept_ratio']==.1 and cfg['mixup_type']=='pairwise'
        assert int(cfg['feature_dim'])==128 and cfg['eeg_backbone_dim']==128
        assert sorted(cfg['train_subject_ids'])==split['training_subjects']
        assert cfg['test_subject_ids']==[subject]
        self.atm = ATMS(feature_dim=128,channels_num=63)
        self.eeg_head = ProjectorLinear(128,128)
        self.image_head = ProjectorLinear(3200,128)
        state = torch.load(atm_path,map_location='cpu',weights_only=False)
        self.atm.load_state_dict(state['model_state_dict'])
        self.eeg_head.load_state_dict(state['eeg_projector_state_dict'])
        self.image_head.load_state_dict(state['img_projector_state_dict'])
        self.ts = CompactDecoder('single')
        self.ts.load_state_dict(torch.load(ts_path,map_location='cpu',weights_only=False)['model'])
        self.requires_grad_(False)
        self.eval()
        self.provenance = {name:dict(path=str(path),sha256=sha256(path))
                           for name,path in [('atm',atm_path),('tsconv',ts_path)]}

    @torch.no_grad()
    def forward(self, eeg, images, subjects, chunk_size=128):
        self.eval()
        # Chunking is for frozen evaluation only: no changes to student BN or negatives.
        eeg_ts, eeg_atm = [], []
        # ATM's legacy subject embedding chooses an unknown token for the WHOLE
        # batch if any ID is out of range. Preserve that convention across chunks.
        ids = subjects
        embedding = self.atm.encoder.enc_embedding.subject_embedding
        if embedding is not None and (subjects>=embedding.subject_embedding.num_embeddings).any():
            ids = torch.full_like(subjects,embedding.subject_embedding.num_embeddings)
        for start in range(0,len(eeg),chunk_size):
            x = eeg[start:start+chunk_size]
            eeg_ts.append(self.ts.encode(x)[0])
            eeg_atm.append(self.eeg_head(self.atm(x,ids[start:start+chunk_size])))
        return [(torch.cat(eeg_ts),self.ts.image_heads[0](images[:,:3200])),
                (torch.cat(eeg_atm),self.image_head(images[:,3200:]))]
