import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from module.eeg_encoder.model import TSConv_parameterizable
from module.projector import ProjectorLinear
from module.loss import ContrastiveLoss
from ensemble_experiments.full_sharing_v2.models import FullPair, Stem, seeded

ARMS = ('G1', 'G2', 'G3')
DESCRIPTIONS = {
    'G1': 'One temporal stem, static spatial readout plus four-query 64D dynamic electrode pooling',
    'G2': 'F4 pathways, shared dense readouts with rank16 private corrections and private normalization',
    'G3': 'One TS backbone with one 64D electrode-attention residual before spatial pooling',
}


class ElectrodeFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1400)
        self.project = nn.Linear(1400, 64)
        self.position = nn.Parameter(torch.empty(1, 63, 64))
        nn.init.normal_(self.position, std=.02)

    def forward(self, x):
        descriptors = x.permute(0, 2, 1, 3).reshape(x.shape[0], 63, 1400)
        return self.project(self.norm(descriptors)) + self.position


class QueryPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ElectrodeFeatures()
        self.queries = nn.Parameter(torch.empty(4, 64))
        nn.init.normal_(self.queries, std=.02)
        self.keys, self.values = nn.Linear(64, 64), nn.Linear(64, 64)
        self.readout = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128),
                                     nn.GELU(), nn.Dropout(.5), nn.LayerNorm(128))

    def forward(self, x):
        h = self.features(x)
        weights = torch.softmax(self.queries @ self.keys(h).transpose(1, 2) / 8., dim=-1)
        return self.readout((weights @ self.values(h)).flatten(1))


class AttentionSide(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ElectrodeFeatures()
        self.norm = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.out = nn.Linear(64, 64)
        self.ff = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128), nn.GELU(),
                                nn.Dropout(.25), nn.Linear(128, 64))
        self.dropout = nn.Dropout(.25)
        self.to_features = nn.Linear(64, 1400)
        self.gain = nn.Parameter(torch.tensor(.1))

    def compute(self, x):
        h = self.features(x)
        q, k, v = self.qkv(self.norm(h)).reshape(x.shape[0],63,3,4,16).permute(2,0,3,1,4)
        attention = self.dropout(torch.softmax((q @ k.transpose(-1,-2))/4., dim=-1))
        h = h + self.dropout(self.out((attention @ v).transpose(1,2).reshape(x.shape[0],63,64)))
        h = h + self.ff(h)
        correction = self.to_features(h).reshape(x.shape[0],63,40,35).permute(0,2,1,3)
        return x + self.gain.tanh()*correction

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.compute, x, use_reentrant=False)
        return self.compute(x)


class CorrectedLinear(nn.Module):
    def __init__(self, shared, rank=16):
        super().__init__()
        self.shared = shared
        self.down = nn.Linear(shared.in_features, rank, bias=False)
        self.up = nn.Linear(rank, shared.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.shared(x) + self.up(self.down(x))


class Decoder(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        if arm == 'G2':
            self.pair = FullPair('F4')
            shared0 = self.pair.readout_a[0]
            shared1 = self.pair.readout_a[1].f[1]
            for j, readout in enumerate((self.pair.readout_a, self.pair.readout_b)):
                readout[0] = seeded(5100+j, lambda: CorrectedLinear(shared0))
                readout[1].f[1] = seeded(5200+j, lambda: CorrectedLinear(shared1))
        else:
            ts = seeded(3300, lambda: TSConv_parameterizable(feature_dim=128, temporal_kernel=30))
            self.stem = Stem(ts.tsconv[0],ts.tsconv[1])
            self.norm = nn.Sequential(ts.tsconv[2],ts.tsconv[3])
            self.spatial = nn.Sequential(*list(ts.tsconv.children())[4:],ts.projection)
            self.readout = ts.proj_eeg
            if arm == 'G1':self.dynamic = seeded(5400, QueryPool)
            else:self.side = seeded(5500, AttentionSide)
            self.eeg_heads = nn.ModuleList([seeded(3301+j, lambda: ProjectorLinear(128,128)) for j in range(2)])
            self.image_heads = nn.ModuleList([seeded(3303+j, lambda: ProjectorLinear(3200,128)) for j in range(2)])
            self.criteria = nn.ModuleList([ContrastiveLoss(.07,1.,1.,False,True,False,False,True) for _ in range(2)])

    def encode(self, eeg, subjects):
        if self.arm=='G2':return self.pair.encode(eeg, subjects)
        temporal = self.stem(eeg.unsqueeze(1))
        h = self.norm(temporal)
        if self.arm=='G3':h=self.side(h)
        static = self.readout(self.spatial(h).flatten(1))
        other = self.dynamic(temporal) if self.arm=='G1' else static
        return [head(x) for head,x in zip(self.eeg_heads,(static,other))]

    def forward(self, eeg, images, subjects):
        heads = self.pair.image_heads if self.arm=='G2' else self.image_heads
        return list(zip(self.encode(eeg,subjects),[h(x) for h,x in zip(heads,images.split(3200,-1))]))

    def losses(self, features, positives):
        criteria = self.pair.criteria if self.arm=='G2' else self.criteria
        return [c.multi_positive_pair_loss(e.float(),i.float(),positives) for c,(e,i) in zip(criteria,features)]

    def shared_blocks(self):
        if self.arm=='G2':
            p=self.pair
            return dict(stem=list(p.stem_a.parameters()),readout=list(p.readout_a[0].shared.parameters())+
                        list(p.readout_a[1].f[1].shared.parameters()))
        return dict(stem=list(self.stem.parameters()))

    def size(self):
        named=list(self.named_parameters(remove_duplicate=False));unique={id(p):p for _,p in named}
        images={id(p):p for n,p in named if 'image_heads.' in n}
        criteria={id(p):p for n,p in named if 'criteria.' in n}
        aliases={}
        for n,p in named:aliases.setdefault(id(p),[]).append(n)
        return dict(parameters=sum(p.numel() for p in unique.values()),
                    trainable_parameters=sum(p.numel() for p in unique.values() if p.requires_grad),
                    eeg_parameters=sum(p.numel() for k,p in unique.items() if k not in images and k not in criteria),
                    image_parameters=sum(p.numel() for p in images.values()),
                    aliases=[v for v in aliases.values() if len(v)>1],alignment_dim=128,
                    stem_filters=40,temporal_kernel=30,attention_width=250 if self.arm=='G2' else 64)
