"""Electrode-preserving spatial alternatives within a single shared decoder."""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from ensemble_experiments.compact_valcon.models import CompactDecoder

ARMS = ('electrode', 'distill')


class ElectrodeAttention(nn.Module):
    def __init__(self, channels=63, filters=40, bins=4, width=32):
        super().__init__()
        self.bins = bins
        self.descriptor = nn.Sequential(nn.LayerNorm(filters*bins), nn.Linear(filters*bins, width), nn.GELU())
        self.position = nn.Parameter(torch.empty(1, channels, width))
        nn.init.normal_(self.position, std=.02)
        self.qk = nn.Linear(width, 2*width, bias=False)
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        b, f, c, t = x.shape
        # Pool only to BUILD the attention matrix; values retain every temporal bin.
        descriptor = F.adaptive_avg_pool2d(x, (c, self.bins)).permute(0,2,1,3).flatten(2)
        q, k = self.qk(self.descriptor(descriptor)+self.position).chunk(2, dim=-1)
        weights = (q @ k.transpose(1,2) / math.sqrt(q.shape[-1])).softmax(-1)
        values = x.permute(0,2,1,3).reshape(b,c,f*t)
        context = (weights @ values).reshape(b,c,f,t).permute(0,2,1,3)
        return x + self.scale.tanh()*context


class ElectrodeDecoder(CompactDecoder):
    def __init__(self, channels=63):
        super().__init__('dual_head', channels)
        self.arm = 'electrode'
        self.electrode_attention = ElectrodeAttention(channels)
        self.spatial_norms = nn.ModuleList([copy.deepcopy(self.backbone.tsconv[5]) for _ in range(2)])
        # Shared spatial weights, branch-private running statistics, no unused BN.
        self.backbone.tsconv[5] = nn.Identity()

    def encode(self, eeg):
        temporal = self.backbone.tsconv[:4](eeg.unsqueeze(1))
        views = [temporal, self.electrode_attention(temporal)]
        outputs = []
        for x, norm, head in zip(views, self.spatial_norms, self.eeg_heads):
            x = norm(self.backbone.tsconv[4](x))
            x = self.backbone.tsconv[7](self.backbone.tsconv[6](x))
            x = self.backbone.projection(x).flatten(1)
            outputs.append(head(self.backbone.proj_eeg(x)))
        return outputs

    def size(self):
        result = super().size()
        result['eeg_parameters'] += sum(p.numel() for p in self.electrode_attention.parameters())
        result['eeg_parameters'] += sum(p.numel() for p in self.spatial_norms.parameters())
        return result


def make_model(arm, channels=63):
    if arm == 'electrode': return ElectrodeDecoder(channels)
    if arm == 'distill': return CompactDecoder('dual_head', channels)
    raise ValueError(arm)


def standardized_scores(features, columns):
    scores = torch.stack([F.normalize(e.float(),dim=-1) @ F.normalize(i.float()[columns],dim=-1).T
                          for e,i in features])
    return ((scores-scores.mean(-1,keepdim=True))/scores.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6)).mean(0)


def distillation_loss(student, teacher, objects, image_ids, temperature=1.0):
    # Unique image gallery avoids repeating nine identical candidate columns.
    keys = torch.stack([objects, image_ids], dim=1)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    n = int(inverse.max())+1
    columns = torch.full((n,),len(inverse),device=inverse.device,dtype=torch.long)
    columns.scatter_reduce_(0,inverse,torch.arange(len(inverse),device=inverse.device),reduce='amin')
    prediction = standardized_scores(student, columns)/temperature
    target = standardized_scores(teacher, columns).detach()/temperature
    return F.kl_div(prediction.log_softmax(-1),target.softmax(-1),reduction='batchmean')*temperature**2
