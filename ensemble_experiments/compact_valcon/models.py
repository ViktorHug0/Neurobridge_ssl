"""Three fixed designs; no teacher, independent full encoders, or test adaptation."""
import torch
from torch import nn
from module.eeg_encoder.model import TSConv_parameterizable
from module.eeg_encoder.atm.atm import ATMS
from module.projector import ProjectorLinear
from module.loss import ContrastiveLoss

ARMS = ('single', 'dual_head', 'hybrid')


class LocalResidual(nn.Module):
    def __init__(self, width=40):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Sequential(nn.Conv1d(width, width, 3, padding=1, groups=width),
                                  nn.GELU(), nn.Conv1d(width, width, 1))

    def forward(self, tokens):
        return tokens + self.conv(self.norm(tokens).transpose(1, 2)).transpose(1, 2)


class GlobalResidual(nn.Module):
    def __init__(self, width=40, length=35):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, length, width))
        nn.init.normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.25, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 80), nn.GELU(),
                                nn.Dropout(0.25), nn.Linear(80, width))

    def forward(self, tokens):
        x = self.norm(tokens + self.position)
        x = tokens + self.attention(x, x, x, need_weights=False)[0]
        return x + self.ff(x)


class CompactDecoder(nn.Module):
    def __init__(self, arm, channels=63):
        super().__init__()
        if arm not in (*ARMS, 'reference_atm'):
            raise ValueError(arm)
        self.arm = arm
        # Image inputs always arrive in the order IV33, IV28.
        self.targets = [28] if arm == 'reference_atm' else [33] if arm == 'single' else [33, 28]
        self.align_dims = [128] * len(self.targets)
        self.width = 128 if arm == 'reference_atm' else 1024 if arm == 'single' else 896
        self.backbone = (ATMS(feature_dim=128, channels_num=channels) if arm == 'reference_atm'
                         else TSConv_parameterizable(feature_dim=self.width, channels_num=channels,
                              temporal_kernel=30, pool_kernel=51, pool_stride=5))
        self.eeg_heads = nn.ModuleList([ProjectorLinear(self.width, d) for d in self.align_dims])
        self.image_heads = nn.ModuleList([ProjectorLinear(3200, d) for d in self.align_dims])
        self.criteria = nn.ModuleList([ContrastiveLoss(0.07, 1., 1., False, True, False, False, True)
                                       for _ in self.align_dims])
        if arm == 'hybrid':
            self.branches = nn.ModuleList([LocalResidual(), GlobalResidual()])

    def encode(self, eeg):
        if self.arm == 'reference_atm':
            subjects = torch.ones(eeg.shape[0], dtype=torch.long, device=eeg.device)
            return [self.eeg_heads[0](self.backbone(eeg, subjects))]
        if self.arm == 'hybrid':
            # The expensive temporal/spatial convolutions and their BN execute ONCE.
            tokens = self.backbone.projection(self.backbone.tsconv(eeg.unsqueeze(1)))
            tokens = tokens.squeeze(2).transpose(1, 2)
            # Shared dense tail weights; inexpensive tail computation is per branch.
            return [head(self.backbone.proj_eeg(branch(tokens).transpose(1, 2).flatten(1)))
                    for head, branch in zip(self.eeg_heads, self.branches)]
        hidden = self.backbone(eeg)
        return [head(hidden) for head in self.eeg_heads]

    def forward(self, eeg, images):
        blocks = dict(zip([33, 28], images.split(3200, dim=-1)))
        return list(zip(self.encode(eeg), [head(blocks[target])
                    for head, target in zip(self.image_heads, self.targets)]))

    def loss(self, features, positives):
        # Mean, not sum, keeps the shared encoder's loss scale comparable across arms.
        return torch.stack([c.multi_positive_pair_loss(e.float(), i.float(), positives)
                            for c, (e, i) in zip(self.criteria, features)]).mean()

    def size(self):
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    eeg_parameters=sum(p.numel() for p in self.backbone.parameters())
                        + sum(p.numel() for p in self.eeg_heads.parameters())
                        + (sum(p.numel() for p in self.branches.parameters()) if self.arm == 'hybrid' else 0),
                    backbone_width=self.width, alignment_dims=self.align_dims)
