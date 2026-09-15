"""Approved F1--F6 topology, with 40 filters and 128D readouts/alignment."""
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from module.eeg_encoder.atm.atm import ATMS, Config, iTransformer
from module.eeg_encoder.model import TSConv_parameterizable
from module.projector import ProjectorLinear
from module.loss import ContrastiveLoss

CONFIGS = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')
DESCRIPTIONS = {
    'F1': 'Compatible independent stems; native attention before ATM convolution',
    'F2': 'F1 with tied temporal convolution weights; two stem computations',
    'F3': 'Independent raw-EEG stems; ATM electrode attention after its stem',
    'F4': 'F3 with one raw-EEG temporal stem computed once',
    'F5': 'F4 plus tied readout dense weights; private adapters and normalization',
    'F6': 'One TS backbone computed once; private local residual and attention branches',
}


def seeded(seed, build):
    state = torch.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        return build()
    finally:
        torch.set_rng_state(state)


class Stem(nn.Module):
    """Only convolution/pooling: checkpointing never repeats BatchNorm updates."""
    def __init__(self, conv, pool):
        super().__init__()
        self.conv, self.pool = conv, pool

    def forward(self, x):
        def compute(value):
            return self.pool(self.conv(value))
        if self.training and torch.is_grad_enabled():
            return checkpoint(compute, x, use_reentrant=False)
        return compute(x)


class ElectrodeResidual(nn.Module):
    """Full ATM-width attention over 63 electrode descriptors of length40*35."""
    def __init__(self, channels=63):
        super().__init__()
        cfg = Config(channels, d_model=250, n_heads=4, e_layers=1, d_ff=256)
        cfg.seq_len = 1400
        self.attention = iTransformer(cfg)
        self.norm = nn.LayerNorm(1400)
        self.output = nn.Linear(250, 1400)
        self.gain = nn.Parameter(torch.tensor(.1))

    def compute(self, x, subjects):
        b, f, c, t = x.shape
        descriptors = x.permute(0, 2, 1, 3).reshape(b, c, f*t)
        h = self.attention.enc_embedding(self.norm(descriptors), None, subjects)
        h, _ = self.attention.encoder(h, attn_mask=None)
        # Explicit electrode rows, excluding the prepended subject token.
        residual = self.output(h[:, 1:c+1]).reshape(b,c,f,t).permute(0,2,1,3)
        return x + self.gain.tanh() * residual

    def forward(self, x, subjects):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.compute, x, subjects, use_reentrant=False)
        return self.compute(x, subjects)


class TemporalContext(nn.Module):
    """Private ATM-width attention over the shared backbone's 35 spatially pooled tokens."""
    def __init__(self):
        super().__init__()
        cfg = Config(35, d_model=250, n_heads=4, e_layers=1, d_ff=256)
        cfg.seq_len = 40
        self.attention = iTransformer(cfg)
        self.norm = nn.LayerNorm(40)
        self.output = nn.Linear(250,128)
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, tokens, subjects):
        h = self.attention.enc_embedding(self.norm(tokens), None, subjects)
        h, _ = self.attention.encoder(h, attn_mask=None)
        return self.gain.tanh() * self.output(h[:,1:36].mean(1))


def identity_adapter():
    layer = nn.Conv2d(40,40,1)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(40).reshape(40,40,1,1))
        layer.bias.zero_()
    return layer


class FullPair(nn.Module):
    def __init__(self, config, channels=63):
        super().__init__()
        if config not in CONFIGS:
            raise ValueError(config)
        self.config = config
        self.targets = [33,28]
        ts = seeded(3300, lambda: TSConv_parameterizable(
            feature_dim=128, channels_num=channels, temporal_kernel=30))
        self.stem_a = Stem(ts.tsconv[0], ts.tsconv[1])
        self.norm_a = nn.Sequential(ts.tsconv[2], ts.tsconv[3])
        self.tail_a = nn.Sequential(*list(ts.tsconv.children())[4:], ts.projection)
        self.readout_a = ts.proj_eeg
        if config in ('F1','F2'):
            atm = seeded(3300, lambda: ATMS(feature_dim=128, channels_num=channels,
                                           temporal_kernel=30))
            patch = atm.enc_eeg[0]
            self.front = atm.encoder
            # Matched initial stems make F1/F2 identical before optimization.
            self.stem_b = copy.deepcopy(self.stem_a)
            if config == 'F2':
                self.stem_b.conv = self.stem_a.conv
            self.norm_b = nn.Sequential(patch.tsconv[2], patch.tsconv[3])
            self.tail_b = nn.Sequential(*list(patch.tsconv.children())[4:], patch.projection[0])
            self.readout_b = atm.proj_eeg
            # Express native ATM position-major dense weights in the common
            # channel-major input convention, preserving its native function.
            with torch.no_grad():
                w = self.readout_b[0].weight
                w.copy_(w.reshape(128,35,40).permute(0,2,1).reshape(128,1400).clone())
        elif config in ('F3','F4','F5'):
            self.stem_b = copy.deepcopy(self.stem_a) if config == 'F3' else self.stem_a
            self.norm_b = copy.deepcopy(self.norm_a)
            self.tail_b = seeded(4301, lambda: nn.Sequential(
                nn.Conv2d(40,40,(channels,1)),nn.BatchNorm2d(40),nn.ELU(),
                nn.Dropout(.5),nn.Conv2d(40,40,1)))
            self.readout_b = copy.deepcopy(self.readout_a)
            self.post_attention = seeded(4302, lambda: ElectrodeResidual(channels))
            # Present in all three controls; private and initially identity.
            self.adapter_a = identity_adapter()
            self.adapter_b = identity_adapter()
            if config == 'F5':
                self.readout_b[0] = self.readout_a[0]
                self.readout_b[1].f[1] = self.readout_a[1].f[1]
        else:
            self.local_residual = seeded(4303, lambda: nn.Sequential(
                nn.LayerNorm(128),nn.Linear(128,32),nn.GELU(),nn.Linear(32,128)))
            self.context = seeded(4304, TemporalContext)
        self.eeg_heads = nn.ModuleList([seeded(3301+j, lambda: ProjectorLinear(128,128)) for j in range(2)])
        self.image_heads = nn.ModuleList([seeded(3303+j, lambda: ProjectorLinear(3200,128)) for j in range(2)])
        self.criteria = nn.ModuleList([ContrastiveLoss(.07,1.,1.,False,True,False,False,True) for _ in range(2)])

    def encode(self, eeg, subjects):
        raw = eeg.unsqueeze(1)
        sa = self.stem_a(raw)
        if self.config in ('F1','F2'):
            sb = self.stem_b(self.front(eeg,None,subjects).unsqueeze(1))
        elif self.config == 'F3':
            sb = self.stem_b(raw)
        else:
            sb = sa
        ta = self.tail_a(self.norm_a(sa))
        if self.config == 'F6':
            shared = self.readout_a(ta.flatten(1))
            ha = shared + self.local_residual(shared)
            hb = shared + self.context(ta.squeeze(2).transpose(1,2), subjects)
        else:
            nb = self.norm_b(sb)
            if self.config in ('F3','F4','F5'):
                nb = self.post_attention(nb,subjects)
            tb = self.tail_b(nb)
            if self.config in ('F3','F4','F5'):
                ta, tb = self.adapter_a(ta), self.adapter_b(tb)
            ha, hb = self.readout_a(ta.flatten(1)), self.readout_b(tb.flatten(1))
        return [head(h) for head,h in zip(self.eeg_heads,(ha,hb))]

    def forward(self, eeg, images, subjects):
        # Raw IV33/IV28 targets, exactly as in the standalone reference trainer.
        image_features = [head(x) for head,x in zip(self.image_heads,images.split(3200,dim=-1))]
        return list(zip(self.encode(eeg,subjects), image_features))

    def losses(self, features, positives):
        return [c.multi_positive_pair_loss(e.float(),i.float(),positives)
                for c,(e,i) in zip(self.criteria,features)]

    def shared_blocks(self):
        blocks = {}
        if self.config in ('F2','F4','F5','F6'):
            blocks['stem'] = list(self.stem_a.parameters())
        if self.config in ('F5','F6'):
            blocks['readout'] = list(self.readout_a[0].parameters()) + list(self.readout_a[1].f[1].parameters())
        return blocks

    def size(self):
        named = list(self.named_parameters(remove_duplicate=False))
        unique = {id(p):p for _,p in named}
        eeg = {id(p):p for n,p in named if not n.startswith(('image_heads.','criteria.'))}
        aliases = {}
        for n,p in named:
            aliases.setdefault(id(p),[]).append(n)
        return dict(parameters=sum(p.numel() for p in unique.values()),
                    eeg_parameters=sum(p.numel() for p in eeg.values()),
                    image_parameters=sum(p.numel() for p in self.image_heads.parameters()),
                    registered_parameters=sum(p.numel() for _,p in named),
                    aliases=[v for v in aliases.values() if len(v)>1],
                    backbone_dim=128,alignment_dim=128,filters=40,
                    activation_checkpointing='pure temporal conv/pool and post-stem attention only')
