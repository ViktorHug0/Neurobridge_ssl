"""TokenREVE (HISTORY.md R-series): scratch REVE blocks with a switchable tokenizer, the rmix encoder of the cv4
ensemble. Its TSConv time tokens reuse TSConv's validated stem (25-tap temporal conv, 25-sample box pool stride 2 at
120 Hz, BN, ELU, one spatial conv over all electrodes); its flatten readouts use TSConv's bottleneck head (Linear -> 64,
residual MLP, LayerNorm, Linear -> feature_dim).
"""
import torch
import torch.nn as nn

from module.eeg_encoder.ortho_encoders import _FusedTemporalConvPool, _head
from module.eeg_encoder.model import DomainBatchNorm2d


def _bn(c, boundary, norm=nn.BatchNorm2d):
    return DomainBatchNorm2d(c, boundary, norm) if boundary > 0 else norm(c)


def _run(layer, z, subject_ids):
    return layer(z, subject_ids) if isinstance(layer, DomainBatchNorm2d) else layer(z)


def _bottleneck_head(in_dim, feature_dim, code=64):
    head = _head(in_dim, code)
    head.append(nn.Linear(code, feature_dim))
    return head


class _Stem(nn.Module):
    """TSConv steps 1-4: (B, C, T) -> (B, F, T'). spatial=False stops before the spatial conv: (B, F, C, T')."""

    def __init__(self, channels_num, filters=40, kernel=25, pool=25, stride=2, drop=0.5, boundary=0, spatial=True):
        super().__init__()
        self.temporal = _FusedTemporalConvPool(filters, kernel, pool, stride)
        self.layers = nn.ModuleList([_bn(filters, boundary), nn.ELU()])
        if spatial:
            self.layers.extend([nn.Conv2d(filters, filters, (channels_num, 1)), _bn(filters, boundary), nn.ELU(),
                                nn.Dropout(drop)])
        self.spatial = spatial

    def n_time(self, t):
        return (t - self.temporal.effective_kernel) // self.temporal.pool_stride + 1

    def forward(self, x, subject_ids=None):
        z = self.temporal(x.unsqueeze(1))
        for layer in self.layers:
            z = _run(layer, z, subject_ids)
        return z.squeeze(2) if self.spatial else z


class TokenREVE(nn.Module):
    """Scratch REVE (br3_things: REVE blocks = RMSNorm, no biases, GEGLU, no final norm; learned positions; per-electrode
    trial z-score at 120 Hz) with a switchable tokenizer (HISTORY.md R-series):
      electrode  one token per electrode, its whole z-scored waveform (= br3)
      virtual    one token per LEARNED virtual channel (linear spatial mix of the z-scored electrodes, TSConv's spatial
                 filter as REVE's tokenizer)
      mixed      the electrode tokens + one token per time patch across all electrodes (global time context)
      time       time-patch tokens only
    readout: 'mean' over tokens (REVE), 'flatten' (TSConv's time-locked readout through the 64-d bottleneck head) or
    'split' (mean of the electrode tokens + time-locked flatten of the time tokens only).
    stride < patch: overlapping time patches; patch2: a second, finer set of time tokens (multi-scale).
    time_source='tsc': the time tokens come from TSConv's stem (temporal conv -> 0.2 s box pool -> spatial conv, 40
    filters x 36 steps), one token per step (avg-pooled by tsc_pool), instead of linear patches of the raw trace."""

    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, tokens='electrode', virtual=64,
                 patch=24, readout='mean', d_model=256, depth=6, heads=8, stride=None, patch2=0, time_source='patch',
                 tsc_pool=1, time_conv=0, attn='full', registers=0, elec_patch=0, type_ffn=False, tsc_axis='time',
                 elec_embed='linear', filt_tokens=False):
        super().__init__()
        from module.eeg_encoder.alljoined_encoders import _SwitchBlock
        self.tokens, self.patch, self.readout = tokens, patch, readout
        self.stride, self.patch2, self.time_source, self.tsc_pool = stride or patch, patch2, time_source, tsc_pool
        pos = lambda n: nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, n, d_model)), std=0.02)
        n = 0
        if tokens == 'virtual':
            self.mix = nn.Linear(channels_num, virtual, bias=False)
        self.elec_patch = elec_patch
        if elec_patch:  # round 6: each electrode split into patch tokens (REVE's fine tokenization), factorised positions
            self.embed = nn.Linear(elec_patch, d_model, bias=False)
            self.pos = pos(channels_num)
            self.epos = pos(eeg_sample_points // elec_patch)
            n += channels_num * self.epos.shape[1]
        elif tokens in ('electrode', 'virtual', 'mixed'):
            if elec_embed == 'conv':  # round 10: TSConv's temporal front end per electrode (16 filters x 25 taps, 0.2 s pool)
                self.embed = nn.Sequential(nn.Conv1d(1, 16, 25), nn.AvgPool1d(25, 2), nn.Flatten(),
                                           nn.Linear(16 * ((eeg_sample_points - 24 - 25) // 2 + 1), d_model, bias=False))
            else:
                self.embed = nn.Linear(eeg_sample_points, d_model, bias=False)
            self.pos = pos(virtual if tokens == 'virtual' else channels_num)
            n += self.pos.shape[1]
        self.tsc_axis = tsc_axis
        if tokens in ('mixed', 'time') and time_source == 'tsc' and tsc_axis == 'filter':
            # round 9: one token per TSConv stem FILTER (its 36-step time course), attention across filters + electrodes
            self.stem = _Stem(channels_num)
            self.patch_embed = nn.Linear(self.stem.n_time(eeg_sample_points), d_model, bias=False)
            self.patch_pos = pos(40)
            n += 40
        elif tokens in ('mixed', 'time') and time_source == 'tsc':
            self.stem = _Stem(channels_num)
            self.patch_embed = nn.Linear(40, d_model, bias=False)
            self.patch_pos = pos(self.stem.n_time(eeg_sample_points) // tsc_pool)
            n += self.patch_pos.shape[1]
        elif tokens in ('mixed', 'time'):
            self.patch_embed = nn.Linear(channels_num * patch, d_model, bias=False)
            self.patch_pos = pos((eeg_sample_points - patch) // self.stride + 1)
            n += self.patch_pos.shape[1]
            if patch2:
                self.patch_embed2 = nn.Linear(channels_num * patch2, d_model, bias=False)
                self.patch_pos2 = pos(eeg_sample_points // patch2)
                n += self.patch_pos2.shape[1]
        self.n_elec = n - (self.patch_pos.shape[1] + (self.patch_pos2.shape[1] if patch2 else 0) if tokens in ('mixed', 'time') else 0)
        self.elec_embed, self.filt_tokens = elec_embed, filt_tokens
        if filt_tokens:  # round 10: + 40 TSConv-stem filter tokens on top of the patch time tokens (union of token sets)
            self.fstem = _Stem(channels_num)
            self.fembed = nn.Linear(self.fstem.n_time(eeg_sample_points), d_model, bias=False)
            self.fpos = pos(40)
            n += 40
        self.blocks = nn.Sequential(*[_SwitchBlock(d_model, heads, 'rms', False, 'geglu', 4) for _ in range(depth)])
        self.reg = pos(registers) if registers else None  # learned register tokens (shared workspace), appended last
        if type_ffn:
            from braindecode.models.reve import RMSNorm
            hidden = int(d_model * 2.66)
            for blk in self.blocks:
                blk.type_ffn = nn.ModuleDict(dict(n2=RMSNorm(d_model), ff1=nn.Linear(d_model, 2 * hidden, bias=False),
                                                  ff2=nn.Linear(hidden, d_model, bias=False)))
                blk.split = self.n_elec
        if attn == 'gated':
            for blk in self.blocks:
                blk.gate = nn.Linear(d_model, d_model, bias=False)
        if attn == 'bipartite':  # each token attends to itself and to the OTHER token type only
            kind = torch.arange(n) >= self.n_elec
            self.register_buffer('mask', (kind[:, None] != kind[None, :]) | torch.eye(n, dtype=torch.bool), persistent=False)
        if time_conv:  # Squeezeformer's conv module on the time-token subsequence after every block (zero-init = rmix)
            self.tconv = nn.ModuleList(nn.Sequential(nn.Conv1d(d_model, d_model, time_conv, padding=time_conv // 2,
                                                               groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, 1))
                                       for _ in range(depth))
            self.tnorm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(depth))
            for m in self.tconv:
                nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)
        if readout == 'attnstat':  # Squeezeformer's readout: attention-weighted mean + std over the tokens
            self.pool_score = nn.Linear(d_model, 1)
        self.head = (nn.Linear(2 * d_model, feature_dim) if readout == 'attnstat' else
                     _bottleneck_head(n * d_model, feature_dim) if readout == 'flatten' else
                     _bottleneck_head(d_model + (n - self.n_elec) * d_model, feature_dim) if readout == 'split' else
                     nn.Linear(d_model, feature_dim))

    def forward(self, x, subject_ids=None):
        z = ((x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)).clamp(-15, 15)  # (B, C, T)
        toks = []
        if self.elec_patch:
            p = z.unfold(-1, self.elec_patch, self.elec_patch)                                # (B, C, P, patch)
            toks.append((self.embed(p) + self.pos[:, :, None] + self.epos[:, None]).flatten(1, 2))
        elif self.tokens in ('electrode', 'virtual', 'mixed'):
            e = self.mix(z.transpose(1, 2)).transpose(1, 2) if self.tokens == 'virtual' else z
            if self.elec_embed == 'conv':
                B, C, T = e.shape
                toks.append(self.embed(e.reshape(B * C, 1, T)).view(B, C, -1) + self.pos)
            else:
                toks.append(self.embed(e) + self.pos)
        if self.tokens in ('mixed', 'time') and self.time_source == 'tsc':
            f = self.stem(z)                                                                   # (B, 40, 36)
            if self.tsc_axis == 'filter':
                toks.append(self.patch_embed(f) + self.patch_pos)                               # 40 filter tokens
            else:
                f = nn.functional.avg_pool1d(f, self.tsc_pool) if self.tsc_pool > 1 else f
                toks.append(self.patch_embed(f.transpose(1, 2)) + self.patch_pos)
        elif self.tokens in ('mixed', 'time'):
            p = z.unfold(-1, self.patch, self.stride)                                         # (B, C, P, patch)
            toks.append(self.patch_embed(p.permute(0, 2, 1, 3).flatten(2)) + self.patch_pos)
            if self.patch2:
                p = z.unfold(-1, self.patch2, self.patch2)
                toks.append(self.patch_embed2(p.permute(0, 2, 1, 3).flatten(2)) + self.patch_pos2)
        if self.filt_tokens:
            toks.append(self.fembed(self.fstem(z)) + self.fpos)
        if self.reg is not None:
            toks.append(self.reg.expand(len(z), -1, -1))
        h = torch.cat(toks, 1)
        if hasattr(self, 'mask'):
            for blk in self.blocks:
                blk.attn_mask = self.mask
        if hasattr(self, 'tconv'):
            for blk, conv, norm in zip(self.blocks, self.tconv, self.tnorm):
                h = blk(h)
                t = h[:, self.n_elec:]
                h = torch.cat([h[:, :self.n_elec], t + conv(norm(t).transpose(1, 2)).transpose(1, 2)], 1)
        else:
            h = self.blocks(h)
        if self.readout == 'attnstat':
            w = torch.softmax(self.pool_score(h), dim=1)                                      # (B, N, 1)
            mu = (w * h).sum(1)
            sd = ((w * (h - mu[:, None]) ** 2).sum(1)).clamp_min(1e-6).sqrt()
            return self.head(torch.cat([mu, sd], 1))
        if self.readout == 'split':
            return self.head(torch.cat([h[:, :self.n_elec].mean(1), h[:, self.n_elec:].flatten(1)], 1))
        return self.head(h.flatten(1) if self.readout == 'flatten' else h.mean(1))


HYBRIDS = {'TokenREVE': TokenREVE}

if __name__ == '__main__':
    x, sid = torch.randn(6, 32, 120), torch.tensor([1, 2, 3, 21, 22, 25])
    for kw in ({'tokens': 'electrode'}, {'tokens': 'virtual'}, {'tokens': 'mixed'},
               {'tokens': 'time', 'patch': 6, 'readout': 'flatten'}, {'tokens': 'mixed', 'readout': 'split'},
               {'tokens': 'mixed', 'stride': 12}, {'tokens': 'mixed', 'patch2': 12},
               {'tokens': 'mixed', 'time_source': 'tsc'}, {'tokens': 'mixed', 'time_source': 'tsc', 'tsc_pool': 4},
               {'tokens': 'mixed', 'time_conv': 3}, {'tokens': 'mixed', 'attn': 'bipartite'},
               {'tokens': 'mixed', 'attn': 'gated'}, {'tokens': 'mixed', 'registers': 4},
               {'tokens': 'mixed', 'elec_patch': 24}, {'tokens': 'mixed', 'type_ffn': True, 'd_model': 224, 'depth': 5},
               {'tokens': 'mixed', 'readout': 'attnstat'}):
        m = TokenREVE(**kw)
        assert m(x).shape == (6, 1536), kw
        m(x).sum().backward()
        assert torch.allclose(m.eval()(x), m(3 * x + 1), atol=1e-3), 'per-electrode z-score: scale/offset-free'
        print('TokenREVE', kw, f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
