"""Hybrid TSConv / Squeezeformer / REVE encoders (ARCH_LOG.md): empirical search over where convolution and attention
act on the signal. All share TSConv's validated stem (25-tap temporal conv, 25-sample box pool stride 2 at 120 Hz,
BN, ELU, one spatial conv over all electrodes) unless the design replaces a piece of it, and TSConv's bottleneck head
(Linear -> 64, residual MLP, LayerNorm, Linear -> feature_dim). Every BatchNorm is per dataset when
domain_bn_boundary > 0 (THINGS subjects >= boundary), unseen subjects take the primary path.

FilterAttnTSConv    attention ACROSS the 40 spatio-temporal filters (token = one filter's 36-step time course)
TimeAttnFlatTSConv  Squeezeformer's attention over time tokens, but TSConv's time-locked flatten readout
VirtualSensorFormer REVE's patch tokens on K learned virtual sensors (spatial projection of the raw EEG)
DynSpatialTSConv    TSConv whose spatial filters get a trial-adaptive residual (learned queries attend over electrodes)
DualPathTSConv      TSConv split into a slow path (the stem) and a fast path (short kernel, short pool), concatenated
TimeFreqXTSConv     time tokens (content) cross-attend to frequency-bin tokens of the trial's own spectrum (context)
TSConvPlus          TSConv with one knob per axis: learned pooling, spatial multiplier, post-spatial sepconv, query readout
"""
import torch
import torch.nn as nn

from ensemble_experiments.architectures.ortho_encoders import (
    _ConformerFeedForward, _DifferentialGatedAttentionBlock, _FastSelfAttention, _FastSqueezeformerBlock,
    _FusedTemporalConvPool, _GatedAttentionBlock, _head,
)
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


class _Block(nn.Module):
    """pre-norm MHSA + FFN."""

    def __init__(self, d, heads, drop, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = _FastSelfAttention(d, heads, drop)
        self.ffn = _ConformerFeedForward(d, expansion, drop)

    def forward(self, z):
        z = z + self.attn(self.norm(z))
        return z + self.ffn(z)


class FilterAttnTSConv(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, filters=40, d_model=64, depth=2,
                 heads=4, drop=0.1, domain_bn_boundary=0):
        super().__init__()
        self.stem = _Stem(channels_num, filters, boundary=domain_bn_boundary)
        t = self.stem.n_time(eeg_sample_points)
        self.embed = nn.Sequential(nn.Linear(t, d_model), nn.LayerNorm(d_model))
        self.filter_pos = nn.Parameter(torch.zeros(1, filters, d_model))
        nn.init.trunc_normal_(self.filter_pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(d_model, heads, drop) for _ in range(depth)])
        self.head = _bottleneck_head(filters * d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        z = self.embed(self.stem(x, subject_ids)) + self.filter_pos  # (B, F, d): one token per filter
        for block in self.blocks:
            z = block(z)
        return self.head(z.flatten(1))


class TimeAttnFlatTSConv(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, filters=40, d_model=128, depth=2,
                 heads=4, drop=0.1, block='mhsa', domain_bn_boundary=0):
        # block (H3, the attention kind): 'mhsa' pre-norm softmax attention + FFN | 'gated' sigmoid-gated attention +
        # SwiGLU | 'diff' differential attention | 'sqf' Squeezeformer block (attention + depthwise conv module + FFN)
        super().__init__()
        self.stem = _Stem(channels_num, filters, boundary=domain_bn_boundary)
        t = self.stem.n_time(eeg_sample_points)
        self.embed = nn.Sequential(nn.Linear(filters, d_model), nn.LayerNorm(d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, t, d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        make = {'mhsa': lambda: _Block(d_model, heads, drop),
                'gated': lambda: _GatedAttentionBlock(d_model, heads, 2, drop),
                'diff': lambda: _DifferentialGatedAttentionBlock(d_model, heads, drop),
                'sqf': lambda: _FastSqueezeformerBlock(d_model, heads, 15, 2, drop, domain_bn_boundary)}[block]
        self.blocks = nn.ModuleList([make() for _ in range(depth)])
        self.head = _bottleneck_head(t * d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        z = self.embed(self.stem(x, subject_ids).transpose(1, 2)) + self.time_pos  # (B, T', d): one token per step
        for block in self.blocks:
            z = block(z, subject_ids) if isinstance(block, _FastSqueezeformerBlock) else block(z)
        return self.head(z.flatten(1))


class VirtualSensorFormer(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, sensors=16, patch=20, patch_stride=10,
                 d_model=96, depth=4, heads=4, drop=0.1, domain_bn_boundary=0):
        super().__init__()
        self.spatial = nn.Conv1d(channels_num, sensors, 1)
        self.norm = _bn(sensors, domain_bn_boundary, nn.BatchNorm1d)
        self.patch, self.patch_stride = patch, patch_stride
        n_patch = (eeg_sample_points - patch) // patch_stride + 1
        self.embed = nn.Sequential(nn.Linear(patch, d_model), nn.LayerNorm(d_model))
        self.sensor_pos = nn.Parameter(torch.zeros(1, sensors, 1, d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, 1, n_patch, d_model))
        nn.init.trunc_normal_(self.sensor_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(d_model, heads, drop) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = _bottleneck_head(d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        v = _run(self.norm, self.spatial(x), subject_ids)                        # (B, K, T)
        p = v.unfold(-1, self.patch, self.patch_stride)                          # (B, K, P, patch)
        z = (self.embed(p) + self.sensor_pos + self.time_pos).flatten(1, 2)      # (B, K*P, d)
        for block in self.blocks:
            z = block(z)
        return self.head(self.out_norm(z).mean(1))


class DynSpatialTSConv(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, filters=40, queries=8, d_key=32,
                 drop=0.5, domain_bn_boundary=0):
        super().__init__()
        self.stem = _Stem(channels_num, filters, boundary=domain_bn_boundary, spatial=False)
        t = self.stem.n_time(eeg_sample_points)
        self.static = nn.Conv2d(filters, filters, (channels_num, 1))
        self.electrode = nn.Parameter(torch.zeros(1, channels_num, filters))
        nn.init.trunc_normal_(self.electrode, std=0.02)
        self.key = nn.Linear(filters, d_key)
        self.query = nn.Parameter(torch.randn(queries, d_key) / d_key ** 0.5)
        self.mix = nn.Conv1d(queries * filters, filters, 1)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)  # starts as TSConv
        self.post = nn.ModuleList([_bn(filters, domain_bn_boundary, nn.BatchNorm1d), nn.ELU(), nn.Dropout(drop),
                                   nn.Conv1d(filters, filters, 1)])
        self.head = _bottleneck_head(filters * t, feature_dim)

    def forward(self, x, subject_ids=None):
        z = self.stem(x, subject_ids)                                            # (B, F, C, T')
        k = self.key(z.mean(-1).transpose(1, 2) + self.electrode)                # (B, C, d_key)
        a = (self.query @ k.transpose(1, 2) / k.shape[-1] ** 0.5).softmax(-1)    # (B, Q, C): trial-adaptive weights
        dyn = torch.einsum('bqc,bfct->bqft', a, z).flatten(1, 2)                 # (B, Q*F, T')
        y = self.static(z).squeeze(2) + self.mix(dyn)
        for layer in self.post:
            y = _run(layer, y, subject_ids)
        return self.head(y.flatten(1))


class DualPathTSConv(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, filters=40, fast_filters=24,
                 fast_kernel=9, fast_pool=5, fast_stride=2, domain_bn_boundary=0):
        super().__init__()
        self.slow = _Stem(channels_num, filters, boundary=domain_bn_boundary)
        self.fast = _Stem(channels_num, fast_filters, fast_kernel, fast_pool, fast_stride, boundary=domain_bn_boundary)
        self.slow_proj = nn.Conv1d(filters, filters, 1)
        self.fast_proj = nn.Conv1d(fast_filters, fast_filters, 1)
        n = filters * self.slow.n_time(eeg_sample_points) + fast_filters * self.fast.n_time(eeg_sample_points)
        self.head = _bottleneck_head(n, feature_dim)

    def forward(self, x, subject_ids=None):
        return self.head(torch.cat([self.slow_proj(self.slow(x, subject_ids)).flatten(1),
                                    self.fast_proj(self.fast(x, subject_ids)).flatten(1)], 1))


class TSConvPlus(nn.Module):
    """TSConv with one knob per axis (H1): learned_pool replaces the fixed 25-sample box by a depthwise strided conv
    (box-initialised = TSConv at init); spatial_mult spatial filters per temporal filter (grouped); post = 'sepconv'
    adds a residual depthwise-temporal + pointwise conv after the spatial conv (EEGNet's block 2); readout = 'query'
    replaces the flatten by `queries` learned queries cross-attending to the time steps; readout = 'moe': `experts`
    flatten readouts mixed by a per-trial softmax gate on the flattened features (zero-init = their average)."""

    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, filters=40, kernel=25, pool=25, stride=2,
                 learned_pool=False, pool_type='avg', temporal_depth=1,
                 spatial_mult=1, post='none', post_kernel=9, readout='flatten', queries=8, d_model=64, heads=4, experts=4,
                 drop=0.5, domain_bn_boundary=0):
        super().__init__()
        b = domain_bn_boundary
        self.learned_pool = learned_pool or pool_type == 'max' or temporal_depth > 1  # unfused path
        if pool_type == 'max' or temporal_depth > 1:  # H6: max pooling / a stacked temporal conv before the pool
            convs = [nn.Conv2d(1, filters, (1, kernel))]
            for _ in range(temporal_depth - 1):
                convs += [nn.ELU(), nn.Conv2d(filters, filters, (1, kernel), groups=filters)]
            self.temporal = nn.Sequential(*convs)
            self.pool = nn.MaxPool2d((1, pool), (1, stride)) if pool_type == 'max' else nn.AvgPool2d((1, pool), (1, stride))
            t = (eeg_sample_points - temporal_depth * (kernel - 1) - pool) // stride + 1
        elif learned_pool:
            self.temporal = nn.Conv2d(1, filters, (1, kernel))
            self.pool = nn.Conv2d(filters, filters, (1, pool), (1, stride), groups=filters, bias=False)
            nn.init.constant_(self.pool.weight, 1 / pool)
            t = (eeg_sample_points - kernel + 1 - pool) // stride + 1
        else:
            self.temporal = _FusedTemporalConvPool(filters, kernel, pool, stride)
            t = (eeg_sample_points - self.temporal.effective_kernel) // stride + 1
        m = filters * spatial_mult
        self.layers = nn.ModuleList([_bn(filters, b), nn.ELU(), nn.Conv2d(filters, m, (channels_num, 1), groups=filters if spatial_mult > 1 else 1),
                                     _bn(m, b), nn.ELU(), nn.Dropout(drop)])
        self.post = nn.Sequential(nn.Conv1d(m, m, post_kernel, padding=post_kernel // 2, groups=m), nn.Conv1d(m, m, 1),
                                  nn.BatchNorm1d(m), nn.ELU(), nn.Dropout(drop)) if post == 'sepconv' else None
        self.proj = nn.Conv1d(m, m, 1)
        self.readout = readout
        if readout == 'query':
            self.embed = nn.Sequential(nn.Linear(m, d_model), nn.LayerNorm(d_model))
            self.time_pos = nn.Parameter(torch.zeros(1, t, d_model)); nn.init.trunc_normal_(self.time_pos, std=0.02)
            self.q = nn.Parameter(torch.randn(1, queries, d_model) * 0.02)
            self.xattn = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
            self.head = _bottleneck_head(queries * d_model, feature_dim)
        elif readout == 'moe':  # SUBJECT_LOG D4: the per-person headroom is in the readout -> K readouts, trial-gated
            self.heads = nn.ModuleList(_bottleneck_head(m * t, feature_dim) for _ in range(experts))
            self.gate = nn.Linear(m * t, experts)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)  # uniform mixture at init
        else:
            self.head = _bottleneck_head(m * t, feature_dim)

    def forward(self, x, subject_ids=None):
        z = self.temporal(x.unsqueeze(1))
        if self.learned_pool:
            z = self.pool(z)
        for layer in self.layers:
            z = _run(layer, z, subject_ids)
        z = z.squeeze(2)                                                          # (B, M, T')
        if self.post is not None:
            z = z + self.post(z)
        z = self.proj(z)
        if self.readout == 'query':
            k = self.embed(z.transpose(1, 2)) + self.time_pos
            return self.head(self.xattn(self.q.expand(len(k), -1, -1), k, k, need_weights=False)[0].flatten(1))
        if self.readout == 'moe':
            f = z.flatten(1); g = torch.softmax(self.gate(f), -1)                   # (B, K) per-trial expert weights
            return torch.einsum('bk,bkd->bd', g, torch.stack([h(f) for h in self.heads], 1))
        return self.head(z.flatten(1))


class TimeFreqXTSConv(nn.Module):
    """Content from time, context from frequency (ARCH_LOG H5): TSConv's stem gives 36 time tokens (the evoked
    content); the trial's own log-amplitude spectrum gives one token per frequency bin (1-30 Hz at 1 Hz, the
    32-electrode topography of that bin = recording / background context, cf. the fingerprint router). One
    self-attention block on the frequency tokens, then the time tokens cross-attend to them through a zero-initialised
    gate (= TSConv with a linear token embedding at init), then TSConv's time-locked flatten and bottleneck head."""

    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, filters=40, d_model=64, heads=4,
                 max_hz=30, drop=0.1, domain_bn_boundary=0):
        super().__init__()
        self.stem = _Stem(channels_num, filters, boundary=domain_bn_boundary)
        t = self.stem.n_time(eeg_sample_points)
        self.embed = nn.Sequential(nn.Linear(filters, d_model), nn.LayerNorm(d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, t, d_model)); nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.bins = slice(1, max_hz + 1)  # rfft bins of a 120-sample trial at 120 Hz = 1 Hz each
        self.freq_embed = nn.Sequential(nn.LayerNorm(channels_num), nn.Linear(channels_num, d_model), nn.LayerNorm(d_model))
        self.freq_pos = nn.Parameter(torch.zeros(1, max_hz, d_model)); nn.init.trunc_normal_(self.freq_pos, std=0.02)
        self.freq_block = _Block(d_model, heads, drop)
        self.q_norm, self.kv_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.xattn = nn.MultiheadAttention(d_model, heads, dropout=drop, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(d_model))
        self.head = _bottleneck_head(t * d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        z = self.embed(self.stem(x, subject_ids).transpose(1, 2)) + self.time_pos             # (B, T', d)
        spec = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs()[..., self.bins]   # (B, C, F)
        f = self.freq_block(self.freq_embed(torch.log(spec + 1e-6).transpose(1, 2)) + self.freq_pos)  # (B, F, d)
        kv = self.kv_norm(f)
        z = z + torch.tanh(self.gate) * self.xattn(self.q_norm(z), kv, kv, need_weights=False)[0]
        return self.head(z.flatten(1))


class TokenREVE(nn.Module):
    """Scratch REVE (br3_things: REVE blocks = RMSNorm, no biases, GEGLU, no final norm; learned positions; per-electrode
    trial z-score at 120 Hz) with a switchable tokenizer (ARCH_LOG R-series):
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
        from ensemble_experiments.architectures.alljoined_encoders import _SwitchBlock
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


class SqfMix(nn.Module):
    """Squeezeformer (sqf_dbn: TSConv stem -> 36 time tokens -> 3 Squeezeformer blocks -> attention-stat pooling) plus
    rmix's winning ingredient (ARCH_LOG R-series round 8): 32 REVE-style electrode tokens (each electrode's whole
    z-scored trace, Linear 120 -> d + electrode position) in the same sequence. Attention and FFN act on all 68 tokens;
    the conformer conv module only on the time tokens (electrode order is not a sequence). Pooling over all tokens."""

    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, domain_bn_boundary=0, **kw):
        super().__init__()
        from ensemble_experiments.architectures.ortho_encoders import FastTSConvSqueezeformer
        kw = dict(dict(d_model=160, depth=3, pool_kernel=25, pool_stride=2, backbone_dim=512), **kw)
        self.sqf = FastTSConvSqueezeformer(feature_dim, eeg_sample_points, channels_num,
                                           domain_bn_boundary=domain_bn_boundary, **kw)
        d = kw['d_model']
        self.e_embed = nn.Sequential(nn.Linear(eeg_sample_points, d), nn.LayerNorm(d))
        self.e_pos = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, channels_num, d)), std=0.02)
        self.n_e = channels_num

    def forward(self, x, subject_ids=None):
        from module.eeg_encoder.model import DomainBatchNorm2d as DBN
        q = self.sqf
        t = q.temporal(x.unsqueeze(1))
        for layer in q.stem_after_pool:
            t = layer(t, subject_ids) if isinstance(layer, DBN) else layer(t)
        t = t.squeeze(2).transpose(1, 2)
        t = q.token_projection(t) + q.position[:, :t.shape[1]]
        zx = ((x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)).clamp(-15, 15)
        h = torch.cat([self.e_embed(zx) + self.e_pos, t], 1)
        for b in q.blocks:
            h = h + b.attn(b.attn_norm(h))
            h = torch.cat([h[:, :self.n_e], h[:, self.n_e:] + b.conv(h[:, self.n_e:], subject_ids)], 1)
            if b.ffn is not None:
                h = h + b.ffn(h)
            h = b.final_norm(h)
        w = q.pool_score(h).softmax(dim=1)
        mean = (w * h).sum(dim=1)
        var = (w * (h - mean.unsqueeze(1)).square()).sum(dim=1)
        return q.head(torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=-1))


HYBRIDS = {c.__name__: c for c in (FilterAttnTSConv, TimeAttnFlatTSConv, VirtualSensorFormer, DynSpatialTSConv,
                                   DualPathTSConv, TSConvPlus, TimeFreqXTSConv, TokenREVE, SqfMix)}

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
    for name, cls in HYBRIDS.items():
        if name in ('TokenREVE', 'SqfMix'):
            continue
        m = cls(domain_bn_boundary=21)
        out = m(x, sid)
        assert out.shape == (6, 1536), name
        out.sum().backward()
        print(name, f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
    m = TimeFreqXTSConv(domain_bn_boundary=21); assert m(x, sid).shape == (6, 1536)
    print('TimeFreqXTSConv', f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
    for kw in ({'block': 'gated'}, {'block': 'diff'}, {'block': 'sqf'}, {'depth': 1, 'd_model': 64}):
        m = TimeAttnFlatTSConv(domain_bn_boundary=21, **kw)
        assert m(x, sid).shape == (6, 1536), kw
        print('TimeAttnFlatTSConv', kw, f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
    for kw in ({'learned_pool': True}, {'spatial_mult': 2}, {'post': 'sepconv'}, {'readout': 'query'},
               {'kernel': 49, 'pool': 37, 'stride': 3}, {'filters': 80}, {'pool_type': 'max'}, {'readout': 'moe'}, {'temporal_depth': 2, 'kernel': 13}, {'learned_pool': True, 'kernel': 13, 'pool': 49, 'stride': 4}):
        m = TSConvPlus(domain_bn_boundary=21, **kw)
        assert m(x, sid).shape == (6, 1536), kw
        print('TSConvPlus', kw, f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
    a, bx = TSConvPlus(learned_pool=True).eval(), TSConvPlus().eval()  # box-initialised learned pool == fused stem
    bx.temporal.weight.data.copy_(a.temporal.weight.data); bx.temporal.bias.data.copy_(a.temporal.bias.data)
    assert torch.allclose(a.pool(a.temporal(x.unsqueeze(1))), bx.temporal(x.unsqueeze(1)), atol=1e-4)
