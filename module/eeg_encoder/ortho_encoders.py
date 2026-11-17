"""Experimental EEG encoder implementations and maintained registry.

TSConv  = free-form temporal conv -> avg-pool -> one global spatial filter (1st order, time domain).
ATM     = iTransformer, attention over the 63 channels-as-tokens             (1st order, time-as-feature).

Contract: forward((B, C, T)) -> (B, feature_dim), where feature_dim is
--eeg_backbone_dim. Only the architectures in ``_REGISTRY`` are selectable. The
historical ``Ortho`` prefix is accepted for maintained legacy checkpoints.
"""
import math

import torch
import torch.nn as nn

from module.eeg_encoder.model import DomainBatchNorm2d, ResidualAdd


def _head(in_dim, feature_dim, dropout=0.5):
    """Same projection head as TSConv/EEGNet, so only the *body* differs."""
    return nn.Sequential(
        nn.Linear(in_dim, feature_dim),
        ResidualAdd(nn.Sequential(
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(dropout),
        )),
        nn.LayerNorm(feature_dim),
    )


class _ConformerFeedForward(nn.Module):
    def __init__(self, d_model, expansion, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion * d_model),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class _ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, drop, domain_bn_boundary=0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.bn = (DomainBatchNorm2d(d_model, domain_bn_boundary, nn.BatchNorm1d) if domain_bn_boundary > 0
                   else nn.BatchNorm1d(d_model))
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, subject_ids=None):
        z = self.norm(x).transpose(1, 2)
        z = nn.functional.glu(self.pointwise_in(z), dim=1)
        z = self.depthwise(z)
        z = nn.functional.silu(self.bn(z, subject_ids) if isinstance(self.bn, DomainBatchNorm2d) else self.bn(z))
        z = self.pointwise_out(z).transpose(1, 2)
        return self.drop(z)


class _FusedTemporalConvPool(nn.Module):
    """Exact Conv25 -> AvgPool51 fusion, evaluated as one stride-5 Conv75.

    The effective 75-tap kernel is the convolution of each learned 25-tap
    temporal filter with the fixed 51-sample box.  This preserves the original
    parameterization while avoiding the very large 226-step activation.
    """

    def __init__(self, out_channels=40, temporal_kernel=25, pool_kernel=51,
                 pool_stride=5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 1, 1, temporal_kernel))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.pool_stride = pool_stride
        self.effective_kernel = temporal_kernel + pool_kernel - 1
        self.box_pad = pool_kernel - 1
        self.register_buffer(
            'box',
            torch.full((1, 1, pool_kernel), 1.0 / pool_kernel),
            persistent=False,
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(temporal_kernel)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        temporal_weight = self.weight.squeeze(2)
        effective_weight = nn.functional.conv1d(
            temporal_weight,
            self.box,
            padding=self.box_pad,  # a Python int: torch.jit.trace turns self.box.shape[-1] - 1 into a traced size
        ).unsqueeze(2)
        return nn.functional.conv2d(
            x,
            effective_weight,
            self.bias,
            stride=(1, self.pool_stride),
        )


class _FastSelfAttention(nn.Module):
    """Standard temporal MHSA through PyTorch's fused SDPA kernel."""

    def __init__(self, d_model, heads, drop):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.drop = drop
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))


class _FastSqueezeformerBlock(nn.Module):
    def __init__(self, d_model, heads, conv_kernel, expansion, drop, domain_bn_boundary=0, ffn=True):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _FastSelfAttention(d_model, heads, drop)
        self.conv = _ConformerConvModule(d_model, conv_kernel, drop, domain_bn_boundary)
        self.ffn = _ConformerFeedForward(d_model, expansion, drop) if ffn else None
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, subject_ids=None):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x, subject_ids)
        if self.ffn is not None:
            x = x + self.ffn(x)
        return self.final_norm(x)


class FastTSConvSqueezeformer(nn.Module):
    """Latency-oriented TSConv-Squeezeformer retaining the same core biases.

    It exactly fuses the adjacent temporal-convolution/average-pool operations,
    then uses three compact blocks, a 2x FFN, and fused SDPA. Width, temporal
    token count, convolution module, and output head match the validated design.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=160, depth=3, heads=4, conv_kernel=15,
                 expansion=2, drop=0.15, stem_filters=40,
                 pool_kernel=51, pool_stride=5, backbone_dim=0, domain_bn_boundary=0, ffn=True):
        # domain_bn_boundary > 0: every BatchNorm keeps separate statistics for subject IDs at or above it
        # (THINGS-EEG2 mapped onto Alljoined); unseen subjects take the primary path (cf. TSConv's)
        # pool_kernel/pool_stride: 51/5 is the 250 Hz design (10 tokens at 120 Hz); 25/2 keeps
        # its durations at 120 Hz. backbone_dim > 0: TSConv's trunk (head at backbone_dim,
        # then Linear -> feature_dim). Defaults reproduce every existing checkpoint.
        super().__init__()
        self.temporal = _FusedTemporalConvPool(
            out_channels=stem_filters,
            temporal_kernel=25,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )
        max_tokens = (eeg_sample_points - self.temporal.effective_kernel) // pool_stride + 1
        if max_tokens <= 0:
            raise ValueError("EEG window is too short for the fused TSConv stem")
        bn2d = (lambda c: DomainBatchNorm2d(c, domain_bn_boundary)) if domain_bn_boundary > 0 else nn.BatchNorm2d
        self.stem_after_pool = nn.Sequential(
            bn2d(stem_filters),
            nn.ELU(),
            nn.Conv2d(stem_filters, stem_filters, (channels_num, 1), (1, 1)),
            bn2d(stem_filters),
            nn.ELU(),
            nn.Dropout(drop),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(stem_filters, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            _FastSqueezeformerBlock(
                d_model=d_model,
                heads=heads,
                conv_kernel=conv_kernel,
                expansion=expansion,
                drop=drop,
                domain_bn_boundary=domain_bn_boundary,
                ffn=ffn,
            )
            for _ in range(depth)
        ])
        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        trunk_dim = backbone_dim or feature_dim
        self.head = _head(2 * d_model, trunk_dim)
        if trunk_dim != feature_dim:
            self.head.append(nn.Linear(trunk_dim, feature_dim))

    def forward(self, x, subject_ids=None):
        z = self.temporal(x.unsqueeze(1))
        for layer in self.stem_after_pool:
            z = layer(z, subject_ids) if isinstance(layer, DomainBatchNorm2d) else layer(z)
        z = z.squeeze(2).transpose(1, 2)
        z = self.token_projection(z) + self.position[:, :z.shape[1]]
        for block in self.blocks:
            z = block(z, subject_ids)

        weights = self.pool_score(z).softmax(dim=1)
        mean = (weights * z).sum(dim=1)
        variance = (weights * (z - mean.unsqueeze(1)).square()).sum(dim=1)
        pooled = torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)
        return self.head(pooled)


class _MultiResolutionSpatialStem(nn.Module):
    """Phase-preserving temporal scales, each immediately mixed over the scalp."""

    def __init__(self, channels_num, filters_per_scale=16, kernels=(9, 25, 49),
                 output_tokens=36, drop=0.15):
        super().__init__()
        self.output_tokens = output_tokens
        self.branches = nn.ModuleList()
        for kernel in kernels:
            if kernel % 2 == 0:
                raise ValueError("multi-resolution kernels must be odd")
            self.branches.append(nn.ModuleDict({
                # Exact Conv -> AvgPool fusion avoids retaining the large
                # full-resolution per-electrode activation for backprop.
                'temporal': _FusedTemporalConvPool(
                    out_channels=filters_per_scale,
                    temporal_kernel=kernel,
                    pool_kernel=51,
                    pool_stride=5,
                ),
                'spatial': nn.Sequential(
                    nn.BatchNorm2d(filters_per_scale),
                    nn.ELU(),
                    # Every temporal filter gets its own whole-scalp projection.
                    nn.Conv2d(filters_per_scale, filters_per_scale,
                              (channels_num, 1), groups=filters_per_scale, bias=False),
                    nn.BatchNorm2d(filters_per_scale),
                    nn.ELU(),
                    nn.Dropout(drop),
                ),
            }))
        self.output_dim = filters_per_scale * len(kernels)

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            z = branch['temporal'](x.unsqueeze(1))
            z = branch['spatial'](z).squeeze(2)               # B, Dscale, T'
            z = nn.functional.adaptive_avg_pool1d(z, self.output_tokens)
            outputs.append(z)
        return torch.cat(outputs, dim=1).transpose(1, 2)      # B, T', D


class _MLPMixerSequenceBlock(nn.Module):
    """Attention-free global temporal mixing plus a gated feature MLP."""

    def __init__(self, tokens, d_model, token_hidden=72, expansion=3, drop=0.15):
        super().__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mixer = nn.Sequential(
            nn.Linear(tokens, token_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(token_hidden, tokens),
            nn.Dropout(drop),
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_in = nn.Linear(d_model, 2 * expansion * d_model)
        self.feature_out = nn.Linear(expansion * d_model, d_model)
        self.feature_drop = nn.Dropout(drop)

    def forward(self, x):
        z = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mixer(z).transpose(1, 2)
        a, gate = self.feature_in(self.feature_norm(x)).chunk(2, dim=-1)
        z = self.feature_out(nn.functional.gelu(a) * torch.sigmoid(gate))
        return x + self.feature_drop(z)


class MultiScaleTSMixer(nn.Module):
    """Multi-resolution ERP filters + whole-scalp projections + MLP-Mixer.

    Unlike frequency-power models, all branches retain waveform sign and timing.
    Unlike the failed electrode-preserving candidates, each branch integrates all
    sensors before any low-dimensional token processing.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 tokens=36, depth=3, filters_per_scale=16, token_hidden=72,
                 expansion=3, drop=0.15):
        super().__init__()
        self.stem = _MultiResolutionSpatialStem(
            channels_num, filters_per_scale=filters_per_scale, kernels=(9, 25, 49),
            output_tokens=tokens, drop=drop,
        )
        d_model = self.stem.output_dim
        self.position = nn.Parameter(torch.zeros(1, tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            _MLPMixerSequenceBlock(tokens, d_model, token_hidden=token_hidden,
                                   expansion=expansion, drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Flattening deliberately retains absolute ERP latency.
        self.head = _head(tokens * d_model, feature_dim)

    def forward(self, x):
        z = self.stem(x) + self.position
        for block in self.blocks:
            z = block(z)
        return self.head(self.norm(z).flatten(1))


def _mlp(dim, hidden, drop):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim), nn.Dropout(drop))


class _CellBatchNorm(nn.Module):
    """BatchNorm of a (B, C, T, d) grid with one channel per (electrode, feature), statistics over batch and time
    (the analogue of TSConv's per-filter BatchNorm)."""

    def __init__(self, channels, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels * d)

    def forward(self, z):
        B, C, T, d = z.shape
        return self.bn(z.transpose(2, 3).reshape(B, C * d, T)).reshape(B, C, d, T).transpose(2, 3)


def _cell_norm(norm, channels, d):
    return {'ln': lambda: nn.LayerNorm(d), 'bn': lambda: _CellBatchNorm(channels, d), 'none': nn.Identity}[norm]()


class _Mixer3Block(nn.Module):
    """Pre-norm residual dense MLPs along time, electrodes, then features of a (B, C, T, d) grid."""

    def __init__(self, channels, times, d, time_hidden, elec_hidden, feat_hidden, drop, norm='ln'):
        super().__init__()
        self.norms = nn.ModuleList([_cell_norm(norm, channels, d) for _ in range(3)])
        self.time = _mlp(times, time_hidden, drop)
        self.elec = _mlp(channels, elec_hidden, drop)
        self.feat = _mlp(d, feat_hidden, drop)

    def forward(self, z):
        z = z + self.time(self.norms[0](z).transpose(2, 3)).transpose(2, 3)
        z = z + self.elec(self.norms[1](z).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return z + self.feat(self.norms[2](z))


class Mixer3(nn.Module):
    """MLP-only encoder on the raw (electrode x time) grid (NIGHT_LOG N65): each cell is lifted to d features, then
    blocks alternate dense MLPs over all time samples (position-specific: absolute latency, no shift invariance),
    over all electrodes and over features. No convolution, no patches; the flatten head keeps latency."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63, d=8, depth=4,
                 time_hidden=128, elec_hidden=64, feat_hidden=32, drop=0.3, domain_bn_boundary=0,
                 time_attn=0, spectral=False, time_tokens=0, norm='ln', band=''):
        # time_attn > 0: one single-head attention over the time axis after block 2, query/key dim time_attn, from
        # each position's whole (electrode x feature) slice, values = the grid itself (content-dependent re-timing)
        # spectral: mix the orthonormal rFFT of each electrode instead of its samples (cells = (Re, Im) per bin)
        # norm 'bn': BatchNorm per (electrode, feature) over batch and time inside the blocks instead of LayerNorm
        # over features (N65: the conv members' error family may come from their in-network BatchNorm); 'none': neither
        # time_tokens > 0: a learned dense time map (shared across electrodes) compresses the samples to this many
        # steps first: a smooth low-dimensional time basis (the mixers overfit: train loss 6.03 vs TSC 6.48, N65)
        # band 'lp': TSConv's 25-sample/stride-2 average pool on the input (its ~5 Hz smoothing, 48 steps); 'hp': the input
        # minus its 25-sample moving average (N65: do the two error families differ by the frequency band they read?)
        super().__init__()
        self.spectral, self.band = spectral, band
        if band == 'lp':
            eeg_sample_points = (eeg_sample_points - 25) // 2 + 1
        self.tcomp = nn.Linear(eeg_sample_points, time_tokens) if time_tokens else None
        n = eeg_sample_points // 2 + 1 if spectral else time_tokens or eeg_sample_points
        self.bn = (DomainBatchNorm2d(channels_num, domain_bn_boundary, nn.BatchNorm1d) if domain_bn_boundary > 0
                   else nn.BatchNorm1d(channels_num))
        self.lift = nn.Linear(2 if spectral else 1, d)
        self.blocks = nn.ModuleList([
            _Mixer3Block(channels_num, n, d, time_hidden, elec_hidden, feat_hidden, drop, norm)
            for _ in range(depth)
        ])
        if time_attn:
            self.attn_norm = nn.LayerNorm(channels_num * d)
            self.qk = nn.Linear(channels_num * d, 2 * time_attn)
            self.qk_pos = nn.Parameter(torch.zeros(n, 2 * time_attn))
            self.attn_out = nn.Linear(d, d)
        self.time_attn = time_attn
        self.norm = _cell_norm(norm, channels_num, d)
        self.head = _head(channels_num * n * d, feature_dim)

    def forward(self, x, subject_ids=None):
        x = self.bn(x, subject_ids) if isinstance(self.bn, DomainBatchNorm2d) else self.bn(x)
        if self.band == 'lp':
            x = nn.functional.avg_pool1d(x, 25, 2)
        elif self.band == 'hp':
            x = x - nn.functional.avg_pool1d(x, 25, 1, padding=12, count_include_pad=False)
        if self.spectral:  # outside the compiled bf16 trunk: cuFFT has no bf16 for length 120
            f = torch.fft.rfft(x.float(), dim=-1, norm='ortho')
            return self._trunk(torch.stack([f.real, f.imag], -1))
        return self._trunk((self.tcomp(x) if self.tcomp is not None else x).unsqueeze(-1))

    # fuses the memory-bound LayerNorm/GELU/dropout chain over the 32 x 120 x d cells: 0.12 vs 0.49 s/step
    # (bf16, batch 1024, d 8; d 16 uncompiled was 0.83 s = 15 h for 40 epochs)
    @torch.compile
    def _trunk(self, x):
        z = self.lift(x)
        for i, block in enumerate(self.blocks):
            z = block(z)
            if i == 1 and self.time_attn:
                B, C, T, d = z.shape
                q, k = (self.qk(self.attn_norm(z.permute(0, 2, 1, 3).reshape(B, T, C * d))) + self.qk_pos).chunk(2, -1)
                a = (q @ k.transpose(1, 2) / self.time_attn ** .5).softmax(-1)
                z = z + self.attn_out(torch.einsum('bts,bcsf->bctf', a, z))
        return self.head(self.norm(z).flatten(1))


class FlatMLP(nn.Module):
    """EEGProject (one dense map over the flattened grid + a residual MLP, no weight sharing) with Mixer3's input
    BatchNorm and head: the unfactorised ablation of Mixer3 (NIGHT_LOG N65)."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63, width=1024, drop=0.3,
                 domain_bn_boundary=0):
        super().__init__()
        self.bn = (DomainBatchNorm2d(channels_num, domain_bn_boundary, nn.BatchNorm1d) if domain_bn_boundary > 0
                   else nn.BatchNorm1d(channels_num))
        self.body = nn.Sequential(
            nn.Linear(channels_num * eeg_sample_points, width),
            ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.Dropout(drop))),
            nn.LayerNorm(width),
        )
        self.head = _head(width, feature_dim)

    def forward(self, x, subject_ids=None):
        x = self.bn(x, subject_ids) if isinstance(self.bn, DomainBatchNorm2d) else self.bn(x)
        return self.head(self.body(x.flatten(1)))


_REGISTRY = {
    'FastTSConvSqueezeformer': FastTSConvSqueezeformer,
    'MultiScaleTSMixer': MultiScaleTSMixer,
    'Mixer3': Mixer3,
    'FlatMLP': FlatMLP,
}
from module.eeg_encoder.hybrid_encoders import HYBRIDS  # noqa: E402
_REGISTRY.update(HYBRIDS)


def canonical_encoder_name(name):
    """Accept the historical ``Ortho`` prefix for maintained checkpoints only."""
    return name[5:] if name.startswith('Ortho') else name


def is_architecture_encoder(name):
    return canonical_encoder_name(name) in _REGISTRY


def build_architecture_encoder(name, feature_dim, eeg_sample_points, channels_num, **kwargs):
    canonical_name = canonical_encoder_name(name)
    if canonical_name not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture encoder {name!r}; have {sorted(_REGISTRY)}"
        )
    return _REGISTRY[canonical_name](
        feature_dim=feature_dim,
        eeg_sample_points=eeg_sample_points,
        channels_num=channels_num,
        **kwargs,
    )
