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
            padding=self.box.shape[-1] - 1,
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
                 pool_kernel=51, pool_stride=5, backbone_dim=0, power_branch=False, domain_bn_boundary=0, ffn=True):
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
        # TSConv's band-power branch (+0.037 there), made cheap: its temporal filter bank and
        # spatial conv have no nonlinearity between them, so they compose into one 63->filters
        # Conv1d of the same 25 taps -- the same function class (rank <= 25 <= filters) without
        # the filters x channels x time intermediate. Then BN -> square -> pool -> log, flattened
        # into the head next to the attentive statistics.
        self.power_branch = nn.Sequential(
            nn.Conv1d(channels_num, stem_filters, 25, bias=False),
            nn.BatchNorm1d(stem_filters),
        ) if power_branch else None
        self.power_pool = nn.AvgPool1d(pool_kernel, pool_stride)
        self.power_dropout = nn.Dropout(drop)
        power_dim = 0
        if power_branch:
            with torch.no_grad():
                power_dim = self._power_features(torch.zeros(2, channels_num, eeg_sample_points)).shape[1]
        trunk_dim = backbone_dim or feature_dim
        self.head = _head(2 * d_model + power_dim, trunk_dim)
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
        if self.power_branch is not None:
            pooled = torch.cat([pooled, self._power_features(x)], dim=-1)
        return self.head(pooled)

    def _power_features(self, x):
        power = self.power_pool(self.power_branch(x).square())
        return self.power_dropout(power.clamp_min(1e-6).log().flatten(1))


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


_REGISTRY = {
    'FastTSConvSqueezeformer': FastTSConvSqueezeformer,
    'MultiScaleTSMixer': MultiScaleTSMixer,
}
from ensemble_experiments.architectures.hybrid_encoders import HYBRIDS  # noqa: E402
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
