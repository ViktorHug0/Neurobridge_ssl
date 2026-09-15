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
import torch.nn.functional as F

from module.eeg_encoder.model import ResidualAdd


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
    def __init__(self, d_model, kernel_size, drop):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)
        z = nn.functional.glu(self.pointwise_in(z), dim=1)
        z = self.depthwise(z)
        z = nn.functional.silu(self.bn(z))
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
    def __init__(self, d_model, heads, conv_kernel, expansion, drop):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _FastSelfAttention(d_model, heads, drop)
        self.conv = _ConformerConvModule(d_model, conv_kernel, drop)
        self.ffn = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x)
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
                 expansion=2, drop=0.15, stem_filters=40):
        super().__init__()
        self.temporal = _FusedTemporalConvPool(
            out_channels=stem_filters,
            temporal_kernel=25,
            pool_kernel=51,
            pool_stride=5,
        )
        max_tokens = (eeg_sample_points - self.temporal.effective_kernel) // 5 + 1
        if max_tokens <= 0:
            raise ValueError("EEG window is too short for the fused TSConv stem")
        self.stem_after_pool = nn.Sequential(
            nn.BatchNorm2d(stem_filters),
            nn.ELU(),
            nn.Conv2d(stem_filters, stem_filters, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(stem_filters),
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
            )
            for _ in range(depth)
        ])
        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.temporal(x.unsqueeze(1))
        z = self.stem_after_pool(z).squeeze(2).transpose(1, 2)
        z = self.token_projection(z) + self.position[:, :z.shape[1]]
        for block in self.blocks:
            z = block(z)

        weights = self.pool_score(z).softmax(dim=1)
        mean = (weights * z).sum(dim=1)
        variance = (weights * (z - mean.unsqueeze(1)).square()).sum(dim=1)
        pooled = torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)
        return self.head(pooled)


class TinySqueezeformer(FastTSConvSqueezeformer):
    """Compute-matched Squeezeformer proxy for the Tiny encoder family.

    It retains the full family's sensor stem, temporal-token attention,
    depthwise convolution, feed-forward path and attentive statistics pooling.
    Only width and depth are reduced: 12 sensor filters, 64-wide tokens and two
    blocks.  At a 128-dimensional backbone this has roughly the same parameter
    count as TinyTSConv, so differences primarily reflect inductive bias.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=64,
            depth=2,
            heads=4,
            conv_kernel=15,
            expansion=2,
            drop=0.15,
            stem_filters=12,
        )


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


class TinyMultiScaleTSMixer(MultiScaleTSMixer):
    """Parameter-matched proxy retaining the full Mixer's inductive biases."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            tokens=24,
            depth=2,
            filters_per_scale=12,
            token_hidden=48,
            expansion=2,
            drop=0.15,
        )


class _GatedSDPAttention(nn.Module):
    """QK-normalized SDPA with a query-dependent gate for every head."""

    def __init__(self, d_model, heads=4, drop=0.2):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.drop = drop
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_drop = nn.Dropout(drop)

    def _heads(self, x, projection):
        batch, tokens, _ = x.shape
        return projection(x).reshape(
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, queries, context=None):
        context = queries if context is None else context
        q = self._heads(queries, self.q_proj)
        k = self._heads(context, self.k_proj)
        v = self._heads(context, self.v_proj)
        # Unit-norm Q/K plus sqrt(d) restores a useful cosine-logit scale while
        # preventing trial-amplitude outliers from controlling attention.
        scale = math.sqrt(self.head_dim)
        q = nn.functional.normalize(q, dim=-1) * scale
        k = nn.functional.normalize(k, dim=-1) * scale
        attended = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop if self.training else 0.0,
        )
        gates = torch.sigmoid(self.gate_proj(queries)).transpose(1, 2).unsqueeze(-1)
        attended = attended * gates
        attended = attended.transpose(1, 2).flatten(2)
        return self.out_drop(self.out_proj(attended))


class _SwiGLU(nn.Module):
    def __init__(self, d_model, expansion=2, drop=0.2):
        super().__init__()
        hidden = expansion * d_model
        self.in_proj = nn.Linear(d_model, 2 * hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.drop(self.out_proj(value * nn.functional.silu(gate)))


class _GatedAttentionBlock(nn.Module):
    """Shallow pre-norm modern Transformer block."""

    def __init__(self, d_model, heads=4, expansion=2, drop=0.2):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = _GatedSDPAttention(d_model, heads=heads, drop=drop)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = _SwiGLU(d_model, expansion=expansion, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class _GatedQueryPool(nn.Module):
    """Learned queries read a variable-size token set without convolution."""

    def __init__(self, d_model, heads=4, queries=4, expansion=2, drop=0.2):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norm = nn.RMSNorm(d_model)
        self.context_norm = nn.RMSNorm(d_model)
        self.attn = _GatedSDPAttention(d_model, heads=heads, drop=drop)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = _SwiGLU(d_model, expansion=expansion, drop=drop)
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, context):
        queries = self.queries.expand(context.shape[0], -1, -1)
        queries = queries + self.attn(
            self.query_norm(queries), self.context_norm(context)
        )
        queries = queries + self.ffn(self.ffn_norm(queries))
        return self.final_norm(queries)


class TinyGatedChannelTransformer(nn.Module):
    """Arm A: pure channel-token attention with an attention-native readout."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=64, depth=2, heads=4, drop=0.2, summary_queries=4):
        super().__init__()
        self.channel_projection = nn.Linear(eeg_sample_points, d_model)
        self.electrode_identity = nn.Parameter(torch.empty(1, channels_num, d_model))
        nn.init.trunc_normal_(self.electrode_identity, std=0.02)
        self.token_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            _GatedAttentionBlock(d_model, heads=heads, drop=drop)
            for _ in range(depth)
        ])
        self.pool = _GatedQueryPool(
            d_model, heads=heads, queries=summary_queries, drop=drop
        )
        self.head = _head(summary_queries * d_model, feature_dim)

    def forward(self, x):
        tokens = self.token_drop(
            self.channel_projection(x) + self.electrode_identity
        )
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.pool(tokens).flatten(1))


class _DifferentialGatedSDPAttention(nn.Module):
    """Differential attention with a learned per-head subtraction strength."""

    def __init__(self, d_model, heads=4, drop=0.1):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q1 = nn.Linear(d_model, d_model, bias=False)
        self.q2 = nn.Linear(d_model, d_model, bias=False)
        self.k1 = nn.Linear(d_model, d_model, bias=False)
        self.k2 = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, heads)
        self.lambda_logit = nn.Parameter(torch.zeros(heads))
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_drop = nn.Dropout(drop)

    def _heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def _qk(self, projection, x):
        return F.normalize(self._heads(projection(x)), dim=-1) * self.head_dim**0.5

    def forward(self, x):
        q1, q2 = self._qk(self.q1, x), self._qk(self.q2, x)
        k1, k2 = self._qk(self.k1, x), self._qk(self.k2, x)
        value = self._heads(self.v(x))
        dropout_p = self.out_drop.p if self.training else 0.0
        first = F.scaled_dot_product_attention(q1, k1, value, dropout_p=dropout_p)
        second = F.scaled_dot_product_attention(q2, k2, value, dropout_p=dropout_p)
        subtraction = torch.sigmoid(self.lambda_logit).view(1, self.heads, 1, 1)
        output = first - subtraction * second

        query_gate = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
        output = output * query_gate
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.out_drop(self.out_proj(output))


class _DifferentialGatedAttentionBlock(nn.Module):
    def __init__(self, d_model, heads=4, drop=0.1, expansion=2):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = _DifferentialGatedSDPAttention(d_model, heads=heads, drop=drop)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = _SwiGLU(d_model, expansion=expansion, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TinyDifferentialChannelTransformer(TinyGatedChannelTransformer):
    """Channel-only model using differential rather than ordinary attention."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        d_model = 60
        depth = 2
        heads = 4
        drop = 0.2
        super().__init__(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=d_model,
            depth=depth,
            heads=heads,
            drop=drop,
            summary_queries=4,
        )
        self.blocks = nn.ModuleList(
            [_DifferentialGatedAttentionBlock(d_model, heads=heads, drop=drop) for _ in range(depth)]
        )


_THINGS_EEG_2_POSITIONS = torch.tensor([
    (-0.0294367,  0.0839171, -0.0069900),  # Fp1
    ( 0.0298723,  0.0848959, -0.0070800),  # Fp2
    (-0.0548397,  0.0685722, -0.0105900),  # AF7
    (-0.0337007,  0.0768371,  0.0212270),  # AF3
    ( 0.0002313,  0.0807710,  0.0354170),  # AFz
    ( 0.0357123,  0.0777259,  0.0219560),  # AF4
    ( 0.0557433,  0.0696568, -0.0107550),  # AF8
    (-0.0702629,  0.0424743, -0.0114200),  # F7
    (-0.0644658,  0.0480353,  0.0169210),  # F5
    (-0.0502438,  0.0531112,  0.0421920),  # F3
    (-0.0274958,  0.0569311,  0.0603420),  # F1
    ( 0.0295142,  0.0576019,  0.0595400),  # F2
    ( 0.0518362,  0.0543048,  0.0408140),  # F4
    ( 0.0679142,  0.0498297,  0.0163670),  # F6
    ( 0.0730431,  0.0444217, -0.0120000),  # F8
    (-0.0840759,  0.0145673, -0.0504290),  # FT9
    (-0.0807750,  0.0141203, -0.0111350),  # FT7
    (-0.0772149,  0.0186433,  0.0244600),  # FC5
    (-0.0601819,  0.0227162,  0.0555440),  # FC3
    (-0.0340619,  0.0260111,  0.0799870),  # FC1
    ( 0.0003761,  0.0273900,  0.0886680),  # FCz
    ( 0.0347841,  0.0264379,  0.0788080),  # FC2
    ( 0.0622931,  0.0237228,  0.0556300),  # FC4
    ( 0.0795341,  0.0199357,  0.0244380),  # FC6
    ( 0.0818151,  0.0154167, -0.0113300),  # FT8
    ( 0.0841131,  0.0143647, -0.0505380),  # FT10
    (-0.0841611, -0.0160187, -0.0093460),  # T7
    (-0.0802801, -0.0137597,  0.0291600),  # C5
    (-0.0653581, -0.0116317,  0.0643580),  # C3
    (-0.0361580, -0.0099839,  0.0897520),  # C1
    ( 0.0004009, -0.0091670,  0.1002440),  # Cz
    ( 0.0376720, -0.0096241,  0.0884120),  # C2
    ( 0.0671179, -0.0109003,  0.0635800),  # C4
    ( 0.0834559, -0.0127763,  0.0292080),  # C6
    ( 0.0850799, -0.0150203, -0.0094900),  # T8
    (-0.0856192, -0.0465147, -0.0457070),  # TP9
    (-0.0848302, -0.0460217, -0.0070560),  # TP7
    (-0.0795922, -0.0465507,  0.0309490),  # CP5
    (-0.0635562, -0.0470088,  0.0656240),  # CP3
    (-0.0355131, -0.0472919,  0.0913150),  # CP1
    ( 0.0003858, -0.0473180,  0.0994320),  # CPz
    ( 0.0383838, -0.0470731,  0.0906950),  # CP2
    ( 0.0666118, -0.0466372,  0.0655800),  # CP4
    ( 0.0833218, -0.0461013,  0.0312060),  # CP6
    ( 0.0855488, -0.0455453, -0.0071300),  # TP8
    ( 0.0861618, -0.0470353, -0.0458690),  # TP10
    (-0.0724343, -0.0734527, -0.0024870),  # P7
    (-0.0672723, -0.0762907,  0.0283820),  # P5
    (-0.0530073, -0.0787878,  0.0559400),  # P3
    (-0.0286203, -0.0805249,  0.0754360),  # P1
    ( 0.0003247, -0.0811150,  0.0826150),  # Pz
    ( 0.0319197, -0.0804871,  0.0767160),  # P2
    ( 0.0556667, -0.0785602,  0.0565610),  # P4
    ( 0.0678877, -0.0759043,  0.0280910),  # P6
    ( 0.0730557, -0.0730683, -0.0025400),  # P8
    (-0.0548404, -0.0975279,  0.0027920),  # PO7
    (-0.0365114, -0.1008529,  0.0371670),  # PO3
    ( 0.0002156, -0.1021780,  0.0506080),  # POz
    ( 0.0367816, -0.1008491,  0.0363970),  # PO4
    ( 0.0556666, -0.0976251,  0.0027300),  # PO8
    (-0.0294134, -0.1124490,  0.0088390),  # O1
    ( 0.0001076, -0.1148920,  0.0146570),  # Oz
    ( 0.0298426, -0.1121560,  0.0088000),  # O2
], dtype=torch.float32)


def _montage_knn_graph(positions, neighbors=6):
    """Symmetric distance-weighted graph with self loops."""
    distance = torch.cdist(positions, positions)
    nearest = distance.topk(neighbors + 1, largest=False).indices[:, 1:]
    mask = torch.zeros_like(distance)
    mask.scatter_(1, nearest, 1.0)
    mask = torch.maximum(mask, mask.T)
    mask.fill_diagonal_(1.0)
    nonself = mask.bool() & ~torch.eye(len(positions), dtype=torch.bool)
    scale = distance[nonself].median().clamp_min(1e-6)
    weight = torch.exp(-distance.square() / (2.0 * scale.square())) * mask
    return weight / weight.sum(dim=-1, keepdim=True)


class _GraphDiffusionBlock(nn.Module):
    """Two-hop sparse diffusion with gated node and virtual-node updates."""

    def __init__(self, d_model, drop=0.2):
        super().__init__()
        self.node_norm = nn.RMSNorm(d_model)
        self.global_norm = nn.RMSNorm(d_model)
        self.local_mix = nn.Linear(3 * d_model, 2 * d_model)
        self.global_to_node = nn.Linear(d_model, d_model, bias=False)
        self.update_gate = nn.Linear(2 * d_model, d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = _SwiGLU(d_model, expansion=2, drop=drop)
        self.global_mix = nn.Linear(2 * d_model, 2 * d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, nodes, global_node, adjacency):
        normalized = self.node_norm(nodes)
        first_hop = torch.einsum("ij,bjd->bid", adjacency, normalized)
        second_hop = torch.einsum("ij,bjd->bid", adjacency, first_hop)
        left, right = self.local_mix(
            torch.cat([normalized, first_hop, second_hop], dim=-1)
        ).chunk(2, dim=-1)
        candidate = F.silu(left) * right
        candidate = candidate + self.global_to_node(
            self.global_norm(global_node)
        ).unsqueeze(1)
        gate = torch.sigmoid(
            self.update_gate(torch.cat([normalized, candidate], dim=-1))
        )
        nodes = nodes + self.drop(gate * candidate)
        nodes = nodes + self.ffn(self.ffn_norm(nodes))

        pooled = self.node_norm(nodes).mean(dim=1)
        left, right = self.global_mix(
            torch.cat([self.global_norm(global_node), pooled], dim=-1)
        ).chunk(2, dim=-1)
        global_node = global_node + self.drop(F.silu(left) * right)
        return nodes, global_node


class TinyGraphDiffusionNet(nn.Module):
    """Sparse montage graph diffusion; no convolution or dense self-attention."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=50, depth=3, neighbors=6, drop=0.2):
        super().__init__()
        if channels_num != len(_THINGS_EEG_2_POSITIONS):
            raise ValueError(
                "TinyGraphDiffusionNet requires the canonical 63-channel "
                "THINGS-EEG-2 montage"
            )
        positions = _THINGS_EEG_2_POSITIONS.clone()
        positions = (positions - positions.mean(dim=0)) / positions.std(dim=0)
        self.register_buffer("positions", positions)
        self.register_buffer(
            "base_adjacency", _montage_knn_graph(positions, neighbors=neighbors)
        )
        self.edge_logits = nn.Parameter(torch.zeros(channels_num, channels_num))
        self.trace_projection = nn.Linear(eeg_sample_points, d_model)
        self.position_projection = nn.Sequential(
            nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.global_node = nn.Parameter(torch.empty(1, d_model))
        nn.init.trunc_normal_(self.global_node, std=0.02)
        self.token_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            _GraphDiffusionBlock(d_model, drop=drop) for _ in range(depth)
        ])
        self.node_norm = nn.RMSNorm(d_model)
        self.global_norm = nn.RMSNorm(d_model)
        self.pool_gate = nn.Linear(d_model, 1)
        self.head = _head(4 * d_model, feature_dim)

    def _adjacency(self):
        symmetric_logits = 0.5 * (self.edge_logits + self.edge_logits.T)
        learned_gate = 0.5 + torch.sigmoid(symmetric_logits)
        adjacency = self.base_adjacency * learned_gate
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x):
        position = self.position_projection(self.positions).unsqueeze(0)
        nodes = self.token_drop(self.trace_projection(x) + position)
        global_node = self.global_node.expand(x.shape[0], -1)
        adjacency = self._adjacency()
        for block in self.blocks:
            nodes, global_node = block(nodes, global_node, adjacency)

        nodes = self.node_norm(nodes)
        weights = torch.sigmoid(self.pool_gate(nodes))
        gated = (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        readout = torch.cat([
            self.global_norm(global_node),
            nodes.mean(dim=1),
            nodes.amax(dim=1),
            gated,
        ], dim=-1)
        return self.head(readout)


_REGISTRY = {
    'FastTSConvSqueezeformer': FastTSConvSqueezeformer,
    'TinySqueezeformer': TinySqueezeformer,
    'MultiScaleTSMixer': MultiScaleTSMixer,
    'TinyMultiScaleTSMixer': TinyMultiScaleTSMixer,
    'TinyGatedChannelTransformer': TinyGatedChannelTransformer,
    'TinyDifferentialChannelTransformer': TinyDifferentialChannelTransformer,
    'TinyGraphDiffusionNet': TinyGraphDiffusionNet,
}


def canonical_encoder_name(name):
    """Accept the historical ``Ortho`` prefix for maintained checkpoints only."""
    return name[5:] if name.startswith('Ortho') else name


def is_architecture_encoder(name):
    return canonical_encoder_name(name) in _REGISTRY


def build_architecture_encoder(name, feature_dim, eeg_sample_points, channels_num):
    canonical_name = canonical_encoder_name(name)
    if canonical_name not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture encoder {name!r}; have {sorted(_REGISTRY)}"
        )
    return _REGISTRY[canonical_name](
        feature_dim=feature_dim,
        eeg_sample_points=eeg_sample_points,
        channels_num=channels_num,
    )
