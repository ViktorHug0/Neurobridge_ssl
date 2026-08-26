"""EEG encoders built to be *architecturally orthogonal* to TSConv and ATM.

TSConv  = free-form temporal conv -> avg-pool -> one global spatial filter (1st order, time domain).
ATM     = iTransformer, attention over the 63 channels-as-tokens             (1st order, time-as-feature).

Every encoder here deliberately breaks one of those two assumptions: second-order
statistics, frequency/power domain, complex time-frequency, no-conv-no-attention
mixing, multi-scale dilation, or recurrence.

Contract: forward((B, C, T)) -> (B, feature_dim), where feature_dim is
--eeg_backbone_dim. Selected with --eeg_encoder_type Ortho<Name>; train.py routes
any name starting with "Ortho" to build_ortho_encoder below.
"""
import math

import torch
import torch.nn as nn

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


class OrthoRiemann(nn.Module):
    """Second-order: per-window spatial covariance -> BiMap -> LogEig -> head (SPDNet-lite).

    Sees only channel co-variation, never the waveform. The covariance is left
    *uncentred* on purpose: (1/T)sum x_t x_t^T contains the evoked mean's outer
    product, so the ERP survives the second-order projection.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 n_windows=8, bimap_dim=24):
        super().__init__()
        self.n_windows = n_windows
        self.win = eeg_sample_points // n_windows
        # One learned Stiefel-ish projection per window (init orthonormal).
        w = torch.stack([torch.linalg.qr(torch.randn(channels_num, bimap_dim))[0]
                         for _ in range(n_windows)])
        self.bimap = nn.Parameter(w)
        iu = torch.triu_indices(bimap_dim, bimap_dim)
        self.register_buffer('iu', iu, persistent=False)
        # off-diagonal entries counted once -> weight by sqrt(2) to keep the Frobenius norm
        self.register_buffer('tri_w', torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)),
                             persistent=False)
        self.register_buffer('eye', torch.eye(bimap_dim), persistent=False)
        n_feat = n_windows * iu.shape[1]
        self.norm = nn.LayerNorm(n_feat)
        self.drop = nn.Dropout(0.25)
        self.head = _head(n_feat, feature_dim)

    def forward(self, x):
        B, C, T = x.shape
        x = x[:, :, :self.n_windows * self.win].reshape(B, C, self.n_windows, self.win)
        x = x.permute(0, 2, 1, 3).float()                      # B, W, C, win
        cov = x @ x.transpose(-1, -2) / self.win               # B, W, C, C
        w = self.bimap.unsqueeze(0)                            # 1, W, C, d
        m = w.transpose(-1, -2) @ cov @ w                      # B, W, d, d
        m = 0.5 * (m + m.transpose(-1, -2))
        # ponytail: trace-relative ridge keeps eigh well-conditioned; drop it and
        # near-degenerate eigenvalues blow up the backward pass.
        ridge = 1e-3 * m.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        m = m + ridge[..., None, None] * self.eye
        ev, U = torch.linalg.eigh(m)
        logm = U @ torch.diag_embed(torch.log(ev.clamp_min(1e-8))) @ U.transpose(-1, -2)
        feat = logm[..., self.iu[0], self.iu[1]] * self.tri_w  # B, W, d(d+1)/2
        feat = self.drop(self.norm(feat.flatten(1)))
        return self.head(feat)


class _FixedFilterBank(nn.Module):
    """Hamming-windowed sinc band-pass FIR bank (fixed, not learned)."""

    def __init__(self, bands, kernel_size=65, sample_rate=250.0):
        super().__init__()
        half = kernel_size // 2
        n = torch.arange(-half, half + 1, dtype=torch.float32)
        window = torch.hamming_window(kernel_size, periodic=False)
        taps = []
        for lo, hi in bands:
            # ideal band-pass = difference of two low-pass sincs; t=0 limit is 2f/fs
            h = 2 * hi / sample_rate * torch.sinc(2 * hi * n / sample_rate) \
                - 2 * lo / sample_rate * torch.sinc(2 * lo * n / sample_rate)
            h = h * window
            taps.append(h - h.mean())                          # zero DC gain
        self.register_buffer('taps', torch.stack(taps).view(len(bands), 1, 1, kernel_size))
        self.pad = half

    def forward(self, x):                                      # B,1,C,T -> B,F,C,T
        return nn.functional.conv2d(x, self.taps, padding=(0, self.pad))


class OrthoSincPow(nn.Module):
    """Frequency domain: fixed filter bank -> per-band spatial filters -> log-variance.

    FBCSP in a net. Log-power pooling is phase-invariant, so this reads envelope
    dynamics that TSConv's linear avg-pool cannot represent at all.
    """

    BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 spatial_per_band=8, n_windows=10):
        super().__init__()
        n_bands = len(self.BANDS)
        self.n_windows = n_windows
        self.win = eeg_sample_points // n_windows
        self.bank = _FixedFilterBank(self.BANDS)
        # grouped => each band gets its own private set of spatial filters
        self.spatial = nn.Conv2d(n_bands, n_bands * spatial_per_band,
                                 (channels_num, 1), groups=n_bands, bias=False)
        self.bn = nn.BatchNorm2d(n_bands * spatial_per_band)
        n_feat = n_bands * spatial_per_band * n_windows
        self.norm = nn.LayerNorm(n_feat)
        self.drop = nn.Dropout(0.25)
        self.head = _head(n_feat, feature_dim)

    def forward(self, x):
        B, C, T = x.shape
        z = self.bank(x.unsqueeze(1))                          # B, F, C, T
        z = self.bn(self.spatial(z)).squeeze(2)                # B, F*S, T
        z = z[:, :, :self.n_windows * self.win].reshape(z.shape[0], z.shape[1],
                                                        self.n_windows, self.win)
        feat = torch.log(z.var(dim=-1) + 1e-6)                 # B, F*S, W
        feat = self.drop(self.norm(feat.flatten(1)))
        return self.head(feat)


class OrthoSpec(nn.Module):
    """Complex time-frequency: STFT (real+imag, phase kept) -> 2-D CNN over (freq, time).

    The input is a genuinely different object -- a complex spectrogram image --
    rather than the raw (channel, time) matrix TSConv and ATM both consume.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 n_fft=64, hop=8, width=96):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels_num, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
            nn.Dropout2d(0.3),
        )
        with torch.no_grad():
            dummy = self._stft(torch.zeros(1, channels_num, eeg_sample_points))
            n_feat = self.net(dummy).flatten(1).shape[1]
        self.head = _head(n_feat, feature_dim)

    def _stft(self, x):
        B, C, T = x.shape
        z = torch.stft(x.reshape(B * C, T).float(), n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window.to(x.device), center=False, return_complex=True)
        z = torch.view_as_real(z)                              # B*C, freq, frames, 2
        z = z.permute(0, 3, 1, 2).reshape(B, C * 2, z.shape[1], z.shape[2])
        # compress dynamic range but keep sign (i.e. keep phase)
        return torch.sign(z) * torch.log1p(z.abs())

    def forward(self, x):
        return self.head(self.net(self._stft(x)).flatten(1))


class OrthoMixer(nn.Module):
    """No convolution, no attention: MLP-Mixer over (time-patch) tokens.

    Token mixing is a dense learned map across patches -- no locality prior, no
    weight sharing over time. That is exactly the prior TSConv is built on.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 patch=25, d_model=256, depth=4, drop=0.2):
        super().__init__()
        self.patch = patch
        n_tok = eeg_sample_points // patch
        self.embed = nn.Linear(channels_num * patch, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                'n1': nn.LayerNorm(d_model),
                'tok': nn.Sequential(nn.Linear(n_tok, n_tok * 2), nn.GELU(),
                                     nn.Linear(n_tok * 2, n_tok), nn.Dropout(drop)),
                'n2': nn.LayerNorm(d_model),
                'ch': nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                    nn.Linear(d_model * 2, d_model), nn.Dropout(drop)),
            }))
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(d_model * 2, feature_dim)

    def forward(self, x):
        B, C, T = x.shape
        n_tok = T // self.patch
        z = x[:, :, :n_tok * self.patch].reshape(B, C, n_tok, self.patch)
        z = z.permute(0, 2, 1, 3).reshape(B, n_tok, C * self.patch)
        z = self.embed(z)
        for b in self.blocks:
            z = z + b['tok'](b['n1'](z).transpose(1, 2)).transpose(1, 2)
            z = z + b['ch'](b['n2'](z))
        z = self.norm(z)
        return self.head(torch.cat([z.mean(1), z.max(1).values], dim=-1))


class OrthoTCN(nn.Module):
    """Multi-scale: spatial mix first, then dilated residual temporal convs.

    TSConv has one temporal scale (a single 25-tap kernel). This stacks
    dilations 1..16 for an exponentially growing receptive field at full
    temporal resolution -- no early 5x downsample.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 width=128, dilations=(1, 2, 4, 8, 16), drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels_num, width, 1), nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 7, padding=3 * d, dilation=d, groups=width),
                nn.Conv1d(width, width, 1),
                nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop))
            for d in dilations])
        self.head = _head(width * 2, feature_dim)

    def forward(self, x):
        z = self.stem(x)
        for b in self.blocks:
            z = z + b(z)
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], dim=-1))


class OrthoConvGRU(nn.Module):
    """Recurrence: short conv front-end, then a bidirectional GRU over time.

    A learned sequential state machine -- neither a fixed-support convolution
    nor a permutation-symmetric attention over channels.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 conv_width=96, hidden=160, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels_num, conv_width, 11, stride=5, padding=5),
            nn.BatchNorm1d(conv_width), nn.GELU(), nn.Dropout(drop))
        self.gru = nn.GRU(conv_width, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=drop)
        self.head = _head(hidden * 4, feature_dim)

    def forward(self, x):
        z = self.stem(x).transpose(1, 2)                       # B, T', conv_width
        out, _ = self.gru(z)
        return self.head(torch.cat([out.mean(1), out.max(1).values], dim=-1))



class OrthoCovPool(nn.Module):
    """Second-order pooling over *learned* feature maps (bilinear/covariance pooling).

    Same orthogonal claim as OrthoRiemann -- the readout is a covariance, not a
    linear pool -- but the covariance is taken over learned temporal filters
    instead of raw electrodes, so the second-order layer sits on features with
    usable SNR rather than on broadband sensor noise.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 n_maps=32, n_windows=4, drop=0.3):
        super().__init__()
        self.n_windows = n_windows
        self.stem = nn.Sequential(
            nn.Conv1d(channels_num, 64, 25, padding=12), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, n_maps, 1), nn.BatchNorm1d(n_maps), nn.GELU(), nn.Dropout(drop))
        self.win = eeg_sample_points // n_windows
        iu = torch.triu_indices(n_maps, n_maps)
        self.register_buffer('iu', iu, persistent=False)
        self.register_buffer('tri_w', torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)),
                             persistent=False)
        n_feat = n_windows * iu.shape[1]
        self.norm = nn.LayerNorm(n_feat)
        self.head = _head(n_feat, feature_dim)

    def forward(self, x):
        z = self.stem(x)                                       # B, M, T
        B, M, T = z.shape
        z = z[:, :, :self.n_windows * self.win].reshape(B, M, self.n_windows, self.win)
        z = z.permute(0, 2, 1, 3)                              # B, W, M, win
        cov = z @ z.transpose(-1, -2) / self.win
        # signed log compresses the dynamic range without needing an eigendecomposition
        cov = torch.sign(cov) * torch.log1p(cov.abs())
        feat = cov[..., self.iu[0], self.iu[1]] * self.tri_w
        return self.head(self.norm(feat.flatten(1)))


class OrthoPerceiver(nn.Module):
    """Latent-bottleneck cross-attention over joint (channel-block x time-patch) tokens.

    ATM attends channels to channels with the whole time course as each token's
    feature. This instead builds spatio-temporal tokens and squeezes them through a
    small set of learned latent queries -- a different attention topology and a
    hard information bottleneck ATM does not have.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 patch=25, d_model=192, n_latents=16, depth=3, heads=4, drop=0.2):
        super().__init__()
        self.patch = patch
        n_tok = eeg_sample_points // patch
        self.embed = nn.Linear(channels_num * patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d_model) * 0.02)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=drop,
                                                          batch_first=True) for _ in range(depth)])
        self.selfa = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=drop,
                                                          batch_first=True) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model),
                                               nn.Linear(d_model, d_model * 2), nn.GELU(),
                                               nn.Linear(d_model * 2, d_model), nn.Dropout(drop))
                                 for _ in range(depth)])
        self.nq = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.nk = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(d_model * n_latents, feature_dim)

    def forward(self, x):
        B, C, T = x.shape
        n_tok = T // self.patch
        kv = x[:, :, :n_tok * self.patch].reshape(B, C, n_tok, self.patch)
        kv = kv.permute(0, 2, 1, 3).reshape(B, n_tok, C * self.patch)
        kv = self.embed(kv) + self.pos
        z = self.latents.expand(B, -1, -1)
        for xa, sa, ff, nq, nk in zip(self.cross, self.selfa, self.ff, self.nq, self.nk):
            k = nk(kv)
            z = z + xa(nq(z), k, k, need_weights=False)[0]
            q = self.norm(z)
            z = z + sa(q, q, q, need_weights=False)[0]
            z = z + ff(z)
        return self.head(self.norm(z).flatten(1))


class _RelativeSelfAttention(nn.Module):
    """Temporal self-attention with a learned relative-position bias."""

    def __init__(self, d_model, heads, max_len, drop):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_len - 1, heads))
        position = torch.arange(max_len)
        relative_index = position[:, None] - position[None, :] + max_len - 1
        self.register_buffer('relative_index', relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        if T > self.relative_index.shape[0]:
            raise ValueError(f"sequence length {T} exceeds configured maximum")
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_bias[self.relative_index[:T, :T]].permute(2, 0, 1)
        weights = self.attn_drop(scores.add(bias.unsqueeze(0)).softmax(dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))


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


class _ConformerBlock(nn.Module):
    """Macaron FFN -> temporal attention -> depthwise conv -> Macaron FFN."""

    def __init__(self, d_model, heads, max_len, conv_kernel, expansion, drop):
        super().__init__()
        self.ffn1 = _ConformerFeedForward(d_model, expansion, drop)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _RelativeSelfAttention(d_model, heads, max_len, drop)
        self.conv = _ConformerConvModule(d_model, conv_kernel, drop)
        self.ffn2 = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class OrthoTemporalConformer(nn.Module):
    """Speech-style Conformer over time tokens, not the legacy EEGConformer.

    A conventional strided Conv1d stem jointly mixes electrodes and nearby samples.
    Every subsequent block alternates content-dependent global temporal attention
    with a gated local depthwise temporal convolution.  No EEG-specific feature is
    constructed and the ERP waveform is retained throughout the body.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=128, depth=4, heads=4, stem_stride=4,
                 conv_kernel=31, expansion=4, drop=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels_num, d_model, 9, stride=stem_stride, padding=4),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        max_tokens = math.ceil(eeg_sample_points / stem_stride)
        self.blocks = nn.ModuleList([
            _ConformerBlock(
                d_model=d_model,
                heads=heads,
                max_len=max_tokens,
                conv_kernel=conv_kernel,
                expansion=expansion,
                drop=drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.stem(x).transpose(1, 2)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        return self.head(torch.cat([z.mean(1), z.max(1).values], dim=-1))


class _SqueezeformerBlock(nn.Module):
    """Compact attention/conv/FFN block for short temporal token sequences."""

    def __init__(self, d_model, heads, max_len, conv_kernel, expansion, drop):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _RelativeSelfAttention(d_model, heads, max_len, drop)
        self.conv = _ConformerConvModule(d_model, conv_kernel, drop)
        self.ffn = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return self.final_norm(x)


class OrthoTSConvSqueezeformer(nn.Module):
    """TSConv's proven sensor stem followed by compact temporal sequence blocks.

    Unlike TSConv, the 36 post-stem time bins remain tokens instead of being
    flattened immediately.  The body then alternates global temporal attention
    and gated local depthwise convolution before attentive statistics pooling.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=160, depth=4, heads=4, conv_kernel=15,
                 expansion=4, drop=0.15):
        super().__init__()
        temporal_kernel = 25
        pool_kernel = 51
        pool_stride = 5
        conv_len = eeg_sample_points - temporal_kernel + 1
        max_tokens = (conv_len - pool_kernel) // pool_stride + 1
        if max_tokens <= 0:
            raise ValueError("EEG window is too short for the TSConv stem")

        self.stem = nn.Sequential(
            nn.Conv2d(1, 40, (1, temporal_kernel), (1, 1)),
            nn.AvgPool2d((1, pool_kernel), (1, pool_stride)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(drop),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(40, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            _SqueezeformerBlock(
                d_model=d_model,
                heads=heads,
                max_len=max_tokens,
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
        z = self.stem(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        z = self.token_projection(z)
        for block in self.blocks:
            z = block(z)

        weights = self.pool_score(z).softmax(dim=1)
        mean = (weights * z).sum(dim=1)
        variance = (weights * (z - mean.unsqueeze(1)).square()).sum(dim=1)
        pooled = torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)
        return self.head(pooled)


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


class OrthoFastTSConvSqueezeformer(nn.Module):
    """Latency-oriented TSConv-Squeezeformer retaining the same core biases.

    Changes relative to OrthoTSConvSqueezeformer are deliberately mechanical:
    exact fusion of the adjacent linear temporal-conv/average-pool operations,
    three rather than four blocks, a 2x rather than 4x FFN, and fused SDPA.
    Width, temporal token count, convolution module and output head are retained.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=160, depth=3, heads=4, conv_kernel=15,
                 expansion=2, drop=0.15):
        super().__init__()
        self.temporal = _FusedTemporalConvPool(
            out_channels=40,
            temporal_kernel=25,
            pool_kernel=51,
            pool_stride=5,
        )
        max_tokens = (eeg_sample_points - self.temporal.effective_kernel) // 5 + 1
        if max_tokens <= 0:
            raise ValueError("EEG window is too short for the fused TSConv stem")
        self.stem_after_pool = nn.Sequential(
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(drop),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(40, d_model),
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


class OrthoFastLite160x2(OrthoFastTSConvSqueezeformer):
    """Depth ablation: retain width 160 and use two lightweight blocks."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=160,
            depth=2,
            heads=4,
            conv_kernel=15,
            expansion=2,
            drop=0.15,
        )


class OrthoFastLite128x3(OrthoFastTSConvSqueezeformer):
    """Width ablation: use width 128 while retaining three blocks."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=128,
            depth=3,
            heads=4,
            conv_kernel=15,
            expansion=2,
            drop=0.15,
        )


class OrthoFastLite128x2(OrthoFastTSConvSqueezeformer):
    """Combined light candidate: width 128 with two lightweight blocks."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=128,
            depth=2,
            heads=4,
            conv_kernel=15,
            expansion=2,
            drop=0.15,
        )


class _RawChannelAttentionBlock(nn.Module):
    """Content-dependent sensor mixing with an exact identity path at init."""

    def __init__(self, sample_points, d_model=128, heads=4, expansion=2, drop=0.15):
        super().__init__()
        self.input_norm = nn.LayerNorm(sample_points)
        self.input_projection = nn.Linear(sample_points, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _FastSelfAttention(d_model, heads, drop)
        self.ffn = _ConformerFeedForward(d_model, expansion, drop)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, sample_points)
        self.drop = nn.Dropout(drop)
        # Start as the proven raw-waveform path. Training only has to learn a
        # useful correction, rather than reconstructing the ERP from scratch.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x):                                    # B, C, T
        z = self.input_projection(self.input_norm(x))
        z = z + self.attn(self.attn_norm(z))
        z = z + self.ffn(z)
        return x + self.drop(self.output_projection(self.output_norm(z)))


class OrthoChannelAttnTSConv(nn.Module):
    """Raw-waveform channel attention feeding the successful fast TSConv body.

    The channel block can learn subject-robust, input-dependent scalp mixing,
    while its zero-initialized output leaves an exact TSConv-Squeezeformer path
    at initialization. This combines ATM's strongest bias with a temporal-token
    body without forcing either representation through a narrow bottleneck.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        self.channel_context = _RawChannelAttentionBlock(
            eeg_sample_points, d_model=128, heads=4, expansion=2, drop=0.15,
        )
        self.encoder = OrthoFastTSConvSqueezeformer(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=128,
            depth=3,
            heads=4,
            conv_kernel=15,
            expansion=2,
            drop=0.15,
        )

    def forward(self, x):
        return self.encoder(self.channel_context(x))


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


class OrthoMultiScaleTSMixer(nn.Module):
    """Multi-resolution ERP filters + whole-scalp projections + MLP-Mixer.

    Unlike frequency-power models, all branches retain waveform sign and timing.
    Unlike the failed electrode-preserving candidates, each branch integrates all
    sensors before any low-dimensional token processing.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 tokens=36, depth=3, drop=0.15):
        super().__init__()
        self.stem = _MultiResolutionSpatialStem(
            channels_num, filters_per_scale=16, kernels=(9, 25, 49),
            output_tokens=tokens, drop=drop,
        )
        d_model = self.stem.output_dim
        self.position = nn.Parameter(torch.zeros(1, tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            _MLPMixerSequenceBlock(tokens, d_model, token_hidden=72,
                                   expansion=3, drop=drop)
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


class _RankedTemporalSpatialStem(nn.Module):
    """TSConv temporal bank with several spatial components per filter."""

    def __init__(self, eeg_sample_points, channels_num, temporal_filters=40,
                 spatial_rank=2, drop=0.15):
        super().__init__()
        self.temporal = _FusedTemporalConvPool(
            out_channels=temporal_filters,
            temporal_kernel=25,
            pool_kernel=51,
            pool_stride=5,
        )
        self.output_tokens = (eeg_sample_points - self.temporal.effective_kernel) // 5 + 1
        self.after_temporal = nn.Sequential(
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Conv2d(
                temporal_filters,
                temporal_filters * spatial_rank,
                (channels_num, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters * spatial_rank),
            nn.ELU(),
            nn.Dropout(drop),
        )
        self.output_dim = temporal_filters * spatial_rank

    def forward(self, x):
        z = self.temporal(x.unsqueeze(1))
        return self.after_temporal(z).squeeze(2).transpose(1, 2)


class OrthoRankedTSBiGRU(nn.Module):
    """Multiple scalp modes per temporal filter followed by bidirectional recurrence.

    TSConv hard-collapses each temporal filter to one spatial mode. Rank two keeps
    two independently learned scalp patterns, while the BiGRU imposes ordered
    temporal state rather than attention or permutation-symmetric statistics.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 hidden=128, drop=0.15):
        super().__init__()
        self.stem = _RankedTemporalSpatialStem(
            eeg_sample_points, channels_num, temporal_filters=40,
            spatial_rank=2, drop=drop,
        )
        self.position = nn.Parameter(torch.zeros(
            1, self.stem.output_tokens, self.stem.output_dim,
        ))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.gru = nn.GRU(
            self.stem.output_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=drop,
        )
        self.norm = nn.LayerNorm(2 * hidden)
        self.pool = _AttentiveTokenStatsPool(2 * hidden)
        self.head = _head(4 * hidden, feature_dim)

    def forward(self, x):
        z = self.stem(x)
        z, _ = self.gru(z + self.position[:, :z.shape[1]])
        return self.head(self.pool(self.norm(z)))


class OrthoMultiScaleSqueezeformer(nn.Module):
    """Multi-resolution ERP stem followed by compact temporal attention/conv blocks.

    This crosses the two strongest new ideas seen so far: the multi-scale stem that
    made the Mixer useful to ensembles, and the Squeezeformer body that produced
    the strongest new solo model. Absolute positions and signed waveforms are
    retained; all electrodes are integrated before the narrow sequence model.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 tokens=36, d_model=144, depth=3, heads=4, drop=0.15):
        super().__init__()
        self.stem = _MultiResolutionSpatialStem(
            channels_num, filters_per_scale=16, kernels=(9, 25, 49),
            output_tokens=tokens, drop=drop,
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.stem.output_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = nn.Parameter(torch.zeros(1, tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            _FastSqueezeformerBlock(
                d_model=d_model,
                heads=heads,
                conv_kernel=15,
                expansion=2,
                drop=drop,
            )
            for _ in range(depth)
        ])
        self.pool = _AttentiveTokenStatsPool(d_model)
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.token_projection(self.stem(x))
        z = z + self.position[:, :z.shape[1]]
        for block in self.blocks:
            z = block(z)
        return self.head(self.pool(z))


class _GatedDilatedTemporalBlock(nn.Module):
    """A WaveNet-style local mixer with a residual feature MLP."""

    def __init__(self, d_model, dilation, kernel=7, expansion=2, drop=0.15):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            2 * d_model,
            kernel,
            padding=(kernel // 2) * dilation,
            dilation=dilation,
            groups=d_model,
        )
        self.temporal_projection = nn.Conv1d(d_model, d_model, 1)
        self.temporal_drop = nn.Dropout(drop)
        self.feature = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        z = self.temporal_norm(x).transpose(1, 2)
        signal, gate = self.depthwise(z).chunk(2, dim=1)
        z = self.temporal_projection(nn.functional.gelu(signal) * torch.sigmoid(gate))
        x = x + self.temporal_drop(z.transpose(1, 2))
        x = x + self.feature(x)
        return self.final_norm(x)


class OrthoRankedDilatedConv(nn.Module):
    """Rank-two whole-scalp stem plus a purely convolutional temporal hierarchy.

    No recurrence or attention is used. Dilations span the complete 36-token ERP
    while the flattened head keeps absolute latency, a useful bias that pooling
    removed from several earlier weak candidates.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=96, dilations=(1, 2, 4, 8), drop=0.15):
        super().__init__()
        self.stem = _RankedTemporalSpatialStem(
            eeg_sample_points, channels_num, temporal_filters=40,
            spatial_rank=2, drop=drop,
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.stem.output_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = nn.Parameter(torch.zeros(1, self.stem.output_tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            _GatedDilatedTemporalBlock(
                d_model=d_model, dilation=dilation, kernel=7,
                expansion=2, drop=drop,
            )
            for dilation in dilations
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(self.stem.output_tokens * d_model, feature_dim)

    def forward(self, x):
        z = self.token_projection(self.stem(x))
        z = z + self.position[:, :z.shape[1]]
        for block in self.blocks:
            z = block(z)
        return self.head(self.norm(z).flatten(1))


class _BidirectionalDiagonalSSMBlock(nn.Module):
    """Lightweight learned exponential state-space scan in both time directions."""

    def __init__(self, d_model, expansion=2, drop=0.15):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, 2 * d_model)
        # Cover short and long ERP time constants at initialization. Each state
        # learns its own decay independently in each temporal direction.
        initial_decay = torch.linspace(0.55, 0.97, d_model)
        initial_logits = torch.logit(initial_decay)
        self.forward_decay_logits = nn.Parameter(initial_logits.clone())
        self.backward_decay_logits = nn.Parameter(initial_logits.flip(0).clone())
        self.output_projection = nn.Linear(2 * d_model, d_model)
        self.state_drop = nn.Dropout(drop)
        self.feature = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _scan(inputs, decay, reverse=False):
        state = torch.zeros_like(inputs[:, 0])
        outputs = [None] * inputs.shape[1]
        indices = range(inputs.shape[1] - 1, -1, -1) if reverse else range(inputs.shape[1])
        for index in indices:
            state = decay * state + (1.0 - decay) * inputs[:, index]
            outputs[index] = state
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        signal, gate = self.input_projection(self.input_norm(x)).chunk(2, dim=-1)
        signal = torch.tanh(signal)
        forward_decay = torch.sigmoid(self.forward_decay_logits).view(1, -1)
        backward_decay = torch.sigmoid(self.backward_decay_logits).view(1, -1)
        forward_state = self._scan(signal, forward_decay)
        backward_state = self._scan(signal, backward_decay, reverse=True)
        state = self.output_projection(torch.cat([forward_state, backward_state], dim=-1))
        x = x + self.state_drop(state * torch.sigmoid(gate))
        x = x + self.feature(x)
        return self.final_norm(x)


class OrthoMultiScaleBiSSM(nn.Module):
    """Multi-scale full-scalp stem plus bidirectional diagonal state-space scans.

    The recurrent exponential memory supplies a genuinely different sequence
    prior from convolution, GRU recurrence, and self-attention while remaining
    compact enough for the 36-token EEG sequence.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 tokens=36, d_model=128, depth=3, drop=0.15):
        super().__init__()
        self.stem = _MultiResolutionSpatialStem(
            channels_num, filters_per_scale=16, kernels=(9, 25, 49),
            output_tokens=tokens, drop=drop,
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.stem.output_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = nn.Parameter(torch.zeros(1, tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            _BidirectionalDiagonalSSMBlock(d_model, expansion=2, drop=drop)
            for _ in range(depth)
        ])
        self.pool = _AttentiveTokenStatsPool(d_model)
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.token_projection(self.stem(x))
        z = z + self.position[:, :z.shape[1]]
        for block in self.blocks:
            z = block(z)
        return self.head(self.pool(z))


class _LatentTemporalCrossBlock(nn.Module):
    """Perceiver cross-attention into a small latent array, then latent self-attention."""

    def __init__(self, d_model, heads=4, expansion=2, drop=0.15):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=drop, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, heads, dropout=drop, batch_first=True,
        )
        self.feature = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, latents, tokens):
        query = self.query_norm(latents)
        key_value = self.token_norm(tokens)
        update, _ = self.cross_attention(
            query, key_value, key_value, need_weights=False,
        )
        latents = latents + update
        query = self.self_norm(latents)
        update, _ = self.self_attention(query, query, query, need_weights=False)
        latents = latents + update
        latents = latents + self.feature(latents)
        return self.final_norm(latents)


class OrthoRankedPerceiver(nn.Module):
    """Rank-two whole-scalp ERP tokens compressed into learned temporal latents.

    The earlier raw Perceiver was weak because it had to discover useful scalp
    projections and temporal abstractions simultaneously. Here the validated
    rank-two stem performs the former before latent cross-attention does the latter.
    """

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=128, latent_tokens=12, depth=3, drop=0.15):
        super().__init__()
        self.stem = _RankedTemporalSpatialStem(
            eeg_sample_points, channels_num, temporal_filters=40,
            spatial_rank=2, drop=drop,
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.stem.output_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = nn.Parameter(torch.zeros(
            1, self.stem.output_tokens, d_model,
        ))
        self.latents = nn.Parameter(torch.zeros(1, latent_tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList([
            _LatentTemporalCrossBlock(
                d_model=d_model, heads=4, expansion=2, drop=drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Latent order is learned and stable, so flattening retains information
        # that mean pooling would discard.
        self.head = _head(latent_tokens * d_model, feature_dim)

    def forward(self, x):
        tokens = self.token_projection(self.stem(x))
        tokens = tokens + self.position[:, :tokens.shape[1]]
        latents = self.latents.expand(x.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, tokens)
        return self.head(self.norm(latents).flatten(1))


class _SharedElectrodeTemporalStem(nn.Module):
    """Apply one temporal encoder with shared weights to every electrode."""

    def __init__(self, eeg_sample_points, d_model, kernel=25, stride=5,
                 pool_kernel=3, pool_stride=3):
        super().__init__()
        conv_tokens = (eeg_sample_points - kernel) // stride + 1
        self.output_tokens = (conv_tokens - pool_kernel) // pool_stride + 1
        if self.output_tokens <= 0:
            raise ValueError("EEG window is too short for the temporal stem")
        self.net = nn.Sequential(
            nn.Conv1d(1, d_model, kernel, stride=stride),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AvgPool1d(pool_kernel, pool_stride),
        )

    def forward(self, x):
        B, C, T = x.shape
        z = self.net(x.reshape(B * C, 1, T))                  # B*C, D, T'
        return z.transpose(1, 2).reshape(B, C, z.shape[-1], z.shape[1])


class _PerStreamTemporalBlock(nn.Module):
    """ConvNeXt-like temporal block independently shared across spatial streams."""

    def __init__(self, d_model, kernel=7, dilation=1, expansion=2, drop=0.15):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("temporal kernel must be odd")
        self.temporal_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel,
            padding=(kernel // 2) * dilation,
            dilation=dilation,
            groups=d_model,
        )
        self.temporal_projection = nn.Linear(d_model, d_model)
        self.temporal_drop = nn.Dropout(drop)
        self.feature = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):                                    # B, S, T, D
        B, S, T, D = x.shape
        z = self.temporal_norm(x).permute(0, 1, 3, 2).reshape(B * S, D, T)
        z = self.depthwise(z).reshape(B, S, D, T).permute(0, 1, 3, 2)
        z = self.temporal_drop(self.temporal_projection(nn.functional.gelu(z)))
        x = x + z
        x = x + self.feature(x)
        return self.final_norm(x)


class _AttentiveTokenStatsPool(nn.Module):
    """Attentive mean and standard deviation over arbitrary token axes."""

    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(8, d_model // 2)),
            nn.Tanh(),
            nn.Linear(max(8, d_model // 2), 1),
        )

    def forward(self, x):
        z = x.reshape(x.shape[0], -1, x.shape[-1])
        weights = self.score(z).softmax(dim=1)
        mean = (weights * z).sum(dim=1)
        variance = (weights * (z - mean.unsqueeze(1)).square()).sum(dim=1)
        return torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)


class _AxialConvMixerBlock(nn.Module):
    """Temporal convolution, fixed low-rank electrode mixing, then feature MLP."""

    def __init__(self, channels_num, d_model, channel_rank=24,
                 temporal_kernel=7, feature_expansion=2, drop=0.15):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal = nn.Conv1d(
            d_model, d_model, temporal_kernel,
            padding=temporal_kernel // 2, groups=d_model,
        )
        self.temporal_drop = nn.Dropout(drop)
        self.channel_norm = nn.LayerNorm(channels_num)
        self.channel_mixer = nn.Sequential(
            nn.Linear(channels_num, channel_rank),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(channel_rank, channels_num),
            nn.Dropout(drop),
        )
        self.feature_mixer = _ConformerFeedForward(d_model, feature_expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):                                    # B, C, T, D
        B, C, T, D = x.shape
        z = self.temporal_norm(x).permute(0, 1, 3, 2).reshape(B * C, D, T)
        z = self.temporal(z).reshape(B, C, D, T).permute(0, 1, 3, 2)
        x = x + self.temporal_drop(nn.functional.gelu(z))

        z = x.permute(0, 2, 3, 1)                            # B, T, D, C
        z = self.channel_mixer(self.channel_norm(z)).permute(0, 3, 1, 2)
        x = x + z
        x = x + self.feature_mixer(x)
        return self.final_norm(x)


class OrthoAxialConvMixer(nn.Module):
    """Electrode-preserving temporal CNN with attention-free axial MLP mixing."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=20, depth=3, channel_rank=20, drop=0.15):
        super().__init__()
        self.stem = _SharedElectrodeTemporalStem(eeg_sample_points, d_model)
        tokens = self.stem.output_tokens
        self.electrode_position = nn.Parameter(torch.zeros(1, channels_num, 1, d_model))
        self.temporal_position = nn.Parameter(torch.zeros(1, 1, tokens, d_model))
        nn.init.trunc_normal_(self.electrode_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        self.blocks = nn.ModuleList([
            _AxialConvMixerBlock(
                channels_num=channels_num,
                d_model=d_model,
                channel_rank=channel_rank,
                feature_expansion=2,
                drop=drop,
            )
            for _ in range(depth)
        ])
        self.pool = _AttentiveTokenStatsPool(d_model)
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.stem(x)
        z = z + self.electrode_position + self.temporal_position[:, :, :z.shape[2]]
        for block in self.blocks:
            z = block(z)
        return self.head(self.pool(z))


class OrthoLateSpatialResNet(nn.Module):
    """Nonlinear per-electrode temporal tower followed by late spatial compression."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=20, n_latents=12, early_depth=3, drop=0.15):
        super().__init__()
        self.stem = _SharedElectrodeTemporalStem(eeg_sample_points, d_model)
        tokens = self.stem.output_tokens
        self.electrode_position = nn.Parameter(torch.zeros(1, channels_num, 1, d_model))
        self.temporal_position = nn.Parameter(torch.zeros(1, 1, tokens, d_model))
        nn.init.trunc_normal_(self.electrode_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        self.early_blocks = nn.ModuleList([
            _PerStreamTemporalBlock(d_model, kernel=7, expansion=2, drop=drop)
            for _ in range(early_depth)
        ])
        self.pre_spatial_norm = nn.LayerNorm(d_model)
        self.spatial = nn.Linear(channels_num, n_latents, bias=False)
        nn.init.orthogonal_(self.spatial.weight)
        self.latent_blocks = nn.ModuleList([
            _PerStreamTemporalBlock(
                d_model, kernel=7, dilation=dilation, expansion=2, drop=drop,
            )
            for dilation in (1, 2, 4)
        ])
        self.pool = _AttentiveTokenStatsPool(d_model)
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.stem(x)
        z = z + self.electrode_position + self.temporal_position[:, :, :z.shape[2]]
        for block in self.early_blocks:
            z = block(z)
        z = self.pre_spatial_norm(z).permute(0, 2, 3, 1)     # B, T, D, C
        z = self.spatial(z).permute(0, 3, 1, 2)              # B, L, T, D
        for block in self.latent_blocks:
            z = block(z)
        return self.head(self.pool(z))


class _TemporalDeepSetBlock(nn.Module):
    """Mean/variance context update, equivariant over the electrode axis."""

    def __init__(self, d_model, expansion=2, drop=0.15):
        super().__init__()
        hidden = expansion * d_model
        self.norm = nn.LayerNorm(d_model)
        self.local_projection = nn.Linear(d_model, hidden)
        self.context_projection = nn.Linear(2 * d_model, hidden, bias=False)
        self.output = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):                                    # B, C, T, D
        z = self.norm(x)
        mean = z.mean(dim=1, keepdim=True)
        std = z.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6).sqrt()
        context = self.context_projection(torch.cat([mean, std], dim=-1))
        update = self.output(self.drop(nn.functional.gelu(
            self.local_projection(z) + context
        )))
        return self.final_norm(x + self.drop(update))


class OrthoTemporalDeepSets(nn.Module):
    """Shared temporal encoder with attention-free DeepSets channel communication."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 d_model=20, temporal_depth=2, set_depth=3, drop=0.15):
        super().__init__()
        self.stem = _SharedElectrodeTemporalStem(eeg_sample_points, d_model)
        tokens = self.stem.output_tokens
        self.electrode_position = nn.Parameter(torch.zeros(1, channels_num, 1, d_model))
        self.temporal_position = nn.Parameter(torch.zeros(1, 1, tokens, d_model))
        nn.init.trunc_normal_(self.electrode_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        self.temporal_blocks = nn.ModuleList([
            _PerStreamTemporalBlock(d_model, kernel=7, expansion=2, drop=drop)
            for _ in range(temporal_depth)
        ])
        self.set_blocks = nn.ModuleList([
            _TemporalDeepSetBlock(d_model, expansion=2, drop=drop)
            for _ in range(set_depth)
        ])
        self.pool = _AttentiveTokenStatsPool(d_model)
        self.head = _head(2 * d_model, feature_dim)

    def forward(self, x):
        z = self.stem(x)
        z = z + self.electrode_position + self.temporal_position[:, :, :z.shape[2]]
        for block in self.temporal_blocks:
            z = block(z)
        for block in self.set_blocks:
            z = block(z)
        return self.head(self.pool(z))


class _ConvNeXt1DBlock(nn.Module):
    """The standard ConvNeXt depthwise/LayerNorm/inverted-bottleneck block in 1-D."""

    def __init__(self, dim, drop=0.1, layer_scale=1e-6):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 4 * dim)
        self.contract = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        z = self.depthwise(x).transpose(1, 2)
        z = self.contract(nn.functional.gelu(self.expand(self.norm(z))))
        z = self.drop(z * self.gamma).transpose(1, 2)
        return residual + z


class OrthoConvNeXt1D(nn.Module):
    """Hierarchical modern ConvNet over the raw multichannel temporal sequence."""

    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63,
                 dims=(96, 192, 256), depths=(2, 2, 4), drop=0.1):
        super().__init__()
        if len(dims) != len(depths):
            raise ValueError("ConvNeXt dims and depths must have equal length")
        self.stem = nn.Conv1d(channels_num, dims[0], 7, stride=2, padding=3)
        self.stem_norm = nn.LayerNorm(dims[0])
        self.stages = nn.ModuleList([
            nn.Sequential(*[_ConvNeXt1DBlock(dim, drop=drop) for _ in range(depth)])
            for dim, depth in zip(dims, depths)
        ])
        self.downsample_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in dims[:-1]
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv1d(dims[i], dims[i + 1], 2, stride=2)
            for i in range(len(dims) - 1)
        ])
        self.final_norm = nn.LayerNorm(dims[-1])
        self.head = _head(2 * dims[-1], feature_dim)

    def forward(self, x):
        z = self.stem(x)
        z = self.stem_norm(z.transpose(1, 2)).transpose(1, 2)
        for index, stage in enumerate(self.stages):
            z = stage(z)
            if index < len(self.downsamples):
                z = self.downsample_norms[index](z.transpose(1, 2)).transpose(1, 2)
                z = self.downsamples[index](z)
        z = self.final_norm(z.transpose(1, 2))
        return self.head(torch.cat([z.mean(1), z.max(1).values], dim=-1))


_REGISTRY = {
    'OrthoRiemann': OrthoRiemann,
    'OrthoSincPow': OrthoSincPow,
    'OrthoSpec': OrthoSpec,
    'OrthoMixer': OrthoMixer,
    'OrthoTCN': OrthoTCN,
    'OrthoConvGRU': OrthoConvGRU,
    'OrthoCovPool': OrthoCovPool,
    'OrthoPerceiver': OrthoPerceiver,
    'OrthoTemporalConformer': OrthoTemporalConformer,
    'OrthoTSConvSqueezeformer': OrthoTSConvSqueezeformer,
    'OrthoFastTSConvSqueezeformer': OrthoFastTSConvSqueezeformer,
    'OrthoFastLite160x2': OrthoFastLite160x2,
    'OrthoFastLite128x3': OrthoFastLite128x3,
    'OrthoFastLite128x2': OrthoFastLite128x2,
    'OrthoChannelAttnTSConv': OrthoChannelAttnTSConv,
    'OrthoMultiScaleTSMixer': OrthoMultiScaleTSMixer,
    'OrthoRankedTSBiGRU': OrthoRankedTSBiGRU,
    'OrthoMultiScaleSqueezeformer': OrthoMultiScaleSqueezeformer,
    'OrthoRankedDilatedConv': OrthoRankedDilatedConv,
    'OrthoMultiScaleBiSSM': OrthoMultiScaleBiSSM,
    'OrthoRankedPerceiver': OrthoRankedPerceiver,
    'OrthoAxialConvMixer': OrthoAxialConvMixer,
    'OrthoLateSpatialResNet': OrthoLateSpatialResNet,
    'OrthoTemporalDeepSets': OrthoTemporalDeepSets,
    'OrthoConvNeXt1D': OrthoConvNeXt1D,
}


def build_ortho_encoder(name, feature_dim, eeg_sample_points, channels_num):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown orthogonal encoder {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](feature_dim=feature_dim, eeg_sample_points=eeg_sample_points,
                           channels_num=channels_num)
