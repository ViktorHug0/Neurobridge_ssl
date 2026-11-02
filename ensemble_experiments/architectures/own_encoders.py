"""AUG_LOG 2026-10-29 decorrelated-encoder search: members designed to be strong but to err differently from the REVE
family (electrode x time patch transformer) and TSC/SQF (shallow temporal-then-spatial convolution).

FourierFormer      transformer over FREQUENCY tokens: one token per rFFT bin up to max_hz, holding the real and imaginary
                   coefficients of every channel (the phase-locked ERP, global in time, local in frequency).
DilatedConvNeXt1D  deep residual temporal network: ConvNeXt-style blocks (dilated depthwise conv, per-dataset BatchNorm,
                   pointwise MLP) that mix channels at every layer; dilations reach the whole 1 s trial.
DilatedConvNeXt1D(input_mode='phase'): the trunk sees only the unit-magnitude analytic phase (cos, sin) of each channel in
                   delta / theta / alpha bands -- phase-locked image signal without the per-person amplitude.
All: per-trial per-channel z-score input, attention mean+std pooling, Linear(feature_dim)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.eeg_encoder.model import DomainBatchNorm2d


def _zscore(x):
    x = x.float()
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)


def _band_phase(x, bands, fs):
    """(batch, channels, time) -> (batch, 2 len(bands) channels, time): cos/sin of the analytic phase per band (FFT Hilbert)."""
    spec = torch.fft.fft(x, dim=-1)
    f = torch.fft.fftfreq(x.shape[-1], 1.0 / fs).to(x.device)
    out = []
    for lo, hi in bands:
        a = torch.fft.ifft(spec * (2.0 * ((f >= lo) & (f < hi))), dim=-1)  # positive band only, doubled = analytic signal
        u = a / a.abs().clamp_min(1e-6)
        out += [u.real, u.imag]
    return torch.cat(out, 1)


class _AttnStatPool(nn.Module):
    """(batch, tokens, d) -> (batch, 2 d): attention-weighted mean and standard deviation."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h):
        w = self.score(h).softmax(dim=1)
        mean = (w * h).sum(1)
        return torch.cat([mean, (w * (h - mean.unsqueeze(1)).square()).sum(1).clamp_min(1e-6).sqrt()], -1)


class FourierFormer(nn.Module):
    def __init__(self, feature_dim, eeg_sample_points, channels_num, max_hz=24.0, fs=120.0, d_model=256, depth=6,
                 heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.n_bins = int(max_hz * eeg_sample_points / fs) + 1
        self.embed = nn.Linear(2 * channels_num, d_model)
        self.freq = nn.Parameter(torch.randn(self.n_bins, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.pool = _AttnStatPool(d_model)
        self.head = nn.Linear(2 * d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        x = _zscore(x)
        spec = torch.fft.rfft(x, dim=-1)[..., :self.n_bins] / math.sqrt(x.shape[-1])  # (batch, channels, bins)
        tokens = torch.cat([spec.real, spec.imag], 1).transpose(1, 2)  # (batch, bins, 2 channels)
        h = self.blocks(self.embed(tokens) + self.freq)
        return self.head(self.pool(h))


class _ConvNeXtBlock1D(nn.Module):
    def __init__(self, d, dilation, kernel, domain_bn_boundary, dropout):
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, groups=d)
        self.domain = domain_bn_boundary > 0
        self.norm = DomainBatchNorm2d(d, domain_bn_boundary, norm=nn.BatchNorm1d) if self.domain else nn.BatchNorm1d(d)
        self.pw1, self.pw2 = nn.Conv1d(d, 4 * d, 1), nn.Conv1d(4 * d, d, 1)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.full((d, 1), 0.1))

    def forward(self, h, subject_ids):
        r = self.dw(h)
        r = self.norm(r, subject_ids) if self.domain else self.norm(r)
        return h + self.gamma * self.drop(self.pw2(F.gelu(self.pw1(r))))


class DilatedConvNeXt1D(nn.Module):
    def __init__(self, feature_dim, eeg_sample_points, channels_num, d=128, kernel=7, dilations=(1, 2, 4, 8, 16, 1, 2, 4),
                 dropout=0.1, domain_bn_boundary=0, input_mode='raw', bands=((1, 4), (4, 8), (8, 13)), fs=120.0):
        super().__init__()
        self.input_mode, self.bands, self.fs = input_mode, tuple(map(tuple, bands)), fs
        self.stem = nn.Conv1d(channels_num * (2 * len(self.bands) if input_mode == 'phase' else 1), d, 5, padding=2)
        self.blocks = nn.ModuleList(_ConvNeXtBlock1D(d, r, kernel, domain_bn_boundary, dropout) for r in dilations)
        self.pool = _AttnStatPool(d)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, feature_dim))

    def forward(self, x, subject_ids=None):
        x = _zscore(x)
        h = self.stem(_band_phase(x, self.bands, self.fs) if self.input_mode == 'phase' else x)
        for b in self.blocks:
            h = b(h, subject_ids)
        return self.head(self.pool(h.transpose(1, 2)))


class SpecPatchFormer(nn.Module):
    """REVE's electrode-token bias in the frequency domain: one token per (electrode, band of `patch` rFFT bins) holding
    the real and imaginary coefficients; learned electrode + band embeddings; transformer; attention mean+std pool."""

    def __init__(self, feature_dim, eeg_sample_points, channels_num, max_hz=24.0, fs=120.0, patch=5, d_model=256, depth=6,
                 heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        n_bins = int(max_hz * eeg_sample_points / fs) + 1
        self.patch, self.n_patch = patch, math.ceil(n_bins / patch)
        self.n_bins = self.n_patch * patch
        self.embed = nn.Linear(2 * patch, d_model)
        self.electrode = nn.Parameter(torch.randn(channels_num, 1, d_model) * 0.02)
        self.band = nn.Parameter(torch.randn(1, self.n_patch, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.pool = _AttnStatPool(d_model)
        self.head = nn.Linear(2 * d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        x = _zscore(x)
        spec = torch.fft.rfft(x, dim=-1)[..., :self.n_bins] / math.sqrt(x.shape[-1])  # (batch, electrodes, bins)
        b, c, _ = spec.shape
        z = torch.stack([spec.real, spec.imag], -1).reshape(b, c, self.n_patch, 2 * self.patch)
        h = (self.embed(z) + self.electrode + self.band).reshape(b, c * self.n_patch, -1)
        return self.head(self.pool(self.blocks(h)))


class DualDomainFormer(nn.Module):
    """One transformer over two token kinds: time patches (all channels x `patch` samples) and FourierFormer's frequency
    tokens (all channels' real/imaginary coefficients per rFFT bin up to max_hz); type + position embeddings."""

    def __init__(self, feature_dim, eeg_sample_points, channels_num, patch=10, max_hz=24.0, fs=120.0, d_model=256, depth=6,
                 heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.patch, self.n_time = patch, eeg_sample_points // patch
        self.n_bins = int(max_hz * eeg_sample_points / fs) + 1
        self.time_embed = nn.Linear(channels_num * patch, d_model)
        self.freq_embed = nn.Linear(2 * channels_num, d_model)
        self.pos = nn.Parameter(torch.randn(self.n_time + self.n_bins, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.pool = _AttnStatPool(d_model)
        self.head = nn.Linear(2 * d_model, feature_dim)

    def forward(self, x, subject_ids=None):
        x = _zscore(x)
        b, c, t = x.shape
        tt = x[..., :self.n_time * self.patch].reshape(b, c, self.n_time, self.patch).permute(0, 2, 1, 3).reshape(b, self.n_time, -1)
        spec = torch.fft.rfft(x, dim=-1)[..., :self.n_bins] / math.sqrt(t)
        ft = torch.cat([spec.real, spec.imag], 1).transpose(1, 2)
        h = torch.cat([self.time_embed(tt), self.freq_embed(ft)], 1) + self.pos
        return self.head(self.pool(self.blocks(h)))


if __name__ == '__main__':
    x, sid = torch.randn(6, 32, 120), torch.tensor([1, 2, 3, 21, 22, 25])
    for m in (FourierFormer(1536, 120, 32), DilatedConvNeXt1D(1536, 120, 32, domain_bn_boundary=21),
              DilatedConvNeXt1D(1536, 120, 32, domain_bn_boundary=21, input_mode='phase'), SpecPatchFormer(1536, 120, 32),
              DualDomainFormer(1536, 120, 32)):
        out = m(x, sid)
        assert out.shape == (6, 1536)
        out.sum().backward()
        m.eval()
        assert torch.allclose(m(x, sid), m(3 * x + 2, sid), atol=1e-3), 'scale/offset-free input'
        print(type(m).__name__, 'ok', f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
    # the frequency tokens carry the whole low-passed trial: inverting them reproduces the <= max_hz signal
    ff = FourierFormer(1536, 120, 32)
    z = _zscore(x)
    spec = torch.fft.rfft(z, dim=-1)
    low = spec.clone(); low[..., ff.n_bins:] = 0
    assert torch.allclose(torch.fft.irfft(low, n=120), torch.fft.irfft(spec * (torch.arange(61) < ff.n_bins), n=120))
    print('frequency tokens ok:', ff.n_bins, 'bins')
    # phase input: unit magnitude, and a pure 6 Hz cosine has theta phase = its own phase (cos part = the signal / its peak)
    t = torch.arange(120) / 120.0
    ph = _band_phase(torch.cos(2 * math.pi * 6 * t).repeat(1, 1, 1), ((4, 8),), 120.0)
    assert torch.allclose(ph[0, 0], torch.cos(2 * math.pi * 6 * t), atol=1e-4) and torch.allclose(ph[0, 0] ** 2 + ph[0, 1] ** 2, torch.ones(120), atol=1e-4)
    print('phase input ok')
