"""Train-only stochastic views for full-data EEG ensemble members."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_spectral_gain(
    eeg: torch.Tensor,
    standard_deviation: float,
    control_points: int = 8,
) -> torch.Tensor:
    """Apply an independent smooth log-spectral gain per trial and channel."""
    if standard_deviation == 0:
        return eeg
    if standard_deviation < 0:
        raise ValueError("standard_deviation must be non-negative")
    if control_points < 2:
        raise ValueError("control_points must be at least 2")

    original_dtype = eeg.dtype
    spectrum = torch.fft.rfft(eeg.float(), dim=-1)
    controls = torch.randn(
        *eeg.shape[:-1],
        control_points,
        device=eeg.device,
        dtype=spectrum.real.dtype,
    ) * standard_deviation
    log_gain = F.interpolate(
        controls,
        size=spectrum.shape[-1],
        mode="linear",
        align_corners=True,
    )
    augmented = torch.fft.irfft(spectrum * log_gain.exp(), n=eeg.shape[-1], dim=-1)
    return augmented.to(original_dtype)


def channel_dropout(eeg: torch.Tensor, probability: float) -> torch.Tensor:
    """Drop channels independently for every trial without test-time rescaling."""
    if not 0 <= probability < 1:
        raise ValueError("probability must be in [0, 1)")
    if probability == 0:
        return eeg

    keep = torch.rand(
        eeg.shape[0], eeg.shape[1], 1, device=eeg.device
    ) >= probability
    empty = ~keep.any(dim=1, keepdim=True)
    if torch.any(empty):
        keep[:, :1] |= empty
    return eeg * keep.to(eeg.dtype)


def stochastic_member_view(
    eeg: torch.Tensor,
    spectral_gain_sd: float,
    spectral_control_points: int,
    channel_drop_probability: float,
) -> torch.Tensor:
    """Draw one member-specific view; repeated calls are independent."""
    view = smooth_spectral_gain(eeg, spectral_gain_sd, spectral_control_points)
    return channel_dropout(view, channel_drop_probability)
