"""Source-only, per-query covariance alignment for inductive EEG decoding."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _trace_normalize(covariance: torch.Tensor, eps: float) -> torch.Tensor:
    scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    return covariance / scale.clamp_min(eps)[..., None, None]


def _inverse_sqrt_newton_schulz(matrix: torch.Tensor, iterations: int = 8):
    """Batched symmetric inverse square root using only fast matrix multiplies."""
    identity = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).expand_as(matrix)
    norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    estimate = matrix / norm
    inverse = identity
    for _ in range(iterations):
        update = 0.5 * (3.0 * identity - inverse @ estimate)
        estimate = estimate @ update
        inverse = update @ inverse
    return inverse / norm.sqrt()


@torch.no_grad()
def estimate_source_covariances(
    eeg_data_list,
    subject_ids,
    device: torch.device,
    max_trials_per_subject: int = 2048,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> dict[int, torch.Tensor]:
    """Estimate equal-trial, trace-normalized covariance per source subject."""
    references = {}
    for subject_id, data in zip(subject_ids, eeg_data_list):
        flat = data.reshape(-1, data.shape[-2], data.shape[-1])
        count = min(len(flat), int(max_trials_per_subject))
        indices = np.linspace(0, len(flat) - 1, count, dtype=np.int64)
        covariance_sum = torch.zeros(
            flat.shape[-2], flat.shape[-2], device=device, dtype=torch.float64
        )
        for start in range(0, count, chunk_size):
            sample = torch.as_tensor(
                np.asarray(flat[indices[start:start + chunk_size]]),
                device=device,
                dtype=torch.float32,
            )
            sample = sample - sample.mean(dim=-1, keepdim=True)
            covariance = sample @ sample.transpose(-1, -2) / sample.shape[-1]
            covariance_sum += _trace_normalize(covariance, eps).double().sum(dim=0)
        references[int(subject_id)] = (covariance_sum / count).float()
    return references


class InductiveQueryCovarianceAlign(nn.Module):
    """Whiten each query from a frozen source prior blended with its own covariance.

    Known source subjects use the mean reference of the other source subjects.
    Any unseen subject uses the mean reference of every source subject. No state is
    updated in ``forward`` and no statistic is shared between queries.
    """

    def __init__(
        self,
        source_references: dict[int, torch.Tensor],
        alpha: float,
        shrinkage: float,
        eps: float = 1e-5,
    ):
        super().__init__()
        if not source_references:
            raise ValueError("source_references cannot be empty")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("shrinkage must be in [0, 1)")

        source_ids = sorted(source_references)
        stacked = torch.stack([source_references[s] for s in source_ids])
        global_reference = stacked.mean(dim=0)
        max_subject_id = max(source_ids)
        lookup = global_reference.repeat(max_subject_id + 1, 1, 1)
        known = torch.zeros(max_subject_id + 1, dtype=torch.bool, device=stacked.device)
        if len(source_ids) > 1:
            total = stacked.sum(dim=0)
            for index, subject_id in enumerate(source_ids):
                lookup[subject_id] = (total - stacked[index]) / (len(source_ids) - 1)
                known[subject_id] = True

        self.alpha = float(alpha)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.register_buffer("global_reference", global_reference)
        self.register_buffer("source_reference_lookup", lookup)
        self.register_buffer("known_source", known)

    def _reference_for(self, subject_ids: torch.Tensor) -> torch.Tensor:
        subject_ids = subject_ids.reshape(-1).long()
        reference = self.global_reference.expand(subject_ids.shape[0], -1, -1)
        in_range = (subject_ids >= 0) & (subject_ids < self.known_source.numel())
        use_source = in_range.clone()
        use_source[in_range] &= self.known_source[subject_ids[in_range]]
        if use_source.any():
            reference = reference.clone()
            reference[use_source] = self.source_reference_lookup[subject_ids[use_source]]
        return reference

    def forward(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        original_dtype = eeg.dtype
        centered = eeg.float() - eeg.float().mean(dim=-1, keepdim=True)
        with torch.no_grad():
            covariance = centered @ centered.transpose(-1, -2) / centered.shape[-1]
            covariance = _trace_normalize(covariance, self.eps)
            reference = self._reference_for(subject_ids).float()
            blended = (1.0 - self.alpha) * reference + self.alpha * covariance
            identity = torch.eye(
                blended.shape[-1], device=blended.device, dtype=blended.dtype
            )
            blended = (
                (1.0 - self.shrinkage) * blended + self.shrinkage * identity
            )
            inverse_sqrt = _inverse_sqrt_newton_schulz(blended)
        return (inverse_sqrt @ centered).to(original_dtype)
