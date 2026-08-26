"""Losses shared by the decorrelated twin-model trainer and its smoke tests."""

from __future__ import annotations

import torch


def row_z(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize every query row without detaching its mean or scale."""
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(eps)


def multi_positive_cross_entropy(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    """Mean log probability assigned to every positive in each valid row."""
    valid = positive_mask.any(dim=1)
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    log_probs = torch.log_softmax(logits[valid], dim=1)
    positives = positive_mask[valid]
    selected = torch.where(positives, log_probs, torch.zeros_like(log_probs))
    return -(selected.sum(dim=1) / positives.sum(dim=1).clamp_min(1)).mean()


def soft_multiple_choice_rescue_loss(
    row_losses_a: torch.Tensor,
    row_losses_b: torch.Tensor,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give extra gradient to the currently better member for each EEG query."""
    if row_losses_a.ndim != 1 or row_losses_a.shape != row_losses_b.shape:
        raise ValueError("rescue inputs must be matching per-query loss vectors")
    if temperature <= 0:
        raise ValueError("rescue temperature must be positive")
    losses = torch.stack((row_losses_a, row_losses_b), dim=1)
    responsibilities = torch.softmax(-losses.detach() / temperature, dim=1)
    return (responsibilities * losses).sum(dim=1).mean(), responsibilities


def ensemble_contrastive_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric multi-positive loss on the mean of two row-z score matrices."""
    fused_eeg_to_image = 0.5 * (row_z(scores_a) + row_z(scores_b))
    fused_image_to_eeg = 0.5 * (row_z(scores_a.T) + row_z(scores_b.T))
    loss = 0.5 * (
        multi_positive_cross_entropy(fused_eeg_to_image, positive_mask)
        + multi_positive_cross_entropy(fused_image_to_eeg, positive_mask.T)
    )
    return loss, fused_eeg_to_image


def deployed_ensemble_contrastive_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    positive_mask: torch.Tensor,
    object_indices: torch.Tensor,
    image_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EEG-to-image loss using the deployed row-z fusion on unique images.

    Training batches contain the same image once per sampled subject. Retrieval
    galleries contain every candidate image once, so duplicate columns must be
    removed before computing the row statistics used by the fusion rule.
    """
    columns = first_unique_image_columns(object_indices, image_indices)
    unique_a = scores_a[:, columns]
    unique_b = scores_b[:, columns]
    unique_positives = positive_mask[:, columns]
    fused = 0.5 * (row_z(unique_a) + row_z(unique_b))
    return multi_positive_cross_entropy(fused, unique_positives), fused


def first_unique_image_columns(
    object_indices: torch.Tensor, image_indices: torch.Tensor
) -> torch.Tensor:
    """Select one differentiable score column per unique (object, image) key."""
    if object_indices.ndim != 1 or image_indices.shape != object_indices.shape:
        raise ValueError("object_indices and image_indices must be matching vectors")
    key = object_indices.to(torch.int64) * 1_000_000 + image_indices.to(torch.int64)
    _, inverse = torch.unique(key, sorted=True, return_inverse=True)
    first = torch.full(
        (int(inverse.max().item()) + 1,),
        object_indices.numel(),
        dtype=torch.long,
        device=object_indices.device,
    )
    rows = torch.arange(object_indices.numel(), device=object_indices.device)
    first.scatter_reduce_(0, inverse, rows, reduce="amin", include_self=True)
    return first


def negative_score_correlation_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    positive_mask: torch.Tensor,
    object_indices: torch.Tensor,
    image_indices: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared per-query correlation after positives and duplicate images are removed."""
    if scores_a.shape != scores_b.shape or scores_a.shape != positive_mask.shape:
        raise ValueError("scores and positive_mask must have the same shape")
    columns = first_unique_image_columns(object_indices, image_indices)
    a = scores_a[:, columns]
    b = scores_b[:, columns]
    negatives = ~positive_mask[:, columns]
    count = negatives.sum(dim=1, keepdim=True)
    valid = count.squeeze(1) >= 2
    if not torch.any(valid):
        zero = scores_a.sum() * 0.0 + scores_b.sum() * 0.0
        return zero, zero.detach()

    weights = negatives.to(scores_a.dtype)
    safe_count = count.clamp_min(1).to(scores_a.dtype)
    centered_a = (a - (a * weights).sum(dim=1, keepdim=True) / safe_count) * weights
    centered_b = (b - (b * weights).sum(dim=1, keepdim=True) / safe_count) * weights
    covariance = (centered_a * centered_b).sum(dim=1) / safe_count.squeeze(1)
    variance_a = centered_a.square().sum(dim=1) / safe_count.squeeze(1)
    variance_b = centered_b.square().sum(dim=1) / safe_count.squeeze(1)
    correlation = covariance / (variance_a * variance_b).clamp_min(eps).sqrt()
    correlation = correlation[valid]
    return correlation.square().mean(), correlation.mean().detach()
