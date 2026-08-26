"""Cheap CPU checks for both custom losses and the twin branch wiring."""

from types import SimpleNamespace

import torch

from ensemble_experiments.decorrelated_models.losses import (
    deployed_ensemble_contrastive_loss,
    ensemble_contrastive_loss,
    first_unique_image_columns,
    negative_score_correlation_loss,
    soft_multiple_choice_rescue_loss,
)
from ensemble_experiments.decorrelated_models.train_twins import TwinTSConvBranch


def test_losses() -> None:
    objects = torch.tensor([0, 0, 1, 1, 2, 2])
    images = torch.tensor([0, 0, 0, 0, 0, 0])
    columns = first_unique_image_columns(objects, images)
    assert columns.tolist() == [0, 2, 4]

    objects = torch.arange(4)
    images = torch.zeros(4, dtype=torch.long)
    positives = torch.tensor([[True, False, False, False]])
    first = torch.tensor([[0.0, -1.0, 0.0, 1.0]], requires_grad=True)
    orthogonal = torch.tensor([[0.0, 1.0, -2.0, 1.0]], requires_grad=True)
    loss, correlation = negative_score_correlation_loss(
        first, orthogonal, positives, objects, images
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(correlation, torch.tensor(0.0), atol=1e-6)

    opposite = (-first.detach().clone()).requires_grad_(True)
    loss, correlation = negative_score_correlation_loss(
        first, opposite, positives, objects, images
    )
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(correlation, torch.tensor(-1.0), atol=1e-6)
    loss.backward()
    assert first.grad is not None and opposite.grad is not None

    row_a = torch.tensor([1.0, 3.0], requires_grad=True)
    row_b = torch.tensor([2.0, 1.0], requires_grad=True)
    rescue, responsibilities = soft_multiple_choice_rescue_loss(row_a, row_b, 0.5)
    assert responsibilities[0, 0] > responsibilities[0, 1]
    assert responsibilities[1, 1] > responsibilities[1, 0]
    assert not responsibilities.requires_grad
    rescue.backward()
    assert row_a.grad is not None and row_b.grad is not None


def test_branch_wiring() -> None:
    # Use reduced synthetic dimensions to keep this suitable for a login node;
    # the same parameterized TSConv code path is exercised.
    torch.set_num_threads(1)
    args = SimpleNamespace(backbone_dim=16, feature_dim=8)
    branch_a = TwinTSConvBranch(8, 100, 32, args)
    branch_b = TwinTSConvBranch(8, 100, 32, args)
    eeg = torch.randn(8, 8, 100)
    image = torch.randn(8, 32)
    objects = torch.arange(4).repeat_interleave(2)
    images = torch.zeros(8, dtype=torch.long)
    positives = objects[:, None].eq(objects[None, :])

    eeg_a, image_a = branch_a.features(eeg, image)
    eeg_b, image_b = branch_b.features(eeg, image)
    scores_a = branch_a.cosine_scores(eeg_a, image_a)
    scores_b = branch_b.cosine_scores(eeg_b, image_b)
    individual = branch_a.individual_loss(eeg_a, image_a, positives)
    row_a = branch_a.individual_row_losses(eeg_a, image_a, positives)
    row_b = branch_b.individual_row_losses(eeg_b, image_b, positives)
    rescue, _ = soft_multiple_choice_rescue_loss(row_a, row_b)
    ensemble, _ = ensemble_contrastive_loss(scores_a, scores_b, positives)
    deployed, fused = deployed_ensemble_contrastive_loss(
        scores_a, scores_b, positives, objects, images
    )
    diversity, _ = negative_score_correlation_loss(
        scores_a, scores_b, positives, objects, images
    )
    total = individual + ensemble + deployed + diversity + rescue
    total.backward()
    assert scores_a.shape == (8, 8)
    assert fused.shape == (8, 4)
    assert any(parameter.grad is not None for parameter in branch_a.parameters())
    assert any(parameter.grad is not None for parameter in branch_b.parameters())


if __name__ == "__main__":
    test_losses()
    test_branch_wiring()
    print("decorrelated twin-model smoke checks passed")
