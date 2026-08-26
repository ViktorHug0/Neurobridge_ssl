"""Self-checks for virtual-subject feature-statistics augmentation."""

import torch

from module.eeg_encoder.model import TSConv_parameterizable
from module.subject_augmentation import subject_coherent_mixstyle, subject_style_manifold
from module.subject_augmentation import (
    coherent_cross_subject_stimulus_mix,
    cross_dataset_residual_subject_mix,
    sample_coherent_subject_weights,
    translate_auxiliary_subjects,
)


def test_coherent_subject_weights_are_convex_and_anchored():
    torch.manual_seed(2)
    subjects, weights = sample_coherent_subject_weights([4, 2, 9], alpha=0.5, self_anchor=0.4)
    assert subjects.tolist() == [2, 4, 9]
    assert torch.allclose(weights.sum(dim=1), torch.ones(3))
    assert torch.all(weights >= 0)
    assert torch.all(torch.diagonal(weights) >= 0.4)


def test_coherent_stimulus_mix_reuses_virtual_subjects():
    # Each real subject is a scalar basis value. Applying the same weight table to two
    # stimuli must preserve their shared virtual-subject offset exactly.
    subjects = torch.tensor([1, 2, 3])
    weights = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ])
    first = torch.tensor([[1.0], [2.0], [4.0]])
    second = first + 10.0
    features = torch.cat([first, second])
    subject_ids = subjects.repeat(2)
    objects = torch.tensor([0, 0, 0, 1, 1, 1])
    images = torch.zeros(6, dtype=torch.long)
    result = coherent_cross_subject_stimulus_mix(
        features, objects, images, subject_ids, subjects, weights
    )
    expected_first = weights @ first
    assert torch.allclose(result[:3], expected_first)
    assert torch.allclose(result[3:], expected_first + 10.0)

    extrapolated = coherent_cross_subject_stimulus_mix(
        features, objects, images, subject_ids, subjects, weights, residual_scale=2.0
    )
    centroid = first.mean(dim=0, keepdim=True)
    expected_extrapolated = centroid + 2.0 * (expected_first - centroid)
    assert torch.allclose(extrapolated[:3], expected_extrapolated)
    assert torch.allclose(extrapolated[3:], expected_extrapolated + 10.0)


def test_coherent_jitter_endpoints_and_convexity():
    torch.manual_seed(12)
    subjects = torch.tensor([1, 2, 3])
    cohort_weights = torch.eye(3)
    features = torch.tensor([[1.0], [3.0], [8.0]])
    object_indices = torch.zeros(3, dtype=torch.long)
    image_indices = torch.zeros(3, dtype=torch.long)
    stable = coherent_cross_subject_stimulus_mix(
        features, object_indices, image_indices, subjects,
        subjects, cohort_weights, jitter=0.0,
    )
    assert torch.equal(stable, features)
    random_endpoint = coherent_cross_subject_stimulus_mix(
        features, object_indices, image_indices, subjects,
        subjects, cohort_weights, jitter=1.0, jitter_alpha=0.5,
    )
    assert torch.all(random_endpoint >= features.min())
    assert torch.all(random_endpoint <= features.max())
    assert not torch.allclose(random_endpoint, features)


def test_cross_dataset_residual_mix_centers_auxiliary_protocol():
    torch.manual_seed(17)
    # Three primary rows and two auxiliary rows for one matched stimulus. Auxiliary
    # rows share a large protocol offset (100), which must cancel after centering.
    features = torch.tensor([[1.0], [3.0], [7.0], [99.0], [103.0]])
    subjects = torch.tensor([1, 2, 3, 101, 102])
    zeros = torch.zeros(5, dtype=torch.long)
    torch.manual_seed(17)
    no_residual = cross_dataset_residual_subject_mix(
        features, zeros, zeros, subjects, partition_boundary=100,
        alpha=0.5, residual_scale=0.0,
    )
    torch.manual_seed(17)
    with_residual = cross_dataset_residual_subject_mix(
        features, zeros, zeros, subjects, partition_boundary=100,
        alpha=0.5, residual_scale=0.5,
    )
    # Every base is a convex primary-domain mixture; no auxiliary protocol offset leaks.
    assert torch.all((no_residual >= 1.0) & (no_residual <= 7.0))
    assert torch.allclose(with_residual[:3], no_residual[:3])
    delta = with_residual[3:] - no_residual[3:]
    assert torch.allclose(delta, torch.tensor([[-1.0], [1.0]]))
    assert torch.allclose(delta.mean(dim=0), torch.zeros(1))


def test_auxiliary_translation_preserves_primary_and_maps_residuals():
    features = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[10.0, 12.0], [20.0, 24.0]],
        [[14.0, 16.0], [28.0, 32.0]],
    ])
    subjects = torch.tensor([1, 101, 102])
    channel_map = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    source_mean = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    target_mean = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    translated = translate_auxiliary_subjects(
        features, subjects, 100, channel_map, source_mean, target_mean
    )
    assert torch.equal(translated[0], features[0])
    expected = target_mean.unsqueeze(0) + torch.einsum(
        'oi,bit->bot', channel_map, features[1:] - source_mean
    )
    assert torch.allclose(translated[1:], expected)
    halfway = translate_auxiliary_subjects(
        features, subjects, 100, channel_map, source_mean, target_mean, blend=0.5
    )
    assert torch.allclose(halfway[1:], 0.5 * (features[1:] + expected))


def test_subject_coherent_style_hits_sampled_statistics():
    torch.manual_seed(7)
    num_subjects, examples, filters, channels, time = 3, 8, 4, 5, 20
    shared = torch.randn(examples, filters, channels, time)
    batches = []
    subject_ids = []
    for subject in range(num_subjects):
        batches.append(shared * (subject + 1.0) + subject * 3.0)
        subject_ids.extend([subject] * examples)
    features = torch.cat(batches)
    subject_ids = torch.tensor(subject_ids)

    transformed, metadata = subject_coherent_mixstyle(
        features, subject_ids, alpha=0.5, return_metadata=True
    )
    assert torch.all(metadata['partners'] != torch.arange(num_subjects))
    for subject in range(num_subjects):
        block = transformed[subject_ids == subject]
        observed_mean = block.mean(dim=(0, 3), keepdim=True)
        observed_std = block.var(dim=(0, 3), unbiased=False, keepdim=True).sqrt()
        assert torch.allclose(observed_mean, metadata['target_mean'][subject:subject + 1], atol=1e-5)
        # The implementation adds eps before sqrt, so allow its tiny variance inflation.
        assert torch.allclose(observed_std, metadata['target_std'][subject:subject + 1], atol=2e-4)


def test_subject_coherent_style_is_differentiable_and_single_subject_noop():
    features = torch.randn(6, 3, 4, 10, requires_grad=True)
    subject_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    out = subject_coherent_mixstyle(features, subject_ids)
    out.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()

    one_subject = torch.randn(4, 3, 4, 10)
    result = subject_coherent_mixstyle(one_subject, torch.zeros(4, dtype=torch.long))
    assert result.data_ptr() == one_subject.data_ptr()


def test_low_rank_style_manifold_hits_sampled_statistics():
    torch.manual_seed(19)
    num_subjects, examples, filters, channels, time = 4, 7, 3, 5, 18
    shared = torch.randn(examples, filters, channels, time)
    features = torch.cat([
        shared * (0.8 + 0.3 * subject) + 0.4 * subject
        for subject in range(num_subjects)
    ])
    subject_ids = torch.arange(num_subjects).repeat_interleave(examples)
    transformed, metadata = subject_style_manifold(
        features, subject_ids, rank=2, latent_scale=0.8, return_metadata=True
    )
    assert metadata['rank'] == 2
    assert metadata['basis'].shape[0] == 2
    assert metadata['sampled_codes'].shape == (num_subjects, 2)
    for subject in range(num_subjects):
        block = transformed[subject_ids == subject]
        observed_mean = block.mean(dim=(0, 3), keepdim=True)
        observed_std = block.var(dim=(0, 3), unbiased=False, keepdim=True).sqrt()
        assert torch.allclose(
            observed_mean, metadata['target_mean'][subject:subject + 1], atol=1e-5
        )
        assert torch.allclose(
            observed_std, metadata['target_std'][subject:subject + 1], atol=2e-4
        )


def test_low_rank_style_manifold_is_differentiable_and_rank_bounded():
    torch.manual_seed(23)
    features = torch.randn(6, 2, 4, 12, requires_grad=True)
    subject_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    out, metadata = subject_style_manifold(
        features, subject_ids, rank=8, return_metadata=True
    )
    assert metadata['rank'] == 2
    out.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()

    one_subject = torch.randn(3, 2, 4, 12)
    result = subject_style_manifold(
        one_subject, torch.zeros(3, dtype=torch.long)
    )
    assert result.data_ptr() == one_subject.data_ptr()


def test_tsconv_temporal_hook():
    torch.manual_seed(3)
    model = TSConv_parameterizable(
        feature_dim=32, eeg_sample_points=250, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
    ).eval()
    eeg = torch.randn(5, 8, 250)
    baseline = model(eeg)
    shifted = model(eeg, temporal_augmentation=lambda features: features + 0.25)
    assert baseline.shape == shifted.shape == (5, 32)
    assert not torch.allclose(baseline, shifted)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
    print('all subject augmentation self-checks passed')
