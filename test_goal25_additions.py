"""Self-checks for the inter-subject additions (run: .venv/bin/python test_goal25_additions.py).

Covers the pieces with real logic in them: the new objectives, the causal operator transfer, the
shared-encoder wiring, and the concept-level sampler/mask.
"""
import collections
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from module.dataset import EEGPreImageDataset
from module.loss import ContrastiveLoss
from module.sampler import GroupedImageBatchSampler
from module.eeg_encoder.model import (
    DomainBatchNorm2d, SubjectBatchNorm2d, TSConv_parameterizable,
)
from train import (SharedSpaceProjector, build_image_positive_mask, build_projector,
                   build_shared_encoder, collect_trainable_parameters,
                   compute_cohort_reconstruction_loss,
                   compute_mutual_reconstruction_loss,
                   cross_subject_operator_transfer, cross_subject_stimulus_mix,
                   exact_stimulus_cross_subject_partners,
                   exact_stimulus_cohort_targets,
                   select_hard_counterfactual_views,
                   select_rows_per_exact_image,
                   set_requires_grad)
from module.subject_augmentation import (
    coherent_subject_time_shift,
    cross_subject_spectral_style_mix,
    sample_coherent_subject_time_shifts,
)
from evaluate import _zero_padded_time_shift, canonicalize_query_latency


def test_objectives():
    torch.manual_seed(0)
    crit = ContrastiveLoss(0.07, 1.0, 1.0, eeg_l2norm=False, img_l2norm=True, text_l2norm=False,
                           learnable=False, is_softplus=True)
    n_img, n_sub, dim = 4, 3, 8
    obj = torch.arange(n_img).repeat_interleave(n_sub)
    img_idx = torch.zeros_like(obj)
    pos = build_image_positive_mask(obj, img_idx)

    image_feature = F.normalize(torch.randn(n_img, dim), dim=1).repeat_interleave(n_sub, dim=0)
    eeg_random = F.normalize(torch.randn(n_img * n_sub, dim), dim=1)

    # Same-stimulus agreement must beat random, and at a sharp temperature hit the log(k) floor
    # of the multi-positive CE (k = positives per row after dropping the self-pair).
    assert crit.brain_to_brain_loss(eeg_random, pos) > crit.brain_to_brain_loss(image_feature, pos) + 0.5
    sharp = ContrastiveLoss(0.01, 1.0, 1.0, False, True, False, False, False)
    assert abs(sharp.brain_to_brain_loss(image_feature, pos).item() - math.log(n_sub - 1)) < 0.01

    # Relational loss is 0 for identical structures -- and must not go NaN on the masked diagonal.
    same = crit.relational_loss(image_feature, image_feature)
    assert torch.isfinite(same) and same < 1e-5, same
    assert crit.relational_loss(eeg_random, image_feature) > 1e-3

    # Row losses average to the query->key half of the batch loss (used for per-subject V-REx risks).
    rows = crit.multi_positive_row_losses(eeg_random, image_feature, pos)
    qk = crit._multi_positive_cross_entropy(
        torch.matmul(eeg_random, F.normalize(image_feature, dim=1).T) * crit._get_logit_scale(), pos)
    assert torch.allclose(rows.mean(), qk, atol=1e-5)

    # MMD: identical clouds -> 0, shifted -> positive.
    x = torch.randn(64, 32)
    assert crit.mk_mmd(x, x.clone()).abs() < 1e-6
    assert crit.mk_mmd(x, torch.randn(64, 32) + 5.0) > 1e-3


def test_operator_transfer():
    torch.manual_seed(0)
    C, T, n = 8, 200, 60
    sources = torch.randn(n, C, T)
    x = torch.cat([torch.einsum('cd,ndt->nct', torch.randn(C, C), sources) for _ in range(3)])
    sid = torch.cat([torch.full((n,), s) for s in range(3)])

    def cov(z):
        f = z.reshape(-1, C, T).double()
        return torch.einsum('nct,ndt->cd', f, f) / (f.shape[0] * T)

    targets = {s: cov(x[sid == s]) for s in range(3)}
    out = cross_subject_operator_transfer(x, sid, shrink=0.0)
    moved = 0
    for s in range(3):
        c = cov(out[sid == s])
        dists = {t: torch.norm(c / c.norm() - targets[t] / targets[t].norm()).item() for t in range(3)}
        nearest = min(dists, key=dists.get)
        assert dists[nearest] < 1e-3, (s, dists)   # lands on a real subject's covariance
        moved += nearest != s
    assert moved >= 1

    one = sid == 0
    assert torch.equal(cross_subject_operator_transfer(x[one], sid[one]), x[one])   # no-op alone
    torch.manual_seed(7)
    assert torch.allclose(cross_subject_operator_transfer(x, sid, 0.0, extrapolate=1e-9), x, atol=1e-3)
    torch.manual_seed(3)
    far = cov(cross_subject_operator_transfer(x, sid, 0.0, extrapolate=4.0)[sid == 0])
    assert min(torch.norm(far / far.norm() - t / t.norm()).item() for t in targets.values()) > 1e-4


def test_counterfactual_row_reduction():
    objects = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
    images = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0])
    one = select_rows_per_exact_image(objects, images, 1)
    two = select_rows_per_exact_image(objects, images, 2)
    assert one.tolist() == [0, 3, 4, 7]
    assert two.tolist() == [0, 1, 3, 4, 5, 7, 8]
    assert torch.equal(objects[one], torch.tensor([0, 0, 1, 2]))

    eeg = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [0.0, 1.0], [1.0, 0.0],
    ])
    image = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [0.0, 1.0],
    ])
    hard = select_hard_counterfactual_views(eeg, image, num_views=2)
    assert hard.tolist() == [2, 3]


def test_group_mix_self_anchor():
    features = torch.tensor([[1.0], [3.0], [8.0]])
    objects = torch.zeros(3, dtype=torch.long)
    images = torch.zeros(3, dtype=torch.long)
    subjects = torch.arange(3)
    torch.manual_seed(31)
    free = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5, mixup_type='group'
    )
    torch.manual_seed(31)
    anchored = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5, mixup_type='group',
        group_self_anchor=0.4,
    )
    assert torch.allclose(anchored, 0.4 * features + 0.6 * free)

    separated_features = torch.tensor([[0.0], [2.0], [100.0], [102.0]])
    separated_subjects = torch.tensor([1, 2, 101, 102])
    partition = (separated_subjects >= 100).long()
    partitioned = cross_subject_stimulus_mix(
        separated_features, torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long), separated_subjects,
        alpha=0.5, mixup_type='group', partition_labels=partition,
    )
    assert torch.all((partitioned[:2] >= 0) & (partitioned[:2] <= 2))
    assert torch.all((partitioned[2:] >= 100) & (partitioned[2:] <= 102))

    # A pseudo-held row must be invariant to arbitrary changes in its own source
    # observation because its Dirichlet diagonal is exactly zero.
    base = torch.tensor([[1.0], [3.0], [8.0]])
    changed = base.clone(); changed[0] = 1000.0
    torch.manual_seed(91)
    loo_base = cross_subject_stimulus_mix(
        base, objects, images, subjects, alpha=0.5,
        mixup_type='leave_one_out_group',
    )
    torch.manual_seed(91)
    loo_changed = cross_subject_stimulus_mix(
        changed, objects, images, subjects, alpha=0.5,
        mixup_type='leave_one_out_group',
    )
    assert torch.equal(loo_base[0], loo_changed[0])
    assert 3.0 <= loo_base[0].item() <= 8.0


def test_real_or_virtual_row_blend_equation():
    # This is the exact shape/broadcast rule used by --subject_mixup_prob.
    original = torch.zeros(4, 2, 3)
    synthetic = torch.ones_like(original)
    use_synthetic = torch.tensor([True, False, False, True])
    view_shape = (synthetic.shape[0],) + (1,) * (synthetic.ndim - 1)
    blended = torch.where(use_synthetic.view(view_shape), synthetic, original)
    assert torch.equal(blended[:, 0, 0], use_synthetic.float())
    assert torch.equal(blended[0], synthetic[0])
    assert torch.equal(blended[1], original[1])


def test_time_varying_subject_mix():
    batch, channels, time_points = 18, 5, 80
    objects = torch.arange(batch) // 9
    images = torch.zeros(batch, dtype=torch.long)
    subjects = torch.arange(batch) % 9
    torch.manual_seed(0)
    random_features = torch.randn(batch, channels, time_points)

    # Disabled and one-control-point modes retain the historical scalar path.
    torch.manual_seed(11)
    random.seed(11)
    scalar = cross_subject_stimulus_mix(
        random_features, objects, images, subjects, alpha=0.5,
        mixup_type='pairwise', time_varying=0,
    )
    torch.manual_seed(11)
    random.seed(11)
    one = cross_subject_stimulus_mix(
        random_features, objects, images, subjects, alpha=0.5,
        mixup_type='pairwise', time_varying=1,
    )
    assert torch.equal(scalar, one)

    torch.manual_seed(11)
    random.seed(11)
    smooth = cross_subject_stimulus_mix(
        random_features, objects, images, subjects, alpha=0.5,
        mixup_type='pairwise', time_varying=3,
    )
    assert not torch.allclose(scalar, smooth)
    # At every time and sensor it remains inside the source group's convex range.
    for group in range(2):
        group_slice = slice(group * 9, (group + 1) * 9)
        lower = random_features[group_slice].min(0).values
        upper = random_features[group_slice].max(0).values
        assert torch.all(smooth[group_slice] >= lower - 1e-5)
        assert torch.all(smooth[group_slice] <= upper + 1e-5)

    torch.manual_seed(19)
    smooth_group = cross_subject_stimulus_mix(
        random_features, objects, images, subjects, alpha=0.5,
        mixup_type='group', time_varying=3,
    )
    assert not torch.allclose(smooth, smooth_group)
    for group in range(2):
        group_slice = slice(group * 9, (group + 1) * 9)
        lower = random_features[group_slice].min(0).values
        upper = random_features[group_slice].max(0).values
        assert torch.all(smooth_group[group_slice] >= lower - 1e-5)
        assert torch.all(smooth_group[group_slice] <= upper + 1e-5)


def test_hybrid_group_pairwise_mix():
    torch.manual_seed(31)
    features = torch.randn(18, 4, 25)
    objects = torch.tensor([0] * 9 + [1] * 9)
    images = torch.zeros(18, dtype=torch.long)
    subjects = torch.arange(9).repeat(2)

    torch.manual_seed(7)
    random.seed(7)
    expected_group = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5, mixup_type='group'
    )
    torch.manual_seed(7)
    random.seed(7)
    all_group = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5,
        mixup_type='hybrid_group_pairwise', time_varying=3,
        hybrid_group_prob=1.0,
    )
    assert torch.equal(all_group, expected_group)

    torch.manual_seed(11)
    random.seed(11)
    _ = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5, mixup_type='group'
    )
    expected_pair = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5,
        mixup_type='pairwise', time_varying=3,
    )
    torch.manual_seed(11)
    random.seed(11)
    all_pair = cross_subject_stimulus_mix(
        features, objects, images, subjects, alpha=0.5,
        mixup_type='hybrid_group_pairwise', time_varying=3,
        hybrid_group_prob=0.0,
    )
    assert torch.equal(all_pair, expected_pair)


def test_hierarchical_repetition_bootstrap():
    dataset = EEGPreImageDataset.__new__(EEGPreImageDataset)
    dataset.bootstrap_repetition_average = True
    dataset.bootstrap_repetition_count = 4
    dataset.cross_subject_average = False
    dataset.subject_mixup_within = False
    dataset.average = False
    dataset.random = False
    dataset.num_objects = 1
    dataset.num_images_per_object = 1
    dataset.num_subjects = 2
    dataset.subject_ids = [11, 12]
    dataset.eeg_data_list = [
        np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1, 1),
        (10 + np.arange(4, dtype=np.float32)).reshape(1, 1, 4, 1, 1),
    ]
    dataset.image_features = np.ones((1, 1, 3), dtype=np.float32)
    dataset.text_feature_dir = ''
    dataset.feature_dim = 3

    assert len(dataset) == 2
    assert dataset.decode_index(0)[:3] == (11, 0, 0)
    assert dataset.decode_index(1)[:3] == (12, 0, 0)
    np.random.seed(9)
    draws = torch.stack([dataset[0][0] for _ in range(2000)]).numpy().ravel()
    # Bootstrap averages vary, but remain unbiased for the real-subject mean.
    assert np.std(draws) > 0.2
    assert abs(float(np.mean(draws)) - 1.5) < 0.05


def test_spectral_style_mix_preserves_phase_and_is_differentiable():
    torch.manual_seed(17)
    time = torch.linspace(0, 2 * math.pi, 64)
    features = torch.stack([
        torch.stack([torch.sin(time), 2.0 * torch.cos(2 * time)]),
        torch.stack([3.0 * torch.sin(time), 0.5 * torch.cos(2 * time)]),
        torch.stack([0.5 * torch.sin(time), 4.0 * torch.cos(2 * time)]),
    ]).requires_grad_()
    objects = torch.zeros(3, dtype=torch.long)
    images = torch.zeros(3, dtype=torch.long)
    subjects = torch.arange(3)
    mixed = cross_subject_spectral_style_mix(
        features, objects, images, subjects, alpha=0.5
    )
    original_fft = torch.fft.rfft(features.detach(), dim=-1)
    mixed_fft = torch.fft.rfft(mixed.detach(), dim=-1)
    active = torch.abs(original_fft) > 1e-4
    original_phase = original_fft / torch.abs(original_fft).clamp_min(1e-6)
    mixed_phase = mixed_fft / torch.abs(mixed_fft).clamp_min(1e-6)
    assert torch.max(torch.abs(mixed_phase[active] - original_phase[active])) < 2e-3
    assert not torch.allclose(torch.abs(mixed_fft), torch.abs(original_fft))
    mixed.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_coherent_subject_time_shift():
    features = torch.arange(18, dtype=torch.float32).reshape(3, 1, 6).requires_grad_()
    subject_ids = torch.tensor([7, 8, 7])
    shifted = coherent_subject_time_shift(
        features,
        subject_ids,
        torch.tensor([7, 8]),
        torch.tensor([2, -1]),
    )
    assert torch.equal(shifted[0, 0], torch.tensor([0., 0., 0., 1., 2., 3.]))
    assert torch.equal(shifted[1, 0], torch.tensor([7., 8., 9., 10., 11., 0.]))
    assert torch.equal(shifted[2, 0], torch.tensor([0., 0., 12., 13., 14., 15.]))
    shifted.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()

    torch.manual_seed(4)
    subjects, shifts = sample_coherent_subject_time_shifts(
        [2, 4, 9], max_shift=4, std=2.0, device='cpu'
    )
    assert torch.equal(subjects, torch.tensor([2, 4, 9]))
    assert torch.all((shifts >= -4) & (shifts <= 4))


def test_single_query_latency_canonicalization():
    torch.manual_seed(21)
    template = torch.randn(3, 40)
    delayed = _zero_padded_time_shift(template.unsqueeze(0), torch.tensor([3]))
    aligned, shifts = canonicalize_query_latency(delayed, template, max_shift=5)
    assert shifts.item() == -3
    assert torch.allclose(aligned[..., :-3], template.unsqueeze(0)[..., :-3])


def test_domain_batch_norm():
    torch.manual_seed(3)
    layer = DomainBatchNorm2d(3, boundary=100)
    layer.train()
    x = torch.cat([torch.randn(4, 3, 2, 5), 8.0 + torch.randn(4, 3, 2, 5)])
    subject_ids = torch.tensor([1, 2, 3, 4, 101, 102, 103, 104])
    out = layer(x, subject_ids)
    assert torch.allclose(out[:4], layer.primary(x[:4]), atol=1e-5)
    assert torch.allclose(out[4:], layer.auxiliary(x[4:]), atol=1e-5)
    assert not torch.allclose(layer.primary.running_mean, layer.auxiliary.running_mean)

    model = TSConv_parameterizable(
        feature_dim=16, eeg_sample_points=100, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=9, pool_kernel=15, pool_stride=3,
        domain_batch_norm_boundary=100,
    )
    encoded = model(torch.randn(6, 8, 100), subject_ids=torch.tensor([1, 2, 3, 101, 102, 103]))
    assert encoded.shape == (6, 16)


def test_subject_random_effect_adapter_population_path():
    torch.manual_seed(13)
    model = TSConv_parameterizable(
        feature_dim=12, eeg_sample_points=100, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=9, pool_kernel=15, pool_stride=3,
        dropout=0.0, head_dropout=0.0,
        subject_adapter_rank=2, subject_adapter_max_id=10,
        subject_adapter_scale=0.2,
    )
    with torch.no_grad():
        model.subject_adapter_projection.weight.normal_(std=0.1)
    trial = torch.randn(1, 8, 100).repeat(2, 1, 1)
    model.train()
    train_out = model(trial, subject_ids=torch.tensor([1, 0]))
    assert not torch.allclose(train_out[0], train_out[1])
    train_out.square().mean().backward()
    assert model.subject_adapter_projection.weight.grad is not None

    model.eval()
    with torch.no_grad():
        held_out = model(trial, subject_ids=torch.tensor([1, 9]))
        population = model(trial, subject_ids=torch.tensor([0, 0]))
    assert torch.allclose(held_out, population)

    population_model = TSConv_parameterizable(
        feature_dim=12, eeg_sample_points=100, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=9, pool_kernel=15, pool_stride=3,
        dropout=0.0, head_dropout=0.0,
        subject_adapter_rank=2, subject_adapter_max_id=10,
        subject_adapter_virtual_prob=0.0,
        subject_adapter_source_ids=[1, 2, 3],
    )
    population_model.train()
    with torch.no_grad():
        population_model.subject_adapter_projection.weight.normal_(std=0.1)
    repeated = torch.randn(1, 8, 100).repeat(4, 1, 1)
    negative_population = population_model(
        repeated, subject_ids=torch.tensor([-1, -1, -2, -2])
    )
    zero_population = population_model(
        repeated, subject_ids=torch.zeros(4, dtype=torch.long)
    )
    assert torch.allclose(negative_population, zero_population)

    virtual_model = TSConv_parameterizable(
        feature_dim=12, eeg_sample_points=100, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=9, pool_kernel=15, pool_stride=3,
        dropout=0.0, head_dropout=0.0,
        subject_adapter_rank=2, subject_adapter_max_id=10,
        subject_adapter_virtual_prob=1.0,
        subject_adapter_source_ids=[1, 2, 3],
    )
    virtual_model.train()
    with torch.no_grad():
        virtual_model.subject_adapter_embedding.weight[1:4].copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        )
        virtual_model.subject_adapter_projection.weight.normal_(std=0.1)
    torch.manual_seed(41)
    virtual = virtual_model(
        repeated, subject_ids=torch.tensor([-1, -1, -2, -2])
    )
    assert torch.allclose(virtual[0], virtual[1])
    assert torch.allclose(virtual[2], virtual[3])
    assert not torch.allclose(virtual[0], virtual[2])


def test_inferred_subject_adapter_unseen_path():
    torch.manual_seed(47)
    model = TSConv_parameterizable(
        feature_dim=12, eeg_sample_points=100, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=9, pool_kernel=15, pool_stride=3,
        dropout=0.0, head_dropout=0.0,
        subject_adapter_rank=2, subject_adapter_max_id=10,
        subject_adapter_scale=0.2, subject_adapter_inferred=True,
    )
    with torch.no_grad():
        model.subject_adapter_projection.weight.normal_(std=0.1)
    trials = torch.stack([
        torch.randn(8, 100),
        3.0 * torch.randn(8, 100) + 2.0,
        torch.randn(8, 100),
    ])
    model.train()
    encoded = model(trials, subject_ids=torch.tensor([1, 2, 0]))
    router_loss, rows = model.subject_adapter_router_loss(torch.tensor([1, 2, 0]))
    assert encoded.shape == (3, 12) and rows == 2 and torch.isfinite(router_loss)
    assert not torch.allclose(
        model._subject_adapter_router_codes[0],
        model._subject_adapter_router_codes[1],
    )
    (encoded.square().mean() + router_loss).backward()
    assert model.subject_adapter_code_head.weight.grad is not None
    assert model.subject_adapter_class_head.weight.grad is not None

    # Evaluation still infers from the observation; numeric held-person IDs do
    # not select an untrained table row.
    model.eval()
    with torch.no_grad():
        held_a = model(trials[:1], subject_ids=torch.tensor([7]))
        held_b = model(trials[:1], subject_ids=torch.tensor([10]))
    assert torch.allclose(held_a, held_b)


def test_subject_batch_norm_population_routing():
    torch.manual_seed(43)
    layer = SubjectBatchNorm2d(2, max_subject_id=3)
    layer.train()
    features = torch.cat([
        torch.randn(4, 2, 2, 5),
        5.0 + torch.randn(4, 2, 2, 5),
        10.0 + torch.randn(4, 2, 2, 5),
    ])
    ids = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    output = layer(features, ids)
    assert torch.allclose(output[:4], layer.paths[0](features[:4]), atol=1e-5)
    assert not torch.allclose(layer.paths[0].running_mean, layer.paths[1].running_mean)
    layer.eval()
    with torch.no_grad():
        routed = layer(features, ids)
        population = layer.paths[0](features)
    assert torch.allclose(routed, population)

    model = TSConv_parameterizable(
        feature_dim=12, eeg_sample_points=100, channels_num=8,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=9, pool_kernel=15, pool_stride=3,
        subject_batch_norm_max_id=3,
    )
    model.train()
    encoded = model(
        torch.randn(6, 8, 100),
        subject_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    assert encoded.shape == (6, 12)


def test_exact_stimulus_mutual_reconstruction():
    torch.manual_seed(29)
    model = TSConv_parameterizable(
        feature_dim=12, eeg_sample_points=40, channels_num=3,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=5, pool_kernel=9, pool_stride=2,
        dropout=0.0, head_dropout=0.0,
        subject_adapter_rank=2, subject_adapter_max_id=5,
        mutual_reconstruction_hidden=8,
    )
    content = torch.randn(6, 12, requires_grad=True)
    eeg = torch.randn(6, 3, 40)
    objects = torch.tensor([0, 0, 0, 1, 1, 1])
    images = torch.zeros(6, dtype=torch.long)
    subjects = torch.tensor([1, 2, 3, 1, 2, 3])
    eligible = torch.tensor([True, True, True, True, False, False])
    partners, valid = exact_stimulus_cross_subject_partners(
        objects, images, subjects, eligible
    )
    assert valid.tolist() == [True, True, True, False, False, False]
    assert torch.all(subjects[partners[valid]] != subjects[valid])
    assert torch.all(objects[partners[valid]] == objects[valid])

    torch.manual_seed(29)
    loss, rows = compute_mutual_reconstruction_loss(
        model, content, eeg, objects, images, subjects, eligible
    )
    assert rows == 3 and torch.isfinite(loss)
    loss.backward()
    assert content.grad is not None
    assert model.mutual_reconstruction_decoder[-1].weight.grad is not None


def test_exact_stimulus_cohort_reconstruction():
    torch.manual_seed(37)
    model = TSConv_parameterizable(
        feature_dim=10, eeg_sample_points=20, channels_num=2,
        temporal_filters=4, spatial_filters=4, projection_filters=4,
        temporal_kernel=5, pool_kernel=7, pool_stride=2,
        dropout=0.0, head_dropout=0.0,
        cohort_reconstruction_hidden=8,
    )
    content = torch.randn(6, 10, requires_grad=True)
    eeg = torch.arange(240, dtype=torch.float32).reshape(6, 2, 20)
    objects = torch.tensor([0, 0, 0, 1, 1, 1])
    images = torch.zeros(6, dtype=torch.long)
    subjects = torch.tensor([1, 2, 3, 1, 2, 3])
    targets, valid = exact_stimulus_cohort_targets(
        eeg, objects, images, subjects
    )
    assert valid.all()
    assert torch.equal(targets[0], eeg[:3].mean(dim=0))
    assert torch.equal(targets[2], targets[0])
    assert torch.equal(targets[3], eeg[3:].mean(dim=0))
    loss, rows = compute_cohort_reconstruction_loss(
        model, content, eeg, objects, images, subjects
    )
    assert rows == 6 and torch.isfinite(loss)
    loss.backward()
    assert content.grad is not None
    assert model.cohort_reconstruction_decoder[-1].weight.grad is not None


def test_shared_encoder():
    shared = build_shared_encoder(16, 32)
    pe = SharedSpaceProjector(build_projector('linear', 8, 16), shared)
    pi = SharedSpaceProjector(build_projector('linear', 12, 16), shared)
    assert pe(torch.randn(4, 8)).shape == (4, 16) and pi(torch.randn(4, 12)).shape == (4, 16)
    assert pe.shared is pi.shared
    assert len(collect_trainable_parameters([pe, pi])) == 8      # shared params counted once
    set_requires_grad(shared, False)
    assert len(collect_trainable_parameters([pe, pi])) == 4      # fine stage freezes only g()


def test_concept_sampler_and_mask():
    class FakeDS:
        def get_image_group_indices(self):
            g, i = {}, 0
            for o in range(4):
                for im in range(5):
                    g[(o, im)] = list(range(i, i + 9)); i += 9
            return g

    ds = FakeDS()
    meta = {idx: k for k, v in ds.get_image_group_indices().items() for idx in v}
    batches = list(GroupedImageBatchSampler(ds, 27, samples_per_image=9, seed=0, images_per_concept=3))
    assert any(max(collections.Counter(meta[i][0] for i in b).values()) >= 18 for b in batches)
    visited_groups = {
        meta[index] for batch in batches for index in batch
    }
    # drop_last can remove only a final partial batch; it must not silently retain
    # just the first images_per_concept images from every concept.
    assert len(visited_groups) >= 18

    obj = torch.tensor([meta[i][0] for i in batches[0]])
    img = torch.tensor([meta[i][1] for i in batches[0]])
    m_img = build_image_positive_mask(obj, img, False)
    m_con = build_image_positive_mask(obj, img, True)
    assert (m_con | m_img).equal(m_con) and m_con.sum() > m_img.sum() and m_con.diagonal().all()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
    print('all self-checks passed')
