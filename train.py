import os
import argparse
import copy
import logging
import math
from datetime import datetime
import json
import random
import time
import sys
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import (
    EEGNet,
    EEGProject,
    EEGConformer,
    TSConv,
    TSConv_parameterizable,
    EEGTransformer,
    TSConv30,
)
from module.loss import ContrastiveLoss
from module.util import (
    _estimate_mu_cov,
    _inv_sqrt_cov,
    apply_orthogonal_map,
    csls_scores,
    fit_soft_assignment_procrustes,
    retrieve_all,
    sinkhorn_normalize,
)
from module.inductive_covariance import (
    InductiveQueryCovarianceAlign,
    estimate_source_covariances,
)
from module.projector import *
from module.sampler import GroupedImageBatchSampler, NeighbourBatchSampler, RepeatGroupBatchSampler
from module.training_plots import save_probe_plot, save_training_plot
from module.eeg_augmentation import RandomTimeShift, RandomGaussianNoise, RandomChannelDropout, RandomSmooth
from module.subject_augmentation import (
    coherent_cross_subject_stimulus_mix,
    cross_dataset_residual_subject_mix,
    sample_coherent_subject_weights,
    subject_coherent_mixstyle,
    subject_style_manifold,
    translate_auxiliary_subjects,
    cross_subject_spectral_style_mix,
    coherent_subject_time_shift,
    sample_coherent_subject_time_shifts,
)


def seed_everything(seed: int = None):
    if seed is None:
        seed = int(time.time()) % (2**32 - 1)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed_everything] All seeds set to: {seed}")
    return seed


def build_image_positive_mask(object_indices, image_indices, concept_positives=False):
    same_object = object_indices.unsqueeze(1) == object_indices.unsqueeze(0)
    if concept_positives:
        # The 200-way test set holds one image per concept, so image-level discrimination inside a
        # concept is never evaluated -- and EEG almost certainly cannot resolve it. Treating all
        # images of a concept as positives spends the capacity on the granularity that is scored.
        return same_object
    same_image = image_indices.unsqueeze(1) == image_indices.unsqueeze(0)
    return same_object & same_image


def _compute_cross_modal_loss_impl(
    criterion,
    eeg_feature,
    image_feature,
    text_feature,
    positive_mask,
    use_multi_positive,
    sample_weights=None,
):
    eeg_confidence = None
    if isinstance(eeg_feature, dict):
        eeg_confidence = eeg_feature.get('confidence')
        eeg_feature = eeg_feature['feature']
    if use_multi_positive:
        loss_contrastive_ie = criterion.multi_positive_pair_loss(
            eeg_feature,
            image_feature,
            positive_mask,
            query_scale=eeg_confidence,
            sample_weights=sample_weights,
        )
        if criterion.beta != 1.0:
            loss_contrastive_te = criterion.multi_positive_pair_loss(
                eeg_feature,
                text_feature,
                positive_mask,
                key_is_text=True,
                query_scale=eeg_confidence,
                sample_weights=sample_weights,
            )
            loss_contrastive = criterion.beta * loss_contrastive_ie + (1 - criterion.beta) * loss_contrastive_te
        else:
            loss_contrastive = loss_contrastive_ie

        if criterion.alpha != 1.0:
            if criterion.mse_on_raw:
                loss_mse = criterion.criterion_mse(eeg_feature, image_feature)
            else:
                eeg_for_mse, image_for_mse, _ = criterion._normalize_inputs(eeg_feature, image_feature)
                loss_mse = criterion.criterion_mse(eeg_for_mse, image_for_mse)
            return criterion.alpha * loss_contrastive + (1 - criterion.alpha) * loss_mse
        return loss_contrastive

    return criterion(eeg_feature, image_feature, text_feature, eeg_confidence=eeg_confidence)


def compute_cross_modal_loss(
    criterion,
    eeg_feature,
    image_feature,
    text_feature,
    positive_mask,
    use_multi_positive,
    sample_weights=None,
):
    """Run contrastive geometry in FP32 even when encoder AMP is active."""
    if not torch.is_autocast_enabled('cuda'):
        return _compute_cross_modal_loss_impl(
            criterion, eeg_feature, image_feature, text_feature,
            positive_mask, use_multi_positive, sample_weights,
        )

    if isinstance(eeg_feature, dict):
        eeg_feature = dict(eeg_feature)
        eeg_feature['feature'] = eeg_feature['feature'].float()
        if eeg_feature.get('confidence') is not None:
            eeg_feature['confidence'] = eeg_feature['confidence'].float()
    else:
        eeg_feature = eeg_feature.float()
    with torch.autocast(device_type='cuda', enabled=False):
        return _compute_cross_modal_loss_impl(
            criterion,
            eeg_feature,
            image_feature.float(),
            text_feature.float(),
            positive_mask,
            use_multi_positive,
            sample_weights,
        )


def compute_subject_mixup_regularization(mixed_eeg_feature, original_eeg_feature, partner_indices, mixed_mask):
    if partner_indices is None or mixed_mask is None or not torch.any(mixed_mask):
        return mixed_eeg_feature.new_tensor(0.0)
    mixed_subset = mixed_eeg_feature[mixed_mask]
    original_subset = original_eeg_feature[mixed_mask]
    partner_subset = original_eeg_feature[partner_indices[mixed_mask]]
    return (
        torch.norm(mixed_subset - original_subset, p=2, dim=1).sum()
        + torch.norm(mixed_subset - partner_subset, p=2, dim=1).sum()
    ) / mixed_eeg_feature.shape[0]


def exact_stimulus_cross_subject_partners(
    object_indices, image_indices, subject_ids, eligible_mask,
):
    """Choose a different eligible person viewing the exact same image."""
    partners = torch.arange(
        subject_ids.shape[0], device=subject_ids.device, dtype=torch.long
    )
    valid = torch.zeros_like(subject_ids, dtype=torch.bool)
    groups = {}
    for row, (obj, image, subject, eligible) in enumerate(zip(
        object_indices.detach().cpu().tolist(),
        image_indices.detach().cpu().tolist(),
        subject_ids.detach().cpu().tolist(),
        eligible_mask.detach().cpu().tolist(),
    )):
        if eligible:
            groups.setdefault((int(obj), int(image)), []).append((row, int(subject)))
    for rows in groups.values():
        for row, subject in rows:
            candidates = [other for other, other_subject in rows if other_subject != subject]
            if not candidates:
                continue
            choice = torch.randint(len(candidates), (), device=subject_ids.device)
            partners[row] = candidates[int(choice)]
            valid[row] = True
    return partners, valid


def compute_mutual_reconstruction_loss(
    model, content_features, real_eeg_targets, object_indices, image_indices,
    subject_ids, eligible_mask,
):
    """Decode target-person EEG from another person's exact-image content."""
    partners, valid = exact_stimulus_cross_subject_partners(
        object_indices, image_indices, subject_ids, eligible_mask
    )
    if not torch.any(valid):
        return content_features.new_tensor(0.0), 0
    target_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
    reconstruction = model.decode_subject_reconstruction(
        content_features[partners[target_rows]], subject_ids[target_rows]
    )
    loss = F.smooth_l1_loss(reconstruction, real_eeg_targets[target_rows])
    return loss, int(target_rows.numel())


def exact_stimulus_cohort_targets(
    eeg_targets, object_indices, image_indices, subject_ids,
):
    """Return one all-source EEG centroid for every complete exact-image group."""
    targets = torch.zeros_like(eeg_targets)
    valid = torch.zeros_like(subject_ids, dtype=torch.bool)
    groups = {}
    for row, (obj, image) in enumerate(zip(
        object_indices.detach().cpu().tolist(),
        image_indices.detach().cpu().tolist(),
    )):
        groups.setdefault((int(obj), int(image)), []).append(row)
    for rows in groups.values():
        row_index = torch.tensor(rows, device=eeg_targets.device, dtype=torch.long)
        if torch.unique(subject_ids[row_index]).numel() < 2:
            continue
        centroid = eeg_targets[row_index].mean(dim=0)
        targets[row_index] = centroid
        valid[row_index] = True
    return targets, valid


def compute_cohort_reconstruction_loss(
    model, content_features, real_eeg_targets, object_indices, image_indices,
    subject_ids,
):
    targets, valid = exact_stimulus_cohort_targets(
        real_eeg_targets, object_indices, image_indices, subject_ids
    )
    if not torch.any(valid):
        return content_features.new_tensor(0.0), 0
    reconstruction = model.decode_cohort_reconstruction(content_features[valid])
    loss = F.smooth_l1_loss(reconstruction, targets[valid])
    return loss, int(valid.sum().item())


def _smooth_lambda_over_time(lam, n_time):
    """Interpolate per-trial Beta control points into a smooth lambda(t)."""
    control = lam.reshape(lam.shape[0], 1, lam.shape[1]).float()
    return F.interpolate(
        control, size=n_time, mode='linear', align_corners=True
    ).to(lam.dtype)


def build_eeg_encoder(args, feature_dim, eeg_sample_points, channels_num):
    model = _build_eeg_encoder(args, feature_dim, eeg_sample_points, channels_num)
    if getattr(args, 'norm_type', 'batch') != 'batch':
        from module.eeg_encoder.model import swap_batchnorm
        swap_batchnorm(model, args.norm_type)
    if getattr(args, 'subject_adapter_dropout', 0) > 0:  # read by TSConv's subject adapter at training time only
        model.subject_adapter_dropout = args.subject_adapter_dropout
    if getattr(args, 'extra_input_adapter', False):  # submodule: trains, EMAs, reloads; unused at test (unseen ids)
        from module.eeg_encoder.model import ExtraInputAdapter
        model.extra_input_adapter = ExtraInputAdapter(channels_num)
    if getattr(args, 'latency_align', 0) > 0:  # a submodule, so it trains, EMAs and reloads with the encoder
        from module.eeg_encoder.model import LatencyAligner
        model.latency_aligner = LatencyAligner(channels_num, args.latency_align)
    return model


def _build_eeg_encoder(args, feature_dim, eeg_sample_points, channels_num):
    from module.eeg_encoder.model import DomainBatchNorm2d
    DomainBatchNorm2d.boundary2 = getattr(args, 'tsconv_domain_bn_boundary2', 0)
    if args.eeg_encoder_type == 'ATM':
        return ATMS(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=getattr(args, 'atm_d_model', 250),
            n_heads=getattr(args, 'atm_n_heads', 4),
            e_layers=getattr(args, 'atm_e_layers', 1),
            d_ff=getattr(args, 'atm_d_ff', 256),
            attention_dropout=getattr(args, 'atm_attention_dropout', 0.25),
            temporal_filters=getattr(args, 'atm_temporal_filters', 40),
            temporal_kernel=getattr(args, 'atm_temporal_kernel', 25),
            pool_kernel=getattr(args, 'atm_pool_kernel', 51),
            pool_stride=getattr(args, 'atm_pool_stride', 5),
            spatial_filters=getattr(args, 'atm_spatial_filters', 40),
            projection_filters=getattr(args, 'atm_projection_filters', 40),
            conv_dropout=getattr(args, 'atm_conv_dropout', 0.5),
            subject_token=not getattr(args, 'atm_no_subject_token', False),
            backbone_dim=getattr(args, 'atm_backbone_dim', 0),
            domain_bn_boundary=getattr(args, 'tsconv_domain_bn_boundary', 0),
        )
    if args.eeg_encoder_type == 'EEGNet':
        return EEGNet(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGProject':
        return EEGProject(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'TSConv':
        return TSConv(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'TSConv_parameterizable':
        head_dropout = None if args.tsconv_head_dropout < 0 else args.tsconv_head_dropout
        return TSConv_parameterizable(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            backbone_dim=args.tsconv_backbone_dim,
            temporal_filters=args.tsconv_temporal_filters,
            temporal_kernel=args.tsconv_temporal_kernel,
            pool_kernel=args.tsconv_pool_kernel,
            pool_stride=args.tsconv_pool_stride,
            spatial_filters=args.tsconv_spatial_filters,
            projection_filters=args.tsconv_projection_filters,
            activation=args.tsconv_activation,
            dropout=args.tsconv_dropout,
            head_dropout=head_dropout,
            power_branch=getattr(args, 'tsconv_power_branch', False),
            head_ln=not getattr(args, 'tsconv_head_no_ln', False),
            batch_norm=not args.tsconv_no_batch_norm,
            conv_bias=not args.tsconv_no_conv_bias,
            domain_batch_norm_boundary=getattr(args, 'tsconv_domain_bn_boundary', 0),
            domain_spatial_boundary=getattr(args, 'tsconv_domain_spatial_boundary', 0),
            subject_batch_norm_max_id=getattr(args, 'tsconv_subject_bn_max_id', 0),
            subject_adapter_rank=getattr(args, 'subject_adapter_rank', 0),
            subject_adapter_max_id=getattr(args, 'subject_adapter_max_id', 32),
            subject_adapter_scale=getattr(args, 'subject_adapter_scale', 0.1),
            subject_adapter_virtual_prob=getattr(args, 'subject_adapter_virtual_prob', 0.0),
            subject_adapter_virtual_scale=getattr(args, 'subject_adapter_virtual_scale', 1.0),
            subject_adapter_source_ids=getattr(args, 'train_subject_ids', []),
            subject_adapter_inferred=getattr(args, 'subject_adapter_inferred', False),
            mutual_reconstruction_hidden=(
                getattr(args, 'mutual_reconstruction_hidden', 0)
                if getattr(args, 'mutual_reconstruction_lambda', 0.0) > 0
                else 0
            ),
            cohort_reconstruction_hidden=(
                getattr(args, 'cohort_reconstruction_hidden', 0)
                if getattr(args, 'cohort_reconstruction_lambda', 0.0) > 0
                else 0
            ),
        )
    if args.eeg_encoder_type == 'TSConv30':
        return TSConv30(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGTransformer':
        return EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGConformer':
        return EEGConformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    from module.eeg_encoder.foundation import (
        build_foundation_encoder,
        is_foundation_encoder,
    )
    if is_foundation_encoder(args.eeg_encoder_type):
        return build_foundation_encoder(
            args.eeg_encoder_type,
            feature_dim,
            channels_num,
            pretrained=not getattr(args, 'fm_no_pretrained', False),
            ch_names=getattr(args, 'selected_channels', None) or None,  # a channel subset names its electrodes
            resample=getattr(args, 'fm_resample', 'resample'),
            input_normalization=getattr(args, 'reve_input_normalization', 'trial_zscore'),
            head=getattr(args, 'fm_head', 'mean'),
            head_dropout=getattr(args, 'fm_head_dropout', 0.0),
            drop_path=getattr(args, 'fm_drop_path', 0.0),
            arch_kwargs=json.loads(getattr(args, 'fm_arch_kwargs', None) or '{}'),
        )
    from ensemble_experiments.architectures.ortho_encoders import (
        build_architecture_encoder,
        is_architecture_encoder,
    )
    if is_architecture_encoder(args.eeg_encoder_type):
        extra = {}
        if args.eeg_encoder_type.removeprefix('Ortho') == 'FastTSConvSqueezeformer':
            # Reuse the TSConv geometry flags; their defaults equal the 250 Hz design.
            extra = dict(
                pool_kernel=getattr(args, 'tsconv_pool_kernel', 51),
                pool_stride=getattr(args, 'tsconv_pool_stride', 5),
                backbone_dim=getattr(args, 'tsconv_backbone_dim', 0),
                d_model=getattr(args, 'sqf_d_model', 160),
                depth=getattr(args, 'sqf_depth', 3),
                power_branch=getattr(args, 'tsconv_power_branch', False),
                domain_bn_boundary=getattr(args, 'tsconv_domain_bn_boundary', 0),
            )
        if args.eeg_encoder_type == 'BridgeTransformer':  # REVE's position bank resolves electrodes by name
            extra['ch_names'] = getattr(args, 'selected_channels', None) or None
        extra.update(json.loads(getattr(args, 'arch_kwargs', None) or '{}'))
        return build_architecture_encoder(
            args.eeg_encoder_type, feature_dim, eeg_sample_points, channels_num, **extra
        )
    raise ValueError(f"Unsupported EEG encoder type: {args.eeg_encoder_type}")


def attach_lora(linear, rank):
    """SUBJECT_LOG D5: low-rank fine-tuning of one nn.Linear in place: forward uses W + B A (A random, B zero, so the
    model starts unchanged); state_dict() stores the merged weight under the usual key and drops the adapters, so
    checkpoints load into the plain architecture (score_track1, grader)."""
    linear.lora_A = nn.Parameter(torch.randn(rank, linear.in_features, device=linear.weight.device) / math.sqrt(linear.in_features))
    linear.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=linear.weight.device))
    linear.forward = lambda x: F.linear(x, linear.weight + linear.lora_B @ linear.lora_A, linear.bias)

    def merged(module, destination, prefix, local_metadata):
        destination[prefix + 'weight'] = (module.weight + module.lora_B @ module.lora_A).detach()
        destination.pop(prefix + 'lora_A', None); destination.pop(prefix + 'lora_B', None)
    linear._register_state_dict_hook(merged)


def spatial_mix_bank(ch_names, max_deg, n=256, seed=0):
    """Lever L1: (n, C, C) re-interpolation matrices for random montage rotations up to max_deg. Electrode i of a
    displaced cap sits at R p_i and records the (spatially smooth) field there, interpolated from the recorded
    electrodes with a Gaussian kernel on the sphere (sigma = 1/3 of the median nearest-neighbour angle, so the
    unrotated matrix is ~identity)."""
    import mne
    pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    p = np.array([pos[c] for c in ch_names], dtype=np.float64)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    ang = lambda a, b: np.arccos(np.clip(a @ b.T, -1.0, 1.0))  # noqa: E731
    sigma = np.median(np.sort(ang(p, p), axis=1)[:, 1]) / 3
    rng, bank = np.random.default_rng(seed), []
    for _ in range(n):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k  # Rodrigues
        w = np.exp(-ang(p @ rot.T, p) ** 2 / (2 * sigma ** 2))
        bank.append(w / w.sum(axis=1, keepdims=True))
    return np.stack(bank).astype(np.float32)


def build_projector(projector_type, input_dim, output_dim, activation='none', topk=0):
    activation = (activation or 'none').lower()
    if activation not in ('none', 'relu', 'relu_gelu', 'topk'):
        raise ValueError(f"Unsupported projector activation: {activation}")
    if activation == 'topk' and topk <= 0:
        raise ValueError('--projector_topk must be > 0 when --projector_activation topk')
    if projector_type in ('subspace', 'subspace_c'):  # lever L3: EEG side k -> output; image/text side: identity
        if input_dim == output_dim:
            return ProjectorDirect(activation=activation, topk=topk)
        return SubspaceProjector(input_dim, output_dim, centered=projector_type == 'subspace_c')
    if projector_type == 'bottleneck_do':  # bottleneck with dropout on the code (VIB-like noise), lift as below
        if input_dim == output_dim:
            return ProjectorDirect(activation=activation, topk=topk)
        return nn.Sequential(nn.Dropout(0.3), nn.Linear(input_dim, output_dim))
    if projector_type == 'bottleneck':  # EEG side: encoder emits a low-rank code (--eeg_backbone_dim r), lifted
        # linearly to the target space; image/text side: identity, so outputs stay in the grader's DINOv2 space
        if input_dim == output_dim:
            return ProjectorDirect(activation=activation, topk=topk)
        return nn.Linear(input_dim, output_dim)
    if projector_type == 'direct':
        if input_dim != output_dim:
            raise ValueError(
                f"ProjectorDirect requires input_dim == output_dim, got {input_dim} != {output_dim}."
            )
        return ProjectorDirect(activation=activation, topk=topk)
    if projector_type == 'linear':
        return ProjectorLinear(input_dim, output_dim, activation=activation, topk=topk)
    if projector_type == 'mlp':
        return ProjectorMLP(input_dim, output_dim, activation=activation, topk=topk)
    raise ValueError(f"Unsupported projector type: {projector_type}")


_SPARSE_PROJECTOR_ACTIVATIONS = frozenset({'relu', 'relu_gelu', 'topk'})


class SharedSpaceProjector(nn.Module):
    """Modality projector followed by an encoder g(.) shared by BOTH modalities (SAMGA).

    Applying the same transform to the EEG and image branches forces them to obey one geometric
    rule instead of meeting through two independent maps. Wrapping the existing projector keeps
    every call site (train loop, eval, subject-adaptation) unchanged.
    """

    def __init__(self, projector, shared):
        super().__init__()
        self.projector = projector
        self.shared = shared

    def forward(self, x):
        return self.shared(self.projector(x))


def build_shared_encoder(feature_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, feature_dim),
    )


def _empty_alignment_sparsity_accumulator():
    return {'l0_sum': 0.0, 'n_samples': 0, 'active_mask': None}


def _update_alignment_sparsity_accumulator(acc, features):
    """Update running L0 stats without storing all samples (O(dim) memory)."""
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return acc
    if acc['active_mask'] is None:
        acc['active_mask'] = np.zeros(features.shape[1], dtype=bool)
    acc['l0_sum'] += float((features > 0).sum(axis=1).sum())
    acc['n_samples'] += features.shape[0]
    acc['active_mask'] |= (features > 0).any(axis=0)
    return acc


def _finalize_alignment_sparsity_accumulator(acc):
    if acc['n_samples'] == 0 or acc['active_mask'] is None:
        return {'l0_mean': 0.0, 'l0_frac': 0.0, 'active_feat_frac': 0.0}
    dim = acc['active_mask'].shape[0]
    l0_mean = acc['l0_sum'] / acc['n_samples']
    active_dims = int(acc['active_mask'].sum())
    return {
        'l0_mean': l0_mean,
        'l0_frac': l0_mean / dim,
        'active_feat_frac': active_dims / dim,
    }


def compute_alignment_sparsity_stats(features):
    """L0 stats on post-ReLU, pre-L2-norm alignment vectors."""
    acc = _empty_alignment_sparsity_accumulator()
    _update_alignment_sparsity_accumulator(acc, features)
    return _finalize_alignment_sparsity_accumulator(acc)


def _things_coral_maps(args, device, n_trials=2000):
    """--things_coral: per THINGS-EEG2 subject (IDs 21-30 in the Alljoined dir) the map R_aj^1/2 R_s^-1/2 that
    recolours its trials to the mean spatial covariance R_aj of the Alljoined TRAINING subjects (CORAL on raw EEG).
    Index = subject id; every other id (Alljoined, test 10**6 clamped to 31) gets the identity, so the test path
    is untouched. Deterministic (seeded sample of training-image trials), so evaluation rebuilds the same maps."""
    def cov(s):
        x = np.load(f'{args.eeg_data_dir}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)
        idx = np.sort(np.random.default_rng(s).choice(len(x), 4 * n_trials, replace=False))
        x = np.asarray(x[idx], np.float64); x = x[np.abs(x).sum((1, 2)) > 0][:n_trials]  # skip unseen-image padding
        x = x - x.mean(-1, keepdims=True)
        R = np.einsum('nct,ndt->cd', x, x) / (len(x) * x.shape[-1])
        return R + 1e-3 * np.trace(R) / 32 * np.eye(32)
    def power(R, p):
        w, V = np.linalg.eigh(R); return (V * w ** p) @ V.T
    R_aj = np.mean([cov(s) for s in args.train_subject_ids if s <= 20], 0)
    A = np.tile(np.eye(32), (32, 1, 1))
    for s in args.train_subject_ids:
        if 21 <= s <= 30:
            A[s] = power(R_aj, .5) @ power(cov(s), -.5)
    return torch.tensor(A, dtype=torch.float32, device=device)


def run_eeg_backbone(model, args, eeg_batch, subject_id_batch, return_intermediate=False):
    if getattr(model, 'extra_input_adapter', None) is not None:
        from module.eeg_encoder.model import UNKNOWN_SUBJECT_ID
        rows = (subject_id_batch > args.extra_subject_offset) & (subject_id_batch < UNKNOWN_SUBJECT_ID)
        if rows.any():
            eeg_batch = eeg_batch.clone()
            eeg_batch[rows] = model.extra_input_adapter(eeg_batch[rows])
    if getattr(model, 'latency_aligner', None) is not None:
        eeg_batch = model.latency_aligner(eeg_batch)
    covariance_aligner = getattr(args, '_inductive_covariance_aligner', None)
    if covariance_aligner is not None:
        eeg_batch = covariance_aligner(eeg_batch, subject_id_batch)
    if getattr(args, 'things_coral', False):
        if getattr(args, '_things_coral', None) is None or args._things_coral.device != eeg_batch.device:
            args._things_coral = _things_coral_maps(args, eeg_batch.device)
        A = args._things_coral[subject_id_batch.clamp(max=31)]
        eeg_batch = torch.einsum('bij,bjt->bit', A.to(eeg_batch.dtype), eeg_batch)
    if getattr(args, 'input_lowpass_hz', 0):  # AUG_LOG: keep only the band that carries cross-subject image information
        f = torch.fft.rfftfreq(eeg_batch.shape[-1], 1 / 120, device=eeg_batch.device)  # cv4 data are 120 Hz
        eeg_batch = torch.fft.irfft(torch.fft.rfft(eeg_batch.float(), dim=-1) * (f < args.input_lowpass_hz), n=eeg_batch.shape[-1], dim=-1).to(eeg_batch.dtype)
    if getattr(args, 'eeg_car', False):  # common average reference, per trial and time point (stateless)
        eeg_batch = eeg_batch - eeg_batch.mean(dim=-2, keepdim=True)
    if getattr(args, 'eeg_instance_norm', False):
        # Per-trial, per-channel standardization. Uses only the trial itself, so it applies
        # identically at train and test time (no test-set statistics -> still pure zero-shot).
        eeg_batch = (eeg_batch - eeg_batch.mean(dim=-1, keepdim=True)) / eeg_batch.std(dim=-1, keepdim=True).clamp_min(1e-5)
    if args.eeg_encoder_type == 'ATM':
        return model(eeg_batch, subject_id_batch)
    elif args.eeg_encoder_type in ['TSConv', 'TSConv_parameterizable']:
        temporal_augmentation = None
        mixstyle_prob = getattr(args, 'subject_mixstyle_prob', 0.0)
        manifold_prob = getattr(args, 'subject_style_manifold_prob', 0.0)
        if model.training and manifold_prob > 0 and torch.rand((), device=eeg_batch.device) < manifold_prob:
            temporal_augmentation = lambda features: subject_style_manifold(
                features,
                subject_id_batch,
                rank=args.subject_style_manifold_rank,
                latent_scale=args.subject_style_manifold_scale,
                max_radius=args.subject_style_manifold_max_radius,
            )
        elif model.training and mixstyle_prob > 0 and torch.rand((), device=eeg_batch.device) < mixstyle_prob:
            temporal_augmentation = lambda features: subject_coherent_mixstyle(
                features,
                subject_id_batch,
                alpha=args.subject_mixstyle_alpha,
                granularity=args.subject_mixstyle_granularity,
            )
        model_kwargs = {}
        if (
            getattr(args, 'tsconv_domain_bn_boundary', 0) > 0
            or getattr(args, 'tsconv_domain_spatial_boundary', 0) > 0
            or getattr(args, 'tsconv_subject_bn_max_id', 0) > 0
            or getattr(args, 'subject_adapter_rank', 0) > 0
        ):
            model_kwargs['subject_ids'] = subject_id_batch
        return model(
            eeg_batch,
            return_intermediate=return_intermediate,
            temporal_augmentation=temporal_augmentation,
            **model_kwargs,
        )
    elif args.eeg_encoder_type == 'TSConv30':
        if return_intermediate:
            raise ValueError('TSConv30 does not expose intermediate features.')
        return model(eeg_batch)
    if getattr(args, 'tsconv_domain_bn_boundary', 0) > 0:  # registry Squeezeformer with per-dataset BatchNorm
        return model(eeg_batch, subject_ids=subject_id_batch)
    return model(eeg_batch)


def _normalize_rows_torch(features, eps=1e-12):
    return features / torch.clamp(features.norm(dim=1, keepdim=True), min=eps)


def _normalize_rows_np(features, eps=1e-12):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, eps, None)


def subject_adaptive_whiten_torch(features, subject_ids, shrink=0.2, diag=False, normalize=True, eps=1e-6):
    if subject_ids is None or features.ndim != 2 or features.shape[0] == 0:
        return features

    processed = torch.empty_like(features)
    for sid in torch.unique(subject_ids):
        mask = subject_ids == sid
        sub_feats = features[mask]
        if sub_feats.shape[0] <= 1:
            processed[mask] = _normalize_rows_torch(sub_feats, eps) if normalize else sub_feats
            continue

        mu = sub_feats.mean(dim=0, keepdim=True)
        centered = sub_feats - mu
        cov = centered.T @ centered / float(max(sub_feats.shape[0] - 1, 1))
        if diag:
            cov = torch.diag(torch.diagonal(cov))
        dim = cov.shape[0]
        eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
        trace_mean = torch.trace(cov) / max(dim, 1)
        cov = (1.0 - float(shrink)) * cov + float(shrink) * trace_mean * eye
        cov = cov + eps * eye
        cov = 0.5 * (cov + cov.T)

        cov_work = cov.to(torch.float64)
        eye_work = eye.to(torch.float64)
        centered_work = centered.to(torch.float64)
        jitter = float(eps)
        for _ in range(5):
            try:
                evals, evecs = torch.linalg.eigh(cov_work)
                break
            except torch._C._LinAlgError:
                cov_work = cov_work + jitter * eye_work
                jitter *= 10.0
        else:
            inv_diag = torch.rsqrt(torch.clamp(torch.diagonal(cov_work), min=eps))
            whitened = centered_work * inv_diag.unsqueeze(0)
            whitened = whitened.to(features.dtype)
            processed[mask] = _normalize_rows_torch(whitened, eps) if normalize else whitened
            continue

        inv_sqrt = evecs @ torch.diag(torch.rsqrt(torch.clamp(evals, min=eps))) @ evecs.T
        whitened = (centered_work @ inv_sqrt).to(features.dtype)
        processed[mask] = _normalize_rows_torch(whitened, eps) if normalize else whitened

    return processed


def _fit_subject_adapt_calibration(query_features, image_features, args):
    normalize = not args.subject_adapt_saw_no_renorm
    query_features = np.asarray(query_features, dtype=np.float32)
    image_features = np.asarray(image_features, dtype=np.float32)
    mu, cov = _estimate_mu_cov(
        query_features,
        shrink=args.subject_adapt_saw_shrink,
        diag=args.subject_adapt_saw_diag,
    )
    whitener = _inv_sqrt_cov(cov)
    transformed = (query_features - mu) @ whitener
    if normalize:
        transformed = _normalize_rows_np(transformed)
    cumulative_map = np.eye(transformed.shape[1], dtype=np.float32)
    for _ in range(max(1, int(args.subject_adapt_soft_procrustes_steps))):
        scores = _normalize_rows_np(transformed) @ _normalize_rows_np(image_features).T
        if args.subject_adapt_csls_k > 0:
            scores = csls_scores(scores, k=args.subject_adapt_csls_k)
        assignment = sinkhorn_normalize(
            scores,
            tau=args.subject_adapt_sinkhorn_tau,
            num_iters=args.subject_adapt_sinkhorn_iters,
        )
        step_map = fit_soft_assignment_procrustes(
            transformed,
            image_features,
            assignment,
            power=args.subject_adapt_soft_procrustes_power,
            normalize_inputs=False,
        )
        if step_map is None:
            break
        transformed = apply_orthogonal_map(transformed, step_map)
        cumulative_map = (cumulative_map @ np.asarray(step_map, dtype=np.float32)).astype(np.float32, copy=False)
    return mu, whitener, cumulative_map, normalize


def _apply_subject_adapt_calibration_torch(features, mu, whitener, ortho_map, normalize_before_map):
    transformed = (features - mu) @ whitener
    if normalize_before_map:
        transformed = _normalize_rows_torch(transformed)
    transformed = transformed @ ortho_map
    return _normalize_rows_torch(transformed)


def compute_subject_adaptation_loss(
    args,
    criterion,
    eeg_feature_batch,
    image_feature_proj,
    text_feature_proj,
    subject_id_batch,
    object_idx_batch,
    image_idx_batch,
):
    if args.subject_adapt_lambda <= 0:
        return eeg_feature_batch.new_tensor(0.0), 0

    total_loss = eeg_feature_batch.new_tensor(0.0)
    valid_subjects = 0
    unique_subjects = torch.unique(subject_id_batch)
    for sid in unique_subjects.tolist():
        mask = subject_id_batch == sid
        subject_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        num_subject_samples = int(subject_indices.numel())
        if num_subject_samples < args.subject_adapt_min_samples_per_subject:
            continue

        split_a_size = int(np.floor(num_subject_samples * float(args.subject_adapt_split_a_ratio)))
        split_a_size = max(1, min(split_a_size, num_subject_samples - 1))
        split_b_size = num_subject_samples - split_a_size
        if split_b_size <= 0:
            continue

        perm = torch.randperm(num_subject_samples, device=subject_indices.device)
        split_a_indices = subject_indices[perm[:split_a_size]]
        split_b_indices = subject_indices[perm[split_a_size:]]
        if split_b_indices.numel() == 0:
            continue

        with torch.no_grad():
            mu_np, whitener_np, ortho_np, normalize_before_map = _fit_subject_adapt_calibration(
                eeg_feature_batch[split_a_indices].detach().float().cpu().numpy(),
                image_feature_proj[split_a_indices].detach().float().cpu().numpy(),
                args,
            )

        mu = torch.tensor(mu_np, device=eeg_feature_batch.device, dtype=eeg_feature_batch.dtype)
        whitener = torch.tensor(whitener_np, device=eeg_feature_batch.device, dtype=eeg_feature_batch.dtype)
        ortho_map = torch.tensor(ortho_np, device=eeg_feature_batch.device, dtype=eeg_feature_batch.dtype)
        adapted_eeg_feature = _apply_subject_adapt_calibration_torch(
            eeg_feature_batch[split_b_indices],
            mu,
            whitener,
            ortho_map,
            normalize_before_map,
        )
        positive_mask = build_image_positive_mask(object_idx_batch[split_b_indices], image_idx_batch[split_b_indices])
        total_loss = total_loss + compute_cross_modal_loss(
            criterion,
            adapted_eeg_feature,
            image_feature_proj[split_b_indices],
            text_feature_proj[split_b_indices],
            positive_mask,
            args.multi_positive_loss,
        )
        valid_subjects += 1

    if valid_subjects == 0:
        return eeg_feature_batch.new_tensor(0.0), 0
    return total_loss / valid_subjects, valid_subjects


def cross_subject_stimulus_mix(
    features, object_indices, image_indices, subject_ids, alpha=1.0, mixup_type='pairwise',
    return_mixup_metadata=False, group_self_anchor=0.0, partition_labels=None,
    time_varying=0, hybrid_group_prob=0.5,
):
    if features.shape[0] < 2:
        if return_mixup_metadata:
            return features, {'partner_indices': None, 'mixed_mask': None}
        return features

    if mixup_type == 'hybrid_group_pairwise':
        if return_mixup_metadata:
            raise ValueError('hybrid_group_pairwise does not provide pairwise metadata')
        group_mixed = cross_subject_stimulus_mix(
            features, object_indices, image_indices, subject_ids,
            alpha=alpha, mixup_type='group', group_self_anchor=group_self_anchor,
            partition_labels=partition_labels, time_varying=0,
        )
        pair_mixed = cross_subject_stimulus_mix(
            features, object_indices, image_indices, subject_ids,
            alpha=alpha, mixup_type='pairwise', partition_labels=partition_labels,
            time_varying=time_varying,
        )
        use_group = torch.rand(features.shape[0], device=features.device) < float(hybrid_group_prob)
        view_shape = (features.shape[0],) + (1,) * (features.ndim - 1)
        return torch.where(use_group.view(view_shape), group_mixed, pair_mixed)

    obj = object_indices.detach().cpu().tolist()
    img = image_indices.detach().cpu().tolist()
    sid = subject_ids.detach().cpu().tolist()

    if partition_labels is None:
        partitions = [None] * len(obj)
    else:
        if partition_labels.shape != subject_ids.shape:
            raise ValueError('partition_labels must match subject_ids')
        partitions = partition_labels.detach().cpu().tolist()
    groups = {}
    for i, (o, im, partition) in enumerate(zip(obj, img, partitions)):
        groups.setdefault((int(o), int(im), partition), []).append(i)

    mixed = features.clone()
    partner_indices = torch.arange(features.shape[0], device=features.device)
    mixed_mask = torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
    dist_alpha = max(float(alpha), 1e-3)

    for indices in groups.values():
        group_size = len(indices)
        if group_size < 2:
            continue

        group_idx = torch.tensor(indices, device=features.device, dtype=torch.long)
        group_features = features[group_idx]

        if mixup_type in {'group', 'leave_one_out_group'}:
            if mixup_type == 'leave_one_out_group':
                # Row i is a pseudo-held subject: synthesize its matched
                # observation using only the other source people.
                concentration = torch.full(
                    (group_size - 1,), dist_alpha,
                    device=features.device, dtype=torch.float32,
                )
                reduced = torch.distributions.Dirichlet(concentration).sample(
                    (group_size,)
                ).to(group_features.dtype)
                weights = torch.zeros(
                    group_size, group_size,
                    device=features.device, dtype=group_features.dtype,
                )
                off_diagonal = ~torch.eye(
                    group_size, device=features.device, dtype=torch.bool
                )
                weights[off_diagonal] = reduced.reshape(-1)
            else:
                concentration = torch.full((group_size,), dist_alpha, device=features.device, dtype=torch.float32)
                if time_varying > 1:
                    controls = torch.distributions.Dirichlet(concentration).sample(
                        (int(time_varying), group_size)
                    ).to(group_features.dtype)
                    curve = controls.permute(1, 2, 0).reshape(
                        1, group_size * group_size, int(time_varying)
                    )
                    weights = F.interpolate(
                        curve.float(), size=group_features.shape[-1],
                        mode='linear', align_corners=True,
                    ).reshape(group_size, group_size, group_features.shape[-1]).to(
                        group_features.dtype
                    )
                else:
                    weights = torch.distributions.Dirichlet(concentration).sample((group_size,)).to(group_features.dtype)
            if group_self_anchor:
                if mixup_type == 'leave_one_out_group':
                    raise ValueError(
                        'group_self_anchor is incompatible with leave_one_out_group'
                    )
                weights = (1.0 - float(group_self_anchor)) * weights
                identity = torch.eye(
                    group_size, device=features.device, dtype=group_features.dtype
                )
                if weights.ndim == 3:
                    identity = identity.unsqueeze(-1)
                weights = weights + float(group_self_anchor) * identity
            if weights.ndim == 3:
                flattened = group_features.reshape(
                    group_size, -1, group_features.shape[-1]
                )
                mixed_group = torch.einsum('abt,bkt->akt', weights, flattened)
                mixed_group = mixed_group.reshape_as(group_features)
            else:
                mixed_group = torch.einsum('ab,b...->a...', weights, group_features)
        else:
            group_subject_ids = [sid[i] for i in indices]
            if len(set(group_subject_ids)) == group_size:
                order = torch.randperm(group_size, device=features.device)
                offset = random.randint(1, group_size - 1)
                partner_pos = torch.empty(group_size, device=features.device, dtype=torch.long)
                partner_pos[order] = torch.roll(order, shifts=offset)
            else:
                partner_pos = []
                for pos, subject_id in enumerate(group_subject_ids):
                    candidates = [j for j, sid_j in enumerate(group_subject_ids) if j != pos and sid_j != subject_id]
                    if not candidates:
                        candidates = [j for j in range(group_size) if j != pos]
                    partner_pos.append(random.choice(candidates))
                partner_pos = torch.tensor(partner_pos, device=features.device, dtype=torch.long)

            partner_indices[group_idx] = group_idx[partner_pos]
            mixed_mask[group_idx] = True
            concentration = torch.full((group_size,), dist_alpha, device=features.device, dtype=torch.float32)
            beta = torch.distributions.Beta(concentration, concentration)
            if time_varying > 1:
                lam = torch.stack(
                    [beta.sample() for _ in range(int(time_varying))], dim=-1
                ).to(group_features.dtype)
                lam = _smooth_lambda_over_time(lam, group_features.shape[-1])
                lam_shape = [group_size] + [1] * (group_features.dim() - 2) + [-1]
            else:
                # Keep the historical scalar sampler bit-for-bit when disabled.
                lam = beta.sample().to(group_features.dtype)
                lam_shape = [group_size] + [1] * (group_features.dim() - 1)
            lam = lam.view(*lam_shape)
            mixed_group = lam * group_features + (1.0 - lam) * group_features[partner_pos]

        mixed[group_idx] = mixed_group

    if return_mixup_metadata:
        return mixed, {'partner_indices': partner_indices, 'mixed_mask': mixed_mask}
    return mixed


def select_rows_per_exact_image(object_indices, image_indices, rows_per_image=1):
    """Select a bounded number of rows from every exact-image group.

    Group SubjectMix needs all source rows to construct its Dirichlet draw, but
    contrastive training need not retain all resulting virtual rows.  Selection
    happens only after synthesis, allowing a batch to contain many more unique
    image negatives without changing the counterfactual generator.
    """
    if rows_per_image <= 0:
        raise ValueError("rows_per_image must be positive")
    groups = {}
    for index, key in enumerate(zip(
        object_indices.detach().cpu().tolist(),
        image_indices.detach().cpu().tolist(),
    )):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)
    selected = [
        index
        for indices in groups.values()
        for index in indices[:int(rows_per_image)]
    ]
    return torch.tensor(selected, device=object_indices.device, dtype=torch.long)


def select_hard_counterfactual_views(eeg_features, image_features, num_views):
    """Select the currently worst aligned draw for every base batch row."""
    if num_views <= 1 or eeg_features.shape != image_features.shape:
        raise ValueError("hard-view selection needs matching features and at least two views")
    if eeg_features.shape[0] % int(num_views) != 0:
        raise ValueError("feature rows must be divisible by num_views")
    base_rows = eeg_features.shape[0] // int(num_views)
    agreement = F.cosine_similarity(eeg_features, image_features, dim=1)
    hard_view = agreement.view(int(num_views), base_rows).argmin(dim=0)
    return hard_view * base_rows + torch.arange(
        base_rows, device=eeg_features.device
    )


def hyperalign_transfer(eeg_batch, subject_ids, rotations, allowed, prob):
    """AUG_LOG: re-render each trial (probability prob) through the Procrustes rotation from its subject to a random
    other training subject of the same dataset (neurips_challenge/hyperalign_maps.py: fitted on repeat-averaged
    responses to the shared training images, so it follows the image-locked signal, not the background covariance)."""
    w = allowed[subject_ids].float()
    has = w.sum(1) > 0
    partner = torch.multinomial(w + (~has).float()[:, None], 1).squeeze(1)  # rows without a partner draw anything, then stay
    use = has & (torch.rand(len(eeg_batch), device=eeg_batch.device) < prob)
    M = torch.where(use[:, None, None], rotations[subject_ids, partner], torch.eye(rotations.shape[-1], device=eeg_batch.device))
    return torch.einsum('bcd,bdt->bct', M, eeg_batch.float()).to(eeg_batch.dtype)


def cross_subject_operator_transfer(eeg_batch, subject_ids, shrink=0.05, extrapolate=0.0, eps=1e-8, boundary=0, partners=None, rotate=False):
    """do(subject := t): re-render a trial through another subject's channel-mixing operator.

    Model the sensor signal as x_s = A_s . (sources), so the subject-specific nuisance is the
    channel mixing A_s, observable through the channel covariance R_s. Mapping
    x_s -> R_t^(1/2) R_s^(-1/2) x_s swaps the recording operator while leaving the stimulus-driven
    sources untouched, so the image label is preserved exactly -- unlike convex mixup, which also
    blends the signals themselves.

    With ``extrapolate > 0`` the target is not an observed subject but a point at parameter
    u ~ U(0, extrapolate) along the SPD geodesic from R_s through R_j:
        M = R_s^(1/2) (R_s^(-1/2) R_j R_s^(-1/2))^(u/2) R_s^(-1/2),   M R_s M' = geodesic(R_s,R_j,u)
    u=1 lands exactly on subject j, u>1 lands *past* it -- a synthetic operator outside the hull of
    the 9 training subjects, which is the regime a held-out subject actually lives in.

    R_s is estimated per batch (~113 trials x 250 samples per subject, plenty for 63x63), which
    also injects a little estimation noise as free regularization. One target is drawn per source
    subject per batch; over an epoch all ordered pairs get covered.
    """
    channels = eeg_batch.shape[-2]
    uniq = torch.unique(subject_ids)
    if uniq.numel() < 2:
        return eeg_batch

    eye = torch.eye(channels, device=eeg_batch.device, dtype=torch.float64)

    def _spd(mat):
        evals, evecs = torch.linalg.eigh(mat)
        return evals.clamp_min(eps), evecs

    covs, sqrt_ops, inv_sqrt_ops = {}, {}, {}
    for s in uniq.tolist():
        flat = eeg_batch[subject_ids == s].reshape(-1, channels, eeg_batch.shape[-1]).to(torch.float64)
        cov = torch.einsum('nct,ndt->cd', flat, flat) / (flat.shape[0] * flat.shape[-1])
        cov = (1.0 - shrink) * cov + shrink * eye * (torch.diagonal(cov).mean())
        evals, evecs = _spd(cov)
        covs[s] = cov
        sqrt_ops[s] = (evecs * evals.sqrt()) @ evecs.T
        inv_sqrt_ops[s] = (evecs * evals.rsqrt()) @ evecs.T

    out = eeg_batch.clone()
    if partners is None:
        partners = draw_op_partners(subject_ids, boundary)
    for s, j in zip(uniq.tolist(), partners):
        if s == j:
            continue
        mask = subject_ids == s
        if extrapolate > 0:
            whitened = inv_sqrt_ops[s] @ covs[j] @ inv_sqrt_ops[s]
            evals, evecs = _spd(whitened)
            u = random.uniform(0.0, extrapolate)
            powered = (evecs * evals.pow(u / 2.0)) @ evecs.T
            transfer = sqrt_ops[s] @ powered @ inv_sqrt_ops[s]
        elif rotate:  # control: partner's spectrum, random spatial orientation
            q, _ = torch.linalg.qr(torch.randn(channels, channels, device=eye.device, dtype=eye.dtype))
            transfer = q @ sqrt_ops[j] @ q.T @ inv_sqrt_ops[s]
        else:
            transfer = sqrt_ops[j] @ inv_sqrt_ops[s]
        out[mask] = torch.einsum('cd,...dt->...ct', transfer.to(eeg_batch.dtype), eeg_batch[mask]).to(out.dtype)  # autocast einsum returns bf16
    return out


def draw_op_partners(subject_ids, boundary=0):
    """One random partner per subject in the batch (same headset only when boundary > 0: THINGS-EEG2 ids >= boundary)."""
    uniq = torch.unique(subject_ids).tolist()
    if boundary:
        return [random.choice([j for j in uniq if (j >= boundary) == (s >= boundary)]) for s in uniq]
    return [random.choice(uniq) for _ in uniq]


def banded_operator_transfer(eeg_batch, subject_ids, edges, fs=120.0, boundary=0, **kw):
    """Operator transfer applied separately in each frequency band [edges[i], edges[i+1]) Hz, same partners in every band."""
    spec = torch.fft.rfft(eeg_batch.float(), dim=-1)
    freqs = torch.fft.rfftfreq(eeg_batch.shape[-1], 1.0 / fs).to(eeg_batch.device)
    partners = draw_op_partners(subject_ids, boundary)
    out = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = torch.fft.irfft(spec * ((freqs >= lo) & (freqs < hi)), n=eeg_batch.shape[-1], dim=-1)
        out = out + cross_subject_operator_transfer(band, subject_ids, partners=partners, **kw)
    return out.to(eeg_batch.dtype)


class _GroupedSubset(Subset):
    """Subset that preserves get_image_group_indices for GroupedImageBatchSampler."""

    def get_image_group_indices(self):
        index_map = {int(src_idx): sub_idx for sub_idx, src_idx in enumerate(self.indices)}
        out = {}
        for key, lst in self.dataset.get_image_group_indices().items():
            sub = [index_map[i] for i in lst if i in index_map]
            if sub:
                out[key] = sub
        return out


class EEGConfidenceHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.net(x)).squeeze(-1)


def set_requires_grad(module, enabled):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def collect_trainable_parameters(modules):
    params = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='training device')
    parser.add_argument(
        '--amp_dtype', default='none', choices=['none', 'bfloat16', 'float16'],
        help='automatic mixed precision for model/projector activations; contrastive loss stays FP32',
    )
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='AdamW weight decay (AVDE full-FT uses 0.05)')
    parser.add_argument('--lr_scheduler', default='none', choices=['none', 'cosine', 'onecycle', 'wsd', 'plateau'], help='optional warmup+cosine LR schedule (per epoch); AVDE full-FT uses cosine. onecycle steps PER OPTIMIZER STEP, matching the NeuralBench Track-1 recipe; wsd = warmup-stable-decay, also per step')
    parser.add_argument('--plateau_patience', default=5, type=int, help='--lr_scheduler plateau (REVE authors): halve the lr once the per-epoch sealed top-5 has not improved for more than this many epochs; --warmup_epochs adds their per-step exponential warmup')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'stableadamw'], help='stableadamw = the REVE authors\' optimizer and settings (betas 0.92/0.999, eps 1e-9)')
    parser.add_argument('--batch_transform_mixed', type=int, default=0, help='with --batch_transform: apply it to random chunks of this many rows of the (multi-subject) batch, with k=2 directions, as the sealed grader serves shuffled mixed batches (0 = per subject group)')
    parser.add_argument('--batch_transform_ab', type=float, nargs=2, default=[0.5, 0.5], help='(a, b) strengths of the training-time --batch_transform (mean, top direction)')
    parser.add_argument('--batch_transform', action='store_true', help='train and evaluate under the grader-side batch transform (score_track1.batch_transform): per subject group in a training batch, subtract half its mean and halve its top principal direction; evaluation uses random-order 64-trial batches')
    parser.add_argument('--input_mixup', action='store_true', help='REVE authors\' mixup: mix each EEG input with a shuffled one (lam ~ U(0,1)) and mix the two contrastive losses')
    parser.add_argument('--wsd_decay_start', default=0.5, type=float, help='wsd: fraction of training after which the lr decays linearly to 0 (warmup fraction is --onecycle_pct_start)')
    parser.add_argument('--fm_layer_decay', default=1.0, type=float, help='foundation-model layer-wise lr decay: transformer block i of L gets lr * decay**(L - i), embeddings decay**(L + 1); the head keeps lr (1 = off)')
    parser.add_argument('--onecycle_pct_start', default=0.1, type=float, help='OneCycleLR warmup fraction (NeuralBench Track-1 uses 0.1)')
    parser.add_argument('--ema_decay', default=0.0, type=float, help='keep an EMA of the EEG encoder weights and evaluate/checkpoint it (0 disables; 0.999 ~ a 1000-step horizon)')
    parser.add_argument('--ema_warmup', action='store_true', help='ramp the EMA decay as min(ema_decay, (1+n)/(10+n)) so early epochs do not lag')
    parser.add_argument('--codabench_eval_every', default=0, type=int, help='every N epochs, estimate the Codabench per-epoch metric on the official input (0 disables)')
    parser.add_argument('--codabench_eval_fraction', default=0.1, type=float, help='fraction of test trials per subject used by --codabench_eval_every')
    parser.add_argument('--sealed_test_metric', action='store_true', help="report the per-epoch test top-1/top-5 under the NeurIPS-2026 Track-1 sealed rule (single-epoch predictions averaged within (subject, image) before retrieval; score_track1.py's subject_agg) instead of on repetition-averaged EEG, which moves to the avg_eeg_* columns")
    parser.add_argument('--sealed_test_fraction', default=0.125, type=float, help='fraction of test trials per subject scored by --sealed_test_metric (seeded: the same trials every epoch); 0.125 = ~10 of 80 repetitions per image')
    parser.add_argument('--sealed_test_every', default=2, type=int, help='score the sealed rule every N epochs (plus the last epoch and every new best); other epochs log NaN test top-1/top-5')
    parser.add_argument('--grad_clip', default=0.0, type=float, help='clip gradient norm to this value (0 disables); NeuralBench Track-1 uses 1.0')
    parser.add_argument('--rep_avg_prob', default=0.0, type=float, help='probability of replacing a training trial by the mean of 2..R random valid repeats of the same subject and image (higher-SNR view, same target)')
    parser.add_argument('--xsubj_mix_prob', default=0.0, type=float, help='Track-1 SubjectMix: probability of mixing a training trial with a trial of the same image from another subject (dataset-side)')
    parser.add_argument('--xsubj_mix_alpha', default=1.0, type=float, help='Beta(alpha, alpha) for --xsubj_mix_prob')
    parser.add_argument('--input_lowpass_hz', default=0.0, type=float, help='zero every input frequency at or above this (FFT along time, 120 Hz data), train and test; 0 = off')
    parser.add_argument('--sam_rho', default=0.0, type=float, help='sharpness-aware minimisation radius; the ascent direction is the previous step\'s gradient (no extra forward pass); 0 = off')
    parser.add_argument('--swa_from_epoch', default=0, type=int, help='from this epoch on, replace the EMA by a uniform average of the weights after every step (SWA/SWAD); 0 = off')
    parser.add_argument('--hyperalign_prob', default=0.0, type=float, help='per-trial probability of re-rendering a training trial through the Procrustes rotation to another training subject (--hyperalign_maps)')
    parser.add_argument('--hyperalign_maps', default='/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/hyperalign_R.npy', type=str, help='(31, 31, C, C) rotations from neurips_challenge/hyperalign_maps.py')
    parser.add_argument('--band_swap_prob', default=0.0, type=float, help='probability of replacing a training trial\'s 8-30 Hz content by that of a random trial of another same-dataset subject (identity-band swap)')
    parser.add_argument('--noise_swap_prob', default=0.0, type=float, help='probability of replacing a training trial by its subject\'s evoked mean for the image + another same-dataset subject\'s single-trial residual (noise transplant)')
    parser.add_argument('--distill_targets', default='', type=str, help='per-trial teacher predictions (export_teacher.py .npy, row = train index)')
    parser.add_argument('--distill_lambda', default=0.0, type=float, help='weight of KL(teacher || student) over the batch images')
    parser.add_argument('--distill_tau', default=0.05, type=float)
    parser.add_argument('--anticorr_lambda', default=0.0, type=float, help='negative-correlation weight: penalise student mass on the in-batch images the --distill_targets members wrongly favour (no imitation)')
    parser.add_argument('--amp_scale_std', default=0.0, type=float, help='training aug: per-channel gain ~ 1 + N(0, std)')
    parser.add_argument('--spatial_jitter_deg', default=0.0, type=float, help='training aug (lever L1): re-interpolate every electrode from its neighbours after a random rotation of the montage by up to this many degrees (simulated cap displacement; 0 = off)')
    parser.add_argument('--spatial_jitter_prob', default=0.5, type=float, help='probability per training trial of --spatial_jitter_deg')
    parser.add_argument('--repeat_group_batches', action='store_true', help='batches of whole (subject, image) groups with all their repeats (lever L2; un-averaged deterministic data); pair with --multi_positive_loss')
    parser.add_argument('--repeat_mean_loss', default=0.0, type=float, help='lever L2: weight of a contrastive loss on the mean prediction of each (subject, image) group in the batch -- the sealed metric averages repeats; needs --repeat_group_batches')
    parser.add_argument('--neighbor_loss', default=0.0, type=float, help='lever L4: weight of auxiliary heads predicting the targets of the RSVP neighbours (sub-XX/train_neighbors.npy, build_alljoined_neighbors.py); training only')
    parser.add_argument('--amp_gain_range', default=0.0, type=float, help='training aug: one global gain per trial, log-uniform in [1/r, r] (0/1 = off); new-subject recording-scale robustness')
    parser.add_argument('--time_mask_len', default=0, type=int, help='training aug: zero one random window of up to this many samples')
    parser.add_argument('--init_encoder', default='', type=str, help='state dict or checkpoint_*.pth to initialise the EEG encoder from (strict=False, matching shapes only)')
    parser.add_argument('--sealed_early_stop_patience', default=0, type=int, help='with --sealed_test_metric: end training once the per-epoch sealed-rule test top-5 has not improved for this many evaluations (0 = off); the epoch reached is kept')
    parser.add_argument('--stop_after_epoch', default=0, type=int, help='end training after this epoch while the lr schedule still spans --num_epochs (0 = off); with --select_best_on fixed this epoch is kept')
    parser.add_argument('--soft_mix_beta', default=0.0, type=float, help='with --soft_target_tau: target = (1-beta) one-hot + beta soft (NeuroCLIP, beta .3)')
    parser.add_argument('--soft_target_tau', default=0.0, type=float, help='> 0: soft EEG->image contrastive targets softmax(image-image cosine / tau); 0 = one-hot')
    parser.add_argument('--asymmetric_loss', action='store_true', help='contrastive loss in the EEG->image direction only, as NeuralBench ClipLoss(symmetric=False) does; default averages both directions')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='linear LR warmup epochs when --lr_scheduler cosine')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='cosine floor LR')
    parser.add_argument(
        '--fm_resample', default='resample', choices=['resample', 'crop'],
        help='how --eeg_encoder_type LaBraM/CBraMod reduces a 250-sample epoch to the 200 samples '
             'the foundation model consumes: resample the same one-second window to 200 Hz '
             '(default), or crop the first 200 samples',
    )
    parser.add_argument(
        '--reve_input_normalization', default='trial_zscore', choices=['trial_zscore', 'none'],
        help="per-trial z-score + clip 15 inside the REVE wrapper, or 'none' to feed the cached "
             "tensors unchanged; the 0.29 reference checkpoint used 'none'",
    )
    parser.add_argument(
        '--fm_unfreeze_last_k', default=0, type=int,
        help='train only the last K foundation-model layers plus the head (0 = full fine-tune); '
             'REVE has 22 layers of 3.14M params each',
    )
    parser.add_argument(
        '--fm_head', default='mean', choices=['mean', 'mlp', 'attnstat', 'flatten', 'multilayer'],
        help="foundation-model head: token mean -> Linear, or (REVE only) REVE's own query-pool + "
             "flatten-all-tokens -> RMSNorm -> dropout -> Linear",
    )
    parser.add_argument('--fm_arch_kwargs', default='', type=str, help='JSON constructor overrides for a --fm_no_pretrained LaBraM/REVE (num_layers, embed_dim, patch_size, ...)')
    parser.add_argument('--fm_drop_path', default=0.0, type=float, help='stochastic depth: LaBraM (its fine-tuning recipe uses 0.1) and REVE (every attention/feed-forward branch, linear in depth)')
    parser.add_argument('--fm_head_dropout', default=0.0, type=float, help='dropout before the --fm_head flatten Linear')
    parser.add_argument(
        '--fm_backbone_lr_scale', default=1.0, type=float,
        help='foundation-model backbone lr = learning_rate * scale; the head and everything else '
             'keep learning_rate (EEG-FM-Bench: 0.1-0.5)',
    )
    parser.add_argument(
        '--fm_head_warmup_epochs', default=0, type=int,
        help='keep the foundation-model backbone frozen for the first N epochs so the head is not '
             'random when the backbone starts moving (REVE two-stage fine-tuning)',
    )
    parser.add_argument(
        '--fm_no_pretrained', action='store_true',
        help='build LaBraM/CBraMod from scratch instead of the public checkpoint (control arm)',
    )
    parser.add_argument('--num_workers', default=0, type=int, help='number of dataloader workers')
    parser.add_argument('--output_dir', default='./result', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--train_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument(
        '--train_subject_weights', default=None, nargs='+', type=float,
        help='positive loss weights aligned with --train_subject_ids; normalized to mean one',
    )
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--data_average', action='store_true')
    parser.add_argument('--data_random', action='store_true')
    parser.add_argument('--init_temperature', default=0.07, type=float)
    parser.add_argument('--t_learnable', action='store_true')
    parser.add_argument('--softplus', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--mse_on_raw', action='store_true', help='ENIGMA-faithful: compute the MSE term on the UNnormalized CLIP target (learns magnitude/geometry; InfoNCE stays normalized). ATM/repo default normalizes the MSE target -> cosine-only -> needs a diffusion prior for magnitude.')
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--img_l2norm', action='store_true')
    parser.add_argument('--text_l2norm', action='store_true')
    parser.add_argument('--eeg_l2norm', action='store_true')
    parser.add_argument(
        '--eeg_confidence_mode',
        type=str,
        choices=['none', 'detached_prenorm', 'learned'],
        default='none',
        help='optional EEG-side confidence scaling while keeping alignment geometry L2-normalized',
    )
    parser.add_argument(
        '--eeg_confidence_hidden_dim',
        default=0,
        type=int,
        help='hidden width for learned EEG confidence head (0 = linear head)',
    )
    parser.add_argument(
        '--save_eeg_norm_metadata',
        action='store_true',
        help='export per-sample pre-normalization EEG alignment norms (training split, best checkpoint)',
    )
    parser.add_argument('--multi_positive_loss', action='store_true', help='enable multi-positive EEG-image contrastive loss')
    parser.add_argument('--neighbour_batch_prob', default=0.0, type=float, help='probability per batch of drawing it from one k-means cluster of the image targets (fine-grained negatives)')
    parser.add_argument('--grouped_batch_sampler', action='store_true', help='sample multiple subjects per exact image in each batch')
    parser.add_argument('--samples_per_image', default=4, type=int, help='number of samples per exact image for grouped batches')
    parser.add_argument('--subject_mixup_within', action='store_true', help='rebuttal arm: Beta-mix 2 of a subject''s OWN reps for the same stimulus (within-subject; isolates the cross-subject factor). Needs un-averaged reps (no --data_average).')
    parser.add_argument('--within_mix_alpha', default=0.5, type=float, help='Beta(a,a) coefficient for --subject_mixup_within')
    parser.add_argument('--bootstrap_repetition_average', action='store_true', help='resample each source subject`s repeated measurements with replacement before cross-subject augmentation; requires un-averaged training data')
    parser.add_argument('--bootstrap_repetition_count', default=4, type=int, help='number of within-subject repetitions in each bootstrap average')
    parser.add_argument('--subject_ea_align', action='store_true', help='rebuttal arm: per-subject Euclidean Alignment (channel-covariance whitening) of raw EEG at load, applied to train/val/test')
    parser.add_argument('--inductive_covariance_align', action='store_true', help='source-only covariance prior blended with each query covariance independently before the EEG encoder')
    parser.add_argument('--inductive_covariance_alpha', default=0.4, type=float, help='single-query covariance weight for --inductive_covariance_align')
    parser.add_argument('--inductive_covariance_shrinkage', default=0.25, type=float, help='identity shrinkage after source/query covariance blending')
    parser.add_argument('--inductive_covariance_ref_trials', default=2048, type=int, help='deterministic training trials sampled per source covariance estimate')
    parser.add_argument('--inductive_covariance_reference_cache', default='', type=str, help='optional fold-specific cache for source covariance estimates')
    parser.add_argument('--cross_subject_average', action='store_true', help='train on synthetic per-image trials, each the mean of a random-size (Beta-drawn) random subset of cross-subject raw recordings')
    parser.add_argument('--xavg_beta_a', default=1.0, type=float, help='Beta(a,b) shape a for the k distribution in --cross_subject_average (a<b skews to fewer recordings)')
    parser.add_argument('--xavg_beta_b', default=1.0, type=float, help='Beta(a,b) shape b for the k distribution in --cross_subject_average')
    parser.add_argument('--xavg_kmin', default=4, type=int, help='minimum recordings averaged per synthetic trial')
    parser.add_argument('--xavg_kmax', default=36, type=int, help='maximum recordings averaged per synthetic trial')
    parser.add_argument('--subject_mixup_mode', type=str, choices=['none', 'raw_eeg', 'embedding'], default='none', help='cross-subject same-stimulus convex mixing mode')
    parser.add_argument('--mixup_type', type=str, choices=['pairwise', 'group', 'hybrid_group_pairwise', 'leave_one_out_group', 'coherent_group', 'meta_residual'], default='pairwise', help='pairwise mixing, per-stimulus group mixing, their rowwise hybrid, leave-one-person-out group mixing, coherent cohorts, or centered auxiliary-cohort residual injection')
    parser.add_argument('--subject_mixup_alpha', default=1.0, type=float, help='beta(alpha, alpha) coefficient for cross-subject same-stimulus mixup')
    parser.add_argument('--subject_mixup_prob', default=1.0, type=float, help='per-row probability of using the synthetic subject; lower values retain pure empirical rows instead of anchoring every mixture')
    parser.add_argument('--subject_mixup_views', default=1, type=int, help='independent full group-mix cohorts per exact image in each batch (1 keeps historical batch shape)')
    parser.add_argument('--subject_mixup_hard_view', action='store_true', help='with multiple group draws, train on the currently worst-aligned draw for each base row')
    parser.add_argument('--subject_mixup_rows_per_image', default=0, type=int, help='after group synthesis retain at most this many virtual rows per exact image (0 keeps all rows)')
    parser.add_argument('--mix_time_varying', default=0, type=int, help='number of smooth temporal control points for pairwise SubjectMix coefficients (0/1 keeps one scalar per trial)')
    parser.add_argument('--hybrid_group_prob', default=0.5, type=float, help='rowwise probability of scalar group mixing in hybrid_group_pairwise; remaining rows use pairwise mixing')
    parser.add_argument('--group_mix_self_anchor', default=0.0, type=float, help='for group mix, retain this fixed fraction of each row`s original subject before adding its Dirichlet mixture')
    parser.add_argument('--subject_mixup_partition_boundary', default=0, type=int, help='if positive, mix subjects below and at/above this ID separately (used to avoid waveform mixing across EEG datasets)')
    parser.add_argument('--meta_residual_scale', default=0.5, type=float, help='scale of centered auxiliary-cohort residuals for --mixup_type meta_residual')
    parser.add_argument('--cross_dataset_translation', default='', type=str, help='NPZ affine channel map fitted on paired training-stimulus cohort means; applied only to auxiliary subjects before mixing')
    parser.add_argument('--cross_dataset_translation_blend', default=1.0, type=float, help='interpolation from raw auxiliary EEG (0) to the fitted primary-domain counterfactual (1)')
    parser.add_argument('--subject_mixup_reg_lambda', default=0.0, type=float, help='weight for raw_eeg pairwise mixup embedding regularization')
    parser.add_argument('--single_emb_stop_grad', action='store_true', help='detach original single-subject embeddings in mixup regularization')
    parser.add_argument('--subject_mixup_reg_on_backbone', action='store_true', help='compute mixup regularization on backbone features instead of aligned embeddings')
    parser.add_argument('--spectral_style_mix_prob', default=0.0, type=float, help='probability per batch of same-stimulus Dirichlet mixing in log Fourier-magnitude space while retaining each trial`s phase')
    parser.add_argument('--spectral_style_mix_alpha', default=0.5, type=float, help='Dirichlet concentration for --spectral_style_mix_prob')
    parser.add_argument('--coherent_latency_max_shift', default=0, type=int, help='maximum absolute training-only latency shift in samples; one sampled shift is reused for every stimulus of a virtual subject for an epoch')
    parser.add_argument('--coherent_latency_std', default=2.0, type=float, help='standard deviation in samples of the bounded epoch-coherent subject latency distribution')
    parser.add_argument('--subject_mixstyle_prob', default=0.0, type=float, help='probability per training batch of subject-coherent MixStyle after the TSConv temporal block (0 disables)')
    parser.add_argument('--subject_mixstyle_alpha', default=0.5, type=float, help='Beta(a,a) interpolation coefficient for virtual-subject temporal styles')
    parser.add_argument('--subject_mixstyle_granularity', choices=['filter', 'filter_channel'], default='filter_channel', help='statistics retained by subject-coherent temporal MixStyle')
    parser.add_argument('--subject_style_manifold_prob', default=0.0, type=float, help='probability per training batch of sampling coherent low-rank subject styles after the TSConv temporal block (0 disables)')
    parser.add_argument('--subject_style_manifold_rank', default=2, type=int, help='PCA rank of the source-subject temporal mean/log-scale manifold')
    parser.add_argument('--subject_style_manifold_scale', default=1.0, type=float, help='standard-deviation multiplier for sampled subject-style latent codes')
    parser.add_argument('--subject_style_manifold_max_radius', default=2.0, type=float, help='per-coordinate truncation radius for Gaussian subject-style codes')
    parser.add_argument('--coherent_mix_scope', choices=['epoch', 'run'], default='epoch', help='how long coherent_group virtual-subject mixture weights are reused')
    parser.add_argument('--coherent_mix_self_anchor', default=0.0, type=float, help='fixed weight retained on each virtual subject`s corresponding real subject (0 <= value < 1)')
    parser.add_argument('--coherent_mix_residual_scale', default=1.0, type=float, help='scale virtual-subject deviations from the per-stimulus cohort centroid; >1 extrapolates beyond the convex hull')
    parser.add_argument('--coherent_mix_jitter', default=0.0, type=float, help='blend stable coherent weights with fresh per-stimulus Dirichlet weights (0=fully coherent, 1=ordinary group mixing)')
    parser.add_argument('--concept_positives', action='store_true', help='treat every image of the same THINGS concept as a positive (test retrieval is concept-level: 200 concepts, 1 image each)')
    parser.add_argument('--images_per_concept', default=1, type=int, help='images of the same concept drawn into one batch; >1 gives --concept_positives something to pull together')
    parser.add_argument('--shared_encoder_hidden', default=0, type=int, help='SAMGA shared encoder g(): hidden width of the MLP applied to BOTH modalities after their projectors (0 disables)')
    parser.add_argument('--mmd_lambda', default=0.0, type=float, help='weight of the multi-kernel MMD cross-modal distribution term (SAMGA coarse stage)')
    parser.add_argument('--coarse_epochs', default=0, type=int, help='epochs of the coarse stage; afterwards the shared encoder is frozen and MMD is dropped (0 = never switch)')
    parser.add_argument('--eeg_instance_norm', action='store_true', help='per-trial per-channel standardization of raw EEG before the encoder (train and test alike)')
    parser.add_argument('--eeg_car', action='store_true', help='common average reference of raw EEG before the encoder, per trial (train and test alike)')
    parser.add_argument('--things_coral', action='store_true', help='recolour each THINGS-EEG2 training subject (IDs 21-30) to the Alljoined training subjects\' mean spatial covariance (CORAL); Alljoined/test inputs untouched')
    parser.add_argument('--op_transfer_prob', default=0.0, type=float, help='probability per batch of re-rendering raw EEG through another subject`s channel-mixing operator (counterfactual do(subject:=t))')
    parser.add_argument('--op_transfer_shrink', default=0.05, type=float, help='covariance shrinkage for --op_transfer_prob')
    parser.add_argument('--boost_gamma', default=0.0, type=float, help='boosting: weight each trial by (1 - ensemble prob of its true image among the batch)^gamma, from --distill_targets (the members\' training predictions); needs --multi_positive_loss')
    parser.add_argument('--op_transfer_bands', default=[], nargs='+', type=float, help='band edges in Hz (e.g. 0 4 8 13 30 61): transfer each band with its own covariances (120 Hz data)')
    parser.add_argument('--op_transfer_cross_dataset', action='store_true', help='allow Alljoined <-> THINGS-EEG2 partners (default: same dataset)')
    parser.add_argument('--op_transfer_rotate', action='store_true', help='control: rotate the partner covariance by a random orthogonal matrix')
    parser.add_argument('--op_transfer_extrapolate', default=0.0, type=float, help='if >0, draw the target operator at u~U(0,this) along the SPD geodesic through a partner subject; u>1 synthesizes subjects outside the training hull')
    parser.add_argument('--brain2brain_lambda', default=0.0, type=float, help='weight for same-stimulus cross-subject EEG<->EEG InfoNCE (stimulus-grounded invariance, no image target)')
    parser.add_argument('--global_negatives', default=0.0, type=float, help='weight of an extra EEG->image cross-entropy against ALL training-image targets (frozen, so exact)')
    parser.add_argument('--global_concept_eps', default=0.0, type=float, help='global-negatives target: move eps of the mass to the other images of the same THINGS concept')
    parser.add_argument('--global_soft_tau', default=0.0, type=float, help='global-negatives target: softmax(image-image cosine / tau) over all training images instead of one-hot')
    parser.add_argument('--global_symmetric', default=0.0, type=float, help='with global negatives: + weight x image->EEG cross-entropy over the batch queries')
    parser.add_argument('--mask_concept_neg', action='store_true', help='in-batch InfoNCE without same-concept different-image negatives (false-negative elimination)')
    parser.add_argument('--loss_trunc', default=0.0, type=float, help='drop this fraction of the highest per-trial InfoNCE losses in each batch (robust to uninformative trials)')
    parser.add_argument('--siglip', action='store_true', help='in-batch pairwise sigmoid loss (SigLIP) instead of InfoNCE')
    parser.add_argument('--siglip_bias', default=-10.0, type=float, help='initial SigLIP bias (learned)')
    parser.add_argument('--global_negatives_replace', action='store_true', help='use the all-training-images cross-entropy INSTEAD of the in-batch contrastive loss')
    parser.add_argument('--relational_lambda', default=0.0, type=float, help='weight for RKD-style relational loss: EEG-EEG similarity structure matches image-image structure')
    parser.add_argument('--pr_lambda', default=0.0, type=float, help='weight of the collapse guard relu(1/PR - 1/pr_floor) on the batch EEG cloud (CVPR E8.11c)')
    parser.add_argument('--pr_floor', default=14.0, type=float, help='participation-ratio floor of the collapse guard (ATM-128 test clouds: 14.3)')
    parser.add_argument('--iso_lambda', default=0.0, type=float, help='weight of the cross-subject isometry loss: mean residual of the best rotation between two source subjects\' embeddings of the batch\'s shared images (CVPR E8.11); logged in REL')
    parser.add_argument('--rsa_lambda', default=0.0, type=float, help='weight of 1 - RSA (Pearson corr of centred EEG-EEG vs image-image cosines, same-image pairs dropped; CVPR E8.4); logged in REL')
    parser.add_argument('--relational_temperature', default=0.07, type=float, help='temperature for --relational_lambda similarity structures')
    parser.add_argument('--auxiliary_alignment_dims', default='', type=str,
                        help='comma-separated training-only alignment widths sharing one encoder')
    parser.add_argument('--auxiliary_alignment_lambda', default=0.0, type=float,
                        help='weight of training-only multi-resolution contrastive supervision')
    parser.add_argument('--auxiliary_fixed_targets', action='store_true',
                        help='AUG_LOG 2026-10-30: auxiliary heads align to the TRUE training-only target embeddings (identity image side; '
                             '--auxiliary_alignment_dims must equal those target dims) instead of a learned projection of them')
    parser.add_argument('--auxiliary_alignment_epochs', default=0, type=int,
                        help='apply auxiliary alignment heads for first N epochs (0 = all)')
    parser.add_argument('--teacher_alignment_lambda', default=0.0, type=float,
                        help='cosine distillation weight from the final training-only target block into the main EEG alignment space')
    parser.add_argument(
        '--auxiliary_image_targets', action='store_true',
        help=(
            'interpret comma-separated --image_feature_dir entries as one primary '
            'inference target followed by training-only semantic targets; auxiliary '
            'heads are not saved in the inference checkpoint'
        ),
    )
    parser.add_argument('--matryoshka_dims', default='', type=str,
                        help='comma-separated prefix widths for nested contrastive supervision of the final embedding')
    parser.add_argument('--matryoshka_lambda', default=0.0, type=float,
                        help='weight of nested-prefix contrastive supervision (single projector and final embedding)')
    parser.add_argument('--matryoshka_epochs', default=0, type=int,
                        help='apply nested-prefix supervision for first N epochs (0 = all epochs)')
    parser.add_argument('--vrex_lambda', default=0.0, type=float, help='weight for V-REx penalty: variance across training subjects of the per-subject cross-modal risk')
    parser.add_argument('--train_saw', action='store_true', help='apply subject-wise whitening to EEG embeddings before cross-modal losses during training')
    parser.add_argument('--train_saw_whiten_image', action='store_true', help='also whiten image embeddings per subject batch when --train_saw is enabled')
    parser.add_argument('--train_saw_shrink', default=0.2, type=float, help='covariance shrinkage for train-time subject whitening')
    parser.add_argument('--train_saw_diag', action='store_true', help='use diagonal covariance for train-time subject whitening')
    parser.add_argument('--train_saw_no_renorm', action='store_true', help='disable L2 renormalization after train-time subject whitening')
    parser.add_argument('--eval_mode', type=str, choices=['plain_cosine', 'saw', 'csls', 'saw_csls'], default='plain_cosine', help='test-time retrieval: cosine, SAW whitening, or SAW + fixed-k CSLS')
    parser.add_argument('--sattc_saw_shrink', default=0.2, type=float, help='covariance shrinkage for subject-adaptive whitening (SAW) during evaluation')
    parser.add_argument('--sattc_saw_diag', action='store_true', help='use diagonal covariance for SAW during evaluation')
    parser.add_argument('--sattc_csls_k', default=2, type=int, help='fixed neighborhood size k for CSLS after SAW (eval_mode saw_csls)')
    parser.add_argument('--sattc_cw', action='store_true', help='apply candidate-side whitening (CW) to the retrieval bank during evaluation')
    parser.add_argument('--sattc_cw_shrink', default=0.05, type=float, help='covariance shrinkage for candidate-side whitening (CW)')
    parser.add_argument('--sattc_cw_diag', action='store_true', help='use diagonal covariance for candidate-side whitening (CW)')
    parser.add_argument('--val_subject_id', default=None, type=int, help='subject ID used for validation model selection')
    parser.add_argument('--val_concept_ratio', default=0.0, type=float, help='hold out this fraction of TRAINING CONCEPTS (all their images, all training subjects) as a validation split for checkpoint selection; keeps all training subjects, unlike --val_subject_id')
    parser.add_argument('--val_concept_seed', default=20260822, type=int, help='seed choosing the held-out concepts; constant across folds and arms on purpose so every arm trains on identical data')
    parser.add_argument('--save_testsel_checkpoint', action='store_true', help='additionally save checkpoint_testsel_best.pth (best held-out test loss) alongside the selected checkpoint, as a matched control for val-selected runs')
    parser.add_argument(
        '--select_best_metric',
        type=str,
        choices=['loss', 'top5'],
        default='loss',
        help=(
            'rank checkpoints by the selection split loss (historical default) or by its '
            'top-5 accuracy; see neurips_challenge/README.md for why top5 is better here'
        ),
    )
    parser.add_argument(
        '--select_best_on',
        type=str,
        choices=['test', 'val', 'fixed'],
        default='test',
        help=(
            'checkpoint rule: held-out test loss, source validation loss, or the '
            'final predeclared epoch (fixed; no label-based checkpoint selection)'
        ),
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=0,
        help=(
            'stop after this many epochs without source-validation loss improvement; '
            '0 disables and nonzero values require --select_best_on val'
        ),
    )
    parser.add_argument(
        '--early_stop_min_delta',
        type=float,
        default=0.0,
        help=(
            'minimum source-validation loss decrease that resets early-stop '
            'patience; checkpoint selection still retains every exact improvement'
        ),
    )
    parser.add_argument('--subject_probe_holdout', action='store_true', help='per-subject held-out split; train linear subject probes (baseline only)')
    parser.add_argument('--subject_probe_holdout_ratio', type=float, default=0.10, help='fraction per train subject reserved for probe validation')
    parser.add_argument('--subject_probe_interval', type=int, default=1, help='train/eval subject probes every N epochs')
    parser.add_argument('--subject_probe_layers', nargs='+', default=['backbone', 'align'], help='layers to probe: temporal, spatial, backbone, align')
    parser.add_argument('--subject_adapt_lambda', default=0.0, type=float, help='weight for unlabeled subject-adaptation loss fit on split A and supervised on split B')
    parser.add_argument('--subject_adapt_split_a_ratio', default=0.5, type=float, help='per-subject within-batch ratio reserved for unlabeled split A')
    parser.add_argument('--subject_adapt_min_samples_per_subject', default=8, type=int, help='minimum number of samples for a subject to contribute to subject adaptation')
    parser.add_argument('--subject_adapt_saw_shrink', default=0.85, type=float, help='SAW shrinkage used when fitting unlabeled subject adaptation')
    parser.add_argument('--subject_adapt_saw_diag', action='store_true', help='use diagonal covariance for subject-adaptation SAW')
    parser.add_argument('--subject_adapt_saw_no_renorm', action='store_true', help='disable pre-map L2 renorm inside subject adaptation')
    parser.add_argument('--subject_adapt_csls_k', default=1, type=int, help='CSLS neighborhood size for subject-adaptation Sinkhorn fitting')
    parser.add_argument('--subject_adapt_sinkhorn_tau', default=0.1, type=float, help='Sinkhorn temperature for subject adaptation')
    parser.add_argument('--subject_adapt_sinkhorn_iters', default=20, type=int, help='Sinkhorn iterations for subject adaptation')
    parser.add_argument('--subject_adapt_soft_procrustes_steps', default=10, type=int, help='number of Sinkhorn+Procrustes refinement steps for subject adaptation')
    parser.add_argument('--subject_adapt_soft_procrustes_power', default=1.0, type=float, help='power exponent for soft assignment weights in subject adaptation')
    parser.add_argument('--eeg_data_dir', default='./things_eeg/data/preprocessed_eeg', type=str, help='where your EEG data are')
    parser.add_argument("--selected_channels", default=[], nargs='*', type=str, help="selected EEG channels, empty means all channels")
    parser.add_argument('--time_window', type=int, default=[0, 250], nargs=2, help='time window for EEG data, in sample points')
    parser.add_argument('--eeg_aug', action='store_true')
    parser.add_argument('--eeg_aug_type', type=str, choices=['noise', 'time_shift', 'channel_dropout', 'smooth'], default='noise', help='eeg augmentation type')
    parser.add_argument(
        '--eeg_encoder_type',
        type=str,
        default='EEGProject',
    )
    parser.add_argument('--init_checkpoint', default='', type=str,
                        help='initialize the encoder and projectors from a compatible training checkpoint')
    parser.add_argument('--init_checkpoint_scope', choices=['all', 'encoder', 'conv'], default='all',
                        help='load all compatible modules, only the EEG encoder, or only TSConv convolution weights from --init_checkpoint')
    parser.add_argument('--freeze_bn_stats', action='store_true', help='BatchNorm layers stay in eval mode during training (few-shot oracles)')
    parser.add_argument('--train_max_trials', default=0, type=int, help='train on a seeded random subset of this many trials (few-shot oracles, SUBJECT_LOG D8)')
    parser.add_argument('--lora_rank', default=0, type=int, help='rank of LoRA adapters on the --lora_regex Linear layers (SUBJECT_LOG D5)')
    parser.add_argument('--lora_regex', default='proj_eeg|eproj', type=str, help='"model.<name>" / "eproj<name>" of the Linear layers to adapt')
    parser.add_argument('--trainable_regex', default='', type=str,
                        help='train only parameters whose "model.<name>" / "eproj.<name>" matches (fine-tune oracles)')
    parser.add_argument('--freeze_eeg_encoder', action='store_true',
                        help='freeze an initialized EEG encoder in eval mode and train only downstream modules')
    parser.add_argument('--freeze_image_projector', action='store_true',
                        help='keep the image projector fixed after full checkpoint initialization')
    parser.add_argument('--per_subject_test_log', action='store_true',
                        help='pooled multi-subject runs: every epoch, log each test subject\'s loss/top1/top5 '
                             '(its own 200-row batch, unique gallery) to per_subject_test.csv, so the best epoch '
                             'can be picked per subject as a single-subject run would')
    parser.add_argument('--eval_unique_gallery', action='store_true',
                        help='rank val/test queries against each DISTINCT stimulus once instead of '
                             'one row per presentation; without it repeated presentations put '
                             'identical vectors in the gallery and cap top-5 at ~5/n_repeats')
    parser.add_argument('--tsconv_head_no_ln', action='store_true', help='drop the LayerNorm at the end of the TSConv head trunk (lets the output norm vary per trial)')
    parser.add_argument('--tsconv_backbone_dim', default=0, type=int, help='width of the TSConv residual trunk before a separate trunk->feature_dim layer (0 = trunk at feature_dim, no extra layer); NeuralBench Track1TSConv uses 1024')
    parser.add_argument('--tsconv_temporal_filters', default=40, type=int, help='temporal filter count for TSConv_parameterizable')
    parser.add_argument('--tsconv_temporal_kernel', default=25, type=int, help='temporal kernel width for TSConv_parameterizable')
    parser.add_argument('--tsconv_power_branch', action='store_true', help='TSConv_parameterizable: add a ShallowConvNet band-power branch (square -> pool -> log) before the trunk')
    parser.add_argument('--tsconv_pool_kernel', default=51, type=int, help='temporal average-pooling kernel for TSConv_parameterizable')
    parser.add_argument('--sqf_d_model', default=160, type=int, help='FastTSConvSqueezeformer token width (divisible by its 4 heads)')
    parser.add_argument('--arch_kwargs', default='', type=str, help='JSON dict of constructor kwargs for registry encoders (ortho_encoders.py)')
    parser.add_argument('--sqf_depth', default=3, type=int, help='FastTSConvSqueezeformer number of blocks')
    parser.add_argument('--tsconv_pool_stride', default=5, type=int, help='temporal average-pooling stride for TSConv_parameterizable')
    parser.add_argument('--tsconv_spatial_filters', default=40, type=int, help='spatial filter count for TSConv_parameterizable')
    parser.add_argument('--tsconv_projection_filters', default=40, type=int, help='1x1 projection filter count for TSConv_parameterizable')
    parser.add_argument('--tsconv_activation', default='elu', choices=['elu', 'gelu', 'relu', 'leaky_relu', 'silu'], help='activation for TSConv_parameterizable')
    parser.add_argument('--tsconv_dropout', default=0.5, type=float, help='backbone dropout for TSConv_parameterizable')
    parser.add_argument('--tsconv_head_dropout', default=-1.0, type=float, help='projection-head dropout for TSConv_parameterizable; negative reuses --tsconv_dropout')
    parser.add_argument('--tsconv_no_batch_norm', action='store_true', help='disable BatchNorm layers in TSConv_parameterizable')
    parser.add_argument('--tsconv_domain_spatial_boundary', default=0, type=int, help='separate TSConv spatial (electrode-mixing) conv for subject IDs at or above this one, e.g. the THINGS-EEG2 subjects mapped onto the Alljoined montage (0 disables)')
    parser.add_argument('--extra_trials_dir', default='', type=str, help='append another stimulus set\'s single trials to training (Alljoined-1: /nasbrain/p20fores/alljoined/alljoined1_120Hz)')
    parser.add_argument('--extra_subject_offset', default=40, type=int, help='subject id of extra trials = offset + their subject number (Alljoined-1: 41..48)')
    parser.add_argument('--extra_input_adapter', action='store_true', help='learned identity-init temporal+channel adapter applied to extra-trial subjects only')
    parser.add_argument('--latency_align', default=0, type=float, help='learned per-trial input time shift within +-this many samples (model.LatencyAligner); 0 = off')
    parser.add_argument('--norm_type', default='batch', choices=['batch', 'group', 'instance'], help='replace every BatchNorm of the encoder by per-trial GroupNorm(1, C) (group) or GroupNorm(C, C) (instance); drops the per-dataset split')
    parser.add_argument('--tsconv_domain_bn_boundary2', default=0, type=int, help='with --tsconv_domain_bn_boundary: a third BatchNorm path for subject IDs >= this (THINGS-EEG1 pseudo-subjects, 31)')
    parser.add_argument('--tsconv_domain_bn_boundary', default=0, type=int, help='use separate primary/auxiliary TSConv BatchNorm paths split at this subject ID (0 disables)')
    parser.add_argument('--tsconv_subject_bn_max_id', default=0, type=int, help='use source-person TSConv BatchNorm paths plus population path 0 (0 disables)')
    parser.add_argument('--tsconv_no_conv_bias', action='store_true', help='disable convolution biases in TSConv_parameterizable')
    parser.add_argument('--atm_d_model', default=250, type=int, help='ATM channel-token hidden dimension')
    parser.add_argument('--atm_n_heads', default=4, type=int, help='ATM attention head count')
    parser.add_argument('--atm_e_layers', default=1, type=int, help='ATM attention layer count')
    parser.add_argument('--atm_d_ff', default=256, type=int, help='ATM feed-forward hidden dimension')
    parser.add_argument('--atm_attention_dropout', default=0.25, type=float, help='ATM attention-stack dropout')
    parser.add_argument('--atm_temporal_filters', default=40, type=int, help='ATM readout temporal filter count')
    parser.add_argument('--atm_temporal_kernel', default=25, type=int, help='ATM readout temporal kernel width')
    parser.add_argument('--atm_pool_kernel', default=51, type=int, help='ATM readout pooling kernel width')
    parser.add_argument('--atm_pool_stride', default=5, type=int, help='ATM readout pooling stride')
    parser.add_argument('--atm_spatial_filters', default=40, type=int, help='ATM readout spatial filter count')
    parser.add_argument('--atm_projection_filters', default=40, type=int, help='ATM readout projection filter count')
    parser.add_argument('--atm_conv_dropout', default=0.5, type=float, help='ATM convolutional readout dropout')
    parser.add_argument('--atm_backbone_dim', default=0, type=int, help='ATM head trunk width; 0 = feature_dim (no separate output Linear)')
    parser.add_argument('--atm_no_subject_token', action='store_true', help='drop the ATM subject token (keeps all 63 channels; grader gives no subject ids)')
    parser.add_argument('--subject_adapter_rank', default=0, type=int, help='rank of learned source-person FiLM random effects at the temporal TSConv block (0 disables)')
    parser.add_argument('--subject_adapter_max_id', default=32, type=int, help='largest subject ID addressable by the source-person adapter; ID 0 is the population path')
    parser.add_argument('--subject_adapter_scale', default=0.1, type=float, help='residual scale of channelwise subject FiLM effects')
    parser.add_argument('--subject_adapter_virtual_prob', default=0.0, type=float, help='probability that each virtual batch identity draws a source-embedding Gaussian random effect')
    parser.add_argument('--subject_adapter_virtual_scale', default=1.0, type=float, help='posterior-predictive scale around the empirical source-adapter distribution')
    parser.add_argument('--subject_adapter_inferred', action='store_true', help='infer a continuous rank-R subject FiLM code from each EEG observation, including unseen people')
    parser.add_argument('--subject_adapter_dropout', default=0.0, type=float, help='probability of replacing a training trial\'s adapter ID by the population ID 0 (trains the path inference uses)')
    parser.add_argument('--subject_adapter_router_lambda', default=0.0, type=float, help='source-ID classification weight anchoring the amortized subject-code router')
    parser.add_argument('--mutual_reconstruction_lambda', default=0.0, type=float, help='weight for exact-stimulus cross-person EEG reconstruction through source random effects')
    parser.add_argument('--mutual_reconstruction_hidden', default=256, type=int, help='hidden width of the training-time conditional EEG reconstruction decoder')
    parser.add_argument('--cohort_reconstruction_lambda', default=0.0, type=float, help='weight for decoding the exact-image source-cohort EEG centroid from each content representation')
    parser.add_argument('--cohort_reconstruction_hidden', default=256, type=int, help='hidden width of the training-only cohort-mean EEG decoder')
    parser.add_argument('--image_aug', action='store_true')
    parser.add_argument('--image_test_aug', action='store_true')
    parser.add_argument('--eeg_test_aug', action='store_true')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')
    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'mlp', 'subspace', 'subspace_c', 'bottleneck', 'bottleneck_do'], default='direct', help='subspace (lever L3): the EEG encoder outputs --eeg_backbone_dim coefficients of the top principal directions of the training targets')
    parser.add_argument(
        '--projector_activation',
        type=str,
        choices=['none', 'relu', 'relu_gelu', 'topk'],
        default='none',
        help='terminal activation on alignment projectors (relu/relu_gelu/topk = Sparse CLIP-style sparse embeddings)',
    )
    parser.add_argument(
        '--projector_topk',
        type=int,
        default=512,
        help='keep top-k dims per sample when --projector_activation topk (paper small-scale ablation uses 512)',
    )
    parser.add_argument(
        '--logit_scale_max',
        type=float,
        default=None,
        help='optional cap on learnable contrastive temperature (Sparse CLIP sparsity knob)',
    )
    parser.add_argument('--feature_dim', type=str, default='512', help='shared alignment-space dimension when projector is not direct')
    parser.add_argument('--eeg_backbone_dim', type=int, default=0, help='EEG encoder output dimension (0 means use image feature dimension)')
    parser.add_argument('--image_feature_dir', default='/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit', type=str, help='where your image feature are')
    parser.add_argument('--aug_image_feature_dirs', default=[], nargs='+', type=str, help='where your augmentation image feature are')
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')
    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    parser.add_argument('--train_object_fraction', default=1.0, type=float, help='train on this fraction of the training concepts (fixed seed-0 subset, all subjects); CVPR E8.8b volume control')
    parser.add_argument('--save_epoch_checkpoints', nargs='*', type=int, default=[], help='also save checkpoint_epXXX.pth at these epochs (1-based; CVPR E8.10 TTA-vs-training-length curves)')
    parser.add_argument('--save_last_checkpoint', action='store_true', help='also write checkpoint_last.pth (final epoch); off by default since nothing reads it and it doubles checkpoint disk use')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    parser.add_argument('--train_rng_seed', type=int, default=None,
                        help='optional RNG reset after module initialization, for matched batch/mixup draws across model variants')
    args = parser.parse_args()
    if args.projector_activation == 'topk' and args.projector_topk <= 0:
        raise ValueError('--projector_topk must be > 0 when --projector_activation topk')
    if args.subject_mixup_reg_lambda < 0:
        raise ValueError("--subject_mixup_reg_lambda must be non-negative.")
    if not 0 <= args.subject_mixup_prob <= 1:
        raise ValueError('--subject_mixup_prob must lie in [0, 1].')
    if args.subject_mixup_views <= 0:
        raise ValueError('--subject_mixup_views must be positive.')
    if args.subject_mixup_views > 1:
        if (
            args.subject_mixup_mode != 'raw_eeg'
            or args.mixup_type != 'group'
            or args.subject_mixup_prob != 1
        ):
            raise ValueError('multiple subject mixup views require pure raw-EEG group mixing.')
        if args.subject_adapter_rank > 0 or args.tsconv_subject_bn_max_id > 0:
            raise ValueError('multiple subject mixup views are not combined with source-person routing.')
    if args.subject_mixup_hard_view:
        if args.subject_mixup_views <= 1:
            raise ValueError('--subject_mixup_hard_view requires multiple subject mixup views.')
        if any((
            args.subject_mixup_reg_lambda > 0,
            args.train_saw,
            args.subject_adapt_lambda > 0,
            args.auxiliary_alignment_lambda > 0,
            args.teacher_alignment_lambda > 0,
            args.cohort_reconstruction_lambda > 0,
            args.brain2brain_lambda > 0,
            args.relational_lambda > 0,
            args.rsa_lambda > 0,
            args.iso_lambda > 0,
            args.pr_lambda > 0,
            args.mmd_lambda > 0,
            args.vrex_lambda > 0,
            args.matryoshka_lambda > 0,
        )):
            raise ValueError('hard-view selection is only supported with the base contrastive objective.')
    if args.subject_mixup_rows_per_image < 0:
        raise ValueError('--subject_mixup_rows_per_image must be non-negative.')
    if args.subject_mixup_rows_per_image > 0:
        if (
            args.subject_mixup_mode != 'raw_eeg'
            or args.mixup_type != 'group'
            or args.subject_mixup_prob != 1
            or args.subject_mixup_views != 1
        ):
            raise ValueError('row-reduced counterfactual batches require one pure raw-EEG group view.')
        if args.subject_mixup_reg_lambda > 0:
            raise ValueError('row-reduced counterfactual batches do not support pairwise mixup regularization.')
    if args.mix_time_varying < 0:
        raise ValueError('--mix_time_varying must be non-negative.')
    if args.mix_time_varying > 1 and args.mixup_type not in {'pairwise', 'group', 'hybrid_group_pairwise'}:
        raise ValueError('--mix_time_varying > 1 currently requires pairwise, group, or hybrid mixing.')
    if not 0 <= args.hybrid_group_prob <= 1:
        raise ValueError('--hybrid_group_prob must lie in [0, 1].')
    if args.coherent_latency_max_shift < 0 or args.coherent_latency_std < 0:
        raise ValueError('--coherent_latency_max_shift and --coherent_latency_std must be non-negative.')
    if args.teacher_alignment_lambda < 0:
        raise ValueError('--teacher_alignment_lambda must be non-negative.')
    if args.teacher_alignment_lambda > 0 and not args.auxiliary_image_targets:
        raise ValueError('--teacher_alignment_lambda requires --auxiliary_image_targets.')
    if not 0.0 <= args.group_mix_self_anchor < 1.0:
        raise ValueError('--group_mix_self_anchor must be in [0, 1).')
    if args.group_mix_self_anchor > 0 and args.mixup_type != 'group':
        raise ValueError('--group_mix_self_anchor requires --mixup_type group.')
    if args.subject_mixup_partition_boundary < 0:
        raise ValueError('--subject_mixup_partition_boundary must be non-negative.')
    if args.tsconv_domain_bn_boundary < 0:
        raise ValueError('--tsconv_domain_bn_boundary must be non-negative.')
    if args.tsconv_subject_bn_max_id < 0:
        raise ValueError('--tsconv_subject_bn_max_id must be non-negative.')
    if args.tsconv_domain_bn_boundary > 0 and args.tsconv_subject_bn_max_id > 0:
        raise ValueError('domain and subject TSConv BatchNorm routing are mutually exclusive.')
    if args.tsconv_domain_bn_boundary > 0:
        if args.eeg_encoder_type.removeprefix('Ortho') not in ('TSConv_parameterizable', 'FastTSConvSqueezeformer', 'ATM',
                                                                 'FactorizedSTFormer', 'ElectrodeTokenTSConv', 'FilterAttnTSConv',
                                                                 'TimeAttnFlatTSConv', 'VirtualSensorFormer', 'DynSpatialTSConv',
                                                                 'DualPathTSConv', 'TSConvPlus', 'TimeFreqXTSConv', 'SqfMix', 'GRUEncoder', 'DilatedConvNeXt1D'):
            raise ValueError('--tsconv_domain_bn_boundary requires TSConv_parameterizable, FastTSConvSqueezeformer or ATM.')
        if args.tsconv_no_batch_norm:
            raise ValueError('--tsconv_domain_bn_boundary is incompatible with --tsconv_no_batch_norm.')
    if args.tsconv_subject_bn_max_id > 0:
        if args.eeg_encoder_type != 'TSConv_parameterizable':
            raise ValueError('--tsconv_subject_bn_max_id requires TSConv_parameterizable.')
        if args.tsconv_no_batch_norm:
            raise ValueError('--tsconv_subject_bn_max_id is incompatible with --tsconv_no_batch_norm.')
        if args.tsconv_subject_bn_max_id < max(
            list(args.train_subject_ids) + list(args.test_subject_ids), default=0
        ):
            raise ValueError('--tsconv_subject_bn_max_id must cover all configured subject IDs.')
        if args.subject_mixup_mode != 'raw_eeg' or args.subject_mixup_prob >= 1:
            raise ValueError('subject BatchNorm requires raw EEG mixing with some real rows.')
    if args.subject_adapter_rank < 0 or args.subject_adapter_scale < 0:
        raise ValueError('--subject_adapter_rank and --subject_adapter_scale must be non-negative.')
    if not 0 <= args.subject_adapter_virtual_prob <= 1 or args.subject_adapter_virtual_scale < 0:
        raise ValueError('subject adapter virtual probability must be in [0,1] and scale non-negative.')
    if args.subject_adapter_rank > 0:
        if args.eeg_encoder_type != 'TSConv_parameterizable':
            raise ValueError('--subject_adapter_rank requires TSConv_parameterizable.')
        all_subject_ids = list(args.train_subject_ids) + list(args.test_subject_ids)
        if args.subject_adapter_max_id < max(all_subject_ids, default=0):
            raise ValueError('--subject_adapter_max_id must cover all configured subject IDs.')
    if args.subject_adapter_virtual_prob > 0:
        if args.subject_adapter_rank <= 0 or args.subject_mixup_mode != 'raw_eeg':
            raise ValueError('virtual adapter sampling requires a subject adapter and raw EEG mixing.')
        if args.tsconv_subject_bn_max_id > 0:
            raise ValueError('virtual adapter sampling is not combined with subject BatchNorm routing.')
    if args.subject_adapter_router_lambda < 0:
        raise ValueError('--subject_adapter_router_lambda must be non-negative.')
    if args.subject_adapter_inferred:
        if args.subject_adapter_rank <= 0 or args.eeg_encoder_type != 'TSConv_parameterizable':
            raise ValueError('--subject_adapter_inferred requires a parameterizable subject adapter.')
        if args.subject_adapter_virtual_prob > 0:
            raise ValueError('inferred and lookup-table virtual subject adapters are mutually exclusive.')
    elif args.subject_adapter_router_lambda > 0:
        raise ValueError('--subject_adapter_router_lambda requires --subject_adapter_inferred.')
    if args.mutual_reconstruction_lambda < 0 or args.mutual_reconstruction_hidden <= 0:
        raise ValueError('--mutual_reconstruction_lambda must be non-negative and its hidden width positive.')
    if args.mutual_reconstruction_lambda > 0:
        if args.subject_adapter_rank <= 0:
            raise ValueError('--mutual_reconstruction_lambda requires --subject_adapter_rank > 0.')
        if args.subject_mixup_mode == 'raw_eeg' and args.subject_mixup_prob >= 1:
            raise ValueError('mutual reconstruction needs some unchanged real rows; set --subject_mixup_prob < 1.')
    if args.cohort_reconstruction_lambda < 0 or args.cohort_reconstruction_hidden <= 0:
        raise ValueError('--cohort_reconstruction_lambda must be non-negative and its hidden width positive.')
    if args.cohort_reconstruction_lambda > 0 and args.eeg_encoder_type != 'TSConv_parameterizable':
        raise ValueError('--cohort_reconstruction_lambda requires TSConv_parameterizable.')
    if args.freeze_image_projector:
        if not args.init_checkpoint or args.init_checkpoint_scope != 'all':
            raise ValueError('--freeze_image_projector requires --init_checkpoint with scope all.')
        if args.shared_encoder_hidden > 0:
            raise ValueError('--freeze_image_projector is not supported with a shared encoder.')
    if args.freeze_eeg_encoder and not args.init_checkpoint:
        raise ValueError('--freeze_eeg_encoder requires --init_checkpoint.')
    if args.subject_mixup_partition_boundary > 0 and args.mixup_type == 'coherent_group':
        raise ValueError('--subject_mixup_partition_boundary is not implemented for coherent_group.')
    if args.meta_residual_scale < 0:
        raise ValueError('--meta_residual_scale must be non-negative.')
    if not 0.0 <= args.cross_dataset_translation_blend <= 1.0:
        raise ValueError('--cross_dataset_translation_blend must be in [0, 1].')
    if args.cross_dataset_translation and args.subject_mixup_partition_boundary <= 0:
        raise ValueError('--cross_dataset_translation requires a positive --subject_mixup_partition_boundary.')
    if args.mixup_type == 'meta_residual':
        if args.subject_mixup_mode != 'raw_eeg' or args.subject_mixup_partition_boundary <= 0:
            raise ValueError('--mixup_type meta_residual requires raw_eeg mode and a positive partition boundary.')
    if args.subject_mixup_reg_lambda > 0 and (args.subject_mixup_mode != 'raw_eeg' or args.mixup_type != 'pairwise'):
        raise ValueError("--subject_mixup_reg_lambda currently requires --subject_mixup_mode raw_eeg and --mixup_type pairwise.")
    if not 0.0 <= args.spectral_style_mix_prob <= 1.0:
        raise ValueError('--spectral_style_mix_prob must be in [0, 1].')
    if args.spectral_style_mix_alpha <= 0:
        raise ValueError('--spectral_style_mix_alpha must be positive.')
    if args.spectral_style_mix_prob > 0 and not args.grouped_batch_sampler:
        raise ValueError('--spectral_style_mix_prob requires --grouped_batch_sampler.')
    if not 0.0 <= args.subject_mixstyle_prob <= 1.0:
        raise ValueError('--subject_mixstyle_prob must be in [0, 1].')
    if args.subject_mixstyle_alpha <= 0:
        raise ValueError('--subject_mixstyle_alpha must be positive.')
    if not 0.0 <= args.subject_style_manifold_prob <= 1.0:
        raise ValueError('--subject_style_manifold_prob must be in [0, 1].')
    if args.subject_style_manifold_rank <= 0:
        raise ValueError('--subject_style_manifold_rank must be positive.')
    if args.subject_style_manifold_scale < 0:
        raise ValueError('--subject_style_manifold_scale must be non-negative.')
    if args.subject_style_manifold_max_radius <= 0:
        raise ValueError('--subject_style_manifold_max_radius must be positive.')
    if args.subject_mixstyle_prob > 0 and args.subject_style_manifold_prob > 0:
        raise ValueError('--subject_mixstyle_prob and --subject_style_manifold_prob are mutually exclusive.')
    if not 0.0 <= args.coherent_mix_self_anchor < 1.0:
        raise ValueError('--coherent_mix_self_anchor must be in [0, 1).')
    if args.coherent_mix_residual_scale < 0:
        raise ValueError('--coherent_mix_residual_scale must be non-negative.')
    if not 0.0 <= args.coherent_mix_jitter <= 1.0:
        raise ValueError('--coherent_mix_jitter must be in [0, 1].')
    if args.mixup_type == 'coherent_group':
        if args.subject_mixup_mode != 'raw_eeg':
            raise ValueError('--mixup_type coherent_group requires --subject_mixup_mode raw_eeg.')
        if not args.grouped_batch_sampler:
            raise ValueError('--mixup_type coherent_group requires --grouped_batch_sampler.')
    if args.subject_mixstyle_prob > 0:
        if args.eeg_encoder_type not in ('TSConv', 'TSConv_parameterizable'):
            raise ValueError('--subject_mixstyle_prob currently requires TSConv or TSConv_parameterizable.')
        if not args.grouped_batch_sampler:
            raise ValueError('--subject_mixstyle_prob requires --grouped_batch_sampler for balanced matched-stimulus batches.')
    if args.subject_style_manifold_prob > 0:
        if args.eeg_encoder_type not in ('TSConv', 'TSConv_parameterizable'):
            raise ValueError('--subject_style_manifold_prob currently requires TSConv or TSConv_parameterizable.')
        if not args.grouped_batch_sampler:
            raise ValueError('--subject_style_manifold_prob requires --grouped_batch_sampler for balanced matched-stimulus batches.')
    if args.subject_probe_holdout:
        if not (0.0 < args.subject_probe_holdout_ratio < 1.0):
            raise ValueError("--subject_probe_holdout_ratio must be strictly between 0 and 1.")
    if args.subject_adapt_lambda > 0:
        if not (0.0 < args.subject_adapt_split_a_ratio < 1.0):
            raise ValueError("--subject_adapt_split_a_ratio must be strictly between 0 and 1.")
        if args.subject_adapt_min_samples_per_subject < 2:
            raise ValueError("--subject_adapt_min_samples_per_subject must be at least 2.")
    if args.train_saw_whiten_image and not args.train_saw:
        raise ValueError("--train_saw_whiten_image requires --train_saw.")
    if args.subject_mixup_within:
        if args.data_average or args.cross_subject_average or args.subject_mixup_mode != 'none':
            raise ValueError("--subject_mixup_within needs un-averaged reps and replaces cross-subject mixup: drop --data_average, --cross_subject_average, and set --subject_mixup_mode none.")
    if args.bootstrap_repetition_average:
        if args.data_average or args.cross_subject_average or args.subject_mixup_within:
            raise ValueError('--bootstrap_repetition_average requires un-averaged data and is incompatible with --data_average, --cross_subject_average, and --subject_mixup_within.')
        if args.bootstrap_repetition_count < 1:
            raise ValueError('--bootstrap_repetition_count must be positive.')
    if args.cross_subject_average:
        conflicts = [name for flag, name in [
            (args.data_average, '--data_average'),
            (args.subject_mixup_mode != 'none', '--subject_mixup_mode'),
            (args.grouped_batch_sampler, '--grouped_batch_sampler'),
            (args.multi_positive_loss, '--multi_positive_loss'),
            (args.train_saw, '--train_saw'),
        ] if flag]
        if conflicts:
            raise ValueError(f"--cross_subject_average is incompatible with: {', '.join(conflicts)}")
        if not (1 <= args.xavg_kmin <= args.xavg_kmax):
            raise ValueError("require 1 <= --xavg_kmin <= --xavg_kmax")
    seed = seed_everything(seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name is not None:
        log_dir = os.path.join(args.output_dir, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}-{args.output_name}")
    else:
        log_dir = os.path.join(args.output_dir, datetime.now().strftime(r'%Y%m%d-%H%M%S'))

    log_dir_suffix = '-'.join(log_dir.split('-')[2:])
    if os.path.exists(args.output_dir):
        for existing_dir in os.listdir(args.output_dir):
            if existing_dir.endswith(log_dir_suffix):
                existing_path = os.path.join(args.output_dir, existing_dir)
                if os.path.exists(os.path.join(existing_path, "result.csv")):
                    print(f"Experiment with the same name '{log_dir_suffix}' already exists. Exiting to avoid overwriting.")
                    sys.exit(0)
                shutil.rmtree(existing_path)
                print(f"Removed incomplete experiment directory '{existing_dir}' to avoid conflicts.")

    writer = SummaryWriter(log_dir=log_dir)

    args_dict = vars(args)
    with open(os.path.join(writer.log_dir, "train_config.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8',
        filename=f'{writer.log_dir}/train.log',
        filemode='w'
    )

    def log(message):
        logging.info(message)
        print(message)

    log('Input arguments:')
    for key, val in vars(args).items():
        log(f'{key:22} {val}')

    if args.val_subject_id is not None and args.val_subject_id in args.train_subject_ids:
        if args.train_subject_weights is not None:
            args.train_subject_weights = [
                weight
                for subject, weight in zip(
                    args.train_subject_ids, args.train_subject_weights
                )
                if subject != args.val_subject_id
            ]
        args.train_subject_ids = [sid for sid in args.train_subject_ids if sid != args.val_subject_id]
        log(f"Removed val_subject_id={args.val_subject_id} from train_subject_ids for clean validation.")
        if len(args.train_subject_ids) == 0:
            raise ValueError("After removing val_subject_id, train_subject_ids is empty.")
    if args.train_subject_weights is not None:
        if not args.multi_positive_loss:
            raise ValueError(
                '--train_subject_weights currently requires --multi_positive_loss.'
            )
        if len(args.train_subject_weights) != len(args.train_subject_ids):
            raise ValueError(
                '--train_subject_weights must have one value per train subject.'
            )
        if any(weight <= 0 for weight in args.train_subject_weights):
            raise ValueError('--train_subject_weights must all be positive.')
        mean_subject_weight = float(np.mean(args.train_subject_weights))
        args.train_subject_weights = [
            float(weight) / mean_subject_weight
            for weight in args.train_subject_weights
        ]
        log(
            'Train subject loss weights: '
            + ', '.join(
                f's{subject}={weight:.3f}'
                for subject, weight in zip(
                    args.train_subject_ids, args.train_subject_weights
                )
            )
        )
    if args.mixup_type == 'coherent_group' and args.samples_per_image < len(args.train_subject_ids):
        raise ValueError(
            '--mixup_type coherent_group requires samples_per_image >= the number of '
            f'training subjects ({len(args.train_subject_ids)}) so every stimulus group is complete.'
        )

    with open(os.path.join(args.output_dir, "last_run.txt"), 'w') as f:
        f.write(writer.log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')
    amp_enabled = args.amp_dtype != 'none'
    if amp_enabled and device.type != 'cuda':
        raise ValueError('--amp_dtype requires a CUDA device')
    amp_dtype = {
        'none': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[args.amp_dtype]
    if amp_enabled:
        torch.set_autocast_dtype('cuda', amp_dtype)
        torch.set_autocast_enabled('cuda', True)
    log(
        f"Automatic mixed precision: {args.amp_dtype} "
        "(contrastive features/logits forced to float32)"
    )
    train_subject_weight_lookup = None
    if args.train_subject_weights is not None:
        train_subject_weight_lookup = torch.ones(
            max(args.train_subject_ids) + 1,
            dtype=torch.float32,
            device=device,
        )
        for subject, weight in zip(
            args.train_subject_ids, args.train_subject_weights
        ):
            train_subject_weight_lookup[subject] = weight
    log(f'Subject mixup mode: {args.subject_mixup_mode}, type: {args.mixup_type} (alpha={args.subject_mixup_alpha})')
    log(
        'Subject-coherent temporal MixStyle: '
        f'prob={args.subject_mixstyle_prob}, alpha={args.subject_mixstyle_alpha}, '
        f'granularity={args.subject_mixstyle_granularity}'
    )
    log(
        'Low-rank temporal subject-style manifold: '
        f'prob={args.subject_style_manifold_prob}, rank={args.subject_style_manifold_rank}, '
        f'scale={args.subject_style_manifold_scale}, max_radius={args.subject_style_manifold_max_radius}'
    )
    log(f'Evaluation mode: {args.eval_mode}')

    hyperalign = None
    if args.hyperalign_prob > 0:  # rotations + which (subject, partner) pairs may be used: same dataset, both training subjects
        rot = torch.as_tensor(np.load(args.hyperalign_maps), dtype=torch.float32, device=device)
        ids = torch.arange(rot.shape[0], device=device)
        tr = torch.zeros(rot.shape[0], dtype=torch.bool, device=device); tr[[i for i in args.train_subject_ids if i < rot.shape[0]]] = True
        hyperalign = (rot, tr[:, None] & tr[None, :] & (ids[:, None] != ids[None, :]) & ((ids[:, None] >= 21) == (ids[None, :] >= 21)))
    cross_dataset_translation = None
    if args.cross_dataset_translation:
        translation_npz = np.load(args.cross_dataset_translation)
        required_keys = {'channel_map', 'source_mean', 'target_mean'}
        missing_keys = required_keys.difference(translation_npz.files)
        if missing_keys:
            raise ValueError(
                f'{args.cross_dataset_translation} is missing translation arrays: '
                f'{sorted(missing_keys)}'
            )
        cross_dataset_translation = tuple(
            torch.as_tensor(translation_npz[key], dtype=torch.float32, device=device)
            for key in ('channel_map', 'source_mean', 'target_mean')
        )
        log(
            'Auxiliary cohort translation: '
            f'{args.cross_dataset_translation}, blend={args.cross_dataset_translation_blend}'
        )

    if args.grouped_batch_sampler and args.data_random:
        raise ValueError("Grouped batching requires deterministic indices. Disable --data_random when using --grouped_batch_sampler.")

    print('\n>>> Loading Train Data <<<')
    if args.eeg_aug:
        if args.eeg_aug_type == 'noise':
            eeg_transform = RandomGaussianNoise(std=0.001)
        elif args.eeg_aug_type == 'time_shift':
            eeg_transform = RandomTimeShift(max_shift=5)
        elif args.eeg_aug_type == 'channel_dropout':
            eeg_transform = RandomChannelDropout(drop_prob=0.1)
        elif args.eeg_aug_type == 'smooth':
            eeg_transform = RandomSmooth(kernel_size=5, smooth_prob=0.3)
    else:
        eeg_transform = None

    train_dataset = EEGPreImageDataset(
        args.train_subject_ids,
        args.eeg_data_dir,
        args.selected_channels,
        args.time_window,
        args.image_feature_dir,
        args.text_feature_dir,
        args.image_aug,
        args.aug_image_feature_dirs,
        args.data_average,
        args.data_random,
        eeg_transform,
        True,
        args.image_test_aug,
        args.eeg_test_aug,
        args.frozen_eeg_prior,
        cross_subject_average=args.cross_subject_average,
        xavg_beta_a=args.xavg_beta_a,
        xavg_beta_b=args.xavg_beta_b,
        xavg_kmin=args.xavg_kmin,
        xavg_kmax=args.xavg_kmax,
        subject_mixup_within=args.subject_mixup_within,
        within_mix_alpha=args.within_mix_alpha,
        bootstrap_repetition_average=args.bootstrap_repetition_average,
        bootstrap_repetition_count=args.bootstrap_repetition_count,
        subject_ea_align=args.subject_ea_align,
    )
    if args.train_object_fraction < 1:  # CVPR E8.8b: same data volume as fewer subjects, all subjects kept
        log(f'train_object_fraction {args.train_object_fraction}: {len(train_dataset.keep_objects(args.train_object_fraction))} training concepts')
    if args.extra_trials_dir:  # Alljoined-1 (export_alljoined1.py): extra single trials with their own image targets
        train_dataset.add_extra_trials(args.extra_trials_dir, args.extra_subject_offset)
    train_dataset.xsubj_mix_prob, train_dataset.xsubj_mix_alpha = args.xsubj_mix_prob, args.xsubj_mix_alpha
    train_dataset.rep_avg_prob, train_dataset.noise_swap_prob, train_dataset.band_swap_prob = args.rep_avg_prob, args.noise_swap_prob, args.band_swap_prob
    train_dataset.amp_scale_std, train_dataset.time_mask_len = args.amp_scale_std, args.time_mask_len
    train_dataset.amp_gain_range = args.amp_gain_range
    if args.spatial_jitter_deg > 0:
        train_dataset.spatial_mix_bank = spatial_mix_bank(args.selected_channels, args.spatial_jitter_deg)
        train_dataset.spatial_jitter_prob = args.spatial_jitter_prob
        log(f'spatial jitter: {len(train_dataset.spatial_mix_bank)} montage rotations up to {args.spatial_jitter_deg} deg, '
            f'p={args.spatial_jitter_prob}')
    if args.distill_lambda > 0 or args.anticorr_lambda > 0 or args.boost_gamma > 0:
        assert not args.data_average and not args.data_random, '--distill_targets rows are single trials'
        train_dataset.distill_targets = np.load(args.distill_targets, mmap_mode='r')
        assert len(train_dataset.distill_targets) == len(train_dataset), 'teacher rows != training trials'

    if args.inductive_covariance_align:
        if args.subject_ea_align:
            raise ValueError(
                '--inductive_covariance_align is incompatible with --subject_ea_align'
            )
        if (
            args.inductive_covariance_reference_cache
            and os.path.isfile(args.inductive_covariance_reference_cache)
        ):
            cached = torch.load(
                args.inductive_covariance_reference_cache,
                map_location=device,
                weights_only=True,
            )
            source_covariances = {
                int(subject): covariance.to(device)
                for subject, covariance in cached.items()
            }
        else:
            source_covariances = estimate_source_covariances(
                train_dataset.eeg_data_list,
                args.train_subject_ids,
                device,
                max_trials_per_subject=args.inductive_covariance_ref_trials,
            )
            if args.inductive_covariance_reference_cache:
                os.makedirs(
                    os.path.dirname(args.inductive_covariance_reference_cache),
                    exist_ok=True,
                )
                torch.save(
                    {
                        subject: covariance.cpu()
                        for subject, covariance in source_covariances.items()
                    },
                    args.inductive_covariance_reference_cache,
                )
        args._inductive_covariance_aligner = InductiveQueryCovarianceAlign(
            source_covariances,
            alpha=args.inductive_covariance_alpha,
            shrinkage=args.inductive_covariance_shrinkage,
        ).to(device)
        log(
            'Inductive covariance alignment: source-only leave-one-source-out train '
            f'prior, all-source unseen prior, alpha={args.inductive_covariance_alpha}, '
            f'shrinkage={args.inductive_covariance_shrinkage}, '
            f'reference_trials={args.inductive_covariance_ref_trials}'
        )

    eeg_sample_points = train_dataset.num_sample_points
    target_dims = getattr(
        train_dataset, 'target_dims', [train_dataset.image_features.shape[-1]]
    )
    if args.auxiliary_image_targets:
        if len(target_dims) < 2:
            raise ValueError(
                '--auxiliary_image_targets requires at least two comma-separated '
                '--image_feature_dir entries'
            )
        image_feature_dim = target_dims[0]
        inference_image_feature_dir = str(args.image_feature_dir).split(',')[0]
        log(
            'Semantic amalgamation targets: primary_dim='
            f'{image_feature_dim}, auxiliary_dims={target_dims[1:]}'
        )
    else:
        image_feature_dim = train_dataset.image_features.shape[-1]
        inference_image_feature_dir = args.image_feature_dir
    target_offsets = np.cumsum([0, *target_dims]).tolist()

    def primary_image_target(features):
        """Return the only image block retained by the inference checkpoint."""
        if not args.auxiliary_image_targets:
            return features
        return features[..., :image_feature_dim]

    def auxiliary_image_target_blocks(features):
        return [
            features[..., target_offsets[index]:target_offsets[index + 1]]
            for index in range(1, len(target_dims))
        ]

    backbone_feature_dim = args.eeg_backbone_dim if args.eeg_backbone_dim > 0 else image_feature_dim
    channels_num = train_dataset.channels_num
    log(f'EEG sample points: {eeg_sample_points}')
    log(f'image feature dimension: {image_feature_dim}')
    log(f'EEG backbone output dimension: {backbone_feature_dim}')
    log(f'alignment feature dimension (--feature_dim): {args.feature_dim}')
    log(f'data length: {len(train_dataset)}')
    log(f'number of channels: {channels_num}')

    pin_memory = device.type == "cuda"
    global_gallery = None
    if args.global_negatives > 0 or args.global_negatives_replace:
        # every training image's frozen target is a negative (16540), not only the batch's: the targets never move,
        # so the full softmax is exact and cheap; same-image duplicates in a batch stop being false negatives
        global_gallery = torch.tensor(np.asarray(train_dataset.image_features[..., :image_feature_dim], dtype=np.float32)
                                      .reshape(-1, image_feature_dim), device=device)
        global_n_img = train_dataset.image_features.shape[1]
        log(f'global negatives: {len(global_gallery)} training images')
    train_main_dataset = train_dataset
    concept_val_dataloader = None
    if args.val_concept_ratio > 0.0:
        # Honest checkpoint selection without giving up a training subject: hold out a
        # fixed slice of training CONCEPTS (every image of them, across every training
        # subject).  Concept-disjoint mirrors the real test condition of 200 unseen
        # concepts; an image-level split would leave concept-mates in training, which
        # --multi_positive_loss is explicitly trained to pull together.
        if args.data_random:
            raise ValueError("--val_concept_ratio requires deterministic indexing (disable --data_random).")
        groups = train_dataset.get_image_group_indices()
        concepts = sorted({key[0] for key in groups})
        holdout_count = min(
            max(1, int(np.floor(len(concepts) * float(args.val_concept_ratio)))),
            len(concepts) - 1,
        )
        # Constant seed on purpose: every fold and every arm must hold out the SAME
        # concepts, otherwise the arms are trained on different data and cannot be
        # ensembled against each other.
        held = set(
            np.random.default_rng(args.val_concept_seed).permutation(concepts)[:holdout_count].tolist()
        )
        concept_val_indices, main_train_indices = [], []
        for key, indices in groups.items():
            (concept_val_indices if key[0] in held else main_train_indices).extend(indices)
        train_main_dataset = (
            _GroupedSubset(train_dataset, main_train_indices)
            if args.grouped_batch_sampler
            else Subset(train_dataset, main_train_indices)
        )
        # Val scores against clean targets, as the test set does: a shallow copy shares every
        # array but reads image_features instead of the --image_aug targets.
        val_view = copy.copy(train_dataset)
        val_view.image_aug = False
        val_view.eeg_transform = None  # and clean EEG: --eeg_aug is for training rows only
        val_view.rep_avg_prob = val_view.noise_swap_prob = val_view.band_swap_prob = 0
        val_view.xsubj_mix_prob = val_view.amp_scale_std = val_view.time_mask_len = val_view.amp_gain_range = 0
        val_view.distill_targets = None
        concept_val_dataloader = DataLoader(
            Subset(val_view, concept_val_indices),
            batch_size=200, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory,
        )
        log(
            f"Concept val split: held_out_concepts={holdout_count}/{len(concepts)} "
            f"(seed={args.val_concept_seed}), train_items={len(main_train_indices)}, "
            f"val_items={len(concept_val_indices)}"
        )
    subj_probe_train_dataloader = None
    subj_probe_val_dataloader = None
    if args.subject_probe_holdout:
        if args.data_random:
            raise ValueError("--subject_probe_holdout requires deterministic indexing (disable --data_random).")
        if len(args.train_subject_ids) <= 1:
            raise ValueError("--subject_probe_holdout requires at least two training subjects.")
        rng = np.random.default_rng(seed + 137)
        main_train_indices = []
        probe_val_indices = []
        if args.grouped_batch_sampler:
            image_group_keys = list(train_dataset.get_image_group_indices().keys())
            holdout_group_count = min(
                max(1, int(np.floor(len(image_group_keys) * float(args.subject_probe_holdout_ratio)))),
                len(image_group_keys) - 1,
            )
            rng.shuffle(image_group_keys)
            probe_group_keys = set(image_group_keys[:holdout_group_count])
            for key, indices in train_dataset.get_image_group_indices().items():
                if key in probe_group_keys:
                    probe_val_indices.extend(indices)
                else:
                    main_train_indices.extend(indices)
            log(
                f"Subject probe holdout: train_groups={len(image_group_keys) - holdout_group_count}, "
                f"probe_val_groups={holdout_group_count}, main_train={len(main_train_indices)}, "
                f"probe_val={len(probe_val_indices)} (ratio={args.subject_probe_holdout_ratio:.3f})"
            )
        else:
            n_subj = len(args.train_subject_ids)
            if len(train_dataset) % n_subj != 0:
                raise ValueError("Train dataset length not divisible by number of train subjects; cannot build per-subject probe split.")
            per_subject_len = len(train_dataset) // n_subj
            if per_subject_len <= 1:
                raise ValueError("Per-subject train length too small for probe holdout.")
            holdout_per_subject = min(
                max(1, int(np.floor(per_subject_len * float(args.subject_probe_holdout_ratio)))),
                per_subject_len - 1,
            )
            for subject_idx in range(n_subj):
                start = subject_idx * per_subject_len
                perm = rng.permutation(np.arange(start, start + per_subject_len, dtype=np.int64))
                probe_val_indices.extend(perm[:holdout_per_subject].tolist())
                main_train_indices.extend(perm[holdout_per_subject:].tolist())
            log(
                f"Subject probe holdout: per_subject={per_subject_len}, main_train={len(main_train_indices)}, "
                f"probe_val={len(probe_val_indices)} (ratio={args.subject_probe_holdout_ratio:.3f})"
            )
        train_main_dataset = _GroupedSubset(train_dataset, main_train_indices) if args.grouped_batch_sampler else Subset(train_dataset, main_train_indices)
        subj_probe_val_dataset = Subset(train_dataset, probe_val_indices)
        pb = 256
        subj_probe_train_dataloader = DataLoader(
            train_main_dataset, batch_size=pb, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=pin_memory
        )
        subj_probe_val_dataloader = DataLoader(
            subj_probe_val_dataset, batch_size=pb, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=pin_memory
        )

    if args.train_max_trials > 0:  # SUBJECT_LOG D8 few-shot oracle: a seeded random subset of K training trials
        keep = np.sort(np.random.default_rng(0).choice(len(train_main_dataset), min(args.train_max_trials, len(train_main_dataset)), replace=False))
        train_main_dataset = Subset(train_main_dataset, keep.tolist())
        log(f'--train_max_trials {args.train_max_trials}: training on {len(keep)} trials')
    if args.repeat_group_batches:
        if train_main_dataset is not train_dataset:
            raise ValueError("--repeat_group_batches needs the full training set (no --val_concept_ratio).")
        batch_sampler = RepeatGroupBatchSampler(len(train_dataset), train_dataset.num_repetitions, args.batch_size, seed=seed)
        dataloader = DataLoader(train_main_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=pin_memory)
        log(f'repeat-group batches: {batch_sampler.groups_per_batch} (subject, image) groups x {batch_sampler.reps} repeats')
    elif args.grouped_batch_sampler:
        batch_sampler = GroupedImageBatchSampler(
            train_main_dataset,
            batch_size=args.batch_size,
            samples_per_image=args.samples_per_image,
            drop_last=True,
            seed=seed,
            images_per_concept=args.images_per_concept,
        )
        dataloader = DataLoader(train_main_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=pin_memory)
        effective_batch_size = batch_sampler.images_per_batch * batch_sampler.samples_per_image
        log(
            f'Grouped batch sampler enabled: images_per_batch={batch_sampler.images_per_batch}, '
            f'samples_per_image={batch_sampler.samples_per_image}, effective_batch_size={effective_batch_size}'
        )
    elif args.neighbour_batch_prob > 0:  # AUG_LOG: fine-grained negatives from k-means clusters of the image targets
        ds = train_dataset
        assert train_main_dataset is ds and not (ds.average or ds.random or ds.subject_mixup_within or ds.bootstrap_repetition_average)
        keys = (np.arange(len(ds)) // ds.num_repetitions) % (ds.num_objects * ds.num_images_per_object)
        feats = np.asarray(ds.image_features[..., :image_feature_dim]).reshape(ds.num_objects * ds.num_images_per_object, -1)
        batch_sampler = NeighbourBatchSampler(keys, feats, args.batch_size, args.neighbour_batch_prob, seed=seed)
        dataloader = DataLoader(train_main_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=pin_memory)
        log(f'neighbour batches: p={args.neighbour_batch_prob}, {len(batch_sampler.clusters)} image clusters')
    else:
        dataloader = DataLoader(
            train_main_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    if args.select_best_on == 'val' and args.val_subject_id is None and args.val_concept_ratio <= 0.0:
        raise ValueError("--select_best_on val requires --val_subject_id or --val_concept_ratio.")
    if args.val_subject_id is not None and args.val_concept_ratio > 0.0:
        raise ValueError("--val_subject_id and --val_concept_ratio are two different validation designs; pick one.")
    if args.subject_probe_holdout and args.val_concept_ratio > 0.0:
        raise ValueError("--subject_probe_holdout and --val_concept_ratio both re-split the train set; pick one.")
    if args.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be nonnegative.")
    if args.boost_gamma > 0 and not (args.multi_positive_loss and args.distill_targets):
        raise ValueError('--boost_gamma needs --multi_positive_loss (the only loss path that takes sample weights) and --distill_targets.')
    if args.repeat_mean_loss > 0 and not args.repeat_group_batches:
        raise ValueError("--repeat_mean_loss needs --repeat_group_batches (batches must hold whole repeat groups).")
    if args.repeat_group_batches and (args.data_average or args.data_random or args.grouped_batch_sampler):
        raise ValueError("--repeat_group_batches needs un-averaged, deterministic data and no --grouped_batch_sampler.")
    if args.sealed_early_stop_patience > 0 and not args.sealed_test_metric:
        raise ValueError("--sealed_early_stop_patience requires --sealed_test_metric.")
    if args.early_stop_patience > 0 and args.select_best_on != 'val':
        raise ValueError("--early_stop_patience requires --select_best_on val.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be nonnegative.")
    if args.val_subject_id is not None:
        if args.val_subject_id in args.test_subject_ids:
            raise ValueError("val_subject_id must be different from test_subject_ids.")

    val_dataloader = concept_val_dataloader
    if args.val_subject_id is not None:
        print('\n>>> Loading Validation Data <<<')
        val_dataset = EEGPreImageDataset(
            [args.val_subject_id],
            args.eeg_data_dir,
            args.selected_channels,
            args.time_window,
            args.image_feature_dir,
            args.text_feature_dir,
            args.image_aug,
            args.aug_image_feature_dirs,
            True,
            False,
            eeg_transform,
            False,
            args.image_test_aug,
            args.eeg_test_aug,
            args.frozen_eeg_prior,
            subject_ea_align=args.subject_ea_align,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    print('\n>>> Loading Test Data <<<')
    test_dataset = EEGPreImageDataset(
        args.test_subject_ids,
        args.eeg_data_dir,
        args.selected_channels,
        args.time_window,
        args.image_feature_dir,
        args.text_feature_dir,
        args.image_aug,
        args.aug_image_feature_dirs,
        True,
        False,
        eeg_transform,
        False,
        args.image_test_aug,
        args.eeg_test_aug,
        args.frozen_eeg_prior,
        subject_ea_align=args.subject_ea_align,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    inference_keys = [
        'eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir',
        'projector', 'projector_activation', 'projector_topk', 'feature_dim', 'eeg_backbone_dim', 'time_window', 'selected_channels',
        'eeg_confidence_mode', 'eeg_confidence_hidden_dim',
        'tsconv_temporal_filters', 'tsconv_temporal_kernel', 'tsconv_pool_kernel', 'tsconv_pool_stride',
        'tsconv_spatial_filters', 'tsconv_projection_filters', 'tsconv_activation', 'tsconv_dropout',
        'tsconv_head_dropout', 'tsconv_no_batch_norm', 'tsconv_no_conv_bias', 'tsconv_power_branch',
        'fm_head', 'fm_head_dropout', 'fm_drop_path', 'arch_kwargs', 'fm_arch_kwargs',
        'tsconv_domain_bn_boundary', 'tsconv_domain_bn_boundary2', 'tsconv_domain_spatial_boundary',
        'norm_type', 'latency_align', 'extra_input_adapter', 'extra_subject_offset',
        'subject_adapter_rank', 'subject_adapter_max_id', 'subject_adapter_scale',
        'subject_adapter_inferred',
        'inductive_covariance_align', 'inductive_covariance_alpha',
        'inductive_covariance_shrinkage', 'inductive_covariance_ref_trials',
        'eval_mode', 'sattc_saw_shrink', 'sattc_saw_diag', 'sattc_csls_k', 'sattc_cw', 'sattc_cw_shrink', 'sattc_cw_diag',
    ]
    inference_config = {k: args_dict[k] for k in inference_keys}
    # Training-only semantic targets must never leak into standalone inference.
    inference_config['image_feature_dir'] = inference_image_feature_dir
    inference_config['feature_dim'] = int(str(args.feature_dim).split(',')[0])
    inference_config['architecture'] = 'baseline'
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['backbone_feature_dim'] = backbone_feature_dim
    inference_config['image_feature_dim'] = image_feature_dim
    with open(os.path.join(writer.log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)

    model = build_eeg_encoder(args, backbone_feature_dim, eeg_sample_points, channels_num).to(device)
    if args.init_encoder:
        # Pretrained trunk (pretrain_denoise.py) or a previous run's checkpoint (curriculum).
        pretrained = torch.load(args.init_encoder, map_location=device, weights_only=False)
        pretrained = pretrained.get('model_state_dict', pretrained)  # a train.py checkpoint works too
        own = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(pretrained, strict=False)
        log(f'--init_encoder: loaded {len(pretrained)}/{len(own)} tensors from {args.init_encoder}')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'EEG Encoder trainable parameters: {num_params / 1e6:.2f}M')
    log(str(model))

    projector_activation = args.projector_activation
    projector_topk = args.projector_topk
    if len(target_dims) > 1 and not args.auxiliary_image_targets:
        raise ValueError(
            'comma-separated --image_feature_dir entries require '
            '--auxiliary_image_targets'
        )
    args.feature_dim = int(str(args.feature_dim).split(',')[0])
    eeg_projector = build_projector(
        args.projector, backbone_feature_dim, args.feature_dim,
        activation=projector_activation, topk=projector_topk,
    ).to(device)
    if args.projector in ('subspace', 'subspace_c'):
        eeg_projector.fit(train_dataset.image_features)  # training images only; stored in the checkpoint
        log(f'subspace head: {backbone_feature_dim} principal directions of the training targets')
    img_projector = build_projector(
        args.projector, image_feature_dim, args.feature_dim,
        activation=projector_activation, topk=projector_topk,
    ).to(device)
    text_projector = build_projector(
        args.projector, image_feature_dim, args.feature_dim,
        activation=projector_activation, topk=projector_topk,
    ).to(device)
    shared_encoder = None
    if args.shared_encoder_hidden > 0:
        align_out_dim = args.feature_dim if args.projector != 'direct' else backbone_feature_dim
        shared_encoder = build_shared_encoder(align_out_dim, args.shared_encoder_hidden).to(device)
        eeg_projector = SharedSpaceProjector(eeg_projector, shared_encoder).to(device)
        img_projector = SharedSpaceProjector(img_projector, shared_encoder).to(device)
        log(f'Shared encoder enabled (hidden={args.shared_encoder_hidden}), coarse stage = '
            f'{args.coarse_epochs} epochs, mmd_lambda={args.mmd_lambda}')

    projector_params = (
        sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in img_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    )
    log(f'Projector trainable parameters: {projector_params / 1e6:.2f}M')

    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if args.init_checkpoint_scope == 'conv':
            convolution_prefixes = ('tsconv.0.', 'tsconv.4.', 'projection.')
            convolution_state = {
                key: value for key, value in initial['model_state_dict'].items()
                if key.startswith(convolution_prefixes)
            }
            if not convolution_state:
                raise ValueError(
                    '--init_checkpoint_scope conv found no compatible TSConv convolution weights'
                )
            incompatible = model.load_state_dict(convolution_state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            if unexpected:
                raise ValueError(f'Unexpected convolution checkpoint keys: {unexpected}')
            log(
                f'Initialized {len(convolution_state)} TSConv convolution tensors from '
                f'{args.init_checkpoint}; normalization, head, and projectors remain fresh'
            )
        else:
            model.load_state_dict(initial['model_state_dict'])
            if args.init_checkpoint_scope == 'all':
                eeg_projector.load_state_dict(initial['eeg_projector_state_dict'])
                img_projector.load_state_dict(initial['img_projector_state_dict'])
                text_projector.load_state_dict(initial['text_projector_state_dict'])
            log(
                f'Initialized {args.init_checkpoint_scope} from {args.init_checkpoint}'
            )
        del initial

    if args.freeze_eeg_encoder:
        set_requires_grad(model, False)
        model.eval()
        log('Frozen initialized EEG encoder in eval mode')

    if args.fm_unfreeze_last_k > 0:
        trainable = model.unfreeze_last_k(args.fm_unfreeze_last_k)
        log(f'Partial fine-tune: last {args.fm_unfreeze_last_k} backbone layers + head, '
            f'{trainable:,} trainable parameters')

    if args.lora_rank > 0:  # SUBJECT_LOG D5: rank-r updates W + B A on the nn.Linear layers matching --lora_regex
        import re
        for prefix, root in (('model.', model), ('eproj', eeg_projector)):
            for name, module in root.named_modules():
                full = prefix + name
                if isinstance(module, nn.Linear) and re.search(args.lora_regex, full):
                    attach_lora(module, args.lora_rank)
                    log(f'LoRA rank {args.lora_rank} on {full} {tuple(module.weight.shape)}')
        args.trainable_regex = 'lora_'  # only the adapters train; the checkpoint stores the merged weights
    if args.trainable_regex:  # SUBJECT_LOG D4: fine-tune only the matching parameters (BN running stats adapt anyway)
        import re
        for prefix, module in (('model.', model), ('eproj.', eeg_projector)):
            for name, parameter in module.named_parameters():
                parameter.requires_grad = re.search(args.trainable_regex, prefix + name) is not None
        log(f'--trainable_regex {args.trainable_regex}: '
            f'{sum(p.numel() for m in (model, eeg_projector) for p in m.parameters() if p.requires_grad):,} trainable parameters')
    if args.freeze_image_projector:
        set_requires_grad(img_projector, False)
        img_projector.eval()
        log('Frozen pretrained image projector for fine-tuning')

    align_dim = args.feature_dim if args.projector != 'direct' else backbone_feature_dim
    if args.teacher_alignment_lambda > 0:
        if len(target_dims) != 2:
            raise ValueError(
                '--teacher_alignment_lambda requires exactly one training-only target block.'
            )
        if int(target_dims[-1]) != int(align_dim):
            raise ValueError(
                f'Teacher target dim {target_dims[-1]} must equal alignment dim {align_dim}.'
            )
        log(
            f'Natural EEG teacher alignment: dim={align_dim}, '
            f'lambda={args.teacher_alignment_lambda}'
        )
    probe_class_map = {sid: idx for idx, sid in enumerate(sorted(args.train_subject_ids))}
    num_probe_classes = len(probe_class_map)
    subj_probe_heads = None
    subj_probe_optimizers = None
    if args.subject_probe_holdout:
        subj_probe_heads = nn.ModuleDict()
        # Determine probe input dimensions
        with torch.no_grad():
            dummy_eeg = torch.zeros(1, channels_num, eeg_sample_points).to(device)
            dummy_sid = torch.tensor([args.train_subject_ids[0]]).to(device)
            bb_out = run_eeg_backbone(model, args, dummy_eeg, dummy_sid, return_intermediate=True)
            
            for layer in args.subject_probe_layers:
                if layer == 'align':
                    in_dim = align_dim
                elif layer in bb_out:
                    in_dim = bb_out[layer].shape[-1]
                else:
                    log(f"Warning: layer '{layer}' not found in backbone output. Skipping probe.")
                    continue
                subj_probe_heads[f'eeg_{layer}'] = nn.Linear(in_dim, num_probe_classes).to(device)
        
        subj_probe_optimizers = {name: optim.Adam(head.parameters(), lr=1e-3) for name, head in subj_probe_heads.items()}

    criterion = ContrastiveLoss(
        args.init_temperature,
        args.alpha,
        args.beta,
        args.eeg_l2norm,
        args.img_l2norm,
        args.text_l2norm,
        args.t_learnable,
        args.softplus,
        logit_scale_max=args.logit_scale_max,
        asymmetric=args.asymmetric_loss,
        mse_on_raw=args.mse_on_raw,
        soft_target_tau=args.soft_target_tau,
        soft_mix_beta=args.soft_mix_beta,
    ).to(device)
    if args.init_checkpoint and args.init_checkpoint_scope == 'all':  # the learned temperature belongs to the head
        saved = torch.load(args.init_checkpoint, map_location=device, weights_only=False).get('criterion_state_dict')
        if saved is not None:
            criterion.load_state_dict(saved)
            log(f'Initialized the loss (logit scale) from {args.init_checkpoint}')
    log(str(criterion))
    auxiliary_dims = [
        int(value) for value in str(args.auxiliary_alignment_dims).split(',')
        if value.strip()
    ]
    matryoshka_dims = sorted(set(
        int(value) for value in str(args.matryoshka_dims).split(',')
        if value.strip()
    ))
    if args.matryoshka_lambda > 0:
        if not matryoshka_dims:
            raise ValueError('--matryoshka_dims is required when --matryoshka_lambda > 0')
        if any(dim <= 0 or dim >= align_dim for dim in matryoshka_dims):
            raise ValueError(
                f'Matryoshka dimensions must be in [1, {align_dim - 1}]'
            )
        log(
            f'Nested final-embedding supervision: dims={matryoshka_dims}, '
            f'lambda={args.matryoshka_lambda}'
        )
    auxiliary_eeg_projectors = None
    auxiliary_img_projectors = None
    if args.auxiliary_alignment_lambda > 0:
        if not auxiliary_dims:
            raise ValueError(
                '--auxiliary_alignment_dims is required when '
                '--auxiliary_alignment_lambda > 0'
            )
        if any(dim <= 0 for dim in auxiliary_dims):
            raise ValueError('Auxiliary alignment dimensions must be positive')
        # Keep all later stochastic training decisions matched to the control.
        auxiliary_rng = torch.random.get_rng_state()
        if args.auxiliary_image_targets:
            auxiliary_target_dims = target_dims[1:]
            if len(auxiliary_dims) == 1:
                auxiliary_dims = auxiliary_dims * len(auxiliary_target_dims)
            if len(auxiliary_dims) != len(auxiliary_target_dims):
                raise ValueError(
                    '--auxiliary_alignment_dims must have one value or one value '
                    'per training-only image target'
                )
        else:
            auxiliary_target_dims = [image_feature_dim] * len(auxiliary_dims)
        auxiliary_eeg_projectors = nn.ModuleList([
            ProjectorLinear(backbone_feature_dim, dim) for dim in auxiliary_dims
        ]).to(device)
        if args.auxiliary_fixed_targets and list(auxiliary_dims) != list(auxiliary_target_dims):
            raise ValueError(f'--auxiliary_fixed_targets needs --auxiliary_alignment_dims {auxiliary_target_dims}')
        auxiliary_img_projectors = nn.ModuleList([
            nn.Identity() if args.auxiliary_fixed_targets else ProjectorLinear(target_dim, dim)
            for target_dim, dim in zip(auxiliary_target_dims, auxiliary_dims)
        ]).to(device)
        torch.random.set_rng_state(auxiliary_rng)
        auxiliary_params = sum(
            parameter.numel()
            for module in (auxiliary_eeg_projectors, auxiliary_img_projectors)
            for parameter in module.parameters()
        )
        log(
            f'Training-only auxiliary alignment heads: dims={auxiliary_dims}, '
            f'lambda={args.auxiliary_alignment_lambda}, '
            f'parameters={auxiliary_params / 1e6:.2f}M'
        )
    eeg_confidence_head = None
    if args.eeg_confidence_mode == 'learned':
        eeg_confidence_head = EEGConfidenceHead(
            align_dim,
            hidden_dim=args.eeg_confidence_hidden_dim,
        ).to(device)
        log(f'EEG confidence head parameters: {sum(p.numel() for p in eeg_confidence_head.parameters() if p.requires_grad) / 1e6:.2f}M')

    sattc_params = {
        'saw_shrink': args.sattc_saw_shrink,
        'saw_diag': args.sattc_saw_diag,
        'csls_k': args.sattc_csls_k,
        'cw_enabled': args.sattc_cw,
        'cw_shrink': args.sattc_cw_shrink,
        'cw_diag': args.sattc_cw_diag,
    }

    total_epochs = args.num_epochs

    module_registry = {
        'model': model,
        'eeg_projector': eeg_projector,
        'img_projector': img_projector,
        'text_projector': text_projector,
    }
    neighbor_heads = neighbor_table = neighbor_targets = None
    if args.neighbor_loss > 0:
        # lever L4: tables indexed [subject id, object, image, rep, offset] -> neighbour's training image k (-1: none)
        # subjects without a table (THINGS-EEG2 exports: no RSVP order) keep -1, i.e. no neighbour loss
        paths = {s: os.path.join(args.eeg_data_dir, f"sub-{s:02d}", "train_neighbors.npy") for s in args.train_subject_ids}
        tables = {s: np.load(f) for s, f in paths.items() if os.path.exists(f)}
        first = next(iter(tables.values()))
        neighbor_table = torch.full((max(args.train_subject_ids) + 1, *first.shape), -1, dtype=torch.long)
        for s, t in tables.items():
            neighbor_table[s] = torch.from_numpy(t.astype(np.int64))
        neighbor_table = neighbor_table.to(device)
        neighbor_targets = torch.as_tensor(np.asarray(train_dataset.image_features, dtype=np.float32)).reshape(
            -1, image_feature_dim).to(device)
        neighbor_heads = nn.ModuleList([nn.Linear(args.feature_dim, args.feature_dim) for _ in range(first.shape[-1])]).to(device)
        module_registry['neighbor_heads'] = neighbor_heads
        log(f'neighbour heads: {first.shape[-1]} RSVP offsets, coverage '
            f'{float((neighbor_table[args.train_subject_ids] >= 0).float().mean()):.3f}')
    if eeg_confidence_head is not None:
        module_registry['eeg_confidence_head'] = eeg_confidence_head
    if auxiliary_eeg_projectors is not None:
        module_registry['auxiliary_eeg_projectors'] = auxiliary_eeg_projectors
        module_registry['auxiliary_img_projectors'] = auxiliary_img_projectors
    siglip_head = None
    if args.siglip:  # learnable SigLIP bias (the EEG norm already plays the temperature)
        siglip_head = nn.Module(); siglip_head.b = nn.Parameter(torch.tensor(float(args.siglip_bias), device=device))
        module_registry['siglip_head'] = siglip_head
    trainable_parameters = (
        collect_trainable_parameters(list(module_registry.values()))
        + ([p for p in criterion.parameters() if p.requires_grad] if args.t_learnable else [])
    )
    fm_backbone = getattr(model, 'fm', None)
    param_groups = [{'params': trainable_parameters}]
    if args.fm_backbone_lr_scale != 1.0 and fm_backbone is not None:
        backbone_ids = {id(p) for p in fm_backbone.parameters()}
        param_groups = [
            {'params': [p for p in trainable_parameters if id(p) not in backbone_ids],
             'lr': args.learning_rate},
            {'params': [p for p in trainable_parameters if id(p) in backbone_ids],
             'lr': args.learning_rate * args.fm_backbone_lr_scale},
        ]
        log(f'Foundation backbone lr {args.learning_rate * args.fm_backbone_lr_scale:g}, '
            f'head lr {args.learning_rate:g}')
    if args.fm_layer_decay != 1.0 and fm_backbone is not None:
        if args.fm_backbone_lr_scale != 1.0:
            raise ValueError('--fm_layer_decay and --fm_backbone_lr_scale are mutually exclusive')
        names = [name for name, _ in fm_backbone.named_parameters()]
        # REVE transformer.layers, EEGPT target_encoder.blocks, LaBraM blocks
        prefix = next(p for p in ('transformer.layers.', 'target_encoder.blocks.', 'blocks.')
                      if any(n.startswith(p) for n in names))
        depth_of = lambda name: int(name[len(prefix):].split('.')[0])
        n_layers = 1 + max(depth_of(n) for n in names if n.startswith(prefix))
        scale_of = {}  # parameter id -> lr multiplier (BEiT-style layer ids: embeddings 0, block i -> i + 1)
        for name, parameter in fm_backbone.named_parameters():
            if name.startswith(prefix):
                layer_id = depth_of(name) + 1
            elif name.startswith(('final_layer', 'cls_query_token', 'norm.', 'target_encoder.norm.')):
                layer_id = n_layers + 1  # REVE's own head (bypassed by return_output); final norms
            else:
                layer_id = 0  # patch/channel/positional embeddings and their norms
            scale_of[id(parameter)] = args.fm_layer_decay ** (n_layers + 1 - layer_id)
        groups = {}
        for parameter in trainable_parameters:
            groups.setdefault(scale_of.get(id(parameter), 1.0), []).append(parameter)
        param_groups = [{'params': ps, 'lr': args.learning_rate * scale}
                        for scale, ps in sorted(groups.items(), reverse=True)]
        log(f'Layer-wise lr decay {args.fm_layer_decay}: {len(param_groups)} groups, lr '
            f'{args.learning_rate:g} (head) down to {args.learning_rate * min(groups):.2g} (embeddings)')
    trainable_ids = {id(p) for p in trainable_parameters}
    if args.optimizer == 'stableadamw':
        from module.stable_adamw import StableAdamW
        optimizer = StableAdamW(param_groups, lr=args.learning_rate, betas=(0.92, 0.999), eps=1e-9,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(
            param_groups,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    grad_scaler = torch.amp.GradScaler(
        'cuda', enabled=(args.amp_dtype == 'float16')
    )
    ema_model = None
    if args.ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        # BatchNorm running stats are averaged too (use_buffers), so the EMA model is
        # self-consistent when it is swapped in for evaluation.
        ema_update = get_ema_multi_avg_fn(args.ema_decay)
        if args.ema_warmup:
            def ema_update(ema_list, cur_list, n):
                # timm/TF-style warmup: the average tracks the live weights early, so it does
                # not lag while they change fast, and reaches ema_decay after ~9k updates.
                decay = min(args.ema_decay, (1.0 + n.item()) / (10.0 + n.item()))
                if torch.is_floating_point(ema_list[0]):
                    torch._foreach_lerp_(ema_list, cur_list, 1.0 - decay)
                else:
                    for ema_t, cur_t in zip(ema_list, cur_list):
                        ema_t.copy_(ema_t * decay + cur_t * (1.0 - decay))
        ema_model = AveragedModel(model, multi_avg_fn=ema_update, use_buffers=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        warmup = max(0, int(args.warmup_epochs))
        floor = args.min_lr / max(args.learning_rate, 1e-12)

        def _lr_lambda(step):
            e = step + 1  # 1-based epoch about to run (scheduler.step() is called at epoch end)
            if e <= warmup:
                return e / max(warmup, 1)
            progress = (e - warmup) / max(total_epochs - warmup, 1)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    elif args.lr_scheduler == 'onecycle':
        # Stepped per optimizer step, not per epoch -- this is the whole point of OneCycle and
        # the reason the per-epoch cosine above is not a substitute.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=total_epochs * len(dataloader),
            pct_start=args.onecycle_pct_start,
            anneal_strategy='cos',
        )
    elif args.lr_scheduler == 'wsd':
        wsd_total = total_epochs * len(dataloader)
        wsd_warmup = max(1, int(args.onecycle_pct_start * wsd_total))
        wsd_decay = int(args.wsd_decay_start * wsd_total)

        def _wsd_lambda(step):
            if step < wsd_warmup:
                return (step + 1) / wsd_warmup
            if step < wsd_decay:
                return 1.0
            return max(0.0, (wsd_total - step) / max(wsd_total - wsd_decay, 1))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _wsd_lambda)
    elif args.lr_scheduler == 'plateau':
        # REVE authors (reve_eeg dt.py): ReduceLROnPlateau(max, 0.5) per epoch, after an exponential per-step warmup
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.plateau_patience)
        warmup_steps = args.warmup_epochs * len(dataloader)
        plateau_warmup = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (10 ** (step / warmup_steps) - 1) / 9 if step < warmup_steps else 1.0)

    if args.train_rng_seed is not None:
        seed_everything(args.train_rng_seed)
        log(
            f'Reset training RNG to {args.train_rng_seed} after module initialization; '
            'batch order and stochastic augmentation draws are matched across variants'
        )

    def forward_architecture(eeg_backbone_batch):
        eeg_feature = eeg_projector(eeg_backbone_batch)
        out = {'eeg_feature': eeg_feature}
        if args.eeg_confidence_mode == 'detached_prenorm':
            out['eeg_confidence'] = eeg_feature.norm(dim=1).detach()
        elif args.eeg_confidence_mode == 'learned':
            out['eeg_confidence'] = eeg_confidence_head(eeg_feature.detach())
        else:
            out['eeg_confidence'] = None
        return out

    head_select_loader = (val_dataloader if args.select_best_on == 'val' and val_dataloader is not None
                          else test_dataloader)

    def build_checkpoint(epoch_idx, loss_value):
        checkpoint = {
            'epoch': epoch_idx,
            'stage': 'joint',
            'architecture': 'baseline',
            'model_state_dict': model.state_dict(),
            'eeg_projector_state_dict': eeg_projector.state_dict(),
            'img_projector_state_dict': img_projector.state_dict(),
            'text_projector_state_dict': text_projector.state_dict(),
            # No optimizer state: nothing in this repo resumes training, and AdamW's two
            # moments per parameter are 2/3 of a REVE checkpoint (840 MB -> 280 MB).
            'loss': loss_value,
        }
        if eeg_confidence_head is not None:
            checkpoint['eeg_confidence_head_state_dict'] = eeg_confidence_head.state_dict()
        if args.t_learnable:
            checkpoint['criterion_state_dict'] = criterion.state_dict()
        if args.inductive_covariance_align:
            checkpoint['inductive_covariance_state_dict'] = (
                args._inductive_covariance_aligner.state_dict()
            )
        return checkpoint

    def maybe_apply_train_saw(eeg_feature_batch, image_feature_proj, subject_id_batch):
        if not args.train_saw:
            return eeg_feature_batch, image_feature_proj
        normalize = not args.train_saw_no_renorm
        eeg_feature_batch = subject_adaptive_whiten_torch(
            eeg_feature_batch,
            subject_id_batch,
            shrink=args.train_saw_shrink,
            diag=args.train_saw_diag,
            normalize=normalize,
        )
        if args.train_saw_whiten_image:
            image_feature_proj = subject_adaptive_whiten_torch(
                image_feature_proj,
                subject_id_batch,
                shrink=args.train_saw_shrink,
                diag=args.train_saw_diag,
                normalize=normalize,
            )
        return eeg_feature_batch, image_feature_proj

    def export_eeg_norm_metadata(split_dataset, split_name, output_path_base):
        rows = []
        export_loader = DataLoader(
            split_dataset,
            batch_size=200,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        model.eval()
        eeg_projector.eval()
        with torch.no_grad():
            for batch in export_loader:
                eeg_batch = batch[0].to(device)
                subject_id_batch = batch[3].to(device)
                object_idx_batch = batch[4]
                image_idx_batch = batch[5]
                repetition_idx_batch = batch[6]

                eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
                eeg_feature_batch = eeg_projector(eeg_backbone_batch)
                eeg_norm_batch = eeg_feature_batch.norm(dim=1).cpu().numpy()

                for i in range(len(eeg_norm_batch)):
                    rows.append({
                        'split': split_name,
                        'subject_id': int(subject_id_batch[i].item()),
                        'object_idx': int(object_idx_batch[i]),
                        'image_idx': int(image_idx_batch[i]),
                        'repetition_idx': int(repetition_idx_batch[i]),
                        'eeg_prenorm': float(eeg_norm_batch[i]),
                    })

        df = pd.DataFrame(rows)
        parquet_path = f"{output_path_base}.parquet"
        csv_path = f"{output_path_base}.csv"
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(parquet_path, index=False)
            log(f"Saved EEG norm metadata: {csv_path} and {parquet_path}")
            return parquet_path
        except Exception as exc:
            log(f"Saved EEG norm metadata CSV: {csv_path} (parquet skipped: {exc})")
            return csv_path

    per_subject_last = {}  # subject -> (loss, top1, top5) of the latest test pass

    def evaluate_split(split_loader, split_name):
        if split_loader is None:
            return float('nan'), float('nan'), float('nan'), {}

        total_loss_local = 0.0
        batch_losses = []  # (subject, loss) per batch, for --per_subject_test_log
        eeg_feature_list = []
        image_feature_list = []
        subjects = []
        object_indices = []
        image_indices = []

        with torch.no_grad():
            for batch in split_loader:
                eeg_batch = batch[0].to(device)
                image_feature_batch = batch[1].to(device)
                text_feature_batch = batch[2].to(device)
                subject_id_batch = batch[3].to(device)
                object_idx_batch = batch[4].to(device)
                image_idx_batch = batch[5].to(device)

                eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
                arch_out = forward_architecture(eeg_backbone_batch)
                eeg_feature_batch = arch_out['eeg_feature']
                eeg_confidence_batch = arch_out['eeg_confidence']
                image_feature_proj = img_projector(
                    primary_image_target(image_feature_batch)
                )
                text_feature_proj = text_projector(
                    primary_image_target(text_feature_batch)
                )
                eeg_feature_for_loss, image_feature_for_loss = maybe_apply_train_saw(
                    eeg_feature_batch,
                    image_feature_proj,
                    subject_id_batch,
                )

                positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
                loss = compute_cross_modal_loss(
                    criterion,
                    {'feature': eeg_feature_for_loss, 'confidence': eeg_confidence_batch},
                    image_feature_for_loss,
                    text_feature_proj,
                    positive_mask,
                    args.multi_positive_loss
                )
                total_loss_local += loss.item()
                if args.per_subject_test_log and split_name == 'test':
                    batch_subjects = subject_id_batch.unique()
                    assert len(batch_subjects) == 1, 'per-subject test loss needs one subject per test batch'
                    batch_losses.append((int(batch_subjects[0]), loss.item()))
                eeg_feature_list.append(eeg_feature_batch.float().cpu().numpy())
                image_feature_list.append(image_feature_proj.float().cpu().numpy())
                subjects.append(subject_id_batch.cpu().numpy())
                object_indices.append(object_idx_batch.cpu().numpy())
                image_indices.append(image_idx_batch.cpu().numpy())

        avg_loss_local = total_loss_local / len(split_loader)
        eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
        image_feature_all = np.concatenate(image_feature_list, axis=0)
        subject_all = np.concatenate(subjects, axis=0)
        object_all = np.concatenate(object_indices, axis=0)
        image_all = np.concatenate(image_indices, axis=0)
        gallery, targets = image_feature_all, None
        if args.eval_unique_gallery:
            # Rank against each distinct stimulus once. The flat per-item gallery holds one row
            # per presentation, so a repeated image contributes many IDENTICAL vectors (40 when
            # un-averaged: 4 reps x 10 subjects) and the target is an arbitrary member of a tied
            # block -- a perfect model tops out near 5/40. That ceiling, not model quality, is
            # what checkpoint selection was reading. This is also the grader's protocol.
            keys = object_all.astype(np.int64) * (int(image_all.max()) + 1) + image_all.astype(np.int64)
            _, first, targets = np.unique(keys, return_index=True, return_inverse=True)
            gallery = image_feature_all[first]
        top5_count, top1_count, total = retrieve_all(
            eeg_feature_all,
            gallery,
            subject_ids=subject_all,
            eval_mode=args.eval_mode,
            sattc_params=sattc_params,
            target_indices=targets,
        )
        top5_acc_local = top5_count / total * 100
        top1_acc_local = top1_count / total * 100
        if batch_losses:
            assert args.eval_unique_gallery, '--per_subject_test_log needs --eval_unique_gallery (pooled gallery repeats images)'
            # Each subject exactly as its own single-subject run scores it: its batch loss, and its
            # queries ranked against the (shared, unique) gallery.
            per_subject_last.clear()
            for subject in sorted({s for s, _ in batch_losses}):
                rows = subject_all == subject
                s5, s1, n = retrieve_all(eeg_feature_all[rows], gallery, subject_ids=subject_all[rows],
                                         eval_mode=args.eval_mode, sattc_params=sattc_params,
                                         target_indices=None if targets is None else targets[rows])
                losses = [l for s, l in batch_losses if s == subject]
                per_subject_last[subject] = (sum(losses) / len(losses), s1 / n * 100, s5 / n * 100)
        sparsity = {}
        if args.projector_activation in _SPARSE_PROJECTOR_ACTIVATIONS:
            eeg_sp = compute_alignment_sparsity_stats(eeg_feature_all)
            img_sp = compute_alignment_sparsity_stats(image_feature_all)
            sparsity = {
                'eeg_l0_mean': eeg_sp['l0_mean'],
                'eeg_l0_frac': eeg_sp['l0_frac'],
                'eeg_active_feat_frac': eeg_sp['active_feat_frac'],
                'img_l0_mean': img_sp['l0_mean'],
                'img_l0_frac': img_sp['l0_frac'],
                'img_active_feat_frac': img_sp['active_feat_frac'],
            }
            log(
                f"{split_name} sparsity: eeg_l0_frac={eeg_sp['l0_frac']:.4f} "
                f"eeg_active_feat_frac={eeg_sp['active_feat_frac']:.4f} "
                f"img_l0_frac={img_sp['l0_frac']:.4f}"
            )
        log(
            f"{split_name}: mode={args.eval_mode} top5 acc {top5_acc_local:.2f}%\ttop1 acc {top1_acc_local:.2f}%\tLoss: {avg_loss_local:.4f}"
        )
        return avg_loss_local, top1_acc_local, top5_acc_local, sparsity

    def collect_split_sparsity(split_loader, split_name):
        """Post-ReLU alignment sparsity only (no loss / retrieval; safe for full train set)."""
        if split_loader is None or args.projector_activation not in _SPARSE_PROJECTOR_ACTIVATIONS:
            return {}

        eeg_acc = _empty_alignment_sparsity_accumulator()
        img_acc = _empty_alignment_sparsity_accumulator()
        with torch.no_grad():
            for batch in split_loader:
                eeg_batch = batch[0].to(device)
                image_feature_batch = batch[1].to(device)
                subject_id_batch = batch[3].to(device)

                eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
                arch_out = forward_architecture(eeg_backbone_batch)
                _update_alignment_sparsity_accumulator(eeg_acc, arch_out['eeg_feature'].cpu().numpy())
                _update_alignment_sparsity_accumulator(
                    img_acc,
                    img_projector(
                        primary_image_target(image_feature_batch)
                    ).cpu().numpy(),
                )

        eeg_sp = _finalize_alignment_sparsity_accumulator(eeg_acc)
        img_sp = _finalize_alignment_sparsity_accumulator(img_acc)
        sparsity = {
            'eeg_l0_mean': eeg_sp['l0_mean'],
            'eeg_l0_frac': eeg_sp['l0_frac'],
            'eeg_active_feat_frac': eeg_sp['active_feat_frac'],
            'img_l0_mean': img_sp['l0_mean'],
            'img_l0_frac': img_sp['l0_frac'],
            'img_active_feat_frac': img_sp['active_feat_frac'],
        }
        log(
            f"{split_name} sparsity: eeg_l0_frac={eeg_sp['l0_frac']:.4f} "
            f"eeg_active_feat_frac={eeg_sp['active_feat_frac']:.4f} "
            f"img_l0_frac={img_sp['l0_frac']:.4f} "
            f"img_active_feat_frac={img_sp['active_feat_frac']:.4f}"
        )
        return sparsity

    def encode_probe_labels(subject_ids: torch.Tensor) -> torch.Tensor:
        ids = [int(x) for x in subject_ids.detach().cpu().tolist()]
        return torch.tensor([probe_class_map[s] for s in ids], dtype=torch.long, device=subject_ids.device)

    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_test_loss = float('inf')
    best_testsel_loss = float('inf')
    best_testsel_epoch = -1
    best_test_epoch = 0
    best_selected_top1 = 0.0   # top1 on whichever split selects; == best_top1_acc unless select_best_on=val
    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'top1_acc': [],
               'test_top5_acc': [], 'val_loss': [], 'val_top1_acc': [], 'val_top5_acc': []}
    if args.sealed_test_metric:
        # Selection would otherwise read the sealed test number itself.
        assert args.select_best_on != 'test', '--sealed_test_metric needs --select_best_on val or fixed'
        history.update(avg_eeg_top1=[], avg_eeg_top5=[])
    sealed_best_top5, sealed_stale = float('-inf'), 0

    sealed_best = {'top5': float('-inf')}

    def sealed_test_scores():
        """(top1, top5) in % under the Track-1 sealed rule, on the current (EMA) weights."""
        from neurips_challenge.score_track1 import score_modules
        start = time.time()
        r = score_modules(model, eeg_projector, img_projector, args, args.test_subject_ids, device, 512, 0,
                          fraction=args.sealed_test_fraction, return_preds=True)
        # per-(subject, image) mean predictions of this epoch (~3 MB): ensembles can take any epoch, no weights needed
        np.savez_compressed(os.path.join(log_dir, f'sealed_preds_ep{epoch:03d}.npz'), mean_norm=r['mean_norm'],
                            **{f's{k}': v.astype(np.float16) for k, v in r['subject_mean_preds'].items()})
        if not os.path.exists(os.path.join(log_dir, 'sealed_gallery.npy')):
            np.save(os.path.join(log_dir, 'sealed_gallery.npy'), r['gallery_normed'])
        if args.save_weights and 100 * r['subject_agg']['top5'] > sealed_best['top5']:
            sealed_best['top5'] = 100 * r['subject_agg']['top5']
            torch.save(build_checkpoint(epoch, -sealed_best['top5']), f"{writer.log_dir}/checkpoint_sealed_best.pth")
        s = r['subject_agg']
        log(f"sealed-rule test top5 {100 * s['top5']:.2f} top1 {100 * s['top1']:.2f} "
            f"(fraction {args.sealed_test_fraction}, {time.time() - start:.0f}s)")
        # per-subject curve: neurips_challenge/cv4_summary.py picks each recipe's epoch budget from it
        with open(os.path.join(log_dir, 'sealed_per_epoch.jsonl'), 'a') as handle:
            handle.write(json.dumps({'epoch': epoch, 'fraction': args.sealed_test_fraction,
                                     'by_subject': {k: 100 * v['top5'] for k, v in r['subject_agg_by_subject'].items()}}) + '\n')
        return 100 * s['top1'], 100 * s['top5']
    probe_history = (
        {'epoch': [], 'eeg_backbone_val_acc': [], 'eeg_align_val_acc': []}
        if args.subject_probe_holdout
        else None
    )
    top1_acc = 0.0
    top5_acc = 0.0
    best_test_sparsity = {}
    last_epoch_train_sparsity = {}

    if args.init_checkpoint:
        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()
        initial_test_loss, initial_top1, initial_top5, initial_sparsity = evaluate_split(
            test_dataloader, 'test-init'
        )
        initial_selected_loss = initial_test_loss
        initial_selected_top1 = initial_top1
        if args.select_best_on == 'val' and val_dataloader is not None:
            initial_selected_loss, initial_selected_top1, _, _ = evaluate_split(
                val_dataloader, 'val-init'
            )
        best_test_loss = initial_selected_loss
        best_top1_acc = initial_top1
        best_top5_acc = initial_top5
        best_selected_top1 = initial_selected_top1
        best_test_epoch = 0
        best_test_sparsity = dict(initial_sparsity)
        if args.save_weights:
            torch.save(
                build_checkpoint(0, initial_selected_loss),
                f"{writer.log_dir}/checkpoint_test_best.pth",
            )
        log(
            f'Initialized best candidate at epoch 0: loss={initial_selected_loss:.4f} '
            f'top1={initial_top1:.2f} top5={initial_top5:.2f}'
        )

    coherent_mix_cohort = None
    if args.mixup_type == 'coherent_group' and args.coherent_mix_scope == 'run':
        coherent_mix_cohort = sample_coherent_subject_weights(
            args.train_subject_ids,
            alpha=args.subject_mixup_alpha,
            self_anchor=args.coherent_mix_self_anchor,
            device=device,
        )

    epochs_without_selection_improvement = 0
    early_stop_reference_loss = best_test_loss
    codabench_history = []
    sam_dir = None  # --sam_rho: unit gradient of the previous step
    for epoch in range(1, total_epochs + 1):
        # Autocast is switched on globally (set_autocast_enabled), so no autocast region ever closes to flush its
        # weight-cast cache: the epoch-end evaluation's no-grad casts survived into the next epoch's first forward,
        # which then had no grad_fn when only the head trained (frozen-backbone probe, 2026-10-09).
        torch.clear_autocast_cache()
        if ema_model is not None and epoch > 1:
            model.load_state_dict(raw_state)
        if args.swa_from_epoch and epoch == args.swa_from_epoch:  # SWAD-style: from here on, evaluate the uniform average of every step's weights
            ema_model = AveragedModel(model, use_buffers=True)
        if args.fm_head_warmup_epochs > 0 and epoch in (1, args.fm_head_warmup_epochs + 1):
            # AdamW skips parameters whose grad is None, so a frozen backbone is simply not
            # stepped; unfreezing restores whatever --fm_unfreeze_last_k left trainable.
            thawed = epoch > args.fm_head_warmup_epochs
            for parameter in model.fm.parameters():
                parameter.requires_grad_(thawed and id(parameter) in trainable_ids)
            log(f'Foundation backbone {"unfrozen" if thawed else "frozen for head warmup"}')
        if args.freeze_eeg_encoder:
            model.eval()
        else:
            model.train()
            if args.freeze_bn_stats:  # SUBJECT_LOG D8: keep the init checkpoint's BatchNorm statistics (few-shot oracles)
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        eeg_projector.train()
        if args.freeze_image_projector:
            img_projector.eval()
        else:
            img_projector.train()
        text_projector.train()
        criterion.train()
        total_loss = 0.0
        total_cl_loss = 0.0
        total_subject_mixup_reg_loss = 0.0
        total_mutual_reconstruction_loss = 0.0
        total_mutual_reconstruction_rows = 0
        total_cohort_reconstruction_loss = 0.0
        total_cohort_reconstruction_rows = 0
        total_subject_router_loss = 0.0
        total_subject_router_rows = 0
        total_subject_adapt_loss = 0.0
        total_subject_adapt_subjects = 0
        total_b2b_loss = 0.0
        total_rel_loss = 0.0
        total_auxiliary_alignment_loss = 0.0
        total_teacher_alignment_loss = 0.0
        total_matryoshka_loss = 0.0
        total_vrex_loss = 0.0
        total_mmd_loss = 0.0
        # Coarse stage: shape the shared geometry with MMD. Fine stage: freeze g() and let the
        # contrastive term refine instance-level retrieval on top of an already-stable space.
        in_coarse_stage = args.coarse_epochs <= 0 or epoch <= args.coarse_epochs
        if shared_encoder is not None and not in_coarse_stage:
            set_requires_grad(shared_encoder, False)
            shared_encoder.eval()
        if args.mixup_type == 'coherent_group' and args.coherent_mix_scope == 'epoch':
            coherent_mix_cohort = sample_coherent_subject_weights(
                args.train_subject_ids,
                alpha=args.subject_mixup_alpha,
                self_anchor=args.coherent_mix_self_anchor,
                device=device,
            )
        latency_subject_ids, latency_shifts = sample_coherent_subject_time_shifts(
            args.train_subject_ids,
            args.coherent_latency_max_shift,
            args.coherent_latency_std,
            device=device,
        )

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}"):
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            optimizer.zero_grad()
            sam_eps = None
            if args.sam_rho > 0 and sam_dir is not None:  # SAM with a stale ascent direction: the previous step's unit gradient
                with torch.no_grad():
                    sam_eps = [args.sam_rho * d for d in sam_dir]
                    torch._foreach_add_(trainable_parameters, sam_eps)
            if args.op_transfer_prob > 0 and random.random() < args.op_transfer_prob:
                op_kw = dict(shrink=args.op_transfer_shrink, extrapolate=args.op_transfer_extrapolate, rotate=args.op_transfer_rotate,
                             boundary=21 if not args.op_transfer_cross_dataset and any(s >= 21 for s in args.train_subject_ids) and min(args.train_subject_ids) < 21 else 0)
                if args.op_transfer_bands:
                    eeg_batch = banded_operator_transfer(eeg_batch, subject_id_batch, args.op_transfer_bands, **op_kw)
                else:
                    eeg_batch = cross_subject_operator_transfer(eeg_batch, subject_id_batch, **op_kw)
            if hyperalign is not None:
                eeg_batch = hyperalign_transfer(eeg_batch, subject_id_batch, *hyperalign, args.hyperalign_prob)
            if cross_dataset_translation is not None:
                eeg_batch = translate_auxiliary_subjects(
                    eeg_batch,
                    subject_id_batch,
                    args.subject_mixup_partition_boundary,
                    *cross_dataset_translation,
                    blend=args.cross_dataset_translation_blend,
                )
            if (
                args.spectral_style_mix_prob > 0
                and random.random() < args.spectral_style_mix_prob
            ):
                eeg_batch = cross_subject_spectral_style_mix(
                    eeg_batch,
                    object_idx_batch,
                    image_idx_batch,
                    subject_id_batch,
                    alpha=args.spectral_style_mix_alpha,
                )
            original_eeg_batch = eeg_batch
            mixup_metadata = None
            backbone_subject_id_batch = subject_id_batch
            reconstruction_eligible_mask = torch.ones_like(
                subject_id_batch, dtype=torch.bool
            )
            if args.subject_mixup_mode == 'raw_eeg':
                synthetic_mask = torch.ones(
                    eeg_batch.shape[0], device=eeg_batch.device, dtype=torch.bool
                )
                if args.mixup_type == 'coherent_group':
                    mixup_out = coherent_cross_subject_stimulus_mix(
                        eeg_batch,
                        object_idx_batch,
                        image_idx_batch,
                        subject_id_batch,
                        coherent_mix_cohort[0],
                        coherent_mix_cohort[1],
                        residual_scale=args.coherent_mix_residual_scale,
                        jitter=args.coherent_mix_jitter,
                        jitter_alpha=args.subject_mixup_alpha,
                    )
                elif args.mixup_type == 'meta_residual':
                    mixup_out = cross_dataset_residual_subject_mix(
                        eeg_batch,
                        object_idx_batch,
                        image_idx_batch,
                        subject_id_batch,
                        partition_boundary=args.subject_mixup_partition_boundary,
                        alpha=args.subject_mixup_alpha,
                        residual_scale=args.meta_residual_scale,
                    )
                else:
                    mix_partition_labels = None
                    if args.subject_mixup_partition_boundary > 0:
                        mix_partition_labels = (
                            subject_id_batch >= args.subject_mixup_partition_boundary
                        ).long()
                    mixup_out = cross_subject_stimulus_mix(
                        eeg_batch,
                        object_idx_batch,
                        image_idx_batch,
                        subject_id_batch,
                        alpha=args.subject_mixup_alpha,
                        mixup_type=args.mixup_type,
                        return_mixup_metadata=args.subject_mixup_reg_lambda > 0,
                        group_self_anchor=args.group_mix_self_anchor,
                        partition_labels=mix_partition_labels,
                        time_varying=args.mix_time_varying,
                        hybrid_group_prob=args.hybrid_group_prob,
                    )
                if args.subject_mixup_reg_lambda > 0:
                    eeg_batch, mixup_metadata = mixup_out
                else:
                    eeg_batch = mixup_out
                if args.subject_mixup_prob < 1:
                    synthetic_mask = torch.rand(
                        eeg_batch.shape[0], device=eeg_batch.device
                    ) < args.subject_mixup_prob
                    view_shape = (eeg_batch.shape[0],) + (1,) * (eeg_batch.ndim - 1)
                    eeg_batch = torch.where(
                        synthetic_mask.view(view_shape), eeg_batch, original_eeg_batch
                    )
                    if mixup_metadata is not None:
                        mixup_metadata['mixed_mask'] &= synthetic_mask
                if args.subject_adapter_rank > 0 or args.tsconv_subject_bn_max_id > 0:
                    # Real rows use their learned source random effect. Virtual
                    # rows use either ID zero (the inference population path)
                    # or a negative virtual-identity key for empirical-Bayes
                    # adapter draws.
                    virtual_ids = (
                        -subject_id_batch
                        if (
                            args.subject_adapter_rank > 0
                            and args.subject_adapter_virtual_prob > 0
                            and args.tsconv_subject_bn_max_id <= 0
                        )
                        else torch.zeros_like(subject_id_batch)
                    )
                    backbone_subject_id_batch = torch.where(
                        synthetic_mask, virtual_ids, subject_id_batch
                    )
                if args.inductive_covariance_align:
                    # A cross-subject mixture has no single owner. Give it the
                    # same all-source prior used by a genuinely unseen subject.
                    backbone_subject_id_batch = torch.where(
                        synthetic_mask,
                        torch.full_like(subject_id_batch, -1),
                        subject_id_batch,
                    )
                reconstruction_eligible_mask = ~synthetic_mask
                if args.subject_mixup_rows_per_image > 0:
                    keep = select_rows_per_exact_image(
                        object_idx_batch,
                        image_idx_batch,
                        args.subject_mixup_rows_per_image,
                    )
                    eeg_batch = eeg_batch[keep]
                    original_eeg_batch = original_eeg_batch[keep]
                    image_feature_batch = image_feature_batch[keep]
                    text_feature_batch = text_feature_batch[keep]
                    subject_id_batch = subject_id_batch[keep]
                    object_idx_batch = object_idx_batch[keep]
                    image_idx_batch = image_idx_batch[keep]
                    backbone_subject_id_batch = backbone_subject_id_batch[keep]
                    reconstruction_eligible_mask = reconstruction_eligible_mask[keep]
                if args.subject_mixup_views > 1:
                    eeg_views = [eeg_batch]
                    for _ in range(1, args.subject_mixup_views):
                        eeg_views.append(cross_subject_stimulus_mix(
                            original_eeg_batch,
                            object_idx_batch,
                            image_idx_batch,
                            subject_id_batch,
                            alpha=args.subject_mixup_alpha,
                            mixup_type='group',
                            group_self_anchor=args.group_mix_self_anchor,
                            time_varying=args.mix_time_varying,
                        ))
                    eeg_batch = torch.cat(eeg_views, dim=0)
                    original_eeg_batch = torch.cat(
                        [original_eeg_batch] * args.subject_mixup_views, dim=0
                    )
                    image_feature_batch = torch.cat(
                        [image_feature_batch] * args.subject_mixup_views, dim=0
                    )
                    text_feature_batch = torch.cat(
                        [text_feature_batch] * args.subject_mixup_views, dim=0
                    )
                    subject_id_batch = subject_id_batch.repeat(args.subject_mixup_views)
                    object_idx_batch = object_idx_batch.repeat(args.subject_mixup_views)
                    image_idx_batch = image_idx_batch.repeat(args.subject_mixup_views)
                    backbone_subject_id_batch = torch.zeros_like(subject_id_batch)
                    reconstruction_eligible_mask = torch.zeros_like(
                        subject_id_batch, dtype=torch.bool
                    )
            if args.coherent_latency_max_shift > 0:
                eeg_batch = coherent_subject_time_shift(
                    eeg_batch,
                    subject_id_batch,
                    latency_subject_ids,
                    latency_shifts,
                )
            if args.input_mixup:
                # REVE authors' mixup (reve_eeg dt.py): one lam ~ U(0,1) per batch, mixed with a shuffled batch
                mix_lam, mix_perm = random.random(), torch.randperm(eeg_batch.shape[0], device=eeg_batch.device)
                eeg_batch = mix_lam * eeg_batch + (1 - mix_lam) * eeg_batch[mix_perm]
            eeg_backbone_batch = run_eeg_backbone(
                model, args, eeg_batch, backbone_subject_id_batch
            )
            if args.batch_transform:  # the transform a test subject's predict() batches get, per subject group here
                from neurips_challenge.score_track1 import batch_transform
                grouped = eeg_backbone_batch.clone()
                if args.batch_transform_mixed:  # sealed condition: shuffled chunks mixing subjects
                    perm = torch.randperm(len(grouped), device=grouped.device)
                    for j in range(0, len(perm), args.batch_transform_mixed):
                        rows = perm[j:j + args.batch_transform_mixed]
                        if len(rows) > 2:
                            grouped[rows] = batch_transform(eeg_backbone_batch[rows], a=args.batch_transform_ab[0], k=2, b=args.batch_transform_ab[1], detach_dir=True)
                else:
                    for sid in torch.unique(backbone_subject_id_batch):
                        rows = backbone_subject_id_batch == sid
                        if rows.sum() > 2:
                            grouped[rows] = batch_transform(eeg_backbone_batch[rows], a=args.batch_transform_ab[0], b=args.batch_transform_ab[1], detach_dir=True)
                eeg_backbone_batch = grouped
            subject_router_loss = eeg_batch.new_tensor(0.0)
            subject_router_rows = 0
            if args.subject_adapter_router_lambda > 0:
                subject_router_loss, subject_router_rows = (
                    model.subject_adapter_router_loss(backbone_subject_id_batch)
                )
            mutual_reconstruction_loss = eeg_batch.new_tensor(0.0)
            mutual_reconstruction_rows = 0
            if args.mutual_reconstruction_lambda > 0:
                mutual_reconstruction_loss, mutual_reconstruction_rows = (
                    compute_mutual_reconstruction_loss(
                        model,
                        eeg_backbone_batch,
                        original_eeg_batch,
                        object_idx_batch,
                        image_idx_batch,
                        subject_id_batch,
                        reconstruction_eligible_mask,
                    )
                )
            cohort_reconstruction_loss = eeg_batch.new_tensor(0.0)
            cohort_reconstruction_rows = 0
            if args.cohort_reconstruction_lambda > 0:
                cohort_reconstruction_loss, cohort_reconstruction_rows = (
                    compute_cohort_reconstruction_loss(
                        model,
                        eeg_backbone_batch,
                        original_eeg_batch,
                        object_idx_batch,
                        image_idx_batch,
                        subject_id_batch,
                    )
                )
            if args.subject_mixup_mode == 'embedding':
                mix_partition_labels = None
                if args.subject_mixup_partition_boundary > 0:
                    mix_partition_labels = (
                        subject_id_batch >= args.subject_mixup_partition_boundary
                    ).long()
                eeg_backbone_batch = cross_subject_stimulus_mix(
                    eeg_backbone_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                    alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type,
                    group_self_anchor=args.group_mix_self_anchor,
                    partition_labels=mix_partition_labels,
                    time_varying=args.mix_time_varying,
                    hybrid_group_prob=args.hybrid_group_prob,
                )
            arch_out = forward_architecture(eeg_backbone_batch)

            loss = eeg_batch.new_tensor(0.0)

            if args.subject_adapter_router_lambda > 0:
                loss = loss + args.subject_adapter_router_lambda * subject_router_loss
                total_subject_router_loss += subject_router_loss.item()
                total_subject_router_rows += subject_router_rows
            if args.mutual_reconstruction_lambda > 0:
                loss = loss + (
                    args.mutual_reconstruction_lambda * mutual_reconstruction_loss
                )
                total_mutual_reconstruction_loss += mutual_reconstruction_loss.item()
                total_mutual_reconstruction_rows += mutual_reconstruction_rows
            if args.cohort_reconstruction_lambda > 0:
                loss = loss + (
                    args.cohort_reconstruction_lambda * cohort_reconstruction_loss
                )
                total_cohort_reconstruction_loss += cohort_reconstruction_loss.item()
                total_cohort_reconstruction_rows += cohort_reconstruction_rows

            eeg_feature_batch = arch_out['eeg_feature']
            eeg_confidence_batch = arch_out['eeg_confidence']
            if args.subject_mixup_reg_lambda > 0:
                reg_feature_batch = eeg_backbone_batch if args.subject_mixup_reg_on_backbone else eeg_feature_batch
                if args.single_emb_stop_grad:
                    with torch.no_grad():
                        original_eeg_feature_batch = run_eeg_backbone(model, args, original_eeg_batch, subject_id_batch)
                        if not args.subject_mixup_reg_on_backbone:
                            original_eeg_feature_batch = forward_architecture(original_eeg_feature_batch)['eeg_feature']
                else:
                    original_eeg_feature_batch = run_eeg_backbone(model, args, original_eeg_batch, subject_id_batch)
                    if not args.subject_mixup_reg_on_backbone:
                        original_eeg_feature_batch = forward_architecture(original_eeg_feature_batch)['eeg_feature']
                subject_mixup_reg_loss = compute_subject_mixup_regularization(
                    reg_feature_batch,
                    original_eeg_feature_batch,
                    mixup_metadata['partner_indices'],
                    mixup_metadata['mixed_mask'],
                )
                loss = loss + args.subject_mixup_reg_lambda * subject_mixup_reg_loss
                total_subject_mixup_reg_loss += subject_mixup_reg_loss.item()
            primary_image_batch = primary_image_target(image_feature_batch)
            image_feature_proj = img_projector(primary_image_batch)
            text_feature_proj = text_projector(
                primary_image_target(text_feature_batch)
            )
            if args.subject_mixup_hard_view:
                keep = select_hard_counterfactual_views(
                    eeg_feature_batch,
                    image_feature_proj,
                    args.subject_mixup_views,
                )
                eeg_backbone_batch = eeg_backbone_batch[keep]
                eeg_feature_batch = eeg_feature_batch[keep]
                image_feature_batch = image_feature_batch[keep]
                text_feature_batch = text_feature_batch[keep]
                image_feature_proj = image_feature_proj[keep]
                text_feature_proj = text_feature_proj[keep]
                subject_id_batch = subject_id_batch[keep]
                object_idx_batch = object_idx_batch[keep]
                image_idx_batch = image_idx_batch[keep]
                if eeg_confidence_batch is not None:
                    eeg_confidence_batch = eeg_confidence_batch[keep]
            positive_mask = build_image_positive_mask(
                object_idx_batch, image_idx_batch, args.concept_positives
            )
            eeg_feature_for_loss, image_feature_for_loss = maybe_apply_train_saw(
                eeg_feature_batch,
                image_feature_proj,
                subject_id_batch,
            )
            auxiliary_active = (
                auxiliary_eeg_projectors is not None
                and (
                    args.auxiliary_alignment_epochs <= 0
                    or epoch <= args.auxiliary_alignment_epochs
                )
            )
            if auxiliary_active:
                auxiliary_losses = []
                if args.auxiliary_image_targets:
                    auxiliary_target_batches = auxiliary_image_target_blocks(
                        image_feature_batch
                    )
                else:
                    auxiliary_target_batches = [
                        image_feature_batch
                    ] * len(auxiliary_img_projectors)
                for auxiliary_eeg, auxiliary_img, auxiliary_target in zip(
                        auxiliary_eeg_projectors, auxiliary_img_projectors,
                        auxiliary_target_batches):
                    auxiliary_losses.append(criterion.multi_positive_pair_loss(
                        auxiliary_eeg(eeg_backbone_batch),
                        auxiliary_img(auxiliary_target),
                        positive_mask,
                    ))
                auxiliary_alignment_loss = torch.stack(auxiliary_losses).mean()
                loss = loss + args.auxiliary_alignment_lambda * auxiliary_alignment_loss
                total_auxiliary_alignment_loss += auxiliary_alignment_loss.item()
            if args.teacher_alignment_lambda > 0:
                teacher_target = auxiliary_image_target_blocks(image_feature_batch)[-1]
                teacher_alignment_loss = (
                    1.0 - F.cosine_similarity(
                        eeg_feature_batch,
                        teacher_target.detach(),
                        dim=1,
                    )
                ).mean()
                loss = loss + args.teacher_alignment_lambda * teacher_alignment_loss
                total_teacher_alignment_loss += teacher_alignment_loss.item()
            boost_w = None
            if args.boost_gamma > 0:  # AUG_LOG boosting: up-weight trials the current ensemble gets wrong (targets unchanged)
                with torch.no_grad():
                    t = batch[7].to(device).float()
                    if t.shape[1] == 1:  # precomputed per-trial p_true (neurips_challenge/boost_weights.py)
                        p_true = t[:, 0]
                    else:
                        p_true = ((F.normalize(t, dim=1) @ F.normalize(image_feature_for_loss.float(), dim=1).T
                                   / args.distill_tau).softmax(dim=1) * positive_mask.float()).sum(1)
                    boost_w = (1 - p_true).clamp_min(0) ** args.boost_gamma
                    boost_w = boost_w / boost_w.mean().clamp_min(1e-8)
            cl_loss = compute_cross_modal_loss(
                criterion,
                {'feature': eeg_feature_for_loss, 'confidence': eeg_confidence_batch},
                image_feature_for_loss,
                text_feature_proj,
                positive_mask,
                args.multi_positive_loss,
                sample_weights=(
                    train_subject_weight_lookup[subject_id_batch]
                    if train_subject_weight_lookup is not None
                    else boost_w
                ),
            )
            if args.input_mixup:  # ... and the loss is lam * L(own targets) + (1 - lam) * L(partner's targets)
                cl_loss = mix_lam * cl_loss + (1 - mix_lam) * compute_cross_modal_loss(
                    criterion,
                    {'feature': eeg_feature_for_loss, 'confidence': eeg_confidence_batch},
                    image_feature_for_loss[mix_perm],
                    text_feature_proj[mix_perm],
                    positive_mask[mix_perm][:, mix_perm],
                    args.multi_positive_loss,
                )
            if args.mask_concept_neg:  # false-negative elimination: same-concept, different-image pairs leave the softmax
                q_m = criterion._normalize_inputs(eeg_feature_for_loss, None)[0]
                lg = (q_m @ F.normalize(image_feature_for_loss, dim=1).T) * criterion._get_logit_scale()
                same_c = object_idx_batch[:, None] == object_idx_batch[None, :]
                lg = lg.masked_fill(same_c & ~positive_mask, float('-inf'))
                cl_loss = -(torch.log_softmax(lg, 1).masked_fill(~positive_mask, float('-inf'))).logsumexp(1).mean()
            if args.loss_trunc > 0:  # robust InfoNCE: drop the batch's highest-loss fraction of trials (label noise:
                # most single trials carry little image information; their large losses only add gradient noise)
                q_t = criterion._normalize_inputs(eeg_feature_for_loss, None)[0]
                lg = (q_t @ F.normalize(image_feature_for_loss, dim=1).T) * criterion._get_logit_scale()
                per = -(torch.log_softmax(lg, 1) + torch.log(positive_mask.float().clamp_min(1e-30))).logsumexp(1)
                keep_n = max(1, int(round(len(per) * (1 - args.loss_trunc))))
                cl_loss = per.topk(keep_n, largest=False).values.mean()
            if siglip_head is not None:  # pairwise sigmoid loss, in-batch; same-image duplicates are positives
                q_s = criterion._normalize_inputs(eeg_feature_for_loss, None)[0]
                z = q_s @ F.normalize(image_feature_for_loss, dim=1).T + siglip_head.b
                y = positive_mask.float() * 2 - 1
                cl_loss = -F.logsigmoid(y * z).sum() / len(z)
            if global_gallery is not None:
                gal = F.normalize(img_projector(global_gallery), dim=1)
                q = criterion._normalize_inputs(eeg_feature_for_loss, None)[0]
                gidx = object_idx_batch.long() * global_n_img + image_idx_batch.long()
                logits_g = (q @ gal.T) * criterion._get_logit_scale()
                if args.global_concept_eps > 0 or args.global_soft_tau > 0:
                    with torch.no_grad():  # soft targets over the whole training gallery
                        if args.global_soft_tau > 0:  # image-image similarity structure (distillation)
                            tgt = torch.softmax((gal[gidx] @ gal.T) / args.global_soft_tau, dim=1)
                        else:
                            tgt = F.one_hot(gidx, len(gal)).float()
                        if args.global_concept_eps > 0:  # spread eps over the other images of the same concept
                            same = (torch.arange(len(gal), device=device)[None] // global_n_img) == (gidx[:, None] // global_n_img)
                            same = same & (torch.arange(len(gal), device=device)[None] != gidx[:, None])
                            tgt = (1 - args.global_concept_eps) * tgt + args.global_concept_eps * same.float() / same.sum(1, keepdim=True)
                    g_loss = -(tgt * F.log_softmax(logits_g, dim=1)).sum(1).mean()
                else:
                    g_loss = F.cross_entropy(logits_g, gidx)
                cl_loss = g_loss if args.global_negatives_replace else cl_loss + args.global_negatives * g_loss
                if args.global_symmetric > 0:  # image -> EEG over the batch's queries (anti-hub); duplicates are positives
                    li = logits_g[:, gidx].T                                   # (B images, B queries)
                    lp = F.log_softmax(li, dim=1)
                    cl_loss = cl_loss + args.global_symmetric * -(torch.logsumexp(lp.masked_fill(~positive_mask, -1e9), 1)).mean()
            loss = loss + cl_loss
            total_cl_loss += cl_loss.item()
            if args.repeat_mean_loss > 0 or neighbor_heads is not None:
                assert eeg_feature_for_loss.shape[0] == subject_id_batch.shape[0], 'L2/L4 losses need unfiltered batches'
            if args.repeat_mean_loss > 0:
                # lever L2: the sealed metric averages raw predictions over repeats, so train the averages too
                key = (subject_id_batch.long() * 4096 + object_idx_batch.long()) * 64 + image_idx_batch.long()
                _, inv = torch.unique(key, return_inverse=True)
                n_groups = int(inv.max()) + 1
                counts = torch.bincount(inv, minlength=n_groups).to(eeg_feature_for_loss.dtype)
                mean_feat = torch.zeros(n_groups, eeg_feature_for_loss.shape[1], device=device,
                                        dtype=eeg_feature_for_loss.dtype).index_add_(0, inv, eeg_feature_for_loss)
                mean_feat = mean_feat / counts[:, None]
                first = torch.full((n_groups,), inv.numel(), device=device, dtype=torch.long).scatter_reduce_(
                    0, inv, torch.arange(inv.numel(), device=device), reduce='amin')
                rm_loss = compute_cross_modal_loss(
                    criterion, {'feature': mean_feat, 'confidence': None}, image_feature_for_loss[first],
                    text_feature_proj[first],
                    build_image_positive_mask(object_idx_batch[first], image_idx_batch[first]), True)
                loss = loss + args.repeat_mean_loss * rm_loss
            if neighbor_heads is not None:
                rep_idx = batch[6].to(device).long()
                nb = neighbor_table[subject_id_batch.long(), object_idx_batch.long(), image_idx_batch.long(), rep_idx]
                nb_losses = []
                for j, head in enumerate(neighbor_heads):
                    valid = nb[:, j] >= 0
                    if int(valid.sum()) < 8:
                        continue
                    k = nb[valid, j]
                    nb_losses.append(compute_cross_modal_loss(
                        criterion, {'feature': head(eeg_feature_for_loss[valid]), 'confidence': None},
                        neighbor_targets[k], text_feature_proj[valid], k[:, None] == k[None, :], True))
                if nb_losses:
                    loss = loss + args.neighbor_loss * torch.stack(nb_losses).mean()
            if args.distill_lambda > 0:
                # Ensemble distillation: match the teacher's softmax over this batch's images.
                teacher = F.normalize(batch[7].to(device).float(), dim=1)
                images = F.normalize(image_feature_for_loss.float(), dim=1)
                student_logits = F.normalize(eeg_feature_for_loss.float(), dim=1) @ images.T / args.distill_tau
                teacher_probs = (teacher @ images.T / args.distill_tau).softmax(dim=1)
                distill_loss = F.kl_div(student_logits.log_softmax(dim=1), teacher_probs, reduction='batchmean')
                loss = loss + args.distill_lambda * distill_loss
            if args.anticorr_lambda > 0:
                # AUG_LOG negative correlation: penalise the probability the student puts on the in-batch images the
                # OTHER members wrongly favour (their mass off the true image). No imitation of their answers.
                others = F.normalize(batch[7].to(device).float(), dim=1)
                images = F.normalize(image_feature_for_loss.float(), dim=1)
                wrong = ~((object_idx_batch[:, None] == object_idx_batch[None, :]) & (image_idx_batch[:, None] == image_idx_batch[None, :]))
                p_s = (F.normalize(eeg_feature_for_loss.float(), dim=1) @ images.T / args.distill_tau).softmax(dim=1)
                p_o = (others @ images.T / args.distill_tau).softmax(dim=1)
                loss = loss + args.anticorr_lambda * (p_s * p_o * wrong).sum(1).mean()
            matryoshka_active = (
                args.matryoshka_lambda > 0
                and (args.matryoshka_epochs <= 0 or epoch <= args.matryoshka_epochs)
            )
            if matryoshka_active:
                nested_losses = [
                    criterion.multi_positive_pair_loss(
                        eeg_feature_for_loss[:, :dim],
                        image_feature_for_loss[:, :dim],
                        positive_mask,
                    )
                    for dim in matryoshka_dims
                ]
                matryoshka_loss = torch.stack(nested_losses).mean()
                loss = loss + args.matryoshka_lambda * matryoshka_loss
                total_matryoshka_loss += matryoshka_loss.item()

            if args.mmd_lambda > 0 and in_coarse_stage:
                mmd_loss = criterion.mk_mmd(eeg_feature_for_loss, image_feature_for_loss)
                loss = loss + args.mmd_lambda * mmd_loss
                total_mmd_loss += mmd_loss.item()

            if args.brain2brain_lambda > 0:
                b2b_loss = criterion.brain_to_brain_loss(eeg_feature_for_loss, positive_mask)
                loss = loss + args.brain2brain_lambda * b2b_loss
                total_b2b_loss += b2b_loss.item()

            if args.relational_lambda > 0:
                rel_loss = criterion.relational_loss(
                    eeg_feature_for_loss, image_feature_for_loss, args.relational_temperature
                )
                loss = loss + args.relational_lambda * rel_loss
                total_rel_loss += rel_loss.item()

            if args.rsa_lambda > 0:
                rsa_loss = criterion.rsa_loss(eeg_feature_for_loss, image_feature_for_loss, positive_mask)
                loss = loss + args.rsa_lambda * rsa_loss
                total_rel_loss += rsa_loss.item()

            if args.iso_lambda > 0:
                iso_loss = criterion.isometry_loss(eeg_feature_for_loss, image_idx_batch, subject_id_batch)
                loss = loss + args.iso_lambda * iso_loss
                total_rel_loss += iso_loss.item()

            if args.pr_lambda > 0:
                loss = loss + args.pr_lambda * criterion.collapse_penalty(eeg_feature_for_loss, args.pr_floor)

            if args.vrex_lambda > 0:
                row_losses = criterion.multi_positive_row_losses(
                    eeg_feature_for_loss, image_feature_for_loss, positive_mask
                )
                risks = torch.stack([
                    row_losses[subject_id_batch == s].mean()
                    for s in torch.unique(subject_id_batch)
                ])
                if risks.numel() > 1:
                    vrex_loss = risks.var(unbiased=False)
                    loss = loss + args.vrex_lambda * vrex_loss
                    total_vrex_loss += vrex_loss.item()

            if args.subject_adapt_lambda > 0:
                subject_adapt_loss, valid_subjects = compute_subject_adaptation_loss(
                    args,
                    criterion,
                    eeg_feature_batch,
                    image_feature_proj,
                    text_feature_proj,
                    subject_id_batch,
                    object_idx_batch,
                    image_idx_batch,
                )
                loss = loss + args.subject_adapt_lambda * subject_adapt_loss
                total_subject_adapt_loss += subject_adapt_loss.item()
                total_subject_adapt_subjects += valid_subjects

            grad_scaler.scale(loss).backward()
            if args.sam_rho > 0:  # back to w; the gradient taken at w + eps is what the optimizer applies
                with torch.no_grad():
                    if sam_eps is not None:
                        torch._foreach_sub_(trainable_parameters, sam_eps)
                    g = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable_parameters]
                    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x.float()) for x in g]))
                    sam_dir = [x / (norm + 1e-12) for x in g] if torch.isfinite(norm) else None
            if args.grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if ema_model is not None:
                ema_model.update_parameters(model)
            if args.lr_scheduler in ('onecycle', 'wsd'):
                scheduler.step()
            elif args.lr_scheduler == 'plateau' and plateau_warmup.last_epoch < warmup_steps:
                plateau_warmup.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/train_cl', total_cl_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_subject_mixup_reg', total_subject_mixup_reg_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_mutual_reconstruction', total_mutual_reconstruction_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_cohort_reconstruction', total_cohort_reconstruction_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_subject_router', total_subject_router_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_subject_adapt', total_subject_adapt_loss / len(dataloader), epoch)
        writer.add_scalar(
            'Loss/train_auxiliary_alignment',
            total_auxiliary_alignment_loss / len(dataloader),
            epoch,
        )
        writer.add_scalar(
            'Loss/train_teacher_alignment',
            total_teacher_alignment_loss / len(dataloader),
            epoch,
        )
        writer.add_scalar(
            'Loss/train_matryoshka',
            total_matryoshka_loss / len(dataloader),
            epoch,
        )
        log(
            f"Epoch [{epoch}/{total_epochs}] TrainLoss={avg_loss:.4f} "
            f"CL={total_cl_loss / len(dataloader):.4f} "
            f"MIXREG={total_subject_mixup_reg_loss / len(dataloader):.4f} "
            f"MRECON={total_mutual_reconstruction_loss / len(dataloader):.4f} "
            f"MRECON_ROWS={total_mutual_reconstruction_rows / len(dataloader):.1f} "
            f"CRECON={total_cohort_reconstruction_loss / len(dataloader):.4f} "
            f"CRECON_ROWS={total_cohort_reconstruction_rows / len(dataloader):.1f} "
            f"SROUTER={total_subject_router_loss / len(dataloader):.4f} "
            f"SROUTER_ROWS={total_subject_router_rows / len(dataloader):.1f} "
            f"SADAPT={total_subject_adapt_loss / len(dataloader):.4f} "
            f"SADAPT_SUBJ={total_subject_adapt_subjects / len(dataloader):.2f} "
            f"B2B={total_b2b_loss / len(dataloader):.4f} "
            f"REL={total_rel_loss / len(dataloader):.4f} "
            f"AUX={total_auxiliary_alignment_loss / len(dataloader):.4f} "
            f"TEACH={total_teacher_alignment_loss / len(dataloader):.4f} "
            f"MRL={total_matryoshka_loss / len(dataloader):.4f} "
            f"VREX={total_vrex_loss / len(dataloader):.4f} "
            f"MMD={total_mmd_loss / len(dataloader):.4f}{'' if in_coarse_stage else '(fine)'}"
        )

        # checkpoint_last.pth is deliberately not written: nothing reads it (evaluate.py and the
        # end-of-run reload both use checkpoint_test_best.pth), and it was 270GB across the run
        # archive. Pass --save_last_checkpoint if you need the final-epoch weights.
        if args.save_weights and args.save_last_checkpoint:
            torch.save(build_checkpoint(epoch, avg_loss), f"{writer.log_dir}/checkpoint_last.pth")

        if ema_model is not None:
            # Everything below -- test/val metrics, selection, checkpoints, the Codabench
            # estimate -- sees the EMA weights; the raw weights are restored before training.
            raw_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_model.module.state_dict())
        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()

        avg_test_loss, top1_acc, top5_acc, test_sparsity = evaluate_split(test_dataloader, "test")
        if args.sealed_test_metric:
            # Challenge runs only: test top-1/top-5 (and result.csv's best ones) follow the sealed
            # rule via the grader's own pass on this run's test data; the loss stays as above.
            history['avg_eeg_top1'].append(top1_acc)
            history['avg_eeg_top5'].append(top5_acc)
            top1_acc = top5_acc = float('nan')
            if epoch % args.sealed_test_every == 0 or epoch == total_epochs:
                top1_acc, top5_acc = sealed_test_scores()
        if per_subject_last:
            path = os.path.join(log_dir, 'per_subject_test.csv')
            with open(path, 'a') as handle:
                if handle.tell() == 0:
                    handle.write('epoch,subject,test_loss,top1,top5\n')
                for subject, (s_loss, s_top1, s_top5) in per_subject_last.items():
                    handle.write(f'{epoch},{subject},{s_loss:.5f},{s_top1:.2f},{s_top5:.2f}\n')
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Acc/top1_test', top1_acc, epoch)
        writer.add_scalar('Acc/top5_test', top5_acc, epoch)
        if test_sparsity:
            writer.add_scalar('Sparsity/eeg_l0_frac_test', test_sparsity['eeg_l0_frac'], epoch)
            writer.add_scalar('Sparsity/img_l0_frac_test', test_sparsity['img_l0_frac'], epoch)

        if epoch == total_epochs and args.projector_activation in _SPARSE_PROJECTOR_ACTIVATIONS:
            train_sparsity_loader = DataLoader(
                train_dataset,
                batch_size=200,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            last_epoch_train_sparsity = collect_split_sparsity(train_sparsity_loader, 'train')
            if last_epoch_train_sparsity:
                writer.add_scalar(
                    'Sparsity/eeg_active_feat_frac_train_last',
                    last_epoch_train_sparsity['eeg_active_feat_frac'],
                    epoch,
                )
                writer.add_scalar(
                    'Sparsity/img_active_feat_frac_train_last',
                    last_epoch_train_sparsity['img_active_feat_frac'],
                    epoch,
                )

        val_loss, val_top1, val_top5 = float('nan'), float('nan'), float('nan')
        if val_dataloader is not None:
            val_loss, val_top1, val_top5, _ = evaluate_split(val_dataloader, "val")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/top1_val', val_top1, epoch)
            writer.add_scalar('Acc/top5_val', val_top5, epoch)

        bb_probe_acc_pct = float('nan')
        clip_probe_acc_pct = float('nan')
        if (
            args.subject_probe_holdout
            and epoch % args.subject_probe_interval == 0
            and subj_probe_heads is not None
            and subj_probe_train_dataloader is not None
            and subj_probe_val_dataloader is not None
        ):
            model.eval()
            eeg_projector.eval()
            for h in subj_probe_heads.values():
                h.train()
            probe_train_loss_sum = {k: 0.0 for k in subj_probe_heads}
            probe_train_batches = 0
            for probe_batch in subj_probe_train_dataloader:
                eeg_p = probe_batch[0].to(device)
                sid_p = probe_batch[3].to(device)
                with torch.no_grad():
                    bb_res = run_eeg_backbone(model, args, eeg_p, sid_p, return_intermediate=True)
                    if not isinstance(bb_res, dict):
                        bb_res = {'backbone': bb_res}
                    # Some encoders expose a raw conv flatten as `backbone` and the
                    # projector input as `output`. Use the latter when available.
                    bb_res['align'] = eeg_projector(bb_res.get('output', bb_res['backbone']))
                
                y = encode_probe_labels(sid_p)
                for name, head in subj_probe_heads.items():
                    subj_probe_optimizers[name].zero_grad()
                    layer_key = name.replace('eeg_', '')
                    feats = bb_res[layer_key]
                    pl = head(feats.detach())
                    pls = F.cross_entropy(pl, y)
                    pls.backward()
                    subj_probe_optimizers[name].step()
                    probe_train_loss_sum[name] += float(pls.item())
                probe_train_batches += 1
            
            for h in subj_probe_heads.values():
                h.eval()
            probe_val_acc_sum = {k: 0.0 for k in subj_probe_heads}
            probe_val_batches = 0
            with torch.no_grad():
                for probe_batch in subj_probe_val_dataloader:
                    eeg_p = probe_batch[0].to(device)
                    sid_p = probe_batch[3].to(device)
                    bb_res = run_eeg_backbone(model, args, eeg_p, sid_p, return_intermediate=True)
                    if not isinstance(bb_res, dict):
                        bb_res = {'backbone': bb_res}
                    bb_res['align'] = eeg_projector(bb_res.get('output', bb_res['backbone']))
                    
                    y = encode_probe_labels(sid_p)
                    for name, head in subj_probe_heads.items():
                        layer_key = name.replace('eeg_', '')
                        feats = bb_res[layer_key]
                        pred = torch.argmax(head(feats), dim=1)
                        probe_val_acc_sum[name] += float((pred == y).float().mean().item())
                    probe_val_batches += 1
            if probe_train_batches > 0 and probe_val_batches > 0:
                parts = []
                for name in subj_probe_heads:
                    tloss = probe_train_loss_sum[name] / probe_train_batches
                    vacc = probe_val_acc_sum[name] / probe_val_batches
                    writer.add_scalar(f'SubjectProbe/{name}_val_acc', vacc, epoch)
                    writer.add_scalar(f'SubjectProbe/{name}_train_loss', tloss, epoch)
                    parts.append(f"{name}:val_acc={vacc:.3f}")
                log("SubjProbe " + " | ".join(parts))
                bb_probe_acc_pct = (probe_val_acc_sum['eeg_backbone'] / probe_val_batches) * 100.0
                clip_probe_acc_pct = (probe_val_acc_sum['eeg_align'] / probe_val_batches) * 100.0

        if probe_history is not None and epoch % args.subject_probe_interval == 0:
            probe_history['epoch'].append(epoch)
            # Log all probe accuracies to history
            for name in subj_probe_heads:
                acc = (probe_val_acc_sum[name] / probe_val_batches) * 100.0
                key = f'{name}_val_acc'
                if key not in probe_history:
                    probe_history[key] = []
                probe_history[key].append(acc)

        if args.codabench_eval_every > 0 and epoch % args.codabench_eval_every == 0:
            # Same code path as the post-run score_track1.py, on a fixed trial subset of the
            # official 120 Hz grader input; the final full score stays the number of record.
            from neurips_challenge.score_track1 import OFFICIAL_EEG_DIR, score_modules
            cb_cfg = copy.copy(args)
            cb_cfg.eeg_data_dir = OFFICIAL_EEG_DIR
            model.eval(); eeg_projector.eval(); img_projector.eval()
            cb_start = time.time()
            cb = score_modules(model, eeg_projector, img_projector, cb_cfg, args.test_subject_ids,
                               device, 512, 0, fraction=args.codabench_eval_fraction)['per_epoch']
            codabench_history.append({'epoch': epoch, **cb})
            pd.DataFrame(codabench_history).to_csv(os.path.join(log_dir, 'codabench_estimates.csv'), index=False)
            writer.add_scalar('Codabench/est_top5', cb['top5'], epoch)
            log(f"codabench estimate ({cb['n_queries']} queries, {time.time() - cb_start:.0f}s): "
                f"top5 {cb['top5']:.4f} top1 {cb['top1']:.4f}")

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(avg_test_loss)
        history['top1_acc'].append(top1_acc)
        history['test_top5_acc'].append(top5_acc)
        history['val_loss'].append(val_loss)
        history['val_top1_acc'].append(val_top1)
        history['val_top5_acc'].append(val_top5)
        # Persist each epoch, including interrupted runs. top1_acc is test top-1.
        metrics_path = os.path.join(log_dir, 'epoch_metrics.csv')
        pd.DataFrame(history).to_csv(metrics_path + '.tmp', index=False)
        os.replace(metrics_path + '.tmp', metrics_path)

        selected_loss = avg_test_loss
        selected_top1 = top1_acc
        selected_top5 = top5_acc
        if args.select_best_on == 'val' and not np.isnan(val_loss):
            selected_loss = val_loss
            selected_top1 = val_top1
            selected_top5 = val_top5
        if args.select_best_metric == 'top5':
            # Contrastive loss keeps rising once the model gets confident, long after ranking
            # is still improving: on all 8 Track-1 sweep runs the val-loss minimum sat 2.5-5.4
            # top-5 points below the val-accuracy maximum. Rank on the metric we report.
            selected_loss = -selected_top5

        if args.select_best_on == 'fixed':
            # Honest refit protocol: the epoch budget was chosen on source validation
            # before this run.  Test metrics are still logged for reporting, but may
            # not influence which weights are retained.
            is_better = epoch == min(args.stop_after_epoch or total_epochs, total_epochs)  # a stop past the schedule = its last epoch
        else:
            is_better = False
            if selected_loss < best_test_loss:
                is_better = True
            elif selected_loss == best_test_loss and selected_top1 > best_top1_acc:
                is_better = True

        if is_better and args.sealed_test_metric and np.isnan(top1_acc):
            # an off-epoch became the best: score it so result.csv's best numbers are never NaN
            top1_acc, top5_acc = sealed_test_scores()
            history['top1_acc'][-1], history['test_top5_acc'][-1] = top1_acc, top5_acc
            pd.DataFrame(history).to_csv(metrics_path + '.tmp', index=False)
            os.replace(metrics_path + '.tmp', metrics_path)
        if is_better:
            best_test_loss = selected_loss
            best_top5_acc = top5_acc
            best_top1_acc = top1_acc
            best_selected_top1 = selected_top1
            best_test_epoch = epoch
            best_test_sparsity = dict(test_sparsity)
            if args.save_weights:
                torch.save(build_checkpoint(epoch, selected_loss), f"{writer.log_dir}/checkpoint_test_best.pth")
        if args.save_weights and epoch in args.save_epoch_checkpoints:  # same (EMA-swapped) weights as evaluated
            torch.save(build_checkpoint(epoch, selected_loss), f"{writer.log_dir}/checkpoint_ep{epoch:03d}.pth")
        if args.save_testsel_checkpoint and avg_test_loss < best_testsel_loss:
            # Matched control: what test-loss selection would have kept on this same
            # trajectory. Never influences training or the selected checkpoint.
            best_testsel_loss = avg_test_loss
            best_testsel_epoch = epoch
            torch.save(build_checkpoint(epoch, avg_test_loss), f"{writer.log_dir}/checkpoint_testsel_best.pth")
        if args.early_stop_patience > 0:
            if selected_loss < early_stop_reference_loss - args.early_stop_min_delta:
                early_stop_reference_loss = selected_loss
                epochs_without_selection_improvement = 0
            else:
                epochs_without_selection_improvement += 1

        if args.lr_scheduler == 'plateau':
            scheduler.step(top5_acc)  # the per-epoch sealed top-5, the same signal as --sealed_early_stop_patience
        elif scheduler is not None and args.lr_scheduler not in ('onecycle', 'wsd'):
            scheduler.step()  # onecycle/wsd already stepped per optimizer step

        if (
            args.early_stop_patience > 0
            and epochs_without_selection_improvement >= args.early_stop_patience
        ):
            log(
                f"Early stopping after epoch {epoch}: no source-validation "
                f"improvement for {epochs_without_selection_improvement} epochs "
                f"(best epoch {best_test_epoch})."
            )
            break
        if args.sealed_early_stop_patience > 0 and not np.isnan(top5_acc):
            if top5_acc > sealed_best_top5:
                sealed_best_top5, sealed_stale = top5_acc, 0
            else:
                sealed_stale += 1
            if sealed_stale >= args.sealed_early_stop_patience:
                log(f"Early stopping after epoch {epoch}: sealed-rule test top-5 not improved for {sealed_stale} "
                    f"evaluations (best {sealed_best_top5:.2f}).")
                if args.save_weights and not is_better:  # --select_best_on fixed keeps the epoch reached
                    torch.save(build_checkpoint(epoch, selected_loss), f"{writer.log_dir}/checkpoint_test_best.pth")
                break
        if epoch == args.stop_after_epoch:
            log(f"Stopping after epoch {epoch} of a {total_epochs}-epoch schedule (--stop_after_epoch).")
            break

    save_training_plot(history, os.path.join(log_dir, 'training_metrics.png'))
    if probe_history is not None:
        probe_history_path = os.path.join(log_dir, 'probe_metrics.csv')
        pd.DataFrame(probe_history).to_csv(probe_history_path, index=False)
        probe_plot_path = os.path.join(log_dir, 'probe_metrics.png')
        save_probe_plot(probe_history, probe_plot_path)
        log(f"Saved probe metrics CSV: {probe_history_path}")
        log(f"Saved probe metrics plot: {probe_plot_path}")

    if args.save_eeg_norm_metadata and args.save_weights:
        best_ckpt_path = os.path.join(writer.log_dir, 'checkpoint_test_best.pth')
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt['model_state_dict'])
            eeg_projector.load_state_dict(best_ckpt['eeg_projector_state_dict'])
            img_projector.load_state_dict(best_ckpt['img_projector_state_dict'])
            text_projector.load_state_dict(best_ckpt['text_projector_state_dict'])
            if eeg_confidence_head is not None and 'eeg_confidence_head_state_dict' in best_ckpt:
                eeg_confidence_head.load_state_dict(best_ckpt['eeg_confidence_head_state_dict'])
            export_eeg_norm_metadata(
                train_dataset,
                'train',
                os.path.join(writer.log_dir, 'eeg_norm_metadata'),
            )
        else:
            log("Skipping EEG norm metadata export: checkpoint_test_best.pth not found.")

    result_dict = {
        'architecture': 'baseline',
        'eval_mode': args.eval_mode,
        'eeg_confidence_mode': args.eeg_confidence_mode,
        'top1 acc': f'{top1_acc:.3f}',
        'top5 acc': f'{top5_acc:.3f}',
        'best top1 acc': f'{best_top1_acc:.3f}',
        'best top5 acc': f'{best_top5_acc:.3f}',
        'best test loss': f'{best_test_loss:.3f}',
        'best epoch': best_test_epoch,
    }
    if device.type == 'cuda':
        result_dict.update({
            'peak cuda allocated gb': f'{torch.cuda.max_memory_allocated(device) / 2**30:.3f}',
            'peak cuda reserved gb': f'{torch.cuda.max_memory_reserved(device) / 2**30:.3f}',
        })
    if args.projector_activation in _SPARSE_PROJECTOR_ACTIVATIONS and best_test_sparsity:
        result_dict.update({
            'eeg_l0_mean': f"{best_test_sparsity['eeg_l0_mean']:.3f}",
            'eeg_l0_frac': f"{best_test_sparsity['eeg_l0_frac']:.3f}",
            'eeg_active_feat_frac': f"{best_test_sparsity['eeg_active_feat_frac']:.3f}",
            'img_l0_mean': f"{best_test_sparsity['img_l0_mean']:.3f}",
            'img_l0_frac': f"{best_test_sparsity['img_l0_frac']:.3f}",
            'img_active_feat_frac': f"{best_test_sparsity['img_active_feat_frac']:.3f}",
        })
    if args.projector_activation in _SPARSE_PROJECTOR_ACTIVATIONS and last_epoch_train_sparsity:
        result_dict.update({
            'eeg_active_feat_frac_train_last': f"{last_epoch_train_sparsity['eeg_active_feat_frac']:.3f}",
            'img_active_feat_frac_train_last': f"{last_epoch_train_sparsity['img_active_feat_frac']:.3f}",
            'eeg_l0_frac_train_last': f"{last_epoch_train_sparsity['eeg_l0_frac']:.3f}",
            'img_l0_frac_train_last': f"{last_epoch_train_sparsity['img_l0_frac']:.3f}",
        })
    if args.t_learnable:
        with torch.no_grad():
            result_dict['learned_logit_scale'] = f"{criterion._get_logit_scale().item():.3f}"
    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(log_dir, 'result.csv'), index=False)
    log(f'best test loss: {best_test_loss:.4f} top5 acc: {best_top5_acc:.2f} top1 acc: {best_top1_acc:.2f} at epoch {best_test_epoch}')
