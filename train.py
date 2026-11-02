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

from module.dataset import EEGPreImageDataset, GroupedImageBatchSampler, RandomGaussianNoise, RandomSmooth
from module.eeg_encoder.atm import ATMS
from module.eeg_encoder.model import (
    EEGNet,
    EEGProject,
    EEGConformer,
    TSConv,
    TSConv_parameterizable,
)
from module.loss import ContrastiveLoss
from module.util import retrieve_all
from module.projector import *
from module.training_plots import save_probe_plot, save_training_plot


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


def build_image_positive_mask(object_indices, image_indices):
    same_object = object_indices.unsqueeze(1) == object_indices.unsqueeze(0)
    same_image = image_indices.unsqueeze(1) == image_indices.unsqueeze(0)
    return same_object & same_image


def _compute_cross_modal_loss_impl(
    criterion,
    eeg_feature,
    image_feature,
    positive_mask,
    use_multi_positive,
):
    if use_multi_positive:
        loss_contrastive = criterion.multi_positive_pair_loss(
            eeg_feature,
            image_feature,
            positive_mask,
        )

        if criterion.alpha != 1.0:
            eeg_for_mse, image_for_mse = criterion._normalize_inputs(eeg_feature, image_feature)
            loss_mse = criterion.criterion_mse(eeg_for_mse, image_for_mse)
            return criterion.alpha * loss_contrastive + (1 - criterion.alpha) * loss_mse
        return loss_contrastive

    return criterion(eeg_feature, image_feature)


def compute_cross_modal_loss(
    criterion,
    eeg_feature,
    image_feature,
    positive_mask,
    use_multi_positive,
):
    """Run contrastive geometry in FP32 even when encoder AMP is active."""
    if not torch.is_autocast_enabled('cuda'):
        return _compute_cross_modal_loss_impl(
            criterion, eeg_feature, image_feature,
            positive_mask, use_multi_positive,
        )

    with torch.autocast(device_type='cuda', enabled=False):
        return _compute_cross_modal_loss_impl(
            criterion,
            eeg_feature.float(),
            image_feature.float(),
            positive_mask,
            use_multi_positive,
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


def build_eeg_encoder(args, feature_dim, eeg_sample_points, channels_num):
    if args.eeg_encoder_type == 'ATM':
        return ATMS(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            d_model=getattr(args, 'atm_d_model', 250),
            d_ff=getattr(args, 'atm_d_ff', 256),
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
            domain_batch_norm_boundary=getattr(args, 'tsconv_domain_bn_boundary', 0),
        )
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
            arch_kwargs=json.loads(getattr(args, 'fm_arch_kwargs', None) or '{}'),
        )
    from module.eeg_encoder.ortho_encoders import (
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
                domain_bn_boundary=getattr(args, 'tsconv_domain_bn_boundary', 0),
            )
        extra.update(json.loads(getattr(args, 'arch_kwargs', None) or '{}'))
        return build_architecture_encoder(
            args.eeg_encoder_type, feature_dim, eeg_sample_points, channels_num, **extra
        )
    raise ValueError(f"Unsupported EEG encoder type: {args.eeg_encoder_type}")


def build_projector(projector_type, input_dim, output_dim, activation='none'):
    activation = (activation or 'none').lower()
    if activation not in ('none', 'relu'):
        raise ValueError(f"Unsupported projector activation: {activation}")
    if projector_type == 'bottleneck':  # EEG side: encoder emits a low-rank code (--eeg_backbone_dim r), lifted
        # linearly to the target space; image/text side: identity, so outputs stay in the grader's DINOv2 space
        if input_dim == output_dim:
            return ProjectorDirect(activation=activation)
        return nn.Linear(input_dim, output_dim)
    if projector_type == 'direct':
        if input_dim != output_dim:
            raise ValueError(
                f"ProjectorDirect requires input_dim == output_dim, got {input_dim} != {output_dim}."
            )
        return ProjectorDirect(activation=activation)
    if projector_type == 'linear':
        return ProjectorLinear(input_dim, output_dim, activation=activation)
    raise ValueError(f"Unsupported projector type: {projector_type}")


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


def run_eeg_backbone(model, args, eeg_batch, subject_id_batch, return_intermediate=False):
    if args.eeg_encoder_type == 'ATM':
        return model(eeg_batch, subject_id_batch)
    elif args.eeg_encoder_type in ['TSConv', 'TSConv_parameterizable']:
        model_kwargs = {}
        if getattr(args, 'tsconv_domain_bn_boundary', 0) > 0:
            model_kwargs['subject_ids'] = subject_id_batch
        return model(eeg_batch, return_intermediate=return_intermediate, **model_kwargs)
    if getattr(args, 'tsconv_domain_bn_boundary', 0) > 0:  # registry Squeezeformer with per-dataset BatchNorm
        return model(eeg_batch, subject_ids=subject_id_batch)
    return model(eeg_batch)


def cross_subject_stimulus_mix(
    features, object_indices, image_indices, subject_ids, alpha=1.0, mixup_type='pairwise',
    return_mixup_metadata=False,
):
    if features.shape[0] < 2:
        if return_mixup_metadata:
            return features, {'partner_indices': None, 'mixed_mask': None}
        return features

    obj = object_indices.detach().cpu().tolist()
    img = image_indices.detach().cpu().tolist()
    sid = subject_ids.detach().cpu().tolist()

    groups = {}
    for i, (o, im) in enumerate(zip(obj, img)):
        groups.setdefault((int(o), int(im)), []).append(i)

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

        if mixup_type == 'group':
            concentration = torch.full((group_size,), dist_alpha, device=features.device, dtype=torch.float32)
            weights = torch.distributions.Dirichlet(concentration).sample((group_size,)).to(group_features.dtype)
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
            lam = beta.sample().to(group_features.dtype)
            lam = lam.view(*([group_size] + [1] * (group_features.dim() - 1)))
            mixed_group = lam * group_features + (1.0 - lam) * group_features[partner_pos]

        mixed[group_idx] = mixed_group

    if return_mixup_metadata:
        return mixed, {'partner_indices': partner_indices, 'mixed_mask': mixed_mask}
    return mixed


def cross_subject_operator_transfer(eeg_batch, subject_ids, shrink=0.05, eps=1e-8, boundary=0):
    """do(subject := t): re-render a trial through another subject's channel-mixing operator.

    Model the sensor signal as x_s = A_s . (sources), so the subject-specific nuisance is the
    channel mixing A_s, observable through the channel covariance R_s. Mapping
    x_s -> R_t^(1/2) R_s^(-1/2) x_s swaps the recording operator while leaving the stimulus-driven
    sources untouched, so the image label is preserved exactly -- unlike convex mixup, which also
    blends the signals themselves.

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

    sqrt_ops, inv_sqrt_ops = {}, {}
    for s in uniq.tolist():
        flat = eeg_batch[subject_ids == s].reshape(-1, channels, eeg_batch.shape[-1]).to(torch.float64)
        cov = torch.einsum('nct,ndt->cd', flat, flat) / (flat.shape[0] * flat.shape[-1])
        cov = (1.0 - shrink) * cov + shrink * eye * (torch.diagonal(cov).mean())
        evals, evecs = _spd(cov)
        sqrt_ops[s] = (evecs * evals.sqrt()) @ evecs.T
        inv_sqrt_ops[s] = (evecs * evals.rsqrt()) @ evecs.T

    out = eeg_batch.clone()
    for s, j in zip(uniq.tolist(), draw_op_partners(subject_ids, boundary)):
        if s == j:
            continue
        mask = subject_ids == s
        transfer = sqrt_ops[j] @ inv_sqrt_ops[s]
        out[mask] = torch.einsum('cd,...dt->...ct', transfer.to(eeg_batch.dtype), eeg_batch[mask]).to(out.dtype)  # autocast einsum returns bf16
    return out


def draw_op_partners(subject_ids, boundary=0):
    """One random partner per subject in the batch (same headset only when boundary > 0: THINGS-EEG2 ids >= boundary)."""
    uniq = torch.unique(subject_ids).tolist()
    if boundary:
        return [random.choice([j for j in uniq if (j >= boundary) == (s >= boundary)]) for s in uniq]
    return [random.choice(uniq) for _ in uniq]


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
    parser.add_argument('--lr_scheduler', default='none', choices=['none', 'cosine', 'onecycle'], help='optional warmup+cosine LR schedule (per epoch); AVDE full-FT uses cosine. onecycle steps PER OPTIMIZER STEP, matching the NeuralBench Track-1 recipe')
    parser.add_argument('--ema_decay', default=0.0, type=float, help='keep an EMA of the EEG encoder weights and evaluate/checkpoint it (0 disables; 0.999 ~ a 1000-step horizon)')
    parser.add_argument('--ema_warmup', action='store_true', help='ramp the EMA decay as min(ema_decay, (1+n)/(10+n)) so early epochs do not lag')
    parser.add_argument('--sealed_test_metric', action='store_true', help="report the per-epoch test top-1/top-5 under the NeurIPS-2026 Track-1 sealed rule (single-epoch predictions averaged within (subject, image) before retrieval; score_track1.py's subject_agg) instead of on repetition-averaged EEG, which moves to the avg_eeg_* columns")
    parser.add_argument('--sealed_test_fraction', default=0.125, type=float, help='fraction of test trials per subject scored by --sealed_test_metric (seeded: the same trials every epoch); 0.125 = ~10 of 80 repetitions per image')
    parser.add_argument('--sealed_test_every', default=2, type=int, help='score the sealed rule every N epochs (plus the last epoch and every new best); other epochs log NaN test top-1/top-5')
    parser.add_argument('--grad_clip', default=0.0, type=float, help='clip gradient norm to this value (0 disables); NeuralBench Track-1 uses 1.0')
    parser.add_argument('--stop_after_epoch', default=0, type=int, help='end training after this epoch while the lr schedule still spans --num_epochs (0 = off); with --select_best_on fixed this epoch is kept')
    parser.add_argument('--asymmetric_loss', action='store_true', help='contrastive loss in the EEG->image direction only, as NeuralBench ClipLoss(symmetric=False) does; default averages both directions')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='linear LR warmup epochs when --lr_scheduler cosine')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='cosine floor LR')
    parser.add_argument('--fm_arch_kwargs', default='', type=str, help='JSON constructor overrides for a --fm_no_pretrained LaBraM/REVE (num_layers, embed_dim, patch_size, ...)')
    parser.add_argument(
        '--fm_no_pretrained', action='store_true',
        help='build LaBraM/CBraMod from scratch instead of the public checkpoint (control arm)',
    )
    parser.add_argument('--num_workers', default=0, type=int, help='number of dataloader workers')
    parser.add_argument('--output_dir', default='./result', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--train_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--data_average', action='store_true')
    parser.add_argument('--init_temperature', default=0.07, type=float)
    parser.add_argument('--t_learnable', action='store_true')
    parser.add_argument('--softplus', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--img_l2norm', action='store_true')
    parser.add_argument('--eeg_l2norm', action='store_true')
    parser.add_argument('--multi_positive_loss', action='store_true', help='enable multi-positive EEG-image contrastive loss')
    parser.add_argument('--grouped_batch_sampler', action='store_true', help='sample multiple subjects per exact image in each batch')
    parser.add_argument('--samples_per_image', default=4, type=int, help='number of samples per exact image for grouped batches')
    parser.add_argument('--subject_mixup_within', action='store_true', help='rebuttal arm: Beta-mix 2 of a subject''s OWN reps for the same stimulus (within-subject; isolates the cross-subject factor). Needs un-averaged reps (no --data_average).')
    parser.add_argument('--cross_subject_average', action='store_true', help='train on synthetic per-image trials, each the mean of a random-size (Beta-drawn) random subset of cross-subject raw recordings')
    parser.add_argument('--subject_mixup_mode', type=str, choices=['none', 'raw_eeg', 'embedding'], default='none', help='cross-subject same-stimulus convex mixing mode')
    parser.add_argument('--mixup_type', type=str, choices=['pairwise', 'group'], default='pairwise', help='pairwise mixing or per-stimulus group mixing')
    parser.add_argument('--subject_mixup_alpha', default=1.0, type=float, help='beta(alpha, alpha) coefficient for cross-subject same-stimulus mixup')
    parser.add_argument('--subject_mixup_reg_lambda', default=0.0, type=float, help='weight for raw_eeg pairwise mixup embedding regularization')
    parser.add_argument('--op_transfer_prob', default=0.0, type=float, help='probability per batch of re-rendering raw EEG through another subject`s channel-mixing operator (counterfactual do(subject:=t))')
    parser.add_argument('--eval_mode', type=str, choices=['plain_cosine', 'saw', 'csls', 'saw_csls'], default='plain_cosine', help='test-time retrieval: cosine, SAW whitening, or SAW + fixed-k CSLS')
    parser.add_argument('--sattc_saw_shrink', default=0.2, type=float, help='covariance shrinkage for subject-adaptive whitening (SAW) during evaluation')
    parser.add_argument('--sattc_csls_k', default=2, type=int, help='fixed neighborhood size k for CSLS after SAW (eval_mode saw_csls)')
    parser.add_argument('--val_subject_id', default=None, type=int, help='subject ID used for validation model selection')
    parser.add_argument('--val_concept_ratio', default=0.0, type=float, help='hold out this fraction of TRAINING CONCEPTS (all their images, all training subjects) as a validation split for checkpoint selection; keeps all training subjects, unlike --val_subject_id')
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
    parser.add_argument('--subject_probe_holdout', action='store_true', help='per-subject held-out split; train linear subject probes (baseline only)')
    parser.add_argument('--subject_probe_holdout_ratio', type=float, default=0.10, help='fraction per train subject reserved for probe validation')
    parser.add_argument('--subject_probe_interval', type=int, default=1, help='train/eval subject probes every N epochs')
    parser.add_argument('--subject_probe_layers', nargs='+', default=['backbone', 'align'], help='layers to probe: temporal, spatial, backbone, align')
    parser.add_argument('--eeg_data_dir', default='./things_eeg/data/preprocessed_eeg', type=str, help='where your EEG data are')
    parser.add_argument("--selected_channels", default=[], nargs='*', type=str, help="selected EEG channels, empty means all channels")
    parser.add_argument('--time_window', type=int, default=[0, 250], nargs=2, help='time window for EEG data, in sample points')
    parser.add_argument('--eeg_aug', action='store_true')
    parser.add_argument('--eeg_aug_type', type=str, choices=['noise', 'smooth'], default='noise', help='eeg augmentation type')
    parser.add_argument(
        '--eeg_encoder_type',
        type=str,
        default='EEGProject',
    )
    parser.add_argument('--init_checkpoint', default='', type=str,
                        help='initialize the encoder and projectors from a compatible training checkpoint')
    parser.add_argument('--eval_unique_gallery', action='store_true',
                        help='rank val/test queries against each DISTINCT stimulus once instead of '
                             'one row per presentation; without it repeated presentations put '
                             'identical vectors in the gallery and cap top-5 at ~5/n_repeats')
    parser.add_argument('--tsconv_backbone_dim', default=0, type=int, help='width of the TSConv residual trunk before a separate trunk->feature_dim layer (0 = trunk at feature_dim, no extra layer); NeuralBench Track1TSConv uses 1024')
    parser.add_argument('--tsconv_temporal_filters', default=40, type=int, help='temporal filter count for TSConv_parameterizable')
    parser.add_argument('--tsconv_temporal_kernel', default=25, type=int, help='temporal kernel width for TSConv_parameterizable')
    parser.add_argument('--tsconv_pool_kernel', default=51, type=int, help='temporal average-pooling kernel for TSConv_parameterizable')
    parser.add_argument('--arch_kwargs', default='', type=str, help='JSON dict of constructor kwargs for registry encoders (ortho_encoders.py)')
    parser.add_argument('--tsconv_pool_stride', default=5, type=int, help='temporal average-pooling stride for TSConv_parameterizable')
    parser.add_argument('--tsconv_spatial_filters', default=40, type=int, help='spatial filter count for TSConv_parameterizable')
    parser.add_argument('--tsconv_projection_filters', default=40, type=int, help='1x1 projection filter count for TSConv_parameterizable')
    parser.add_argument('--tsconv_activation', default='elu', choices=['elu', 'gelu', 'relu', 'leaky_relu', 'silu'], help='activation for TSConv_parameterizable')
    parser.add_argument('--tsconv_dropout', default=0.5, type=float, help='backbone dropout for TSConv_parameterizable')
    parser.add_argument('--tsconv_head_dropout', default=-1.0, type=float, help='projection-head dropout for TSConv_parameterizable; negative reuses --tsconv_dropout')
    parser.add_argument('--tsconv_domain_bn_boundary', default=0, type=int, help='use separate primary/auxiliary TSConv BatchNorm paths split at this subject ID (0 disables)')
    parser.add_argument('--atm_d_model', default=250, type=int, help='ATM channel-token hidden dimension')
    parser.add_argument('--atm_d_ff', default=256, type=int, help='ATM feed-forward hidden dimension')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')
    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'bottleneck'], default='direct')
    parser.add_argument(
        '--projector_activation',
        type=str,
        choices=['none', 'relu'],
        default='none',
        help='terminal activation on alignment projectors (relu = Sparse CLIP-style sparse embeddings)',
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
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')
    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    args = parser.parse_args()
    if args.subject_mixup_reg_lambda < 0:
        raise ValueError("--subject_mixup_reg_lambda must be non-negative.")
    if args.tsconv_domain_bn_boundary < 0:
        raise ValueError('--tsconv_domain_bn_boundary must be non-negative.')
    if args.tsconv_domain_bn_boundary > 0:
        if args.eeg_encoder_type.removeprefix('Ortho') not in ('TSConv_parameterizable', 'FastTSConvSqueezeformer', 'ATM'):
            raise ValueError('--tsconv_domain_bn_boundary requires TSConv_parameterizable, FastTSConvSqueezeformer or ATM.')
    if args.subject_mixup_reg_lambda > 0 and (args.subject_mixup_mode != 'raw_eeg' or args.mixup_type != 'pairwise'):
        raise ValueError("--subject_mixup_reg_lambda currently requires --subject_mixup_mode raw_eeg and --mixup_type pairwise.")
    if args.subject_probe_holdout:
        if not (0.0 < args.subject_probe_holdout_ratio < 1.0):
            raise ValueError("--subject_probe_holdout_ratio must be strictly between 0 and 1.")
    if args.subject_mixup_within:
        if args.data_average or args.cross_subject_average or args.subject_mixup_mode != 'none':
            raise ValueError("--subject_mixup_within needs un-averaged reps and replaces cross-subject mixup: drop --data_average, --cross_subject_average, and set --subject_mixup_mode none.")
    if args.cross_subject_average:
        conflicts = [name for flag, name in [
            (args.data_average, '--data_average'),
            (args.subject_mixup_mode != 'none', '--subject_mixup_mode'),
            (args.grouped_batch_sampler, '--grouped_batch_sampler'),
            (args.multi_positive_loss, '--multi_positive_loss'),
        ] if flag]
        if conflicts:
            raise ValueError(f"--cross_subject_average is incompatible with: {', '.join(conflicts)}")
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
        args.train_subject_ids = [sid for sid in args.train_subject_ids if sid != args.val_subject_id]
        log(f"Removed val_subject_id={args.val_subject_id} from train_subject_ids for clean validation.")
        if len(args.train_subject_ids) == 0:
            raise ValueError("After removing val_subject_id, train_subject_ids is empty.")
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
    log(f'Subject mixup mode: {args.subject_mixup_mode}, type: {args.mixup_type} (alpha={args.subject_mixup_alpha})')
    log(f'Evaluation mode: {args.eval_mode}')

    print('\n>>> Loading Train Data <<<')
    if args.eeg_aug:
        if args.eeg_aug_type == 'noise':
            eeg_transform = RandomGaussianNoise(std=0.001)
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
        args.data_average,
        False,
        eeg_transform,
        True,
        False,
        args.frozen_eeg_prior,
        cross_subject_average=args.cross_subject_average,
        subject_mixup_within=args.subject_mixup_within,
    )

    eeg_sample_points = train_dataset.num_sample_points
    image_feature_dim = train_dataset.image_features.shape[-1]
    backbone_feature_dim = args.eeg_backbone_dim if args.eeg_backbone_dim > 0 else image_feature_dim
    channels_num = train_dataset.channels_num
    log(f'EEG sample points: {eeg_sample_points}')
    log(f'image feature dimension: {image_feature_dim}')
    log(f'EEG backbone output dimension: {backbone_feature_dim}')
    log(f'alignment feature dimension (--feature_dim): {args.feature_dim}')
    log(f'data length: {len(train_dataset)}')
    log(f'number of channels: {channels_num}')

    pin_memory = device.type == "cuda"
    train_main_dataset = train_dataset
    concept_val_dataloader = None
    if args.val_concept_ratio > 0.0:
        # Honest checkpoint selection without giving up a training subject: hold out a
        # fixed slice of training CONCEPTS (every image of them, across every training
        # subject).  Concept-disjoint mirrors the real test condition of 200 unseen
        # concepts; an image-level split would leave concept-mates in training, which
        # --multi_positive_loss is explicitly trained to pull together.
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
            np.random.default_rng(20260822).permutation(concepts)[:holdout_count].tolist()
        )
        concept_val_indices, main_train_indices = [], []
        for key, indices in groups.items():
            (concept_val_indices if key[0] in held else main_train_indices).extend(indices)
        train_main_dataset = (
            _GroupedSubset(train_dataset, main_train_indices)
            if args.grouped_batch_sampler
            else Subset(train_dataset, main_train_indices)
        )
        # Val scores clean EEG, as the test set does: a shallow copy shares every array
        # but drops the transform (--eeg_aug is for training rows only).
        val_view = copy.copy(train_dataset)
        val_view.eeg_transform = None
        concept_val_dataloader = DataLoader(
            Subset(val_view, concept_val_indices),
            batch_size=200, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory,
        )
        log(
            f"Concept val split: held_out_concepts={holdout_count}/{len(concepts)} "
            f"(seed=20260822), train_items={len(main_train_indices)}, "
            f"val_items={len(concept_val_indices)}"
        )
    subj_probe_train_dataloader = None
    subj_probe_val_dataloader = None
    if args.subject_probe_holdout:
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

    if args.grouped_batch_sampler:
        batch_sampler = GroupedImageBatchSampler(
            train_main_dataset,
            batch_size=args.batch_size,
            samples_per_image=args.samples_per_image,
            drop_last=True,
            seed=seed,
        )
        dataloader = DataLoader(train_main_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=pin_memory)
        effective_batch_size = batch_sampler.images_per_batch * batch_sampler.samples_per_image
        log(
            f'Grouped batch sampler enabled: images_per_batch={batch_sampler.images_per_batch}, '
            f'samples_per_image={batch_sampler.samples_per_image}, effective_batch_size={effective_batch_size}'
        )
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
    if args.early_stop_patience > 0 and args.select_best_on != 'val':
        raise ValueError("--early_stop_patience requires --select_best_on val.")
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
            True,
            False,
            eeg_transform,
            False,
            False,
            args.frozen_eeg_prior,
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
        True,
        False,
        eeg_transform,
        False,
        False,
        args.frozen_eeg_prior,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    inference_keys = [
        'eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir',
        'projector', 'projector_activation', 'feature_dim', 'eeg_backbone_dim', 'time_window', 'selected_channels',
        'tsconv_temporal_filters', 'tsconv_temporal_kernel', 'tsconv_pool_kernel', 'tsconv_pool_stride',
        'tsconv_spatial_filters', 'tsconv_projection_filters', 'tsconv_activation', 'tsconv_dropout',
        'tsconv_head_dropout',
        'arch_kwargs', 'fm_arch_kwargs',
        'tsconv_domain_bn_boundary',
        'eval_mode', 'sattc_saw_shrink', 'sattc_csls_k',
    ]
    inference_config = {k: args_dict[k] for k in inference_keys}
    inference_config['feature_dim'] = int(str(args.feature_dim).split(',')[0])
    inference_config['architecture'] = 'baseline'
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['backbone_feature_dim'] = backbone_feature_dim
    inference_config['image_feature_dim'] = image_feature_dim
    with open(os.path.join(writer.log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)

    model = build_eeg_encoder(args, backbone_feature_dim, eeg_sample_points, channels_num).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'EEG Encoder trainable parameters: {num_params / 1e6:.2f}M')
    log(str(model))

    projector_activation = args.projector_activation
    args.feature_dim = int(str(args.feature_dim).split(',')[0])
    eeg_projector = build_projector(
        args.projector, backbone_feature_dim, args.feature_dim,
        activation=projector_activation,
    ).to(device)
    img_projector = build_projector(
        args.projector, image_feature_dim, args.feature_dim,
        activation=projector_activation,
    ).to(device)
    text_projector = build_projector(
        args.projector, image_feature_dim, args.feature_dim,
        activation=projector_activation,
    ).to(device)

    projector_params = (
        sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in img_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    )
    log(f'Projector trainable parameters: {projector_params / 1e6:.2f}M')

    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial['model_state_dict'])
        eeg_projector.load_state_dict(initial['eeg_projector_state_dict'])
        img_projector.load_state_dict(initial['img_projector_state_dict'])
        text_projector.load_state_dict(initial['text_projector_state_dict'])
        log(f'Initialized all modules from {args.init_checkpoint}')
        del initial

    align_dim = args.feature_dim if args.projector != 'direct' else backbone_feature_dim
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
        args.eeg_l2norm,
        args.img_l2norm,
        args.t_learnable,
        args.softplus,
        logit_scale_max=args.logit_scale_max,
        asymmetric=args.asymmetric_loss,
    ).to(device)
    if args.init_checkpoint:  # the learned temperature belongs to the head
        saved = torch.load(args.init_checkpoint, map_location=device, weights_only=False).get('criterion_state_dict')
        if saved is not None:
            criterion.load_state_dict(saved)
            log(f'Initialized the loss (logit scale) from {args.init_checkpoint}')
    log(str(criterion))

    sattc_params = {
        'saw_shrink': args.sattc_saw_shrink,
        'csls_k': args.sattc_csls_k,
    }

    total_epochs = args.num_epochs

    module_registry = {
        'model': model,
        'eeg_projector': eeg_projector,
        'img_projector': img_projector,
        'text_projector': text_projector,
    }
    trainable_parameters = (
        collect_trainable_parameters(list(module_registry.values()))
        + ([p for p in criterion.parameters() if p.requires_grad] if args.t_learnable else [])
    )
    optimizer = optim.AdamW(
        [{'params': trainable_parameters}],
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
            pct_start=0.1,
            anneal_strategy='cos',
        )

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
        if args.t_learnable:
            checkpoint['criterion_state_dict'] = criterion.state_dict()
        return checkpoint

    def evaluate_split(split_loader, split_name):
        if split_loader is None:
            return float('nan'), float('nan'), float('nan'), {}

        total_loss_local = 0.0
        eeg_feature_list = []
        image_feature_list = []
        subjects = []
        object_indices = []
        image_indices = []

        with torch.no_grad():
            for batch in split_loader:
                eeg_batch = batch[0].to(device)
                image_feature_batch = batch[1].to(device)
                subject_id_batch = batch[3].to(device)
                object_idx_batch = batch[4].to(device)
                image_idx_batch = batch[5].to(device)

                eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
                eeg_feature_batch = eeg_projector(eeg_backbone_batch)
                image_feature_proj = img_projector(image_feature_batch)

                positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
                loss = compute_cross_modal_loss(
                    criterion,
                    eeg_feature_batch,
                    image_feature_proj,
                    positive_mask,
                    args.multi_positive_loss
                )
                total_loss_local += loss.item()
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
        sparsity = {}
        if args.projector_activation == 'relu':
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
        if split_loader is None or args.projector_activation != 'relu':
            return {}

        eeg_acc = _empty_alignment_sparsity_accumulator()
        img_acc = _empty_alignment_sparsity_accumulator()
        with torch.no_grad():
            for batch in split_loader:
                eeg_batch = batch[0].to(device)
                image_feature_batch = batch[1].to(device)
                subject_id_batch = batch[3].to(device)

                eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
                _update_alignment_sparsity_accumulator(eeg_acc, eeg_projector(eeg_backbone_batch).cpu().numpy())
                _update_alignment_sparsity_accumulator(
                    img_acc,
                    img_projector(image_feature_batch).cpu().numpy(),
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
    sealed_best = {'top5': float('-inf')}

    def sealed_test_scores():
        """(top1, top5) in % under the Track-1 sealed rule, on the current (EMA) weights."""
        from neurips_challenge.score_track1 import score_modules
        start = time.time()
        r = score_modules(model, eeg_projector, img_projector, args, args.test_subject_ids, device, 512, 0,
                          fraction=args.sealed_test_fraction)
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

    epochs_without_selection_improvement = 0
    early_stop_reference_loss = best_test_loss
    for epoch in range(1, total_epochs + 1):
        # Autocast is switched on globally (set_autocast_enabled), so no autocast region ever closes to flush its
        # weight-cast cache: the epoch-end evaluation's no-grad casts survived into the next epoch's first forward,
        # which then had no grad_fn when only the head trained (frozen-backbone probe, 2026-10-09).
        torch.clear_autocast_cache()
        if ema_model is not None and epoch > 1:
            model.load_state_dict(raw_state)
        model.train()
        eeg_projector.train()
        img_projector.train()
        text_projector.train()
        criterion.train()
        total_loss = 0.0
        total_cl_loss = 0.0
        total_subject_mixup_reg_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}"):
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            optimizer.zero_grad()
            if args.op_transfer_prob > 0 and random.random() < args.op_transfer_prob:
                eeg_batch = cross_subject_operator_transfer(
                    eeg_batch, subject_id_batch,
                    boundary=21 if any(s >= 21 for s in args.train_subject_ids) and min(args.train_subject_ids) < 21 else 0,
                )
            original_eeg_batch = eeg_batch
            mixup_metadata = None
            if args.subject_mixup_mode == 'raw_eeg':
                mixup_out = cross_subject_stimulus_mix(
                    eeg_batch,
                    object_idx_batch,
                    image_idx_batch,
                    subject_id_batch,
                    alpha=args.subject_mixup_alpha,
                    mixup_type=args.mixup_type,
                    return_mixup_metadata=args.subject_mixup_reg_lambda > 0,
                )
                if args.subject_mixup_reg_lambda > 0:
                    eeg_batch, mixup_metadata = mixup_out
                else:
                    eeg_batch = mixup_out
            eeg_backbone_batch = run_eeg_backbone(
                model, args, eeg_batch, subject_id_batch
            )
            if args.subject_mixup_mode == 'embedding':
                eeg_backbone_batch = cross_subject_stimulus_mix(
                    eeg_backbone_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                    alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type,
                )
            eeg_feature_batch = eeg_projector(eeg_backbone_batch)

            loss = eeg_batch.new_tensor(0.0)

            if args.subject_mixup_reg_lambda > 0:
                original_eeg_feature_batch = run_eeg_backbone(model, args, original_eeg_batch, subject_id_batch)
                original_eeg_feature_batch = eeg_projector(original_eeg_feature_batch)
                subject_mixup_reg_loss = compute_subject_mixup_regularization(
                    eeg_feature_batch,
                    original_eeg_feature_batch,
                    mixup_metadata['partner_indices'],
                    mixup_metadata['mixed_mask'],
                )
                loss = loss + args.subject_mixup_reg_lambda * subject_mixup_reg_loss
                total_subject_mixup_reg_loss += subject_mixup_reg_loss.item()
            image_feature_proj = img_projector(image_feature_batch)
            positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
            cl_loss = compute_cross_modal_loss(
                criterion,
                eeg_feature_batch,
                image_feature_proj,
                positive_mask,
                args.multi_positive_loss,
            )
            loss = loss + cl_loss
            total_cl_loss += cl_loss.item()

            grad_scaler.scale(loss).backward()
            if args.grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if ema_model is not None:
                ema_model.update_parameters(model)
            if args.lr_scheduler == 'onecycle':
                scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/train_cl', total_cl_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_subject_mixup_reg', total_subject_mixup_reg_loss / len(dataloader), epoch)
        log(
            f"Epoch [{epoch}/{total_epochs}] TrainLoss={avg_loss:.4f} "
            f"CL={total_cl_loss / len(dataloader):.4f} "
            f"MIXREG={total_subject_mixup_reg_loss / len(dataloader):.4f}"
        )

        if ema_model is not None:
            # Everything below -- test/val metrics, selection, checkpoints -- sees the EMA weights; the raw weights are restored before training.
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
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Acc/top1_test', top1_acc, epoch)
        writer.add_scalar('Acc/top5_test', top5_acc, epoch)
        if test_sparsity:
            writer.add_scalar('Sparsity/eeg_l0_frac_test', test_sparsity['eeg_l0_frac'], epoch)
            writer.add_scalar('Sparsity/img_l0_frac_test', test_sparsity['img_l0_frac'], epoch)

        if epoch == total_epochs and args.projector_activation == 'relu':
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
        if args.save_testsel_checkpoint and avg_test_loss < best_testsel_loss:
            # Matched control: what test-loss selection would have kept on this same
            # trajectory. Never influences training or the selected checkpoint.
            best_testsel_loss = avg_test_loss
            best_testsel_epoch = epoch
            torch.save(build_checkpoint(epoch, avg_test_loss), f"{writer.log_dir}/checkpoint_testsel_best.pth")
        if args.early_stop_patience > 0:
            if selected_loss < early_stop_reference_loss:
                early_stop_reference_loss = selected_loss
                epochs_without_selection_improvement = 0
            else:
                epochs_without_selection_improvement += 1

        if scheduler is not None and args.lr_scheduler != 'onecycle':
            scheduler.step()  # onecycle already stepped per optimizer step

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

    result_dict = {
        'architecture': 'baseline',
        'eval_mode': args.eval_mode,
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
    if args.projector_activation == 'relu' and best_test_sparsity:
        result_dict.update({
            'eeg_l0_mean': f"{best_test_sparsity['eeg_l0_mean']:.3f}",
            'eeg_l0_frac': f"{best_test_sparsity['eeg_l0_frac']:.3f}",
            'eeg_active_feat_frac': f"{best_test_sparsity['eeg_active_feat_frac']:.3f}",
            'img_l0_mean': f"{best_test_sparsity['img_l0_mean']:.3f}",
            'img_l0_frac': f"{best_test_sparsity['img_l0_frac']:.3f}",
            'img_active_feat_frac': f"{best_test_sparsity['img_active_feat_frac']:.3f}",
        })
    if args.projector_activation == 'relu' and last_epoch_train_sparsity:
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
