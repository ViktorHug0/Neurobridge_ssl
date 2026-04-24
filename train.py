import os
import argparse
import logging
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
from module.eeg_encoder.model import EEGNet, EEGProject, TSConv, EEGTransformer, TSConv30 
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
from module.projector import *
from module.sampler import GroupedImageBatchSampler
from module.training_plots import save_probe_plot, save_training_plot
from module.eeg_augmentation import RandomTimeShift, RandomGaussianNoise, RandomChannelDropout, RandomSmooth


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


def resolve_abstraction_image_feature_dirs(image_feature_dir: str) -> list[str]:
    base_name = os.path.basename(os.path.normpath(image_feature_dir))
    parent_dir = os.path.dirname(os.path.normpath(image_feature_dir))
    return [
        os.path.join(parent_dir, base_name.replace("layer28", "layer25", 1)),
        os.path.join(parent_dir, base_name.replace("layer28", "layer31", 1)),
    ]


def build_image_positive_mask(object_indices, image_indices):
    same_object = object_indices.unsqueeze(1) == object_indices.unsqueeze(0)
    same_image = image_indices.unsqueeze(1) == image_indices.unsqueeze(0)
    return same_object & same_image


def compute_cross_modal_loss(criterion, eeg_feature, image_feature, text_feature, positive_mask, use_multi_positive):
    if use_multi_positive:
        loss_contrastive_ie = criterion.multi_positive_pair_loss(eeg_feature, image_feature, positive_mask)
        if criterion.beta != 1.0:
            loss_contrastive_te = criterion.multi_positive_pair_loss(
                eeg_feature, text_feature, positive_mask, key_is_text=True
            )
            loss_contrastive = criterion.beta * loss_contrastive_ie + (1 - criterion.beta) * loss_contrastive_te
        else:
            loss_contrastive = loss_contrastive_ie

        if criterion.alpha != 1.0:
            eeg_for_mse, image_for_mse, _ = criterion._normalize_inputs(eeg_feature, image_feature)
            loss_mse = criterion.criterion_mse(eeg_for_mse, image_for_mse)
            return criterion.alpha * loss_contrastive + (1 - criterion.alpha) * loss_mse
        return loss_contrastive

    return criterion(eeg_feature, image_feature, text_feature)


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
        return ATMS(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGNet':
        return EEGNet(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGProject':
        return EEGProject(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'TSConv':
        return TSConv(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'TSConv30':
        return TSConv30(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGTransformer':
        return EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    raise ValueError(f"Unsupported EEG encoder type: {args.eeg_encoder_type}")


def build_projector(projector_type, input_dim, output_dim):
    if projector_type == 'direct':
        if input_dim != output_dim:
            raise ValueError(
                f"ProjectorDirect requires input_dim == output_dim, got {input_dim} != {output_dim}."
            )
        return ProjectorDirect()
    if projector_type == 'linear':
        return ProjectorLinear(input_dim, output_dim)
    if projector_type == 'mlp':
        return ProjectorMLP(input_dim, output_dim)
    raise ValueError(f"Unsupported projector type: {projector_type}")


def run_eeg_backbone(model, args, eeg_batch, subject_id_batch):
    if args.eeg_encoder_type == 'ATM':
        return model(eeg_batch, subject_id_batch)
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
    features, object_indices, image_indices, subject_ids, alpha=1.0, mixup_type='pairwise', return_mixup_metadata=False
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
            lam = torch.distributions.Beta(concentration, concentration).sample().to(group_features.dtype)
            lam_shape = [group_size] + [1] * (group_features.dim() - 1)
            lam = lam.view(*lam_shape)
            mixed_group = lam * group_features + (1.0 - lam) * group_features[partner_pos]

        mixed[group_idx] = mixed_group

    if return_mixup_metadata:
        return mixed, {'partner_indices': partner_indices, 'mixed_mask': mixed_mask}
    return mixed


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
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int, help='number of dataloader workers')
    parser.add_argument('--output_dir', default='./result', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--train_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--data_average', action='store_true')
    parser.add_argument('--data_random', action='store_true')
    parser.add_argument('--init_temperature', default=0.07, type=float)
    parser.add_argument('--t_learnable', action='store_true')
    parser.add_argument('--softplus', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--img_l2norm', action='store_true')
    parser.add_argument('--text_l2norm', action='store_true')
    parser.add_argument('--eeg_l2norm', action='store_true')
    parser.add_argument('--multi_positive_loss', action='store_true', help='enable multi-positive EEG-image contrastive loss')
    parser.add_argument('--grouped_batch_sampler', action='store_true', help='sample multiple subjects per exact image in each batch')
    parser.add_argument('--samples_per_image', default=4, type=int, help='number of samples per exact image for grouped batches')
    parser.add_argument('--subject_mixup_mode', type=str, choices=['none', 'raw_eeg', 'embedding'], default='none', help='cross-subject same-stimulus convex mixing mode')
    parser.add_argument('--mixup_type', type=str, choices=['pairwise', 'group'], default='pairwise', help='pairwise mixing or full same-stimulus group mixing')
    parser.add_argument('--subject_mixup_alpha', default=1.0, type=float, help='beta(alpha, alpha) coefficient for cross-subject same-stimulus mixup')
    parser.add_argument('--subject_mixup_reg_lambda', default=0.0, type=float, help='weight for raw_eeg pairwise mixup embedding regularization')
    parser.add_argument('--single_emb_stop_grad', action='store_true', help='detach original single-subject embeddings in mixup regularization')
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
    parser.add_argument('--select_best_on', type=str, choices=['test', 'val'], default='test', help='which split selects the best checkpoint')
    parser.add_argument('--subject_probe_holdout', action='store_true', help='per-subject held-out split; train linear subject probes (baseline only)')
    parser.add_argument('--subject_probe_holdout_ratio', type=float, default=0.10, help='fraction per train subject reserved for probe validation')
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
    parser.add_argument('--eeg_encoder_type', type=str, choices=['ATM', "EEGNet", "EEGProject", "TSConv", "EEGTransformer", "TSConv30"], default='EEGProject')
    parser.add_argument('--image_aug', action='store_true')
    parser.add_argument('--image_test_aug', action='store_true')
    parser.add_argument('--eeg_test_aug', action='store_true')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')
    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'mlp'], default='direct')
    parser.add_argument('--feature_dim', type=int, default=512, help='shared alignment-space dimension when projector is not direct')
    parser.add_argument('--eeg_backbone_dim', type=int, default=0, help='EEG encoder output dimension (0 means use image feature dimension)')
    parser.add_argument('--image_feature_dir', default='/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit', type=str, help='where your image feature are')
    parser.add_argument('--aug_image_feature_dirs', default=[], nargs='+', type=str, help='where your augmentation image feature are')
    parser.add_argument('--sample_abstraction_levels', action='store_true', help='uniformly sample InternViT layer27/28/29 image embeddings per image during training')
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')
    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    args = parser.parse_args()
    if args.subject_mixup_reg_lambda < 0:
        raise ValueError("--subject_mixup_reg_lambda must be non-negative.")
    if args.subject_mixup_reg_lambda > 0 and (args.subject_mixup_mode != 'raw_eeg' or args.mixup_type != 'pairwise'):
        raise ValueError("--subject_mixup_reg_lambda currently requires --subject_mixup_mode raw_eeg and --mixup_type pairwise.")
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
    if args.sample_abstraction_levels and args.image_aug:
        raise ValueError("--sample_abstraction_levels cannot be combined with --image_aug.")

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
    log(f'Subject mixup mode: {args.subject_mixup_mode}, type: {args.mixup_type} (alpha={args.subject_mixup_alpha})')
    log(f'Evaluation mode: {args.eval_mode}')

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

    abstraction_image_feature_dirs = []
    if args.sample_abstraction_levels:
        abstraction_image_feature_dirs = resolve_abstraction_image_feature_dirs(args.image_feature_dir)

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
        abstraction_image_feature_dirs,
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

    if args.select_best_on == 'val' and args.val_subject_id is None:
        raise ValueError("--select_best_on val requires --val_subject_id.")
    if args.val_subject_id is not None:
        if args.val_subject_id in args.test_subject_ids:
            raise ValueError("val_subject_id must be different from test_subject_ids.")

    val_dataloader = None
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
    )
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    inference_keys = [
        'eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir',
        'projector', 'feature_dim', 'eeg_backbone_dim', 'time_window', 'selected_channels',
        'eval_mode', 'sattc_saw_shrink', 'sattc_saw_diag', 'sattc_csls_k', 'sattc_cw', 'sattc_cw_shrink', 'sattc_cw_diag',
    ]
    inference_config = {k: args_dict[k] for k in inference_keys}
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

    eeg_projector = build_projector(args.projector, backbone_feature_dim, args.feature_dim).to(device)
    img_projector = build_projector(args.projector, image_feature_dim, args.feature_dim).to(device)
    text_projector = build_projector(args.projector, image_feature_dim, args.feature_dim).to(device)
    projector_params = (
        sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in img_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    )
    log(f'Projector trainable parameters: {projector_params / 1e6:.2f}M')

    align_dim = args.feature_dim if args.projector != 'direct' else backbone_feature_dim
    probe_class_map = {sid: idx for idx, sid in enumerate(sorted(args.train_subject_ids))}
    num_probe_classes = len(probe_class_map)
    subj_probe_heads = None
    subj_probe_optimizers = None
    if args.subject_probe_holdout:
        subj_probe_heads = nn.ModuleDict({
            'eeg_backbone': nn.Linear(backbone_feature_dim, num_probe_classes).to(device),
            'eeg_align': nn.Linear(align_dim, num_probe_classes).to(device),
        })
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
    ).to(device)
    log(str(criterion))

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
    optimizer = optim.AdamW(
        collect_trainable_parameters(list(module_registry.values()))
        + ([p for p in criterion.parameters() if p.requires_grad] if args.t_learnable else []),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    def forward_architecture(eeg_backbone_batch):
        return {'eeg_feature': eeg_projector(eeg_backbone_batch)}

    def build_checkpoint(epoch_idx, loss_value, optimizer_obj):
        checkpoint = {
            'epoch': epoch_idx,
            'stage': 'joint',
            'architecture': 'baseline',
            'model_state_dict': model.state_dict(),
            'eeg_projector_state_dict': eeg_projector.state_dict(),
            'img_projector_state_dict': img_projector.state_dict(),
            'text_projector_state_dict': text_projector.state_dict(),
            'optimizer_state_dict': optimizer_obj.state_dict(),
            'loss': loss_value,
        }
        if args.t_learnable:
            checkpoint['criterion_state_dict'] = criterion.state_dict()
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

    def evaluate_split(split_loader, split_name):
        if split_loader is None:
            return float('nan'), float('nan'), float('nan')

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
                text_feature_batch = batch[2].to(device)
                subject_id_batch = batch[3].to(device)
                object_idx_batch = batch[4].to(device)
                image_idx_batch = batch[5].to(device)

                eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
                arch_out = forward_architecture(eeg_backbone_batch)
                eeg_feature_batch = arch_out['eeg_feature']
                image_feature_proj = img_projector(image_feature_batch)
                text_feature_proj = text_projector(text_feature_batch)
                eeg_feature_for_loss, image_feature_for_loss = maybe_apply_train_saw(
                    eeg_feature_batch,
                    image_feature_proj,
                    subject_id_batch,
                )

                positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
                loss = compute_cross_modal_loss(
                    criterion,
                    eeg_feature_for_loss,
                    image_feature_for_loss,
                    text_feature_proj,
                    positive_mask,
                    args.multi_positive_loss
                )
                total_loss_local += loss.item()
                eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
                image_feature_list.append(image_feature_proj.cpu().numpy())
                subjects.append(subject_id_batch.cpu().numpy())
                object_indices.append(object_idx_batch.cpu().numpy())
                image_indices.append(image_idx_batch.cpu().numpy())

        avg_loss_local = total_loss_local / len(split_loader)
        eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
        image_feature_all = np.concatenate(image_feature_list, axis=0)
        subject_all = np.concatenate(subjects, axis=0)
        object_all = np.concatenate(object_indices, axis=0)
        image_all = np.concatenate(image_indices, axis=0)
        top5_count, top1_count, total = retrieve_all(
            eeg_feature_all,
            image_feature_all,
            subject_ids=subject_all,
            eval_mode=args.eval_mode,
            sattc_params=sattc_params,
        )
        top5_acc_local = top5_count / total * 100
        top1_acc_local = top1_count / total * 100
        log(
            f"{split_name}: mode={args.eval_mode} top5 acc {top5_acc_local:.2f}%\ttop1 acc {top1_acc_local:.2f}%\tLoss: {avg_loss_local:.4f}"
        )
        return avg_loss_local, top1_acc_local, top5_acc_local

    def encode_probe_labels(subject_ids: torch.Tensor) -> torch.Tensor:
        ids = [int(x) for x in subject_ids.detach().cpu().tolist()]
        return torch.tensor([probe_class_map[s] for s in ids], dtype=torch.long, device=subject_ids.device)

    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_test_loss = float('inf')
    best_test_epoch = 0
    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'top1_acc': []}
    probe_history = (
        {'epoch': [], 'eeg_backbone_val_acc': [], 'eeg_align_val_acc': []}
        if args.subject_probe_holdout
        else None
    )
    top1_acc = 0.0
    top5_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        eeg_projector.train()
        img_projector.train()
        text_projector.train()
        criterion.train()
        total_loss = 0.0
        total_cl_loss = 0.0
        total_subject_mixup_reg_loss = 0.0
        total_subject_adapt_loss = 0.0
        total_subject_adapt_subjects = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}"):
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            optimizer.zero_grad()
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
            eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
            if args.subject_mixup_mode == 'embedding':
                eeg_backbone_batch = cross_subject_stimulus_mix(
                    eeg_backbone_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                    alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type
                )
            arch_out = forward_architecture(eeg_backbone_batch)

            loss = eeg_batch.new_tensor(0.0)

            eeg_feature_batch = arch_out['eeg_feature']
            if args.subject_mixup_reg_lambda > 0:
                if args.single_emb_stop_grad:
                    with torch.no_grad():
                        original_eeg_feature_batch = forward_architecture(
                            run_eeg_backbone(model, args, original_eeg_batch, subject_id_batch)
                        )['eeg_feature']
                else:
                    original_eeg_feature_batch = forward_architecture(
                        run_eeg_backbone(model, args, original_eeg_batch, subject_id_batch)
                    )['eeg_feature']
                subject_mixup_reg_loss = compute_subject_mixup_regularization(
                    eeg_feature_batch,
                    original_eeg_feature_batch,
                    mixup_metadata['partner_indices'],
                    mixup_metadata['mixed_mask'],
                )
                loss = loss + args.subject_mixup_reg_lambda * subject_mixup_reg_loss
                total_subject_mixup_reg_loss += subject_mixup_reg_loss.item()
            image_feature_proj = img_projector(image_feature_batch)
            text_feature_proj = text_projector(text_feature_batch)
            eeg_feature_for_loss, image_feature_for_loss = maybe_apply_train_saw(
                eeg_feature_batch,
                image_feature_proj,
                subject_id_batch,
            )
            positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
            cl_loss = compute_cross_modal_loss(
                criterion,
                eeg_feature_for_loss,
                image_feature_for_loss,
                text_feature_proj,
                positive_mask,
                args.multi_positive_loss
            )
            loss = loss + cl_loss
            total_cl_loss += cl_loss.item()

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

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/train_cl', total_cl_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_subject_mixup_reg', total_subject_mixup_reg_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_subject_adapt', total_subject_adapt_loss / len(dataloader), epoch)
        log(
            f"Epoch [{epoch}/{total_epochs}] TrainLoss={avg_loss:.4f} "
            f"CL={total_cl_loss / len(dataloader):.4f} "
            f"MIXREG={total_subject_mixup_reg_loss / len(dataloader):.4f} "
            f"SADAPT={total_subject_adapt_loss / len(dataloader):.4f} "
            f"SADAPT_SUBJ={total_subject_adapt_subjects / len(dataloader):.2f}"
        )

        if args.save_weights:
            torch.save(build_checkpoint(epoch, avg_loss, optimizer), f"{writer.log_dir}/checkpoint_last.pth")

        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()

        avg_test_loss, top1_acc, top5_acc = evaluate_split(test_dataloader, "test")
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Acc/top1_test', top1_acc, epoch)
        writer.add_scalar('Acc/top5_test', top5_acc, epoch)

        val_loss, val_top1, val_top5 = float('nan'), float('nan'), float('nan')
        if val_dataloader is not None:
            val_loss, val_top1, val_top5 = evaluate_split(val_dataloader, "val")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/top1_val', val_top1, epoch)
            writer.add_scalar('Acc/top5_val', val_top5, epoch)

        bb_probe_acc_pct = float('nan')
        clip_probe_acc_pct = float('nan')
        if (
            args.subject_probe_holdout
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
                    bb = run_eeg_backbone(model, args, eeg_p, sid_p)
                    al = eeg_projector(bb)
                y = encode_probe_labels(sid_p)
                for name, head in subj_probe_heads.items():
                    subj_probe_optimizers[name].zero_grad()
                    feats = bb if name == 'eeg_backbone' else al
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
                    bb = run_eeg_backbone(model, args, eeg_p, sid_p)
                    al = eeg_projector(bb)
                    y = encode_probe_labels(sid_p)
                    for name, head in subj_probe_heads.items():
                        feats = bb if name == 'eeg_backbone' else al
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

        if probe_history is not None:
            probe_history['epoch'].append(epoch)
            probe_history['eeg_backbone_val_acc'].append(bb_probe_acc_pct)
            probe_history['eeg_align_val_acc'].append(clip_probe_acc_pct)

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(avg_test_loss)
        history['top1_acc'].append(top1_acc)

        selected_loss = avg_test_loss
        selected_top1 = top1_acc
        if args.select_best_on == 'val' and not np.isnan(val_loss):
            selected_loss = val_loss
            selected_top1 = val_top1

        is_better = False
        if selected_loss < best_test_loss:
            is_better = True
        elif selected_loss == best_test_loss and selected_top1 > best_top1_acc:
            is_better = True

        if is_better:
            best_test_loss = selected_loss
            best_top5_acc = top5_acc
            best_top1_acc = top1_acc
            best_test_epoch = epoch
            if args.save_weights:
                torch.save(build_checkpoint(epoch, selected_loss, optimizer), f"{writer.log_dir}/checkpoint_test_best.pth")

    save_training_plot(history, os.path.join(log_dir, 'training_metrics.png'))
    if probe_history is not None:
        probe_plot_path = os.path.join(log_dir, 'probe_metrics.png')
        save_probe_plot(probe_history, probe_plot_path)
        log(f"Saved probe metrics plot: {probe_plot_path}")

    result_dict = {
        'architecture': 'baseline',
        'eval_mode': args.eval_mode,
        'top1 acc': f'{top1_acc:.2f}',
        'top5 acc': f'{top5_acc:.2f}',
        'best top1 acc': f'{best_top1_acc:.2f}',
        'best top5 acc': f'{best_top5_acc:.2f}',
        'best test loss': f'{best_test_loss:.4f}',
        'best epoch': best_test_epoch,
    }
    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(log_dir, 'result.csv'), index=False)
    log(f'best test loss: {best_test_loss:.4f} top5 acc: {best_top5_acc:.2f} top1 acc: {best_top1_acc:.2f} at epoch {best_test_epoch}')