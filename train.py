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
from module.util import retrieve_all
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


def build_image_positive_mask(object_indices, image_indices):
    same_object = object_indices.unsqueeze(1) == object_indices.unsqueeze(0)
    same_image = image_indices.unsqueeze(1) == image_indices.unsqueeze(0)
    return same_object & same_image


def build_cross_subject_positive_mask(object_indices, image_indices, subject_ids):
    same_image = build_image_positive_mask(object_indices, image_indices)
    different_subject = subject_ids.unsqueeze(1) != subject_ids.unsqueeze(0)
    return same_image & different_subject


def sample_cross_subject_partner_indices(object_indices, image_indices, subject_ids):
    obj = object_indices.detach().cpu().tolist()
    img = image_indices.detach().cpu().tolist()
    sid = subject_ids.detach().cpu().tolist()

    groups = {}
    for idx, (o, im) in enumerate(zip(obj, img)):
        groups.setdefault((int(o), int(im)), []).append(idx)

    partner_indices = []
    valid_anchor_indices = []
    for idx, (o, im, s) in enumerate(zip(obj, img, sid)):
        candidates = [j for j in groups[(int(o), int(im))] if j != idx and sid[j] != s]
        if len(candidates) == 0:
            continue
        partner_indices.append(random.choice(candidates))
        valid_anchor_indices.append(idx)

    if len(valid_anchor_indices) == 0:
        empty = object_indices.new_empty((0,), dtype=torch.long)
        return empty, empty
    return (
        torch.tensor(valid_anchor_indices, device=object_indices.device, dtype=torch.long),
        torch.tensor(partner_indices, device=object_indices.device, dtype=torch.long),
    )


def compute_pair_infonce_loss(anchor_feature, positive_feature, criterion, tau):
    if criterion.eeg_l2norm:
        anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
        positive_feature = F.normalize(positive_feature, p=2, dim=1)

    if tau is None or tau <= 0:
        logit_scale = criterion._get_logit_scale()
    else:
        logit_scale = 1.0 / tau
    logits = torch.matmul(anchor_feature, positive_feature.T) * logit_scale
    labels = torch.arange(anchor_feature.shape[0], device=anchor_feature.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


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


def compute_relic_prediction_consistency_loss(criterion, eeg_feature, image_feature, positive_mask):
    eeg_feature, image_feature, _ = criterion._normalize_inputs(eeg_feature, image_feature, None)
    logits = torch.matmul(eeg_feature, image_feature.T) * criterion._get_logit_scale()
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    pair_mask = positive_mask.clone()
    pair_mask.fill_diagonal_(False)
    if not torch.any(pair_mask):
        return logits.new_tensor(0.0)

    src_idx, dst_idx = pair_mask.nonzero(as_tuple=True)
    log_probs_src = log_probs[src_idx]
    log_probs_dst = log_probs[dst_idx]
    probs_src = probs[src_idx]
    probs_dst = probs[dst_idx]

    kl_forward = torch.sum(probs_src * (log_probs_src - log_probs_dst), dim=-1)
    kl_reverse = torch.sum(probs_dst * (log_probs_dst - log_probs_src), dim=-1)
    return (0.5 * (kl_forward + kl_reverse)).mean()


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


def cross_subject_stimulus_mix(features, object_indices, image_indices, subject_ids, alpha=1.0, mixup_type='pairwise'):
    if features.shape[0] < 2:
        return features

    obj = object_indices.detach().cpu().tolist()
    img = image_indices.detach().cpu().tolist()
    sid = subject_ids.detach().cpu().tolist()

    groups = {}
    for i, (o, im) in enumerate(zip(obj, img)):
        groups.setdefault((int(o), int(im)), []).append(i)

    mixed = features.clone()
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

            concentration = torch.full((group_size,), dist_alpha, device=features.device, dtype=torch.float32)
            lam = torch.distributions.Beta(concentration, concentration).sample().to(group_features.dtype)
            lam_shape = [group_size] + [1] * (group_features.dim() - 1)
            lam = lam.view(*lam_shape)
            mixed_group = lam * group_features + (1.0 - lam) * group_features[partner_pos]

        mixed[group_idx] = mixed_group

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
    parser.add_argument('--eeg_l2_norm_ssl', action='store_true', help='L2-normalize EEG embeddings only for cross-subject SSL loss')
    parser.add_argument('--multi_positive_loss', action='store_true', help='enable multi-positive EEG-image contrastive loss')
    parser.add_argument('--grouped_batch_sampler', action='store_true', help='sample multiple subjects per exact image in each batch')
    parser.add_argument('--samples_per_image', default=4, type=int, help='number of samples per exact image for grouped batches')
    parser.add_argument('--subject_mixup_mode', type=str, choices=['none', 'raw_eeg', 'embedding'], default='none', help='cross-subject same-stimulus convex mixing mode')
    parser.add_argument('--mixup_type', type=str, choices=['pairwise', 'group'], default='pairwise', help='pairwise mixing or full same-stimulus group mixing')
    parser.add_argument('--subject_mixup_alpha', default=1.0, type=float, help='beta(alpha, alpha) coefficient for cross-subject same-stimulus mixup')
    parser.add_argument('--ssl_lambda', default=0.0, type=float, help='weight for EEG-only cross-subject SSL loss')
    parser.add_argument('--relic_lambda', default=0.0, type=float, help='weight for RELIC-style prediction consistency across same-image different-subject EEG views')
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
    parser.add_argument('--architecture', type=str, choices=['baseline', 'clap_adapter'], default='baseline', help='training architecture')
    parser.add_argument('--pretrain_epochs', default=0, type=int, help='baseline pretraining epochs before freezing the EEG encoder and training the CLAP adapter')
    parser.add_argument('--adapter_hidden_dim', default=256, type=int, help='hidden dimension of the frozen-adapter residual MLP')
    parser.add_argument('--adapter_alpha', default=1.0, type=float, help='inference-time alpha for the frozen-adapter residual MLP')
    parser.add_argument('--clap_loss_lambda', default=1.0, type=float, help='weight for CLAP stage-2 paired InfoNCE')
    parser.add_argument('--clap_tau', default=0.5, type=float, help='temperature for CLAP stage-2 paired InfoNCE (<=0 uses base logit scale)')
    parser.add_argument('--clap_transfer', action='store_true', default=True, help='whether to apply the CLAP adapter to the image branch at inference')
    parser.add_argument('--no_clap_transfer', action='store_false', dest='clap_transfer', help='disable CLAP adapter transfer to image branch')
    parser.add_argument('--clap_mse_lambda', default=0.0, type=float, help='weight for identity anchor MSE loss in stage-2')
    parser.add_argument('--val_subject_id', default=None, type=int, help='subject ID used for validation model selection')
    parser.add_argument('--select_best_on', type=str, choices=['test', 'val'], default='test', help='which split selects the best checkpoint')
    parser.add_argument('--subject_probe_holdout', action='store_true', help='per-subject held-out split; train linear subject probes (baseline only)')
    parser.add_argument('--subject_probe_holdout_ratio', type=float, default=0.10, help='fraction per train subject reserved for probe validation')
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
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')
    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    args = parser.parse_args()
    if args.subject_probe_holdout:
        if not (0.0 < args.subject_probe_holdout_ratio < 1.0):
            raise ValueError("--subject_probe_holdout_ratio must be strictly between 0 and 1.")
    if args.train_saw_whiten_image and not args.train_saw:
        raise ValueError("--train_saw_whiten_image requires --train_saw.")

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

    if args.subject_probe_holdout and args.architecture != 'baseline':
        raise ValueError("--subject_probe_holdout is only supported with --architecture baseline.")

    with open(os.path.join(args.output_dir, "last_run.txt"), 'w') as f:
        f.write(writer.log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')
    log(f'Subject mixup mode: {args.subject_mixup_mode}, type: {args.mixup_type} (alpha={args.subject_mixup_alpha})')
    log(f'Evaluation mode: {args.eval_mode}')

    if args.grouped_batch_sampler and args.data_random:
        raise ValueError("Grouped batching requires deterministic indices. Disable --data_random when using --grouped_batch_sampler.")
    if args.relic_lambda > 0 and not args.grouped_batch_sampler:
        raise ValueError("RELIC prediction consistency requires --grouped_batch_sampler to form same-image cross-subject pairs.")
    if args.relic_lambda > 0 and args.samples_per_image < 2:
        raise ValueError("RELIC prediction consistency requires --samples_per_image >= 2.")

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
        args.frozen_eeg_prior
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

    if args.architecture == 'clap_adapter' and not args.grouped_batch_sampler:
        raise ValueError("clap_adapter requires --grouped_batch_sampler to ensure cross-subject pairs.")
    if args.architecture == 'clap_adapter' and args.samples_per_image < 2:
        raise ValueError("clap_adapter requires --samples_per_image >= 2.")
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
            args.frozen_eeg_prior
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
        args.frozen_eeg_prior
    )
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    inference_keys = [
        'eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir', 'architecture',
        'projector', 'feature_dim', 'eeg_backbone_dim', 'pretrain_epochs', 'adapter_hidden_dim',
        'adapter_alpha', 'time_window', 'selected_channels',
        'eval_mode', 'sattc_saw_shrink', 'sattc_saw_diag', 'sattc_csls_k', 'sattc_cw', 'sattc_cw_shrink', 'sattc_cw_diag',
    ]
    inference_config = {k: args_dict[k] for k in inference_keys}
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['backbone_feature_dim'] = backbone_feature_dim
    inference_config['image_feature_dim'] = image_feature_dim
    with open(os.path.join(writer.log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)

    model = build_eeg_encoder(args, backbone_feature_dim, eeg_sample_points, channels_num).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'EEG Encoder trainable parameters: {num_params / 1e6:.2f}M')
    log(str(model))

    eeg_adapter = None
    if args.architecture == 'baseline':
        eeg_projector = build_projector(args.projector, backbone_feature_dim, args.feature_dim).to(device)
    else:
        eeg_projector = build_projector(args.projector, backbone_feature_dim, args.feature_dim).to(device)
        eeg_adapter = ResidualAdapter(args.feature_dim, args.adapter_hidden_dim, args.adapter_alpha).to(device)

    img_projector = build_projector(args.projector, image_feature_dim, args.feature_dim).to(device)
    text_projector = build_projector(args.projector, image_feature_dim, args.feature_dim).to(device)
    projector_params = (
        sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in img_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    )
    if eeg_adapter is not None:
        projector_params += sum(p.numel() for p in eeg_adapter.parameters() if p.requires_grad)
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
        args.eeg_l2_norm_ssl,
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

    if args.architecture == 'clap_adapter':
        if args.pretrain_epochs <= 0 or args.pretrain_epochs >= args.num_epochs:
            raise ValueError(f"{args.architecture} requires 0 < pretrain_epochs < num_epochs.")
        stage_schedule = [('pretrain', args.pretrain_epochs), ('adapter', args.num_epochs - args.pretrain_epochs)]
    else:
        stage_schedule = [('joint', args.num_epochs)]
    total_epochs = sum(item[1] for item in stage_schedule)
    log(f"Stage schedule: {stage_schedule} (total_epochs={total_epochs})")

    module_registry = {
        'model': model,
        'eeg_projector': eeg_projector,
        'img_projector': img_projector,
        'text_projector': text_projector,
        'eeg_adapter': eeg_adapter,
    }

    def get_epoch_stage(epoch_idx):
        running = 0
        for stage_name, length in stage_schedule:
            running += length
            if epoch_idx <= running:
                return stage_name
        return stage_schedule[-1][0]

    def configure_stage(stage_name):
        if args.architecture == 'baseline':
            active = {'model', 'eeg_projector', 'img_projector', 'text_projector'}
        else:
            if stage_name == 'pretrain':
                active = {'model', 'eeg_projector', 'img_projector', 'text_projector'}
            else:
                active = {'eeg_adapter'}
        lr = args.learning_rate

        for name, module in module_registry.items():
            if module is None:
                continue
            enabled = name in active
            set_requires_grad(module, enabled)
            if enabled:
                module.train()
            else:
                module.eval()

        criterion.train()
        trainable_modules = [module_registry[name] for name in sorted(active) if module_registry[name] is not None]
        trainable_parameters = collect_trainable_parameters(trainable_modules)
        if args.t_learnable:
            trainable_parameters.extend([p for p in criterion.parameters() if p.requires_grad])
        if len(trainable_parameters) == 0:
            raise ValueError(f"No trainable parameters active for stage '{stage_name}'.")
        optimizer = optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        return optimizer

    def forward_architecture(eeg_backbone_batch, stage_name):
        output = {'eeg_feature': None, 'ssl_feature': None, 'z_content': None}
        if args.architecture == 'baseline':
            output['eeg_feature'] = eeg_projector(eeg_backbone_batch)
            output['ssl_feature'] = output['eeg_feature']
            return output

        base_feature = eeg_projector(eeg_backbone_batch)
        output['z_content'] = base_feature
        if stage_name == 'adapter':
            output['eeg_feature'] = eeg_adapter(base_feature)
        else:
            output['eeg_feature'] = base_feature
        output['ssl_feature'] = output['eeg_feature']
        return output

    def build_checkpoint(epoch_idx, loss_value, stage_name, optimizer_obj):
        checkpoint = {
            'epoch': epoch_idx,
            'stage': stage_name,
            'architecture': args.architecture,
            'model_state_dict': model.state_dict(),
            'eeg_projector_state_dict': eeg_projector.state_dict(),
            'img_projector_state_dict': img_projector.state_dict(),
            'text_projector_state_dict': text_projector.state_dict(),
            'optimizer_state_dict': optimizer_obj.state_dict(),
            'loss': loss_value,
        }
        if eeg_adapter is not None:
            checkpoint['eeg_adapter_state_dict'] = eeg_adapter.state_dict()
        if args.t_learnable:
            checkpoint['criterion_state_dict'] = criterion.state_dict()
        return checkpoint

    def maybe_apply_clap_adapter_to_image(image_feature_proj, stage_name):
        if args.architecture == 'clap_adapter' and args.clap_transfer and stage_name == 'adapter' and eeg_adapter is not None:
            return eeg_adapter(image_feature_proj)
        return image_feature_proj

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

    def evaluate_split(split_loader, stage_name, split_name):
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
                arch_out = forward_architecture(eeg_backbone_batch, stage_name)
                eeg_feature_batch = arch_out['eeg_feature']
                image_feature_proj = img_projector(image_feature_batch)
                image_feature_proj = maybe_apply_clap_adapter_to_image(image_feature_proj, stage_name)
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
    optimizer = None
    active_stage = None
    top1_acc = 0.0
    top5_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        stage = get_epoch_stage(epoch)
        if stage != active_stage:
            optimizer = configure_stage(stage)
            active_stage = stage
            log(f"Switched to stage '{stage}' at epoch {epoch}.")

        total_loss = 0.0
        total_ssl_loss = 0.0
        total_relic_loss = 0.0
        total_cl_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [{stage}]"):
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            optimizer.zero_grad()
            if args.subject_mixup_mode == 'raw_eeg':
                eeg_batch = cross_subject_stimulus_mix(
                    eeg_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                    alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type
                )
            eeg_backbone_batch = run_eeg_backbone(model, args, eeg_batch, subject_id_batch)
            if args.subject_mixup_mode == 'embedding':
                eeg_backbone_batch = cross_subject_stimulus_mix(
                    eeg_backbone_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                    alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type
                )
            arch_out = forward_architecture(eeg_backbone_batch, stage)

            loss = eeg_batch.new_tensor(0.0)

            eeg_feature_batch = arch_out['eeg_feature']
            image_feature_proj = img_projector(image_feature_batch)
            image_feature_proj = maybe_apply_clap_adapter_to_image(image_feature_proj, stage)
            text_feature_proj = text_projector(text_feature_batch)
            eeg_feature_for_loss, image_feature_for_loss = maybe_apply_train_saw(
                eeg_feature_batch,
                image_feature_proj,
                subject_id_batch,
            )
            if args.architecture == 'clap_adapter' and stage == 'adapter':
                valid_anchor_idx, partner_idx = sample_cross_subject_partner_indices(
                    object_idx_batch, image_idx_batch, subject_id_batch
                )
                if valid_anchor_idx.numel() > 0:
                    anchor_feature = eeg_feature_batch[valid_anchor_idx]
                    positive_feature = eeg_feature_batch[partner_idx]
                    cl_loss = compute_pair_infonce_loss(anchor_feature, positive_feature, criterion, args.clap_tau)
                    loss = loss + args.clap_loss_lambda * cl_loss
                    
                    if args.clap_mse_lambda > 0:
                        mse_loss = F.mse_loss(eeg_feature_batch, arch_out['z_content'].detach())
                        loss = loss + args.clap_mse_lambda * mse_loss
                    
                    total_cl_loss += cl_loss.item()
                else:
                    cl_loss = eeg_feature_batch.sum() * 0.0
                    loss = loss + cl_loss
            else:
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

            if args.ssl_lambda > 0:
                ssl_positive_mask = build_cross_subject_positive_mask(object_idx_batch, image_idx_batch, subject_id_batch)
                ssl_loss = criterion.self_similarity_loss(arch_out['ssl_feature'], ssl_positive_mask)
                loss = loss + args.ssl_lambda * ssl_loss
                total_ssl_loss += ssl_loss.item()

            if args.relic_lambda > 0:
                relic_positive_mask = build_cross_subject_positive_mask(object_idx_batch, image_idx_batch, subject_id_batch)
                relic_loss = compute_relic_prediction_consistency_loss(
                    criterion,
                    eeg_feature_for_loss,
                    image_feature_for_loss,
                    relic_positive_mask,
                )
                loss = loss + args.relic_lambda * relic_loss
                total_relic_loss += relic_loss.item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/train_cl', total_cl_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_ssl', total_ssl_loss / len(dataloader), epoch)
        writer.add_scalar('Loss/train_relic', total_relic_loss / len(dataloader), epoch)
        log(
            f"Epoch [{epoch}/{total_epochs}] Stage={stage} TrainLoss={avg_loss:.4f} "
            f"CL={total_cl_loss / len(dataloader):.4f} SSL={total_ssl_loss / len(dataloader):.4f} "
            f"RELIC={total_relic_loss / len(dataloader):.4f}"
        )

        if args.save_weights:
            torch.save(build_checkpoint(epoch, avg_loss, stage, optimizer), f"{writer.log_dir}/checkpoint_last.pth")

        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()
        if eeg_adapter is not None:
            eeg_adapter.eval()

        avg_test_loss, top1_acc, top5_acc = evaluate_split(test_dataloader, stage, "test")
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Acc/top1_test', top1_acc, epoch)
        writer.add_scalar('Acc/top5_test', top5_acc, epoch)

        val_loss, val_top1, val_top5 = float('nan'), float('nan'), float('nan')
        if val_dataloader is not None:
            val_loss, val_top1, val_top5 = evaluate_split(val_dataloader, stage, "val")
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
                torch.save(build_checkpoint(epoch, selected_loss, stage, optimizer), f"{writer.log_dir}/checkpoint_test_best.pth")

    save_training_plot(history, os.path.join(log_dir, 'training_metrics.png'))
    if probe_history is not None:
        probe_plot_path = os.path.join(log_dir, 'probe_metrics.png')
        save_probe_plot(probe_history, probe_plot_path)
        log(f"Saved probe metrics plot: {probe_plot_path}")

    result_dict = {
        'architecture': args.architecture,
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