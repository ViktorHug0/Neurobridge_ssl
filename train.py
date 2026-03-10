import os
import argparse
import logging
from datetime import datetime
import json
import random
import time
import sys
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import EEGNet, EEGProject, TSConv, EEGTransformer
from module.loss import ContrastiveLoss
from module.util import retrieve_all
from module.projector import *
from module.sampler import GroupedImageBatchSampler
from module.domain_generalization import SubjectClassifier, grad_reverse, decorrelation_loss, coral_subject_loss
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


def save_training_plot(history, out_path):
    epochs = history['epoch']
    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax_loss.plot(epochs, history['test_loss'], label='Test Loss', color='blue', linestyle='--', linewidth=2)
    ax_acc.plot(epochs, history['top1_acc'], label='Top-1 Acc', color='black', linestyle='--', linewidth=2)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_acc.set_ylabel('Top-1 Accuracy (%)')
    ax_acc.set_ylim(0, 40)
    ax_loss.grid(alpha=0.3)

    lines_1, labels_1 = ax_loss.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_eeg_encoder(args, feature_dim, eeg_sample_points, channels_num):
    if args.eeg_encoder_type == 'ATM':
        return ATMS(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGNet':
        return EEGNet(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGProject':
        return EEGProject(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'TSConv':
        return TSConv(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    if args.eeg_encoder_type == 'EEGTransformer':
        return EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    raise ValueError(f"Unsupported EEG encoder type: {args.eeg_encoder_type}")


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
    parser.add_argument('--val_subject_ids', default=[], nargs='*', type=int)
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--strict_dg', action='store_true')
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
    parser.add_argument('--ssl_lambda', default=0.0, type=float, help='weight for EEG-only cross-subject SSL loss')
    parser.add_argument('--ssl_projector_dim', default=128, type=int, help='bottleneck dimension for the EEG-only SSL head')
    parser.add_argument('--style_dim', default=128, type=int, help='style head dimension')
    parser.add_argument('--adv_lambda', default=0.0, type=float, help='subject adversarial loss weight on content')
    parser.add_argument('--style_lambda', default=0.0, type=float, help='subject classification loss weight on style')
    parser.add_argument('--decor_lambda', default=0.0, type=float, help='content-style decorrelation loss weight')
    parser.add_argument('--coral_lambda', default=0.0, type=float, help='source-subject CORAL loss weight on content')
    parser.add_argument('--episodic_source_holdout', action='store_true', help='enable source-held-out episodic meta-training')
    parser.add_argument('--episodic_lambda', default=0.2, type=float, help='weight for meta-heldout source loss')
    parser.add_argument('--eeg_data_dir', default='./things_eeg/data/preprocessed_eeg', type=str, help='where your EEG data are')
    parser.add_argument("--selected_channels", default=[], nargs='*', type=str, help="selected EEG channels, empty means all channels")
    parser.add_argument('--time_window', type=int, default=[0, 250], nargs=2, help='time window for EEG data, in sample points')
    parser.add_argument('--eeg_aug', action='store_true')
    parser.add_argument('--eeg_aug_type', type=str, choices=['noise', 'time_shift', 'channel_dropout', 'smooth'], default='noise', help='eeg augmentation type')
    parser.add_argument('--eeg_encoder_type', type=str, choices=['ATM', "EEGNet", "EEGProject", "TSConv", "EEGTransformer"], default='EEGProject')
    parser.add_argument('--image_aug', action='store_true')
    parser.add_argument('--image_test_aug', action='store_true')
    parser.add_argument('--eeg_test_aug', action='store_true')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')
    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'mlp'], default='direct')
    parser.add_argument('--feature_dim', type=int, default=512, help='output dimension when projector is not direct')
    parser.add_argument('--image_feature_dir', default='/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit', type=str, help='where your image feature are')
    parser.add_argument('--aug_image_feature_dirs', default=[], nargs='+', type=str, help='where your augmentation image feature are')
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')
    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    args = parser.parse_args()

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

    with open(os.path.join(args.output_dir, "last_run.txt"), 'w') as f:
        f.write(writer.log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')

    if args.grouped_batch_sampler and args.data_random:
        raise ValueError("Grouped batching requires deterministic indices. Disable --data_random when using --grouped_batch_sampler.")
    if args.strict_dg and len(args.val_subject_ids) == 0:
        raise ValueError("Strict DG requires --val_subject_ids.")
    if any(s in args.test_subject_ids for s in args.val_subject_ids):
        raise ValueError("Validation subjects must not overlap test subjects.")

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

    train_subject_ids = [sid for sid in args.train_subject_ids if sid not in set(args.val_subject_ids)]
    if len(train_subject_ids) == 0:
        raise ValueError("No training subjects left after removing validation subjects.")

    train_dataset = EEGPreImageDataset(
        train_subject_ids,
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
    feature_dim = train_dataset.image_features.shape[-1]
    channels_num = train_dataset.channels_num
    log(f'EEG sample points: {eeg_sample_points}')
    log(f'feature dimension: {feature_dim}')
    log(f'data length: {len(train_dataset)}')
    log(f'number of channels: {channels_num}')

    if args.grouped_batch_sampler:
        batch_sampler = GroupedImageBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            samples_per_image=args.samples_per_image,
            drop_last=True,
            seed=seed,
        )
        dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)
        effective_batch_size = batch_sampler.images_per_batch * batch_sampler.samples_per_image
        log(
            f'Grouped batch sampler enabled: images_per_batch={batch_sampler.images_per_batch}, '
            f'samples_per_image={batch_sampler.samples_per_image}, effective_batch_size={effective_batch_size}'
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers
        )

    val_dataloader = None
    if len(args.val_subject_ids) > 0:
        print('\n>>> Loading Val Data <<<')
        val_dataset = EEGPreImageDataset(
            args.val_subject_ids,
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

    inference_config = {k: args_dict[k] for k in ['eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir']}
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['feature_dim'] = feature_dim
    with open(os.path.join(writer.log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)

    model = build_eeg_encoder(args, feature_dim, eeg_sample_points, channels_num).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'EEG Encoder trainable parameters: {num_params / 1e6:.2f}M')
    log(str(model))

    if args.projector == 'direct':
        eeg_projector = ProjectorDirect().to(device)
        img_projector = ProjectorDirect().to(device)
        text_projector = ProjectorDirect().to(device)
    elif args.projector == 'linear':
        eeg_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
        img_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
        text_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
    else:
        eeg_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        img_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        text_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)

    projector_params = (
        sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in img_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    )
    log(f'Projector trainable parameters: {projector_params / 1e6:.2f}M')

    style_projector = ProjectorLinear(feature_dim, args.style_dim).to(device)
    subject_id_to_idx = {sid: i for i, sid in enumerate(sorted(train_subject_ids))}
    content_subject_classifier = SubjectClassifier(args.feature_dim, len(subject_id_to_idx)).to(device)
    style_subject_classifier = SubjectClassifier(args.style_dim, len(subject_id_to_idx)).to(device)
    subject_criterion = torch.nn.CrossEntropyLoss()

    eeg_ssl_projector = None
    if args.ssl_lambda > 0:
        eeg_ssl_projector = ProjectorMLP(feature_dim, args.ssl_projector_dim).to(device)
        ssl_params = sum(p.numel() for p in eeg_ssl_projector.parameters() if p.requires_grad)
        log(f'EEG-only SSL projector trainable parameters: {ssl_params / 1e6:.2f}M')
        log(f'EEG-only SSL enabled with lambda={args.ssl_lambda:.4f} and bottleneck dim={args.ssl_projector_dim}')

    criterion = ContrastiveLoss(
        args.init_temperature,
        args.alpha,
        args.beta,
        args.eeg_l2norm,
        args.img_l2norm,
        args.text_l2norm,
        args.t_learnable,
        args.softplus
    ).to(device)
    log(str(criterion))

    trainable_parameters = (
        list(model.parameters())
        + list(eeg_projector.parameters())
        + list(img_projector.parameters())
        + list(text_projector.parameters())
        + list(style_projector.parameters())
        + list(content_subject_classifier.parameters())
        + list(style_subject_classifier.parameters())
    )
    if eeg_ssl_projector is not None:
        trainable_parameters.extend(list(eeg_ssl_projector.parameters()))
    if args.t_learnable:
        trainable_parameters.extend([p for p in criterion.parameters() if p.requires_grad])

    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_select_loss = float('inf')
    best_select_epoch = 0
    best_val_top1_acc = 0.0
    best_val_top5_acc = 0.0
    best_test_top1_at_select = 0.0
    best_test_top5_at_select = 0.0
    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'top1_acc': []}

    def evaluate_split(split_dataloader, split_name):
        total_loss = 0.0
        eeg_feature_list = []
        image_feature_list = []
        with torch.no_grad():
            for batch in split_dataloader:
                eeg_batch = batch[0].to(device)
                image_feature_batch = batch[1].to(device)
                text_feature_batch = batch[2].to(device)
                subject_id_batch = batch[3].to(device)
                object_idx_batch = batch[4].to(device)
                image_idx_batch = batch[5].to(device)

                if args.eeg_encoder_type == 'ATM':
                    eeg_backbone_batch = model(eeg_batch, subject_id_batch)
                else:
                    eeg_backbone_batch = model(eeg_batch)

                eeg_feature_batch = eeg_projector(eeg_backbone_batch)
                image_feature_batch = img_projector(image_feature_batch)
                text_feature_batch = text_projector(text_feature_batch)
                positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
                loss = compute_cross_modal_loss(
                    criterion,
                    eeg_feature_batch,
                    image_feature_batch,
                    text_feature_batch,
                    positive_mask,
                    args.multi_positive_loss
                )
                total_loss += loss.item()
                eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
                image_feature_list.append(image_feature_batch.cpu().numpy())

        avg_loss = total_loss / len(split_dataloader)
        eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
        image_feature_all = np.concatenate(image_feature_list, axis=0)
        top5_count, top1_count, total = retrieve_all(eeg_feature_all, image_feature_all, args.data_average)
        top5_acc = top5_count / total * 100
        top1_acc = top1_count / total * 100
        writer.add_scalar(f'Loss/{split_name}', avg_loss, epoch)
        writer.add_scalar(f'Acc/top1_{split_name}', top1_acc, epoch)
        writer.add_scalar(f'Acc/top5_{split_name}', top5_acc, epoch)
        return avg_loss, top1_acc, top5_acc

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        eeg_projector.train()
        img_projector.train()
        text_projector.train()
        style_projector.train()
        content_subject_classifier.train()
        style_subject_classifier.train()
        if eeg_ssl_projector is not None:
            eeg_ssl_projector.train()

        total_loss = 0.0
        total_ssl_loss = 0.0
        epoch_meta_sid = None
        if args.episodic_source_holdout:
            epoch_meta_sid = sorted(train_subject_ids)[(epoch - 1) % len(train_subject_ids)]
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]"):
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            optimizer.zero_grad()
            if args.eeg_encoder_type == 'ATM':
                eeg_backbone_batch = model(eeg_batch, subject_id_batch)
            else:
                eeg_backbone_batch = model(eeg_batch)

            eeg_feature_batch = eeg_projector(eeg_backbone_batch)
            eeg_style_batch = style_projector(eeg_backbone_batch)
            image_feature_batch = img_projector(image_feature_batch)
            text_feature_batch = text_projector(text_feature_batch)

            if args.episodic_source_holdout:
                meta_mask = subject_id_batch == epoch_meta_sid
                base_mask = ~meta_mask
                if torch.sum(base_mask) == 0:
                    base_mask = torch.ones_like(meta_mask, dtype=torch.bool)
                    meta_mask = torch.zeros_like(meta_mask, dtype=torch.bool)
                base_positive_mask = build_image_positive_mask(object_idx_batch[base_mask], image_idx_batch[base_mask])
                loss = compute_cross_modal_loss(
                    criterion,
                    eeg_feature_batch[base_mask],
                    image_feature_batch[base_mask],
                    text_feature_batch[base_mask],
                    base_positive_mask,
                    args.multi_positive_loss
                )
                if torch.any(meta_mask):
                    meta_positive_mask = build_image_positive_mask(object_idx_batch[meta_mask], image_idx_batch[meta_mask])
                    meta_loss = compute_cross_modal_loss(
                        criterion,
                        eeg_feature_batch[meta_mask],
                        image_feature_batch[meta_mask],
                        text_feature_batch[meta_mask],
                        meta_positive_mask,
                        args.multi_positive_loss
                    )
                    loss = loss + args.episodic_lambda * meta_loss
            else:
                positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
                loss = compute_cross_modal_loss(
                    criterion,
                    eeg_feature_batch,
                    image_feature_batch,
                    text_feature_batch,
                    positive_mask,
                    args.multi_positive_loss
                )

            ssl_loss = eeg_feature_batch.new_tensor(0.0)
            if eeg_ssl_projector is not None:
                eeg_ssl_feature_batch = eeg_ssl_projector(eeg_backbone_batch)
                ssl_positive_mask = build_cross_subject_positive_mask(object_idx_batch, image_idx_batch, subject_id_batch)
                ssl_loss = criterion.self_similarity_loss(eeg_ssl_feature_batch, ssl_positive_mask)
                loss = loss + args.ssl_lambda * ssl_loss
                total_ssl_loss += ssl_loss.item()

            subject_idx_batch = torch.tensor(
                [subject_id_to_idx[int(s.item())] for s in subject_id_batch],
                device=device,
                dtype=torch.long,
            )
            adv_logits = content_subject_classifier(grad_reverse(eeg_feature_batch, 1.0))
            style_logits = style_subject_classifier(eeg_style_batch)
            adv_loss = subject_criterion(adv_logits, subject_idx_batch)
            style_loss = subject_criterion(style_logits, subject_idx_batch)
            decor_loss = decorrelation_loss(eeg_feature_batch, eeg_style_batch)
            coral_loss = coral_subject_loss(eeg_feature_batch, subject_id_batch)
            loss = (
                loss
                + args.adv_lambda * adv_loss
                + args.style_lambda * style_loss
                + args.decor_lambda * decor_loss
                + args.coral_lambda * coral_loss
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        if eeg_ssl_projector is not None:
            writer.add_scalar('Loss/train_ssl', total_ssl_loss / len(dataloader), epoch)
        log(f"Epoch [{epoch}/{args.num_epochs}] Train Loss: {avg_loss:.4f}")

        if args.save_weights:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'eeg_projector_state_dict': eeg_projector.state_dict(),
                'style_projector_state_dict': style_projector.state_dict(),
                'content_subject_classifier_state_dict': content_subject_classifier.state_dict(),
                'style_subject_classifier_state_dict': style_subject_classifier.state_dict(),
                'img_projector_state_dict': img_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            if eeg_ssl_projector is not None:
                checkpoint['eeg_ssl_projector_state_dict'] = eeg_ssl_projector.state_dict()
            torch.save(checkpoint, f"{writer.log_dir}/checkpoint_last.pth")

        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()
        style_projector.eval()
        content_subject_classifier.eval()
        style_subject_classifier.eval()
        if eeg_ssl_projector is not None:
            eeg_ssl_projector.eval()
        avg_test_loss, top1_acc, top5_acc = evaluate_split(test_dataloader, 'test')
        log(f"[test] top5 acc {top5_acc:.2f}%\ttop1 acc {top1_acc:.2f}%\tLoss: {avg_test_loss:.4f}")
        if val_dataloader is not None:
            avg_val_loss, val_top1_acc, val_top5_acc = evaluate_split(val_dataloader, 'val')
            log(f"[val] top5 acc {val_top5_acc:.2f}%\ttop1 acc {val_top1_acc:.2f}%\tLoss: {avg_val_loss:.4f}")
        else:
            avg_val_loss, val_top1_acc, val_top5_acc = avg_test_loss, top1_acc, top5_acc

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(avg_test_loss)
        history['top1_acc'].append(top1_acc)

        is_better = False
        if avg_val_loss < best_select_loss:
            is_better = True
        elif avg_val_loss == best_select_loss and val_top1_acc > best_val_top1_acc:
            is_better = True

        if is_better:
            best_select_loss = avg_val_loss
            best_val_top1_acc = val_top1_acc
            best_val_top5_acc = val_top5_acc
            best_top5_acc = top5_acc
            best_top1_acc = top1_acc
            best_test_top1_at_select = top1_acc
            best_test_top5_at_select = top5_acc
            best_select_epoch = epoch
            if args.save_weights:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'eeg_projector_state_dict': eeg_projector.state_dict(),
                    'style_projector_state_dict': style_projector.state_dict(),
                    'content_subject_classifier_state_dict': content_subject_classifier.state_dict(),
                    'style_subject_classifier_state_dict': style_subject_classifier.state_dict(),
                    'img_projector_state_dict': img_projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }
                if eeg_ssl_projector is not None:
                    checkpoint['eeg_ssl_projector_state_dict'] = eeg_ssl_projector.state_dict()
                torch.save(checkpoint, f"{writer.log_dir}/checkpoint_test_best.pth")

    save_training_plot(history, os.path.join(log_dir, 'training_metrics.png'))

    result_dict = {
        'top1 acc': f'{top1_acc:.2f}',
        'top5 acc': f'{top5_acc:.2f}',
        'best top1 acc': f'{best_test_top1_at_select:.2f}',
        'best top5 acc': f'{best_test_top5_at_select:.2f}',
        'best val loss': f'{best_select_loss:.4f}',
        'best val top1 acc': f'{best_val_top1_acc:.2f}',
        'best val top5 acc': f'{best_val_top5_acc:.2f}',
        'best epoch': best_select_epoch,
    }
    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(log_dir, 'result.csv'), index=False)
    log(
        f'best val loss: {best_select_loss:.4f} '
        f'best test top5/top1: {best_test_top5_at_select:.2f}/{best_test_top1_at_select:.2f} '
        f'at epoch {best_select_epoch}'
    )