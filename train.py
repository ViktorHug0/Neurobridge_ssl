import os
import argparse
import copy
import logging
import math
import re
from datetime import datetime
import json
import random
import time
import sys
import shutil
from types import SimpleNamespace

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd

from module.augment import (
    RandomGaussianNoise,
    RandomSmooth,
    cross_subject_operator_transfer,
    functional_operator_transfer,
    cross_subject_stimulus_mix,
)
from module.dataset import EEGPreImageDataset, ExtraTrials, GroupedImageBatchSampler, GroupedSubset, RepeatPairSampler, TestRepTrials
from module.eeg_encoder.model import build_eeg_encoder, run_eeg_backbone
from module.loss import (
    ContrastiveLoss,
    build_image_positive_mask,
    compute_cross_modal_loss,
    compute_subject_mixup_regularization,
)
from module.util import collect_trainable_parameters, retrieve_all, seed_everything
from module.projector import build_projector
from module.teacher import Teachers
from module.training_plots import (
    empty_alignment_sparsity_accumulator,
    finalize_alignment_sparsity_accumulator,
    update_alignment_sparsity_accumulator,
    compute_alignment_sparsity_stats,
    save_probe_plot,
    save_training_plot,
)


def log(message):
    logging.info(message)
    print(message)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='training device')
    parser.add_argument(
        '--amp_dtype', default='none', choices=['none', 'bfloat16', 'float16'],
        help='automatic mixed precision for model/projector activations; contrastive loss stays FP32',
    )
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--layer_decay', default=1.0, type=float, help='< 1: layer-wise learning-rate decay for a pretrained foundation encoder (LaBraM authors use 0.65): block i of L trains at lr x decay^(L - i), the embeddings at decay^(L + 1), norm / head at lr')
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
    parser.add_argument('--backbone_checkpoint_chunks', default=0, type=int, help='> 1: run the EEG backbone on this many batch chunks with activation checkpointing (fits wide TSConv at batch 1024; BatchNorm sees chunk statistics)')
    parser.add_argument('--group_dro_rho', type=float, default=0.0, help='> 0: replace the batch-mean contrastive loss by the soft maximum over the batch people of their mean loss, (1/rho) log mean_d exp(rho L_d)')
    parser.add_argument('--js_groups', type=int, default=0, help='> 1: split the batch people into this many groups (subject id mod K) and shrink the mean gradient per coordinate by (1 - s^2 / (K gbar^2))_+, s^2 = between-group variance (James-Stein)')
    parser.add_argument('--person_centre_train', action='store_true', help='subtract from every training output the mean of its same-person batch-mates (leave-one-out) and add the batch mean, as the test-time centring does')
    parser.add_argument('--distill_targets', default='', type=str, help='per-trial teacher predictions (neurips_challenge/export_teacher_cv4.py .npy, row = train index)')
    parser.add_argument('--distill_lambda', default=0.0, type=float, help='weight of KL(teacher || student) over the batch images')
    parser.add_argument('--distill_tau', default=0.005, type=float)
    parser.add_argument('--kd_teacher', default='', help='ensemble distillation (scripts/subject_key/KEY_LOG.md): a file from scripts/subject_key/export_teacher.py with the members\' embeddings of every clean training trial; the student matches softmax(mean teacher cosine / --kd_tau) over the batch images (a SubjectMix pair gets the same mix of its two trials\' teacher cosines)')
    parser.add_argument('--kd_lambda', default=1.0, type=float)
    parser.add_argument('--kd_tau', default=0.05, type=float)
    parser.add_argument('--kd_members', type=int, nargs='*', default=[], help='with --kd_teacher and --aux_targets --aux_dim: one teacher member per student head (main head first, then the --aux_targets heads in order); each head matches ITS member instead of the main head matching the ensemble mean (multi-head distillation, KEY_LOG K25)')
    parser.add_argument('--save_epochs', type=int, nargs='*', default=[], help='also save checkpoint_epXX.pth at these epochs (epoch-budget checks under the sealed metric, NIGHT_LOG N71)')
    parser.add_argument('--concept_smooth', default=0.0, type=float, help='> 0: training target = (1 - a) * image embedding + a * mean (L2-normalised) embedding of its concept\'s training images (held-out images excluded): less memorisation of exact training images (NIGHT_LOG N72)')
    parser.add_argument('--holdout_images', default=0, type=int, help='> 0: drop this many training images (seed 0, flat ids object*10+image; holdout_images()) from training and from the --bank_lambda bank: held-out people x these unseen images = a large novel-image gallery proxy (logs/holdout_eval.py, NIGHT_LOG N71)')
    parser.add_argument('--holdout_avoid', default='', help='targets.npz of a --extra_trials_dir cohort on THINGS images (obj, image): --holdout_images are drawn among the OTHER training images, so the cohort keeps all its trials (scripts/mi/nq09_proxy2.py scores this second proxy)')
    parser.add_argument('--holdout_concepts', default=0, type=int, help='> 0: like --holdout_images but drops ALL 10 images of this many concepts (seed 0): the new-concept variant of the clean proxy')
    parser.add_argument('--bank_lambda', default=0.0, type=float, help='> 0: add lambda * cross-entropy of each trial over ALL training-image embeddings (cosine / --bank_tau), target = its own image: retrieval among ~16.5k images, not just the batch (NIGHT_LOG N71)')
    parser.add_argument('--bank_tau', default=0.05, type=float)
    parser.add_argument('--train_test_reps', default=0, type=int, help='> 0: also train on this many evenly spaced single-trial test reps per image of every training subject (final sealed models; README "Final sealed model")')
    parser.add_argument('--anti_teacher_lambda', default=0.0, type=float, help='> 0: add lambda * mean_i sum_{wrong j} p_student(j|i) p_teacher(j|i) over the batch images (--distill_targets teacher, --distill_tau): penalise confusions shared with the ensemble, NIGHT_LOG N67')
    parser.add_argument('--input_band', default='', choices=['', 'lp', 'hp', 'bp', 'vhp'], help="'lp': each channel's 25-sample (0.2 s at 120 Hz) moving average only; 'hp': the input minus it; 'bp': the 5-sample moving average minus it, ~4-12 Hz; 'vhp': the input minus the 5-sample moving average, > ~15 Hz (NIGHT_LOG N65)")
    parser.add_argument('--prior_lambda', default=0.0, type=float, help='> 0: add lambda * KL(batch-mean softmax over the training-image bank || uniform), each trial\'s bank cosines z-scored, temperature --prior_tau (NIGHT_LOG N66: reshape the output prior)')
    parser.add_argument('--prior_tau', default=0.25, type=float)
    parser.add_argument('--band_drop', default=0.0, type=float, help='training only: probability that a trial keeps one random frequency band (or two adjacent ones) of its input (NIGHT_LOG N65)')
    parser.add_argument('--atypical_weight', default=0.0, type=float, help='> 0: loss weight exp(-a z) per trial, z = standardised cosine of its image target to the batch-mean target: emphasises atypical images, which carry the top-5 hits at large galleries (MI_NEURIPS_LOG E01, G03)')
    parser.add_argument('--time_jitter', default=0, type=int, help='training only: shift every trial by a random whole number of samples in [-n, n] (edges replicated): robustness to a latency offset of an unseen cohort (MI_NEURIPS_LOG E02, G01)')
    parser.add_argument('--time_warp', default=0.0, type=float, help='training only: per trial, the content after the stimulus onset (sample 24) is stretched in time by a random factor in [1 - a, 1 + a] (round 11: augmentations re-tested on never-trained images)')
    parser.add_argument('--time_mask', default=0, type=int, help='training only: a random window of this many samples is zeroed in every trial')
    parser.add_argument('--adaptive_filter', default=0.0, type=float, help='a > 0: every trial (training and test) is replaced by D C_s^-1 x, C_s = (1 - a) C + a D with C the trial`s own channel covariance and D its diagonal: a per-trial adaptive spatial noise filter (identity at a = 1)')
    parser.add_argument('--xperson_lambda', default=0.0, type=float, help='JEPA-style auxiliary loss (TokenREVE arch_kwargs jepa=k, needs --grouped_batch_sampler): lambda * (1 - cosine) between the predictor output of a trial`s mid-depth latent and the stop-gradient final latent of ANOTHER person`s trial of the same image')
    parser.add_argument('--things_noise', default=0.0, type=float, help='training only: with this probability a THINGS-EEG2 trial (ids >= 21) gets real Alljoined single-trial noise added, (a - b) / sqrt 2 for two Alljoined batch trials of one person: the cleaner dataset is trained at the noise level of the target one (MI_NEURIPS_LOG H01)')
    parser.add_argument('--things_filter', default='', help='npy (taps, channels): THINGS-EEG2 trials become sum_i F[i] * x(t - i) per electrode (tap i = later by i samples, so it carries the lag; fitted by scripts/mi/ne07_things_adapter.py, MI_NEURIPS_LOG E07); replaces --things_lag')
    parser.add_argument('--extra_trials_dir', default='', help='another cohort with its own images added to training (MSS, ds005589: export/sub-XX_ses-YY.npz + targets.npz from scripts/mi/nq01 / nq02); its people get subject ids 100 + XX')
    parser.add_argument('--extra_joint_loss', action='store_true', help='control: one contrastive loss over the batch, extra-cohort images as negatives of our trials and vice versa (default: one loss per image set); halves the new-person score (MI_NEURIPS_LOG Q04)')
    parser.add_argument('--extra_shuffle_targets', action='store_true', help='control: the extra trials get the image target of a random extra trial')
    parser.add_argument('--repeat_mix', default=0.0, type=float, help='probability that a training trial is replaced by its mean with another repeat of the same person and image (single-trial datasets): cleaner examples of the same response, an SNR augmentation for the few-repeat sealed regime')
    parser.add_argument('--aux_targets', nargs='*', default=[], help='auxiliary image targets, training only: image-feature dirs (image_train.npy, same image index as --image_feature_dir); each is predicted from the input of model.head by one linear head, same contrastive loss as the main target')
    parser.add_argument('--aux_lambda', default=0.5, type=float, help='weight of each --aux_targets loss')
    parser.add_argument('--aux_dim', default=0, type=int, help='> 0: every --aux_targets space gets its own alignment space of this size (EEG head Linear(width, aux_dim) against a learned linear projection of the image features, like the main head) instead of a head into the raw feature space, and the heads are saved (aux_state_dict) to score a query in all the spaces (scripts/subject_key/KEY_LOG.md K24)')
    parser.add_argument('--aux_concept_lambda', default=0.0, type=float, help='auxiliary loss, training only: cross-entropy over the 1,654 THINGS training concepts from the input of model.head (one linear head)')
    parser.add_argument('--soft_label', default=0.0, type=float, help='SoftCLIP targets: (1 - a) one-hot + a softmax(image-image cosine / --soft_tau) over the batch images (MI_NEURIPS_LOG round 11, T2)')
    parser.add_argument('--soft_tau', default=0.05, type=float)
    parser.add_argument('--repeat_agg', default=0.0, type=float, help='> 0: batches of B/2 trials + another repeat of the same person and image each (RepeatPairSampler); loss = the single-trial loss of each half + this x the loss of the pair\'s raw mean output, which is what the grader ranks (MI_NEURIPS_LOG Y17)')
    parser.add_argument('--soft_until', default=0, type=int, help='> 0: --soft_label up to this epoch, hard targets after (MI_NEURIPS_LOG Y11: soft labels raise per-trial top-5 but lower the gain of the repeat average)')
    parser.add_argument('--alljoined_after', default=0, type=int, help='> 0: after this epoch the THINGS-EEG2 trials (ids >= 21) are dropped from every batch, so the last epochs see the target cohort only (round 11, T4)')
    parser.add_argument('--open_bank', default='', type=str, help='.npy of image embeddings with no EEG, added as columns of the EEG -> image loss (needs --soft_label; MI_NEURIPS_LOG Y02)')
    parser.add_argument('--open_bank_sep', default=0.0, type=float, help='weight of a softmax over the --open_bank images alone: target = softmax(image-bank cosine / --soft_tau)')
    parser.add_argument('--open_bank_soft', action='store_true', help='the soft-label share is spread over batch AND --open_bank images by similarity (default: bank images are negatives)')
    parser.add_argument('--gce_q', default=0.0, type=float, help='generalised cross-entropy (1 - p^q) / q instead of -log p for the EEG -> image loss (noise-robust; round 11, T3)')
    parser.add_argument('--extra_bank', action='store_true', help='the --extra_trials_dir trials are scored against ALL the images of their cohort (extra_bank_loss) instead of the cohort images of the batch')
    parser.add_argument('--extra_repeat', default=1, type=int, help='the --extra_trials_dir trials are seen this many times per epoch (a small cohort is otherwise a few percent of the batches)')
    parser.add_argument('--extra_shift', default=0, type=int, help='time shift of the --extra_trials_dir trials in samples (negative = earlier); MSS needs -3 against Alljoined (scripts/mi/nq03_mss_zeroshot.py)')
    parser.add_argument('--things_lag', default=0, type=int, help='shift the trials of THINGS-EEG2 subjects (ids >= 21 in the Alljoined dir) later by this many samples: Alljoined responses lag THINGS-EEG2 by ~42 ms = 5 samples at 120 Hz (scripts/mi/MI_NEURIPS_LOG.md D03)')
    parser.add_argument('--subject_key', default='keep', choices=['keep', 'demean', 'swap', 'swap_split', 'split', 'comb', 'comb_split'], help="the subject's image-independent mean ERP (scripts/mi/MI_LOG.md E07: the encoder uses it as a subject key). demean: subtract it, train and test (test subject: mean of its own test trials, label-free); swap: training only, replace it by a random training subject's (swap_split, ATM: only in what the readout sees, the attention map comes from the raw trial); split (ATM): the key only sets the electrode attention map, the readout sees the key-free signal; comb / comb_split: as demean / split with the key estimated from the trial itself, as its 200 ms-periodic part (scripts/subject_key/KEY_LOG.md K36): no other data of the subject, train and test")
    parser.add_argument('--train_image_targets', default=[], nargs='+', type=str, help='(16540, D) .npy replacing the TRAINING image targets only (test / sealed scoring keep image_feature_dir): one file = fixed target (e.g. the NeuroBridge CPA prior aug_cpa4.npy, logs/aug_features.py); several = a uniformly random one per trial and step (augmented views)')
    parser.add_argument('--image_mixup_prob', default=0.0, type=float, help='probability per batch of cross-image mixup: EEG and image target both lam*x + (1-lam)*x[perm], lam ~ U(0,1) per trial (virtual images; the clean proxy scales steeply with distinct training images, NIGHT_LOG 11-10)')
    parser.add_argument('--func_transfer_prob', default=0.0, type=float, help='probability per batch of re-rendering each person through another person`s decoder-fitted spatial operator (--func_transfer_ops); takes precedence over --op_transfer_prob on that batch')
    parser.add_argument('--func_transfer_ops', default=[], nargs='+', help='files {person id: 32x32 operator} from scripts/mi/ne19_fit_operators.py')
    parser.add_argument('--op_donors', default='', help='npz with S (K, channels, channels): recording operators of people of other cohorts (scripts/mi/nq08_op_donors.py); half of the --op_transfer_prob partners are then drawn among them')
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
    parser.add_argument('--atm_e_layers', default=1, type=int, help='ATM: number of attention layers over the electrode tokens (scripts/subject_key/KEY_LOG.md, msL_atm)')
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
        if args.eeg_encoder_type.removeprefix('Ortho') not in ('TSConv_parameterizable', 'FastTSConvSqueezeformer', 'ATM', 'Mixer3', 'FlatMLP'):
            raise ValueError('--tsconv_domain_bn_boundary requires TSConv_parameterizable, FastTSConvSqueezeformer, ATM, Mixer3 or FlatMLP.')
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
    if args.sealed_test_metric:
        # Selection would otherwise read the sealed test number itself.
        assert args.select_best_on != 'test', '--sealed_test_metric needs --select_best_on val or fixed'
    return args


def setup_run(args):
    """Seed, run directory, train_config.json, logging and device. Returns (seed, log_dir, writer, device)."""
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
    with open(os.path.join(log_dir, "train_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8',
        filename=f'{log_dir}/train.log',
        filemode='w'
    )
    log('Input arguments:')
    for key, val in vars(args).items():
        log(f'{key:22} {val}')

    if args.val_subject_id is not None and args.val_subject_id in args.train_subject_ids:
        args.train_subject_ids = [sid for sid in args.train_subject_ids if sid != args.val_subject_id]
        log(f"Removed val_subject_id={args.val_subject_id} from train_subject_ids for clean validation.")
        if len(args.train_subject_ids) == 0:
            raise ValueError("After removing val_subject_id, train_subject_ids is empty.")
    with open(os.path.join(args.output_dir, "last_run.txt"), 'w') as f:
        f.write(log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')
    amp_enabled = args.amp_dtype != 'none'
    if amp_enabled and device.type != 'cuda':
        raise ValueError('--amp_dtype requires a CUDA device')
    if amp_enabled:
        torch.set_autocast_dtype('cuda', {'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.amp_dtype])
        torch.set_autocast_enabled('cuda', True)
    log(f"Automatic mixed precision: {args.amp_dtype} (contrastive features/logits forced to float32)")
    log(f'Subject mixup mode: {args.subject_mixup_mode}, type: {args.mixup_type} (alpha={args.subject_mixup_alpha})')
    log(f'Evaluation mode: {args.eval_mode}')
    return seed, log_dir, writer, device


def hold_out_val_concepts(args, train_dataset, pin_memory):
    """Honest checkpoint selection without giving up a training subject: hold out a fixed slice of
    training CONCEPTS (every image of them, across every training subject). Concept-disjoint mirrors
    the real test condition of 200 unseen concepts; an image-level split would leave concept-mates in
    training, which --multi_positive_loss is explicitly trained to pull together.
    Returns (train_main_dataset, concept_val_dataloader)."""
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
        GroupedSubset(train_dataset, main_train_indices)
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
    return train_main_dataset, concept_val_dataloader


def hold_out_probe_split(args, train_dataset, seed, pin_memory):
    """Per-subject held-out split for the linear subject probes.
    Returns (train_main_dataset, probe_train_dataloader, probe_val_dataloader)."""
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
    train_main_dataset = GroupedSubset(train_dataset, main_train_indices) if args.grouped_batch_sampler else Subset(train_dataset, main_train_indices)
    pb = 256
    probe_train_dataloader = DataLoader(
        train_main_dataset, batch_size=pb, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=pin_memory
    )
    probe_val_dataloader = DataLoader(
        Subset(train_dataset, probe_val_indices), batch_size=pb, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=pin_memory
    )
    return train_main_dataset, probe_train_dataloader, probe_val_dataloader


def build_datasets(args, seed, device):
    """Train / val / test datasets and loaders, plus the EEG and feature shapes."""
    print('\n>>> Loading Train Data <<<')
    if args.eeg_aug:
        if args.eeg_aug_type == 'noise':
            eeg_transform = RandomGaussianNoise(std=0.001)
        elif args.eeg_aug_type == 'smooth':
            eeg_transform = RandomSmooth(kernel_size=5, smooth_prob=0.3)
    else:
        eeg_transform = None

    sources = (args.eeg_data_dir, args.selected_channels, args.time_window, args.image_feature_dir, args.text_feature_dir)
    train_dataset = EEGPreImageDataset(
        args.train_subject_ids, *sources, average=args.data_average, eeg_transform=eeg_transform, train=True,
        frozen_eeg_prior=args.frozen_eeg_prior, cross_subject_average=args.cross_subject_average,
        subject_mixup_within=args.subject_mixup_within,
    )
    if len(args.train_image_targets) == 1:
        train_dataset.image_features = np.load(args.train_image_targets[0]).reshape(train_dataset.image_features.shape)
    if args.distill_lambda > 0 or args.anti_teacher_lambda > 0:
        assert not args.data_average, '--distill_targets rows are single trials'
        train_dataset.distill_targets = np.load(args.distill_targets, mmap_mode='r')
        assert len(train_dataset.distill_targets) == len(train_dataset), 'teacher rows != training trials'
    data = SimpleNamespace(
        train_dataset=train_dataset,
        eeg_sample_points=train_dataset.num_sample_points,
        image_feature_dim=train_dataset.image_features.shape[-1],
        channels_num=train_dataset.channels_num,
        pin_memory=device.type == "cuda",
        probe_train_loader=None,
        probe_val_loader=None,
    )
    data.backbone_feature_dim = args.eeg_backbone_dim if args.eeg_backbone_dim > 0 else data.image_feature_dim
    log(f'EEG sample points: {data.eeg_sample_points}')
    log(f'image feature dimension: {data.image_feature_dim}')
    log(f'EEG backbone output dimension: {data.backbone_feature_dim}')
    log(f'alignment feature dimension (--feature_dim): {args.feature_dim}')
    log(f'data length: {len(train_dataset)}')
    log(f'number of channels: {data.channels_num}')

    train_dataset.repeat_mix = args.repeat_mix
    train_main_dataset = train_dataset
    if args.holdout_images or args.holdout_concepts:
        assert not (args.data_average or train_dataset.random or args.val_concept_ratio), 'holdout indexes single trials'
        i = np.arange(len(train_dataset))
        per_subject = train_dataset.num_objects * train_dataset.num_images_per_object * train_dataset.num_repetitions
        flat = (i % per_subject) // train_dataset.num_repetitions  # object * images_per_object + image
        train_main_dataset = (GroupedSubset if args.grouped_batch_sampler else Subset)(train_dataset, i[~np.isin(flat, held_out(args))].tolist())
        log(f'held out {len(held_out(args))} training images: data length {len(train_main_dataset)}')
    data.val_loader = None
    if args.val_concept_ratio > 0.0:
        train_main_dataset, data.val_loader = hold_out_val_concepts(args, train_dataset, data.pin_memory)
    if args.subject_probe_holdout:
        train_main_dataset, data.probe_train_loader, data.probe_val_loader = hold_out_probe_split(
            args, train_dataset, seed, data.pin_memory
        )

    if args.train_test_reps:
        assert not (args.grouped_batch_sampler or args.distill_lambda or args.anti_teacher_lambda), 'test trials have no image groups or teacher rows'
        test_trials = EEGPreImageDataset(args.train_subject_ids, *sources, average=False, train=False)
        train_main_dataset = ConcatDataset([train_main_dataset, TestRepTrials(test_trials, args.train_test_reps, train_dataset.num_objects)])
        log(f'+ {args.train_test_reps} test reps per test image and subject: data length {len(train_main_dataset)}')

    if args.extra_trials_dir:
        assert not (args.grouped_batch_sampler or args.distill_lambda or args.anti_teacher_lambda), 'extra trials have no image groups or teacher rows'
        # ponytail: lag_things shifts every id >= 21 later by --things_lag, the extra people included, so their own shift is stored net of it
        extra = ExtraTrials(args.extra_trials_dir, 100, 100000, args.extra_shift - args.things_lag, train_dataset.feature_dim)
        if args.extra_shuffle_targets:
            extra.img = np.random.default_rng(0).permutation(extra.img)
        if (args.holdout_images or args.holdout_concepts) and not args.holdout_avoid:  # a cohort on THINGS images keeps only the images the proxy trains on
            o, flat = extra.obj[extra.img], extra.obj[extra.img] * train_dataset.num_images_per_object + extra.image[extra.img]
            k = (o >= 100000) | ((o < train_dataset.num_objects) & ~np.isin(flat, held_out(args)))
            extra.X, extra.img, extra.subj = extra.X[k], extra.img[k], extra.subj[k]
        train_main_dataset = ConcatDataset([train_main_dataset] + [extra] * args.extra_repeat)
        log(f'+ {len(extra)} extra trials of {len(set(extra.subj.tolist()))} people from {args.extra_trials_dir}: data length {len(train_main_dataset)}')

    if args.grouped_batch_sampler:
        batch_sampler = GroupedImageBatchSampler(
            train_main_dataset,
            batch_size=args.batch_size,
            samples_per_image=args.samples_per_image,
            drop_last=True,
            seed=seed,
        )
        data.train_loader = DataLoader(train_main_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=data.pin_memory)
        effective_batch_size = batch_sampler.images_per_batch * batch_sampler.samples_per_image
        log(
            f'Grouped batch sampler enabled: images_per_batch={batch_sampler.images_per_batch}, '
            f'samples_per_image={batch_sampler.samples_per_image}, effective_batch_size={effective_batch_size}'
        )
    elif args.repeat_agg:
        assert not (args.data_average or train_dataset.random or args.alljoined_after) and isinstance(train_main_dataset, (Subset, EEGPreImageDataset)), '--repeat_agg indexes single trials of one dataset'
        data.train_loader = DataLoader(train_main_dataset, batch_sampler=RepeatPairSampler(getattr(train_main_dataset, 'indices', range(len(train_dataset))), len(train_dataset),
                                       train_dataset.num_repetitions, args.batch_size, seed=seed), num_workers=args.num_workers, pin_memory=data.pin_memory)
    else:
        data.train_loader = DataLoader(train_main_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                       num_workers=args.num_workers, pin_memory=data.pin_memory)

    if args.val_subject_id is not None:
        print('\n>>> Loading Validation Data <<<')
        val_dataset = EEGPreImageDataset([args.val_subject_id], *sources, average=True, eeg_transform=eeg_transform,
                                         train=False, frozen_eeg_prior=args.frozen_eeg_prior)
        data.val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    print('\n>>> Loading Test Data <<<')
    test_dataset = EEGPreImageDataset(args.test_subject_ids, *sources, average=True, eeg_transform=eeg_transform,
                                      train=False, frozen_eeg_prior=args.frozen_eeg_prior)
    data.test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)
    if args.subject_key in ('demean', 'swap', 'swap_split', 'split'):
        global SUBJECT_KEYS
        SUBJECT_KEYS = torch.zeros(max(args.train_subject_ids + args.test_subject_ids) + 1, train_dataset.num_channels, train_dataset.num_sample_points, device=device)
        for ds in (train_dataset, test_dataset):  # test subject: mean over its own (unlabelled) test trials
            for sid, x in zip(ds.subject_ids, ds.eeg_data_list):
                x = np.asarray(x, dtype=np.float32).reshape(-1, *x.shape[-2:])  # single-trial caches pad missing repeats with zeros
                SUBJECT_KEYS[sid] = torch.tensor(x.sum(0) / x.reshape(len(x), -1).any(1).sum(), device=device)
    return data


def things_noise(eeg_batch, subject_id_batch, p):
    """--things_noise p: THINGS-EEG2 trials + (a - b) / sqrt 2, a and b two Alljoined trials of ONE person in the batch
    (their key cancels; single-trial noise variance, the evoked parts are small against it)."""
    aj = torch.nonzero(subject_id_batch < 21)[:, 0]
    aj = aj[torch.argsort(subject_id_batch[aj] + torch.rand(len(aj), device=aj.device))]  # grouped by person, random order inside
    same = subject_id_batch[aj[:-1]] == subject_id_batch[aj[1:]]
    th = torch.nonzero((subject_id_batch >= 21) & (torch.rand(len(subject_id_batch), device=aj.device) < p))[:, 0]
    if not same.any() or not len(th):
        return eeg_batch
    noise = (eeg_batch[aj[:-1][same]] - eeg_batch[aj[1:][same]]) / 2 ** .5
    eeg_batch = eeg_batch.clone()
    eeg_batch[th] += noise[torch.randint(len(noise), (len(th),), device=aj.device)]
    return eeg_batch


def lag_things(args, eeg_batch, subject_id_batch):
    """--things_lag n: THINGS-EEG2 trials x(t) -> x(t - n); the first sample fills the gap.
    --things_filter: a per-electrode FIR over the later shifts instead (same edge rule)."""
    later = lambda n: torch.cat([eeg_batch[..., :1].expand(*eeg_batch.shape[:-1], n), eeg_batch[..., :-n]], -1) if n else eeg_batch
    if args.things_filter:
        global THINGS_FILTER
        if THINGS_FILTER is None:
            THINGS_FILTER = torch.as_tensor(np.load(args.things_filter)).float()
        taps = THINGS_FILTER.to(eeg_batch.device, eeg_batch.dtype)
        shifted = sum(taps[i][:, None] * later(i) for i in range(len(taps)))
    elif args.things_lag > 0:
        shifted = later(args.things_lag)
    else:
        return eeg_batch
    return torch.where((subject_id_batch >= 21).view(-1, *[1] * (eeg_batch.dim() - 1)), shifted, eeg_batch)


THINGS_FILTER = None
SUBJECT_KEYS = None


def apply_subject_key(args, eeg_batch, subject_id_batch, train):
    if args.subject_key == 'keep' or (args.subject_key in ('swap', 'swap_split') and not train):
        return eeg_batch
    if args.subject_key in ('comb', 'comb_split'):  # the key is the steady-state response to the 5 Hz image stream: period = 50 samples at 250 Hz
        per = eeg_batch.unflatten(-1, (-1, 50)).mean(-2)
        content = eeg_batch - (per - per.mean(-1, keepdim=True)).repeat(1, 1, eeg_batch.shape[-1] // 50)
        return content if args.subject_key == 'comb' else torch.stack([content, eeg_batch], dim=1)  # the attention map comes from the raw trial
    if args.subject_key == 'split':
        return torch.stack([eeg_batch - SUBJECT_KEYS[subject_id_batch], SUBJECT_KEYS[subject_id_batch]], dim=1)
    raw, eeg_batch = eeg_batch, eeg_batch - SUBJECT_KEYS[subject_id_batch]
    if args.subject_key in ('swap', 'swap_split'):
        ids = torch.tensor(args.train_subject_ids, device=eeg_batch.device)
        pick = lambda pool: pool[torch.randint(len(pool), (len(eeg_batch),), device=eeg_batch.device)]
        lo, hi = ids[ids < 21], ids[ids >= 21]  # partner from the same dataset (ids >= 21 = THINGS-EEG2 exported next to Alljoined: keys 2x larger, r .09 with Alljoined keys)
        eeg_batch = eeg_batch + SUBJECT_KEYS[torch.where(subject_id_batch >= 21, pick(hi), pick(lo)) if len(lo) and len(hi) else pick(ids)]
    return torch.stack([eeg_batch, raw], dim=1) if args.subject_key == 'swap_split' else eeg_batch


def write_inference_config(args, data, log_dir):
    inference_keys = [
        'eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir',
        'projector', 'projector_activation', 'feature_dim', 'eeg_backbone_dim', 'time_window', 'selected_channels',
        'tsconv_temporal_filters', 'tsconv_temporal_kernel', 'tsconv_pool_kernel', 'tsconv_pool_stride',
        'tsconv_spatial_filters', 'tsconv_projection_filters', 'tsconv_activation', 'tsconv_dropout',
        'tsconv_head_dropout',
        'arch_kwargs', 'fm_arch_kwargs',
        'tsconv_domain_bn_boundary', 'input_band',
        'eval_mode', 'sattc_saw_shrink', 'sattc_csls_k',
    ]
    inference_config = {k: vars(args)[k] for k in inference_keys}
    inference_config['feature_dim'] = int(str(args.feature_dim).split(',')[0])
    inference_config['architecture'] = 'baseline'
    inference_config['eeg_sample_points'] = data.eeg_sample_points
    inference_config['backbone_feature_dim'] = data.backbone_feature_dim
    inference_config['image_feature_dim'] = data.image_feature_dim
    with open(os.path.join(log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)


def build_model(args, data, device):
    """Encoder, the three projectors and the loss, optionally initialised from --init_checkpoint.
    Also turns args.feature_dim into an int."""
    model = build_eeg_encoder(args, data.backbone_feature_dim, data.eeg_sample_points, data.channels_num).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'EEG Encoder trainable parameters: {num_params / 1e6:.2f}M')
    log(str(model))

    args.feature_dim = int(str(args.feature_dim).split(',')[0])
    eeg_projector = build_projector(args.projector, data.backbone_feature_dim, args.feature_dim, activation=args.projector_activation).to(device)
    img_projector = build_projector(args.projector, data.image_feature_dim, args.feature_dim, activation=args.projector_activation).to(device)
    text_projector = build_projector(args.projector, data.image_feature_dim, args.feature_dim, activation=args.projector_activation).to(device)
    projector_params = sum(p.numel() for proj in (eeg_projector, img_projector, text_projector)
                           for p in proj.parameters() if p.requires_grad)
    log(f'Projector trainable parameters: {projector_params / 1e6:.2f}M')

    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial['model_state_dict'])
        eeg_projector.load_state_dict(initial['eeg_projector_state_dict'])
        img_projector.load_state_dict(initial['img_projector_state_dict'])
        text_projector.load_state_dict(initial['text_projector_state_dict'])
        log(f'Initialized all modules from {args.init_checkpoint}')
        del initial
    nets = SimpleNamespace(model=model, eeg_projector=eeg_projector, img_projector=img_projector,
                           text_projector=text_projector, criterion=None)
    nets.aux = None
    if args.aux_targets or args.aux_concept_lambda > 0:  # the pre-head feature is caught by a hook; the heads are dropped after training
        head_in = model.head if hasattr(model, 'head') else model.proj_eeg  # ATM names its head proj_eeg
        pre = {}; head_in.register_forward_hook(lambda m, i, o: pre.__setitem__('x', i[0]))
        banks = [F.normalize(torch.as_tensor(np.load(os.path.join(d, 'image_train.npy')).astype(np.float32)).flatten(0, 1), dim=1).to(device) for d in args.aux_targets]
        width = next(m for m in head_in.modules() if isinstance(m, nn.Linear)).in_features
        if args.aux_dim:  # several spaces next to an encoder that already fills a 10 GB card: keep the banks in half precision
            banks = [b.half() for b in banks]
        nets.aux = SimpleNamespace(pre=pre, banks=banks, heads=nn.ModuleList([nn.Linear(width, args.aux_dim or b.shape[1]) for b in banks] + [nn.Linear(width, 1654)] * (args.aux_concept_lambda > 0)).to(device),
                                   img=nn.ModuleList([nn.Linear(b.shape[1], args.aux_dim) for b in banks] if args.aux_dim else []).to(device))
        log(f'auxiliary targets: {[tuple(b.shape) for b in banks]} from the {width}-d input of model.head, lambda {args.aux_lambda}')

    # Before the loss on purpose: the probe heads (and their dummy train-mode forward) draw from the torch RNG.
    probes = build_subject_probes(args, nets, data, device) if args.subject_probe_holdout else None

    nets.criterion = ContrastiveLoss(
        args.init_temperature, args.alpha, args.eeg_l2norm, args.img_l2norm, args.t_learnable, args.softplus,
        logit_scale_max=args.logit_scale_max, asymmetric=args.asymmetric_loss,
    ).to(device)
    nets.criterion.soft_label, nets.criterion.soft_tau, nets.criterion.gce_q = args.soft_label, args.soft_tau, args.gce_q
    if args.open_bank:
        nets.criterion.bank, nets.criterion.bank_soft, nets.criterion.bank_sep = F.normalize(torch.tensor(np.load(args.open_bank)).float(), dim=1).to(device), args.open_bank_soft, args.open_bank_sep
    if args.init_checkpoint:  # the learned temperature belongs to the head
        saved = torch.load(args.init_checkpoint, map_location=device, weights_only=False).get('criterion_state_dict')
        if saved is not None:
            nets.criterion.load_state_dict(saved)
            log(f'Initialized the loss (logit scale) from {args.init_checkpoint}')
    log(str(nets.criterion))
    return nets, probes


def build_subject_probes(args, nets, data, device):
    """Linear subject-ID probes on the requested layers, with their optimizers and loaders."""
    align_dim = args.feature_dim if args.projector != 'direct' else data.backbone_feature_dim
    class_map = {sid: idx for idx, sid in enumerate(sorted(args.train_subject_ids))}
    heads = nn.ModuleDict()
    # Determine probe input dimensions
    with torch.no_grad():
        dummy_eeg = torch.zeros(1, data.channels_num, data.eeg_sample_points).to(device)
        dummy_sid = torch.tensor([args.train_subject_ids[0]]).to(device)
        bb_out = run_eeg_backbone(nets.model, args, dummy_eeg, dummy_sid, return_intermediate=True)

        for layer in args.subject_probe_layers:
            if layer == 'align':
                in_dim = align_dim
            elif layer in bb_out:
                in_dim = bb_out[layer].shape[-1]
            else:
                log(f"Warning: layer '{layer}' not found in backbone output. Skipping probe.")
                continue
            heads[f'eeg_{layer}'] = nn.Linear(in_dim, len(class_map)).to(device)

    optimizers = {name: optim.Adam(head.parameters(), lr=1e-3) for name, head in heads.items()}
    return SimpleNamespace(heads=heads, optimizers=optimizers, class_map=class_map,
                           train_loader=data.probe_train_loader, val_loader=data.probe_val_loader,
                           history={'epoch': [], 'eeg_backbone_val_acc': [], 'eeg_align_val_acc': []})


def build_optimization(args, nets, train_loader, device):
    """AdamW over every trainable module (+ the temperature when learnable), grad scaler, EMA and LR schedule."""
    opt = SimpleNamespace(scheduler=None, ema_model=None)
    opt.trainable_parameters = (
        collect_trainable_parameters([nets.model, nets.eeg_projector, nets.img_projector, nets.text_projector])
        + (list(nets.aux.heads.parameters()) + list(nets.aux.img.parameters()) if nets.aux else [])
        + ([p for p in nets.criterion.parameters() if p.requires_grad] if args.t_learnable else [])
    )
    groups = [{'params': opt.trainable_parameters}]
    if args.layer_decay < 1:
        depth = lambda n: int(m.group(1)) + 1 if (m := re.search(r'^fm\.blocks\.(\d+)\.', n)) else 0 if n.startswith('fm.') and not n.startswith(('fm.norm', 'fm.temporal_embedding')) else None  # the temporal embedding is re-initialised for the 120-sample window: full lr
        by = {}
        for n, p in nets.model.named_parameters():
            if p.requires_grad and depth(n) is not None:
                by.setdefault(depth(n), []).append(p)
        low = {id(p) for ps in by.values() for p in ps}
        groups = [{'params': [p for p in opt.trainable_parameters if id(p) not in low]}] + [{'params': ps, 'lr': args.learning_rate * args.layer_decay ** (max(by) + 1 - d)} for d, ps in sorted(by.items())]
    opt.optimizer = optim.AdamW(
        groups,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    opt.grad_scaler = torch.amp.GradScaler('cuda', enabled=(args.amp_dtype == 'float16'))
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
        opt.ema_model = AveragedModel(nets.model, multi_avg_fn=ema_update, use_buffers=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    total_epochs = args.num_epochs
    if args.lr_scheduler == 'cosine':
        warmup = max(0, int(args.warmup_epochs))
        floor = args.min_lr / max(args.learning_rate, 1e-12)

        def _lr_lambda(step):
            e = step + 1  # 1-based epoch about to run (scheduler.step() is called at epoch end)
            if e <= warmup:
                return e / max(warmup, 1)
            progress = (e - warmup) / max(total_epochs - warmup, 1)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        opt.scheduler = optim.lr_scheduler.LambdaLR(opt.optimizer, _lr_lambda)
    elif args.lr_scheduler == 'onecycle':
        # Stepped per optimizer step, not per epoch -- this is the whole point of OneCycle and
        # the reason the per-epoch cosine above is not a substitute.
        opt.scheduler = optim.lr_scheduler.OneCycleLR(
            opt.optimizer,
            max_lr=[group['lr'] for group in opt.optimizer.param_groups],
            total_steps=total_epochs * len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
        )
    return opt


def set_train_mode(nets, mode):
    for module in (nets.model, nets.eeg_projector, nets.img_projector, nets.text_projector):
        module.train(mode)


def save_checkpoint(nets, args, path, epoch_idx, loss_value):
    checkpoint = {
        'epoch': epoch_idx,
        'stage': 'joint',
        'architecture': 'baseline',
        'model_state_dict': nets.model.state_dict(),
        'eeg_projector_state_dict': nets.eeg_projector.state_dict(),
        'img_projector_state_dict': nets.img_projector.state_dict(),
        'text_projector_state_dict': nets.text_projector.state_dict(),
        # No optimizer state: nothing in this repo resumes training, and AdamW's two
        # moments per parameter are 2/3 of a REVE checkpoint (840 MB -> 280 MB).
        'loss': loss_value,
    }
    if args.t_learnable:
        checkpoint['criterion_state_dict'] = nets.criterion.state_dict()
    if args.aux_dim:
        checkpoint['aux_state_dict'] = {'heads': nets.aux.heads.state_dict(), 'img': nets.aux.img.state_dict()}
    torch.save(checkpoint, path)


_PRIOR_BANK = None
_BANK = None


def holdout_images(n):
    """the --holdout_images set: n flat training-image ids (object * 10 + image), seed 0 (logs/holdout_eval.py recomputes it)"""
    return np.sort(np.random.default_rng(0).choice(16540, n, replace=False))


def held_out(args):
    """flat training-image ids dropped by --holdout_images (random images) or --holdout_concepts (all images of concepts)"""
    if args.holdout_concepts:
        return np.sort((np.random.default_rng(0).choice(1654, args.holdout_concepts, replace=False)[:, None] * 10 + np.arange(10)).ravel())
    if args.holdout_avoid:  # second proxy: the held-out images are drawn outside a cohort's image set, which stays fully trainable
        T = np.load(args.holdout_avoid); pool = np.setdiff1d(np.arange(16540), (T['obj'] * 10 + T['image'])[T['obj'] < 1654])
        return np.sort(np.random.default_rng(0).choice(pool, args.holdout_images, replace=False))
    return holdout_images(args.holdout_images)


_VIEWS = None


def random_view_targets(args, object_idx, image_idx, device):
    """--train_image_targets with several files: each trial's target = one of the views, uniformly at random"""
    global _VIEWS
    if _VIEWS is None:
        _VIEWS = torch.stack([torch.as_tensor(np.load(f).reshape(16540, -1), dtype=torch.float16) for f in args.train_image_targets]).to(device)
    flat = object_idx * 10 + image_idx
    return _VIEWS[torch.randint(len(_VIEWS), (len(flat),), device=device), flat].float()


_CONCEPT_MEAN = None


def concept_smoothed(image_feature, object_idx, args):
    """(1 - a) * target + a * its concept's mean normalised training-image embedding; test-rep trials (object >= 1654) unchanged"""
    global _CONCEPT_MEAN
    if _CONCEPT_MEAN is None:
        bank = torch.as_tensor(np.load(os.path.join(args.image_feature_dir, 'image_train.npy')), dtype=torch.float32)
        bank = F.normalize(bank, dim=-1)  # (objects, 10, D)
        keep = torch.ones(bank.shape[:2])
        if args.holdout_images or args.holdout_concepts:
            keep.view(-1)[torch.as_tensor(held_out(args))] = 0
        _CONCEPT_MEAN = ((bank * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp_min(1)).to(image_feature.device)
    a = args.concept_smooth
    seen = object_idx < len(_CONCEPT_MEAN)
    target = F.normalize(image_feature.float(), dim=1)
    mixed = (1 - a) * target[seen] + a * _CONCEPT_MEAN[object_idx[seen]]
    out = image_feature.float().clone(); out[seen] = mixed
    return out


def bank_loss(eeg_feature, object_idx, image_idx, args):
    """cross-entropy of each trial over every training-image embedding except the held-out ones, target its own image"""
    global _BANK
    if _BANK is None:
        bank = np.load(os.path.join(args.image_feature_dir, 'image_train.npy'))
        bank = F.normalize(torch.as_tensor(bank.reshape(-1, bank.shape[-1]), dtype=torch.float32, device=eeg_feature.device), dim=1)
        keep = torch.ones(len(bank), dtype=torch.bool, device=bank.device)
        if args.holdout_images or args.holdout_concepts:
            keep[torch.as_tensor(held_out(args), device=bank.device)] = False
        row = torch.full((len(bank),), -1, dtype=torch.long, device=bank.device)
        row[keep] = torch.arange(int(keep.sum()), device=bank.device)  # flat image id -> bank row
        _BANK = (bank[keep], row)
    bank, row = _BANK
    flat = object_idx * 10 + image_idx
    keep = flat < len(row)  # test-rep trials (finals) have no bank row
    with torch.autocast(device_type='cuda', enabled=False):
        logits = F.normalize(eeg_feature[keep].float(), dim=1) @ bank.T / args.bank_tau
        return F.cross_entropy(logits, row[flat[keep]])


_XBANK = None


def extra_bank_loss(eeg_feature, object_idx, image_idx, nets, args):
    """--extra_bank: cross-entropy of each extra-cohort trial over ALL the images of its cohort (the criterion's own logits), target its image.
    The in-batch loss of a small cohort has few negatives (about 20 at 15 % of a 128 batch); the cohort's whole image set gives 1,900 (ds007964)."""
    global _XBANK
    if _XBANK is None:
        T = np.load(os.path.join(args.extra_trials_dir, 'targets.npz'))
        key = torch.as_tensor(T['obj'] * 10 + T['image'] if 'obj' in T else (100000 + np.arange(len(T['F']))) * 10, device=eeg_feature.device)
        order = key.argsort(); _XBANK = (torch.as_tensor(T['F'], dtype=torch.float32, device=eeg_feature.device), key[order], order)
    bank, key, order = _XBANK
    with torch.autocast(device_type='cuda', enabled=False):
        e, b = nets.criterion._normalize_inputs(eeg_feature.float(), nets.img_projector(bank).float())
        return F.cross_entropy(e @ b.T * nets.criterion._get_logit_scale(), order[torch.searchsorted(key, object_idx * 10 + image_idx)])


def output_prior_loss(eeg_feature, args):
    """KL of the batch-mean softmax over the training-image bank to uniform: the output prior (which images the model
    tends to predict; each architecture's error family, NIGHT_LOG N65) pushed toward equipartition, the training
    targets' own marginal."""
    global _PRIOR_BANK
    if _PRIOR_BANK is None:
        bank = np.load(os.path.join(args.image_feature_dir, 'image_train.npy'))
        _PRIOR_BANK = F.normalize(torch.as_tensor(bank.reshape(-1, bank.shape[-1]), dtype=torch.float32, device=eeg_feature.device), dim=1)
    with torch.autocast(device_type='cuda', enabled=False):
        s = F.normalize(eeg_feature.float(), dim=1) @ _PRIOR_BANK.T
        q = ((s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True).clamp_min(1e-6) / args.prior_tau).softmax(1).mean(0)
        return (q * (q * len(q)).clamp_min(1e-12).log()).sum()


def warp_mask(x, warp, mask, onset=24):
    """--time_warp a: x(t) -> x(onset + (t - onset) / s), s ~ U(1 - a, 1 + a) per trial (linear interpolation, edges replicated);
    --time_mask n: n consecutive samples zeroed per trial."""
    T = x.shape[-1]; t = torch.arange(T, device=x.device); v = lambda i: i.view(len(i), *[1] * (x.dim() - 2), -1)
    if warp > 0:
        s = 1 + warp * (2 * torch.rand(len(x), 1, device=x.device) - 1)
        p = (onset + (t - onset) / s).clamp(0, T - 1); lo = p.floor().long().clamp(max=T - 2); w = v(p - lo).to(x.dtype)
        x = x.gather(-1, v(lo).expand_as(x)) * (1 - w) + x.gather(-1, v(lo + 1).expand_as(x)) * w
    if mask > 0:
        a = torch.randint(0, T - mask + 1, (len(x), 1), device=x.device)
        x = x * v((t < a) | (t >= a + mask)).to(x.dtype)
    return x


def person_centre(eeg_feature_batch, subject_id_batch):
    """--person_centre_train: x - mean(same-person batch-mates) + batch mean (mirror of score_track1.batch_transform_wknn with true ids)."""
    _, inv = subject_id_batch.unique(return_inverse=True)
    n = torch.bincount(inv).float()[inv, None]
    sums = torch.zeros(int(inv.max()) + 1, eeg_feature_batch.shape[1], device=eeg_feature_batch.device, dtype=eeg_feature_batch.dtype).index_add_(0, inv, eeg_feature_batch)
    mates = (sums[inv] - eeg_feature_batch) / (n - 1).clamp_min(1)
    return torch.where(n > 1, eeg_feature_batch - mates, eeg_feature_batch) + eeg_feature_batch.mean(0, keepdim=True)


def kd_loss(nets, args, eeg_feature, image_feature, subject_ids, flat, mixup_metadata, member=None):
    """KL(softmax(teacher cosine / tau) || softmax(student cosine / tau)) over the batch images; a SubjectMix pair gets the same mix of its two trials' teacher cosines."""
    teacher_cos = nets.teachers.cosine(subject_ids, flat, member)
    if mixup_metadata is not None:
        teacher_cos = mixup_metadata['lam'][:, None] * teacher_cos + (1 - mixup_metadata['lam'][:, None]) * teacher_cos[mixup_metadata['partner_indices']]
    with torch.autocast(device_type='cuda', enabled=False):
        student_logits = F.normalize(eeg_feature.float(), dim=1) @ F.normalize(image_feature.float(), dim=1).T / args.kd_tau
        return args.kd_lambda * F.kl_div(student_logits.log_softmax(dim=1), (teacher_cos / args.kd_tau).softmax(dim=1), reduction='batchmean')


def train_one_epoch(nets, opt, args, device, train_loader, writer, epoch):
    total_epochs = args.num_epochs
    set_train_mode(nets, True)
    nets.criterion.train()
    total_loss = 0.0
    total_cl_loss = 0.0
    total_subject_mixup_reg_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
        eeg_batch = batch[0].to(device)
        image_feature_batch = batch[1].to(device)
        subject_id_batch = batch[3].to(device)
        object_idx_batch = batch[4].to(device)
        image_idx_batch = batch[5].to(device)
        real = batch[0].flatten(1).abs().amax(1).to(device) > 0  # --repeat_agg: a missing repeat is zero padding
        if args.alljoined_after and epoch > args.alljoined_after:
            keep = subject_id_batch < 21
            eeg_batch, image_feature_batch, subject_id_batch, object_idx_batch, image_idx_batch = (v[keep] for v in (eeg_batch, image_feature_batch, subject_id_batch, object_idx_batch, image_idx_batch))
        if len(args.train_image_targets) > 1:
            image_feature_batch = random_view_targets(args, object_idx_batch, image_idx_batch, device).to(image_feature_batch.dtype)

        opt.optimizer.zero_grad()
        eeg_batch = lag_things(args, eeg_batch, subject_id_batch)
        if args.things_noise > 0:
            eeg_batch = things_noise(eeg_batch, subject_id_batch, args.things_noise)
        if args.time_jitter > 0:
            t = torch.arange(eeg_batch.shape[-1], device=device) - torch.randint(-args.time_jitter, args.time_jitter + 1, (len(eeg_batch), 1), device=device)
            eeg_batch = eeg_batch.gather(-1, t.clamp(0, eeg_batch.shape[-1] - 1).view(len(eeg_batch), *[1] * (eeg_batch.dim() - 2), -1).expand_as(eeg_batch))
        if args.time_warp > 0 or args.time_mask > 0:
            eeg_batch = warp_mask(eeg_batch, args.time_warp, args.time_mask)
        eeg_batch = apply_subject_key(args, eeg_batch, subject_id_batch, train=True)
        if args.func_transfer_prob > 0 and random.random() < args.func_transfer_prob:
            eeg_batch = functional_operator_transfer(eeg_batch, subject_id_batch, args.func_transfer_ops)
        elif args.op_transfer_prob > 0 and random.random() < args.op_transfer_prob:
            own = subject_id_batch < 100  # --extra_trials_dir people pass through: the operators are estimated per batch and per person, and with them a person has too few trials per batch
            eeg_batch = eeg_batch.clone()
            eeg_batch[own] = cross_subject_operator_transfer(
                eeg_batch[own], subject_id_batch[own],
                boundary=21 if any(s >= 21 for s in args.train_subject_ids) and min(args.train_subject_ids) < 21 else 0,
                donors=args.op_donors,
            )
        if args.image_mixup_prob > 0 and random.random() < args.image_mixup_prob:
            lam = torch.rand(len(eeg_batch), 1, device=device)
            perm = torch.randperm(len(eeg_batch), device=device)
            eeg_batch = lam.view(-1, *[1] * (eeg_batch.dim() - 1)) * eeg_batch + (1 - lam.view(-1, *[1] * (eeg_batch.dim() - 1))) * eeg_batch[perm]
            image_feature_batch = lam * image_feature_batch + (1 - lam) * image_feature_batch[perm]
        original_eeg_batch = eeg_batch
        mixup_metadata = None
        if args.subject_mixup_mode == 'raw_eeg':
            mixup_out = cross_subject_stimulus_mix(
                eeg_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type,
                return_mixup_metadata=args.subject_mixup_reg_lambda > 0 or nets.teachers is not None,
            )
            if args.subject_mixup_reg_lambda > 0 or nets.teachers is not None:
                eeg_batch, mixup_metadata = mixup_out
            else:
                eeg_batch = mixup_out
        if args.backbone_checkpoint_chunks > 1:
            eeg_backbone_batch = torch.cat([
                torch.utils.checkpoint.checkpoint(run_eeg_backbone, nets.model, args, x, sid, use_reentrant=False)
                for x, sid in zip(eeg_batch.chunk(args.backbone_checkpoint_chunks), subject_id_batch.chunk(args.backbone_checkpoint_chunks))])
        else:
            eeg_backbone_batch = run_eeg_backbone(nets.model, args, eeg_batch, subject_id_batch)
        if args.subject_mixup_mode == 'embedding':
            eeg_backbone_batch = cross_subject_stimulus_mix(
                eeg_backbone_batch, object_idx_batch, image_idx_batch, subject_id_batch,
                alpha=args.subject_mixup_alpha, mixup_type=args.mixup_type,
            )
        eeg_feature_batch = nets.eeg_projector(eeg_backbone_batch)
        if args.person_centre_train:
            eeg_feature_batch = person_centre(eeg_feature_batch, subject_id_batch)

        loss = eeg_batch.new_tensor(0.0)

        if args.subject_mixup_reg_lambda > 0:
            original_eeg_feature_batch = run_eeg_backbone(nets.model, args, original_eeg_batch, subject_id_batch)
            original_eeg_feature_batch = nets.eeg_projector(original_eeg_feature_batch)
            subject_mixup_reg_loss = compute_subject_mixup_regularization(
                eeg_feature_batch, original_eeg_feature_batch,
                mixup_metadata['partner_indices'], mixup_metadata['mixed_mask'],
            )
            loss = loss + args.subject_mixup_reg_lambda * subject_mixup_reg_loss
            total_subject_mixup_reg_loss += subject_mixup_reg_loss.item()
        if args.concept_smooth > 0:
            image_feature_batch = concept_smoothed(image_feature_batch, object_idx_batch, args)
        image_feature_proj = nets.img_projector(image_feature_batch)
        positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
        if args.group_dro_rho > 0 or args.js_groups > 1 or args.atypical_weight > 0:
            # per-anchor terms of the (asymmetric, single-positive) EEG->image loss, for per-person aggregation
            assert args.asymmetric_loss and not args.multi_positive_loss and args.alpha == 1.0
            with torch.autocast(device_type='cuda', enabled=False):
                eeg_n, img_n = nets.criterion._normalize_inputs(eeg_feature_batch.float(), image_feature_proj.float())
                rows = F.cross_entropy(eeg_n @ img_n.T * nets.criterion._get_logit_scale(), torch.arange(len(eeg_n), device=device), reduction='none')
            if args.group_dro_rho > 0:
                _, inv = subject_id_batch.unique(return_inverse=True)
                person_loss = torch.zeros(int(inv.max()) + 1, device=device).index_add_(0, inv, rows) / torch.bincount(inv)
                cl_loss = (torch.logsumexp(args.group_dro_rho * person_loss, 0) - math.log(len(person_loss))) / args.group_dro_rho
            elif args.atypical_weight > 0:
                # weight every anchor by how atypical its image is (low cosine to the batch-mean target): the sealed hits are atypical images (E01)
                c = F.normalize(image_feature_batch.float(), dim=1) @ F.normalize(image_feature_batch.float().mean(0), dim=0)
                w = torch.exp(-args.atypical_weight * (c - c.mean()) / c.std())
                cl_loss = (rows * w / w.mean()).mean()
            else:
                cl_loss = rows.mean()
        elif args.extra_trials_dir and not args.extra_joint_loss:
            # each cohort is its own retrieval problem: no negatives across cohorts. In one loss the model reads the cohort from the EEG and adds an output offset
            # toward that cohort's images (MSS: another image set; ds007964: 1,900 THINGS images among ours), which costs retrieval for our people (MI_NEURIPS_LOG Q04, Q09)
            ex = subject_id_batch >= 100
            terms = [compute_cross_modal_loss(nets.criterion, eeg_feature_batch[m], image_feature_proj[m], positive_mask[m][:, m], args.multi_positive_loss) * m.sum()
                     for m in ((~ex,) if args.extra_bank else (ex, ~ex)) if m.sum() > 1]
            if args.extra_bank and ex.any():
                terms.append(extra_bank_loss(eeg_feature_batch[ex], object_idx_batch[ex], image_idx_batch[ex], nets, args) * ex.sum())
            cl_loss = sum(terms) / len(ex)
        elif args.repeat_agg:  # first half = anchors, second half = another repeat of each (RepeatPairSampler): the same image twice is a false negative in one loss
            h = len(eeg_feature_batch) // 2; A, P = slice(0, h), slice(h, 2 * h); ok = real[A] & real[P]
            cl_loss = sum(compute_cross_modal_loss(nets.criterion, eeg_feature_batch[s], image_feature_proj[s], positive_mask[s, s], args.multi_positive_loss) for s in (A, P)) / 2
            mean = (eeg_feature_batch[A] + eeg_feature_batch[P]) / 2  # raw mean, as the grader averages the repeats of a (person, image)
            cl_loss = cl_loss + args.repeat_agg * compute_cross_modal_loss(nets.criterion, mean[ok], image_feature_proj[A][ok], positive_mask[A, A][ok][:, ok], args.multi_positive_loss)
        else:
            cl_loss = compute_cross_modal_loss(
                nets.criterion, eeg_feature_batch, image_feature_proj, positive_mask, args.multi_positive_loss,
            )
        loss = loss + cl_loss
        total_cl_loss += cl_loss.item()
        if nets.aux:
            flat = object_idx_batch * 10 + image_idx_batch; ok = (flat < 16540) & (subject_id_batch < 100)  # training images of our people only
            for k, (head, bank) in enumerate(zip(nets.aux.heads, nets.aux.banks)):
                target = nets.aux.img[k](bank[flat[ok]].float()) if args.aux_dim else bank[flat[ok]]
                head_out = head(nets.aux.pre['x'][ok])
                loss = loss + args.aux_lambda * compute_cross_modal_loss(nets.criterion, head_out, target, positive_mask[ok][:, ok], args.multi_positive_loss)
                if args.kd_members:  # THINGS-EEG2 batches only (ok is all True there)
                    loss = loss + kd_loss(nets, args, head_out, target, subject_id_batch, flat, mixup_metadata, args.kd_members[k + 1])
            if args.aux_concept_lambda > 0:
                loss = loss + args.aux_concept_lambda * F.cross_entropy(nets.aux.heads[-1](nets.aux.pre['x'][ok]).float(), object_idx_batch[ok])
        if args.xperson_lambda > 0:
            pred, target = next(m.aux for m in nets.model.modules() if hasattr(m, 'aux'))
            other = positive_mask & (subject_id_batch[:, None] != subject_id_batch[None])  # same image, another person
            has = other.any(1)
            partner = torch.multinomial(other[has].float(), 1)[:, 0]
            xp_loss = (1 - F.cosine_similarity(pred[has].float(), target.detach()[partner].float())).mean()
            loss = loss + args.xperson_lambda * xp_loss
        if args.bank_lambda > 0:
            loss = loss + args.bank_lambda * bank_loss(eeg_feature_batch, object_idx_batch, image_idx_batch, args)
        if args.prior_lambda > 0:
            loss = loss + args.prior_lambda * output_prior_loss(eeg_feature_batch, args)
        if nets.teachers is not None:
            loss = loss + kd_loss(nets, args, eeg_feature_batch, image_feature_proj, subject_id_batch, object_idx_batch * 10 + image_idx_batch, mixup_metadata, args.kd_members[0] if args.kd_members else None)
        if args.distill_lambda > 0:
            # ensemble distillation: match the teacher's softmax over this batch's images
            with torch.autocast(device_type='cuda', enabled=False):
                teacher = F.normalize(batch[7].to(device).float(), dim=1)
                images = F.normalize(image_feature_proj.float(), dim=1)
                student_logits = F.normalize(eeg_feature_batch.float(), dim=1) @ images.T / args.distill_tau
                teacher_probs = (teacher @ images.T / args.distill_tau).softmax(dim=1)
                loss = loss + args.distill_lambda * F.kl_div(student_logits.log_softmax(dim=1), teacher_probs, reduction='batchmean')

        if args.anti_teacher_lambda > 0:
            # shared-confusion penalty: probability that student AND teacher put on the same WRONG batch image
            with torch.autocast(device_type='cuda', enabled=False):
                teacher = F.normalize(batch[7].to(device).float(), dim=1)
                images = F.normalize(image_feature_proj.float(), dim=1)
                p_s = (F.normalize(eeg_feature_batch.float(), dim=1) @ images.T / args.distill_tau).softmax(dim=1)
                p_t = (teacher @ images.T / args.distill_tau).softmax(dim=1)
                loss = loss + args.anti_teacher_lambda * (p_s * p_t * ~positive_mask).sum(1).mean()

        if args.js_groups > 1:
            # ponytail: K extra backward passes through one graph (~K x backward cost)
            assert not args.subject_mixup_reg_lambda and not args.distill_lambda, 'js_groups shrinks the contrastive gradient only'
            params = [p for p in opt.trainable_parameters if p.requires_grad]
            group = subject_id_batch % args.js_groups
            grads = [torch.autograd.grad(opt.grad_scaler.scale(rows[group == d].mean()), params, retain_graph=True, allow_unused=True)
                     for d in group.unique()]
            for j, p in enumerate(params):
                if all(g[j] is None for g in grads):
                    continue
                g = torch.stack([gd[j] if gd[j] is not None else torch.zeros_like(p) for gd in grads])
                mean = g.mean(0)
                p.grad = mean * (1 - g.var(0) / (len(g) * mean.square()).clamp_min(1e-30)).clamp_min(0)
        else:
            opt.grad_scaler.scale(loss).backward()
        if args.grad_clip > 0:
            opt.grad_scaler.unscale_(opt.optimizer)
            torch.nn.utils.clip_grad_norm_(opt.trainable_parameters, args.grad_clip)
        opt.grad_scaler.step(opt.optimizer)
        opt.grad_scaler.update()
        if opt.ema_model is not None:
            opt.ema_model.update_parameters(nets.model)
        if args.lr_scheduler == 'onecycle':
            opt.scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Loss/train_cl', total_cl_loss / len(train_loader), epoch)
    writer.add_scalar('Loss/train_subject_mixup_reg', total_subject_mixup_reg_loss / len(train_loader), epoch)
    log(
        f"Epoch [{epoch}/{total_epochs}] TrainLoss={avg_loss:.4f} "
        f"CL={total_cl_loss / len(train_loader):.4f} "
        f"MIXREG={total_subject_mixup_reg_loss / len(train_loader):.4f}"
    )
    return avg_loss


def _sparsity_summary(eeg_sp, img_sp):
    return {
        'eeg_l0_mean': eeg_sp['l0_mean'],
        'eeg_l0_frac': eeg_sp['l0_frac'],
        'eeg_active_feat_frac': eeg_sp['active_feat_frac'],
        'img_l0_mean': img_sp['l0_mean'],
        'img_l0_frac': img_sp['l0_frac'],
        'img_active_feat_frac': img_sp['active_feat_frac'],
    }


def evaluate_split(nets, args, device, split_loader, split_name):
    """(loss, top1 %, top5 %, sparsity) of retrieval on one split; NaNs when the loader is None."""
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
            eeg_batch = lag_things(args, eeg_batch, subject_id_batch)
            eeg_batch = apply_subject_key(args, eeg_batch, subject_id_batch, train=False)
            eeg_backbone_batch = run_eeg_backbone(nets.model, args, eeg_batch, subject_id_batch)
            eeg_feature_batch = nets.eeg_projector(eeg_backbone_batch)
            if args.person_centre_train:  # scored as trained (a one-person batch, i.e. the LOSO test subject, is left as it is: scripts/subject_key K03 calibrates it)
                eeg_feature_batch = person_centre(eeg_feature_batch, subject_id_batch)
            image_feature_proj = nets.img_projector(image_feature_batch)

            positive_mask = build_image_positive_mask(object_idx_batch, image_idx_batch)
            loss = compute_cross_modal_loss(
                nets.criterion, eeg_feature_batch, image_feature_proj, positive_mask, args.multi_positive_loss
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
        eeg_feature_all, gallery, subject_ids=subject_all, eval_mode=args.eval_mode,
        sattc_params={'saw_shrink': args.sattc_saw_shrink, 'csls_k': args.sattc_csls_k}, target_indices=targets,
    )
    top5_acc_local = top5_count / total * 100
    top1_acc_local = top1_count / total * 100
    sparsity = {}
    if args.projector_activation == 'relu':
        eeg_sp = compute_alignment_sparsity_stats(eeg_feature_all)
        img_sp = compute_alignment_sparsity_stats(image_feature_all)
        sparsity = _sparsity_summary(eeg_sp, img_sp)
        log(
            f"{split_name} sparsity: eeg_l0_frac={eeg_sp['l0_frac']:.4f} "
            f"eeg_active_feat_frac={eeg_sp['active_feat_frac']:.4f} "
            f"img_l0_frac={img_sp['l0_frac']:.4f}"
        )
    log(
        f"{split_name}: mode={args.eval_mode} top5 acc {top5_acc_local:.2f}%\ttop1 acc {top1_acc_local:.2f}%\tLoss: {avg_loss_local:.4f}"
    )
    return avg_loss_local, top1_acc_local, top5_acc_local, sparsity


def collect_train_sparsity(nets, args, device, data, writer, epoch):
    """Post-ReLU alignment sparsity over the full train set (no loss / retrieval; O(dim) memory)."""
    loader = DataLoader(data.train_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers,
                        pin_memory=data.pin_memory)
    eeg_acc = empty_alignment_sparsity_accumulator()
    img_acc = empty_alignment_sparsity_accumulator()
    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            subject_id_batch = batch[3].to(device)

            eeg_backbone_batch = run_eeg_backbone(nets.model, args, eeg_batch, subject_id_batch)
            update_alignment_sparsity_accumulator(eeg_acc, nets.eeg_projector(eeg_backbone_batch).cpu().numpy())
            update_alignment_sparsity_accumulator(img_acc, nets.img_projector(image_feature_batch).cpu().numpy())

    eeg_sp = finalize_alignment_sparsity_accumulator(eeg_acc)
    img_sp = finalize_alignment_sparsity_accumulator(img_acc)
    log(
        f"train sparsity: eeg_l0_frac={eeg_sp['l0_frac']:.4f} "
        f"eeg_active_feat_frac={eeg_sp['active_feat_frac']:.4f} "
        f"img_l0_frac={img_sp['l0_frac']:.4f} "
        f"img_active_feat_frac={img_sp['active_feat_frac']:.4f}"
    )
    writer.add_scalar('Sparsity/eeg_active_feat_frac_train_last', eeg_sp['active_feat_frac'], epoch)
    writer.add_scalar('Sparsity/img_active_feat_frac_train_last', img_sp['active_feat_frac'], epoch)
    return _sparsity_summary(eeg_sp, img_sp)


def sealed_test_scores(nets, args, device, log_dir, epoch, sealed_best):
    """(top1, top5) in % under the Track-1 sealed rule, on the current (EMA) weights."""
    from neurips_challenge.score_track1 import score_modules
    start = time.time()
    r = score_modules(nets.model, nets.eeg_projector, nets.img_projector, args, args.test_subject_ids, device, 512, 0,
                      fraction=args.sealed_test_fraction)
    if args.save_weights and 100 * r['subject_agg']['top5'] > sealed_best['top5']:
        sealed_best['top5'] = 100 * r['subject_agg']['top5']
        save_checkpoint(nets, args, f"{log_dir}/checkpoint_sealed_best.pth", epoch, -sealed_best['top5'])
    s = r['subject_agg']
    log(f"sealed-rule test top5 {100 * s['top5']:.2f} top1 {100 * s['top1']:.2f} "
        f"(fraction {args.sealed_test_fraction}, {time.time() - start:.0f}s)")
    # per-subject curve: neurips_challenge/cv4_summary.py picks each recipe's epoch budget from it
    with open(os.path.join(log_dir, 'sealed_per_epoch.jsonl'), 'a') as handle:
        handle.write(json.dumps({'epoch': epoch, 'fraction': args.sealed_test_fraction,
                                 'by_subject': {k: 100 * v['top5'] for k, v in r['subject_agg_by_subject'].items()}}) + '\n')
    return 100 * s['top1'], 100 * s['top5']


def run_subject_probes(nets, args, device, probes, writer, epoch):
    """Train the subject probes one pass on frozen features, score them on the probe split, log to history."""
    def encode_probe_labels(subject_ids):
        ids = [int(x) for x in subject_ids.detach().cpu().tolist()]
        return torch.tensor([probes.class_map[s] for s in ids], dtype=torch.long, device=subject_ids.device)

    def probe_features(probe_batch):
        eeg_p = probe_batch[0].to(device)
        sid_p = probe_batch[3].to(device)
        bb_res = run_eeg_backbone(nets.model, args, eeg_p, sid_p, return_intermediate=True)
        if not isinstance(bb_res, dict):
            bb_res = {'backbone': bb_res}
        # Some encoders expose a raw conv flatten as `backbone` and the
        # projector input as `output`. Use the latter when available.
        bb_res['align'] = nets.eeg_projector(bb_res.get('output', bb_res['backbone']))
        return bb_res, encode_probe_labels(sid_p)

    for h in probes.heads.values():
        h.train()
    probe_train_loss_sum = {k: 0.0 for k in probes.heads}
    probe_train_batches = 0
    for probe_batch in probes.train_loader:
        with torch.no_grad():
            bb_res, y = probe_features(probe_batch)
        for name, head in probes.heads.items():
            probes.optimizers[name].zero_grad()
            pls = F.cross_entropy(head(bb_res[name.replace('eeg_', '')].detach()), y)
            pls.backward()
            probes.optimizers[name].step()
            probe_train_loss_sum[name] += float(pls.item())
        probe_train_batches += 1

    for h in probes.heads.values():
        h.eval()
    probe_val_acc_sum = {k: 0.0 for k in probes.heads}
    probe_val_batches = 0
    with torch.no_grad():
        for probe_batch in probes.val_loader:
            bb_res, y = probe_features(probe_batch)
            for name, head in probes.heads.items():
                pred = torch.argmax(head(bb_res[name.replace('eeg_', '')]), dim=1)
                probe_val_acc_sum[name] += float((pred == y).float().mean().item())
            probe_val_batches += 1
    if probe_train_batches > 0 and probe_val_batches > 0:
        parts = []
        for name in probes.heads:
            tloss = probe_train_loss_sum[name] / probe_train_batches
            vacc = probe_val_acc_sum[name] / probe_val_batches
            writer.add_scalar(f'SubjectProbe/{name}_val_acc', vacc, epoch)
            writer.add_scalar(f'SubjectProbe/{name}_train_loss', tloss, epoch)
            parts.append(f"{name}:val_acc={vacc:.3f}")
        log("SubjProbe " + " | ".join(parts))

    probes.history['epoch'].append(epoch)
    for name in probes.heads:
        probes.history.setdefault(f'{name}_val_acc', []).append((probe_val_acc_sum[name] / probe_val_batches) * 100.0)


def write_epoch_metrics(history, log_dir):
    # Persist each epoch, including interrupted runs. top1_acc is test top-1.
    metrics_path = os.path.join(log_dir, 'epoch_metrics.csv')
    pd.DataFrame(history).to_csv(metrics_path + '.tmp', index=False)
    os.replace(metrics_path + '.tmp', metrics_path)


def save_outputs(args, device, nets, log_dir, history, probes, best, last, last_epoch_train_sparsity):
    """Training plot, probe metrics, result.csv and the final summary line."""
    save_training_plot(history, os.path.join(log_dir, 'training_metrics.png'))
    if probes is not None:
        probe_history_path = os.path.join(log_dir, 'probe_metrics.csv')
        pd.DataFrame(probes.history).to_csv(probe_history_path, index=False)
        probe_plot_path = os.path.join(log_dir, 'probe_metrics.png')
        save_probe_plot(probes.history, probe_plot_path)
        log(f"Saved probe metrics CSV: {probe_history_path}")
        log(f"Saved probe metrics plot: {probe_plot_path}")

    result_dict = {
        'architecture': 'baseline',
        'eval_mode': args.eval_mode,
        'top1 acc': f'{last.top1:.3f}',
        'top5 acc': f'{last.top5:.3f}',
        'best top1 acc': f'{best.top1:.3f}',
        'best top5 acc': f'{best.top5:.3f}',
        'best test loss': f'{best.loss:.3f}',
        'best epoch': best.epoch,
    }
    if device.type == 'cuda':
        result_dict.update({
            'peak cuda allocated gb': f'{torch.cuda.max_memory_allocated(device) / 2**30:.3f}',
            'peak cuda reserved gb': f'{torch.cuda.max_memory_reserved(device) / 2**30:.3f}',
        })
    if args.projector_activation == 'relu' and best.sparsity:
        result_dict.update({key: f'{value:.3f}' for key, value in best.sparsity.items()})
    if args.projector_activation == 'relu' and last_epoch_train_sparsity:
        result_dict.update({
            f'{key}_train_last': f'{last_epoch_train_sparsity[key]:.3f}'
            for key in ('eeg_active_feat_frac', 'img_active_feat_frac', 'eeg_l0_frac', 'img_l0_frac')
        })
    if args.t_learnable:
        with torch.no_grad():
            result_dict['learned_logit_scale'] = f"{nets.criterion._get_logit_scale().item():.3f}"
    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(log_dir, 'result.csv'), index=False)
    log(f'best test loss: {best.loss:.4f} top5 acc: {best.top5:.2f} top1 acc: {best.top1:.2f} at epoch {best.epoch}')


def evaluate_init_checkpoint(nets, args, device, data, log_dir, best):
    """Score the --init_checkpoint weights as the epoch-0 best candidate."""
    set_train_mode(nets, False)
    initial_test_loss, initial_top1, initial_top5, initial_sparsity = evaluate_split(
        nets, args, device, data.test_loader, 'test-init'
    )
    initial_selected_loss = initial_test_loss
    if args.select_best_on == 'val' and data.val_loader is not None:
        initial_selected_loss, _, _, _ = evaluate_split(nets, args, device, data.val_loader, 'val-init')
    best.loss = initial_selected_loss
    best.top1 = initial_top1
    best.top5 = initial_top5
    best.epoch = 0
    best.sparsity = dict(initial_sparsity)
    if args.save_weights:
        save_checkpoint(nets, args, f"{log_dir}/checkpoint_test_best.pth", 0, initial_selected_loss)
    log(
        f'Initialized best candidate at epoch 0: loss={initial_selected_loss:.4f} '
        f'top1={initial_top1:.2f} top5={initial_top5:.2f}'
    )


def main(args):
    seed, log_dir, writer, device = setup_run(args)
    data = build_datasets(args, seed, device)
    write_inference_config(args, data, log_dir)
    nets, probes = build_model(args, data, device)
    nets.teachers = Teachers(args.kd_teacher, args, device) if args.kd_teacher else None
    opt = build_optimization(args, nets, data.train_loader, device)
    total_epochs = args.num_epochs

    best = SimpleNamespace(top1=0.0, top5=0.0, loss=float('inf'), epoch=0, sparsity={}, testsel_loss=float('inf'))
    last = SimpleNamespace(top1=0.0, top5=0.0)  # the final epoch's test top-1/top-5 (result.csv)
    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'top1_acc': [],
               'test_top5_acc': [], 'val_loss': [], 'val_top1_acc': [], 'val_top5_acc': []}
    if args.sealed_test_metric:
        history.update(avg_eeg_top1=[], avg_eeg_top5=[])
    sealed_best = {'top5': float('-inf')}
    last_epoch_train_sparsity = {}
    if args.init_checkpoint:
        evaluate_init_checkpoint(nets, args, device, data, log_dir, best)

    epochs_without_selection_improvement = 0
    early_stop_reference_loss = best.loss
    for epoch in range(1, total_epochs + 1):
        # Autocast is switched on globally (set_autocast_enabled), so no autocast region ever closes to flush its
        # weight-cast cache: the epoch-end evaluation's no-grad casts survived into the next epoch's first forward,
        # which then had no grad_fn when only the head trained (frozen-backbone probe, 2026-10-09).
        torch.clear_autocast_cache()
        if opt.ema_model is not None and epoch > 1:
            nets.model.load_state_dict(raw_state)
        if args.soft_until:
            nets.criterion.soft_label = args.soft_label if epoch <= args.soft_until else 0.0
        avg_loss = train_one_epoch(nets, opt, args, device, data.train_loader, writer, epoch)

        if opt.ema_model is not None:
            # Everything below -- test/val metrics, selection, checkpoints -- sees the EMA weights; the raw weights are restored before training.
            raw_state = copy.deepcopy(nets.model.state_dict())
            nets.model.load_state_dict(opt.ema_model.module.state_dict())
        set_train_mode(nets, False)

        avg_test_loss, last.top1, last.top5, test_sparsity = evaluate_split(nets, args, device, data.test_loader, "test")
        if args.sealed_test_metric:
            # Challenge runs only: test top-1/top-5 (and result.csv's best ones) follow the sealed
            # rule via the grader's own pass on this run's test data; the loss stays as above.
            history['avg_eeg_top1'].append(last.top1)
            history['avg_eeg_top5'].append(last.top5)
            last.top1 = last.top5 = float('nan')
            if epoch % args.sealed_test_every == 0 or epoch == total_epochs:
                last.top1, last.top5 = sealed_test_scores(nets, args, device, log_dir, epoch, sealed_best)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Acc/top1_test', last.top1, epoch)
        writer.add_scalar('Acc/top5_test', last.top5, epoch)
        if test_sparsity:
            writer.add_scalar('Sparsity/eeg_l0_frac_test', test_sparsity['eeg_l0_frac'], epoch)
            writer.add_scalar('Sparsity/img_l0_frac_test', test_sparsity['img_l0_frac'], epoch)
        if epoch == total_epochs and args.projector_activation == 'relu':
            last_epoch_train_sparsity = collect_train_sparsity(nets, args, device, data, writer, epoch)

        val_loss, val_top1, val_top5 = float('nan'), float('nan'), float('nan')
        if data.val_loader is not None:
            val_loss, val_top1, val_top5, _ = evaluate_split(nets, args, device, data.val_loader, "val")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/top1_val', val_top1, epoch)
            writer.add_scalar('Acc/top5_val', val_top5, epoch)

        if probes is not None and epoch % args.subject_probe_interval == 0:
            run_subject_probes(nets, args, device, probes, writer, epoch)

        for key, value in [('epoch', epoch), ('train_loss', avg_loss), ('test_loss', avg_test_loss),
                           ('top1_acc', last.top1), ('test_top5_acc', last.top5), ('val_loss', val_loss),
                           ('val_top1_acc', val_top1), ('val_top5_acc', val_top5)]:
            history[key].append(value)
        write_epoch_metrics(history, log_dir)

        selected_loss, selected_top1, selected_top5 = avg_test_loss, last.top1, last.top5
        if args.select_best_on == 'val' and not np.isnan(val_loss):
            selected_loss, selected_top1, selected_top5 = val_loss, val_top1, val_top5
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
            is_better = selected_loss < best.loss or (selected_loss == best.loss and selected_top1 > best.top1)

        if is_better and args.sealed_test_metric and np.isnan(last.top1):
            # an off-epoch became the best: score it so result.csv's best numbers are never NaN
            last.top1, last.top5 = sealed_test_scores(nets, args, device, log_dir, epoch, sealed_best)
            history['top1_acc'][-1], history['test_top5_acc'][-1] = last.top1, last.top5
            write_epoch_metrics(history, log_dir)
        if is_better:
            best.loss, best.top5, best.top1, best.epoch = selected_loss, last.top5, last.top1, epoch
            best.sparsity = dict(test_sparsity)
            if args.save_weights:
                save_checkpoint(nets, args, f"{log_dir}/checkpoint_test_best.pth", epoch, selected_loss)
        if epoch in args.save_epochs:
            save_checkpoint(nets, args, f"{log_dir}/checkpoint_ep{epoch:02d}.pth", epoch, selected_loss)
        if args.save_testsel_checkpoint and avg_test_loss < best.testsel_loss:
            # Matched control: what test-loss selection would have kept on this same
            # trajectory. Never influences training or the selected checkpoint.
            best.testsel_loss = avg_test_loss
            save_checkpoint(nets, args, f"{log_dir}/checkpoint_testsel_best.pth", epoch, avg_test_loss)
        if args.early_stop_patience > 0:
            if selected_loss < early_stop_reference_loss:
                early_stop_reference_loss = selected_loss
                epochs_without_selection_improvement = 0
            else:
                epochs_without_selection_improvement += 1

        if opt.scheduler is not None and args.lr_scheduler != 'onecycle':
            opt.scheduler.step()  # onecycle already stepped per optimizer step

        if (
            args.early_stop_patience > 0
            and epochs_without_selection_improvement >= args.early_stop_patience
        ):
            log(
                f"Early stopping after epoch {epoch}: no source-validation "
                f"improvement for {epochs_without_selection_improvement} epochs "
                f"(best epoch {best.epoch})."
            )
            break
        if epoch == args.stop_after_epoch:
            log(f"Stopping after epoch {epoch} of a {total_epochs}-epoch schedule (--stop_after_epoch).")
            break

    save_outputs(args, device, nets, log_dir, history, probes, best, last, last_epoch_train_sparsity)


if __name__ == '__main__':
    main(parse_args())
