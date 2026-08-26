"""Jointly train two EEG encoders with a negative-score decorrelation leash."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ensemble_experiments.decorrelated_models.losses import (
    deployed_ensemble_contrastive_loss,
    ensemble_contrastive_loss,
    negative_score_correlation_loss,
    row_z,
    soft_multiple_choice_rescue_loss,
)
from module.dataset import EEGPreImageDataset
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import TSConv_parameterizable
from module.loss import ContrastiveLoss
from module.projector import ProjectorLinear
from module.sampler import GroupedImageBatchSampler
from train import (
    _GroupedSubset,
    build_image_positive_mask,
    cross_subject_stimulus_mix,
    seed_everything,
)


class TwinTSConvBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        image_dim: int,
        args,
        encoder_type: str = "TSConv_parameterizable",
        backbone_dim: int | None = None,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        backbone_dim = backbone_dim or args.backbone_dim
        feature_dim = feature_dim or args.feature_dim
        if encoder_type == "ATM":
            self.encoder = ATMS(
                feature_dim=backbone_dim,
                eeg_sample_points=samples,
                channels_num=channels,
            )
        elif encoder_type == "TSConv_parameterizable":
            self.encoder = TSConv_parameterizable(
                feature_dim=backbone_dim,
                eeg_sample_points=samples,
                channels_num=channels,
                temporal_filters=40,
                temporal_kernel=30,
                pool_kernel=51,
                pool_stride=5,
                spatial_filters=40,
                projection_filters=40,
                activation="elu",
                dropout=0.5,
            )
        else:
            raise ValueError(f"unsupported encoder type: {encoder_type}")
        self.eeg_projector = ProjectorLinear(backbone_dim, feature_dim)
        self.img_projector = ProjectorLinear(image_dim, feature_dim)
        self.criterion = ContrastiveLoss(
            init_temperature=0.07,
            alpha=1.0,
            beta=1.0,
            eeg_l2norm=False,
            img_l2norm=True,
            text_l2norm=False,
            learnable=False,
            is_softplus=True,
        )

    def features(
        self,
        eeg: torch.Tensor,
        image: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == "ATM":
            backbone = self.encoder(eeg, subject_ids)
        else:
            backbone = self.encoder(eeg)
        return self.eeg_projector(backbone), self.img_projector(image)

    @staticmethod
    def cosine_scores(eeg_feature: torch.Tensor, image_feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(eeg_feature, dim=1) @ F.normalize(image_feature, dim=1).T

    def individual_loss(
        self,
        eeg_feature: torch.Tensor,
        image_feature: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.criterion.multi_positive_pair_loss(
            eeg_feature, image_feature, positive_mask
        )

    def individual_row_losses(
        self,
        eeg_feature: torch.Tensor,
        image_feature: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-query EEG-to-image loss for retrieval-directed rescue training."""
        return self.criterion.multi_positive_row_losses(
            eeg_feature, image_feature, positive_mask
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dump-root", default="results/things_eeg/synthetic_subjects/ensemble_screen/dumps")
    parser.add_argument("--held-subject", type=int, required=True)
    parser.add_argument("--train-subject-ids", type=int, nargs="+", required=True)
    parser.add_argument("--eeg-data-dir", required=True)
    parser.add_argument("--image-feature-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--backbone-dim", type=int, default=1024)
    parser.add_argument(
        "--encoder-a",
        choices=["TSConv_parameterizable", "ATM"],
        default="TSConv_parameterizable",
    )
    parser.add_argument(
        "--encoder-b",
        choices=["TSConv_parameterizable", "ATM"],
        default="TSConv_parameterizable",
    )
    parser.add_argument("--feature-dim-a", type=int)
    parser.add_argument("--feature-dim-b", type=int)
    parser.add_argument("--backbone-dim-a", type=int)
    parser.add_argument("--backbone-dim-b", type=int)
    parser.add_argument("--lambda-div", type=float, required=True)
    parser.add_argument("--beta-ensemble", type=float, required=True)
    parser.add_argument("--decorrelation-start-epoch", type=int, default=11)
    parser.add_argument("--gamma-rescue", type=float, default=0.0)
    parser.add_argument("--rescue-temperature", type=float, default=0.5)
    parser.add_argument("--rescue-start-epoch", type=int, default=1)
    parser.add_argument(
        "--selection-protocol", choices=["test", "valcon"], default="test"
    )
    parser.add_argument("--val-concept-ratio", type=float, default=0.10)
    parser.add_argument("--val-concept-seed", type=int, default=20260822)
    parser.add_argument("--seed-a", type=int, default=3300)
    parser.add_argument("--seed-b", type=int, default=3301)
    parser.add_argument("--train-rng-seed", type=int, default=7330)
    parser.add_argument("--mixup-alpha", type=float, default=0.5)
    parser.add_argument(
        "--freeze-member",
        choices=["none", "a", "b"],
        default="none",
        help="Freeze one pretrained branch; only the other branch is optimized.",
    )
    parser.add_argument(
        "--frozen-checkpoint",
        help="Checkpoint containing branch_state_dict for --freeze-member.",
    )
    parser.add_argument(
        "--fusion-loss-mode",
        choices=["symmetric_batch", "deployed_unique"],
        default="symmetric_batch",
        help="Use deployed_unique for frozen-member fusion-aware training.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("decorrelated_twins")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_loaders(args, device: torch.device):
    dataset_kwargs = dict(
        eeg_data_dir=args.eeg_data_dir,
        selected_channels=[],
        time_window=[0, 250],
        image_feature_dir=args.image_feature_dir,
        text_feature_dir="",
        image_aug=False,
        aug_image_feature_dirs=[],
        average=True,
        _random=False,
    )
    train_dataset = EEGPreImageDataset(
        args.train_subject_ids, train=True, **dataset_kwargs
    )
    test_dataset = EEGPreImageDataset(
        [args.held_subject], train=False, **dataset_kwargs
    )
    train_main_dataset = train_dataset
    val_loader = None
    if args.selection_protocol == "valcon":
        groups = train_dataset.get_image_group_indices()
        concepts = sorted({key[0] for key in groups})
        holdout_count = min(
            max(1, int(np.floor(len(concepts) * args.val_concept_ratio))),
            len(concepts) - 1,
        )
        held_concepts = set(
            np.random.default_rng(args.val_concept_seed)
            .permutation(concepts)[:holdout_count]
            .tolist()
        )
        val_indices, train_indices = [], []
        for key, indices in groups.items():
            (val_indices if key[0] in held_concepts else train_indices).extend(indices)
        train_main_dataset = _GroupedSubset(train_dataset, train_indices)
        val_loader = DataLoader(
            Subset(train_dataset, val_indices),
            batch_size=200,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(
            f"[valcon] concepts={holdout_count}/{len(concepts)} "
            f"seed={args.val_concept_seed} train_items={len(train_indices)} "
            f"val_items={len(val_indices)}"
        )

    sampler = GroupedImageBatchSampler(
        train_main_dataset,
        batch_size=args.batch_size,
        samples_per_image=9,
        drop_last=True,
        seed=args.train_rng_seed,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_main_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_dataset, train_loader, val_loader, test_loader


def build_branch(
    seed: int,
    train_dataset,
    args,
    device: torch.device,
    member: str,
) -> TwinTSConvBranch:
    seed_everything(seed)
    encoder_type = getattr(args, f"encoder_{member}")
    backbone_dim = getattr(args, f"backbone_dim_{member}") or args.backbone_dim
    feature_dim = getattr(args, f"feature_dim_{member}") or args.feature_dim
    branch = TwinTSConvBranch(
        channels=train_dataset.channels_num,
        samples=train_dataset.num_sample_points,
        image_dim=train_dataset.image_features.shape[-1],
        args=args,
        encoder_type=encoder_type,
        backbone_dim=backbone_dim,
        feature_dim=feature_dim,
    )
    return branch.to(device)


def correctness_and_top5(scores: torch.Tensor, positive_mask: torch.Tensor):
    prediction = scores.argmax(dim=1)
    top1 = positive_mask.gather(1, prediction[:, None]).squeeze(1)
    top5_indices = scores.topk(min(5, scores.shape[1]), dim=1).indices
    top5 = positive_mask.gather(1, top5_indices).any(dim=1)
    return prediction, top1, top5


def safe_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum().div(denominator.clamp_min(1e-12)).item())


def pair_metrics(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, float]:
    z_a, z_b = row_z(scores_a), row_z(scores_b)
    fused = 0.5 * (z_a + z_b)
    pred_a, correct_a, top5_a = correctness_and_top5(scores_a, positive_mask)
    pred_b, correct_b, top5_b = correctness_and_top5(scores_b, positive_mask)
    _, correct_pair, top5_pair = correctness_and_top5(fused, positive_mask)
    accuracy_a = float(correct_a.float().mean().mul(100).item())
    accuracy_b = float(correct_b.float().mean().mul(100).item())
    pair_accuracy = float(correct_pair.float().mean().mul(100).item())
    best = max(accuracy_a, accuracy_b)
    mean = 0.5 * (accuracy_a + accuracy_b)
    oracle = float((correct_a | correct_b).float().mean().mul(100).item())
    both_wrong = ~correct_a & ~correct_b
    wrong_agreement = (
        float((pred_a[both_wrong] == pred_b[both_wrong]).float().mean().item())
        if torch.any(both_wrong)
        else float("nan")
    )
    positive_a = scores_a.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    positive_b = scores_b.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    negative_a = scores_a.masked_fill(positive_mask, float("-inf")).max(dim=1).values
    negative_b = scores_b.masked_fill(positive_mask, float("-inf")).max(dim=1).values
    headroom = oracle - best
    return {
        "member_a_top1": accuracy_a,
        "member_b_top1": accuracy_b,
        "member_a_top5": float(top5_a.float().mean().mul(100).item()),
        "member_b_top5": float(top5_b.float().mean().mul(100).item()),
        "pair_top1": pair_accuracy,
        "pair_top5": float(top5_pair.float().mean().mul(100).item()),
        "gain_over_best": pair_accuracy - best,
        "gain_over_mean": pair_accuracy - mean,
        "score_correlation": safe_correlation(z_a, z_b),
        "margin_correlation": safe_correlation(positive_a - negative_a, positive_b - negative_b),
        "correctness_correlation": safe_correlation(correct_a.float(), correct_b.float()),
        "prediction_disagreement": float((pred_a != pred_b).float().mean().item()),
        "double_fault": float(both_wrong.float().mean().item()),
        "wrong_winner_agreement": wrong_agreement,
        "oracle_top1": oracle,
        "oracle_headroom_over_best": headroom,
        "oracle_headroom_realized": (
            (pair_accuracy - best) / headroom if headroom > 0 else float("nan")
        ),
    }


@torch.no_grad()
def evaluate(branch_a, branch_b, loader, device, rescue_temperature: float = 0.5):
    branch_a.eval()
    branch_b.eval()
    eeg_a, image_a, eeg_b, image_b = [], [], [], []
    objects, images = [], []
    for batch in loader:
        eeg = batch[0].to(device, non_blocking=True)
        image = batch[1].to(device, non_blocking=True)
        subject = batch[3].to(device, non_blocking=True)
        feature_a, target_a = branch_a.features(eeg, image, subject)
        feature_b, target_b = branch_b.features(eeg, image, subject)
        eeg_a.append(feature_a.cpu())
        image_a.append(target_a.cpu())
        eeg_b.append(feature_b.cpu())
        image_b.append(target_b.cpu())
        objects.append(batch[4].cpu())
        images.append(batch[5].cpu())

    feature_a = torch.cat(eeg_a).to(device)
    target_a = torch.cat(image_a).to(device)
    feature_b = torch.cat(eeg_b).to(device)
    target_b = torch.cat(image_b).to(device)
    object_indices = torch.cat(objects).to(device)
    image_indices = torch.cat(images).to(device)
    positive_mask = build_image_positive_mask(object_indices, image_indices)
    scores_a = branch_a.cosine_scores(feature_a, target_a)
    scores_b = branch_b.cosine_scores(feature_b, target_b)
    loss_a = branch_a.individual_loss(feature_a, target_a, positive_mask)
    loss_b = branch_b.individual_loss(feature_b, target_b, positive_mask)
    row_losses_a = branch_a.individual_row_losses(feature_a, target_a, positive_mask)
    row_losses_b = branch_b.individual_row_losses(feature_b, target_b, positive_mask)
    rescue_loss, rescue_responsibilities = soft_multiple_choice_rescue_loss(
        row_losses_a, row_losses_b, rescue_temperature
    )
    pair_loss, _ = ensemble_contrastive_loss(scores_a, scores_b, positive_mask)
    negative_corr_sq, negative_corr = negative_score_correlation_loss(
        scores_a,
        scores_b,
        positive_mask,
        object_indices,
        image_indices,
    )
    metrics = pair_metrics(scores_a, scores_b, positive_mask)
    metrics.update({
        "member_a_test_loss": float(loss_a.item()),
        "member_b_test_loss": float(loss_b.item()),
        "pair_test_loss": float(pair_loss.item()),
        "rescue_test_loss": float(rescue_loss.item()),
        "rescue_assignment_max_weight": float(
            rescue_responsibilities.max(dim=1).values.mean().item()
        ),
        "negative_score_correlation": float(negative_corr.item()),
        "negative_score_correlation_squared": float(negative_corr_sq.item()),
    })
    embeddings = {
        "a_eeg": feature_a.cpu().numpy(),
        "a_image": target_a.cpu().numpy(),
        "b_eeg": feature_b.cpu().numpy(),
        "b_image": target_b.cpu().numpy(),
        "object": object_indices.cpu().numpy(),
        "image_idx": image_indices.cpu().numpy(),
        "pair_scores": (0.5 * (row_z(scores_a) + row_z(scores_b))).cpu().numpy(),
    }
    return metrics, embeddings


@torch.no_grad()
def evaluate_selection_losses(
    branch_a,
    branch_b,
    loader,
    device,
    rescue_temperature: float,
) -> dict[str, float]:
    """Evaluate source-concept validation losses without constructing a huge score matrix."""
    branch_a.eval()
    branch_b.eval()
    totals = {"member_a": 0.0, "member_b": 0.0, "pair": 0.0, "rescue": 0.0}
    examples = 0
    for batch in loader:
        eeg = batch[0].to(device, non_blocking=True)
        image = batch[1].to(device, non_blocking=True)
        subject = batch[3].to(device, non_blocking=True)
        objects = batch[4].to(device, non_blocking=True)
        images = batch[5].to(device, non_blocking=True)
        positive_mask = build_image_positive_mask(objects, images)
        feature_a, target_a = branch_a.features(eeg, image, subject)
        feature_b, target_b = branch_b.features(eeg, image, subject)
        member_a = branch_a.individual_loss(feature_a, target_a, positive_mask)
        member_b = branch_b.individual_loss(feature_b, target_b, positive_mask)
        pair, _ = ensemble_contrastive_loss(
            branch_a.cosine_scores(feature_a, target_a),
            branch_b.cosine_scores(feature_b, target_b),
            positive_mask,
        )
        rescue, _ = soft_multiple_choice_rescue_loss(
            branch_a.individual_row_losses(feature_a, target_a, positive_mask),
            branch_b.individual_row_losses(feature_b, target_b, positive_mask),
            rescue_temperature,
        )
        count = eeg.shape[0]
        examples += count
        totals["member_a"] += float(member_a.item()) * count
        totals["member_b"] += float(member_b.item()) * count
        totals["pair"] += float(pair.item()) * count
        totals["rescue"] += float(rescue.item()) * count
    return {key: value / examples for key, value in totals.items()}


def checkpoint(branch: nn.Module, epoch: int, metrics: dict) -> dict:
    return {
        "epoch": epoch,
        "branch_state_dict": branch.state_dict(),
        "metrics": metrics,
    }


def load_frozen_branch(branch: nn.Module, checkpoint_path: str, device) -> int:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("branch_state_dict")
    if state is None:
        raise ValueError(f"{checkpoint_path} has no branch_state_dict")
    branch.load_state_dict(state, strict=True)
    branch.requires_grad_(False)
    branch.eval()
    return int(payload.get("epoch", -1))


def selection_improved(
    candidate_loss: float,
    candidate_top1: float,
    best: dict[str, float],
) -> bool:
    """Mirror train.py: minimize held-out loss, then use top-1 as a tie-break."""
    return candidate_loss < best["loss"] or (
        candidate_loss == best["loss"] and candidate_top1 > best["top1"]
    )


def main() -> None:
    args = parse_args()
    if args.lambda_div < 0 or args.beta_ensemble < 0 or args.gamma_rescue < 0:
        raise ValueError("lambda, beta, and gamma must be non-negative")
    if args.rescue_temperature <= 0:
        raise ValueError("rescue-temperature must be positive")
    if args.selection_protocol == "valcon" and not 0 < args.val_concept_ratio < 1:
        raise ValueError("ValCon requires 0 < val-concept-ratio < 1")
    if args.decorrelation_start_epoch < 1 or args.rescue_start_epoch < 1:
        raise ValueError("auxiliary-loss start epochs must be positive")
    if args.held_subject in args.train_subject_ids:
        raise ValueError("held subject cannot appear in train-subject-ids")
    if (args.freeze_member == "none") != (args.frozen_checkpoint is None):
        raise ValueError(
            "--freeze-member and --frozen-checkpoint must be provided together"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.csv"
    if result_path.exists():
        print(f"exists {result_path}; nothing to do")
        return
    logger = configure_logging(output_dir)
    (output_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_dataset, train_loader, val_loader, test_loader = build_loaders(args, device)
    branch_a = build_branch(args.seed_a, train_dataset, args, device, "a")
    branch_b = build_branch(args.seed_b, train_dataset, args, device, "b")
    frozen_epoch = -1
    if args.freeze_member == "a":
        frozen_epoch = load_frozen_branch(
            branch_a, args.frozen_checkpoint, device
        )
    elif args.freeze_member == "b":
        frozen_epoch = load_frozen_branch(
            branch_b, args.frozen_checkpoint, device
        )
    seed_everything(args.train_rng_seed)
    parameters = list(branch_a.parameters()) + list(branch_b.parameters())
    optimizer = AdamW(
        [parameter for parameter in parameters if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    logger.info(
        "arm=%s held=%02d selection=%s lambda=%.4g beta=%.4g gamma=%.4g rescue_tau=%.4g "
        "encoders=(%s,%s) freeze=%s frozen_epoch=%d fusion_mode=%s "
        "branch_params=(%.2fM, %.2fM)",
        args.arm,
        args.held_subject,
        args.selection_protocol,
        args.lambda_div,
        args.beta_ensemble,
        args.gamma_rescue,
        args.rescue_temperature,
        args.encoder_a,
        args.encoder_b,
        args.freeze_member,
        frozen_epoch,
        args.fusion_loss_mode,
        sum(p.numel() for p in branch_a.parameters()) / 1e6,
        sum(p.numel() for p in branch_b.parameters()) / 1e6,
    )

    best_a = {"loss": float("inf"), "epoch": -1, "top1": float("-inf")}
    best_b = {"loss": float("inf"), "epoch": -1, "top1": float("-inf")}
    best_pair = {"loss": float("inf"), "epoch": -1, "top1": float("nan")}
    selection_tag = "val" if args.selection_protocol == "valcon" else "test"
    path_a = output_dir / f"checkpoint_member_a_{selection_tag}_best.pth"
    path_b = output_dir / f"checkpoint_member_b_{selection_tag}_best.pth"
    path_pair = output_dir / f"checkpoint_pair_{selection_tag}_best.pth"
    best_a_testctl = {"loss": float("inf"), "epoch": -1, "top1": float("-inf")}
    best_b_testctl = {"loss": float("inf"), "epoch": -1, "top1": float("-inf")}
    path_a_testctl = output_dir / "checkpoint_member_a_testsel_control.pth"
    path_b_testctl = output_dir / "checkpoint_member_b_testsel_control.pth"
    history = []

    for epoch in range(1, args.num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        branch_a.eval() if args.freeze_member == "a" else branch_a.train()
        branch_b.eval() if args.freeze_member == "b" else branch_b.train()
        active_lambda = args.lambda_div if epoch >= args.decorrelation_start_epoch else 0.0
        active_gamma = args.gamma_rescue if epoch >= args.rescue_start_epoch else 0.0
        totals = {"loss": 0.0, "a": 0.0, "b": 0.0, "ensemble": 0.0,
                  "diversity": 0.0, "negative_corr": 0.0, "rescue": 0.0,
                  "rescue_max_weight": 0.0}
        for batch in train_loader:
            eeg = batch[0].to(device, non_blocking=True)
            image = batch[1].to(device, non_blocking=True)
            subject = batch[3].to(device, non_blocking=True)
            object_indices = batch[4].to(device, non_blocking=True)
            image_indices = batch[5].to(device, non_blocking=True)
            eeg = cross_subject_stimulus_mix(
                eeg,
                object_indices,
                image_indices,
                subject,
                alpha=args.mixup_alpha,
                mixup_type="pairwise",
            )
            positive_mask = build_image_positive_mask(object_indices, image_indices)
            if args.freeze_member == "a":
                with torch.no_grad():
                    feature_a, target_a = branch_a.features(eeg, image, subject)
                feature_b, target_b = branch_b.features(eeg, image, subject)
            elif args.freeze_member == "b":
                feature_a, target_a = branch_a.features(eeg, image, subject)
                with torch.no_grad():
                    feature_b, target_b = branch_b.features(eeg, image, subject)
            else:
                feature_a, target_a = branch_a.features(eeg, image, subject)
                feature_b, target_b = branch_b.features(eeg, image, subject)
            individual_a = branch_a.individual_loss(feature_a, target_a, positive_mask)
            individual_b = branch_b.individual_loss(feature_b, target_b, positive_mask)
            row_losses_a = branch_a.individual_row_losses(
                feature_a, target_a, positive_mask
            )
            row_losses_b = branch_b.individual_row_losses(
                feature_b, target_b, positive_mask
            )
            rescue_loss, rescue_responsibilities = soft_multiple_choice_rescue_loss(
                row_losses_a, row_losses_b, args.rescue_temperature
            )
            scores_a = branch_a.cosine_scores(feature_a, target_a)
            scores_b = branch_b.cosine_scores(feature_b, target_b)
            if args.fusion_loss_mode == "deployed_unique":
                ensemble_loss, _ = deployed_ensemble_contrastive_loss(
                    scores_a,
                    scores_b,
                    positive_mask,
                    object_indices,
                    image_indices,
                )
            else:
                ensemble_loss, _ = ensemble_contrastive_loss(
                    scores_a, scores_b, positive_mask
                )
            diversity_loss, negative_corr = negative_score_correlation_loss(
                scores_a,
                scores_b,
                positive_mask,
                object_indices,
                image_indices,
            )
            if args.freeze_member == "a":
                individual_objective = individual_b
            elif args.freeze_member == "b":
                individual_objective = individual_a
            else:
                individual_objective = individual_a + individual_b
            loss = (
                individual_objective
                + args.beta_ensemble * ensemble_loss
                + active_lambda * diversity_loss
                + active_gamma * rescue_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["a"] += float(individual_a.item())
            totals["b"] += float(individual_b.item())
            totals["ensemble"] += float(ensemble_loss.item())
            totals["diversity"] += float(diversity_loss.item())
            totals["negative_corr"] += float(negative_corr.item())
            totals["rescue"] += float(rescue_loss.item())
            totals["rescue_max_weight"] += float(
                rescue_responsibilities.max(dim=1).values.mean().item()
            )

        batches = len(train_loader)
        test_metrics, _ = evaluate(
            branch_a, branch_b, test_loader, device, args.rescue_temperature
        )
        if args.selection_protocol == "valcon":
            selection_losses = evaluate_selection_losses(
                branch_a,
                branch_b,
                val_loader,
                device,
                args.rescue_temperature,
            )
            selected_a_loss = selection_losses["member_a"]
            selected_b_loss = selection_losses["member_b"]
            selected_pair_loss = selection_losses["pair"]
            selected_a_tie = selected_b_tie = selected_pair_tie = 0.0
        else:
            selection_losses = {
                "member_a": test_metrics["member_a_test_loss"],
                "member_b": test_metrics["member_b_test_loss"],
                "pair": test_metrics["pair_test_loss"],
                "rescue": test_metrics["rescue_test_loss"],
            }
            selected_a_loss = selection_losses["member_a"]
            selected_b_loss = selection_losses["member_b"]
            selected_pair_loss = selection_losses["pair"]
            selected_a_tie = test_metrics["member_a_top1"]
            selected_b_tie = test_metrics["member_b_top1"]
            selected_pair_tie = test_metrics["pair_top1"]
        row = {
            "epoch": epoch,
            "active_lambda": active_lambda,
            "active_gamma": active_gamma,
            **{f"train_{key}": value / batches for key, value in totals.items()},
            **{f"selection_{key}_loss": value for key, value in selection_losses.items()},
            **test_metrics,
        }
        if device.type == "cuda":
            row["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            row["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        history.append(row)
        with (output_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        if selection_improved(
            selected_a_loss, selected_a_tie, best_a
        ):
            best_a = {
                "loss": selected_a_loss,
                "epoch": epoch,
                "top1": selected_a_tie,
            }
            torch.save(checkpoint(branch_a, epoch, test_metrics), path_a)
        if selection_improved(
            selected_b_loss, selected_b_tie, best_b
        ):
            best_b = {
                "loss": selected_b_loss,
                "epoch": epoch,
                "top1": selected_b_tie,
            }
            torch.save(checkpoint(branch_b, epoch, test_metrics), path_b)
        if selection_improved(
            selected_pair_loss, selected_pair_tie, best_pair
        ):
            best_pair = {
                "loss": selected_pair_loss,
                "epoch": epoch,
                "top1": test_metrics["pair_top1"],
            }
            torch.save({
                "epoch": epoch,
                "branch_a_state_dict": branch_a.state_dict(),
                "branch_b_state_dict": branch_b.state_dict(),
                "metrics": test_metrics,
            }, path_pair)
        if args.selection_protocol == "valcon":
            if selection_improved(
                test_metrics["member_a_test_loss"],
                test_metrics["member_a_top1"],
                best_a_testctl,
            ):
                best_a_testctl = {
                    "loss": test_metrics["member_a_test_loss"],
                    "epoch": epoch,
                    "top1": test_metrics["member_a_top1"],
                }
                torch.save(checkpoint(branch_a, epoch, test_metrics), path_a_testctl)
            if selection_improved(
                test_metrics["member_b_test_loss"],
                test_metrics["member_b_top1"],
                best_b_testctl,
            ):
                best_b_testctl = {
                    "loss": test_metrics["member_b_test_loss"],
                    "epoch": epoch,
                    "top1": test_metrics["member_b_top1"],
                }
                torch.save(checkpoint(branch_b, epoch, test_metrics), path_b_testctl)
        logger.info(
            "epoch=%02d lambda=%.4g gamma=%.4g train=%.4f ind=(%.4f,%.4f) "
            "ens=%.4f rescue=%.4f rescue_w=%.3f div=%.4f negcorr=%.4f "
            "test_solo=(%.2f,%.2f) pair=%.2f corr=%.4f",
            epoch,
            active_lambda,
            active_gamma,
            row["train_loss"],
            row["train_a"],
            row["train_b"],
            row["train_ensemble"],
            row["train_rescue"],
            row["train_rescue_max_weight"],
            row["train_diversity"],
            row["train_negative_corr"],
            test_metrics["member_a_top1"],
            test_metrics["member_b_top1"],
            test_metrics["pair_top1"],
            test_metrics["score_correlation"],
        )
        if device.type == "cuda":
            logger.info(
                "epoch=%02d peak_gpu=(allocated=%.2fGiB,reserved=%.2fGiB)",
                epoch,
                row["peak_gpu_allocated_gib"],
                row["peak_gpu_reserved_gib"],
            )

    if args.freeze_member == "none":
        selected_a = torch.load(path_a, map_location=device, weights_only=False)
        selected_b = torch.load(path_b, map_location=device, weights_only=False)
        branch_a.load_state_dict(selected_a["branch_state_dict"])
        branch_b.load_state_dict(selected_b["branch_state_dict"])
    else:
        selected_pair = torch.load(path_pair, map_location=device, weights_only=False)
        branch_a.load_state_dict(selected_pair["branch_a_state_dict"])
        branch_b.load_state_dict(selected_pair["branch_b_state_dict"])
        selected_a = {"epoch": frozen_epoch if args.freeze_member == "a" else selected_pair["epoch"]}
        selected_b = {"epoch": frozen_epoch if args.freeze_member == "b" else selected_pair["epoch"]}
    final_metrics, embeddings = evaluate(
        branch_a, branch_b, test_loader, device, args.rescue_temperature
    )
    final_metrics.update({
        "architecture": f"decorrelated_{args.encoder_a}_plus_{args.encoder_b}",
        "member_a_encoder": args.encoder_a,
        "member_b_encoder": args.encoder_b,
        "arm": args.arm,
        "lambda_div": args.lambda_div,
        "beta_ensemble": args.beta_ensemble,
        "gamma_rescue": args.gamma_rescue,
        "freeze_member": args.freeze_member,
        "frozen_checkpoint": args.frozen_checkpoint or "",
        "frozen_checkpoint_epoch": frozen_epoch,
        "fusion_loss_mode": args.fusion_loss_mode,
        "rescue_temperature": args.rescue_temperature,
        "selection_protocol": args.selection_protocol,
        "val_concept_ratio": (
            args.val_concept_ratio if args.selection_protocol == "valcon" else 0.0
        ),
        "val_concept_seed": (
            args.val_concept_seed if args.selection_protocol == "valcon" else -1
        ),
        "member_a_selection_loss": best_a["loss"],
        "member_b_selection_loss": best_b["loss"],
        "pair_selection_loss": best_pair["loss"],
        "member_a_best_epoch": selected_a["epoch"],
        "member_b_best_epoch": selected_b["epoch"],
        "pair_contemporaneous_best_epoch": best_pair["epoch"],
        "pair_contemporaneous_best_top1": best_pair["top1"],
        # The primary pair always combines the two independently selected members.
        "best top1 acc": final_metrics["pair_top1"],
        "best top5 acc": final_metrics["pair_top5"],
    })
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_metrics))
        writer.writeheader()
        writer.writerow(final_metrics)

    dump_root = Path(args.dump_root)
    dump_root.mkdir(parents=True, exist_ok=True)
    common = {"object": embeddings["object"], "image_idx": embeddings["image_idx"]}
    np.savez_compressed(
        dump_root / f"decor_{args.arm}_a-sub{args.held_subject:02d}.npz",
        eeg=embeddings["a_eeg"], image=embeddings["a_image"], **common,
    )
    np.savez_compressed(
        dump_root / f"decor_{args.arm}_b-sub{args.held_subject:02d}.npz",
        eeg=embeddings["b_eeg"], image=embeddings["b_image"], **common,
    )
    np.savez_compressed(
        output_dir / "individual_best_pair_scores.npz",
        scores=embeddings["pair_scores"], **common,
    )
    logger.info("individual-best final metrics: %s", json.dumps(final_metrics, sort_keys=True))

    if args.selection_protocol == "valcon":
        selected_a_ctl = torch.load(
            path_a_testctl, map_location=device, weights_only=False
        )
        selected_b_ctl = torch.load(
            path_b_testctl, map_location=device, weights_only=False
        )
        branch_a.load_state_dict(selected_a_ctl["branch_state_dict"])
        branch_b.load_state_dict(selected_b_ctl["branch_state_dict"])
        control_metrics, control_embeddings = evaluate(
            branch_a, branch_b, test_loader, device, args.rescue_temperature
        )
        control_metrics.update({
            "architecture": f"decorrelated_{args.encoder_a}_plus_{args.encoder_b}",
            "member_a_encoder": args.encoder_a,
            "member_b_encoder": args.encoder_b,
            "arm": args.arm + "_testsel_control",
            "lambda_div": args.lambda_div,
            "beta_ensemble": args.beta_ensemble,
            "gamma_rescue": args.gamma_rescue,
            "rescue_temperature": args.rescue_temperature,
            "selection_protocol": "matched_test_control_from_valcon_trajectory",
            "val_concept_ratio": args.val_concept_ratio,
            "val_concept_seed": args.val_concept_seed,
            "member_a_selection_loss": best_a_testctl["loss"],
            "member_b_selection_loss": best_b_testctl["loss"],
            "member_a_best_epoch": selected_a_ctl["epoch"],
            "member_b_best_epoch": selected_b_ctl["epoch"],
            "best top1 acc": control_metrics["pair_top1"],
            "best top5 acc": control_metrics["pair_top5"],
        })
        control_result_path = output_dir / "result_testsel_control.csv"
        with control_result_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(control_metrics))
            writer.writeheader()
            writer.writerow(control_metrics)
        control_common = {
            "object": control_embeddings["object"],
            "image_idx": control_embeddings["image_idx"],
        }
        control_prefix = f"decor_{args.arm}_ctl"
        np.savez_compressed(
            dump_root / f"{control_prefix}_a-sub{args.held_subject:02d}.npz",
            eeg=control_embeddings["a_eeg"],
            image=control_embeddings["a_image"],
            **control_common,
        )
        np.savez_compressed(
            dump_root / f"{control_prefix}_b-sub{args.held_subject:02d}.npz",
            eeg=control_embeddings["b_eeg"],
            image=control_embeddings["b_image"],
            **control_common,
        )
        np.savez_compressed(
            output_dir / "testsel_control_pair_scores.npz",
            scores=control_embeddings["pair_scores"],
            **control_common,
        )
        logger.info(
            "matched test-selected control metrics: %s",
            json.dumps(control_metrics, sort_keys=True),
        )


if __name__ == "__main__":
    main()
