"""Learn one subject-independent candidate scorer on pooled all-ten data.

Unlike the query router, this model scores each of the 200 image candidates
from its cross-member score/rank agreement pattern.  The exact same shared
function and checkpoint are used for every subject; subject identity and
candidate identity are not features.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learned_global_router import accuracy, prepare_records
from learned_pooled_router import hard_negative_loss, load_pooled_records


def candidate_features(scores: torch.Tensor) -> torch.Tensor:
    """Return query-local features with shape (queries, candidates, features)."""

    query_count, member_count, candidate_count = scores.shape
    member_scores = scores.transpose(1, 2)

    order = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(candidate_count, device=scores.device)
    rank_values = rank_values.view(1, 1, -1).expand(query_count, member_count, -1)
    ranks.scatter_(2, order, rank_values)
    percentiles = (1.0 - ranks.float() / max(candidate_count - 1, 1)).transpose(1, 2)

    products = [
        member_scores[..., left] * member_scores[..., right]
        for left in range(member_count)
        for right in range(left + 1, member_count)
    ]
    pairwise_products = torch.stack(products, dim=-1)
    consensus = torch.stack(
        [
            member_scores.mean(dim=-1),
            member_scores.max(dim=-1).values,
            member_scores.std(dim=-1, unbiased=False),
        ],
        dim=-1,
    )
    winners = scores.argmax(dim=-1)
    vote_fraction = F.one_hot(winners, num_classes=candidate_count).float().mean(dim=1)
    top5_fraction = (ranks < min(5, candidate_count)).float().mean(dim=1)
    return torch.cat(
        [
            member_scores,
            percentiles,
            pairwise_products,
            consensus,
            vote_fraction.unsqueeze(-1),
            top5_fraction.unsqueeze(-1),
        ],
        dim=-1,
    )


class ResidualCandidateScorer(nn.Module):
    def __init__(self, feature_count: int, variant: str, hidden_dim: int, scale: float):
        super().__init__()
        self.scale = scale
        if variant == "linear":
            self.correction = nn.Linear(feature_count, 1)
            final = self.correction
        elif variant == "mlp":
            self.correction = nn.Sequential(
                nn.Linear(feature_count, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            final = self.correction[-1]
        else:
            raise ValueError(f"unknown variant: {variant}")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        uniform = scores.mean(dim=1)
        correction = torch.tanh(self.correction(features).squeeze(-1))
        return uniform + self.scale * correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="ensemble_experiments/test_selected/pooled_candidate_config.json"
    )
    parser.add_argument("--variant", required=True, choices=("linear", "mlp"))
    parser.add_argument("--dump-root", default=None)
    parser.add_argument(
        "--output-root",
        default="results/things_eeg/ensemble50_testselected/pooled_candidate_scorer",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if args.variant not in config["variants"]:
        raise ValueError(f"variant {args.variant} is not frozen in config")
    if config["score_transform"]["name"] != "row_z_signed_power":
        raise ValueError("only row_z_signed_power is implemented")
    constraints = config["inference_constraints"]
    if any(bool(value) for value in constraints.values()):
        raise ValueError("candidate scorer config permits a subject-dependent rule")

    train_cfg = config["training"]
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    dump_root = Path(args.dump_root or config["data_contract"]["dump_root"])
    records = load_pooled_records(config, dump_root)
    score_cfg = config["score_transform"]
    scores, labels, reporting_subjects = prepare_records(
        records, device, float(score_cfg["power"]), float(score_cfg["epsilon"])
    )
    features = candidate_features(scores)
    feature_mean = features.mean(dim=(0, 1), keepdim=True)
    feature_std = features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
    features = (features - feature_mean) / feature_std

    model = ResidualCandidateScorer(
        feature_count=features.shape[-1],
        variant=args.variant,
        hidden_dim=int(train_cfg["hidden_dim"]),
        scale=float(train_cfg["residual_scale"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch_size = int(train_cfg["batch_size"])
    final_losses = {}

    model.train()
    for epoch in range(int(train_cfg["epochs"])):
        order = torch.randperm(len(labels), generator=generator)
        epoch_values = []
        for cpu_indices in order.split(batch_size):
            indices = cpu_indices.to(device)
            fused = model(features[indices], scores[indices])
            cross_entropy = F.cross_entropy(
                fused / float(train_cfg["fusion_temperature"]), labels[indices]
            )
            hard_negative = hard_negative_loss(
                fused,
                labels[indices],
                float(train_cfg["hard_negative_margin"]),
                float(train_cfg["hard_negative_temperature"]),
            )
            loss = (
                float(train_cfg["cross_entropy_weight"]) * cross_entropy
                + float(train_cfg["hard_negative_weight"]) * hard_negative
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_values.append(
                (float(loss.detach()), float(cross_entropy.detach()), float(hard_negative.detach()))
            )
        if epoch == int(train_cfg["epochs"]) - 1:
            final_losses = {
                "total": float(np.mean([value[0] for value in epoch_values])),
                "cross_entropy": float(np.mean([value[1] for value in epoch_values])),
                "hard_negative": float(np.mean([value[2] for value in epoch_values])),
            }

    model.eval()
    with torch.no_grad():
        fused = model(features, scores)
        uniform = scores.mean(dim=1)
        member_correct = scores.argmax(dim=-1) == labels.unsqueeze(1)
        per_subject = []
        for subject in [int(value) for value in config["subjects"]]:
            mask = reporting_subjects == subject
            per_subject.append(
                {
                    "subject": subject,
                    "uniform_top1": accuracy(uniform[mask], labels[mask]),
                    "scorer_top1": accuracy(fused[mask], labels[mask]),
                    "oracle_top1": float(member_correct[mask].any(dim=1).float().mean() * 100.0),
                }
            )

    output_dir = Path(args.output_root) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "candidate-scorer-all10.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
            "members": config["members"],
            "variant": args.variant,
            "config": config,
        },
        checkpoint,
    )
    report = {
        "protocol_version": config["protocol_version"],
        "protocol_name": config["protocol_name"],
        "variant": args.variant,
        "interpretation": (
            "Pooled development result: labels from all ten reported subjects were used "
            "to fit one shared candidate-scoring function."
        ),
        "config": str(config_path),
        "members": config["members"],
        "single_checkpoint": str(checkpoint),
        "subject_identity_is_scorer_input": False,
        "per_subject": per_subject,
        "uniform_mean_top1": float(np.mean([row["uniform_top1"] for row in per_subject])),
        "scorer_mean_top1": float(np.mean([row["scorer_top1"] for row in per_subject])),
        "oracle_mean_top1": float(np.mean([row["oracle_top1"] for row in per_subject])),
        "final_epoch_losses": final_losses,
    }
    (output_dir / "candidate_scorer_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key.endswith("mean_top1")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
