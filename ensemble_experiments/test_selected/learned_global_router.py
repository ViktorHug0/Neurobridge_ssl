"""Train and evaluate the fixed learned ensemble-routing rule.

The member list, score transform, feature extractor, architecture, optimizer,
regularization, epoch count, and seed come from one immutable JSON config.  For
outer subject ``h``, router labels are read only from that fold's source records;
the target record is loaded only after fitting.  No target subject identifier is
an input feature, no other target query is used, and there is no per-fold search.

The fitted weights differ across LOSO folds because each fold has a different
source training set, just as the EEG encoders do.  The learning rule is global
and fixed.  A byte-identical fitted router cannot be both trained and evaluated
honestly on the same ten labeled subjects; such a deployment model needs an
external held-out subject cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ScoreRecord:
    subject: int
    scores: np.ndarray  # queries x members x candidates
    labels: np.ndarray


def load_dump(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    required = {"eeg", "image", "subject", "object"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    eeg = data["eeg"].astype(np.float32)
    image = data["image"].astype(np.float32)
    if eeg.ndim != 2 or image.ndim != 2 or len(eeg) != len(image):
        raise ValueError(f"invalid embedding shapes in {path}: {eeg.shape}, {image.shape}")
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    score = eeg @ image.T
    objects = np.asarray(data["object"])
    subjects = np.asarray(data["subject"])
    if score.shape[0] != score.shape[1]:
        raise ValueError(f"router expects square retrieval scores, got {score.shape} in {path}")
    if len(objects) != score.shape[0] or len(subjects) != score.shape[0]:
        raise ValueError(f"metadata length mismatch in {path}")
    return score.astype(np.float32), objects, subjects


def load_record(record: dict, members: list[str]) -> ScoreRecord:
    matrices = []
    reference_objects = None
    reference_subjects = None
    for member in members:
        if member not in record["dumps"]:
            raise KeyError(f"record for subject {record['subject']} lacks member {member}")
        matrix, objects, subjects = load_dump(record["dumps"][member])
        if reference_objects is None:
            reference_objects = objects
            reference_subjects = subjects
        else:
            if not np.array_equal(objects, reference_objects):
                raise ValueError("member dumps use different query/object ordering")
            if not np.array_equal(subjects, reference_subjects):
                raise ValueError("member dumps use different subject ordering")
        matrices.append(matrix)
    assert reference_objects is not None and reference_subjects is not None
    unique_subjects = np.unique(reference_subjects)
    if unique_subjects.tolist() != [int(record["subject"])]:
        raise ValueError(
            f"dump subject {unique_subjects.tolist()} does not match manifest "
            f"subject {record['subject']}"
        )
    # THINGS-EEG2 test dumps contain one query and one candidate per object in
    # matching order.  This check makes the diagonal target assumption explicit.
    if len(np.unique(reference_objects)) != len(reference_objects):
        raise ValueError("router requires one unique candidate per test object")
    labels = np.arange(len(reference_objects), dtype=np.int64)
    return ScoreRecord(
        subject=int(record["subject"]),
        scores=np.stack(matrices, axis=1),
        labels=labels,
    )


def transform_scores(scores: torch.Tensor, power: float, epsilon: float) -> torch.Tensor:
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(epsilon)
    z = (scores - mean) / std
    return torch.sign(z) * torch.abs(z).pow(power)


def router_features(scores: torch.Tensor, feature_temperature: float) -> torch.Tensor:
    """Query-local, label-free reliability features for every member.

    ``scores`` has shape ``(queries, members, candidates)``.  No statistic is
    shared across queries, which keeps inference inductive rather than
    transductive.
    """

    candidates = scores.shape[-1]
    top = torch.topk(scores, k=min(10, candidates), dim=-1).values
    maximum = top[..., 0]
    second = top[..., min(1, top.shape[-1] - 1)]
    margin = maximum - second
    top5_mean = top[..., : min(5, top.shape[-1])].mean(dim=-1)
    top10_mean = top.mean(dim=-1)
    probabilities = torch.softmax(scores / feature_temperature, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / math.log(candidates)
    row_l2 = scores.square().mean(dim=-1).sqrt()

    winners = scores.argmax(dim=-1)
    support = (winners.unsqueeze(2) == winners.unsqueeze(1)).float().mean(dim=2)
    consensus = scores.mean(dim=1)
    consensus_norm = consensus.square().sum(dim=-1).sqrt().clamp_min(1e-6)
    member_norm = scores.square().sum(dim=-1).sqrt().clamp_min(1e-6)
    consensus_similarity = (scores * consensus.unsqueeze(1)).sum(dim=-1)
    consensus_similarity = consensus_similarity / (
        member_norm * consensus_norm.unsqueeze(1)
    )
    consensus_at_winner = consensus.gather(1, winners)
    consensus_winner = consensus.argmax(dim=-1, keepdim=True)
    member_at_consensus = scores.gather(
        2, consensus_winner.unsqueeze(1).expand(-1, scores.shape[1], -1)
    ).squeeze(-1)
    return torch.stack(
        [
            maximum,
            margin,
            top5_mean,
            top10_mean,
            entropy,
            row_l2,
            support,
            consensus_similarity,
            consensus_at_winner,
            member_at_consensus,
        ],
        dim=-1,
    )


class GlobalQueryRouter(nn.Module):
    """A low-capacity shared member scorer with fixed member identity biases."""

    def __init__(self, member_count: int, feature_count: int, hidden_dim: int, uniform_floor: float):
        super().__init__()
        if not 0.0 <= uniform_floor <= 1.0:
            raise ValueError("uniform_floor must be in [0, 1]")
        self.member_count = member_count
        self.uniform_floor = uniform_floor
        self.shared = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.member_bias = nn.Parameter(torch.zeros(member_count))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        routing_logits = self.shared(features).squeeze(-1) + self.member_bias
        adaptive = torch.softmax(routing_logits, dim=1)
        uniform = torch.full_like(adaptive, 1.0 / self.member_count)
        return self.uniform_floor * uniform + (1.0 - self.uniform_floor) * adaptive


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(dim=-1) == labels).float().mean().item() * 100.0)


def prepare_records(
    records: list[ScoreRecord],
    device: torch.device,
    power: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.from_numpy(np.concatenate([record.scores for record in records])).to(device)
    labels = torch.from_numpy(np.concatenate([record.labels for record in records])).to(device)
    subjects = torch.from_numpy(
        np.concatenate(
            [np.full(len(record.labels), record.subject, dtype=np.int64) for record in records]
        )
    ).to(device)
    return transform_scores(scores, power, epsilon), labels, subjects


def fit_outer(
    train_records: list[ScoreRecord],
    test_record: ScoreRecord,
    members: list[str],
    config: dict,
    device: torch.device,
) -> tuple[dict, dict]:
    score_cfg = config["score_transform"]
    router_cfg = config["router"]
    # Give every outer fold the same initialization and minibatch RNG contract.
    torch.manual_seed(int(router_cfg["seed"]))
    train_scores, train_labels, _ = prepare_records(
        train_records, device, float(score_cfg["power"]), float(score_cfg["epsilon"])
    )
    test_scores, test_labels, _ = prepare_records(
        [test_record], device, float(score_cfg["power"]), float(score_cfg["epsilon"])
    )
    train_features = router_features(
        train_scores, float(router_cfg["feature_temperature"])
    )
    test_features = router_features(
        test_scores, float(router_cfg["feature_temperature"])
    )
    feature_mean = train_features.mean(dim=(0, 1), keepdim=True)
    feature_std = train_features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
    train_features = (train_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std

    model = GlobalQueryRouter(
        member_count=len(members),
        feature_count=train_features.shape[-1],
        hidden_dim=int(router_cfg["hidden_dim"]),
        uniform_floor=float(router_cfg["uniform_floor"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(router_cfg["learning_rate"]),
        weight_decay=float(router_cfg["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(router_cfg["seed"]))
    batch_size = int(router_cfg["batch_size"])
    fusion_temperature = float(router_cfg["fusion_temperature"])
    uniform_kl_lambda = float(router_cfg["uniform_kl_lambda"])

    model.train()
    for _ in range(int(router_cfg["epochs"])):
        order = torch.randperm(len(train_labels), generator=generator)
        for cpu_indices in order.split(batch_size):
            indices = cpu_indices.to(device)
            weights = model(train_features[indices])
            fused = (weights.unsqueeze(-1) * train_scores[indices]).sum(dim=1)
            retrieval_loss = F.cross_entropy(
                fused / fusion_temperature, train_labels[indices]
            )
            uniform_kl = (
                weights * (weights.clamp_min(1e-12).log() + math.log(len(members)))
            ).sum(dim=1).mean()
            loss = retrieval_loss + uniform_kl_lambda * uniform_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        weights = model(test_features)
        routed_scores = (weights.unsqueeze(-1) * test_scores).sum(dim=1)
        uniform_scores = test_scores.mean(dim=1)
        individual_correct = test_scores.argmax(dim=-1) == test_labels.unsqueeze(1)
        metrics = {
            "subject": test_record.subject,
            "router_top1": accuracy(routed_scores, test_labels),
            "uniform_top1": accuracy(uniform_scores, test_labels),
            "oracle_top1": float(individual_correct.any(dim=1).float().mean().item() * 100.0),
            "mean_weights": {
                member: float(value)
                for member, value in zip(members, weights.mean(dim=0).cpu().tolist())
            },
            "mean_max_weight": float(weights.max(dim=1).values.mean().item()),
        }
    state = {
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "members": members,
        "config": config,
    }
    return metrics, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ensemble_experiments/test_selected/router_config.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/things_eeg/ensemble50_testselected/learned_router/router_runs",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    members = list(config["members"])
    if list(manifest["members"]) != members:
        raise ValueError("manifest member order differs from the frozen router config")
    if config["score_transform"]["name"] != "row_z_signed_power":
        raise ValueError("only row_z_signed_power is implemented")
    constraints = config["inference_constraints"]
    if any(
        constraints[key]
        for key in ("uses_subject_id_feature", "uses_other_test_queries", "uses_target_labels")
    ):
        raise ValueError("router config violates inductive inference constraints")

    seed = int(config["router"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for outer in [int(v) for v in config["outer_subjects"]]:
        fold = manifest["folds"][str(outer)]
        if int(fold["outer_subject"]) != outer:
            raise ValueError(f"outer mismatch in manifest fold {outer}")
        train_subjects = [int(record["subject"]) for record in fold["router_train"]]
        if outer in train_subjects or len(set(train_subjects)) != len(train_subjects):
            raise ValueError(f"invalid router source subjects for outer {outer}")
        train_records = [load_record(record, members) for record in fold["router_train"]]
        test_record = load_record(fold["router_test"], members)
        if test_record.subject != outer:
            raise ValueError(f"target record mismatch for outer {outer}")
        metrics, state = fit_outer(train_records, test_record, members, config, device)
        rows.append(metrics)
        torch.save(state, output_dir / f"router-outer{outer:02d}.pth")
        print(
            f"outer={outer:02d} uniform={metrics['uniform_top1']:.2f} "
            f"router={metrics['router_top1']:.2f} oracle={metrics['oracle_top1']:.2f}",
            flush=True,
        )

    report = {
        "protocol_version": config["protocol_version"],
        "members": members,
        "manifest": str(Path(args.manifest)),
        "config": str(Path(args.config)),
        "per_subject": rows,
        "uniform_mean_top1": float(np.mean([row["uniform_top1"] for row in rows])),
        "router_mean_top1": float(np.mean([row["router_top1"] for row in rows])),
        "oracle_mean_top1": float(np.mean([row["oracle_top1"] for row in rows])),
    }
    (output_dir / "router_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key.endswith("top1")}, indent=2))


if __name__ == "__main__":
    main()
