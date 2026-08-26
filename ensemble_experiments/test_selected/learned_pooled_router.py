"""Fit one query router on the pooled all-ten-subject development set.

This is the protocol-v2 router requested for the test-selected ensemble track.
There is exactly one member pool, feature normalization, fitted state dict, and
scoring function for all ten subjects.  Subject identity is used only to group
metrics in the final report; it is never passed to the router.

The fit intentionally uses labeled queries from all ten subjects.  Its all-ten
accuracy is therefore a pooled development result, not an estimate on unseen
subjects.  This is explicit in both the config and the saved report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from learned_global_router import (
    GlobalQueryRouter,
    ScoreRecord,
    accuracy,
    load_record,
    prepare_records,
    router_features,
)


def load_pooled_records(config: dict, dump_root: Path) -> list[ScoreRecord]:
    members = list(config["members"])
    pattern = config["data_contract"]["dump_pattern"]
    records = []
    for subject in [int(value) for value in config["subjects"]]:
        record = {
            "subject": subject,
            "dumps": {
                member: str(dump_root / pattern.format(member=member, subject=subject))
                for member in members
            },
        }
        missing = [path for path in record["dumps"].values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"subject {subject} is missing {len(missing)} member dumps: {missing}"
            )
        records.append(load_record(record, members))
    return records


def hard_negative_loss(
    fused_scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Smooth top-1 margin against the strongest incorrect candidate."""

    if temperature <= 0:
        raise ValueError("hard_negative_temperature must be positive")
    true_scores = fused_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    label_mask = F.one_hot(labels, num_classes=fused_scores.shape[1]).bool()
    strongest_negative = fused_scores.masked_fill(label_mask, -torch.inf).max(dim=1).values
    violation = (strongest_negative - true_scores + margin) / temperature
    return F.softplus(violation).mean() * temperature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="ensemble_experiments/test_selected/pooled_router_config.json"
    )
    parser.add_argument("--dump-root", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/things_eeg/ensemble50_testselected/pooled_router",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["score_transform"]["name"] != "row_z_signed_power":
        raise ValueError("only row_z_signed_power is implemented")
    subjects = [int(value) for value in config["subjects"]]
    data_contract = config["data_contract"]
    if [int(value) for value in data_contract["fit_subjects"]] != subjects:
        raise ValueError("fit_subjects must be the same ordered all-ten subject list")
    if [int(value) for value in data_contract["report_subjects"]] != subjects:
        raise ValueError("report_subjects must be the same ordered all-ten subject list")
    constraints = config["inference_constraints"]
    required_false = (
        "uses_subject_id_feature",
        "uses_subject_specific_parameters",
        "uses_other_queries_at_inference",
        "member_pool_changes_by_subject",
        "scoring_rule_changes_by_subject",
        "fitted_model_changes_by_subject",
    )
    if any(bool(constraints[key]) for key in required_false):
        raise ValueError("pooled router config permits a subject-dependent inference rule")

    router_cfg = config["router"]
    score_cfg = config["score_transform"]
    seed = int(router_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    dump_root = Path(args.dump_root or data_contract["dump_root"])
    records = load_pooled_records(config, dump_root)
    members = list(config["members"])

    scores, labels, reporting_subjects = prepare_records(
        records,
        device,
        float(score_cfg["power"]),
        float(score_cfg["epsilon"]),
    )
    features = router_features(scores, float(router_cfg["feature_temperature"]))
    feature_mean = features.mean(dim=(0, 1), keepdim=True)
    feature_std = features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
    features = (features - feature_mean) / feature_std

    model = GlobalQueryRouter(
        member_count=len(members),
        feature_count=features.shape[-1],
        hidden_dim=int(router_cfg["hidden_dim"]),
        uniform_floor=float(router_cfg["uniform_floor"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(router_cfg["learning_rate"]),
        weight_decay=float(router_cfg["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch_size = int(router_cfg["batch_size"])
    member_count = len(members)
    final_losses = {}

    model.train()
    for epoch in range(int(router_cfg["epochs"])):
        order = torch.randperm(len(labels), generator=generator)
        epoch_losses = []
        for cpu_indices in order.split(batch_size):
            indices = cpu_indices.to(device)
            weights = model(features[indices])
            fused = (weights.unsqueeze(-1) * scores[indices]).sum(dim=1)
            cross_entropy = F.cross_entropy(
                fused / float(router_cfg["fusion_temperature"]), labels[indices]
            )
            hard_negative = hard_negative_loss(
                fused,
                labels[indices],
                float(router_cfg["hard_negative_margin"]),
                float(router_cfg["hard_negative_temperature"]),
            )
            uniform_kl = (
                weights
                * (weights.clamp_min(1e-12).log() + math.log(member_count))
            ).sum(dim=1).mean()
            loss = (
                float(router_cfg["cross_entropy_weight"]) * cross_entropy
                + float(router_cfg["hard_negative_weight"]) * hard_negative
                + float(router_cfg["uniform_kl_lambda"]) * uniform_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(
                (float(loss.detach()), float(cross_entropy.detach()), float(hard_negative.detach()))
            )
        if epoch == int(router_cfg["epochs"]) - 1:
            final_losses = {
                "total": float(np.mean([value[0] for value in epoch_losses])),
                "cross_entropy": float(np.mean([value[1] for value in epoch_losses])),
                "hard_negative": float(np.mean([value[2] for value in epoch_losses])),
            }

    model.eval()
    with torch.no_grad():
        weights = model(features)
        routed_scores = (weights.unsqueeze(-1) * scores).sum(dim=1)
        uniform_scores = scores.mean(dim=1)
        member_correct = scores.argmax(dim=-1) == labels.unsqueeze(1)
        per_subject = []
        for subject in subjects:
            mask = reporting_subjects == subject
            per_subject.append(
                {
                    "subject": subject,
                    "uniform_top1": accuracy(uniform_scores[mask], labels[mask]),
                    "router_top1": accuracy(routed_scores[mask], labels[mask]),
                    "oracle_top1": float(member_correct[mask].any(dim=1).float().mean() * 100.0),
                    "mean_weights": {
                        member: float(value)
                        for member, value in zip(
                            members, weights[mask].mean(dim=0).cpu().tolist()
                        )
                    },
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "members": members,
        "config": config,
    }
    torch.save(state, output_dir / "pooled-router-all10.pth")
    report = {
        "protocol_version": config["protocol_version"],
        "protocol_name": config["protocol_name"],
        "interpretation": (
            "Pooled development result: labels from all ten reported subjects were used "
            "to fit one shared router."
        ),
        "config": str(config_path),
        "dump_root": str(dump_root),
        "members": members,
        "single_checkpoint": str(output_dir / "pooled-router-all10.pth"),
        "subject_identity_is_router_input": False,
        "per_subject": per_subject,
        "uniform_mean_top1": float(np.mean([row["uniform_top1"] for row in per_subject])),
        "router_mean_top1": float(np.mean([row["router_top1"] for row in per_subject])),
        "oracle_mean_top1": float(np.mean([row["oracle_top1"] for row in per_subject])),
        "global_mean_weights": {
            member: float(value)
            for member, value in zip(members, weights.mean(dim=0).cpu().tolist())
        },
        "mean_max_weight": float(weights.max(dim=1).values.mean()),
        "final_epoch_losses": final_losses,
    }
    (output_dir / "pooled_router_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key.endswith("top1")}, indent=2))


if __name__ == "__main__":
    main()
