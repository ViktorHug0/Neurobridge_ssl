#!/usr/bin/env python3
"""Few-shot fine-tuning comparison: projector-only vs full-model vs adapter.

Given a SAGE-zero-shot checkpoint (trained on 9 subjects), fine-tune on N labeled
EEG-image pairs from the held-out subject's training set and evaluate 200-way
retrieval on the test set (plain cosine, inductive).

Methods:
  - baseline: no fine-tuning (SAGE-zero-shot)
  - projector_only: freeze encoder, fine-tune eeg_projector
  - full_finetune: fine-tune encoder + eeg_projector
  - adapter: freeze everything, add low-rank residual adapter after projector
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared import (
    DEFAULT_OUTPUT_ROOT,
    LowRankResidualAdapter,
    TTAParams,
    add_common_args,
    aggregate_results,
    build_eval_args,
    build_test_dataset,
    build_train_dataset,
    cosine_scores,
    ensure_output_dir,
    evaluate_scores,
    find_checkpoint_dir,
    load_json,
    load_modules,
    normalize_rows,
    params_from_args,
    write_config,
)
from train import build_eeg_encoder, build_projector, run_eeg_backbone

DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "fewshot_finetuning")

CALIBRATION_SIZES_DEFAULT = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, "all"]
SEEDS_DEFAULT = [3300, 3301, 3302]


def evaluate_cosine_only(query_features, image_features):
    return evaluate_scores(cosine_scores(query_features, image_features))


def _encode_with_modules(eeg_data, image_data, subject_ids, modules, eval_args, batch_size=512):
    """Encode raw EEG/image tensors through the current model state."""
    device = modules["device"]
    n = eeg_data.shape[0]
    eeg_features, img_features = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            eeg_batch = eeg_data[start:end].to(device)
            img_batch = image_data[start:end].to(device)
            subj_batch = subject_ids[start:end].to(device)
            backbone = run_eeg_backbone(modules["model"], eval_args, eeg_batch, subj_batch)
            eeg_features.append(modules["eeg_projector"](backbone).cpu().numpy())
            img_features.append(modules["img_projector"](img_batch).cpu().numpy())
    return (
        np.concatenate(eeg_features, axis=0).astype(np.float32),
        np.concatenate(img_features, axis=0).astype(np.float32),
    )


def _get_raw_data_tensors(dataset, eval_args, subject_id):
    """Extract raw EEG/image tensors from dataset for fine-tuning."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    eeg_list, img_list, obj_list = [], [], []
    for batch in loader:
        eeg_list.append(batch[0])
        img_list.append(batch[1])
        obj_list.append(batch[4])
    eeg_all = torch.cat(eeg_list, dim=0)
    img_all = torch.cat(img_list, dim=0)
    obj_all = torch.cat(obj_list, dim=0)
    order = torch.argsort(obj_all)
    return eeg_all[order], img_all[order], obj_all[order]


def _infonce_loss(eeg_emb, img_emb, temperature=0.07):
    """Symmetric InfoNCE loss on L2-normalized embeddings."""
    eeg_norm = F.normalize(eeg_emb, dim=1)
    img_norm = F.normalize(img_emb, dim=1)
    logits = eeg_norm @ img_norm.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_e2i = F.cross_entropy(logits, labels)
    loss_i2e = F.cross_entropy(logits.T, labels)
    return (loss_e2i + loss_i2e) / 2.0


def _enable_grad(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_backbone_batched(model, eval_args, eeg, subject_ids, device, batch_size: int) -> torch.Tensor:
    """Run frozen EEG encoder in chunks to avoid OOM on large calibration sets."""
    chunks = []
    with torch.no_grad():
        for start in range(0, eeg.shape[0], batch_size):
            end = min(start + batch_size, eeg.shape[0])
            backbone = run_eeg_backbone(
                model, eval_args, eeg[start:end].to(device), subject_ids[start:end].to(device),
            )
            chunks.append(backbone.detach())
    return torch.cat(chunks, dim=0)


def _encode_image_batched(img_proj, images, device, batch_size: int) -> torch.Tensor:
    """Project image features in chunks."""
    chunks = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            chunks.append(img_proj(images[start:end].to(device)).detach())
    return torch.cat(chunks, dim=0)


def finetune_projector_only(
    modules, eval_args, train_eeg, train_img, subject_ids,
    val_eeg, val_img, val_subject_ids, test_eeg_raw, test_img_raw, test_subject_ids,
    config,
):
    """Freeze encoder, fine-tune only eeg_projector."""
    device = modules["device"]
    model = modules["model"]
    eeg_proj = copy.deepcopy(modules["eeg_projector"]).to(device)
    img_proj = modules["img_projector"]

    _enable_grad(eeg_proj)
    model.eval()
    img_proj.eval()
    eeg_proj.train()

    optimizer = torch.optim.AdamW(eeg_proj.parameters(), lr=config["lr"], weight_decay=config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    enc_bs = config["encode_batch_size"]
    backbone_train = _encode_backbone_batched(model, eval_args, train_eeg, subject_ids, device, enc_bs)
    img_targets = _encode_image_batched(img_proj, train_img, device, enc_bs)

    best_state, best_val = None, -1.0
    stale = 0
    bs = min(config["batch_size"], backbone_train.shape[0])

    for epoch in range(config["max_epochs"]):
        eeg_proj.train()
        perm = torch.randperm(backbone_train.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, backbone_train.shape[0], bs):
            idx = perm[start:start + bs]
            eeg_emb = eeg_proj(backbone_train[idx])
            loss = _infonce_loss(eeg_emb, img_targets[idx], config["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if val_eeg is not None and (epoch + 1) % config["eval_every"] == 0:
            eeg_proj.eval()
            with torch.no_grad():
                val_backbone = _encode_backbone_batched(
                    model, eval_args, val_eeg, val_subject_ids, device, enc_bs,
                )
                val_emb = eeg_proj(val_backbone).cpu().numpy()
                val_img_emb = _encode_image_batched(img_proj, val_img, device, enc_bs).cpu().numpy()
            val_metrics = evaluate_cosine_only(val_emb, val_img_emb)
            if val_metrics["top1_acc"] > best_val:
                best_val = val_metrics["top1_acc"]
                best_state = copy.deepcopy(eeg_proj.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    break

    if best_state is not None:
        eeg_proj.load_state_dict(best_state)
    eeg_proj.eval()

    with torch.no_grad():
        test_backbone = _encode_backbone_batched(
            model, eval_args, test_eeg_raw, test_subject_ids, device, enc_bs,
        )
        test_eeg_emb = eeg_proj(test_backbone).cpu().numpy()
        test_img_emb = _encode_image_batched(img_proj, test_img_raw, device, enc_bs).cpu().numpy()
    del eeg_proj, backbone_train, img_targets
    _free_cuda()
    return evaluate_cosine_only(test_eeg_emb, test_img_emb)


def finetune_full_model(
    modules, eval_args, train_eeg, train_img, subject_ids,
    val_eeg, val_img, val_subject_ids, test_eeg_raw, test_img_raw, test_subject_ids,
    config,
):
    """Fine-tune encoder + eeg_projector."""
    device = modules["device"]
    model = copy.deepcopy(modules["model"]).to(device)
    eeg_proj = copy.deepcopy(modules["eeg_projector"]).to(device)
    img_proj = modules["img_projector"]

    _enable_grad(model)
    _enable_grad(eeg_proj)
    img_proj.eval()
    model.train()
    eeg_proj.train()

    params = list(model.parameters()) + list(eeg_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    enc_bs = config["encode_batch_size"]
    img_targets = _encode_image_batched(img_proj, train_img, device, enc_bs)

    best_state, best_val = None, -1.0
    stale = 0
    bs = min(config["batch_size"], train_eeg.shape[0])

    for epoch in range(config["max_epochs"]):
        model.train()
        eeg_proj.train()
        perm = torch.randperm(train_eeg.shape[0])
        for start in range(0, train_eeg.shape[0], bs):
            idx = perm[start:start + bs]
            backbone = run_eeg_backbone(model, eval_args, train_eeg[idx].to(device), subject_ids[idx].to(device))
            eeg_emb = eeg_proj(backbone)
            loss = _infonce_loss(eeg_emb, img_targets[idx], config["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if val_eeg is not None and (epoch + 1) % config["eval_every"] == 0:
            model.eval()
            eeg_proj.eval()
            with torch.no_grad():
                val_backbone = _encode_backbone_batched(
                    model, eval_args, val_eeg, val_subject_ids, device, enc_bs,
                )
                val_emb = eeg_proj(val_backbone).cpu().numpy()
                val_img_emb = _encode_image_batched(img_proj, val_img, device, enc_bs).cpu().numpy()
            val_metrics = evaluate_cosine_only(val_emb, val_img_emb)
            if val_metrics["top1_acc"] > best_val:
                best_val = val_metrics["top1_acc"]
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "eeg_proj": copy.deepcopy(eeg_proj.state_dict()),
                }
                stale = 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        eeg_proj.load_state_dict(best_state["eeg_proj"])
    model.eval()
    eeg_proj.eval()

    with torch.no_grad():
        test_backbone = _encode_backbone_batched(
            model, eval_args, test_eeg_raw, test_subject_ids, device, enc_bs,
        )
        test_eeg_emb = eeg_proj(test_backbone).cpu().numpy()
        test_img_emb = _encode_image_batched(img_proj, test_img_raw, device, enc_bs).cpu().numpy()
    del model, eeg_proj, img_targets
    _free_cuda()
    return evaluate_cosine_only(test_eeg_emb, test_img_emb)


def finetune_adapter(
    modules, eval_args, train_eeg, train_img, subject_ids,
    val_eeg, val_img, val_subject_ids, test_eeg_raw, test_img_raw, test_subject_ids,
    config,
):
    """Freeze everything, add low-rank residual adapter after projector."""
    device = modules["device"]
    model = modules["model"]
    eeg_proj = modules["eeg_projector"]
    img_proj = modules["img_projector"]

    model.eval()
    eeg_proj.eval()
    img_proj.eval()

    feature_dim = config["feature_dim"]
    adapter = LowRankResidualAdapter(feature_dim, rank=config["adapter_rank"]).to(device)
    adapter.train()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["adapter_lr"], weight_decay=config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    enc_bs = config["encode_batch_size"]
    backbone_train = _encode_backbone_batched(model, eval_args, train_eeg, subject_ids, device, enc_bs)
    with torch.no_grad():
        eeg_base = eeg_proj(backbone_train).detach()
    img_targets = _encode_image_batched(img_proj, train_img, device, enc_bs)

    best_state, best_val = None, -1.0
    stale = 0
    bs = min(config["batch_size"], eeg_base.shape[0])

    for epoch in range(config["max_epochs"]):
        adapter.train()
        perm = torch.randperm(eeg_base.shape[0])
        for start in range(0, eeg_base.shape[0], bs):
            idx = perm[start:start + bs]
            eeg_emb = adapter(eeg_base[idx])
            loss = _infonce_loss(eeg_emb, img_targets[idx], config["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if val_eeg is not None and (epoch + 1) % config["eval_every"] == 0:
            adapter.eval()
            with torch.no_grad():
                val_backbone = _encode_backbone_batched(
                    model, eval_args, val_eeg, val_subject_ids, device, enc_bs,
                )
                val_base = eeg_proj(val_backbone)
                val_emb = adapter(val_base).cpu().numpy()
                val_img_emb = _encode_image_batched(img_proj, val_img, device, enc_bs).cpu().numpy()
            val_metrics = evaluate_cosine_only(val_emb, val_img_emb)
            if val_metrics["top1_acc"] > best_val:
                best_val = val_metrics["top1_acc"]
                best_state = copy.deepcopy(adapter.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()

    with torch.no_grad():
        test_backbone = _encode_backbone_batched(
            model, eval_args, test_eeg_raw, test_subject_ids, device, enc_bs,
        )
        test_base = eeg_proj(test_backbone)
        test_eeg_emb = adapter(test_base).cpu().numpy()
        test_img_emb = _encode_image_batched(img_proj, test_img_raw, device, enc_bs).cpu().numpy()
    del adapter, backbone_train, eeg_base, img_targets
    _free_cuda()
    return evaluate_cosine_only(test_eeg_emb, test_img_emb)


def _parse_calibration_sizes(values, total: int) -> list[int]:
    sizes = []
    for v in values:
        if str(v).lower() == "all":
            s = total
        else:
            s = int(v)
        if 1 <= s <= total and s not in sizes:
            sizes.append(s)
    return sorted(sizes)


def run_subject(source_run_dir: str, subject_id: int, args, config: dict):
    """Run all fine-tuning methods for one subject across sample sizes and seeds."""
    checkpoint_dir = find_checkpoint_dir(source_run_dir, subject_id)
    train_cfg = load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = build_eval_args(train_cfg, eval_cfg, args, subject_id)

    test_dataset = build_test_dataset(eval_args, subject_id, average=True)
    train_dataset = build_train_dataset(eval_args, subject_id, average=True)

    modules = load_modules(eval_args, checkpoint_dir, test_dataset)
    device = modules["device"]

    # Get raw tensors
    train_eeg, train_img, train_obj = _get_raw_data_tensors(train_dataset, eval_args, subject_id)
    test_eeg, test_img, test_obj = _get_raw_data_tensors(test_dataset, eval_args, subject_id)
    total_train = train_eeg.shape[0]

    train_subject_ids = torch.full((total_train,), subject_id, dtype=torch.long)
    test_subject_ids = torch.full((test_eeg.shape[0],), subject_id, dtype=torch.long)

    config["feature_dim"] = int(eval_args.feature_dim)

    enc_bs = config.get("encode_batch_size", 64)
    # Baseline (no fine-tuning)
    with torch.no_grad():
        test_backbone = _encode_backbone_batched(
            modules["model"], eval_args, test_eeg, test_subject_ids, device, enc_bs,
        )
        test_eeg_emb = modules["eeg_projector"](test_backbone).cpu().numpy()
        test_img_emb = _encode_image_batched(
            modules["img_projector"], test_img, device, enc_bs,
        ).cpu().numpy()
    baseline_metrics = evaluate_cosine_only(test_eeg_emb, test_img_emb)
    _free_cuda()

    cal_sizes = _parse_calibration_sizes(args.calibration_sizes, total_train)
    rows = []

    for cal_size in cal_sizes:
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 100000 + int(subject_id) * 1000 + int(cal_size))
            cal_idx = np.sort(rng.choice(total_train, size=cal_size, replace=False))

            cal_eeg = train_eeg[cal_idx]
            cal_img = train_img[cal_idx]
            cal_subj = train_subject_ids[cal_idx]

            # Split into train/val (80/20) if enough samples
            if cal_size >= 50:
                n_val = max(10, cal_size // 5)
                n_train = cal_size - n_val
                val_perm = np.random.default_rng(seed).permutation(cal_size)
                train_idx = val_perm[:n_train]
                val_idx = val_perm[n_train:]
                ft_eeg, ft_img, ft_subj = cal_eeg[train_idx], cal_img[train_idx], cal_subj[train_idx]
                v_eeg, v_img, v_subj = cal_eeg[val_idx], cal_img[val_idx], cal_subj[val_idx]
            else:
                ft_eeg, ft_img, ft_subj = cal_eeg, cal_img, cal_subj
                v_eeg, v_img, v_subj = None, None, None

            # Baseline row (same for all cal sizes/seeds but useful for plotting)
            rows.append({
                "subject_id": int(subject_id), "seed": int(seed),
                "calibration_size": int(cal_size), "method": "baseline",
                "top1_acc": baseline_metrics["top1_acc"],
                "top5_acc": baseline_metrics["top5_acc"],
            })

            # Method 1: projector_only
            t0 = time.time()
            metrics = finetune_projector_only(
                modules, eval_args, ft_eeg, ft_img, ft_subj,
                v_eeg, v_img, v_subj, test_eeg, test_img, test_subject_ids, config,
            )
            rows.append({
                "subject_id": int(subject_id), "seed": int(seed),
                "calibration_size": int(cal_size), "method": "projector_only",
                "top1_acc": metrics["top1_acc"], "top5_acc": metrics["top5_acc"],
                "time_s": round(time.time() - t0, 1),
            })

            # Method 2: full_finetune
            t0 = time.time()
            config_full = {**config, "lr": config["lr_full"]}
            metrics = finetune_full_model(
                modules, eval_args, ft_eeg, ft_img, ft_subj,
                v_eeg, v_img, v_subj, test_eeg, test_img, test_subject_ids, config_full,
            )
            rows.append({
                "subject_id": int(subject_id), "seed": int(seed),
                "calibration_size": int(cal_size), "method": "full_finetune",
                "top1_acc": metrics["top1_acc"], "top5_acc": metrics["top5_acc"],
                "time_s": round(time.time() - t0, 1),
            })

            # Method 3: adapter
            t0 = time.time()
            metrics = finetune_adapter(
                modules, eval_args, ft_eeg, ft_img, ft_subj,
                v_eeg, v_img, v_subj, test_eeg, test_img, test_subject_ids, config,
            )
            rows.append({
                "subject_id": int(subject_id), "seed": int(seed),
                "calibration_size": int(cal_size), "method": "adapter",
                "top1_acc": metrics["top1_acc"], "top5_acc": metrics["top5_acc"],
                "time_s": round(time.time() - t0, 1),
            })

            print(
                f"  sub={subject_id:02d} seed={seed} cal={cal_size:>5d} | "
                f"base={baseline_metrics['top1_acc']:.1f} "
                f"proj={rows[-3]['top1_acc']:.1f} "
                f"full={rows[-2]['top1_acc']:.1f} "
                f"adapt={rows[-1]['top1_acc']:.1f}"
            )

    for row in rows:
        row["total_train_samples"] = total_train
        row["checkpoint_dir"] = checkpoint_dir
    return rows


def _write_summaries(df: pd.DataFrame, output_dir: str):
    df.to_csv(os.path.join(output_dir, "subject_results.csv"), index=False)
    avg = aggregate_results(df, ["method", "calibration_size"])
    avg.to_csv(os.path.join(output_dir, "average_results.csv"), index=False)

    # Delta vs baseline
    baseline_avg = avg[avg["method"] == "baseline"][["calibration_size", "top1_mean"]].rename(
        columns={"top1_mean": "baseline_top1"}
    )
    delta = avg.merge(baseline_avg, on="calibration_size", how="left")
    delta["delta_top1"] = delta["top1_mean"] - delta["baseline_top1"]
    delta.to_csv(os.path.join(output_dir, "delta_vs_baseline.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--calibration_sizes", nargs="+", default=CALIBRATION_SIZES_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--lr_projector", type=float, default=3e-4)
    parser.add_argument("--lr_full", type=float, default=1e-4)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--adapter_rank", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch_size_ft", type=int, default=256)
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=64,
        help="Batch size for frozen encoder/image forward passes (lower if OOM).",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=5)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    source_run_dir = os.path.abspath(args.source_run_dir)
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source_run_dir does not exist: {source_run_dir}")

    params = params_from_args(args)
    output_dir = ensure_output_dir(args, "fewshot_finetuning")

    config = {
        "lr": args.lr_projector,
        "lr_full": args.lr_full,
        "adapter_lr": args.adapter_lr,
        "adapter_rank": args.adapter_rank,
        "wd": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "temperature": args.temperature,
        "batch_size": args.batch_size_ft,
        "encode_batch_size": args.encode_batch_size,
        "eval_every": args.eval_every,
        "feature_dim": 0,  # filled per subject
    }

    write_config(output_dir, args, params)

    all_rows = []
    for subject_id in args.subjects:
        print(f"[fewshot_finetuning] subject={int(subject_id):02d}")
        all_rows.extend(run_subject(source_run_dir, subject_id, args, config))
        _write_summaries(pd.DataFrame(all_rows), output_dir)

    subject_df = pd.DataFrame(all_rows)
    _write_summaries(subject_df, output_dir)

    print(f"\nSaved results to: {output_dir}")
    print(f"Grid: {len(args.subjects)} subjects × {len(args.calibration_sizes)} sizes × {len(args.seeds)} seeds × 4 methods")


if __name__ == "__main__":
    main()
