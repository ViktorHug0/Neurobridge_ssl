import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from module.dataset import EEGPreImageDataset
from module.dataset_profiles import load_dataset_profile
from module.projector import ResidualAdapter
from module.util import retrieve_all
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything

DEFAULT_SUBJECT_IDS = [8]
DEFAULT_SELECTED_CHANNELS = []
DEFAULT_TIME_WINDOW = [0, 250]


def _load_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _pick_single_subject(subject_value):
    if isinstance(subject_value, list):
        if len(subject_value) == 0:
            raise ValueError("Expected at least one test subject ID.")
        return int(subject_value[0])
    return int(subject_value)


def _prepare_output_dir(output_dir, output_name):
    if output_name is not None:
        log_dir = os.path.join(output_dir, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}-{output_name}")
    else:
        log_dir = os.path.join(output_dir, datetime.now().strftime(r"%Y%m%d-%H%M%S"))

    log_dir_suffix = "-".join(log_dir.split("-")[2:])
    if os.path.exists(output_dir):
        for existing_dir in os.listdir(output_dir):
            if existing_dir.endswith(log_dir_suffix):
                existing_path = os.path.join(output_dir, existing_dir)
                if os.path.exists(os.path.join(existing_path, "result.csv")):
                    print(f"Evaluation with the same name '{log_dir_suffix}' already exists. Exiting to avoid overwriting.")
                    sys.exit(0)
                shutil.rmtree(existing_path)
                print(f"Removed incomplete evaluation directory '{existing_dir}' to avoid conflicts.")

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _build_eval_args(train_cfg, eval_cfg, cli_args):
    merged = {}
    merged.update(train_cfg)
    merged.update(eval_cfg)

    merged["eval_mode"] = cli_args.eval_mode
    if cli_args.sattc_saw_shrink is not None:
        merged["sattc_saw_shrink"] = cli_args.sattc_saw_shrink
    if cli_args.sattc_saw_diag:
        merged["sattc_saw_diag"] = True
    if cli_args.sattc_csls_k is not None:
        merged["sattc_csls_k"] = cli_args.sattc_csls_k
    if cli_args.sattc_cw:
        merged["sattc_cw"] = True
    if cli_args.sattc_cw_shrink is not None:
        merged["sattc_cw_shrink"] = cli_args.sattc_cw_shrink
    if cli_args.sattc_cw_diag:
        merged["sattc_cw_diag"] = True

    if cli_args.device is not None:
        merged["device"] = cli_args.device
    if cli_args.num_workers is not None:
        merged["num_workers"] = cli_args.num_workers
    if cli_args.batch_size is not None:
        merged["eval_batch_size"] = cli_args.batch_size
    if cli_args.eval_topk is not None:
        merged["eval_topk"] = cli_args.eval_topk
    if cli_args.dataset_name is not None:
        merged["dataset_name"] = cli_args.dataset_name
    if cli_args.dataset_config_path is not None:
        merged["dataset_config_path"] = cli_args.dataset_config_path
    if cli_args.test_dataset_name is not None:
        merged["test_dataset_name"] = cli_args.test_dataset_name

    test_subject_id = cli_args.test_subject_id
    if test_subject_id is None:
        test_subject_id = _pick_single_subject(merged.get("test_subject_ids", [1]))
    merged["test_subject_id"] = int(test_subject_id)
    return SimpleNamespace(**merged)


def _pick_profile_subject_id(test_subject_id, profile):
    if test_subject_id is not None:
        return int(test_subject_id)
    subject_ids = profile.get("test_subject_ids", DEFAULT_SUBJECT_IDS)
    if isinstance(subject_ids, list) and len(subject_ids) > 0:
        return int(subject_ids[0])
    return int(subject_ids)


def _profile_value_or_default(cli_value, default_value, profile_value):
    if profile_value is None:
        return cli_value
    if cli_value == default_value:
        return profile_value
    return cli_value


def _build_eval_dataset(eval_args):
    dataset_name = getattr(eval_args, "dataset_name", "things_eeg2")
    dataset_config_path = getattr(eval_args, "dataset_config_path", None)
    profile = load_dataset_profile(dataset_name, dataset_config_path)
    if dataset_name == "mixed":
        test_dataset_name = getattr(eval_args, "test_dataset_name", None) or profile.get("test_dataset_name")
        component_config_path = None
        for component in profile.get("components", []):
            if component.get("name") == test_dataset_name:
                component_config_path = component.get("config_path")
                break
        profile = load_dataset_profile(test_dataset_name, component_config_path)
    test_subject_id = _pick_profile_subject_id(getattr(eval_args, "test_subject_id", None), profile)
    eeg_data_dir = _profile_value_or_default(getattr(eval_args, "eeg_data_dir", './things_eeg/data/preprocessed_eeg'), './things_eeg/data/preprocessed_eeg', profile.get("eeg_data_dir"))
    image_feature_dir = _profile_value_or_default(
        getattr(eval_args, "image_feature_dir", '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit'),
        '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit',
        profile.get("image_feature_dir"),
    )
    text_feature_dir = _profile_value_or_default(getattr(eval_args, "text_feature_dir", './data/things_eeg/text_feature/BLIP2'), './data/things_eeg/text_feature/BLIP2', profile.get("text_feature_dir"))
    selected_channels = _profile_value_or_default(list(getattr(eval_args, "selected_channels", DEFAULT_SELECTED_CHANNELS)), DEFAULT_SELECTED_CHANNELS, profile.get("selected_channels"))
    time_window = _profile_value_or_default(list(getattr(eval_args, "time_window", DEFAULT_TIME_WINDOW)), DEFAULT_TIME_WINDOW, profile.get("time_window"))
    dataset = EEGPreImageDataset(
        [test_subject_id],
        eeg_data_dir,
        selected_channels,
        time_window,
        image_feature_dir,
        text_feature_dir,
        _to_bool(getattr(eval_args, "image_aug", False)),
        getattr(eval_args, "aug_image_feature_dirs", []),
        _to_bool(getattr(eval_args, "data_average", True), True),
        False,
        None,
        False,
        _to_bool(getattr(eval_args, "image_test_aug", False)),
        _to_bool(getattr(eval_args, "eeg_test_aug", False)),
        _to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )
    eval_batch_size = getattr(eval_args, "eval_batch_size", None) or profile.get("eval_batch_size", 200)
    eval_topk = getattr(eval_args, "eval_topk", None) or profile.get("topk", 5)
    return dataset, int(test_subject_id), int(eval_batch_size), int(eval_topk)


def _forward_feature(args, modules, eeg_backbone_batch):
    architecture = getattr(args, "architecture", "baseline")
    if architecture == "baseline":
        return modules["eeg_projector"](eeg_backbone_batch)

    if architecture == "clap_adapter":
        base_feature = modules["eeg_projector"](eeg_backbone_batch)
        checkpoint_stage = modules["checkpoint_stage"]
        if checkpoint_stage == "adapter" and modules["eeg_adapter"] is not None:
            return modules["eeg_adapter"](base_feature)
        return base_feature

    raise ValueError(f"Unsupported architecture in evaluate.py: {architecture}")


def _maybe_apply_image_branch(args, modules, image_feature_proj):
    architecture = getattr(args, "architecture", "baseline")
    if architecture == "clap_adapter":
        if _to_bool(getattr(args, "clap_transfer", True), True) and modules["checkpoint_stage"] == "adapter":
            if modules["eeg_adapter"] is not None:
                return modules["eeg_adapter"](image_feature_proj)
    return image_feature_proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["things_eeg2", "alljoined", "mixed"], default=None)
    parser.add_argument("--dataset_config_path", type=str, default=None)
    parser.add_argument("--test_dataset_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Directory containing checkpoint_test_best.pth and train_config.json")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--output_name", required=True, type=str)
    parser.add_argument("--eval_mode", required=True, type=str, choices=["plain_cosine", "saw", "saw_csls"])
    parser.add_argument("--test_subject_id", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_topk", type=int, default=None)

    parser.add_argument("--sattc_saw_shrink", type=float, default=None)
    parser.add_argument("--sattc_saw_diag", action="store_true")
    parser.add_argument("--sattc_csls_k", type=int, default=None)
    parser.add_argument("--sattc_cw", action="store_true")
    parser.add_argument("--sattc_cw_shrink", type=float, default=None)
    parser.add_argument("--sattc_cw_diag", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_test_best.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint_test_best.pth in '{checkpoint_dir}'")

    train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_cfg = _load_json(os.path.join(checkpoint_dir, "evaluate_config.json"))
    eval_args = _build_eval_args(train_cfg, eval_cfg, args)

    seed_everything(seed=args.seed if args.seed is not None else train_cfg.get("seed"))

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = _prepare_output_dir(output_dir, args.output_name)

    with open(os.path.join(log_dir, "evaluate_runtime_config.json"), "w") as f:
        json.dump(vars(eval_args), f, indent=4)

    device = torch.device(eval_args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    test_dataset, resolved_test_subject_id, eval_batch_size, eval_topk = _build_eval_dataset(eval_args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=getattr(eval_args, "num_workers", 0),
    )

    eeg_sample_points = test_dataset.num_sample_points
    channels_num = test_dataset.channels_num
    image_feature_dim = test_dataset.image_features.shape[-1]
    backbone_feature_dim = getattr(eval_args, "eeg_backbone_dim", 0) or image_feature_dim

    model = build_eeg_encoder(eval_args, backbone_feature_dim, eeg_sample_points, channels_num).to(device)
    eeg_projector = None
    img_projector = build_projector(eval_args.projector, image_feature_dim, eval_args.feature_dim).to(device)
    eeg_adapter = None

    architecture = getattr(eval_args, "architecture", checkpoint.get("architecture", "baseline"))
    if architecture == "baseline":
        eeg_projector = build_projector(eval_args.projector, backbone_feature_dim, eval_args.feature_dim).to(device)
    elif architecture == "clap_adapter":
        eeg_projector = build_projector(eval_args.projector, backbone_feature_dim, eval_args.feature_dim).to(device)
        eeg_adapter = ResidualAdapter(eval_args.feature_dim, eval_args.adapter_hidden_dim, getattr(eval_args, "adapter_alpha", 1.0)).to(device)
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")

    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    img_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    if eeg_adapter is not None and "eeg_adapter_state_dict" in checkpoint:
        eeg_adapter.load_state_dict(checkpoint["eeg_adapter_state_dict"])
    modules = {
        "eeg_projector": eeg_projector,
        "img_projector": img_projector,
        "eeg_adapter": eeg_adapter,
        "checkpoint_stage": checkpoint.get("stage", "joint"),
    }

    model.eval()
    eeg_projector.eval()
    img_projector.eval()
    if eeg_adapter is not None:
        eeg_adapter.eval()

    eeg_feature_list = []
    image_feature_list = []
    subject_list = []
    object_list = []
    image_idx_list = []

    with torch.no_grad():
        for batch in test_loader:
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            subject_id_batch = batch[3].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            eeg_backbone_batch = run_eeg_backbone(model, eval_args, eeg_batch, subject_id_batch)
            eeg_feature_batch = _forward_feature(eval_args, modules, eeg_backbone_batch)
            image_feature_proj = img_projector(image_feature_batch)
            image_feature_proj = _maybe_apply_image_branch(eval_args, modules, image_feature_proj)

            eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
            image_feature_list.append(image_feature_proj.cpu().numpy())
            subject_list.append(subject_id_batch.cpu().numpy())
            object_list.append(object_idx_batch.cpu().numpy())
            image_idx_list.append(image_idx_batch.cpu().numpy())

    eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
    image_feature_all = np.concatenate(image_feature_list, axis=0)
    subject_all = np.concatenate(subject_list, axis=0)
    object_all = np.concatenate(object_list, axis=0)
    image_all = np.concatenate(image_idx_list, axis=0)

    sattc_params = {
        "saw_shrink": getattr(eval_args, "sattc_saw_shrink", 0.2),
        "saw_diag": _to_bool(getattr(eval_args, "sattc_saw_diag", False)),
        "csls_k": getattr(eval_args, "sattc_csls_k", 12),
        "cw_enabled": _to_bool(getattr(eval_args, "sattc_cw", False)),
        "cw_shrink": getattr(eval_args, "sattc_cw_shrink", 0.05),
        "cw_diag": _to_bool(getattr(eval_args, "sattc_cw_diag", False)),
    }

    top5_count, top1_count, total = retrieve_all(
        eeg_feature_all,
        image_feature_all,
        _to_bool(getattr(eval_args, "data_average", True), True),
        subject_ids=subject_all,
        object_indices=object_all,
        image_indices=image_all,
        eval_mode=eval_args.eval_mode,
        sattc_params=sattc_params,
        topk_value=eval_topk,
    )
    top5_acc = top5_count / total * 100.0
    top1_acc = top1_count / total * 100.0

    result_dict = {
        "architecture": architecture,
        "dataset_name": getattr(eval_args, "dataset_name", "things_eeg2"),
        "eval_mode": eval_args.eval_mode,
        "top1 acc": f"{top1_acc:.2f}",
        f"top{eval_topk} acc": f"{top5_acc:.2f}",
        "best top1 acc": f"{top1_acc:.2f}",
        f"best top{eval_topk} acc": f"{top5_acc:.2f}",
        "best test loss": f"{float(checkpoint.get('loss', float('nan'))):.4f}",
        "best epoch": int(checkpoint.get("epoch", -1)),
    }
    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(log_dir, "result.csv"), index=False)

    summary = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "test_subject_id": resolved_test_subject_id,
        "eval_mode": eval_args.eval_mode,
        "top1_acc": top1_acc,
        f"top{eval_topk}_acc": top5_acc,
    }
    with open(os.path.join(log_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(
        f"eval_mode={eval_args.eval_mode} subject={resolved_test_subject_id} "
        f"top1={top1_acc:.2f} top{eval_topk}={top5_acc:.2f} checkpoint_epoch={checkpoint.get('epoch', -1)}"
    )


if __name__ == "__main__":
    main()
