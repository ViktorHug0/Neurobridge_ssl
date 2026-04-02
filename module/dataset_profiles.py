import json
import os
from copy import deepcopy


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _dataset_profile_dir():
    return os.path.join(_repo_root(), "configs", "datasets")


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_path(path: str | None):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


def _normalize_profile(profile):
    normalized = deepcopy(profile)
    for key in (
        "eeg_data_dir",
        "image_feature_dir",
        "text_feature_dir",
        "adapter_output_dir",
        "raw_config_path",
        "image_root",
        "image_set_dir",
    ):
        if key in normalized:
            normalized[key] = _normalize_path(normalized[key])

    for component in normalized.get("components", []):
        if "config_path" in component:
            component["config_path"] = _normalize_path(component["config_path"])
    return normalized


def load_dataset_profile(dataset_name: str, dataset_config_path: str | None = None):
    builtin_path = os.path.join(_dataset_profile_dir(), f"{dataset_name}.json")
    if not os.path.isfile(builtin_path):
        raise FileNotFoundError(f"Unknown dataset profile '{dataset_name}': '{builtin_path}' not found.")

    profile = _read_json(builtin_path)
    if dataset_config_path:
        profile = _deep_update(profile, _read_json(dataset_config_path))
    profile.setdefault("dataset_name", dataset_name)
    return _normalize_profile(profile)


def resolve_dataset_component(component_name: str, dataset_config_path: str | None = None):
    profile = load_dataset_profile(component_name, dataset_config_path)
    profile.setdefault("weight", 1.0)
    return profile
