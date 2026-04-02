import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from extract_feature import resolve_split_dir
from module.dataset_profiles import load_dataset_profile


def _load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def _read_eeg(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    raise TypeError(f"Unsupported EEG container at '{path}'")


def _mvnn(epoched_train, epoched_test):
    from sklearn.discriminant_analysis import _cov
    import scipy

    session_data = [epoched_test, epoched_train]

    sigma_part = np.empty((len(session_data), session_data[0].shape[2], session_data[0].shape[2]))
    for p in range(sigma_part.shape[0]):
        sigma_cond = np.empty((session_data[p].shape[0], session_data[0].shape[2], session_data[0].shape[2]))
        for i in range(session_data[p].shape[0]):
            cond_data = session_data[p][i]
            sigma_cond[i] = np.mean(
                [_cov(np.transpose(cond_data[e]), shrinkage="auto") for e in range(cond_data.shape[0])],
                axis=0,
            )
        sigma_part[p] = sigma_cond.mean(axis=0)
    sigma_tot = sigma_part[1]
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

    whitened_test = np.reshape(
        (
            np.reshape(session_data[0], (-1, session_data[0].shape[2], session_data[0].shape[3])).swapaxes(1, 2)
            @ sigma_inv
        ).swapaxes(1, 2),
        session_data[0].shape,
    )
    whitened_train = np.reshape(
        (
            np.reshape(session_data[1], (-1, session_data[1].shape[2], session_data[1].shape[3])).swapaxes(1, 2)
            @ sigma_inv
        ).swapaxes(1, 2),
        session_data[1].shape,
    )

    return whitened_train, whitened_test


def _save_eeg_subject(path, eeg_data, eeg_obj):
    np.save(
        path,
        {
            "preprocessed_eeg_data": eeg_data,
            "ch_names": eeg_obj.get("ch_names"),
            "times": eeg_obj.get("times"),
        },
        allow_pickle=True,
    )


def _apply_subject_mvnn(output_eeg_dir, subjects):
    print(f"Applying MVNN to {len(subjects)} exported Alljoined subjects...")
    pbar = tqdm(subjects, desc="MVNN")
    for subject in pbar:
        pbar.set_postfix({"subject": subject})
        subject_dir = os.path.join(output_eeg_dir, subject)
        train_path = os.path.join(subject_dir, "train.npy")
        test_path = os.path.join(subject_dir, "test.npy")
        train_obj = _read_eeg(train_path)
        test_obj = _read_eeg(test_path)
        train_data = np.asarray(train_obj["preprocessed_eeg_data"], dtype=np.float32)
        test_data = np.asarray(test_obj["preprocessed_eeg_data"], dtype=np.float32)
        train_mvnn, test_mvnn = _mvnn(train_data, test_data)
        _save_eeg_subject(train_path, train_mvnn.astype(np.float32, copy=False), train_obj)
        _save_eeg_subject(test_path, test_mvnn.astype(np.float32, copy=False), test_obj)


def _update_export_metadata(output_eeg_dir, normalization):
    info_path = os.path.join(output_eeg_dir, "info.json")
    if os.path.isfile(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        info["normalization"] = normalization
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    dataset_info_path = os.path.join(output_eeg_dir, "dataset_info.json")
    if os.path.isfile(dataset_info_path):
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)
        dataset_info["normalization"] = normalization
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)


def _filter_metadata(df, partition):
    df = df[df["partition"] == partition]
    if "dropped" in df.columns:
        df = df[df["dropped"] == False]
    return df.reset_index(drop=True)


def _select_channels(eeg_data, selected_channels, all_channels):
    if not selected_channels:
        return eeg_data, list(all_channels)
    indices = [list(all_channels).index(ch) for ch in selected_channels]
    return eeg_data[:, indices, :], [all_channels[i] for i in indices]


def _resolve_image_fields(image_path, image_root=None):
    basename = os.path.basename(str(image_path))
    resolved_path = os.path.join(image_root, basename) if image_root else str(image_path)
    return basename, resolved_path


def _coerce_optional_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _split_class_name(dirname):
    if "_" not in dirname:
        return dirname
    prefix, suffix = dirname.split("_", 1)
    return suffix if prefix.isdigit() else dirname


def _row_lookup_key(row):
    category_num = int(row["category_num"])
    category_img_num = _coerce_optional_int(row.get("category_img_num"))
    if category_img_num is not None:
        return ("category_img_num", category_num, category_img_num)
    basename, _ = _resolve_image_fields(row["image_path"])
    return ("image_basename", category_num, basename)


def _item_lookup_key(item):
    category_num = int(item["category_num"])
    category_img_num = _coerce_optional_int(item.get("category_img_num"))
    if category_img_num is not None:
        return ("category_img_num", category_num, category_img_num)
    return ("image_basename", category_num, item["image_basename"])


def _build_stimulus_index(rows, image_root=None):
    unique = {}
    for row in rows:
        basename, resolved_path = _resolve_image_fields(row["image_path"], image_root=image_root)
        category_img_num = _coerce_optional_int(row.get("category_img_num"))
        key = (int(row["category_num"]), category_img_num, str(row["category_name"]), basename)
        unique[key] = {
            "category_num": int(row["category_num"]),
            "category": str(row["category_name"]),
            "category_img_num": category_img_num,
            "image_basename": basename,
            "image_path": resolved_path,
        }

    sorted_keys = sorted(
        unique.keys(),
        key=lambda item: (item[0], item[1] if item[1] is not None else float("inf"), item[3]),
    )
    object_ids = sorted({key[0] for key in sorted_keys})
    object_idx_map = {object_id: idx for idx, object_id in enumerate(object_ids)}
    per_object_counts = defaultdict(int)
    stimulus_index = []
    for stimulus_id, key in enumerate(sorted_keys):
        object_id, category_img_num, category, basename = key
        if category_img_num is None:
            image_idx = per_object_counts[object_id]
            per_object_counts[object_id] += 1
        else:
            image_idx = category_img_num
        stimulus_index.append(
            {
                "stimulus_id": stimulus_id,
                "object_idx": object_idx_map[object_id],
                "image_idx": image_idx,
                "category_num": object_id,
                "category": category,
                "category_img_num": category_img_num,
                "image_basename": basename,
                "image_path": unique[key]["image_path"],
            }
        )
    return stimulus_index


def _build_stimulus_index_from_image_set(rows, image_set_dir, split_name):
    split_dir = resolve_split_dir(image_set_dir, split_name)
    metadata_by_basename = {}
    metadata_by_category = defaultdict(dict)
    for row in rows:
        basename = os.path.basename(str(row["image_path"]))
        category_num = int(row["category_num"])
        category_name = str(row["category_name"])
        category_img_num = _coerce_optional_int(row.get("category_img_num"))
        metadata_by_basename[basename] = {
            "category_num": category_num,
            "category": category_name,
            "category_img_num": category_img_num,
        }
        if category_img_num is not None:
            metadata_by_category[category_name][category_img_num] = {
                "category_num": category_num,
                "category": category_name,
                "category_img_num": category_img_num,
            }

    stimulus_index = []
    stimulus_id = 0
    image_classes = sorted(os.listdir(split_dir))
    for object_idx, image_class in enumerate(image_classes):
        class_dir = os.path.join(split_dir, image_class)
        if not os.path.isdir(class_dir):
            continue
        class_name = _split_class_name(image_class)
        image_files = sorted(os.listdir(class_dir))
        for image_idx, image_file in enumerate(image_files):
            meta = metadata_by_category.get(class_name, {}).get(image_idx)
            if meta is None:
                meta = metadata_by_basename.get(image_file)
            if meta is None:
                raise ValueError(
                    f"Image '{image_file}' from '{split_dir}' could not be matched to Alljoined metadata "
                    f"for category '{class_name}' and image_idx={image_idx}. Please confirm the image set "
                    "matches THINGS-EEG-2."
                )
            stimulus_index.append(
                {
                    "stimulus_id": stimulus_id,
                    "object_idx": object_idx,
                    "image_idx": image_idx,
                    "category_num": meta["category_num"],
                    "category": meta["category"],
                    "category_img_num": meta.get("category_img_num"),
                    "image_basename": image_file,
                    "image_path": os.path.join(class_dir, image_file),
                }
            )
            stimulus_id += 1

    unique_metadata_count = len({_row_lookup_key(row) for row in rows})
    if len(stimulus_index) != unique_metadata_count:
        raise ValueError(
            f"Image-set-derived stimulus count ({len(stimulus_index)}) does not match metadata stimulus count "
            f"({unique_metadata_count}) for split '{split_name}'."
        )
    return stimulus_index


def _group_trials(rows, eeg_data, stimulus_lookup, image_root=None):
    grouped = defaultdict(list)
    if len(rows) != eeg_data.shape[0]:
        raise ValueError(f"Metadata rows ({len(rows)}) do not match EEG trials ({eeg_data.shape[0]}).")
    for row_idx, row in enumerate(rows):
        key = _row_lookup_key(row)
        grouped[stimulus_lookup[key]["stimulus_id"]].append(eeg_data[row_idx])
    return grouped


def _count_trials_per_stimulus(rows, stimulus_lookup):
    counts = defaultdict(int)
    for row in rows:
        key = _row_lookup_key(row)
        if key not in stimulus_lookup:
            raise ValueError(f"Metadata contains an unknown stimulus key for row '{key}'.")
        counts[stimulus_lookup[key]["stimulus_id"]] += 1
    return counts


def _determine_target_repetitions(min_count, requested_repetitions):
    if min_count is None:
        raise ValueError("No grouped trials found while building the adapter export.")
    if requested_repetitions is None or requested_repetitions <= 0:
        return min_count
    # We now allow requested_repetitions > min_count; the script will cycle/repeat
    # trials for stimuli that fall short of the target.
    return requested_repetitions


def _export_split(
    config,
    subjects,
    split_name,
    partition,
    eeg_key,
    output_eeg_dir,
    selected_channels,
    image_root,
    requested_repetitions,
    image_set_dir=None,
):
    dataset_times = None
    dataset_channels = None
    stimulus_index = None
    stimulus_lookup = None
    min_repetitions = None

    print(f"Scanning metadata for {len(subjects)} subjects in split '{split_name}'...")
    pbar = tqdm(subjects, desc=f"Scanning {split_name}")
    for subject in pbar:
        pbar.set_postfix({"subject": subject})
        metadata_path = config["metadata_path"].format(subject=subject)
        metadata = _filter_metadata(pd.read_parquet(metadata_path, engine="pyarrow"), partition)
        rows = metadata.to_dict("records")
        if stimulus_index is None:
            if image_set_dir:
                stimulus_index = _build_stimulus_index_from_image_set(rows, image_set_dir, split_name)
            else:
                stimulus_index = _build_stimulus_index(rows, image_root=image_root)
            stimulus_lookup = {_item_lookup_key(item): item for item in stimulus_index}
        counts = _count_trials_per_stimulus(rows, stimulus_lookup)
        if len(counts) != len(stimulus_index):
            raise ValueError(f"Subject '{subject}' is missing stimuli for split '{split_name}'.")
        subject_min = min(counts.values())
        min_repetitions = subject_min if min_repetitions is None else min(min_repetitions, subject_min)

    target_repetitions = _determine_target_repetitions(min_repetitions, requested_repetitions)
    print(f"Using {target_repetitions} repetitions for split '{split_name}'.")
    split_index_path = os.path.join(output_eeg_dir, f"stimulus_index_{split_name}.json")
    with open(split_index_path, "w") as f:
        json.dump(stimulus_index, f, indent=2)

    print(f"Saving {len(subjects)} subjects for split '{split_name}'...")
    pbar = tqdm(subjects, desc=f"Saving {split_name}")
    for subject in pbar:
        pbar.set_postfix({"subject": subject})
        metadata_path = config["metadata_path"].format(subject=subject)
        eeg_path = config[eeg_key].format(subject=subject)
        metadata = _filter_metadata(pd.read_parquet(metadata_path, engine="pyarrow"), partition)
        rows = metadata.to_dict("records")
        eeg_obj = _read_eeg(eeg_path)
        eeg_data = eeg_obj["preprocessed_eeg_data"]
        eeg_data, chosen_channels = _select_channels(eeg_data, selected_channels, eeg_obj["ch_names"])
        if dataset_channels is None:
            dataset_channels = chosen_channels
        if dataset_times is None and "times" in eeg_obj:
            dataset_times = np.asarray(eeg_obj["times"]).tolist()
        grouped = _group_trials(rows, eeg_data, stimulus_lookup, image_root=image_root)
        subject_array = []
        for item in stimulus_index:
            trials = grouped[item["stimulus_id"]]
            if len(trials) >= target_repetitions:
                final_trials = trials[:target_repetitions]
            else:
                # Cycle/repeat trials to reach target_repetitions
                final_trials = (trials * (target_repetitions // len(trials) + 1))[:target_repetitions]
            
            stacked = np.stack(final_trials, axis=0).astype(np.float32, copy=False)
            subject_array.append(stacked)
        subject_array = np.stack(subject_array, axis=0)
        subject_dir = os.path.join(output_eeg_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)
        np.save(
            os.path.join(subject_dir, "train.npy" if split_name == "train" else "test.npy"),
            {
                "preprocessed_eeg_data": subject_array,
                "ch_names": dataset_channels,
                "times": dataset_times,
            },
            allow_pickle=True,
        )

    return {
        "stimulus_index_path": os.path.basename(split_index_path),
        "num_stimuli": len(stimulus_index),
        "num_repetitions": target_repetitions,
        "channels": dataset_channels,
        "times": dataset_times,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Alljoined-1.6M into a Neurobridge-compatible flat adapter format.")
    parser.add_argument("--dataset_profile", type=str, default="alljoined", choices=["alljoined"])
    parser.add_argument("--dataset_config_path", type=str, default=None)
    parser.add_argument("--raw_config_path", type=str, default=None, help="Override the ENIGMA raw dataset config path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Adapter output root. Defaults to the dataset profile output dir.")
    parser.add_argument("--subjects", nargs="+", type=int, default=None, help="Subset of subject IDs to export, e.g. 1 2 3.")
    parser.add_argument("--image_root", type=str, default=None, help="Optional root used to resolve image basenames to absolute image paths.")
    parser.add_argument("--image_set_dir", type=str, default=None, help="THINGS image set directory used to force the exact feature ordering from extract_feature.py.")
    parser.add_argument("--train_repetitions", type=int, default=0, help="Training repetitions per stimulus (0 uses the minimum available count).")
    parser.add_argument("--test_repetitions", type=int, default=0, help="Test repetitions per stimulus (0 uses the minimum available count).")
    parser.add_argument("--mvnn", action="store_true", help="Apply THINGS-style MVNN per subject after exporting train/test arrays.")
    parser.add_argument("--existing_output_only", action="store_true", help="Skip raw export and apply MVNN to an existing adapter export under --output_dir.")
    args = parser.parse_args()

    profile = load_dataset_profile(args.dataset_profile, args.dataset_config_path)
    output_root = os.path.abspath(args.output_dir or profile.get("adapter_output_dir", "./data/alljoined_adapter"))
    output_eeg_dir = os.path.join(output_root, "eeg")
    os.makedirs(output_eeg_dir, exist_ok=True)
    requested_subjects = None
    if args.subjects:
        requested_subjects = {f"sub-{subject_id:02d}" for subject_id in args.subjects}

    if args.existing_output_only:
        if not args.mvnn:
            raise ValueError("--existing_output_only currently requires --mvnn.")
        subjects = sorted(
            subject
            for subject in os.listdir(output_eeg_dir)
            if subject.startswith("sub-") and os.path.isdir(os.path.join(output_eeg_dir, subject))
        )
        if requested_subjects is not None:
            subjects = [subject for subject in subjects if subject in requested_subjects]
        if not subjects:
            raise ValueError(f"No exported subjects found under '{output_eeg_dir}'.")
        _apply_subject_mvnn(output_eeg_dir, subjects)
        _update_export_metadata(output_eeg_dir, "mvnn")
        print(f"Applied MVNN to existing Alljoined adapter export in '{output_root}' for {len(subjects)} subjects.")
        return

    raw_config_path = args.raw_config_path or profile.get("raw_config_path")
    if raw_config_path is None:
        raise ValueError("A raw Alljoined config path is required.")
    config = _load_config(raw_config_path)
    image_set_dir = args.image_set_dir or profile.get("image_set_dir")

    subjects = config["subjects"]
    if requested_subjects is not None:
        subjects = [subject for subject in subjects if subject in requested_subjects]
    selected_channels = profile.get("selected_channels", config.get("channels", []))

    train_meta = _export_split(
        config,
        subjects,
        split_name="train",
        partition="stim_train",
        eeg_key="eeg_trials_train",
        output_eeg_dir=output_eeg_dir,
        selected_channels=selected_channels,
        image_root=args.image_root,
        requested_repetitions=args.train_repetitions,
        image_set_dir=image_set_dir,
    )
    test_meta = _export_split(
        config,
        subjects,
        split_name="test",
        partition="stim_test",
        eeg_key="eeg_trials_test",
        output_eeg_dir=output_eeg_dir,
        selected_channels=selected_channels,
        image_root=args.image_root,
        requested_repetitions=args.test_repetitions,
        image_set_dir=image_set_dir,
    )

    if args.mvnn:
        _apply_subject_mvnn(output_eeg_dir, subjects)
        _update_export_metadata(output_eeg_dir, "mvnn")

    with open(os.path.join(output_eeg_dir, "info.json"), "w") as f:
        json.dump(
            {
                "ch_names": train_meta["channels"],
                "times": train_meta["times"],
                "normalization": "mvnn" if args.mvnn else None,
            },
            f,
            indent=2,
        )
    with open(os.path.join(output_eeg_dir, "dataset_info.json"), "w") as f:
        json.dump(
            {
                "dataset_name": "alljoined",
                "layout": "flat",
                "stimulus_index_train": train_meta["stimulus_index_path"],
                "stimulus_index_test": test_meta["stimulus_index_path"],
                "num_stimuli_train": train_meta["num_stimuli"],
                "num_stimuli_test": test_meta["num_stimuli"],
                "num_repetitions_train": train_meta["num_repetitions"],
                "num_repetitions_test": test_meta["num_repetitions"],
                "image_set_dir": image_set_dir,
                "shared_image_feature_dir": profile.get("image_feature_dir"),
                "shared_text_feature_dir": profile.get("text_feature_dir"),
                "normalization": "mvnn" if args.mvnn else None,
            },
            f,
            indent=2,
        )

    print(f"Saved Alljoined adapter export to '{output_root}' for {len(subjects)} subjects.")


if __name__ == "__main__":
    main()
