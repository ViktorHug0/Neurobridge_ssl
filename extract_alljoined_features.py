import argparse
import os
import shutil

import numpy as np

from module.dataset_profiles import load_dataset_profile


def _flatten_stimulus_features(features: np.ndarray) -> np.ndarray:
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features.reshape(features.shape[0] * features.shape[1], features.shape[2])
    raise ValueError(f"Unsupported shared feature shape: {features.shape}")


def _load_stimulus_index_count(eeg_data_dir, split_name):
    import json

    with open(os.path.join(eeg_data_dir, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
    rel_path = dataset_info[f'stimulus_index_{split_name}']
    path = rel_path if os.path.isabs(rel_path) else os.path.join(eeg_data_dir, rel_path)
    with open(path, 'r') as f:
        return len(json.load(f))


def _validate_feature_bank(feature_dir, split_name, stimulus_count, filename):
    path = os.path.join(feature_dir, filename)
    features = np.load(path)
    flat = _flatten_stimulus_features(features)
    if flat.shape[0] != stimulus_count:
        raise ValueError(
            f"Feature bank '{path}' contains {flat.shape[0]} stimuli, but the Alljoined adapter expects {stimulus_count}."
        )
    return path


def _mirror_file(src, dst, mode):
    if os.path.lexists(dst):
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'symlink':
        os.symlink(src, dst)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(f'Unsupported mode: {mode}')


def main():
    parser = argparse.ArgumentParser(
        description='Validate and optionally mirror the shared THINGS feature bank for the Alljoined adapter.'
    )
    parser.add_argument('--dataset_profile', type=str, default='alljoined', choices=['alljoined'])
    parser.add_argument('--dataset_config_path', type=str, default=None)
    parser.add_argument('--eeg_data_dir', type=str, default=None)
    parser.add_argument('--source_image_feature_dir', type=str, default=None)
    parser.add_argument('--source_text_feature_dir', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None, help='Optional target root for mirrored links/copies.')
    parser.add_argument('--mode', type=str, choices=['validate', 'symlink', 'copy'], default='validate')
    parser.add_argument('--skip_text', action='store_true')
    args = parser.parse_args()

    profile = load_dataset_profile(args.dataset_profile, args.dataset_config_path)
    eeg_data_dir = args.eeg_data_dir or profile['eeg_data_dir']
    image_feature_dir = args.source_image_feature_dir or profile['image_feature_dir']
    text_feature_dir = args.source_text_feature_dir or profile.get('text_feature_dir')

    train_count = _load_stimulus_index_count(eeg_data_dir, 'train')
    test_count = _load_stimulus_index_count(eeg_data_dir, 'test')

    train_image_path = _validate_feature_bank(image_feature_dir, 'train', train_count, 'image_train.npy')
    test_image_path = _validate_feature_bank(image_feature_dir, 'test', test_count, 'image_test.npy')
    print(f'Validated shared image features: train={train_image_path}, test={test_image_path}')

    train_text_path = None
    test_text_path = None
    if text_feature_dir and not args.skip_text:
        train_text_path = _validate_feature_bank(text_feature_dir, 'train', train_count, 'train.npy')
        test_text_path = _validate_feature_bank(text_feature_dir, 'test', test_count, 'test.npy')
        print(f'Validated shared text features: train={train_text_path}, test={test_text_path}')

    if args.mode == 'validate':
        return

    if not args.output_root:
        raise ValueError('--output_root is required when mode is symlink or copy.')

    image_out = os.path.join(args.output_root, 'image_feature')
    _mirror_file(train_image_path, os.path.join(image_out, 'image_train.npy'), args.mode)
    _mirror_file(test_image_path, os.path.join(image_out, 'image_test.npy'), args.mode)

    if train_text_path and test_text_path:
        text_out = os.path.join(args.output_root, 'text_feature')
        _mirror_file(train_text_path, os.path.join(text_out, 'train.npy'), args.mode)
        _mirror_file(test_text_path, os.path.join(text_out, 'test.npy'), args.mode)


if __name__ == '__main__':
    main()
