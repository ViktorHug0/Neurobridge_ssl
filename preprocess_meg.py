import os
import shutil
import argparse
import gc

import mne
import pandas as pd
import numpy as np

from module.util import dump_pretty


CHUNK_EPOCHS = 128
NORMALIZE_BATCH_OBJECTS = 8
EPS = 1e-8
EXPECTED_TRAIN_SHAPE = (1654, 12, 1)
EXPECTED_TEST_SHAPE = (200, 1, 12)


def precision_to_dtype(precision: str):
    if precision == "fp16":
        return np.float16
    if precision == "fp32":
        return np.float32
    if precision == "fp64":
        return np.float64
    raise ValueError(f"Unsupported precision: {precision}")


def split_epoch_indices(events: np.ndarray, image_concept_list: list[int]):
    sorted_indices = np.argsort(events[:, 2])
    sorted_event_ids = events[sorted_indices, 2]

    valid_mask = sorted_event_ids != 999999
    valid_sorted_indices = sorted_indices[valid_mask]
    valid_event_ids = sorted_event_ids[valid_mask]

    unique_event_ids, counts = np.unique(valid_event_ids, return_counts=True)
    zs_event_ids = unique_event_ids[counts == 12]

    test_mask = np.isin(valid_event_ids, zs_event_ids)
    test_indices = valid_sorted_indices[test_mask]

    training_indices = valid_sorted_indices[~test_mask]
    training_event_ids = valid_event_ids[~test_mask]
    test_set_categories = {image_concept_list[event_id - 1] for event_id in zs_event_ids}
    keep_training_mask = np.array(
        [image_concept_list[event_id - 1] not in test_set_categories for event_id in training_event_ids],
        dtype=bool,
    )
    training_indices = training_indices[keep_training_mask]

    return training_indices, test_indices, zs_event_ids


def fill_flat_memmap(epochs, indices, flat_memmap, n_times_out, target_dtype, track_stats=False):
    channel_sum = None
    channel_sumsq = None
    if track_stats:
        n_channels = flat_memmap.shape[1]
        channel_sum = np.zeros((n_channels,), dtype=np.float64)
        channel_sumsq = np.zeros((n_channels,), dtype=np.float64)

    for start in range(0, len(indices), CHUNK_EPOCHS):
        end = min(start + CHUNK_EPOCHS, len(indices))
        chunk_indices = indices[start:end]
        chunk = epochs[chunk_indices].get_data(tmin=0.0, tmax=1.0)[:, :, :n_times_out]

        if track_stats:
            chunk64 = chunk.astype(np.float64, copy=False)
            channel_sum += chunk64.sum(axis=(0, 2))
            channel_sumsq += np.square(chunk64).sum(axis=(0, 2))

        flat_memmap[start:end] = chunk.astype(target_dtype, copy=False)

        del chunk
        if track_stats:
            del chunk64

    flat_memmap.flush()
    return channel_sum, channel_sumsq


def normalize_memmap_inplace(array_memmap, mean, std, target_dtype):
    compute_dtype = np.float64 if target_dtype == np.float64 else np.float32
    for start in range(0, array_memmap.shape[0], NORMALIZE_BATCH_OBJECTS):
        end = min(start + NORMALIZE_BATCH_OBJECTS, array_memmap.shape[0])
        batch = np.asarray(array_memmap[start:end], dtype=compute_dtype)
        batch = (batch - mean) / std
        array_memmap[start:end] = batch.astype(target_dtype, copy=False)
        del batch

    array_memmap.flush()


if __name__ == "__main__":
    # Get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='./data/things_meg', type=str, help="raw data directory")
    parser.add_argument('--output_meg_dir', default='./data/things_meg/preprocessed_meg', type=str, help="output directory")
    parser.add_argument('--output_image_dir', default='./data/things_meg/image_set', type=str, help="output directory")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp64", "fp32", "fp16"], help="precision: float32 or float16")
    parser.add_argument('--zscore', action="store_true")
    args = parser.parse_args()

    # Print input arguments
    print('\nInput arguments:')
    for key, val in vars(args).items():
        print('{:20} {}'.format(key, val))

    meg_dir = os.path.join(args.raw_data_dir, 'raw_meg')
    image_dir = os.path.join(args.raw_data_dir, 'image_set', 'object_images')

    image_concept_list = []
    total_images = 0
    concept_idx = 1
    file_list = []

    for concept_name in sorted(os.listdir(image_dir)):
        concept_path = os.path.join(image_dir, concept_name)
        if os.path.isdir(concept_path):
            files = sorted(os.listdir(concept_path))
            total_images += len(files)
            file_list.extend([os.path.join(concept_path, f) for f in files])
            image_concept_list.extend([concept_idx] * len(files))
            concept_idx += 1

    print(f"Number of concepts: {concept_idx - 1}")
    print(f"There are a total of {total_images} files in the image directory '{image_dir}'")

    target_dtype = precision_to_dtype(args.precision)

    for sub_id in range(1, 5):
        print(f"\nProcessing Subject {sub_id}...")
        save_dir = os.path.join(args.output_meg_dir, 'sub-'+format(sub_id, '02'))
        train_path = os.path.join(save_dir, 'train.npy')
        test_path = os.path.join(save_dir, 'test.npy')
        subject_done = os.path.isfile(train_path) and os.path.isfile(test_path)
        if subject_done:
            if sub_id != 4:
                print(f"Subject {sub_id} already processed, skipping...")
                continue
            print(f"Subject {sub_id} already processed, but re-processing for meta information...")

        fif_file = os.path.join(meg_dir, f"sub-{format(sub_id,'02')}", f"preprocessed_P{sub_id}-epo.fif")
        epochs = mne.read_epochs(fif_file, preload=False)

        print("Num of events:", len(epochs.events))
        ch_names = epochs.ch_names
        print("Num of Channels:", len(ch_names))
        print("Channel Names:", ch_names)

        training_indices, test_indices, zs_event_ids = split_epoch_indices(epochs.events, image_concept_list)
        print("Zero-shot Event IDs:", zs_event_ids)
        print("Number of events in the training set:", len(training_indices))
        print("Number of events in the zero-shot test set:", len(test_indices))

        n_channels = len(ch_names)
        cropped_times = epochs.times[(epochs.times >= 0.0) & (epochs.times <= 1.0)]
        n_times_out = len(cropped_times) - 1
        times = np.round(cropped_times[:n_times_out], 3)

        if len(training_indices) != np.prod(EXPECTED_TRAIN_SHAPE):
            raise ValueError(
                f"Unexpected training epoch count for subject {sub_id}: "
                f"got {len(training_indices)}, expected {np.prod(EXPECTED_TRAIN_SHAPE)}"
            )
        if len(test_indices) != np.prod(EXPECTED_TEST_SHAPE):
            raise ValueError(
                f"Unexpected test epoch count for subject {sub_id}: "
                f"got {len(test_indices)}, expected {np.prod(EXPECTED_TEST_SHAPE)}"
            )

        train_shape = (*EXPECTED_TRAIN_SHAPE, n_channels, n_times_out)
        test_shape = (*EXPECTED_TEST_SHAPE, n_channels, n_times_out)

        os.makedirs(save_dir, exist_ok=True)
        for output_path in (train_path, test_path):
            if os.path.exists(output_path):
                os.remove(output_path)

        train_data = np.lib.format.open_memmap(train_path, mode='w+', dtype=target_dtype, shape=train_shape)
        test_data = np.lib.format.open_memmap(test_path, mode='w+', dtype=target_dtype, shape=test_shape)
        flat_train = train_data.reshape(-1, n_channels, n_times_out)
        flat_test = test_data.reshape(-1, n_channels, n_times_out)

        train_sum, train_sumsq = fill_flat_memmap(
            epochs,
            training_indices,
            flat_train,
            n_times_out,
            target_dtype,
            track_stats=args.zscore,
        )
        fill_flat_memmap(
            epochs,
            test_indices,
            flat_test,
            n_times_out,
            target_dtype,
            track_stats=False,
        )

        if args.zscore:
            sample_count = flat_train.shape[0] * n_times_out
            mean_per_channel = train_sum / sample_count
            var_per_channel = np.maximum(train_sumsq / sample_count - np.square(mean_per_channel), EPS)
            mean_train = mean_per_channel.reshape(1, 1, 1, n_channels, 1)
            std_train = np.sqrt(var_per_channel).reshape(1, 1, 1, n_channels, 1)

            normalize_memmap_inplace(train_data, mean_train, std_train, target_dtype)
            normalize_memmap_inplace(test_data, mean_train, std_train, target_dtype)
            print("Z-score normalization applied.")

        print("train data shape:", train_data.shape)
        print("test data shape:", test_data.shape)

        train_data.flush()
        test_data.flush()
        del flat_train, flat_test, train_data, test_data, epochs
        gc.collect()

    # save info file
    info_dict = {
        "ch_names": ch_names,
        "times": times.tolist(),
        "baseline_duration": 0.2,
        "after_duration": 1.0,
        "normalization": "zscore" if args.zscore else "none",
        "sfreq": 200,
        "precision": args.precision,
        "train_data_shape": train_shape,
        "test_data_shape": test_shape
    }
    with open(os.path.join(args.output_meg_dir, "info.json"), "w", encoding="utf-8") as f:
        dump_pretty(info_dict, f, indent=4, ensure_ascii=False)

    ################################################################################
    # processing image files

    path = os.path.join(args.raw_data_dir, "sourcedata/sample_attributes_P1.csv")
    df = pd.read_csv(path)

    print(f"CSV file '{path}' has been loaded, containing {len(df)} rows.")

    train_df = df[(df['category_nr'] <= 1854) & (df['test_image_nr'].isna())]
    test_df = df[(df['category_nr'] <= 1854) & (df['trial_type'] == 'test')]
    train_df = train_df[~train_df['category_nr'].isin(test_df['category_nr'])]

    train_df = train_df.sort_values(by=['category_nr'])
    test_df = test_df.sort_values(by=['category_nr'])

    base_dir = os.path.join(args.raw_data_dir, 'image_set/object_images')

    print(f"Training set has {len(train_df)} images.")
    dst_dir = os.path.join(args.output_image_dir, 'train_images')
    for _, row in train_df.iterrows():
        category_nr = str(row['category_nr']).zfill(5)
        src_path = row['image_path'].replace('images_meg/', base_dir + '/')
        concept = os.path.basename(os.path.dirname(src_path))
        dst_path = f"{dst_dir}/{category_nr}_{concept}/{os.path.basename(src_path)}"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    print(f"Test set has {len(test_df)} images.")
    dst_dir = os.path.join(args.output_image_dir, 'test_images')
    for _, row in test_df.iterrows():
        category_nr = str(row['category_nr']).zfill(5)
        src_path = row['image_path'].replace('images_test_meg/', base_dir + '/')
        concept = os.path.basename(src_path).rsplit('_', 1)[0]
        src_path = os.path.join(os.path.dirname(src_path), concept, os.path.basename(src_path))
        dst_path = f"{dst_dir}/{category_nr}_{concept}/{os.path.basename(src_path)}"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)