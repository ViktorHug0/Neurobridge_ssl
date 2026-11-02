import glob
import math
import os
import json
import random
import hashlib
import tempfile
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import BatchSampler, Dataset
from tqdm import tqdm
import gc


def _resolve_eeg_file(subject_dir: str, train: bool) -> str:
    """
    Support both this repo's convention (train.npy/test.npy) and NICE-EEG's
    convention (preprocessed_eeg_training.npy/preprocessed_eeg_test.npy).
    """
    if train:
        candidates = ["train.npy", "preprocessed_eeg_training.npy"]
    else:
        candidates = ["test.npy", "preprocessed_eeg_test.npy"]

    for name in candidates:
        p = os.path.join(subject_dir, name)
        if os.path.isfile(p):
            return p

    if not train:
        matches = sorted(glob.glob(os.path.join(subject_dir, "test*.npy")))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find {'train' if train else 'test'} EEG file in '{subject_dir}'. "
        f"Tried: {', '.join(candidates)}"
    )


def _load_eeg_container(path: str):
    """
    np.load may return:
    - ndarray (this repo)
    - dict with keys like 'preprocessed_eeg_data', 'ch_names', 'times' (NICE-EEG)
    """
    obj = np.load(path, allow_pickle=True)
    # NICE-EEG stores a dict directly
    if isinstance(obj, dict):
        return obj
    # Some npy dicts come back as 0-d object arrays
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    return obj


def _eeg_cache_key(
    *,
    subject_id: int,
    train: bool,
    average: bool,
    selected_channels: list[str],
    time_window: list[int],
) -> str:
    """
    Build a stable key for caching the *processed* EEG array for a single subject.
    We intentionally exclude augmentation/transforms (which may be stochastic).
    """
    payload = {
        "v": 1,
        "subject_id": int(subject_id),
        "train": bool(train),
        "average": bool(average),
        "selected_channels": list(selected_channels),
        "time_window": list(time_window),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _eeg_cache_path(eeg_data_dir: str, key: str) -> str:
    cache_dir = os.path.join(eeg_data_dir, ".cache", "neurobridge_eeg")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.npy")


def _fp16_memmap(array: np.ndarray, path: str) -> np.ndarray:
    """An fp16 copy of `array` on disk, read back as a memmap.

    `array.astype(np.float16)` on the memmap opened above would pull the whole thing into
    anonymous RAM and keep it resident for the life of the dataset -- and train, val and test
    each pay it. That is why un-averaged runs needed ~94 GB while NeuralBench, which cuts epochs
    lazily from disk, holds ~30 GB for the same data. The conversion is chunked so neither the
    source nor the destination is ever fully resident.
    """
    if not os.path.isfile(path):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".npy", dir=os.path.dirname(path), delete=False
            ) as handle:
                tmp = handle.name
            out = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.float16, shape=array.shape
            )
            for start in range(0, array.shape[0], 64):
                out[start:start + 64] = array[start:start + 64]
            out.flush()
            del out
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.remove(tmp)
    return np.load(path, mmap_mode="r")


def _standardize_eeg_array(eeg_obj, train: bool, num_images_per_object: int = 10) -> np.ndarray:
    """
    Convert various EEG array layouts into this repo's internal format:
      train: (num_objects, num_images_per_object, num_reps, num_channels, num_time)
      test:  (num_objects, 1, num_reps, num_channels, num_time)
    """
    # NICE-EEG dict
    if isinstance(eeg_obj, dict):
        if "preprocessed_eeg_data" in eeg_obj:
            x = eeg_obj["preprocessed_eeg_data"]
        else:
            raise KeyError(f"Unsupported EEG dict keys: {list(eeg_obj.keys())}")
    else:
        x = eeg_obj

    if not isinstance(x, np.ndarray):
        raise TypeError(f"EEG data must be a numpy array after loading, got {type(x)}")

    # This repo format already:
    # - train: (1654, 10, 4, 63, 250)
    # - test:  (200, 1, 80, 63, 250)
    if x.ndim == 5:
        return x

    # NICE-EEG training: (16540, 4, 63, 250) == (1654*10, reps, ch, t)
    if train and x.ndim == 4:
        n = x.shape[0]
        if n % num_images_per_object != 0:
            raise ValueError(
                f"Cannot reshape training EEG of shape {x.shape}: "
                f"first dim {n} not divisible by num_images_per_object={num_images_per_object}"
            )
        num_objects = n // num_images_per_object
        return x.reshape(num_objects, num_images_per_object, x.shape[1], x.shape[2], x.shape[3])

    # NICE-EEG test: (200, 80, 63, 250) == (objects, reps, ch, t)
    if (not train) and x.ndim == 4:
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])

    raise ValueError(f"Unsupported EEG array shape: {x.shape}")


def _process_eeg_array(
    eeg_data: np.ndarray,
    *,
    selected_idx: list[int] | None,
    time_window: list[int],
    average: bool,
) -> np.ndarray:
    """
    Memory-aware EEG processing:
    - apply time window
    - optionally average over repetitions
    - optionally select channels

    Expected standardized shapes (from `_standardize_eeg_array`):
      train: (num_objects, num_images_per_object, num_reps, num_channels, num_time)
      test:  (num_objects, 1,                   num_reps, num_channels, num_time)
    Returns:
      if average: (num_objects, num_images_per_object, num_sel_channels, num_time_window)
      else:       (num_objects, num_images_per_object, num_reps, num_sel_channels, num_time_window)
    """
    if eeg_data.ndim != 5:
        raise ValueError(f"Expected standardized EEG with ndim=5, got shape={eeg_data.shape}")

    start, end = time_window
    end = min(end, eeg_data.shape[-1])
    if end <= start:
        raise ValueError(f"Invalid time_window={time_window} for EEG length={eeg_data.shape[-1]}")

    # If no channel selection is requested, keep all channels.
    if selected_idx is None:
        # NOTE: this can still be very large; we keep it vectorized and rely on caller to size resources.
        if average:
            # (obj, img, ch, t)
            return eeg_data[..., start:end].mean(axis=2, dtype=np.float32)
        # (obj, img, rep, ch, t)
        return eeg_data[..., start:end].astype(np.float32, copy=False)

    # Selected channels path: avoid materializing large intermediate arrays by processing one channel at a time.
    n_obj, n_img, n_rep, _, _ = eeg_data.shape
    n_t = end - start
    n_ch = len(selected_idx)

    if average:
        out = np.empty((n_obj, n_img, n_ch, n_t), dtype=np.float32)
        for j, ch in enumerate(selected_idx):
            # Slice is a view; mean allocates only (n_obj, n_img, n_t) ~ small.
            out[:, :, j, :] = eeg_data[:, :, :, ch, start:end].mean(axis=2, dtype=np.float32)
        return out

    out = np.empty((n_obj, n_img, n_rep, n_ch, n_t), dtype=np.float32)
    for j, ch in enumerate(selected_idx):
        out[:, :, :, j, :] = eeg_data[:, :, :, ch, start:end]
    return out


class EEGPreImageDataset(Dataset):
    def __init__(
        self, 
        subject_ids: list[int], 
        eeg_data_dir: str,
        selected_channels: list[str],
        time_window: list[int],
        image_feature_dir: str,
        text_feature_dir: str,
        average: bool = True, 
        _random: bool = False,
        eeg_transform=None,
        train=True,
        eeg_test_aug=False,
        frozen_eeg_prior=False,
        cross_subject_average=False,
        subject_mixup_within=False,
    ):
        super().__init__()
        self.subject_ids = subject_ids
        self.average = average
        self.random = _random
        self.eeg_transform = eeg_transform
        self.augment_indices = []
        self.image_feature_dir = image_feature_dir
        self.text_feature_dir = text_feature_dir
        self.train = train
        self.eeg_test_aug = eeg_test_aug
        self.frozen_eeg_prior = frozen_eeg_prior
        self.cross_subject_average = cross_subject_average
        self.subject_mixup_within = subject_mixup_within
        if cross_subject_average and average:
            raise ValueError("cross_subject_average needs un-averaged reps; disable data_average.")
        if subject_mixup_within and average:
            raise ValueError("subject_mixup_within needs un-averaged reps; disable data_average.")
        self.info = {}
        info_json_path = os.path.join(eeg_data_dir, "info.json")
        if os.path.isfile(info_json_path):
            self.info = json.load(open(info_json_path, 'r'))
        
        # Things-EEG
        # o+p: ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        # o+p+t: ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2', 'FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8', 'TP7', 'TP9', 'TP10', 'TP8']
        # f: ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8']
        # c: ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']
        # t: ['FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8']
        # o: ['PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        # p: ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8']
        self.all_channels = self.info.get('ch_names', None)
        selected_idx = None
        if len(selected_channels) > 0 and self.all_channels is not None:
            selected_idx = [self.all_channels.index(ch) for ch in selected_channels]
        
        self.eeg_data_list = []
        for subject_id in tqdm(subject_ids, desc="Subjects", position=0):
            subject_dir = os.path.join(eeg_data_dir, f"sub-{subject_id:02}")
            eeg_data_path = _resolve_eeg_file(subject_dir, train)

            eeg_cache_key = _eeg_cache_key(
                subject_id=subject_id,
                train=train,
                average=self.average,
                selected_channels=selected_channels,
                time_window=time_window,
            )
            eeg_cache_path = _eeg_cache_path(eeg_data_dir, eeg_cache_key)
            if os.path.isfile(eeg_cache_path):
                # Processed LOSO caches are large (~1 GB per averaged subject) and
                # read-only in the main training path. Memory mapping lets several
                # independent models colocated on one GPU share the same page-cache
                # pages instead of each holding a private ~9 GB copy. Materialize a
                # writable array only for options that mutate the cached samples.
                eeg_data = np.load(eeg_cache_path, mmap_mode="r")
                if self.frozen_eeg_prior:
                    eeg_data = np.array(eeg_data, copy=True)
            else:
                eeg_container = _load_eeg_container(eeg_data_path)
                if isinstance(eeg_container, dict):
                    if self.all_channels is None and "ch_names" in eeg_container:
                        self.all_channels = list(eeg_container["ch_names"])
                        self.info["ch_names"] = self.all_channels
                    if "times" in eeg_container and "times" not in self.info:
                        self.info["times"] = eeg_container["times"]

                if len(selected_channels) > 0:
                    if self.all_channels is None:
                        raise ValueError(
                            "selected_channels was provided, but channel names could not be found. "
                            "Expected either '<eeg_data_dir>/info.json' with 'ch_names' or EEG dict files "
                            "containing 'ch_names'."
                        )
                    selected_idx = []
                    for ch in selected_channels:
                        if ch not in self.all_channels:
                            raise ValueError(
                                f"Channel '{ch}' not found in available channels: {self.all_channels}"
                            )
                        selected_idx.append(self.all_channels.index(ch))

                eeg_data = _standardize_eeg_array(eeg_container, train=train)
                eeg_data = _process_eeg_array(
                    eeg_data,
                    selected_idx=selected_idx,
                    time_window=time_window,
                    average=self.average,
                )

                # Atomic write to avoid corrupt cache files in concurrent runs.
                tmp_cache = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".npy", dir=os.path.dirname(eeg_cache_path), delete=False
                    ) as f:
                        np.save(f, eeg_data)
                        tmp_cache = f.name
                    os.replace(tmp_cache, eeg_cache_path)
                finally:
                    if tmp_cache is not None and os.path.exists(tmp_cache):
                        os.remove(tmp_cache)

            self.channels_num = eeg_data.shape[-2]
            
            # If it's the training set and a transform is specified, apply the EEG data transformation
            if self.frozen_eeg_prior:
                if self.eeg_transform is not None and (self.train or self.eeg_test_aug):
                    for object_idx in tqdm(range(eeg_data.shape[0]), desc="Objects", position=1, leave=False):
                        for image_idx in range(eeg_data.shape[1]):
                            if not self.average:
                                for repetition_idx in range(eeg_data.shape[2]):
                                    eeg_data[object_idx, image_idx, repetition_idx] = self.eeg_transform(eeg_data[object_idx, image_idx, repetition_idx])
                            else:
                                eeg_data[object_idx, image_idx] = self.eeg_transform(eeg_data[object_idx, image_idx])

            if not self.average and eeg_data.dtype != np.float16:
                # Backed by disk, not RAM -- see _fp16_memmap. Mixing/averaging upcasts per item.
                eeg_data = _fp16_memmap(eeg_data, eeg_cache_path[:-4] + "_fp16.npy")
            self.eeg_data_list.append(eeg_data)
        
        self.num_subjects = len(self.eeg_data_list)
        self.num_objects = eeg_data.shape[0]
        self.num_images_per_object = eeg_data.shape[1]
        if not self.average:
            self.num_repetitions = eeg_data.shape[2]
        self.num_channels = eeg_data.shape[-2]
        self.num_sample_points = eeg_data.shape[-1]
        self._image_group_indices = None
        
        split_file = "image_train.npy" if train else "image_test.npy"
        self.image_feature_path = os.path.join(self.image_feature_dir, split_file)
        if self.text_feature_dir is not None and self.text_feature_dir != '':
            if train:
                self.text_feature_path = os.path.join(self.text_feature_dir, "train.npy")
            else:
                self.text_feature_path = os.path.join(self.text_feature_dir, "test.npy")

        self.image_features = np.load(self.image_feature_path)
        self.feature_dim = self.image_features.shape[-1]
        if self.text_feature_dir is not None and self.text_feature_dir != '':
            self.text_features = np.load(self.text_feature_path)
    
    def __len__(self):
        if self.cross_subject_average:
            return self.num_objects * self.num_images_per_object
        if self.subject_mixup_within:
            return self.num_objects * self.num_images_per_object * self.num_subjects
        if self.average and self.random:
            length = self.num_objects * self.num_images_per_object
        elif self.average and not self.random:
            length = self.num_objects * self.num_images_per_object * self.num_subjects
        elif not self.average and self.random:
            length = self.num_objects * self.num_images_per_object
        else:  # not self.average and not self.random
            length = self.num_objects * self.num_images_per_object * self.num_repetitions * self.num_subjects
        return length
        
    def __getitem__(self, index):
        # When averaging, use default 0
        repetition_idx = 0

        # Cross-subject average: one synthetic trial per image = mean of a random-size
        # (Beta-drawn) random subset of the raw reps pooled across all training subjects.
        if self.cross_subject_average:
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            pool = np.concatenate(
                [self.eeg_data_list[s][object_idx][image_idx] for s in range(self.num_subjects)],
                axis=0,
            ).astype(np.float32, copy=False)  # (num_subjects * num_reps, ch, time); upcast fp16->fp32 to average
            eeg_data, _ = _random_average_trial(pool, 4, 36, 1.0, 1.0)
            if not self.frozen_eeg_prior and self.eeg_transform is not None and self.train:
                eeg_data = self.eeg_transform(eeg_data)
            image_feature = self.image_features[object_idx][image_idx]
            if self.text_feature_dir is not None and self.text_feature_dir != '':
                text_feature = self.text_features[object_idx][image_idx]
            else:
                text_feature = np.zeros((self.feature_dim,))
            return (
                torch.tensor(eeg_data, dtype=torch.float32),
                torch.tensor(image_feature, dtype=torch.float32),
                torch.tensor(text_feature, dtype=torch.float32),
                -1, object_idx, image_idx, 0,  # subject_id=-1: synthetic trial has no owner
            )

        # Within-subject Mixup: Beta convex combination of 2 of THIS subject's own reps for the same
        # stimulus (isolates whether the gain needs cross-subject pairing). One item per (subject,image).
        if self.subject_mixup_within:
            subject_idx = index // (self.num_objects * self.num_images_per_object)
            object_idx = (index % (self.num_objects * self.num_images_per_object)) // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            reps = self.eeg_data_list[subject_idx][object_idx][image_idx].astype(np.float32)  # (R, C, T)
            i, j = random.sample(range(reps.shape[0]), 2)
            lam = random.betavariate(0.5, 0.5)
            eeg_data = lam * reps[i] + (1.0 - lam) * reps[j]
            image_feature = self.image_features[object_idx][image_idx]
            if self.text_feature_dir is not None and self.text_feature_dir != '':
                text_feature = self.text_features[object_idx][image_idx]
            else:
                text_feature = np.zeros((self.feature_dim,))
            return (
                torch.tensor(eeg_data, dtype=torch.float32),
                torch.tensor(image_feature, dtype=torch.float32),
                torch.tensor(text_feature, dtype=torch.float32),
                self.subject_ids[subject_idx], object_idx, image_idx, 0,
            )

        # Average & Random: Loop through objects and images, random subject
        if self.average and self.random:
            subject_idx = random.randint(0, len(self.subject_ids) - 1)
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx]
        
        # Average & Not Random: Loop through all subjects and images
        elif self.average and not self.random:
            subject_idx = index // (self.num_objects * self.num_images_per_object)
            object_idx = (index % (self.num_objects * self.num_images_per_object)) // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx]
        
        # Not Average & Random: Loop through objects and images, random subject and repetition
        elif not self.average and self.random:
            subject_idx = random.randint(0, self.num_subjects - 1)
            repetition_idx = random.randint(0, self.num_repetitions - 1)
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx][repetition_idx]
        
        # Not Average & Not Random: Complete loop through all EEG data
        else:
            subject_idx = index // (self.num_objects * self.num_images_per_object * self.num_repetitions)
            object_idx = (index % (self.num_objects * self.num_images_per_object * self.num_repetitions)) // (self.num_images_per_object * self.num_repetitions)
            image_idx = (index % (self.num_images_per_object * self.num_repetitions)) // self.num_repetitions
            repetition_idx = index % self.num_repetitions
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx][repetition_idx]
            if self.train:  # ponytail: RNG parity with the removed rep-avg/xsubj/noise/band-swap draws (one each)
                for _ in range(4):
                    random.random()
        
        # If it's the training set and a transform is specified, apply the EEG data transformation
        if not self.frozen_eeg_prior:
            if self.eeg_transform is not None and (self.train or self.eeg_test_aug):
                # copy: un-averaged trials are a read-only fp16 memmap and transforms write in place
                eeg_data = self.eeg_transform(np.array(eeg_data, dtype=np.float32))
        
        image_feature = self.image_features[object_idx][image_idx]

        if self.text_feature_dir is not None and self.text_feature_dir != '':
            text_feature = self.text_features[object_idx][image_idx]
        else:
            text_feature = np.zeros((self.feature_dim,))
        
        return (
            torch.tensor(eeg_data, dtype=torch.float32), 
            torch.tensor(image_feature, dtype=torch.float32), 
            torch.tensor(text_feature, dtype=torch.float32),
            self.subject_ids[subject_idx], 
            object_idx, 
            image_idx, 
            repetition_idx
        )

    def decode_index(self, index: int):
        repetition_idx = 0

        if self.subject_mixup_within:
            subject_idx = index // (self.num_objects * self.num_images_per_object)
            object_idx = (index % (self.num_objects * self.num_images_per_object)) // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            return self.subject_ids[subject_idx], object_idx, image_idx, 0
        if self.average and self.random:
            subject_idx = None
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
        elif self.average and not self.random:
            subject_idx = index // (self.num_objects * self.num_images_per_object)
            object_idx = (index % (self.num_objects * self.num_images_per_object)) // self.num_images_per_object
            image_idx = index % self.num_images_per_object
        elif not self.average and self.random:
            subject_idx = None
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
        else:
            subject_idx = index // (self.num_objects * self.num_images_per_object * self.num_repetitions)
            object_idx = (index % (self.num_objects * self.num_images_per_object * self.num_repetitions)) // (self.num_images_per_object * self.num_repetitions)
            image_idx = (index % (self.num_images_per_object * self.num_repetitions)) // self.num_repetitions
            repetition_idx = index % self.num_repetitions

        subject_id = None if subject_idx is None else self.subject_ids[subject_idx]
        return subject_id, object_idx, image_idx, repetition_idx

    def get_image_group_indices(self):
        if self._image_group_indices is None:
            if self.random:
                raise ValueError("Grouped image sampling requires deterministic dataset indexing; disable data_random.")

            image_group_indices = defaultdict(list)
            for index in range(len(self)):
                _, object_idx, image_idx, _ = self.decode_index(index)
                image_group_indices[(object_idx, image_idx)].append(index)
            self._image_group_indices = dict(image_group_indices)

        return self._image_group_indices


def _random_average_trial(pool, kmin, kmax, beta_a, beta_b):
    """Mean of a random-size (Beta-drawn) random subset of pool rows. pool: (P, ...).
    Returns (averaged_trial, k). Uses python `random` so DataLoader workers reseed it."""
    P = pool.shape[0]
    kmin = max(1, min(kmin, P))
    kmax = max(kmin, min(kmax, P))
    u = random.betavariate(beta_a, beta_b)
    k = max(kmin, min(int(round(kmin + u * (kmax - kmin))), kmax))
    return pool[random.sample(range(P), k)].mean(axis=0), k


class RandomGaussianNoise:
    """
    Adds random Gaussian noise to the EEG signal.
    std indicates the standard deviation of the noise.
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, size=eeg_data.shape)
        return eeg_data + noise


class RandomSmooth:
    """
    A simple smoothing operation, which can be understood as a simple convolution / moving average along the time axis.
    kernel_size indicates the size of the moving average kernel.
    """
    def __init__(self, kernel_size=5, smooth_prob=0.5):
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        ch, time_len = eeg_data.shape
        smoothed = np.copy(eeg_data)
        # Ensure kernel size is odd
        for c in range(ch):
            if np.random.rand() < self.smooth_prob:
                for t in range(time_len):
                    left = max(0, t - self.kernel_size // 2)
                    right = min(time_len, t + self.kernel_size // 2 + 1)
                    smoothed[c, t] = np.mean(eeg_data[c, left:right])
        return smoothed


class GroupedImageBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, samples_per_image=4, drop_last=True, seed=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if samples_per_image <= 0:
            raise ValueError("samples_per_image must be positive.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        image_groups = dataset.get_image_group_indices()
        if not image_groups:
            raise ValueError("No grouped image indices found in dataset.")

        self.group_keys = list(image_groups.keys())
        self.image_groups = {key: list(indices) for key, indices in image_groups.items()}

        smallest_group = min(len(indices) for indices in self.image_groups.values())
        max_group = max(len(indices) for indices in self.image_groups.values())
        self.samples_per_image = min(samples_per_image, smallest_group)
        
        # Calculate how many passes through the unique images are needed to see all samples
        self.num_passes = math.ceil(max_group / self.samples_per_image)

        if self.samples_per_image <= 0:
            raise ValueError("Grouped sampler found an empty image group.")

        self.images_per_batch = batch_size // self.samples_per_image
        if self.images_per_batch <= 0:
            raise ValueError(
                f"batch_size={batch_size} is too small for samples_per_image={self.samples_per_image}."
            )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        batch = []
        groups_in_batch = 0

        for _ in range(self.num_passes):
            shuffled_keys = list(self.group_keys)
            rng.shuffle(shuffled_keys)

            for key in shuffled_keys:
                group_indices = list(self.image_groups[key])
                rng.shuffle(group_indices)
                batch.extend(group_indices[:self.samples_per_image])
                groups_in_batch += 1

                if groups_in_batch == self.images_per_batch:
                    yield batch
                    batch = []
                    groups_in_batch = 0

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_encounters = len(self.group_keys) * self.num_passes
        if self.drop_last:
            return total_encounters // self.images_per_batch
        return math.ceil(total_encounters / self.images_per_batch)


if __name__ == '__main__':
    import numpy as _np, statistics as _st
    _pool = _np.arange(36 * 3).reshape(36, 3).astype(float)
    for a, b in [(1, 1), (1, 3), (3, 1)]:
        ks = [_random_average_trial(_pool, 4, 36, a, b)[1] for _ in range(2000)]
        assert all(4 <= k <= 36 for k in ks), (a, b, min(ks), max(ks))
        assert _random_average_trial(_pool, 4, 36, a, b)[0].shape == (3,)
    lo = _st.mean(_random_average_trial(_pool, 4, 36, 1, 3)[1] for _ in range(5000))
    hi = _st.mean(_random_average_trial(_pool, 4, 36, 3, 1)[1] for _ in range(5000))
    assert lo < hi, (lo, hi)  # a<b skews to fewer recordings, a>b to more
    print(f'cross_subject_average self-check OK (mean k: low-skew={lo:.1f} < high-skew={hi:.1f})')

