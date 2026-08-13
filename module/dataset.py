import glob
import os
import json
import random
import hashlib
import tempfile
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import gc

from module.util import _inv_sqrt_cov


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


def _load_image_target_blocks(target_dirs: list[str], split_file: str):
    """Load one primary image target plus optional normalized auxiliary blocks."""
    if not target_dirs:
        raise ValueError("At least one image feature directory is required")
    dims = [
        np.load(os.path.join(directory, split_file), mmap_mode="r").shape[-1]
        for directory in target_dirs
    ]
    if len(target_dirs) == 1:
        return np.load(os.path.join(target_dirs[0], split_file)), dims

    blocks = []
    for target_index, directory in enumerate(target_dirs):
        block = np.load(os.path.join(directory, split_file)).astype(np.float32)
        if target_index > 0:
            block /= np.clip(
                np.linalg.norm(block, axis=-1, keepdims=True), 1e-8, None
            )
        blocks.append(block)
    return np.concatenate(blocks, axis=-1), dims


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
        image_aug: bool, 
        aug_image_feature_dirs: list[str], 
        average: bool = True, 
        _random: bool = False,
        eeg_transform=None,
        train=True,
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=False,
        cross_subject_average=False,
        xavg_beta_a=1.0,
        xavg_beta_b=1.0,
        xavg_kmin=4,
        xavg_kmax=36,
        subject_mixup_within=False,
        within_mix_alpha=0.5,
        subject_ea_align=False,
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
        self.image_aug = image_aug
        self.image_test_aug = image_test_aug
        self.eeg_test_aug = eeg_test_aug
        self.frozen_eeg_prior = frozen_eeg_prior
        self.cross_subject_average = cross_subject_average
        self.xavg_beta_a = xavg_beta_a
        self.xavg_beta_b = xavg_beta_b
        self.xavg_kmin = xavg_kmin
        self.xavg_kmax = xavg_kmax
        self.subject_mixup_within = subject_mixup_within
        self.within_mix_alpha = within_mix_alpha
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
                eeg_data = np.load(eeg_cache_path)
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

            if subject_ea_align:
                # Euclidean Alignment: whiten each subject by its own channel-covariance R^-1/2 so
                # subjects share a common covariance before averaging ("a smarter cross-subject mean").
                Xf = eeg_data.reshape(-1, eeg_data.shape[-2], eeg_data.shape[-1]).astype(np.float32)
                R = np.einsum('nct,ndt->cd', Xf, Xf) / (Xf.shape[0] * Xf.shape[-1])
                W = _inv_sqrt_cov(R)
                eeg_data = np.einsum('cd,...dt->...ct', W, eeg_data.astype(np.float32)).astype(eeg_data.dtype, copy=False)
            if not self.average:
                # Un-averaged reps for 9 subjects ~= 35GB fp32, OOMs a 31GB box; fp16 halves it to
                # ~18GB (fork-workers share it copy-on-write). Mixing/averaging upcasts to fp32 per item.
                eeg_data = eeg_data.astype(np.float16)
            self.eeg_data_list.append(eeg_data)
        
        self.num_subjects = len(self.eeg_data_list)
        self.num_objects = eeg_data.shape[0]
        self.num_images_per_object = eeg_data.shape[1]
        if not self.average:
            self.num_repetitions = eeg_data.shape[2]
        self.num_channels = eeg_data.shape[-2]
        self.num_sample_points = eeg_data.shape[-1]
        self._image_group_indices = None
        
        if self.image_aug:
            self.aug_image_features = []
            for aug_image_feature_dir in aug_image_feature_dirs:
                if train:
                    aug_image_feature_path = os.path.join(aug_image_feature_dir, "train.npy")
                else:
                    aug_image_feature_path = os.path.join(aug_image_feature_dir, "test.npy")
                aug_image_feature = np.load(aug_image_feature_path)
                self.aug_image_features.append(aug_image_feature)
        
        split_file = "image_train.npy" if train else "image_test.npy"
        # A comma-separated image_feature_dir means several image targets at once. Preserve
        # the primary block byte-for-byte so its training path remains matched to a normal
        # single-target run. Training-only auxiliary blocks are L2-normalized before they are
        # concatenated; target_dims tells the model where to split them again.
        target_dirs = [d for d in str(self.image_feature_dir).split(',') if d]
        self.image_feature_path = os.path.join(target_dirs[0], split_file)
        if self.text_feature_dir is not None and self.text_feature_dir != '':
            if train:
                self.text_feature_path = os.path.join(self.text_feature_dir, "train.npy")
            else:
                self.text_feature_path = os.path.join(self.text_feature_dir, "test.npy")

        self.image_features, self.target_dims = _load_image_target_blocks(
            target_dirs, split_file
        )
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
            eeg_data, _ = _random_average_trial(
                pool, self.xavg_kmin, self.xavg_kmax, self.xavg_beta_a, self.xavg_beta_b
            )
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
            lam = random.betavariate(self.within_mix_alpha, self.within_mix_alpha)
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
        
        # If it's the training set and a transform is specified, apply the EEG data transformation
        if not self.frozen_eeg_prior:
            if self.eeg_transform is not None and (self.train or self.eeg_test_aug):
                eeg_data = self.eeg_transform(eeg_data)
        
        if self.image_aug:
            if self.train or self.image_test_aug:
                aug_idx = random.randint(0, len(self.aug_image_features) - 1)
                rep_idx = random.randint(0, self.aug_image_features[0].shape[0] - 1)
                image_feature = self.aug_image_features[aug_idx][rep_idx][object_idx][image_idx]
            else:
                image_feature = self.image_features[object_idx][image_idx]
        else:
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

    # Euclidean Alignment: whitening R^-1/2 must make the per-subject channel covariance ~= I.
    rng = _np.random.default_rng(0)
    A = rng.standard_normal((8, 8)); Xf = _np.einsum('cd,ndt->nct', A, rng.standard_normal((500, 8, 40))).astype(_np.float32)
    R = _np.einsum('nct,ndt->cd', Xf, Xf) / (Xf.shape[0] * Xf.shape[-1])
    W = _inv_sqrt_cov(R); Xa = _np.einsum('cd,ndt->nct', W, Xf)
    Ra = _np.einsum('nct,ndt->cd', Xa, Xa) / (Xa.shape[0] * Xa.shape[-1])
    assert _np.allclose(Ra, _np.eye(8), atol=0.05), _np.abs(Ra - _np.eye(8)).max()
    print('EA self-check OK (aligned covariance ~= I)')
