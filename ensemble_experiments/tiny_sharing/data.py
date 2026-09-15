"""Inner-fold data plumbing for the tiny sharing screen (plan sec. 7, Phase B).

Batches are built in-process rather than through a DataLoader so that the exact
same pairwise-mixed EEG tensor is reproducible across all ten conditions from
(epoch, step) alone.
"""

import os

import numpy as np
import torch

REPO = "/nasbrain/p20fores/Neurobridge_SSL"
EEG_DIR = os.environ.get(
    "EEG_DATA_DIR", "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"
)
FEATURE_ROOT = os.path.join(REPO, "data/things_eeg/image_feature")
TARGET_A = "InternViT-6B_layer33_mean_8bit"   # TS branch
TARGET_B = "InternViT-6B_layer28_mean_8bit"   # ATM branch

NUM_CONCEPTS = 1654
IMAGES_PER_CONCEPT = 10
HELD_CONCEPTS = 165
CONCEPT_SEED = 20260822
GALLERY_SEED = 3300
GALLERY = 200

# outer slot -> (outer subject excluded entirely, inner validation subject)
SCREEN_SLOTS = {1: (1, 2), 2: (2, 3), 3: (3, 4)}


def held_out_concepts():
    """Same rule train.py uses for --val_concept_ratio 0.1 at seed 20260822."""
    order = np.random.default_rng(CONCEPT_SEED).permutation(np.arange(NUM_CONCEPTS))
    return np.sort(order[:HELD_CONCEPTS])


def _subject_eeg(subject_id, concepts):
    """Averaged train EEG for one subject, restricted to `concepts`.

    Reuses the repo's own processed cache; falls back to building it through
    module.dataset if the cache is cold.
    """
    from module.dataset import _eeg_cache_key, _eeg_cache_path

    key = _eeg_cache_key(
        subject_id=subject_id, train=True, average=True,
        selected_channels=[], time_window=[0, 250],
    )
    path = _eeg_cache_path(EEG_DIR, key)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"cold EEG cache for sub-{subject_id:02d} at {path}; run a normal train.py "
            "fold once to populate it"
        )
    mm = np.load(path, mmap_mode="r")           # (1654, 10, 63, 250)
    return np.ascontiguousarray(mm[concepts])   # fp16


def _image_features(concepts=None):
    out = []
    for target in (TARGET_A, TARGET_B):
        arr = np.load(os.path.join(FEATURE_ROOT, target, "image_train.npy"), mmap_mode="r")
        out.append(np.ascontiguousarray(arr if concepts is None else arr[concepts]))
    return out


def gallery_panels(n_items=HELD_CONCEPTS * IMAGES_PER_CONCEPT):
    """Fixed 200-candidate panels covering every validation query exactly once.

    Full panels are 200 queries against themselves. The trailing short panel is
    padded to 200 candidates with the first 150 images of the same permutation;
    only its own 50 rows are scored as queries.
    """
    order = np.random.default_rng(GALLERY_SEED).permutation(n_items)
    panels = []
    for start in range(0, n_items, GALLERY):
        block = order[start:start + GALLERY]
        if len(block) == GALLERY:
            panels.append((block, block))
            continue
        pad = order[:GALLERY - len(block)]
        panels.append((block, np.concatenate([block, pad])))
    return panels


class FoldData:
    """One screening slot: 8 training subjects, 1 inner validation subject."""

    def __init__(self, slot, device):
        self.slot = slot
        self.outer, self.inner = SCREEN_SLOTS[slot]
        self.device = device

        held = held_out_concepts()
        keep = np.setdiff1d(np.arange(NUM_CONCEPTS), held)
        self.held, self.keep = held, keep
        self.train_subjects = [
            s for s in range(1, 11) if s not in (self.outer, self.inner)
        ]

        # ---- training tensors: (S, C_keep, 10, 63, 250) fp16 in host RAM ----
        self.train_eeg = np.stack(
            [_subject_eeg(s, keep) for s in self.train_subjects]
        )
        img_a, img_b = _image_features(keep)
        self.train_img_a = torch.from_numpy(img_a).to(device)   # fp16, (C,10,3200)
        self.train_img_b = torch.from_numpy(img_b).to(device)

        self.keys = np.array(
            [(c, i) for c in range(len(keep)) for i in range(IMAGES_PER_CONCEPT)]
        )
        self.samples_per_image = len(self.train_subjects)

        # ---- validation tensors: inner subject on the 165 held concepts ----
        val_eeg = _subject_eeg(self.inner, held).reshape(-1, 63, 250)
        self.val_eeg = torch.from_numpy(val_eeg).float().to(device)
        v_a, v_b = _image_features(held)
        self.val_img_a = torch.from_numpy(v_a.reshape(-1, v_a.shape[-1])).to(device)
        self.val_img_b = torch.from_numpy(v_b.reshape(-1, v_b.shape[-1])).to(device)
        self.panels = gallery_panels(self.val_eeg.shape[0])
        # loss chunks: same permutation, no padding, every row exactly once
        order = np.random.default_rng(GALLERY_SEED).permutation(self.val_eeg.shape[0])
        self.loss_chunks = [order[i:i + GALLERY] for i in range(0, len(order), GALLERY)]

    # ---------------- batching ----------------

    def steps_per_epoch(self, images_per_batch):
        return len(self.keys) // images_per_batch

    def epoch_key_order(self, epoch, seed=GALLERY_SEED):
        return np.random.default_rng(seed + epoch).permutation(len(self.keys))

    def batch(self, key_ids, epoch, step, alpha=0.5):
        """One grouped, pairwise-mixed batch. Deterministic in (epoch, step)."""
        groups = self.keys[key_ids]                     # (G, 2)
        g = len(groups)
        s = self.samples_per_image
        eeg = self.train_eeg[:, groups[:, 0], groups[:, 1]]   # (S, G, 63, 250)
        eeg = torch.from_numpy(np.ascontiguousarray(eeg)).to(
            self.device, non_blocking=True
        ).float().permute(1, 0, 2, 3)                         # (G, S, 63, 250)

        rng = np.random.default_rng((CONCEPT_SEED, epoch, step))
        eeg = self._pairwise_mix(eeg, rng, alpha)
        eeg = eeg.reshape(g * s, 63, 250)

        gc = torch.from_numpy(groups[:, 0]).to(self.device)
        gi = torch.from_numpy(groups[:, 1]).to(self.device)
        obj, img = gc.repeat_interleave(s), gi.repeat_interleave(s)
        mask = (obj[:, None] == obj[None, :]) & (img[:, None] == img[None, :])

        ta = self.train_img_a[gc, gi].float().repeat_interleave(s, 0)
        tb = self.train_img_b[gc, gi].float().repeat_interleave(s, 0)
        return eeg, ta, tb, mask

    def _pairwise_mix(self, eeg, rng, alpha):
        """Same-stimulus cross-subject pairwise Mixup, one lambda per row.

        Vectorized restatement of train.py's ``cross_subject_stimulus_mix``
        pairwise branch for the case every group holds distinct subjects:
        a random within-group permutation rolled by a random non-zero offset
        (so no row mixes with itself) and a Beta(a,a) weight per row.
        """
        g, s = eeg.shape[0], eeg.shape[1]
        order = np.argsort(rng.random((g, s)), axis=1)
        offset = rng.integers(1, s, size=g)
        partner = np.empty((g, s), dtype=np.int64)
        rolled = np.take_along_axis(
            order, (np.arange(s)[None, :] + offset[:, None]) % s, axis=1
        )
        np.put_along_axis(partner, order, rolled, axis=1)

        lam = torch.from_numpy(rng.beta(alpha, alpha, size=(g, s))).to(
            eeg.device, torch.float32
        )[:, :, None, None]
        partner_t = torch.from_numpy(partner).to(eeg.device)
        mixed = eeg.gather(
            1, partner_t[:, :, None, None].expand_as(eeg)
        )
        return lam * eeg + (1.0 - lam) * mixed
