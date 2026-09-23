"""Epoch the start-kit's 120 Hz continuous EEG cache into this repo's .npy layout.

NeuralBench's `EegExtractor` cache stores 80 continuous recordings (10 subjects x 4 sessions x
{train,test}), keyed only by (subject, session, split) -- there is no stimulus or repetition in
its key, so unlike the DINOv2 image cache it cannot simply be re-keyed. The per-trial onsets live
separately, in the `TimelineLoader` parquet files. This script joins the two.

What is inherited from NeuralBench, unmodified, because it is already baked into the cached array:
  - resampling to 120 Hz
  - 0.1-75 Hz band-pass, 50/60 Hz notch
  - RobustScaler fit per continuous recording, per channel

What this script must apply itself, because NeuralBench applies it downstream of that cache
(neuro.py:448-465 -- verified: the stored array is unclamped, min -72.06):
  - baseline correction, and
  - clamping.

Everything else is ours: the channel order, the array layout, and the stimulus ordering, which is
made byte-identical to `extract_feature.py`'s enumeration so EEG index i lines up with image
feature index i.

Output per subject: train.npy (1654, 10, 4, 63, T), test.npy (200, 1, 80, 63, T), plus an
info.json holding ch_names -- which module/dataset.py:284 needs for --selected_channels to work
at all, and which the existing 250 Hz directory is missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

CH_ROOT = "/nasbrain/p20fores/Neurips_challenge"
CACHE_DIR = os.path.join(
    CH_ROOT,
    "cache/neuralset.extractors.neuro.EegExtractor._get_data,1",
    "frequency=120,filter=(0.10,75),name=EegExtractor,scaler=RobustScaler,notch_filter=(50,60)-8c9c9e77",
)
PARQUET_GLOB = os.path.join(
    CH_ROOT, "cache/name=Gifford2022Local-*/name=TimelineLoader-*/cache/data/*.parquet"
)
IMAGE_SET_DIR = "/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set"

# preprocess_eeg.py:238-244. The cache stores acquisition order; we permute to this.
NB_CHANNEL_ORDER = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]

# reps_per_session matters: NICE-EEG's preprocessing keeps max_rep presentations *per session*
# (preprocessing_utils.py), and ~170 training / 56 test images were actually shown one extra time
# in a session. Capping globally instead of per session would give those images 3 reps from one
# session and 1 from the other, which is a different trial composition, not just a different
# subset. The surplus presentations are genuine image trials -- catch trials (stim 99999) are
# already absent from the parquets, dropped by mne.annotations_from_events.
SPLITS = {
    "train": {"dir": "training_images", "images_per_object": 10, "reps": 4, "reps_per_session": 2},
    "test": {"dir": "test_images", "images_per_object": 1, "reps": 80, "reps_per_session": 20},
}


def load_recordings(cache_dir: str):
    """(subject, session, split) -> (memmap, ch_names in cache order)."""
    recordings = {}
    for info_path in sorted(glob.glob(os.path.join(cache_dir, "*-info.jsonl"))):
        for line in open(info_path):
            record = json.loads(line)
            key = json.loads(record["#key"].rsplit("_", 2)[0])["timeline"]
            blob = os.path.join(cache_dir, record["data"]["filename"])
            array = np.memmap(
                blob,
                dtype=record["data"]["dtype"],
                mode="r",
                offset=record["data"]["offset"],
                shape=tuple(record["data"]["shape"]),
            )
            header = record["header"]["content"]
            with open(os.path.join(cache_dir, header["filename"]), "rb") as handle:
                handle.seek(header["offset"])
                meta = json.loads(handle.read(header["length"]))
            recordings[(int(key["subject"]), int(key["session"]), key["split"])] = (
                array,
                meta["ch_names"].split(","),
                float(record["frequency"]),
            )
    return recordings


def stimulus_index(image_set_dir: str, split: str):
    """(concept_dir, filename) -> (object_idx, image_idx), in extract_feature.py's order."""
    root = os.path.join(image_set_dir, SPLITS[split]["dir"])
    index = {}
    for object_idx, concept in enumerate(sorted(os.listdir(root))):
        for image_idx, filename in enumerate(sorted(os.listdir(os.path.join(root, concept)))):
            index[(concept, filename)] = (object_idx, image_idx)
    return index


def load_events(parquet_glob: str):
    """(subject, session, split) -> DataFrame of Image rows."""
    events = {}
    for path in sorted(glob.glob(parquet_glob)):
        fields = dict(re.findall(r"(\w+)=([\w]+)", os.path.basename(path)))
        frame = pd.read_parquet(path, columns=["type", "onset", "filepath"])
        frame = frame[frame.type == "Image"]
        events[(int(fields["subject"]), int(fields["session"]), fields["split"])] = frame
    return events


def epoch_one(array, start_sample, n_samples, baseline_slice, clamp):
    """Slice one epoch, baseline-correct, clamp. Mirrors neuro.py:448-465."""
    # np.array, not np.asarray: the source is a read-only memmap, and with no baseline
    # correction there is no intermediate copy before the in-place clamp below.
    segment = np.array(array[start_sample : start_sample + n_samples], dtype=np.float32)
    if segment.shape[0] != n_samples:
        return None  # trial runs past the end of the recording
    if baseline_slice is not None:
        lo, hi = baseline_slice
        segment = segment - segment[lo:hi].mean(axis=0, keepdims=True)
    if clamp is not None:
        np.clip(segment, -clamp, clamp, out=segment)
    return segment


def build_subject(subject, split, recordings, events, index, args, n_objects):
    spec = SPLITS[split]
    freq = args.frequency
    n_times = int(round(args.duration * freq))

    # Read enough context to cover both the analysis window and the baseline, exactly as
    # neuro.py:427-434 extends the window when the baseline falls outside it.
    read_start = args.window_start
    read_stop = args.window_start + args.duration
    if args.baseline is not None:
        read_start = min(read_start, args.window_start + args.baseline[0])
        read_stop = max(read_stop, args.window_start + args.baseline[1])
    read_offset = int(round(read_start * freq))
    n_read = int(round((read_stop - read_start) * freq))
    window_lo = int(round((args.window_start - read_start) * freq))
    baseline_slice = None
    if args.baseline is not None:
        base_lo = int(round((args.window_start + args.baseline[0] - read_start) * freq))
        baseline_slice = (base_lo, base_lo + int(round((args.baseline[1] - args.baseline[0]) * freq)))

    out = np.zeros(
        (n_objects, spec["images_per_object"], spec["reps"], len(NB_CHANNEL_ORDER), n_times),
        dtype=np.float32,
    )
    filled = defaultdict(int)
    filled_session = defaultdict(int)
    overflow = skipped = 0

    for session in sorted({s for (sub, s, sp) in recordings if sub == subject and sp == split}):
        array, cache_channels, cache_freq = recordings[(subject, session, split)]
        assert abs(cache_freq - freq) < 1e-6, f"expected {freq} Hz, cache says {cache_freq}"
        permutation = [cache_channels.index(name) for name in NB_CHANNEL_ORDER]
        frame = events[(subject, session, split)]
        for onset, filepath in zip(frame.onset.to_numpy(), frame.filepath.to_numpy()):
            parts = filepath.split("/")
            key = (parts[-2], parts[-1])
            object_idx, image_idx = index[key]
            rep = filled[(object_idx, image_idx)]
            if (
                rep >= spec["reps"]
                or filled_session[(object_idx, image_idx, session)] >= spec["reps_per_session"]
            ):
                overflow += 1
                continue
            segment = epoch_one(
                array,
                int(round(onset * freq)) + read_offset,
                n_read,
                baseline_slice,
                args.clamp,
            )
            if segment is None:
                skipped += 1
                continue
            window = segment[window_lo : window_lo + n_times]
            out[object_idx, image_idx, rep] = window[:, permutation].T
            filled[(object_idx, image_idx)] += 1
            filled_session[(object_idx, image_idx, session)] += 1

    expected = n_objects * spec["images_per_object"] * spec["reps"]
    total = sum(filled.values())
    return out, {"filled": total, "expected": expected, "overflow": overflow, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--parquet_glob", default=PARQUET_GLOB)
    parser.add_argument("--image_set_dir", default=IMAGE_SET_DIR)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--window_start",
        type=float,
        default=-0.2,
        help="seconds relative to stimulus onset; -0.2 reproduces NeuralBench, 0.0 this repo",
    )
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--baseline",
        type=float,
        nargs=2,
        default=[0.0, 0.2],
        help="relative to the WINDOW start, matching neuro.py; pass none via --no_baseline",
    )
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--clamp", type=float, default=20.0)
    parser.add_argument(
        "--frequency",
        type=float,
        default=120.0,
        help="sampling rate of --cache_dir; checked against the cache's own metadata",
    )
    args = parser.parse_args()
    if args.no_baseline:
        args.baseline = None

    os.makedirs(args.out_dir, exist_ok=True)
    recordings = load_recordings(args.cache_dir)
    events = load_events(args.parquet_glob)
    print(f"recordings: {len(recordings)}  event tables: {len(events)}")

    # The window arithmetic below is in samples, so a wrong --frequency would silently produce
    # epochs of the right shape from the wrong time span.
    cached_freqs = {freq for _, _, freq in recordings.values()}
    if cached_freqs != {args.frequency}:
        raise SystemExit(
            f"--frequency {args.frequency} does not match the cache, which holds {sorted(cached_freqs)}"
        )

    indices = {split: stimulus_index(args.image_set_dir, split) for split in SPLITS}
    n_objects = {
        split: 1 + max(o for o, _ in idx.values()) for split, idx in indices.items()
    }
    print({split: (n_objects[split], len(indices[split])) for split in SPLITS})

    for subject in args.subjects:
        subject_dir = os.path.join(args.out_dir, f"sub-{subject:02d}")
        os.makedirs(subject_dir, exist_ok=True)
        for split in ("train", "test"):
            array, stats = build_subject(
                subject, split, recordings, events, indices[split], args, n_objects[split]
            )
            np.save(os.path.join(subject_dir, f"{split}.npy"), array)
            print(
                f"sub-{subject:02d} {split}: {array.shape} "
                f"filled {stats['filled']}/{stats['expected']} "
                f"overflow {stats['overflow']} skipped {stats['skipped']}",
                flush=True,
            )
            del array

    times = np.arange(n_times) / freq + args.window_start
    with open(os.path.join(args.out_dir, "info.json"), "w") as handle:
        json.dump(
            {
                "ch_names": NB_CHANNEL_ORDER,
                "sfreq": float(freq),
                "times": times.tolist(),
                "window_start": args.window_start,
                "duration": args.duration,
                "baseline": args.baseline,
                "clamp": args.clamp,
                "source": f"NeuralBench EegExtractor {freq} Hz cache, epoched by build_120hz_cache.py",
            },
            handle,
            indent=2,
        )
    print(f"wrote {os.path.join(args.out_dir, 'info.json')}")


if __name__ == "__main__":
    main()
