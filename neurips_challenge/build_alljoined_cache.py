"""Epoch neuralbench's 120 Hz Xu2025Alljoined (Alljoined-1.6M) cache into this repo's .npy layout.

Sibling of build_120hz_cache.py (THINGS-EEG2); same split of responsibilities:
  - inherited from neuralbench's EegExtractor cache: 120 Hz, 0.1-75 Hz band-pass, 50/60 Hz notch,
    RobustScaler per recording (here one recording = one ~5 min RSVP run, 1520 per the study);
  - applied here, as neuralbench does downstream of the cache: the -0.2..0.8 s window, baseline over its
    first 0.2 s, clamp at 20.

Alljoined stimulus <k>.jpg is byte-identical to the k-th THINGS-EEG2 image (sorted training concepts/files,
then sorted test), so k -> (k // 10, k % 10) for training and (k - 16540, 0) for test lines EEG up with the
existing THINGS-EEG2 image features (DINOv2 track1 targets, InternViT, ...).

Output per subject: train.npy (1654, 10, 4, 32, 120) and test.npy (200, 1, 80, 32, 120), plus counts.npy
files holding how many real trials fed each (image) slot, and one info.json with ch_names.
Training images were shown 4-5 times (first 4 kept, chronologically); a slot with fewer real trials (markers
dropped by neuralfetch as collided) is filled by cycling that image's real trials.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

CACHE = "/nasbrain/p20fores/alljoined/nb_cache"
EEG_DIR = os.path.join(
    CACHE,
    "neuralset.extractors.neuro.EegExtractor._get_data,1",
    "frequency=120,filter=(0.10,75),name=EegExtractor,scaler=RobustScaler,notch_filter=(50,60)-8c9c9e77",
)
PARQUET_GLOB = os.path.join(CACHE, "name=Xu2025Alljoined-*/name=TimelineLoader-*/cache/data/*.parquet")
N_TRAIN_IMAGES = 16540
SPLITS = {"stim_train": ("train", 1654, 10, 4), "stim_test": ("test", 200, 1, 80)}
FREQ = 120.0


def load_recordings():
    """(subject, session, run) -> (memmap (time, channel), ch_names)."""
    out = {}
    for info in glob.glob(os.path.join(EEG_DIR, "*-info.jsonl")):
        for line in open(info):
            r = json.loads(line)
            key = json.loads(r["#key"].rsplit("_", 2)[0])["timeline"]
            assert float(r["frequency"]) == FREQ and float(r["start"]) == 0.0, r["#key"]
            d, h = r["data"], r["header"]["content"]
            out[(int(key["subject"]), int(key["session"]), int(key["run"]))] = (
                np.memmap(os.path.join(EEG_DIR, d["filename"]), dtype=d["dtype"], mode="r",
                          offset=d["offset"], shape=tuple(d["shape"])),
                h["ch_names"].split(","),
            )
    return out


def load_events(subject):
    frames = []
    for path in glob.glob(PARQUET_GLOB):
        if f"subject={subject}-" not in os.path.basename(path):
            continue
        f = pd.read_parquet(path, columns=["type", "start", "label", "stim_file", "session", "run"])
        frames.append(f[(f.type == "Image") & f.label.isin(list(SPLITS))])
    return pd.concat(frames).sort_values(["session", "run", "start"])


def slot(label, stim):
    k = int(stim.split(".")[0])
    return (k // 10, k % 10) if label == "stim_train" else (k - N_TRAIN_IMAGES, 0)


def build_subject(subject, recordings, ch_names, args):
    n_times = int(round(args.duration * FREQ))
    base = int(round(0.2 * FREQ))  # neuro.baseline [0, 0.2] relative to the window start
    offset = int(round(args.window_start * FREQ))
    events = load_events(subject)
    trials = defaultdict(list)  # (label, obj, img) -> [epochs], chronological
    missing_rec = skipped = 0
    for (session, run), frame in events.groupby(["session", "run"]):
        rec = recordings.get((subject, int(session), int(run)))
        if rec is None:
            missing_rec += 1
            continue
        array, names = rec
        perm = [names.index(c) for c in ch_names]
        for start, label, stim in zip(frame.start.to_numpy(), frame.label.to_numpy(), frame.stim_file.to_numpy()):
            s = int(round(start * FREQ)) + offset
            if s < 0 or s + n_times > array.shape[0]:
                skipped += 1
                continue
            seg = np.array(array[s:s + n_times][:, perm], dtype=np.float32).T  # (channel, time)
            seg -= seg[:, :base].mean(axis=1, keepdims=True)
            np.clip(seg, -args.clamp, args.clamp, out=seg)
            trials[(label, *slot(label, stim))].append(seg)
    out = {}
    for label, (name, n_obj, n_img, reps) in SPLITS.items():
        arr = np.zeros((n_obj, n_img, reps, len(ch_names), n_times), dtype=np.float32)
        counts = np.zeros((n_obj, n_img), dtype=np.int16)
        for o in range(n_obj):
            for i in range(n_img):
                got = trials.get((label, o, i), [])
                counts[o, i] = min(len(got), reps)
                for r in range(reps if got else 0):
                    arr[o, i, r] = got[r % len(got)]  # ponytail: cycle real trials into short slots
        out[name] = (arr, counts)
    return out, dict(missing_recordings=missing_rec, skipped=skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--window_start", type=float, default=-0.2)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--clamp", type=float, default=20.0)
    args = parser.parse_args()

    recordings = load_recordings()
    ch_names = next(iter(recordings.values()))[1]
    assert all(set(n) == set(ch_names) for _, n in recordings.values())
    print(f"recordings: {len(recordings)}  channels: {len(ch_names)}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    for subject in args.subjects:
        built, stats = build_subject(subject, recordings, ch_names, args)
        sub_dir = os.path.join(args.out_dir, f"sub-{subject:02d}")
        os.makedirs(sub_dir, exist_ok=True)
        for name, (arr, counts) in built.items():
            np.save(os.path.join(sub_dir, f"{name}.npy"), arr)
            np.save(os.path.join(sub_dir, f"{name}_counts.npy"), counts)
            reps = arr.shape[2]
            print(f"sub-{subject:02d} {name} {arr.shape}: real trials {int(counts.sum())}/{counts.size * reps}, "
                  f"slots empty {int((counts == 0).sum())}, short {int(((counts > 0) & (counts < reps)).sum())} {stats}",
                  flush=True)
    json.dump({"ch_names": ch_names, "sfreq": FREQ,
               "times": (np.arange(int(round(args.duration * FREQ))) / FREQ + args.window_start).tolist(),
               "window_start": args.window_start, "duration": args.duration, "baseline": [0.0, 0.2],
               "clamp": args.clamp, "source": "neuralbench Xu2025Alljoined EegExtractor 120 Hz cache, "
                                              "epoched by neurips_challenge/build_alljoined_cache.py"},
              open(os.path.join(args.out_dir, "info.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
