"""Add THINGS-EEG1 (ds003825: 50 subjects, 1 trial per image, 10 Hz RSVP) to the Alljoined 120 Hz dir.

Built from the RAW BrainVision files with the grader's neuralbench chain, so the noise treatment matches the
Alljoined / THINGS-EEG2 caches exactly (neuralset MneRaw._preprocess_raw + track1_tsconv.yaml, then
build_alljoined_cache.py's epoching):
  re-reference to Fz (EEG2's online reference; preprocess_eeg1.py) -> pick the 32 Alljoined electrodes ->
  notch 50/60 Hz + harmonics (phase zero) -> band-pass 0.1-75 Hz -> resample 120 Hz -> RobustScaler per channel
  over the whole recording -> window -0.2..0.8 s at round(onset * 120) -> baseline = first 0.2 s -> clamp 20.
Trials: non-target presentations of EEG2's 16540 TRAINING images only (no test leakage), first occurrence.

The grid layout is (1654, 10, 4, 32, 120) per subject, but EEG1 has ONE trial per image, so four EEG1 subjects
are packed into one pseudo-subject (sub-31, sub-32, ...): repetition slot r = the r-th subject of the group. No
averaging: training draws single repetitions, so every sample stays one single trial. An image a subject never
saw is filled by cycling the group's other trials of that image. Train-only: no test.npy.

    export_things_eeg1_to_alljoined.py [--groups 0 1 2] [--check]
"""
import argparse
import json
import os
import sys

import mne
import numpy as np
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_eeg1 import CHANNELS_ORDER, eeg2_image_index, read_events  # noqa: E402

RAW = "/nasbrain/ProCOM-EEG/EEGNet_Things-EGG1/data/ds003825"
IMAGES = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"
AJ = "/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench"
OFFSET, GROUP, FREQ, N_TIMES, BASE, START = 30, 4, 120.0, 120, 24, -24   # -0.2..0.8 s, baseline first 0.2 s
CH = json.load(open(f"{AJ}/info.json"))["ch_names"]
SUBJECTS = sorted(int(d[4:]) for d in os.listdir(RAW) if d.startswith("sub-"))
GROUPS = [SUBJECTS[i:i + GROUP] for i in range(0, len(SUBJECTS) - GROUP + 1, GROUP)]


def load(s, image_index):
    """(16540, 32, 120) float32 single trials of subject s, and the (16540,) present mask."""
    d = f"{RAW}/sub-{s:02d}/eeg/sub-{s:02d}_task-rsvp"
    seen, trials = set(), []
    for sample, idx in read_events(f"{d}_events.tsv", image_index):
        if idx not in seen:
            seen.add(idx); trials.append((sample, idx))
    raw = mne.io.read_raw_brainvision(f"{d}_eeg.vhdr", preload=True, verbose="ERROR")
    absent = [c for c in CHANNELS_ORDER if c not in raw.ch_names]   # the online reference (Cz, or FCz for 49/50)
    if len(absent) == 1 and "Fz" in raw.ch_names:
        mne.add_reference_channels(raw, absent[0], copy=False)
        raw.set_eeg_reference(["Fz"], verbose="ERROR")
    raw.pick(CH)
    sfreq = raw.info["sfreq"]
    notch = [f for base in (50.0, 60.0) for f in np.arange(base, min(sfreq / 2, 301), base)]
    raw.notch_filter(notch, phase="zero", filter_length="auto", verbose="ERROR")
    raw.filter(0.1, 75.0, verbose="ERROR")
    raw.resample(FREQ, verbose="ERROR")
    x = RobustScaler().fit_transform(raw.get_data().T).T.astype(np.float32)   # (32, T), CH order
    out = np.zeros((16540, len(CH), N_TIMES), np.float32)
    present = np.zeros(16540, bool)
    for sample, idx in trials:
        t = int(round(sample / sfreq * FREQ)) + START
        if t < 0 or t + N_TIMES > x.shape[1]:
            continue
        seg = x[:, t:t + N_TIMES].copy()
        seg -= seg[:, :BASE].mean(1, keepdims=True)
        out[idx] = np.clip(seg, -20, 20)
        present[idx] = True
    return out, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, nargs="*", default=list(range(len(GROUPS))))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    assert "Fz" not in CH, "Fz is the re-reference: it reads 0 and cannot be one of the served electrodes"
    image_index, _ = eeg2_image_index(IMAGES)
    for g in a.groups:
        sid, members = OFFSET + 1 + g, GROUPS[g]
        out = f"{AJ}/sub-{sid:02d}/train.npy"
        if a.check:
            arr = np.load(out, mmap_mode="r")
            assert arr.shape == (1654, 10, GROUP, len(CH), N_TIMES) and np.isfinite(arr[:50]).all(), out
            print(out, f"mean|x| {float(np.abs(arr[:100]).mean()):.3f}")
            continue
        if os.path.exists(out):
            print(f"sub-{sid:02d}: exists, skipping", flush=True); continue
        xs, ps = zip(*(load(s, image_index) for s in members))
        X, P = np.stack(xs, 1), np.stack(ps, 1)                                # (16540, 4, 32, 120), (16540, 4)
        for i in np.where(~P.all(1))[0]:
            have = np.where(P[i])[0]
            if len(have) == 0:
                continue                                                       # never shown in this group: left zero
            for k, r in enumerate(np.where(~P[i])[0]):
                X[i, r] = X[i, have[k % len(have)]]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out + ".tmp.npy", X.reshape(1654, 10, GROUP, len(CH), N_TIMES))
        os.replace(out + ".tmp.npy", out)
        json.dump({"eeg1_subjects": members, "present_per_subject": P.sum(0).tolist(), "empty_images": int((~P.any(1)).sum())},
                  open(f"{AJ}/sub-{sid:02d}/eeg1_meta.json", "w"))
        print(f"sub-{sid:02d} <- EEG1 {members}: present {P.sum(0).tolist()}, empty images {int((~P.any(1)).sum())}", flush=True)


if __name__ == "__main__":
    main()
