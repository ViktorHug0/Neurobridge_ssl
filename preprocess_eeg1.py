"""Preprocess THINGS-EEG1 (OpenNeuro ds003825, 50 subjects) into the THINGS-EEG2 layout.

Produces, per subject, a NICE-EEG-style dict identical in structure to
Things-EEG2/Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_training.npy:

    {"preprocessed_eeg_data": (16540, 1, 63, 250) float32,
     "ch_names": <63 names, EEG2 order>,
     "times":    <300 values, -0.2 .. 0.996>}

so the existing loader (`_resolve_eeg_file` / `_load_eeg_container`) reads it unchanged and
EEG1 subjects can be appended to `--train_subject_ids`.

Why the image filter matters: EEG1's 1,854 concepts are a strict SUPERSET of EEG2's
1,654 train + 200 test concepts, and 177 of EEG2's 200 test IMAGES are shown in EEG1.
Training on unfiltered EEG1 would leak the test set and void the zero-shot claim. Keeping
only trials whose image is one of EEG2's 16,540 training images excludes every test concept
and test image in one filter, and aligns the image axis with the existing feature bank.

Image axis order == extract_feature.py's: sorted(concept dirs) x sorted(filenames), 10 per
concept, so index = concept_idx * 10 + image_idx.

EEG1 shows each image ONCE per subject (rep axis = 1), unlike EEG2's 4. The ~24 images per
subject that a given subject never saw are filled with that subject's mean over the same
concept, and flagged in the saved "missing_mask".

The two datasets also use different online references (EEG1: Cz, EEG2: Fz), which is undone
exactly in process_subject() -- see the comment there.
"""
import argparse
import csv
import os
from collections import defaultdict

import mne
import numpy as np
import scipy
from sklearn.discriminant_analysis import _cov
from tqdm import tqdm

# EEG2 channel order (preprocess_eeg.py)
CHANNELS_ORDER = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                  'O1', 'Oz', 'O2']


def eeg2_image_index(image_set_dir):
    """filename -> flat index, matching extract_feature.py's sorted/sorted walk."""
    train_dir = os.path.join(image_set_dir, "training_images")
    index, concept_of = {}, {}
    for c_i, concept in enumerate(sorted(os.listdir(train_dir))):
        for i, fname in enumerate(sorted(os.listdir(os.path.join(train_dir, concept)))):
            index[fname] = c_i * 10 + i
            concept_of[c_i * 10 + i] = c_i
    return index, concept_of


def validation_images(events_tsv, n_images=150):
    """This subject's repeated-presentation images, capped to a fixed count.

    EEG1's validation block is the ONLY part of the dataset with repetitions (12 vs 1), so it is
    the only basis for an intra-subject test set with usable SNR. Each subject saw a different
    ~160 of the 200 validation images (10 subjects share only 19), so the set cannot be shared;
    capping at the cohort minimum (150) keeps the retrieval chance level identical across
    subjects. Sorted for determinism -- the image-feature rows are built with the same rule.
    """
    reps = {}
    with open(events_tsv) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("istarget", "0") != "0":
                continue
            fname = row["stim"].replace("\\", "/").split("/")[-1]
            reps.setdefault(fname, []).append(int(row["onset"]) - 1)
    return sorted(f for f, s in reps.items() if len(s) >= 12)[:n_images], reps


def read_events(events_tsv, image_index):
    """Non-target trials whose image is an EEG2 training image -> (sample, flat_index)."""
    keep = []
    with open(events_tsv) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("istarget", "0") != "0":
                continue
            # 'stim' is the only filename column present in all 50 subjects' headers
            fname = row["stim"].replace("\\", "/").split("/")[-1]
            idx = image_index.get(fname)
            if idx is not None:
                # events.tsv 'onset' is a 1-based BrainVision marker position; MNE indexes raw
                # samples from 0. Verified against events_from_annotations: offset is exactly -1
                # for all 22,248 markers.
                keep.append((int(row["onset"]) - 1, idx))
    return keep


def mvnn_whiten(data):
    """Multivariate noise normalisation, same math as preprocess_eeg.py's mvnn().

    Covariance per image condition across time points, averaged over conditions, then
    data <- data @ sigma^-1/2. EEG1 has one repetition per image, so the per-condition
    covariance is estimated from that single epoch's 250 time points.
    """
    sigma = np.mean([np.mean([_cov(np.transpose(data[i, r]), shrinkage='auto')
                              for r in range(data.shape[1])], axis=0)
                     for i in tqdm(range(data.shape[0]), desc="  mvnn cov", leave=False)], axis=0)
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
    shape = data.shape
    flat = data.reshape(-1, shape[2], shape[3])
    return np.real((flat.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)).reshape(shape)


def zscore_channelwise(data):
    mean = data.mean(axis=(0, 1, 3), keepdims=True)
    std = np.maximum(data.std(axis=(0, 1, 3), keepdims=True), 1e-8)
    return (data - mean) / std


def process_subject(sub, args, image_index, concept_of):
    eeg_dir = os.path.join(args.raw_data_dir, sub, "eeg")
    vhdr = os.path.join(eeg_dir, f"{sub}_task-rsvp_eeg.vhdr")
    events_tsv = os.path.join(eeg_dir, f"{sub}_task-rsvp_events.tsv")

    if args.export_test:
        return process_subject_test(sub, vhdr, events_tsv, args)

    trials = read_events(events_tsv, image_index)
    # One repetition per image: EEG1 shows each image once, but its repeated test-stimulus
    # block can re-show a handful of EEG2 training images. Keep the first occurrence so the
    # rep axis is uniform.
    seen, unique = set(), []
    for sample, idx in trials:
        if idx not in seen:
            seen.add(idx)
            unique.append((sample, idx))

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")

    # EEG1 records against Cz for 48 subjects, and against FCz for sub-49/50 (different, denser
    # cap), so that one electrode is absent from the file. EEG2 records against Fz (Fz absent,
    # Cz present). Re-referencing is linear and exact, so EEG1 can be put on EEG2's reference
    # without touching EEG2: restore the absent reference as zeros -> channels hold V_i - V_ref
    # -> subtract Fz -> V_i - V_Fz, exactly EEG2's reference. Fz then reads ~0 and is dropped.
    absent = [c for c in CHANNELS_ORDER if c not in raw.ch_names]
    if len(absent) == 1 and "Fz" in raw.ch_names:
        mne.add_reference_channels(raw, absent[0], copy=False)
        raw.set_eeg_reference(["Fz"], verbose="ERROR")
        raw.drop_channels(["Fz"])
    missing = [c for c in CHANNELS_ORDER if c not in raw.ch_names]
    if missing:
        raise ValueError(f"{sub}: channels absent from EEG1 recording: {missing}")

    # Grootswagers et al. 2022 (Sci Data, Methods): "filtered using a Hamming windowed FIR filter
    # with 0.1 Hz highpass and 100 Hz lowpass filters ... downsampled to 250 Hz". Off by default so
    # the first export stays a like-for-like match to EEG2's pipeline.
    if args.highpass or args.lowpass:
        raw.filter(l_freq=args.highpass or None, h_freq=args.lowpass or None,
                   method="fir", fir_window="hamming", verbose="ERROR")

    n_time = int(round(args.after_duration * args.rfreq))
    out = np.zeros((args.n_images, 1, len(CHANNELS_ORDER), n_time), dtype=np.float32)
    filled = np.zeros(args.n_images, dtype=bool)

    # Chunked epoching: 16.5k epochs x 63 ch x 1201 samples in float64 is ~10GB in one go.
    times = None
    for start in tqdm(range(0, len(unique), args.chunk), desc=f"  {sub} epochs", leave=False):
        chunk = unique[start:start + args.chunk]
        events = np.array([[s, 0, i + 1] for s, i in chunk], dtype=int)
        # With a 100 ms SOA the -200..0 ms "baseline" holds the two PRECEDING images' responses,
        # so subtracting it injects neighbour signal into every trial. The dataset authors applied
        # no baseline correction for exactly this kind of stream; --no_baseline reproduces that.
        epochs = mne.Epochs(raw, events, tmin=-args.baseline_duration, tmax=args.after_duration,
                            baseline=None if args.no_baseline else (None, 0),
                            preload=True, verbose="ERROR")
        epochs.resample(args.rfreq, verbose="ERROR")
        epochs.reorder_channels(CHANNELS_ORDER)
        if times is None:
            times = np.round(epochs.times, 3)
        data = epochs.get_data(copy=False)[:, :, int(args.baseline_duration * args.rfreq):]
        # Epochs may drop trials near recording edges; epochs.events keeps the mapping.
        for row, ev in zip(data, epochs.events[:, 2]):
            out[ev - 1, 0] = row[:, :n_time]
            filled[ev - 1] = True
        del epochs, data

    # Fill never-seen images with this subject's mean over the same concept.
    if (~filled).any():
        by_concept = {}
        for i in np.where(filled)[0]:
            by_concept.setdefault(concept_of[i], []).append(i)
        for i in np.where(~filled)[0]:
            peers = by_concept.get(concept_of[i])
            if peers:
                out[i, 0] = out[peers, 0].mean(axis=0)

    if args.mvnn:
        out = mvnn_whiten(out).astype(np.float32)
    elif args.zscore:
        out = zscore_channelwise(out).astype(np.float32)

    return out, ~filled, times


def _prepare_raw(vhdr, args):
    """Read, put on EEG2's Fz reference, and optionally bandpass -- shared by both exports."""
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    absent = [c for c in CHANNELS_ORDER if c not in raw.ch_names]
    if len(absent) == 1 and "Fz" in raw.ch_names:
        mne.add_reference_channels(raw, absent[0], copy=False)
        raw.set_eeg_reference(["Fz"], verbose="ERROR")
        raw.drop_channels(["Fz"])
    missing = [c for c in CHANNELS_ORDER if c not in raw.ch_names]
    if missing:
        raise ValueError(f"channels absent from EEG1 recording: {missing}")
    if args.highpass or args.lowpass:
        raw.filter(l_freq=args.highpass or None, h_freq=args.lowpass or None,
                   method="fir", fir_window="hamming", verbose="ERROR")
    return raw


def process_subject_test(sub, vhdr, events_tsv, args):
    """Intra-subject test set from the validation block: (n_images, 12, 63, 250)."""
    images, reps = validation_images(events_tsv, args.n_test_images)
    if len(images) < args.n_test_images:
        raise ValueError(f"{sub}: only {len(images)} validation images (needs {args.n_test_images}); "
                         "this subject has no usable validation block")

    raw = _prepare_raw(vhdr, args)
    n_time = int(round(args.after_duration * args.rfreq))
    events, order = [], []
    for i, fname in enumerate(images):
        for sample in sorted(reps[fname])[:args.n_test_reps]:
            events.append([sample, 0, i + 1])
            order.append(i)

    epochs = mne.Epochs(raw, np.array(events, dtype=int),
                        tmin=-args.baseline_duration, tmax=args.after_duration,
                        baseline=None if args.no_baseline else (None, 0),
                        preload=True, verbose="ERROR")
    epochs.resample(args.rfreq, verbose="ERROR")
    epochs.reorder_channels(CHANNELS_ORDER)
    times = np.round(epochs.times, 3)
    data = epochs.get_data(copy=False)[:, :, int(args.baseline_duration * args.rfreq):]

    out = np.zeros((len(images), args.n_test_reps, len(CHANNELS_ORDER), n_time), dtype=np.float32)
    seen = defaultdict(int)
    for row, cond in zip(data, epochs.events[:, 2]):
        i = cond - 1
        if seen[i] < args.n_test_reps:
            out[i, seen[i]] = row[:, :n_time]
            seen[i] += 1
    if args.mvnn:
        out = mvnn_whiten(out).astype(np.float32)
    elif args.zscore:
        out = zscore_channelwise(out).astype(np.float32)
    return out, np.array(images), times


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--raw_data_dir', default='/nasbrain/ProCOM-EEG/EEGNet_Things-EGG1/data/ds003825')
    p.add_argument('--image_set_dir', default='/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set')
    p.add_argument('--output_dir', default='/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Preprocessed_data_250Hz')
    p.add_argument('--subjects', default='', type=str, help="e.g. '1 2 3'; empty means all 50")
    p.add_argument('--rfreq', default=250, type=int)
    p.add_argument('--baseline_duration', default=.2, type=float)
    p.add_argument('--after_duration', default=1.0, type=float)
    p.add_argument('--n_images', default=16540, type=int)
    p.add_argument('--chunk', default=4000, type=int)
    p.add_argument('--export_test', action='store_true', help="export the intra-subject test set from the validation block instead of the training set")
    p.add_argument('--n_test_images', default=150, type=int, help="test images per subject (150 = the cohort minimum, so chance level is identical across subjects)")
    p.add_argument('--n_test_reps', default=12, type=int)
    p.add_argument('--no_baseline', action='store_true', help="skip baseline correction (the dataset authors' choice; the -200..0 window contains two prior stimuli at 100ms SOA)")
    p.add_argument('--highpass', default=0.0, type=float, help="FIR highpass in Hz (paper: 0.1); 0 disables")
    p.add_argument('--lowpass', default=0.0, type=float, help="FIR lowpass in Hz (paper: 100); 0 disables")
    p.add_argument('--mvnn', action='store_true')
    p.add_argument('--zscore', action='store_true')
    args = p.parse_args()
    if args.mvnn == args.zscore:
        raise SystemExit("pick exactly one of --mvnn / --zscore (EEG2's published data used mvnn)")

    image_index, concept_of = eeg2_image_index(args.image_set_dir)
    if len(image_index) != args.n_images:
        raise SystemExit(f"expected {args.n_images} EEG2 training images, found {len(image_index)}")

    subs = ([f"sub-{int(s):02d}" for s in args.subjects.split()] if args.subjects
            else sorted(d for d in os.listdir(args.raw_data_dir) if d.startswith("sub-")))

    for sub in subs:
        save_dir = os.path.join(args.output_dir, sub)
        fname = "preprocessed_eeg_test.npy" if args.export_test else "preprocessed_eeg_training.npy"
        target = os.path.join(save_dir, fname)
        if os.path.isfile(target):
            print(f"{sub}: already done, skipping")
            continue
        print(f"\n=== {sub} ===", flush=True)
        data, aux, times = process_subject(sub, args, image_index, concept_of)
        os.makedirs(save_dir, exist_ok=True)
        payload = {"preprocessed_eeg_data": data, "ch_names": CHANNELS_ORDER, "times": times}
        # image_files pins the row order so the per-subject image features can be sliced to match.
        payload["image_files" if args.export_test else "missing_mask"] = aux
        np.save(target, payload)
        note = (f"test images: {len(aux)}" if args.export_test
                else f"images never seen by this subject: {aux.sum()}")
        print(f"{sub}: saved {data.shape} | {note}", flush=True)
