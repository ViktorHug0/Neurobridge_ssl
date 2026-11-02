"""Per training trial, the images shown just before and after it in the RSVP stream (lever L4).

At 5 Hz the -0.2..0.8 s epoch also holds the responses to the previous image (onset -0.2 s) and the next
three (0.2, 0.4, 0.6 s). train.py --neighbor_loss supervises auxiliary heads with those images' targets.
Trial -> (object, image, rep) slots follow build_alljoined_cache.py exactly: chronological occurrences of
each image, epochs whose window leaves the recording skipped, short slots cycled.

Output: sub-XX/train_neighbors.npy, int32 (1654, 10, 4, len(OFFSETS)) holding the neighbour's training-image
index k = object * 10 + image, or -1 when the neighbour is not a training image (test images are never used:
their targets are evaluation data) or the trial is at a run boundary.

    build_alljoined_neighbors.py [--subjects 1 2 ...]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_alljoined_cache import FREQ, N_TRAIN_IMAGES, load_events, load_recordings  # noqa: E402

OFFSETS = (-1, 1, 2)
OUT = "/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench"


def neighbours_for(subject, recordings, window_start=-0.2, duration=1.0, reps=4):
    n_times, offset = int(round(duration * FREQ)), int(round(window_start * FREQ))
    occ = defaultdict(list)  # (obj, img) -> [neighbour ids per kept occurrence], chronological
    for (session, run), frame in load_events(subject).groupby(["session", "run"]):
        rec = recordings.get((subject, int(session), int(run)))
        if rec is None:
            continue
        length = rec[0].shape[0]
        starts, labels, stims = frame.start.to_numpy(), frame.label.to_numpy(), frame.stim_file.to_numpy()
        ids = [int(s.split(".")[0]) if lab == "stim_train" else -1 for s, lab in zip(stims, labels)]
        for t, (start, lab) in enumerate(zip(starts, labels)):
            s = int(round(start * FREQ)) + offset
            if lab != "stim_train" or s < 0 or s + n_times > length:
                continue  # same skips as the cache builder
            k = ids[t]
            nb = [ids[t + o] if 0 <= t + o < len(ids) else -1 for o in OFFSETS]
            occ[(k // 10, k % 10)].append(nb)
    out = np.full((N_TRAIN_IMAGES // 10, 10, reps, len(OFFSETS)), -1, dtype=np.int32)
    for (o, i), got in occ.items():
        for r in range(reps):
            out[o, i, r] = got[r % len(got)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    args = p.parse_args()
    recordings = load_recordings()
    for s in args.subjects:
        nb = neighbours_for(s, recordings)
        np.save(os.path.join(OUT, f"sub-{s:02d}", "train_neighbors.npy"), nb)
        print(f"sub-{s:02d}: neighbour coverage {[(o, round(float((nb[..., j] >= 0).mean()), 3)) for j, o in enumerate(OFFSETS)]}",
              flush=True)


if __name__ == "__main__":
    main()
