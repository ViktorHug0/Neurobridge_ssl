#!/usr/bin/env python
"""Check a preprocessed THINGS-EEG1 subject against THINGS-EEG2's format and scale.

The scale comparison is the one that matters for cohort merging: if EEG1 and EEG2 sit at
different amplitudes after MVNN, the encoder can separate the datasets trivially and the
extra subjects stop being exchangeable with the originals.
"""
import sys

import numpy as np

sys.path.insert(0, "/nasbrain/p20fores/Neurobridge_SSL")
from module.dataset import _load_eeg_container  # noqa: E402

E1 = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_training.npy"
E2 = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/sub-01/preprocessed_eeg_training.npy"

a, b = _load_eeg_container(E1), _load_eeg_container(E2)
da, db = a["preprocessed_eeg_data"], b["preprocessed_eeg_data"]
print("EEG1 keys:", list(a.keys()))
print("EEG1", da.shape, da.dtype, "| EEG2", db.shape, db.dtype)
print("ch_names identical:", list(a["ch_names"]) == list(b["ch_names"]))
print("times len/first/last: EEG1", len(a["times"]), a["times"][0], a["times"][-1],
      "| EEG2", len(b["times"]), b["times"][0], b["times"][-1])
print("missing_mask sum:", int(a["missing_mask"].sum()))

print("\n-- amplitude after MVNN (both should be ~unit scale) --")
for nm, d in (("EEG1", da), ("EEG2", db)):
    s = d[:2000]
    print(f"  {nm}: std={s.std():.3f} mean={s.mean():+.4f} "
          f"p1={np.percentile(s, 1):+.2f} p99={np.percentile(s, 99):+.2f}")

print("\n-- per-channel std ratio EEG1/EEG2 --")
r = da[:2000].std(axis=(0, 1, 3)) / db[:2000].std(axis=(0, 1, 3))
print("  first 8:", np.round(r[:8], 2), "| median:", round(float(np.median(r)), 3),
      "| min/max:", round(float(r.min()), 2), "/", round(float(r.max()), 2))

# Epoch alignment check. A sample-offset bug in event->epoch mapping survives every format
# check above, but shows up immediately as a missing/shifted early visual response. Restrict
# to 0-300 ms so RSVP entrainment at later lags cannot dominate the argmax.
print("\n-- epoch alignment: occipital (O1/Oz/O2) grand mean, 0-300 ms --")
oi = [list(a["ch_names"]).index(c) for c in ("O1", "Oz", "O2")]
for nm, d in (("EEG1", da), ("EEG2", db)):
    ev = d[:2000][:, :, oi].mean(axis=(0, 1, 2))
    early = ev[:75]  # 300 ms at 250 Hz
    peak = int(np.argmax(np.abs(early)) / 250 * 1000)
    course = "  ".join(f"{int(t/250*1000):>3}ms:{early[t]:+.2f}" for t in range(0, 75, 10))
    print(f"  {nm}: |peak| at {peak} ms")
    print(f"      {course}")
