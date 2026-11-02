"""Add the 10 THINGS-EEG2 subjects to the Alljoined 120 Hz dir as sub-21..sub-30.

Both caches are the same neuralbench grader preprocessing (120 Hz, -0.2-0.8 s, baseline [0, 0.2], clamp 20)
and the same image order, so the only work is keeping the 32 Emotiv electrodes, in Alljoined's order.
The mapping is recorded in the Alljoined info.json under "things_eeg2_subjects".

    export_things_to_alljoined.py [--check]
"""
import json
import os
import sys

import numpy as np

THINGS = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench"
AJ = "/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench"
OFFSET = 20

t_info = json.load(open(f"{THINGS}/info.json"))
a_info = json.load(open(f"{AJ}/info.json"))
for key in ("sfreq", "window_start", "duration", "baseline", "clamp"):
    assert t_info[key] == a_info[key], (key, t_info[key], a_info[key])
pick = [t_info["ch_names"].index(c) for c in a_info["ch_names"]]

for s in range(1, 11):
    dst = f"{AJ}/sub-{s + OFFSET:02d}"
    os.makedirs(dst, exist_ok=True)
    for split in ("train", "test"):
        src = np.load(f"{THINGS}/sub-{s:02d}/{split}.npy", mmap_mode="r")
        out = f"{dst}/{split}.npy"
        if "--check" in sys.argv:
            got = np.load(out, mmap_mode="r")
            assert np.array_equal(got[:3], src[:3][..., pick, :]), out
            continue
        # chunked through a memmap: a whole train array (1 GB at 32 ch) does not fit the 1.9 GB interactive slot
        dst_arr = np.lib.format.open_memmap(out + ".tmp.npy", mode="w+", dtype=src.dtype,
                                            shape=src.shape[:-2] + (len(pick), src.shape[-1]))
        for i in range(0, len(src), 100):
            dst_arr[i:i + 100] = src[i:i + 100][..., pick, :]
        dst_arr.flush()
        del dst_arr
        os.replace(out + ".tmp.npy", out)
    print(f"THINGS sub-{s:02d} -> {dst}", flush=True)

a_info["things_eeg2_subjects"] = {str(s + OFFSET): s for s in range(1, 11)}
with open(f"{AJ}/info.json.tmp", "w") as handle:
    json.dump(a_info, handle, indent=1)
os.replace(f"{AJ}/info.json.tmp", f"{AJ}/info.json")
