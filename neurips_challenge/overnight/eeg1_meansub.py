"""THINGS-EEG1 variant sub-43..54 = sub-31..42 minus each EEG1 subject's grand-average epoch (its rep slot's mean
over all 16540 images): removes the 10 Hz steady-state response that Alljoined/THINGS-EEG2 (5 Hz) do not have,
keeping the image-specific deviations. Train-only, same layout."""
import json, os
import numpy as np
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
for s in range(31, 43):
    out = f'{D}/sub-{s + 12:02d}/train.npy'
    if os.path.exists(out):
        continue
    X = np.load(f'{D}/sub-{s:02d}/train.npy')                        # (1654, 10, 4, 32, 120)
    X -= X.mean(axis=(0, 1), keepdims=True)                           # per rep slot (= per EEG1 subject)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + '.tmp.npy', np.clip(X, -20, 20)); os.replace(out + '.tmp.npy', out)
    print(s + 12, 'mean|x|', round(float(np.abs(X[:100]).mean()), 3), flush=True)
