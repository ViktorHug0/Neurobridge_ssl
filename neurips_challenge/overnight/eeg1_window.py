"""THINGS-EEG1 variant sub-55..66 = sub-31..42 with each epoch faded to 0 between 0.35 and 0.45 s after onset (cosine).
At 10 Hz / 50 ms stimuli the image-specific signal sits at ~0.05-0.35 s (Robinson et al. 2019; Grootswagers et al.
2019/2022); later samples are the next images. Epoch -0.2..0.8 s at 120 Hz: onset = sample 24."""
import os
import numpy as np
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
t = (np.arange(120) - 24) / 120.0
w = np.clip((0.45 - t) / 0.10, 0, 1)
w = (0.5 - 0.5 * np.cos(np.pi * w)).astype(np.float32)          # 1 up to 0.35 s, cosine to 0 at 0.45 s
assert w[24] == 1 and w[-1] == 0
for s in range(31, 43):
    out = f'{D}/sub-{s + 24:02d}/train.npy'
    if os.path.exists(out):
        continue
    X = np.load(f'{D}/sub-{s:02d}/train.npy') * w
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + '.tmp.npy', X); os.replace(out + '.tmp.npy', out)
    print(s + 24, flush=True)
