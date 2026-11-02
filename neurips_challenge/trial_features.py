"""Label-free per-trial features of the raw test EEG (all 20 Alljoined subjects), in test.npy order (same as
trial_cache.py). 'subject-level' features use the subject's other trials (a stateful predict() could accumulate them).
    trial_features.py -> results/alljoined/track1/routing/trial_features.npz  ({s}_{name}: (n_trials,))"""
import numpy as np
D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
out = {}
for s in range(1, 21):
    x = np.load(f'{D}/sub-{s:02d}/test.npy').reshape(-1, 32, 120).astype(np.float32)
    n = len(x)
    erp = x.mean(0)
    flat = x.reshape(n, -1); fc = flat - flat.mean(0)
    _, _, Vt = np.linalg.svd(fc[np.random.default_rng(s).choice(n, 4000, replace=False)], full_matrices=False)
    z = fc @ Vt[:20].T; z /= z.std(0)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2                      # 61 bins of 1 Hz
    tot = spec[..., 1:].sum((1, 2))
    e = erp.ravel() - erp.mean()
    f = dict(
        logstd=np.log(x.std((1, 2))),
        logmax=np.log(np.abs(x).max((1, 2))),
        clamp=(np.abs(x) >= 19.99).mean((1, 2)),
        erpcorr=(fc @ e) / (np.linalg.norm(fc, axis=1) * np.linalg.norm(e) + 1e-9),
        pcadist=np.log((z ** 2).sum(1)),
        lowfrac=spec[..., 1:4].sum((1, 2)) / tot,                     # 1-3 Hz drift
        highfrac=spec[..., 30:].sum((1, 2)) / tot,                    # >30 Hz muscle
        baseratio=np.log(x[..., :24].std((1, 2)) / x[..., 24:].std((1, 2))),
        chanmaxz=np.log(x.std(2).max(1) / np.median(x.std(2), 1)),     # one bad channel
    )
    out.update({f's{s}_{k}': v.astype(np.float32) for k, v in f.items()})
    print(s, n, {k: round(float(np.median(v)), 3) for k, v in f.items()}, flush=True)
np.savez('results/alljoined/track1/routing/trial_features.npz', **out)
