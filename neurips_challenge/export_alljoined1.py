"""Alljoined-1 (NEMAR nm000133: 8 subjects, BioSemi 64 ch, 512 Hz, 960 NSD shared images, 300 ms on, SOA ~0.8 s)
through the grader's neuralbench chain, as export_things_eeg1_to_alljoined.py does for THINGS-EEG1:
  average reference over the 64 electrodes (BioSemi records CMS-referenced; the authors' choice) -> pick the 32
  Alljoined-1.6M electrodes -> notch 50/60 Hz + harmonics -> band-pass 0.1-75 Hz -> resample 120 Hz -> RobustScaler
  per channel per recording -> window -0.2..0.8 s at round(onset * 120) -> baseline = first 0.2 s -> clamp 20.
Out: <OUT>/sub-XX.npz with X (n, 32, 120) float16 single trials (all sessions) and nsd (n,) NSD image ids.
    export_alljoined1.py [subjects...]
"""
import glob, json, os, sys
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

RAW = '/nasbrain/p20fores/alljoined/nm000133'
OUT = '/nasbrain/p20fores/alljoined/alljoined1_120Hz'
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']
FREQ, N_TIMES, BASE, START = 120.0, 120, 24, -24


def session(bdf):
    ev = pd.read_csv(bdf.replace('_eeg.bdf', '_events.tsv'), sep='\t')
    raw = mne.io.read_raw_bdf(bdf, preload=True, verbose='ERROR')
    raw.pick([c for c in raw.ch_names if c in mne.channels.make_standard_montage('biosemi64').ch_names])
    assert len(raw.ch_names) == 64, raw.ch_names
    raw.set_eeg_reference('average', verbose='ERROR')
    raw.pick(CH)
    sfreq = raw.info['sfreq']
    raw.notch_filter([f for b in (50.0, 60.0) for f in np.arange(b, min(sfreq / 2, 301), b)], phase='zero', verbose='ERROR')
    raw.filter(0.1, 75.0, verbose='ERROR')
    raw.resample(FREQ, verbose='ERROR')
    x = RobustScaler().fit_transform(raw.get_data().T).T.astype(np.float32)
    X, ids = [], []
    for onset, stim in zip(ev.onset, ev.stim_file):
        t = int(round(onset * FREQ)) + START
        if t < 0 or t + N_TIMES > x.shape[1]:
            continue
        seg = x[:, t:t + N_TIMES].copy()
        seg -= seg[:, :BASE].mean(1, keepdims=True)
        X.append(np.clip(seg, -20, 20)); ids.append(int(os.path.basename(stim).split('.')[0]))
    return X, ids


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in [int(a) for a in sys.argv[1:]] or range(1, 9):
        X, ids = [], []
        for bdf in sorted(glob.glob(f'{RAW}/sub-{s:02d}/ses-*/eeg/*_eeg.bdf')):
            a, b = session(bdf); X += a; ids += b
        np.savez(f'{OUT}/sub-{s:02d}.npz', X=np.stack(X).astype(np.float16), nsd=np.array(ids))
        print(f'sub-{s:02d}: {len(X)} trials, {len(set(ids))} images, mean|x| {np.abs(np.stack(X[:500])).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
