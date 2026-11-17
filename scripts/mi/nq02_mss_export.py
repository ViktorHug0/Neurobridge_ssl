"""Q02 MSS (OpenNeuro ds005589) RSVP runs -> single trials in our layout (32 Alljoined electrodes, 120 Hz, -0.2..0.8 s).
Curry .cdt read directly (float32, sample-major; names and counts from the .cdt.dpa text file: no curryreader needed).
Chain as export_alljoined1.py (October): average reference over the 122 scalp electrodes -> the 32 Alljoined electrodes
(present ones copied, the 9 absent ones by spherical-spline interpolation from all 122, standard_1005 positions) -> notch
50/60 Hz + harmonics -> band-pass 0.1-75 Hz -> resample 120 Hz -> RobustScaler per channel per run -> window -0.2..0.8 s
at round(onset * 120 / 1000) -> baseline = first 0.2 s -> clamp 20.
Trials = events 1..20 (position in the 20-image sequence), matched in order to the run's image.csv; special images (99) dropped.
-> <D>/export/sub-XX_ses-YY.npz: X (n, 32, 120) float16, img (n,) index into targets.npz names, run, seq, pos
    nq02_mss_export.py <sub> <ses> [--keep-raw]"""
import glob, json, os, re, sys
import mne, numpy as np, pandas as pd
from mne.channels.interpolation import _make_interpolation_matrix
from sklearn.preprocessing import RobustScaler

D = '/nasbrain/p20fores/mss_ds005589'
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']
NAMES = {r: i for i, r in enumerate(n for c in sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/meta/*.csv') if os.path.basename(f)[0].islower()) for n in pd.read_csv(f'{D}/meta/{c}.csv', dtype=str).image_file_name)}  # nq01's order
M5 = {k.lower(): v for k, v in mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos'].items()}


def read_cdt(path):
    txt = open(path + '.dpa', errors='ignore').read()
    n, c, sf = (int(re.search(k + r'\s*=\s*(\d+)', txt).group(1)) for k in ('NumSamples', 'NumChannels', 'SampleFreqHz'))
    names = [l.strip() for k, v in re.findall(r'(LABELS\w*) START_LIST[^\n]*\n(.*?)\n\1 END_LIST', txt, re.S) for l in v.split('\n')]
    assert len(names) == c and os.path.getsize(path) == 4 * n * c, (len(names), c, os.path.getsize(path))
    return np.memmap(path, np.float32, 'r').reshape(n, c), names, sf


def spatial(names):
    """(32, n_scalp) matrix: average reference, then copy or spline-interpolate to the Alljoined electrodes; and the scalp columns."""
    scalp = [i for i, n in enumerate(names) if n.lower() in M5]
    low = [names[i].lower() for i in scalp]; pos = np.array([M5[n] for n in low]); pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    W = np.zeros((len(CH), len(scalp)))
    for r, c in enumerate(CH):
        if c.lower() in low:
            W[r, low.index(c.lower())] = 1
        else:
            t = M5[c.lower()][None]; W[r] = _make_interpolation_matrix(pos, t / np.linalg.norm(t))[0]
    k = low.index('oz'); keep = [i for i in range(len(low)) if i != k]  # self-check: Oz rebuilt from the others must look like a local average
    w = _make_interpolation_matrix(pos[keep], pos[k:k + 1])[0]; assert abs(w.sum() - 1) < .05 and w.max() < .6, (w.sum(), w.max())
    return W @ (np.eye(len(scalp)) - 1 / len(scalp)), scalp, [c for c in CH if c.lower() not in low]


def run(cdt):
    x, names, sf = read_cdt(cdt); W, scalp, absent = spatial(names)
    y = W @ np.asarray(x[:, scalp], dtype=np.float64).T * 1e-6  # (32, T) volts
    raw = mne.io.RawArray(y, mne.create_info(CH, sf, 'eeg'), verbose='ERROR')
    raw.notch_filter([f for b in (50.0, 60.0) for f in np.arange(b, min(sf / 2, 301), b)], phase='zero', verbose='ERROR')
    raw.filter(0.1, 75.0, verbose='ERROR'); raw.resample(120.0, verbose='ERROR')
    z = RobustScaler().fit_transform(raw.get_data().T).T.astype(np.float32)
    ev = pd.read_csv(cdt.replace('_eeg.cdt', '_events.tsv'), sep='\t'); ev = ev[ev.eventnumber.between(1, 20)]
    files = [os.path.basename(l.strip().replace('\\', '/')).replace('_resized.jpg', '') for l in open(cdt.replace('_eeg.cdt', '_image.csv')) if '_resized' in l]
    assert len(files) == len(ev) == 2000, (len(files), len(ev))
    X, img, seq, pos = [], [], [], []
    for k, (onset, code, name) in enumerate(zip(ev.onset, ev.eventnumber, files)):
        t = int(round(onset * 120 / sf)) - 24
        if t < 0 or t + 120 > z.shape[1]:
            continue
        s = z[:, t:t + 120].copy(); s -= s[:, :24].mean(1, keepdims=True)
        X.append(np.clip(s, -20, 20)); img.append(NAMES[name]); seq.append(k // 20); pos.append(int(code))
    return np.stack(X).astype(np.float16), img, seq, pos, absent


if __name__ == '__main__':
    sub, ses = sys.argv[1], sys.argv[2]; out = f'{D}/export/sub-{sub}_ses-{ses}.npz'; os.makedirs(f'{D}/export', exist_ok=True)
    parts = []
    for c in sorted(glob.glob(f'{D}/raw/sub-{sub}/ses-{ses}/*task-rsvp*_eeg.cdt')):
        try:
            parts.append(run(c))
        except (AssertionError, AttributeError) as e:  # a truncated or damaged run (the dataset lists some under derivatives/corrupted_raw_data): the session keeps its other runs
            print(f'sub-{sub} ses-{ses}: skipped {os.path.basename(c)}: {type(e).__name__} {e}', flush=True)
    X = np.concatenate([p[0] for p in parts])
    np.savez(out, X=X, img=np.concatenate([p[1] for p in parts]), run=np.concatenate([[r] * len(p[1]) for r, p in enumerate(parts)]),
             seq=np.concatenate([p[2] for p in parts]), pos=np.concatenate([p[3] for p in parts]))
    print(f'sub-{sub} ses-{ses}: {len(parts)} runs, {len(X)} trials, {len(set(np.concatenate([p[1] for p in parts]).tolist()))} images, mean|x| {np.abs(X[:500].astype(np.float32)).mean():.3f}, interpolated {parts[0][4]}', flush=True)
    if '--keep-raw' not in sys.argv:
        for f in glob.glob(f'{D}/raw/sub-{sub}/ses-{ses}/*_eeg.cdt'):
            os.remove(f)
