"""Q06 ds007964 (OpenNeuro, CC0; Kim, Quek, Moerel et al.: 19 people, 127 electrodes + reference FCz at 1000 Hz, RSVP 100 ms
image + 200 ms blank, 1,900 THINGS images of 190 concepts x 3) -> single trials in our layout (32 Alljoined electrodes,
120 Hz, -0.2..0.8 s). All 32 electrodes are recorded (FCz = the reference, rebuilt by the average reference): no interpolation.
Chain as nq02_mss_export.py: average reference over the 128 electrodes -> the 32 -> notch 50/60 Hz + harmonics -> band-pass
0.1-75 Hz -> resample 120 Hz -> RobustScaler per channel -> window at round(onset * 120 / 1000) - 24 -> baseline = first 0.2 s
-> clamp 20. Trials = the 'E  1' markers (image on), in order = the rows of events.tsv; target trials (response task) dropped.
The 1,900 images are THINGS images we already have grader targets for: 1,540 THINGS-EEG2 training images (object, image as
in training, so the positive mask pairs them with our trials of the same image), 24 test images and 336 others.
-> <D>/targets.npz: names, F (1900, 1536), obj, image (THINGS-EEG2 training ids; test images obj 1654 + concept; others obj 2000 + row), subject_offset 200
-> <D>/export/sub-XX.npz: X (n, 32, 120) float16, img (n,) row of targets.npz, seq
    nq06_ds007964_export.py <sub> [--keep-raw]"""
import json, os, sys
import mne, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler

D = '/nasbrain/p20fores/ds007964'; R = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1'
EEG2 = '/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set'
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']


def targets():
    names = sorted(pd.read_csv(f'{D}/raw/sub-01_task-rsvp_events.tsv', sep='\t').stimname.unique()); ls = lambda s: [sorted(os.listdir(f'{EEG2}/{s}/{d}')) for d in sorted(os.listdir(f'{EEG2}/{s}'))]
    tr = {fn: (o, i) for o, fs in enumerate(ls('training_images')) for i, fn in enumerate(fs)}; te = {fs[0]: o for o, fs in enumerate(ls('test_images'))}
    ex = {os.path.basename(l.strip()): j for j, l in enumerate(open(f'{R}/things_extra_names.txt'))}
    Ftr, Fte, Fex = np.load(f'{R}/image_train.npy', mmap_mode='r').reshape(16540, 1536), np.load(f'{R}/image_test.npy').reshape(200, 1536), np.load(f'{R}/things_extra.npy', mmap_mode='r')
    F, obj, image = [], [], []
    for r, n in enumerate(names):
        if n in tr:
            o, i = tr[n]; F.append(Ftr[o * 10 + i]); obj.append(o); image.append(i)
        elif n in te:
            F.append(Fte[te[n]]); obj.append(1654 + te[n]); image.append(0)
        else:
            F.append(Fex[ex[n]]); obj.append(2000 + r); image.append(0)
    np.savez(f'{D}/targets.npz', names=np.array(names), F=np.stack(F).astype(np.float32), obj=np.array(obj), image=np.array(image), subject_offset=200)
    print('targets: %d images, %d training, %d test, %d other' % (len(names), sum(o < 1654 for o in obj), sum(1654 <= o < 2000 for o in obj), sum(o >= 2000 for o in obj)), flush=True)


def chain(y, sf):
    """(32, T) volts at sf Hz, average-referenced, in CH order -> the grader chain at 120 Hz (notch, band-pass, resample, RobustScaler per channel)"""
    raw = mne.io.RawArray(y, mne.create_info(CH, sf, 'eeg'), verbose='ERROR')
    raw.notch_filter([f for b in (50.0, 60.0) for f in np.arange(b, min(sf / 2, 301), b)], phase='zero', verbose='ERROR')
    raw.filter(0.1, 75.0, verbose='ERROR'); raw.resample(120.0, verbose='ERROR')
    return RobustScaler().fit_transform(raw.get_data().T).T.astype(np.float32)


if __name__ == '__main__':
    if not os.path.exists(f'{D}/targets.npz'):
        targets()
    sub = sys.argv[1]; p = f'{D}/raw/sub-{sub}_task-rsvp_'; os.makedirs(f'{D}/export', exist_ok=True); row = {n: i for i, n in enumerate(np.load(f'{D}/targets.npz')['names'])}
    raw = mne.io.read_raw_brainvision(p + 'eeg.vhdr', preload=True, verbose='ERROR'); sf = raw.info['sfreq']; x = raw.get_data(); names = [c.lower() for c in raw.ch_names]
    ref = x.sum(0) / (len(x) + 1)  # average over the 127 recorded electrodes and the reference (0): FCz becomes -ref
    y = np.stack([-ref if c == 'FCz' else x[names.index(c.lower())] - ref for c in CH]); del x
    on = np.array([a['onset'] for a in raw.annotations if a['description'].endswith('E  1')]) * sf; ev = pd.read_csv(p + 'events.tsv', sep='\t')
    assert len(on) == len(ev) and np.abs(on / 4 - ev.onset).max() < 2, (len(on), len(ev))  # 5,700 presentations (sub-04: 5,510)
    z = chain(y, sf); X, img, seq = [], [], []
    for o, n, s, tg in zip(on, ev.stimname, ev.sequencenumber, ev.istarget):
        t = int(round(o * 120 / sf)) - 24
        if tg or t < 0 or t + 120 > z.shape[1]:
            continue
        w = z[:, t:t + 120].copy(); w -= w[:, :24].mean(1, keepdims=True); X.append(np.clip(w, -20, 20)); img.append(row[n]); seq.append(s)
    X = np.stack(X).astype(np.float16); np.savez(f'{D}/export/sub-{sub}.npz', X=X, img=np.array(img), seq=np.array(seq))
    print(f'sub-{sub}: {len(X)} trials, {len(set(img))} images, mean|x| {np.abs(X[:500].astype(np.float32)).mean():.3f}', flush=True)
    if '--keep-raw' not in sys.argv:
        os.remove(p + 'eeg.eeg')
