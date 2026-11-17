"""Q06 ds007964 (Kim et al. 2026: 19 people, 127 channels, 100 ms image + 200 ms blank, 1,900 THINGS images x 3): export to
our trial layout and zero-shot check with a time-shift sweep (as P06 / Q03). Its images are THINGS training images, so the
grader targets and the clean-proxy split already exist: the check uses only its images among the 5,000 never-trained ones.
Chain as the other exports: average reference over the scalp channels -> the 32 Alljoined electrodes (copied, or
spline-interpolated when absent) -> notch 50/60 -> 0.1-75 Hz -> 120 Hz -> RobustScaler per channel -> -0.2..0.8 s -> baseline -> clamp 20.
events.tsv onsets are in quarter-rate samples (x 4 = the 1,000 Hz marker positions).
    nq04_ds007964.py <subjects, e.g. 01,03> [member recipe:checkpoint]"""
import pandas as pd, mne
from sklearn.preprocessing import RobustScaler
from mne.channels.interpolation import _make_interpolation_matrix
from nc_lib import *

D = '/nasbrain/p20fores/ds007964'; SET = '/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set/training_images'
CH = json.load(open(f'{DATA}/info.json'))['ch_names']; M5 = {k.lower(): v for k, v in mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos'].items()}
INDEX = {fn: i for i, fn in enumerate(fn for c in sorted(os.listdir(SET)) for fn in sorted(os.listdir(f'{SET}/{c}')))}
assert len(INDEX) == 16540


def export(s):
    p = f'{D}/export_sub-{s}.npz'
    if os.path.exists(p):
        return np.load(p)
    raw = mne.io.read_raw_brainvision(f'{D}/raw/sub-{s}_task-rsvp_eeg.vhdr', preload=True, verbose='ERROR')
    scalp = [c for c in raw.ch_names if c.lower() in M5]; raw.pick(scalp); sf = raw.info['sfreq']
    pos = np.array([M5[c.lower()] for c in scalp]); pos /= np.linalg.norm(pos, axis=1, keepdims=True); low = [c.lower() for c in scalp]
    W = np.zeros((len(CH), len(scalp)))
    for r, c in enumerate(CH):
        if c.lower() in low:
            W[r, low.index(c.lower())] = 1
        else:
            t = M5[c.lower()][None]; W[r] = _make_interpolation_matrix(pos, t / np.linalg.norm(t))[0]
    y = (W @ (np.eye(len(scalp)) - 1 / len(scalp))) @ raw.get_data()
    r2 = mne.io.RawArray(y, mne.create_info(CH, sf, 'eeg'), verbose='ERROR')
    r2.notch_filter([f for b in (50.0, 60.0) for f in np.arange(b, min(sf / 2, 301), b)], phase='zero', verbose='ERROR'); r2.filter(0.1, 75.0, verbose='ERROR'); r2.resample(120.0, verbose='ERROR')
    z = RobustScaler().fit_transform(r2.get_data().T).T.astype(np.float32); ev = pd.read_csv(f'{D}/raw/sub-{s}_task-rsvp_events.tsv', sep='\t'); ev = ev[ev.istarget == 0]
    X, img = [], []
    for onset, name in zip(ev.onset, ev.stimname):
        t = int(round(onset * 4 * 120 / sf)) - 24
        if name in INDEX and 0 <= t and t + 120 <= z.shape[1]:
            e = z[:, t:t + 120].copy(); e -= e[:, :24].mean(1, keepdims=True); X.append(np.clip(e, -20, 20)); img.append(INDEX[name])
    np.savez(p, X=np.stack(X).astype(np.float16), img=np.array(img))
    print(f'sub-{s}: {len(scalp)} scalp channels, interpolated {[c for c in CH if c.lower() not in low]}, {len(X)} of {len(ev)} trials on THINGS training images, {len(set(img))} images, {np.isin(img, HELD).sum()} trials on never-trained images', flush=True)
    return np.load(p)


if __name__ == '__main__':
    subs = sys.argv[1].split(','); recipe, ck = (sys.argv[2] if len(sys.argv) > 2 else 'mi_rmix_ho5k_lag5:checkpoint_ep10.pth').split(':')
    data = {s: export(s) for s in subs}; f = load(recipe, 1, ck); gal = gallery(); t = torch.arange(120); pos_in_held = {int(i): k for k, i in enumerate(HELD)}
    print(recipe, ck, '| reference, new Alljoined people: single-trial top-5 0.73 %, P(target > random) .66 | chance 0.10 %, .5')
    for s_ in range(-8, 9, 2):
        g = (t - s_).clamp(0, 119); line = []
        for s, d in data.items():
            m = np.isin(d['img'], HELD); x = torch.tensor(d['X'][m].astype(np.float32))[:, :, g]; img = torch.tensor([pos_in_held[int(i)] for i in d['img'][m]])
            r = rank(scaled(f(x, bs=1000)), img, gal).float(); line.append('%s: %.2f %% / %.3f (n %d)' % (s, 100 * (r < 5).float().mean(), 1 - r.mean() / 4999, len(r)))
        print('  shift %+2d (%+4.0f ms): %s' % (s_, 1000 * s_ / 120, '   '.join(line)), flush=True)
