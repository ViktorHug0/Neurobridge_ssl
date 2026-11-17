"""Q10 ds004018 (OpenNeuro, CC0; Grootswagers, Robinson, Carlson 2019: 16 people, 63 electrodes + reference Cz at 1000 Hz,
200 object images (segmented objects, 4 per category) in RSVP at 5 Hz (run 1: 40 presentations per image) or 20 Hz (run 2,
not used: too fast for our window) -> single trials in our layout, chain of nq06. All 32 Alljoined electrodes are recorded
or are the reference (rebuilt by the average reference). Targets = DINOv2-giant hidden state 26, token mean, as nq01.
-> <D>/targets.npz: names, F (200, 1536); subject_offset 300       <D>/export/sub-XX.npz: X (n, 32, 120) float16, img, seq
    nq10_ds004018.py targets | <sub> [--keep-raw]"""
import glob, os, sys
import mne, numpy as np, pandas as pd
from nq06_ds007964_export import CH, chain

D = '/nasbrain/p20fores/ds004018'
if sys.argv[1] == 'targets':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from torchvision.transforms.functional import to_tensor
    names = sorted(os.path.basename(f) for f in glob.glob(f'{D}/stimuli/stim*.png'))[:200]; assert len(names) == 200 and names[-1].startswith('stim200'), names[-1]
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', do_rescale=False); model = AutoModel.from_pretrained('facebook/dinov2-giant', torch_dtype=torch.float32).cuda().eval(); out = []
    for b in range(0, 200, 8):
        ims = []
        for n in names[b:b + 8]:  # transparent background -> the mid-grey of the screen
            im = Image.open(f'{D}/stimuli/{n}').convert('RGBA'); bg = Image.new('RGBA', im.size, (128, 128, 128, 255)); ims.append(to_tensor(Image.alpha_composite(bg, im).convert('RGB')))
        with torch.no_grad():
            out.append(model(pixel_values=proc(images=ims, return_tensors='pt')['pixel_values'].cuda(), output_hidden_states=True).hidden_states[26].float().mean(1).cpu())
    np.savez(f'{D}/targets.npz', names=np.array(names), F=torch.cat(out).numpy().astype(np.float32), subject_offset=300); print('targets saved', flush=True)
else:
    sub = sys.argv[1]; p = f'{D}/raw/sub-{sub}_task-rsvp_run-01_'; os.makedirs(f'{D}/export', exist_ok=True); row = {n: i for i, n in enumerate(np.load(f'{D}/targets.npz')['names'])}
    raw = mne.io.read_raw_brainvision(p + 'eeg.vhdr', preload=True, verbose='ERROR'); sf = raw.info['sfreq']; x = raw.get_data(); names = [c.lower() for c in raw.ch_names]
    ref = x.sum(0) / (len(x) + 1)  # average over the 63 recorded electrodes and the reference (0): Cz becomes -ref
    y = np.stack([x[names.index(c.lower())] - ref if c.lower() in names else -ref for c in CH]); assert [c for c in CH if c.lower() not in names] in ([], ['Cz']); del x
    ev = pd.read_csv(p + 'events.tsv', sep='\t'); soa = np.median(np.diff(ev.onset)); assert 195 < soa < 205, soa  # run 1 = the 5 Hz run, onsets in samples
    z = chain(y, sf); X, img, seq = [], [], []
    for o, n, s, tg in zip(ev.onset, ev.stimulusname, ev.trialnumber, ev.istarget):
        t = int(round(o * 120 / sf)) - 24
        if tg or n not in row or t < 0 or t + 120 > z.shape[1]:
            continue
        w = z[:, t:t + 120].copy(); w -= w[:, :24].mean(1, keepdims=True); X.append(np.clip(w, -20, 20)); img.append(row[n]); seq.append(s)
    X = np.stack(X).astype(np.float16); np.savez(f'{D}/export/sub-{sub}.npz', X=X, img=np.array(img), seq=np.array(seq))
    print(f'sub-{sub}: {len(X)} trials, {len(set(img))} images, mean|x| {np.abs(X[:500].astype(np.float32)).mean():.3f}', flush=True)
    if '--keep-raw' not in sys.argv:
        os.remove(p + 'eeg.eeg')
