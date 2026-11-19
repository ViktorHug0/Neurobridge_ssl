"""K16 (side check of K15's prediction): does the anchored re-centring help INTRA-subject models, which are trained on
4-repetition averages of one subject and tested on that subject's 80-repetition averages?
Runs of results/things_eeg/cvpr/intra/<arm>/seed2025 (one model per subject, 17 occipito-parietal electrodes).
plain / calibrated with the image anchor (mean image embedding of the training images) / causal running mean.
Usage: k16_intra_calib.py <arm> [<arm> ...]"""
import pandas as pd
from sk_lib import *
from mi_lib import _DS
from module.dataset import EEGPreImageDataset

out = []
for ARM in sys.argv[1:]:
    for d in sorted(glob.glob(f'{REPO}/results/things_eeg/cvpr/intra/{ARM}/seed2025/*-sub-*')):
        c = json.load(open(f'{d}/train_config.json'))
        s = c['test_subject_ids'][0]
        ds = EEGPreImageDataset([s], c['eeg_data_dir'], c['selected_channels'], [0, 250], c['image_feature_dir'], '', True, False, None, False, False, False)
        x = T(ds.eeg_data_list[0].reshape(-1, *ds.eeg_data_list[0].shape[2:]))
        feat = c['image_feature_dir'].rstrip('/')
        for split in ('train', 'test'):
            a = np.load(f'{feat}/image_{split}.npy'); _DS[('img', split)] = a.reshape(-1, a.shape[-1])
        _DS['img_dir'] = feat
        m = GM(d, channels=x.shape[1])
        g, anchor = img_emb(m), img_emb(m, 'train').mean(0)
        e = nz(m.embed(x, s), dim=1)
        hit = lambda q, idx: (nz(q, dim=1) @ g.T).argmax(1) == idx
        p = torch.randperm(200, generator=torch.Generator().manual_seed(s))
        run_mean = e[p].cumsum(0) / torch.arange(1, 201)[:, None]
        out.append(dict(arm=ARM, sub=s, plain=top1(e, g), cal=top1(e - e.mean(0) + anchor, g), causal=100 * hit(e[p] - run_mean + anchor, p).float().mean().item()))
        print(out[-1], flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k16_intra_calib.csv', index=False)
print(df.groupby('arm').mean(numeric_only=True).drop(columns='sub').round(2).to_string())
