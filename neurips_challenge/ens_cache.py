"""Exact per-member cache of grader_sim.py --shuffle (sealed condition) for fast ensemble search.
Every step after a member's own batch transform is linear: grader_sim sums the members' transformed batch outputs and
the official metric averages predictions per (subject, image). So the ensemble's aggregated prediction = sum over
members of each member's aggregated transformed output, and any subset can be scored exactly from per-member caches.
Per member and fold: the same chronological + shuffled stream (seed fold*100+7), batches of 64, output / batch mean
norm, then plain | batch_transform(k=2) | batch_transform_fp (LDA on the fold's 15 training subjects + THINGS), then
the per-(subject, image) mean (official group_means). Saved to results/alljoined/track1/cv4/ensemble_cache/<recipe>.npz.
    ens_cache.py <recipe> [...]        ens_search.py scores subsets.
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, DATA, run_dirs, FOLDS  # noqa: E402
from neurips_challenge.official_metrics import group_means  # noqa: E402
from neurips_challenge.score_track1 import (batch_transform, batch_transform_fp, build_modules, build_test_dataset,  # noqa: E402
                                            eeg_fingerprint, fit_fingerprint_lda)

OUT = '/nasbrain/p20fores/Neurobridge_SSL/results/alljoined/track1/cv4/ensemble_cache'
TAGS = ('plain', 'transformed', 'fp')


@torch.no_grad()
def cache(recipe, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    path = f'{OUT}/{recipe}.npz'
    if os.path.exists(path):
        return
    acc = {t: [] for t in TAGS}; keys_all, G = [], None
    for k, subs in FOLDS.items():
        d = run_dirs(recipe)[k]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                        int(cfg.time_window[1]) - int(cfg.time_window[0]))
        lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
        xs, img, subj, raw = [], [], [], []
        for s in subs:  # = grader_sim's stream construction
            order = chronological(s)
            r = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            r = dict(zip(np.sort(order).tolist(), r))
            ds = build_test_dataset(cfg, [s])
            G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
            for i in order:
                eeg, _, _, _, obj, _, _ = ds[int(i)]
                xs.append(torch.as_tensor(eeg)); img.append(int(obj)); subj.append(s); raw.append(torch.as_tensor(r[int(i)]))
        o = np.random.default_rng(k * 100 + 7).permutation(len(img))
        xs, raw = [xs[i] for i in o], [raw[i] for i in o]; img, subj = np.array(img)[o], np.array(subj)[o]
        outs = {t: [] for t in TAGS}
        for b in range(0, len(img), 64):
            X = torch.stack(xs[b:b + 64]).to(device)
            p = eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float()
            p = p / p.norm(dim=1).mean()
            F = eeg_fingerprint(torch.stack(raw[b:b + 64]).to(device))
            outs['plain'].append(p.cpu().numpy())
            outs['transformed'].append(batch_transform(p, k=2).cpu().numpy())
            outs['fp'].append(batch_transform_fp(p, F, lda).cpu().numpy())
        for t in TAGS:
            keys, means = group_means(np.concatenate(outs[t]), np.stack([subj, img], 1))
            acc[t].append(means.astype(np.float32))
        keys_all.append(keys)
        print(f'{recipe} fold {k}: {len(img)} trials', flush=True)
        del model, eproj; torch.cuda.empty_cache()
    os.makedirs(OUT, exist_ok=True)
    np.savez(path + '.tmp.npz', keys=np.concatenate(keys_all), G=np.asarray(G), **{t: np.concatenate(acc[t]) for t in TAGS})
    os.replace(path + '.tmp.npz', path)


if __name__ == '__main__':
    for r in sys.argv[1:]:
        cache(r)
