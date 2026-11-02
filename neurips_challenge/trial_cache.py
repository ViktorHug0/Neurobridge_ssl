"""Per-trial cache for routing experiments: for each cv4 run (best-epoch checkpoint) and test subject, every trial's
raw prediction dotted with the normalised gallery, (n_trials, 200) float16, plus its image label, norm and the raw
prediction itself (float16; label-free features such as member agreement need it).
The grader averages raw predictions per (subject, image) and then ranks by cosine; for one query the cosine's
denominator is shared by all gallery items, so ranking needs only sum_t w_t (p_t . g). Any per-trial weighting or
per-trial mixing of members can therefore be scored from these dots alone.
Trial order = test.npy order (image-major, then repeat), checked against the file on the first batch.
    trial_cache.py <recipe> [--device cpu] [--subjects 2 4]   -> <run>/trial_dots.npz
"""
import argparse, json, os, sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.cv4_ensemble import DATA, run_dirs  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset, _normalize  # noqa: E402


@torch.no_grad()
def cache_run(d, device, only=None, adabn=False):
    from train import run_eeg_backbone
    out = os.path.join(d, 'trial_dots_adabn.npz' if adabn else 'trial_dots.npz')
    if os.path.exists(out) and only is None:
        return
    cfg = SimpleNamespace(**json.load(open(os.path.join(d, 'train_config.json'))))
    cfg.eeg_data_dir = DATA
    name = 'checkpoint_sealed_best.pth' if os.path.exists(os.path.join(d, 'checkpoint_sealed_best.pth')) else 'checkpoint_test_best.pth'
    ckpt = torch.load(os.path.join(d, name), map_location=device, weights_only=False)
    n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
    subjects = only or cfg.test_subject_ids
    model, eproj, iproj = build_modules(cfg, ckpt, build_test_dataset(cfg, [subjects[0]]), device, n_times)
    if adabn:  # test-time BatchNorm: every BN normalises with the 64-trial batch's own statistics (AdaBN-style)
        for mod in model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.train(); mod.track_running_stats = False; mod.running_mean = mod.running_var = None
    res = {}
    for s in subjects:
        ds = build_test_dataset(cfg, [s])
        G = _normalize(iproj(torch.tensor(ds.image_features[:, 0, :ds.target_dims[0]], dtype=torch.float32, device=device)).float().cpu().numpy())
        dots, norms, labels, embs = [], [], [], []
        raw = np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)
        perm = np.random.default_rng(s).permutation(len(ds)) if adabn else None  # chronological-like: random within subject
        loader = DataLoader(torch.utils.data.Subset(ds, perm.tolist()), batch_size=64) if adabn else DataLoader(ds, batch_size=512, shuffle=False)
        for b, (eeg, _, _, sb, obj, _, _) in enumerate(loader):
            if b == 0 and not adabn:
                e = eeg[:4].numpy().reshape(4, -1)  # the loader rounds slightly: compare by correlation
                assert all(np.corrcoef(e[i], raw[i].ravel())[0, 1] > 0.9999 for i in range(4)), 'trial order != test.npy order'
            p = eproj(run_eeg_backbone(model, cfg, eeg.to(device), torch.full_like(sb, 10 ** 6).to(device))).float().cpu().numpy()
            dots.append((p @ G.T).astype(np.float16)); embs.append(p.astype(np.float16)); norms.append(np.linalg.norm(p, axis=1)); labels.append(obj.numpy())
        inv = np.argsort(perm) if adabn else slice(None)  # back to test.npy order
        res[f's{s}_dots'] = np.concatenate(dots)[inv]; res[f's{s}_norm'] = np.concatenate(norms)[inv]; res[f's{s}_label'] = np.concatenate(labels)[inv]
        res[f's{s}_emb'] = np.concatenate(embs)[inv]; res['gallery'] = G  # raw embeddings (identity projector: DINOv2 space)
        print(f'{d} subject {s}: {len(res[f"s{s}_label"])} trials', flush=True)
    if only is None:
        np.savez(out + '.tmp.npz', **res); os.replace(out + '.tmp.npz', out)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('recipes', nargs='+'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--subjects', type=int, nargs='*'); p.add_argument('--fold', type=int); p.add_argument('--adabn', action='store_true')
    p.add_argument('--dirs', action='store_true', help='the positional args are run directories (e.g. the s11 testbed)')
    a = p.parse_args()
    if a.dirs:
        for d in a.recipes:
            cache_run(d, torch.device(a.device), None, a.adabn)
        sys.exit()
    for r in a.recipes:
        for k, d in sorted(run_dirs(r).items()):
            if a.fold and k != a.fold:
                continue
            res = cache_run(d, torch.device(a.device), a.subjects, a.adabn)
            if a.subjects:  # smoke test: subject-aggregated top-5 must match the run's own score
                ref = json.load(open(os.path.join(d, 'track1_score_sealed.json' if os.path.exists(os.path.join(d, 'checkpoint_sealed_best.pth')) else 'track1_score.json')))['subject_agg_by_subject']
                for s in a.subjects:
                    D, L = res[f's{s}_dots'].astype(np.float64), res[f's{s}_label']
                    M = np.zeros((200, 200)); np.add.at(M, L, D)
                    t5 = np.mean([(M[i] > M[i, i]).sum() < 5 for i in range(200)])
                    print(f'subject {s}: top5 {100 * t5:.2f} vs run {100 * ref[str(s)]["top5"]:.2f}')
                    assert abs(t5 - ref[str(s)]['top5']) < 0.011
