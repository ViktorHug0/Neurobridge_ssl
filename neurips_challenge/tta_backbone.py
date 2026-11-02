"""Model-internal identity features for in-batch person grouping (TTA_LOG T20). The members' OUTPUTS carry no usable
identity (T3: purity 0.24); intermediate layers may. Caches, per cv4 fold, a layer's activations for the grader
stream (same order as tta_cache.py) and for 1000 training-image trials of each of the fold's 15 Alljoined training
people (offline WCCN statistics):
  <OUT>/bb_<recipe>_<layer>_f<k>.npy          (N, d) float16, stream order
  <OUT>/bb_<recipe>_<layer>_f<k>_train.npz    X (15000, d) float16, y person
layers: 'proj_in' = input of TSConv's 1x1 filter-mixing conv (40 x 36 = 1440, D5's per-person map lives here),
        'backbone' = run_eeg_backbone output (input of the projection head).
    tta_backbone.py <recipe> <layer>
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, DATA, run_dirs, FOLDS  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset  # noqa: E402
from neurips_challenge.tta_cache import OUT  # noqa: E402


@torch.no_grad()
def main(recipe, layer, device=torch.device('cuda:0'), n_train=1000):
    from train import run_eeg_backbone
    from module.dataset import EEGPreImageDataset
    for k, subs in FOLDS.items():
        path = f'{OUT}/bb_{recipe}_{layer}_f{k}.npy'
        if os.path.exists(path):
            continue
        d = run_dirs(recipe)[k]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                        int(cfg.time_window[1]) - int(cfg.time_window[0]))
        grab = {}
        if layer == 'proj_in':
            h = model.projection.register_forward_pre_hook(lambda m, a: grab.__setitem__('z', a[0].flatten(1)))
        sid = lambda n: torch.full((n,), 10 ** 6, device=device)

        def feats(X):
            z = run_eeg_backbone(model, cfg, X.to(device), sid(len(X)))
            return (grab['z'] if layer == 'proj_in' else z).float().cpu().numpy().astype(np.float16)
        xs, idx = [], []
        for s in subs:  # = tta_cache.py's stream construction
            order = chronological(s); ds = build_test_dataset(cfg, [s])
            for i in order:
                xs.append(torch.as_tensor(ds[int(i)][0])); idx.append(int(i))
        o = np.random.default_rng(k * 100 + 7).permutation(len(xs))
        xs, idx = [xs[i] for i in o], np.array(idx)[o]
        assert np.array_equal(np.load(f'{OUT}/fold{k}.npz')['idx'], idx), 'stream differs from tta_cache'
        test = np.concatenate([feats(torch.stack(xs[b:b + 256])) for b in range(0, len(xs), 256)])
        train = [s for s in range(1, 21) if s not in subs]
        tds = EEGPreImageDataset(train, DATA, cfg.selected_channels, cfg.time_window, cfg.image_feature_dir,
                                 getattr(cfg, 'text_feature_dir', ''), False, [], False, False, None, True)
        per = len(tds) // len(train); X, y = [], []
        for j, s in enumerate(train):  # contiguous blocks per person (as feature_coral.py)
            rows = np.random.default_rng([k, s]).choice(per, n_train, replace=False) + j * per
            X.append(np.concatenate([feats(torch.stack([torch.as_tensor(tds[int(i)][0]) for i in rows[b:b + 250]])) for b in range(0, n_train, 250)]))
            y.append(np.full(n_train, s))
        if layer == 'proj_in':
            h.remove()
        np.savez(f'{OUT}/bb_{recipe}_{layer}_f{k}_train.npz', X=np.concatenate(X), y=np.concatenate(y))
        np.save(path + '.tmp.npy', test); os.replace(path + '.tmp.npy', path)
        print(f'{recipe} {layer} fold {k}: test {test.shape} train {sum(len(x) for x in X)}', flush=True)
        del model, eproj; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
