"""Clean calibration data for static maps: each cv fold model's outputs for its HELD-OUT subjects on their
TRAINING-split trials (16540 training images x 4 repeats; images disjoint from the 200 evaluation images, subjects
unseen by that model). Saves per subject: per-image mean output over the 4 repeats (16540, D) float16 and 4000
random single-trial outputs with their image index.
    train_split_cache.py <recipe>  -> results/alljoined/track1/routing/trainQ_<recipe>_s<subject>.npz"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.cv4_ensemble import DATA, run_dirs  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset  # noqa: E402

OUT = 'results/alljoined/track1/routing'


@torch.no_grad()
def main(recipe, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    for k, d in sorted(run_dirs(recipe).items()):
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
        n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [cfg.test_subject_ids[0]]), device, n_times)
        for s in cfg.test_subject_ids:
            out = f'{OUT}/trainQ_{recipe}_s{s}.npz'
            if os.path.exists(out):
                continue
            x = np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r')        # (1654, 10, 4, 32, 120)
            x = x.reshape(-1, 4, 32, x.shape[-1])                               # (16540 images, 4 reps, ...)
            means = np.zeros((len(x), 1536), np.float32)
            rng = np.random.default_rng(s); pick = set(rng.choice(len(x) * 4, 4000, replace=False).tolist())
            singles, single_img = [], []
            for i in range(0, len(x), 128):
                b = torch.as_tensor(np.asarray(x[i:i + 128], np.float32)).reshape(-1, 32, x.shape[-1]).to(device)
                p = eproj(run_eeg_backbone(model, cfg, b, torch.full((len(b),), 10 ** 6, device=device))).float().cpu().numpy()
                p = p.reshape(-1, 4, p.shape[-1]); means[i:i + len(p)] = p.mean(1)
                for j in range(len(p) * 4):
                    if (i * 4 + j) in pick:
                        singles.append(p[j // 4, j % 4]); single_img.append(i + j // 4)
            np.savez(out, means=means.astype(np.float16), singles=np.array(singles, np.float16), single_img=np.array(single_img))
            print(f'{recipe} fold {k} subject {s} cached', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
