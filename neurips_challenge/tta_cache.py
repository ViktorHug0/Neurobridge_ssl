"""Per-TRIAL cache of the cv4 grader stream, for the lean test-time adaptation (lean_tta.py; HISTORY.md):
  <OUT>/fold<k>.npz          img, subj, idx (dataset index), F (eeg_fingerprint), G (gallery)
  <OUT>/<recipe>_f<k>.npy    raw member outputs (N, 1536) float16, in stream order (batches = consecutive 64 rows)
Stream = the sealed condition: each test subject's real trials in recording order, the fold's 5 subjects pooled and
permuted with seed fold*100+7, so every 64-trial predict() batch mixes participants.
    tta_cache.py <recipe> [...]
"""
import glob, json, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.build_alljoined_cache import PARQUET_GLOB, N_TRAIN_IMAGES  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset, eeg_fingerprint  # noqa: E402

DATA = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
ROOT = '/nasbrain/p20fores/Neurobridge_SSL/results/alljoined/track1/cv4'
OUT = f'{ROOT}/tta_cache'
FOLDS = {1: [2, 4, 5, 11, 15], 2: [7, 14, 16, 17, 20], 3: [3, 8, 10, 12, 19], 4: [1, 6, 9, 13, 18]}  # = cv4.sh


def chronological(subject):
    """dataset indices (image * 80 + repeat) of the subject's real test trials, in recording order."""
    frames = [pd.read_parquet(p, columns=['type', 'start', 'label', 'stim_file', 'session', 'run'])
              for p in glob.glob(PARQUET_GLOB) if f'subject={subject}-' in os.path.basename(p)]
    ev = pd.concat(frames); ev = ev[(ev.type == 'Image') & (ev.label == 'stim_test')].sort_values(['session', 'run', 'start'])
    counts = np.load(f'{DATA}/sub-{subject:02d}/test_counts.npy').ravel()
    seen = np.zeros(200, int); idx = []
    for stim in ev.stim_file:
        i = int(stim.split('.')[0]) - N_TRAIN_IMAGES
        if seen[i] < min(counts[i], 80):
            idx.append(i * 80 + seen[i])
        seen[i] += 1
    return np.array(idx)


@torch.no_grad()
def cache(recipe, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    os.makedirs(OUT, exist_ok=True)
    run_dirs = {int(f.split('/')[-4][4:]): os.path.dirname(f) for f in glob.glob(f'{ROOT}/{recipe}/fold*/*/*/track1_score.json')}
    for k, subs in FOLDS.items():
        path = f'{OUT}/{recipe}_f{k}.npy'
        if os.path.exists(path) or k not in run_dirs:
            continue
        d = run_dirs[k]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        # checkpoint_sealed_best = the epoch with the best sealed score on these test people (known optimism, accepted:
        # the final retrain uses the 4-fold-optimal epoch)
        ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                        int(cfg.time_window[1]) - int(cfg.time_window[0]))
        common = f'{OUT}/fold{k}.npz'
        xs, img, subj, idx, raw, G = [], [], [], [], [], None
        for s in subs:
            order = chronological(s)
            r = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            r = dict(zip(np.sort(order).tolist(), r))
            ds = build_test_dataset(cfg, [s])
            G = ds.image_features[:, 0] if G is None else G
            for i in order:
                eeg, _, _, _, obj, _, _ = ds[int(i)]
                xs.append(torch.as_tensor(eeg)); img.append(int(obj)); subj.append(s); idx.append(int(i))
                raw.append(torch.as_tensor(r[int(i)]))
        o = np.random.default_rng(k * 100 + 7).permutation(len(img))
        xs, raw = [xs[i] for i in o], [raw[i] for i in o]
        img, subj, idx = np.array(img)[o], np.array(subj)[o], np.array(idx)[o]
        out = []
        for b in range(0, len(img), 64):
            X = torch.stack(xs[b:b + 64]).to(device)
            out.append(eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float().cpu().numpy())
        if not os.path.exists(common):
            F = np.concatenate([eeg_fingerprint(torch.stack(raw[b:b + 256]).to(device)).cpu().numpy() for b in range(0, len(raw), 256)])
            np.savez(common + f'.{recipe}.tmp.npz', img=img, subj=subj, idx=idx, F=F, G=np.asarray(G))
            os.replace(common + f'.{recipe}.tmp.npz', common)
        else:
            assert np.array_equal(np.load(common)['idx'], idx), 'stream differs from the cached fold'
        np.save(path + '.tmp.npy', np.concatenate(out).astype(np.float16)); os.replace(path + '.tmp.npy', path)
        print(f'{recipe} fold {k}: {len(img)} trials', flush=True)
        del model, eproj; torch.cuda.empty_cache()


if __name__ == '__main__':
    for r in sys.argv[1:]:
        cache(r)
