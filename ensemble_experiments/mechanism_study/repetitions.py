"""Matched repetition-count curves: every member sees the SAME r recordings.

Unlike the older repetition-view ensemble, this uses only r total repetitions
per prediction. No averaging across partitions is used to report accuracy.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from module.dataset import EEGPreImageDataset, _eeg_cache_key, _eeg_cache_path
from ensemble_experiments.synthetic_subjects.evaluate_repetition_ensemble import (
    _load_config, _build_modules, _encode_group_means,
)
from .common import REPO, OUTPUT, COMMITTEES, unit, margin_metrics, load_member, write_json


def checkpoint_paths(subject):
    manifest = json.loads((REPO / 'ensemble_experiments/legacy/scratch_claude/manifest.json').read_text())
    paths = {n: REPO / manifest[n]['folds'][str(subject)] for n in COMMITTEES['diverse3']}
    for seed in (3300, 3301, 3302):
        root = REPO / f'results/things_eeg/inter-subjects/mixup_20260421-190931/mix_raw_eeg_pairwise_linear_a0p5_seed{seed}'
        matches = list(root.glob(f'*-sub-{subject:02d}/checkpoint_test_best.pth'))
        if len(matches) != 1:
            raise ValueError(f'Expected one checkpoint: {root}, subject={subject}, found {matches}')
        paths[f'p{seed}'] = matches[0].parent
    for path in paths.values():
        if not (path / 'checkpoint_test_best.pth').is_file():
            raise FileNotFoundError(path)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--counts', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 80])
    p.add_argument('--output', type=Path, default=OUTPUT / 'repetitions')
    args = p.parse_args()
    out = args.output / f'sub-{args.subject:02d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('Requested CUDA is unavailable')
    paths = checkpoint_paths(args.subject)
    saved = {}
    for name, path in paths.items():
        dest = out / f'{name}.npz'
        if dest.exists():
            with np.load(dest) as f:
                saved[name] = {k: f[k] for k in f.files}
            continue
        cfg = _load_config(path, args.device, args.subject)
        if cfg.time_window != [0, 250] or cfg.selected_channels:
            raise ValueError('This matched study requires full, identical channel/time input')
        dataset_kwargs = dict(subject_ids=[args.subject], eeg_data_dir=cfg.eeg_data_dir,
            selected_channels=cfg.selected_channels, time_window=cfg.time_window,
            image_feature_dir=cfg.image_feature_dir, text_feature_dir='', image_aug=False,
            aug_image_feature_dirs=[], _random=False, train=False)
        dataset = EEGPreImageDataset(average=True, **dataset_kwargs)
        key = _eeg_cache_key(subject_id=args.subject, train=False, average=False,
            selected_channels=cfg.selected_channels, time_window=cfg.time_window)
        cache = Path(_eeg_cache_path(cfg.eeg_data_dir, key))
        if not cache.exists():
            EEGPreImageDataset(average=False, **dataset_kwargs)
        raw = np.load(cache, mmap_mode='r')[:, 0]
        checkpoint = torch.load(path / 'checkpoint_test_best.pth', map_location='cpu', weights_only=False)
        model, ep, ip = _build_modules(cfg, checkpoint, dataset, args.device)
        with torch.inference_mode():
            image = unit(ip(torch.tensor(dataset.image_features[:, 0], dtype=torch.float32, device=args.device)).cpu().numpy())
        values = {}
        for repeat in range(args.repeats):
            rng = np.random.default_rng(20260907 + args.subject * 1000 + repeat)
            permutation = np.stack([rng.permutation(raw.shape[1]) for _ in range(len(raw))])
            for count in args.counts:
                if count > raw.shape[1]:
                    raise ValueError('Requested more recordings than available')
                if count == raw.shape[1] and repeat:
                    continue
                indices = permutation[:, None, :count]
                features = _encode_group_means(raw, indices, model, ep, cfg, args.subject, args.device, 32)[0]
                values[f'r{count}_draw{repeat}'] = unit(features) @ image.T
        # Regression check against the audited historical dump before trusting curves.
        if raw.shape[1] in args.counts:
            reference, _, _ = load_member(name, args.subject)
            full = values[f'r{raw.shape[1]}_draw0']
            max_error = float(np.abs(full - reference).max())
            write_json(out / f'{name}_provenance.json', {'checkpoint': str(path), 'config': vars(cfg),
                'full_average_max_score_error': max_error, 'n_repetitions': raw.shape[1]})
            if max_error > 0.002:
                raise AssertionError(f'{name}: full-average score mismatch {max_error}; inspect provenance')
        np.savez_compressed(dest, **values)
        saved[name] = values
        print('encoded', name, 'subject', args.subject, flush=True)
        del model, ep, ip, dataset, checkpoint, raw
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    for committee in ['seed3', 'diverse3']:
        names = COMMITTEES[committee]
        for key in saved[names[0]]:
            count, draw = key[1:].split('_draw')
            scores = np.stack([saved[n][key] for n in names])
            rows.append(dict(subject=args.subject, committee=committee, repetitions=int(count), draw=int(draw), **margin_metrics(scores)))
    pd.DataFrame(rows).to_csv(out / 'curves.csv', index=False)
    write_json(out / 'complete.json', {'protocol': 'historical test-selected checkpoints; fixed members, matched recordings', 'counts': args.counts, 'repeats': args.repeats})


if __name__ == '__main__':
    main()
