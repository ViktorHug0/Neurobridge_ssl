"""End-to-end check of a stateless batch post-processing under the grader's conditions, on the cv4 folds.
- Trial order: each subject's real test trials in recording order (neuralset timeline sorted by session, run,
  start, as the warm-up loader serves them), the fold's 5 subjects back to back, batches of 64 cut across subject
  boundaries; with --shuffle (the SEALED condition announced by the organisers), all the fold's trials are
  shuffled together, so every batch mixes participants. predict(X) sees X only, no state across calls.
- Members: each fold's best-epoch checkpoints; each member's batch output is divided by its batch mean norm
  (stateless scaling), optionally transformed, then members are summed.
- Metric: the official top5_acc_subject_agg (official_metrics.py, verbatim from the competition repo).
    grader_sim.py [--shuffle] [--k K] <recipe> [<recipe> ...]
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.cv4_ensemble import DATA, run_dirs  # noqa: E402
from neurips_challenge.score_track1 import batch_transform, batch_transform_fp, batch_transform_wknn, build_modules, build_test_dataset, eeg_fingerprint, fit_fingerprint_lda, fit_fingerprint_wccn, FP_AB, identity_features, RICH  # noqa: E402
from neurips_challenge.build_alljoined_cache import PARQUET_GLOB, N_TRAIN_IMAGES  # noqa: E402
from neurips_challenge.official_metrics import group_means, topk_accuracy  # noqa: E402
from neurips_challenge.trial_routing import FOLDS  # noqa: E402
import glob  # noqa: E402


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


def official(y_pred, img, subj, G):
    keys, pred_sa = group_means(y_pred, np.stack([subj, img], 1))
    n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    out = {'pooled': topk_accuracy(n(pred_sa) @ n(G).T, keys[:, 1], k=5)}
    for s in np.unique(subj):
        m = keys[:, 0] == s
        out[int(s)] = topk_accuracy(n(pred_sa[m]) @ n(G).T, keys[m, 1], k=5)
    return out


@torch.no_grad()
def main(recipes, shuffle=False, n_dirs=1, device=torch.device('cuda:0'), fp=False):
    from train import run_eeg_backbone
    res = {'plain': {}, 'transformed': {}, **({'fp': {}, 'wknn': {}, 'wknnr': {}, 'wknnl': {}, 'wknnrl': {}, 'wknna': {}} if fp else {})}
    for k, subs in FOLDS.items():
        members = []
        for r in recipes:
            d = run_dirs(r)[k]
            cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
            ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
            n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
            model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device, n_times)
            members.append((cfg, model, eproj))
        stream_x, stream_img, stream_subj, G = [[] for _ in members], [], [], None
        stream_raw = []  # the grader's raw X (test.npy trial), for the fingerprint router
        if fp:  # offline router fit: the fold's 15 training subjects + the 10 THINGS-EEG2 subjects
            lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
            wccn = {k_: v.to(device) for k_, v in fit_fingerprint_wccn(DATA, [s for s in range(1, 21) if s not in subs]).items()}  # TTA_LOG T10
            wccnr = {k_: v.to(device) for k_, v in fit_fingerprint_wccn(DATA, [s for s in range(1, 21) if s not in subs], groups=RICH).items()}  # T9
            wccna = {k_: v.to(device) for k_, v in fit_fingerprint_wccn(DATA, [s for s in range(1, 21) if s not in subs], cols=FP_AB).items()}  # T38
        for s in subs:
            order = chronological(s)
            raw = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)]) if fp else None
            raw = dict(zip(np.sort(order).tolist(), raw)) if fp else None
            for m, (cfg, _, _) in enumerate(members):  # each member reads the data with its own config
                ds = build_test_dataset(cfg, [s])
                G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
                for i in order:
                    eeg, _, _, _, obj, _, _ = ds[int(i)]
                    stream_x[m].append(torch.as_tensor(eeg))
                    if m == 0:
                        stream_img.append(int(obj)); stream_subj.append(s)
                        if fp:
                            stream_raw.append(torch.as_tensor(raw[int(i)]))
        stream_img, stream_subj = np.array(stream_img), np.array(stream_subj)
        if shuffle:  # sealed condition: batches drawn from all participants' trials at random
            o = np.random.default_rng(k * 100 + 7).permutation(len(stream_img))  # k = fold index
            stream_x = [[xs[i] for i in o] for xs in stream_x]; stream_img, stream_subj = stream_img[o], stream_subj[o]
            stream_raw = [stream_raw[i] for i in o] if fp else stream_raw
        outs = {t: [] for t in res}
        for b in range(0, len(stream_img), 64):
            per = []
            for m, (cfg, model, eproj) in enumerate(members):
                X = torch.stack(stream_x[m][b:b + 64]).to(device)
                per.append(eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float())
            per = [p / p.norm(dim=1).mean() for p in per]
            outs['plain'].append(sum(per).cpu().numpy())
            outs['transformed'].append(sum(batch_transform(p, k=n_dirs) for p in per).cpu().numpy())
            if fp:
                F = eeg_fingerprint(torch.stack(stream_raw[b:b + 64]).to(device))
                outs['fp'].append(sum(batch_transform_fp(p, F, lda) for p in per).cpu().numpy())
                outs['wknn'].append(batch_transform_wknn(per, F, wccn).cpu().numpy())
                Fr = identity_features(torch.stack(stream_raw[b:b + 64]).to(device))
                outs['wknnr'].append(batch_transform_wknn(per, Fr, wccnr).cpu().numpy())
                outs['wknnl'].append(batch_transform_wknn(per, F, wccn, c=0.85, d=0.93).cpu().numpy())  # T32 one shrink of the summed output
                outs['wknnrl'].append(batch_transform_wknn(per, Fr, wccnr, c=0.85, d=0.93).cpu().numpy())
                outs['wknna'].append(batch_transform_wknn(per, F[:, FP_AB], wccna, c=1., d='auto').cpu().numpy())  # T40 lean
        for tag in outs:
            r = official(np.concatenate(outs[tag]), stream_img, stream_subj, np.asarray(G))
            res[tag].update({s: r[s] for s in subs})
            print(f'fold {k} {tag}: pooled top5_subject_agg {100 * r["pooled"]:.2f}', flush=True)
        print(f'fold {k}: {len(stream_img)} real trials streamed', flush=True)
    for tag, v in res.items():
        print(f'{"+".join(recipes)} {tag}: mean over 20 subjects {100 * np.mean(list(v.values())):.2f}')
    for tag in [t for t in res if t != 'plain']:
        d = np.array([res[tag][s] - res['plain'][s] for s in res['plain']])
        print(f'{tag} - plain: {100 * d.mean():+.2f}, better on {(d > 0).sum()}/20, worse on {(d < 0).sum()}')
    if fp:
        d = np.array([res['fp'][s] - res['transformed'][s] for s in res['plain']])
        print(f'fp - transformed: {100 * d.mean():+.2f}, better on {(d > 0).sum()}/20, worse on {(d < 0).sum()}')
        d = np.array([res['wknn'][s] - res['fp'][s] for s in res['plain']])
        print(f'wknn - fp: {100 * d.mean():+.2f}, better on {(d > 0).sum()}/20, worse on {(d < 0).sum()}')
        for tag in ('wknnr', 'wknnl', 'wknnrl', 'wknna'):
            d = np.array([res[tag][s] - res['fp'][s] for s in res['plain']])
            print(f'{tag} - fp: {100 * d.mean():+.2f}, better on {(d > 0).sum()}/20, worse on {(d < 0).sum()}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('recipes', nargs='+'); ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--k', type=int, default=2); ap.add_argument('--fp', action='store_true')
    a = ap.parse_args(); main(a.recipes, a.shuffle, a.k, fp=a.fp)
