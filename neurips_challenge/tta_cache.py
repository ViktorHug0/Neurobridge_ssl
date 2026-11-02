"""Per-TRIAL cache of the grader_sim --shuffle stream for test-time-adaptation experiments (TTA_LOG.md, 2026-10-20).
ens_cache.py stores per-(subject, image) means after a fixed transform; this stores what predict() sees and emits
before any transform, so any batch-level method can be replayed exactly:
  <OUT>/fold<k>.npz          img, subj, idx (dataset index), F (eeg_fingerprint), G (gallery), lda (m0, sd, W)
  <OUT>/<recipe>_f<k>.npy    raw member outputs (N, 1536) float16, in stream order (batches = consecutive 64 rows)
Same stream as ens_cache.py / grader_sim.py --shuffle: chronological per subject, permuted with seed fold*100+7.
  <OUT>/<recipe>_f<k>_s<d>.npy  the same with the input shifted by d samples (edge-padded): test-time augmentation
    tta_cache.py <recipe> [...] [--shifts -2 -1 1 2]
"""
import glob, json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, DATA, FOLDS  # noqa: E402
from neurips_challenge import cv4_ensemble  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset, eeg_fingerprint, fit_fingerprint_lda  # noqa: E402

OUT = '/nasbrain/p20fores/Neurobridge_SSL/results/alljoined/track1/cv4/tta_cache'
SETS = {'cv4': (OUT, FOLDS),  # s11 = the sealed cohort size: 2 splits of 11 unseen people (s11.sh), models trained on 9
        's11': (OUT.replace('/cv4/', '/s11/'), {1: [2, 3, 4, 5, 7, 11, 14, 15, 16, 17, 20], 2: [1, 2, 3, 6, 8, 9, 10, 12, 13, 18, 19]})}


CH = ['Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz',
      'O2', 'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2', 'F6', 'Fp2']  # every member's channel order
MIR = [CH.index(c if not c[-1].isdigit() else c[:-1] + str(int(c[-1]) + (1 if int(c[-1]) % 2 else -1))) for c in CH]


def shift(X, d):
    """(B, C, T) shifted by d samples along time, edge-padded (d > 0: later); d = 'm': left-right mirrored scalp (T44)"""
    if d == 'm':
        return X[:, MIR]
    if d == 'n':  # AUG_LOG: polarity-inverted input (is sub-09 sign-flipped?)
        return -X
    if d == 0:
        return X
    return torch.cat([X[..., :1].expand(*X.shape[:-1], d), X[..., :-d]], -1) if d > 0 else \
        torch.cat([X[..., -d:], X[..., -1:].expand(*X.shape[:-1], -d)], -1)


def soup(a, b):
    """T50: uniform weight average of two checkpoints of the same run (float tensors; buffers like BN counts kept from a)"""
    out = dict(a)
    for key in ('model_state_dict', 'eeg_projector_state_dict'):
        out[key] = {n: (v + b[key][n]) / 2 if v.is_floating_point() else v for n, v in a[key].items()}
    return out


TRACK1 = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1'


def bridge(src, lam=1e-2):
    """AUG_LOG 2026-10-30 target diversity: linear ridge map (no intercept, so norms and centring carry over) from a
    member's training target space to the grader layer, fitted on L2-normalised training-image features only"""
    n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    ld = lambda d: n(np.load(f'{d}/image_train.npy').astype(np.float64).reshape(-1, np.load(f'{d}/image_train.npy', mmap_mode='r').shape[-1]))
    X, Y = ld(src), ld(TRACK1)
    return np.linalg.solve(X.T @ X + lam * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1]), X.T @ Y).astype(np.float32)


@torch.no_grad()
def cache(recipe, shifts=(0,), set_='cv4', device=torch.device('cuda:0'), mc=0, last=False, soup_=False):
    from train import run_eeg_backbone
    out_dir, folds = SETS[set_]
    os.makedirs(out_dir, exist_ok=True)
    root = cv4_ensemble.ROOT.replace('/cv4', f'/{set_}')
    run_dirs = lambda r: {int(f.split('/')[-4][4:]): os.path.dirname(f) for f in glob.glob(f'{root}/{r}/fold*/*/*/track1_score.json')}
    for k, subs in folds.items():
        paths = {d: f'{out_dir}/{recipe}' + (f'_mc{mc}' if mc else '') + ('_last' if last else '') + ('_soup' if soup_ else '') + f'_f{k}' + (f'_s{d}' if d else '') + '.npy' for d in shifts}
        if all(os.path.exists(p_) for p_ in paths.values()):
            continue
        if k not in run_dirs(recipe):  # AUG_LOG screens: only some folds trained yet
            continue
        d = run_dirs(recipe)[k]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        tgt = str(cfg.image_feature_dir).split(',')[0].rstrip('/')
        W = None if os.path.basename(tgt) == os.path.basename(TRACK1) else bridge(tgt)
        # checkpoint_sealed_best = the epoch with the best sealed score ON THESE TEST PEOPLE (optimistic); --last: the
        # fixed-budget final epoch (checkpoint_test_best under --select_best_on fixed), as a sealed retrain would use
        ckpt = torch.load(f'{d}/checkpoint_{"test" if last else "sealed"}_best.pth', map_location=device, weights_only=False)
        if soup_:  # sealed_best + the final epoch, averaged
            ckpt = soup(ckpt, torch.load(f'{d}/checkpoint_test_best.pth', map_location=device, weights_only=False))
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                        int(cfg.time_window[1]) - int(cfg.time_window[0]))
        if mc:  # dropout layers only back in train mode (BatchNorm stays in eval)
            drops = [m for m in list(model.modules()) + list(eproj.modules()) if isinstance(m, torch.nn.Dropout) and m.p > 0]
            [m.train() for m in drops]; print(f'{recipe}: {len(drops)} active dropout layers, p = {sorted({m.p for m in drops})}', flush=True)
        common = f'{out_dir}/fold{k}.npz'
        xs, img, subj, idx, raw, G = [], [], [], [], [], None
        for s in subs:  # = ens_cache.py's stream construction
            order = chronological(s)
            r = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            r = dict(zip(np.sort(order).tolist(), r))
            ds = build_test_dataset(cfg, [s])
            G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
            for i in order:
                eeg, _, _, _, obj, _, _ = ds[int(i)]
                xs.append(torch.as_tensor(eeg)); img.append(int(obj)); subj.append(s); idx.append(int(i))
                raw.append(torch.as_tensor(r[int(i)]))
        o = np.random.default_rng(k * 100 + 7).permutation(len(img))
        xs, raw = [xs[i] for i in o], [raw[i] for i in o]
        img, subj, idx = np.array(img)[o], np.array(subj)[o], np.array(idx)[o]
        out = {d: [] for d in shifts}
        for b in range(0, len(img), 64):
            X = torch.stack(xs[b:b + 64]).to(device)
            for d in shifts:
                run = lambda: eproj(run_eeg_backbone(model, cfg, shift(X, d), torch.full((len(X),), 10 ** 6, device=device))).float()
                out[d].append((sum(run() for _ in range(mc)) / mc if mc else run()).cpu().numpy())  # mc: MC-dropout mean (T48)
        if not os.path.exists(common):
            F = np.concatenate([eeg_fingerprint(torch.stack(raw[b:b + 256]).to(device)).cpu().numpy() for b in range(0, len(raw), 256)])
            lda = {k_: v.cpu().numpy() for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
            np.savez(common + f'.{recipe}.tmp.npz', img=img, subj=subj, idx=idx, F=F, G=np.asarray(G), **{f'lda_{k_}': v for k_, v in lda.items()})
            os.replace(common + f'.{recipe}.tmp.npz', common)
        else:
            assert np.array_equal(np.load(common)['idx'], idx), 'stream differs from the cached fold'
        for d, p_ in paths.items():
            z = np.concatenate(out[d])
            if W is not None:  # the member's own-space output, to refit the bridge offline
                np.save(p_.replace('.npy', '_raw.npy'), z.astype(np.float16))
            np.save(p_ + '.tmp.npy', (z if W is None else z @ W).astype(np.float16)); os.replace(p_ + '.tmp.npy', p_)
        print(f'{recipe} fold {k}: {len(img)} trials', flush=True)
        del model, eproj; torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('recipes', nargs='+'); ap.add_argument('--shifts', nargs='+', type=int, default=[]); ap.add_argument('--set', default='cv4'); ap.add_argument('--mc', type=int, default=0); ap.add_argument('--mirror', action='store_true'); ap.add_argument('--negate', action='store_true'); ap.add_argument('--last', action='store_true'); ap.add_argument('--soup', action='store_true')
    a = ap.parse_args()
    X = torch.arange(5.).view(1, 1, 5)
    assert shift(X, 2).tolist() == [[[0, 0, 0, 1, 2]]] and shift(X, -1).tolist() == [[[1, 2, 3, 4, 4]]]
    for r in a.recipes:
        cache(r, tuple([0] + a.shifts + (['m'] if a.mirror else []) + (['n'] if a.negate else [])), a.set, mc=a.mc, last=a.last, soup_=a.soup)
