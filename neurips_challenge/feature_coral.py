"""Label-free re-colouring of TSConv's 40 filter outputs (SUBJECT_LOG D7, 2026-10-16). D5: fine-tuning ONLY the 40x40 1x1
filter-mixing conv per person recovers +7.3 (58% of the head headroom); the per-person updates are moderately low-rank
and mostly person-specific. D6: shifting features to zero inside the head breaks it. So: map each trial's 40 filter
outputs (B, 40, 1, 36) to the TRAINING population's statistics, x' = mu_pop + C_pop^1/2 C^-1/2 (x - mu), with (mu, C)
from the person (all their trials: oracle grouping, still no labels) or from the 64-trial mixed batch (deployable);
full covariance or diagonal; strength alpha. TSC-bn64 cv4, grader_sim stream (= feature_transform.py), official metric.
    feature_coral.py [recipe]
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, official, DATA, run_dirs, FOLDS  # noqa: E402
from neurips_challenge.score_track1 import (batch_transform_fp, build_modules, build_test_dataset, eeg_fingerprint,  # noqa: E402
                                            fit_fingerprint_lda)

CFGS = [('none', 1., 'plain'), ('none', 1., 'fp')] + \
       [(src, a, o) for src in ('subj_full', 'subj_diag', 'batch_full', 'batch_diag') for a in (1., .5) for o in ('plain', 'fp')]


def stats(z):
    """z (N, 40, 1, T) -> mean (40,), covariance (40, 40) over trials x time"""
    f = z.squeeze(2).transpose(1, 2).reshape(-1, z.shape[1]).double()
    mu = f.mean(0); c = f - mu
    return mu, c.T @ c / len(c)


def recolour(z, mu, C, pop, diag, alpha, eps=1e-3):
    mu_p, C_p = pop
    if diag:
        A = torch.diag((torch.diag(C_p) / (torch.diag(C) + eps)).sqrt())
    else:
        w, V = torch.linalg.eigh(C + eps * torch.eye(len(C), dtype=C.dtype, device=C.device))
        wp, Vp = torch.linalg.eigh(C_p)
        A = (Vp * wp.clamp_min(0).sqrt()) @ Vp.T @ (V * w.rsqrt()) @ V.T          # C_p^1/2 C^-1/2
    f = z.squeeze(2).transpose(1, 2).double()                                      # (B, T, 40)
    out = mu_p + (f - mu) @ A.T
    return (f + alpha * (out - f)).transpose(1, 2).unsqueeze(2).to(z.dtype)


@torch.no_grad()
def main(recipe, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    from module.dataset import EEGPreImageDataset
    res = {c: {} for c in CFGS}; state = {}

    def hook(m, a):
        z = a[0]
        if state['mode'] == 'collect':
            state['feats'].append(z.detach()); return None
        src, alpha = state['cfg'][0], state['cfg'][1]
        if src == 'none':
            return None
        if src.startswith('subj'):  # per-trial person statistics (oracle grouping): one recolouring per person in the batch
            out = z.clone()
            for s in np.unique(state['ids_b']):
                r = torch.as_tensor(state['ids_b'] == s, device=z.device)
                out[r] = recolour(z[r], *state['subj'][s], state['pop'], src.endswith('diag'), alpha)
            return (out,)
        return (recolour(z, *stats(z), state['pop'], src.endswith('diag'), alpha),)

    for k, subs in FOLDS.items():
        d = run_dirs(recipe)[k]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                        int(cfg.time_window[1]) - int(cfg.time_window[0]))
        h = model.projection.register_forward_pre_hook(hook)
        sid = lambda n: torch.full((n,), 10 ** 6, device=device)
        run = lambda X: eproj(run_eeg_backbone(model, cfg, X, sid(len(X)))).float()
        # population statistics: TRAINING-image trials of 5 of the fold's training Alljoined subjects (contiguous blocks)
        train_subs = [s for s in range(1, 21) if s not in subs][:5]
        tds = EEGPreImageDataset(train_subs, DATA, cfg.selected_channels, cfg.time_window, cfg.image_feature_dir,
                                 getattr(cfg, 'text_feature_dir', ''), False, [], False, False, None, True)
        per = len(tds) // len(train_subs)
        state.update(mode='collect', feats=[])
        for s_i in range(len(train_subs)):
            idx = range(s_i * per, s_i * per + 2048)
            for j in range(0, 2048, 256):
                run(torch.stack([torch.as_tensor(tds[i][0]) for i in list(idx)[j:j + 256]]).to(device))
        state['pop'] = stats(torch.cat(state['feats']))
        lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
        xs, img, subj, raw, G = [], [], [], [], None
        for s in subs:
            order = chronological(s)
            r_ = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            r_ = dict(zip(np.sort(order).tolist(), r_))
            ds = build_test_dataset(cfg, [s])
            G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
            for i in order:
                eeg, _, _, _, obj, _, _ = ds[int(i)]
                xs.append(torch.as_tensor(eeg)); img.append(int(obj)); subj.append(s); raw.append(torch.as_tensor(r_[int(i)]))
        img, subj = np.array(img), np.array(subj)
        state['subj'] = {}
        for s in subs:  # person statistics from all their trials (oracle grouping, label-free)
            state.update(mode='collect', feats=[])
            rows = np.where(subj == s)[0]
            for j in range(0, len(rows), 512):
                run(torch.stack([xs[i] for i in rows[j:j + 512]]).to(device))
            state['subj'][s] = stats(torch.cat(state['feats']))
        o = np.random.default_rng(k * 100 + 7).permutation(len(img))
        xs = [xs[i] for i in o]; raw = [raw[i] for i in o]; img, subj = img[o], subj[o]
        outs = {c: [] for c in CFGS}; state['mode'] = 'apply'
        for b in range(0, len(img), 64):
            X = torch.stack(xs[b:b + 64]).to(device); F = eeg_fingerprint(torch.stack(raw[b:b + 64]).to(device))
            state['ids_b'] = subj[b:b + 64]
            for c in CFGS:
                state['cfg'] = c
                p = run(X); p = p / p.norm(dim=1).mean()
                outs[c].append((batch_transform_fp(p, F, lda) if c[2] == 'fp' else p).cpu().numpy())
        for c in CFGS:
            r = official(np.concatenate(outs[c]), img, subj, np.asarray(G))
            res[c].update({s: r[s] for s in subs})
        h.remove()
        print(f'fold {k}: plain {100 * np.mean([res[CFGS[0]][s] for s in subs]):.2f} router {100 * np.mean([res[CFGS[1]][s] for s in subs]):.2f}', flush=True)
        del model; torch.cuda.empty_cache()
    for ref_key in (('none', 1., 'plain'), ('none', 1., 'fp')):
        ref = res[ref_key]
        print(f'--- vs {ref_key}')
        for c in CFGS:
            if c[2] != ref_key[2]:
                continue
            dl = np.array([res[c][s] - ref[s] for s in ref])
            print(f'RESULT {c[0]:11s} a{c[1]} {c[2]:5s} {100 * np.mean(list(res[c].values())):.2f}  {100 * dl.mean():+.2f} ({(dl > 0).sum()}/{(dl < 0).sum()})', flush=True)
    json.dump({' '.join(map(str, c)): {str(s): v for s, v in res[c].items()} for c in CFGS},
              open(f'results/alljoined/track1/routing/feature_coral_{recipe}.json', 'w'))


if __name__ == '__main__':
    torch.manual_seed(0)
    z = torch.randn(512, 40, 1, 36, dtype=torch.float64) * 2 + 1
    pop = stats(torch.randn(4096, 40, 1, 36, dtype=torch.float64) @ torch.eye(36, dtype=torch.float64))
    y = recolour(z, *stats(z), pop, diag=False, alpha=1.)
    mu, C = stats(y)
    assert torch.allclose(mu, pop[0], atol=1e-6) and torch.allclose(C, pop[1], atol=1e-2), 'recoloured stats = population stats'
    assert torch.allclose(recolour(z, *stats(z), pop, False, 0.), z), 'alpha 0 = identity'
    main(sys.argv[1] if len(sys.argv) > 1 else 'tsc_dbn40_bn64_things')
