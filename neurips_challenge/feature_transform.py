"""Feature-space batch transform (SUBJECT_LOG D6, user 2026-10-16). D4: ~90% of the per-person fine-tune headroom is in
the HEAD (time-locked features -> embedding); D1: no output-space map beats centring. So: is there low-rank batch-level
structure UPSTREAM of the head's nonlinearity that a stateless, label-free transform can remove?
TSC-bn64 (cv4, 4 folds, 20 subjects), grader_sim --shuffle stream (= poe_sim.py), official metric. The head is
proj_eeg = [Linear 1440 -> 64, Residual(GELU, Linear 64 -> 64), LayerNorm] then eproj Linear 64 -> 1536. Transform at
  L0  the 1440-d flattened time-locked features (input of proj_eeg)
  L1  the 64-d bottleneck after proj_eeg.0 (before the nonlinear residual MLP + LayerNorm)
with batch_transform (centring a, top-k shrink b) or batch_transform_fp (the deployed router), then optionally the
deployed output router. Reference: no feature transform + output router (= the deployed single-model pipeline).
    feature_transform.py [recipe]   (default tsc_dbn40_bn64_things)
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, official, DATA, run_dirs, FOLDS  # noqa: E402
from neurips_challenge.score_track1 import (batch_transform, batch_transform_fp, build_modules, build_test_dataset,  # noqa: E402
                                            eeg_fingerprint, fit_fingerprint_lda)

FEAT = {'none': None, 'c.5': dict(a=.5, k=0), 'c1': dict(a=1., k=0), 'k2': dict(a=.5, k=2, b=.5), 'fp': 'fp'}
CFGS = [('none', 'none', 'plain'), ('none', 'none', 'k2'), ('none', 'none', 'fp')] + \
       [(lv, f, o) for lv in ('L0', 'L1') for f in ('c.5', 'c1', 'k2', 'fp') for o in ('plain', 'fp')]


@torch.no_grad()
def main(recipe, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    res = {c: {} for c in CFGS}
    state = {}  # what the hooks apply to the current batch

    def feat(z):
        op = FEAT[state['f']]
        if op is None:
            return z
        return batch_transform_fp(z, state['F'], state['lda']) if op == 'fp' else batch_transform(z, **op)

    for k, subs in FOLDS.items():
        d = run_dirs(recipe)[k]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                        int(cfg.time_window[1]) - int(cfg.time_window[0]))
        h0 = model.proj_eeg.register_forward_pre_hook(lambda m, a: (feat(a[0]),) if state['lv'] == 'L0' else None)
        h1 = model.proj_eeg[0].register_forward_hook(lambda m, a, out: feat(out) if state['lv'] == 'L1' else None)
        lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
        xs, img, subj, raw, G = [], [], [], [], None
        for s in subs:  # = grader_sim's stream (poe_sim.py)
            order = chronological(s)
            r_ = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            r_ = dict(zip(np.sort(order).tolist(), r_))
            ds = build_test_dataset(cfg, [s])
            G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
            for i in order:
                eeg, _, _, _, obj, _, _ = ds[int(i)]
                xs.append(torch.as_tensor(eeg)); img.append(int(obj)); subj.append(s); raw.append(torch.as_tensor(r_[int(i)]))
        o = np.random.default_rng(k * 100 + 7).permutation(len(img))
        xs = [xs[i] for i in o]; raw = [raw[i] for i in o]; img, subj = np.array(img)[o], np.array(subj)[o]
        outs = {c: [] for c in CFGS}
        for b in range(0, len(img), 64):
            X = torch.stack(xs[b:b + 64]).to(device)
            state.update(F=eeg_fingerprint(torch.stack(raw[b:b + 64]).to(device)), lda=lda)
            for c in CFGS:
                state['lv'], state['f'] = c[0], c[1]
                p = eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float()
                p = p / p.norm(dim=1).mean()
                p = batch_transform(p) if c[2] == 'k2' else batch_transform_fp(p, state['F'], lda) if c[2] == 'fp' else p
                outs[c].append(p.cpu().numpy())
        for c in CFGS:
            r = official(np.concatenate(outs[c]), img, subj, np.asarray(G))
            res[c].update({s: r[s] for s in subs})
        h0.remove(); h1.remove()
        print(f'fold {k}: plain {100 * np.mean([res[CFGS[0]][s] for s in subs]):.2f} '
              f'router {100 * np.mean([res[CFGS[2]][s] for s in subs]):.2f}', flush=True)
        del model; torch.cuda.empty_cache()
    ref = res[('none', 'none', 'fp')]
    for c in CFGS:
        dlt = np.array([res[c][s] - ref[s] for s in ref])
        print(f'RESULT {" ".join(c):18s} {100 * np.mean(list(res[c].values())):.2f}  vs output router {100 * dlt.mean():+.2f} '
              f'({(dlt > 0).sum()}/{(dlt < 0).sum()})', flush=True)
    json.dump({' '.join(c): {str(s): v for s, v in res[c].items()} for c in CFGS},
              open(f'results/alljoined/track1/routing/feature_transform_{recipe}.json', 'w'))


if __name__ == '__main__':
    z = torch.randn(64, 1440); assert torch.allclose(batch_transform(z, a=1., k=0).mean(0), torch.zeros(1440), atol=1e-5)
    main(sys.argv[1] if len(sys.argv) > 1 else 'tsc_dbn40_bn64_things')
