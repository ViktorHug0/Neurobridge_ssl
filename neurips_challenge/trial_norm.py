"""Per-trial L2 normalisation before the official subject aggregation (2026-10-19). The scorer averages RAW predictions
per (subject, image), so a trial's output norm is its vote weight; our members only divide each batch by its mean norm.
Same grader_sim --shuffle stream as ens_cache.py, official metric, cv4 20 subjects. Variants of the summed ensemble:
  ref      sum_m fp(p_m / batch mean norm)          (= ens_search, the 40.05 reference)
  pre      sum_m fp(l2(p_m))                        normalise each trial before the router
  mem      sum_m l2(fp(p_m / mean norm))            each member's routed trial output gets an equal vote
  ens      l2(sum_m fp(p_m / mean norm))            the last step of predict(): equal vote per trial
  pre_ens  l2(sum_m fp(l2(p_m)))
    trial_norm.py [member ...]    (default: the reference ensemble, plus rmix for the swap)
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, official, DATA, run_dirs, FOLDS  # noqa: E402
from neurips_challenge.score_track1 import (batch_transform_fp, build_modules, build_test_dataset, eeg_fingerprint,  # noqa: E402
                                            fit_fingerprint_lda)

REF = ['reve_d6_things', 'sqf_dbn_things', 'tsc_dbn40_bn64_things']
ENSEMBLES = {'ref3': REF, 'rmix_swap': ['rmix_things'] + REF[1:]}
l2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
VARIANTS = {'ref': lambda A, B: sum(A), 'pre': lambda A, B: sum(B), 'mem': lambda A, B: sum(map(l2, A)),
            'ens': lambda A, B: l2(sum(A)), 'pre_ens': lambda A, B: l2(sum(B))}


@torch.no_grad()
def main(members, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    res = {}  # (ensemble or member, variant) -> {subject: top-5}
    for k, subs in FOLDS.items():
        lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
        out, img, subj, G = {}, None, None, None
        for r in members:
            d = run_dirs(r)[k]
            cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
            ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
            model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                            int(cfg.time_window[1]) - int(cfg.time_window[0]))
            xs, img, subj, raw = [], [], [], []
            for s in subs:  # = ens_cache.py's stream construction
                order = chronological(s)
                r_ = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
                r_ = dict(zip(np.sort(order).tolist(), r_))
                ds = build_test_dataset(cfg, [s])
                G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
                for i in order:
                    eeg, _, _, _, obj, _, _ = ds[int(i)]
                    xs.append(torch.as_tensor(eeg)); img.append(int(obj)); subj.append(s); raw.append(torch.as_tensor(r_[int(i)]))
            o = np.random.default_rng(k * 100 + 7).permutation(len(img))
            xs, raw = [xs[i] for i in o], [raw[i] for i in o]; img, subj = np.array(img)[o], np.array(subj)[o]
            A, B = [], []
            for b in range(0, len(img), 64):
                X = torch.stack(xs[b:b + 64]).to(device)
                p = eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float()
                F = eeg_fingerprint(torch.stack(raw[b:b + 64]).to(device))
                A.append(batch_transform_fp(p / p.norm(dim=1).mean(), F, lda).cpu().numpy())
                B.append(batch_transform_fp(p / p.norm(dim=1, keepdim=True), F, lda).cpu().numpy())
            out[r] = (np.concatenate(A), np.concatenate(B))
            print(f'{r} fold {k}: {len(img)} trials', flush=True)
            del model, eproj; torch.cuda.empty_cache()
        G = np.asarray(G)
        groups = {r: [r] for r in members} | {e: m for e, m in ENSEMBLES.items() if all(x in out for x in m)}
        for e, m in groups.items():
            A, B = [out[x][0] for x in m], [out[x][1] for x in m]
            for v, f in VARIANTS.items():
                sc = official(f(A, B), img, subj, G)
                res.setdefault((e, v), {}).update({s: sc[s] for s in subs})
        print(f'fold {k}: ' + '  '.join(f'{e} {100 * np.mean([res[(e, "ref")][s] for s in subs]):.2f}' for e in groups), flush=True)
    for (e, v), r in res.items():
        dl = np.array([r[s] - res[(e, 'ref')][s] for s in r])
        print(f'RESULT {e:22s} {v:8s} {100 * np.mean(list(r.values())):.2f}  {100 * dl.mean():+.2f} ({(dl > 0).sum()}/{(dl < 0).sum()})', flush=True)
    json.dump({f'{e} {v}': {str(s): x for s, x in r.items()} for (e, v), r in res.items()},
              open('results/alljoined/track1/routing/trial_norm.json', 'w'))


if __name__ == '__main__':
    a = [np.ones((2, 3)), 3 * np.ones((2, 3))]
    assert np.allclose(VARIANTS['mem'](a, a), 2 * l2(a[0])) and np.allclose(np.linalg.norm(VARIANTS['ens'](a, a), axis=1), 1)
    main(sys.argv[1:] or REF + ['rmix_things'])
