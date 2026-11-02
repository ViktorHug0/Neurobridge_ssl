"""Joint ensemble combination through a proxy gallery (user idea, 2026-10-15; ALIGN_LOG A30): grader_sim.py --shuffle
--fp with the 3 reference members, where the members see each other's votes. predict() has no gallery, so the
16,540 TRAINING-image DINOv2 targets serve as candidates. Per trial, member m's fp-transformed output t_m gives
cosines c_m over the candidates; the product of experts is softmax(sum_m c_m / tau), and its posterior-mean embedding
g_cons = sum_j p(j) g_j is the consensus. Variants (all per trial, stateless):
  base      sum_m t_m                                   (= grader_sim --fp, must reproduce 40.05)
  agree     sum_m w_m t_m, w = M * softmax_m(beta * cos(t_m, g_cons))   (members that agree with the consensus vote more)
  mix       sum_m t_m + lam * |sum_m t_m| * g_cons       (pull the prediction toward the consensus image embedding)
    poe_sim.py [recipes...]   (default: the reference REVE-d6 + SQF-dbn + TSC-bn64)
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import chronological, official, DATA, run_dirs, FOLDS  # noqa: E402
from neurips_challenge.score_track1 import (batch_transform_fp, build_modules, build_test_dataset, eeg_fingerprint,  # noqa: E402
                                            fit_fingerprint_lda)

TARGETS = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy'
VARIANTS = ['base'] + [f'agree_t{t}_b{b}' for t in (0.02, 0.05) for b in (5, 20)] + \
           [f'mix_t{t}_l{l}' for t in (0.02, 0.05) for l in (0.25, 0.5)]


def combine(ts, Gt):
    """ts: list of (B, D) member outputs -> {variant: (B, D)}."""
    nrm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    S = torch.stack(ts)                                   # (M, B, D)
    C = nrm(S) @ Gt.T                                     # (M, B, N) cosines to the candidates
    out = {'base': S.sum(0)}
    for t in (0.02, 0.05):
        p = torch.softmax(C.sum(0) / t, dim=-1)           # product of experts over the candidates
        g = nrm(p @ Gt)                                   # (B, D) consensus embedding
        a = (nrm(S) * g).sum(-1)                          # (M, B) agreement of each member with the consensus
        for b in (5, 20):
            w = len(ts) * torch.softmax(b * a, dim=0)
            out[f'agree_t{t}_b{b}'] = (w[..., None] * S).sum(0)
        for lam in (0.25, 0.5):
            out[f'mix_t{t}_l{lam}'] = S.sum(0) + lam * S.sum(0).norm(dim=-1, keepdim=True) * g
    return out


@torch.no_grad()
def main(recipes, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    Gt = torch.tensor(np.load(TARGETS).reshape(-1, 1536), dtype=torch.float32, device=device)
    Gt = Gt / Gt.norm(dim=-1, keepdim=True)
    res = {v: {} for v in VARIANTS}
    for k, subs in FOLDS.items():
        members = []
        for r in recipes:
            d = run_dirs(r)[k]
            cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
            ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
            model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device,
                                            int(cfg.time_window[1]) - int(cfg.time_window[0]))
            members.append((cfg, model, eproj))
        lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
        xs, img, subj, raw, G = [[] for _ in members], [], [], [], None
        for s in subs:  # = grader_sim's stream
            order = chronological(s)
            r_ = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            r_ = dict(zip(np.sort(order).tolist(), r_))
            for m, (cfg, _, _) in enumerate(members):
                ds = build_test_dataset(cfg, [s])
                G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
                for i in order:
                    eeg, _, _, _, obj, _, _ = ds[int(i)]
                    xs[m].append(torch.as_tensor(eeg))
                    if m == 0:
                        img.append(int(obj)); subj.append(s); raw.append(torch.as_tensor(r_[int(i)]))
        o = np.random.default_rng(k * 100 + 7).permutation(len(img))
        xs = [[x[i] for i in o] for x in xs]; raw = [raw[i] for i in o]; img, subj = np.array(img)[o], np.array(subj)[o]
        outs = {v: [] for v in VARIANTS}
        for b in range(0, len(img), 64):
            F = eeg_fingerprint(torch.stack(raw[b:b + 64]).to(device))
            ts = []
            for m, (cfg, model, eproj) in enumerate(members):
                X = torch.stack(xs[m][b:b + 64]).to(device)
                p = eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float()
                ts.append(batch_transform_fp(p / p.norm(dim=1).mean(), F, lda))
            for v, y in combine(ts, Gt).items():
                outs[v].append(y.cpu().numpy())
        for v in VARIANTS:
            r = official(np.concatenate(outs[v]), img, subj, np.asarray(G))
            res[v].update({s: r[s] for s in subs})
        print(f'fold {k}: base {100 * np.mean([res["base"][s] for s in subs]):.2f}', flush=True)
        del members; torch.cuda.empty_cache()
    for v in VARIANTS:
        d = np.array([res[v][s] - res['base'][s] for s in res['base']])
        print(f'RESULT {v:18s} {100 * np.mean(list(res[v].values())):.2f}  vs base {100 * d.mean():+.2f} '
              f'({(d > 0).sum()}/{(d < 0).sum()})', flush=True)


if __name__ == '__main__':
    ts = [torch.randn(4, 8) for _ in range(3)]; Gt = torch.nn.functional.normalize(torch.randn(50, 8), dim=-1)
    assert torch.allclose(combine(ts, Gt)['base'], sum(ts))
    main(sys.argv[1:] or ['reve_d6_things', 'sqf_dbn_things', 'tsc_dbn40_bn64_things'])
