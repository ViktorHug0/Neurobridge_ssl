"""grader_sim --shuffle --fp, but the batch_transform_fp statistics (batch mean, fingerprint regression, top-k output
directions) are estimated on norm-robust copies Q of each member's batch and then applied to the raw outputs P, so a
few large-norm (over-confident) trials do not dominate them. Variants: base (Q = P, must equal grader_sim's fp),
unit (Q = unit trials x batch-mean norm), cap (Q = trials with norm capped at the batch median).
    robust_stats_sim.py <recipe> [<recipe> ...]
"""
import json, sys
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from neurips_challenge.grader_sim import chronological, official, DATA, run_dirs, FOLDS
from neurips_challenge.score_track1 import batch_transform_fp, build_modules, build_test_dataset, eeg_fingerprint, fit_fingerprint_lda


def bt_fp_stats(P, Q, F, lda, p=6, c=1.0, a=0.5, k=3, b=0.5):
    mu = Q.mean(0, keepdim=True); C = P - mu; CQ = Q - mu
    p = min(p, (len(P) - 1) // 8)
    if p > 0:
        Fp = ((F - lda['m0']) / lda['sd']) @ lda['W']
        Fp = (Fp - Fp.mean(0)) / (Fp.std(0) + 1e-6)
        Z = torch.linalg.svd(Fp, full_matrices=False)[0][:, :p]
        B = Z.T @ CQ
        C = C - c * Z @ B; CQ = CQ - c * Z @ B
    if k and len(P) > k:
        V = torch.linalg.svd(CQ.float(), full_matrices=False)[2][:k]
        C = C - b * (C @ V.T) @ V
    return C + (1 - a) * mu


def unit(P):
    nm = P.norm(dim=1, keepdim=True); return P / nm * nm.mean()


def cap(P):
    nm = P.norm(dim=1, keepdim=True); return P * (nm.median() / nm).clamp(max=1)


VARIANTS = {'base': lambda P: P, 'unit': unit, 'cap': cap}


@torch.no_grad()
def main(recipes, device=torch.device('cuda:0')):
    from train import run_eeg_backbone
    res = {t: {} for t in VARIANTS}
    for k, subs in FOLDS.items():
        members = []
        for r in recipes:
            d = run_dirs(r)[k]
            cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
            ckpt = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=device, weights_only=False)
            model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [subs[0]]), device, int(cfg.time_window[1]) - int(cfg.time_window[0]))
            members.append((cfg, model, eproj))
        lda = {k_: v.to(device) for k_, v in fit_fingerprint_lda(DATA, [s for s in range(1, 21) if s not in subs] + list(range(21, 31))).items() if k_ != 'bands'}
        stream_x, stream_img, stream_subj, stream_raw, G = [[] for _ in members], [], [], [], None
        for s in subs:
            order = chronological(s)
            raw = np.asarray(np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)[np.sort(order)])
            raw = dict(zip(np.sort(order).tolist(), raw))
            for m, (cfg, _, _) in enumerate(members):
                ds = build_test_dataset(cfg, [s])
                G = ds.image_features[:, 0, :ds.target_dims[0]] if G is None else G
                for i in order:
                    eeg, _, _, _, obj, _, _ = ds[int(i)]
                    stream_x[m].append(torch.as_tensor(eeg))
                    if m == 0:
                        stream_img.append(int(obj)); stream_subj.append(s); stream_raw.append(torch.as_tensor(raw[int(i)]))
        o = np.random.default_rng(k * 100 + 7).permutation(len(stream_img))  # same shuffle as grader_sim --shuffle
        stream_x = [[xs[i] for i in o] for xs in stream_x]; stream_img, stream_subj = np.array(stream_img)[o], np.array(stream_subj)[o]
        stream_raw = [stream_raw[i] for i in o]
        outs = {t: [] for t in VARIANTS}
        for b in range(0, len(stream_img), 64):
            per = []
            for m, (cfg, model, eproj) in enumerate(members):
                X = torch.stack(stream_x[m][b:b + 64]).to(device)
                per.append(eproj(run_eeg_backbone(model, cfg, X, torch.full((len(X),), 10 ** 6, device=device))).float())
            per = [p / p.norm(dim=1).mean() for p in per]
            F = eeg_fingerprint(torch.stack(stream_raw[b:b + 64]).to(device))
            for t, q in VARIANTS.items():
                outs[t].append(sum(bt_fp_stats(p, q(p), F, lda) for p in per).cpu().numpy())
        for t in outs:
            r = official(np.concatenate(outs[t]), stream_img, stream_subj, np.asarray(G))
            res[t].update({s: r[s] for s in subs})
            print(f'fold {k} {t}: pooled {100 * r["pooled"]:.2f}', flush=True)
    for t, v in res.items():
        print(f'RESULT {"+".join(recipes)} {t}: mean over 20 subjects {100 * np.mean(list(v.values())):.2f}')
    for t in [t for t in res if t != 'base']:
        d = np.array([res[t][s] - res['base'][s] for s in res['base']])
        print(f'RESULT {t} - base: {100 * d.mean():+.2f}, better on {(d > 0).sum()}/20, worse on {(d < 0).sum()}')


if __name__ == '__main__':
    P, F = torch.randn(64, 16), torch.randn(64, 656)
    lda = {'m0': torch.zeros(656), 'sd': torch.ones(656), 'W': torch.randn(656, 8)}
    assert torch.allclose(bt_fp_stats(P, P, F, lda), batch_transform_fp(P, F, lda), atol=1e-4)  # Q = P reproduces fp
    main(sys.argv[1:])
