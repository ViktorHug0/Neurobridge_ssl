"""Zero-shot transfer of the cv4 Track-1 models to Alljoined-1 (export_alljoined1.py): 8 unseen subjects, 960 NSD
images, BioSemi, 300 ms stimuli at SOA ~0.8 s. Per subject, predictions are averaged over each image's repeats
(the sealed aggregation), then retrieved among the 960 images' DINOv2 track-1 targets (top-5 chance 0.52%).
Views: full epoch, and 'early' = only -0.2..0.35 s kept (rest = the subject's mean epoch), the span where the two
paradigms should still look alike. Also a model-free check: ridge EEG(repeat-mean) -> DINOv2 on 800 images of 7
subjects, retrieval of the 160 held-out images for the 8th (chance top-5 3.1%).
    zeroshot_alljoined1.py <recipe> [<recipe> ...]
"""
import json, os, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.score_track1 import build_modules, build_test_dataset  # noqa: E402
from neurips_challenge.cv4_ensemble import DATA, run_dirs  # noqa: E402
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

D = '/nasbrain/p20fores/alljoined/alljoined1_120Hz'
SUBS = range(1, 9)
dev = torch.device('cuda')
n = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def targets(ids):
    f = f'{D}/dinov2_track1.npz'
    if not os.path.exists(f):
        from transformers import AutoModel
        from neurips_challenge.build_dinov2_multilayer import run
        m = AutoModel.from_pretrained('facebook/dinov2-giant', torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda().eval()
        u = sorted(set(ids))
        F = run(m, [f'/nasbrain/p20fores/alljoined/nm000133/stimuli/nsd/{i}.png' for i in u], states=[26])[:, 0].numpy()
        np.savez(f, ids=np.array(u), F=F); del m; torch.cuda.empty_cache()
    z = np.load(f)
    return z['ids'], z['F']


def topk(P, G, y, k):  # P (n, d) image-mean predictions, G (m, d) gallery, y (n,) gallery rows
    s = n(P) @ n(G).T
    return float(np.mean([(s[i] > s[i, y[i]]).sum() < k for i in range(len(y))]))


def ridge_check(data, ids, G):
    rng = np.random.default_rng(0); test_img = set(rng.choice(ids, 160, replace=False).tolist())
    feats = {s: {i: data[s][0][data[s][1] == i].astype(np.float32).mean(0).ravel() for i in ids} for s in SUBS}
    row = {i: r for r, i in enumerate(ids)}
    res = []
    for s in SUBS:
        tr = [(feats[t][i], G[row[i]]) for t in SUBS if t != s for i in ids if i not in test_img]
        X = np.stack([a for a, _ in tr]); Y = np.stack([b for _, b in tr]); mu = X.mean(0); sd = X.std(0) + 1e-6
        Xn = (X - mu) / sd; W = np.linalg.solve(Xn.T @ Xn + 1e4 * np.eye(Xn.shape[1]), Xn.T @ (Y - Y.mean(0)))
        te = sorted(test_img); P = np.stack([(feats[s][i] - mu) / sd for i in te]) @ W
        res.append(topk(P, G[[row[i] for i in te]], np.arange(len(te)), 5))
    print(f'ridge (model-free, 160-way held-out images): top-5 {100 * np.mean(res):.2f} (chance 3.12) per subj {np.round(100 * np.array(res), 1).tolist()}', flush=True)


@torch.no_grad()
def main(recipes):
    data = {s: (lambda z: (z['X'], z['nsd']))(np.load(f'{D}/sub-{s:02d}.npz')) for s in SUBS}
    ids, G = targets(np.concatenate([data[s][1] for s in SUBS]))
    row = {i: r for r, i in enumerate(ids)}
    ridge_check(data, ids, G)
    from train import run_eeg_backbone
    agg = {}  # (recipe or 'ens', view) -> list of per (fold, subject) top-5
    for k in FOLDS:
        ens = {}
        for r in recipes:
            d = run_dirs(r)[k]
            cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
            ck = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=dev, weights_only=False)
            model, eproj, _ = build_modules(cfg, ck, build_test_dataset(cfg, [FOLDS[k][0]]), dev, 120)
            for s in SUBS:
                X, y = data[s]
                tpl = torch.as_tensor(X[:3000].astype(np.float32).mean(0), device=dev)
                for view in ('full', 'early'):
                    out = []
                    for b in range(0, len(X), 256):
                        x = torch.as_tensor(X[b:b + 256].astype(np.float32), device=dev)
                        if view == 'early':
                            x[..., 66:] = tpl[..., 66:]
                        out.append(eproj(run_eeg_backbone(model, cfg, x, torch.full((len(x),), 10 ** 6, device=dev))).float().cpu().numpy())
                    p = np.concatenate(out)
                    u = np.unique(y); P = np.stack([p[y == i].mean(0) for i in u])
                    gy = np.array([row[i] for i in u])
                    agg.setdefault((r, view), []).append(topk(P, G, gy, 5))
                    e = ens.setdefault((s, view), [0, gy]); e[0] = e[0] + P / np.linalg.norm(p, axis=1).mean()
            del model, eproj; torch.cuda.empty_cache()
            print(f'fold {k} {r}: full {100 * np.mean(agg[(r, "full")]):.2f} early {100 * np.mean(agg[(r, "early")]):.2f} (running mean)', flush=True)
        for (s, view), (P, gy) in ens.items():
            agg.setdefault(('ens', view), []).append(topk(P, G, gy, 5))
    for (r, view), v in sorted(agg.items()):
        print(f'RESULT {r:28s} {view:5s} top-5 (960-way, chance 0.52) {100 * np.mean(v):.2f} +- {100 * np.std(v) / np.sqrt(len(v)):.2f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
