"""What does the EEG output norm encode? Fold-1 models on the fold-1 test subjects (unseen), every test trial.
Per model: norm spread (CV within subject), AUC of norm for a correct single-trial top-5 (200-way), Spearman with
the input trial RMS, and the sealed subject-aggregated top-5 when repeats are averaged raw (the grader's way:
norm-weighted), after per-trial L2 normalisation (equal weights), or with norm^2 weights.
    norm_probe.py <recipe> [<recipe> ...]
"""
import json, sys
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from neurips_challenge.score_track1 import build_modules, build_test_dataset
from neurips_challenge.cv4_ensemble import DATA, run_dirs
from neurips_challenge.trial_routing import FOLDS
from train import run_eeg_backbone

dev = torch.device('cuda')
n = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def agg_top5(p, y, G, w):
    P = np.stack([(w[y == i, None] * p[y == i]).sum(0) for i in range(200)])
    s = n(P) @ n(G).T
    return float(np.mean([(s[i] > s[i, i]).sum() < 5 for i in range(200)]))


@torch.no_grad()
def main(recipes):
    for r in recipes:
        d = run_dirs(r)[1]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ck = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=dev, weights_only=False)
        model, eproj, _ = build_modules(cfg, ck, build_test_dataset(cfg, [FOLDS[1][0]]), dev, 120)
        out = {k: [] for k in ('cv', 'auc', 'rho_rms', 'raw', 'unit', 'sq')}
        for s in FOLDS[1]:
            ds = build_test_dataset(cfg, [s])
            G = ds.image_features[:, 0, :ds.target_dims[0]]
            X, y = [], []
            for i in range(len(ds)):
                eeg, _, _, _, obj, _, _ = ds[i]; X.append(np.asarray(eeg, np.float32)); y.append(int(obj))
            X, y = np.stack(X), np.array(y)
            p = np.concatenate([eproj(run_eeg_backbone(model, cfg, torch.as_tensor(X[b:b + 512], device=dev),
                                                        torch.full((len(X[b:b + 512]),), 10 ** 6, device=dev))).float().cpu().numpy()
                                for b in range(0, len(X), 512)])
            nm = np.linalg.norm(p, axis=1)
            sc = n(p) @ n(G).T
            correct = np.array([(sc[t] > sc[t, y[t]]).sum() < 5 for t in range(len(y))])
            out['cv'].append(nm.std() / nm.mean())
            out['auc'].append(roc_auc_score(correct, nm))
            out['rho_rms'].append(spearmanr(nm, np.sqrt((X ** 2).mean((1, 2)))).correlation)
            out['raw'].append(agg_top5(p, y, G, np.ones(len(y))))
            out['unit'].append(agg_top5(p, y, G, 1 / nm))
            out['sq'].append(agg_top5(p, y, G, nm))
        m = {k: round(float(np.mean(v)) * (100 if k in ('raw', 'unit', 'sq') else 1), 3) for k, v in out.items()}
        print(f'RESULT {r}: norm CV {m["cv"]} | AUC(norm -> correct top-5) {m["auc"]} | rho(norm, input RMS) {m["rho_rms"]} | '
              f'subject-agg top-5: raw mean {m["raw"]}, per-trial unit {m["unit"]}, norm^2-weighted {m["sq"]}', flush=True)
        del model, eproj; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1:])
