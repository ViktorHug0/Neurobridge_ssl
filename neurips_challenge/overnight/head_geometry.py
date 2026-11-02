"""Where does the per-trial output norm come from? Fold-1 models, fold-1 test subject 2 (unseen). With x the input of
the final 1536-d linear (e = W x + b) and delta = x - mean_trials(x):
  log|W delta| = log|delta| (magnitude reaching the head) + log g(delta) (gain of its direction, g = |W u|, u unit)
and the share of var(log|W delta|) carried by each term; rho(|e|, .) for |x|, |delta|, g and, when a LayerNorm feeds
the head, its input magnitude |h| (which the LN discards); the singular-value spread of the effective head
(W diag(gamma) behind an LN) and |mean e| / mean |e|.
    head_geometry.py <recipe> [<recipe> ...]
"""
import json, sys
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from neurips_challenge.score_track1 import build_modules, build_test_dataset
from neurips_challenge.cv4_ensemble import DATA, run_dirs
from train import run_eeg_backbone

dev = torch.device('cuda')


@torch.no_grad()
def main(recipes, subject=2, n=6000):
    for r in recipes:
        d = run_dirs(r)[1]
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        ck = torch.load(f'{d}/checkpoint_sealed_best.pth', map_location=dev, weights_only=False)
        ds = build_test_dataset(cfg, [subject])
        model, eproj, _ = build_modules(cfg, ck, ds, dev, 120)
        lins = [m for m in list(model.modules()) + list(eproj.modules()) if isinstance(m, torch.nn.Linear) and m.out_features == 1536]
        head = lins[-1]
        lns = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm) and m.normalized_shape[0] == head.in_features]
        got = {}
        head.register_forward_hook(lambda m, i, o: got.update(x=i[0].float().mean(1) if i[0].ndim == 3 else i[0].float(),
                                                             e=o.float().mean(1) if o.ndim == 3 else o.float()))  # REVE: head per token, then mean
        if lns:
            lns[-1].register_forward_hook(lambda m, i, o: got.update(h=i[0].float(), z=o.float()))
        idx = np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
        X = torch.as_tensor(np.stack([np.asarray(ds[int(i)][0], np.float32) for i in idx]), device=dev)
        acc = {k: [] for k in ('x', 'e', 'h', 'z')}
        for b in range(0, len(X), 500):
            out = eproj(run_eeg_backbone(model, cfg, X[b:b + 500], torch.full((len(X[b:b + 500]),), 10 ** 6, device=dev))).float()
            for k in acc:
                if k in got:
                    acc[k].append(got[k].cpu())
            assert torch.allclose(got['e'], out, atol=1e-3), 'the hooked linear is not the output layer'
        x, e = torch.cat(acc['x']).double(), torch.cat(acc['e']).double()
        z_ = torch.cat(acc['z']).double() if acc['z'] else None
        ln_feeds_head = z_ is not None and z_.shape == x.shape and torch.allclose(z_, x, atol=1e-4)
        W = head.weight.double().cpu()
        A = W * lns[-1].weight.double().cpu() if ln_feeds_head else W  # effective map of the normalised code
        s = torch.linalg.svdvals(A)
        delta = x - x.mean(0)
        Wd = delta @ W.T
        mag, gain = delta.norm(dim=1), Wd.norm(dim=1) / delta.norm(dim=1)
        lt, lm, lg = Wd.norm(dim=1).log(), mag.log(), gain.log()
        var = lt.var()
        en = e.norm(dim=1)
        rho = lambda a: round(float(spearmanr(en, a).correlation), 2)
        print(f'RESULT {r}: head {head.in_features}->1536{" behind LN" if ln_feeds_head else " (no norm before it)"} | '
              f'norm CV {float(en.std() / en.mean()):.2f} | |mean e|/mean|e| {float(e.mean(0).norm() / en.mean()):.2f} | '
              f'sv spread s1/s_med {float(s[0] / s.median()):.1f}, s_med/s_min {float(s.median() / s[-1]):.1f} | '
              f'var(log|W delta|) share: magnitude {float(lm.var() / var):.2f}, direction gain {float(lg.var() / var):.2f}, '
              f'2cov {float(2 * torch.cov(torch.stack([lm, lg]))[0, 1] / var):.2f} | '
              f'rho(|e|, |x|) {rho(x.norm(dim=1))}, rho(|e|, gain) {rho(gain)}, rho(|e|, |delta|) {rho(mag)}'
              + (f', rho(|e|, pre-LN |h|) {rho(torch.cat(acc["h"]).double().norm(dim=1))}' if ln_feeds_head else ''), flush=True)
        del model, eproj; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1:])
