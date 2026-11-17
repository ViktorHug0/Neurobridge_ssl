"""R08 the packaged submissions on cohorts recorded with another timing (ds004018 about +5 samples, ds007964 about +4/+5
against Alljoined-1.6M; neither was used to train these packages). The package without alignment (sub16) against the same
members behind the alignment step (sub19): 64-trial batches mixing the cohort's people, gallery = the cohort's images.
Single trial and 3 trials of a (person, image) averaged as the grader does: top-5 % and P(target > random image).
    nr08_package_on_cohorts.py <package dir> [<package dir> ...]"""
import glob, importlib.util, json, sys, types
from pathlib import Path
import numpy as np, torch

DEV = torch.device('cuda:0'); S = Path('/nasbrain/p20fores/Neurobridge_SSL/neurips_challenge/submissions'); nz = lambda x: x / x.norm(dim=1, keepdim=True)
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object; sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']
for cohort in ('ds004018', 'ds007964'):
    D = f'/nasbrain/p20fores/{cohort}'; G = nz(torch.tensor(np.load(f'{D}/targets.npz')['F'])).to(DEV); X, img, who = [], [], []
    for p in sorted(glob.glob(f'{D}/export/sub-*.npz'))[:8]:
        d = np.load(p); X.append(d['X'][:4000].astype(np.float32)); img.append(d['img'][:4000]); who += [p] * len(X[-1])
    X, img = torch.tensor(np.concatenate(X)), np.concatenate(img); o = np.random.default_rng(0).permutation(len(X)); X, img, who = X[o], img[o], np.array(who)[o]
    groups = {}
    for i, kk in enumerate(zip(who.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 3])); t1 = torch.as_tensor(img); t3 = t1[g3[:, 0]]
    for name in sys.argv[1:]:
        spec = importlib.util.spec_from_file_location('submission', S / name / 'submission.py'); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
        m = sub.Solver.load_model(None, {'submission_dir': S / name, 'ch_names': CH, 'device': DEV})
        y = torch.cat([m.predict(X[b:b + 64].to(DEV)).float().cpu() for b in range(0, len(X), 64)])
        def read(q, t):
            Sc = nz(q).to(DEV) @ G.T; tg = Sc.gather(1, t.to(DEV)[:, None]); return 100 * ((Sc > tg).sum(1) < 5).float().mean().item(), (Sc < tg).float().mean().item() * len(G) / (len(G) - 1)
        print('%s (%d trials, gallery %d, chance top-5 %.2f %%) | %-34s single trial %.2f %% / %.3f | 3 trials averaged %.2f %% / %.3f' % (cohort, len(X), len(G), 500 / len(G), name, *read(y, t1), *read(y[g3].mean(1), t3)), flush=True)
        del m; torch.cuda.empty_cache()
