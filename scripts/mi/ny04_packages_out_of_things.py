"""Y04 do the gains of the packages hold when the TARGETS are outside THINGS? Every proxy gallery so far has THINGS targets; a 40k gallery of
THINGS targets among COCO / ImageNet distractors rewards any THINGS-ward bias of the outputs (Y03, obr01), and the sealed gallery is mostly
outside THINGS. The only EEG on non-THINGS images on disk: ds004018 (16 people, another lab, 200 segmented objects, 5 Hz RSVP); control with
THINGS targets from another lab: ds007964 (19 people, 1,900 THINGS images). Packages WITH the alignment step (the cohorts are 4-5 samples
off), 64-trial batches mixing people, r=3 groups (3 trials of a person and image, as the grader). Galleries: the cohort's images; + 40,000
COCO / ImageNet images; + the 14.4k THINGS proxy gallery. Top-5 %, AUC = P(target > a gallery image); change against the FIRST package in %
with a paired bootstrap over groups.
    ny04_packages_out_of_things.py <package dir> [<package dir> ...]"""
import glob, importlib.util, json, os, sys, types
from pathlib import Path
import numpy as np, torch

DEV = torch.device('cuda:0'); R = '/nasbrain/p20fores/Neurobridge_SSL'; S = Path(f'{R}/neurips_challenge/submissions'); nz = lambda x: x / x.norm(dim=1, keepdim=True)
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object; sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']; F = f'{R}/data/things_eeg/image_feature/dinov2-giant_track1'
OPEN = nz(torch.tensor(np.load(f'{F}/open_gallery45k.npy')[:40000]).float()); T = np.load(f'{F}/image_train.npy').reshape(16540, 1536)
THINGS = nz(torch.cat([torch.tensor(T[np.sort(np.random.default_rng(0).choice(16540, 5000, replace=False))]).float(), torch.tensor(np.load(f'{F}/things_extra.npy')).float()]))
for cohort in ('ds004018', 'ds007964'):
    D = f'/nasbrain/p20fores/{cohort}'; G0 = nz(torch.tensor(np.load(f'{D}/targets.npz')['F']).float()); X, img, who = [], [], []
    for p in sorted(glob.glob(f'{D}/export/sub-*.npz')):
        d = np.load(p); X.append(d['X'][:6000].astype(np.float32)); img.append(d['img'][:6000]); who += [p] * len(X[-1])
    X, img = torch.tensor(np.concatenate(X)), np.concatenate(img); o = np.random.default_rng(0).permutation(len(X)); X, img, who = X[o], img[o], np.array(who)[o]
    groups = {}
    for i, kk in enumerate(zip(who.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.concatenate([np.array(v[:len(v) // 3 * 3]).reshape(-1, 3) for v in groups.values() if len(v) >= 3])); t = torch.as_tensor(img)[g3[:, 0]]
    gals = {'own %d' % len(G0): G0, '+ 40k COCO / ImageNet': torch.cat([G0, OPEN]), '+ 14.4k THINGS': torch.cat([G0, THINGS])}
    print(f'{cohort}: {len(set(who.tolist()))} people, {len(X)} trials, {len(g3)} groups of 3', flush=True); H = {}
    for name in sys.argv[1:]:
        c = f'{R}/results/things_eeg/mi/nc/y04_{cohort}_{name}.npy'
        if not os.path.exists(c):
            spec = importlib.util.spec_from_file_location('submission', S / name / 'submission.py'); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
            m = sub.Solver.load_model(None, {'submission_dir': S / name, 'ch_names': CH, 'device': DEV})
            np.save(c, torch.cat([m.predict(X[b:b + 64].to(DEV)).float().cpu() for b in range(0, len(X), 64)]).numpy().astype(np.float16)); del m; torch.cuda.empty_cache()
        q = nz(torch.tensor(np.load(c)).float()[g3].mean(1))
        for gn, G in gals.items():
            r, G = [], G.to(DEV)
            for b in range(0, len(q), 4096):
                Sc = q[b:b + 4096].to(DEV) @ G.T; r.append((Sc > Sc.gather(1, t[b:b + 4096].to(DEV)[:, None])).sum(1).cpu())
            H[(name, gn)] = torch.cat(r).numpy()
    bi = np.random.default_rng(0).integers(0, len(g3), (1000, len(g3))); ref = sys.argv[1]
    for gn, G in gals.items():
        for name in sys.argv[1:]:
            a, b = H[(name, gn)], H[(ref, gn)]; o = '%-22s | %-38s top-5 %6.3f %% (chance %.3f)  AUC %.4f' % (gn, name, 100 * (a < 5).mean(), 500 / len(G), 1 - a.mean() / (len(G) - 1))
            if name != ref:
                for lab, k in (('top-5', 5), ('top-20', 20), ('top-100', 100)):
                    o += ' | %s %+5.1f +- %.1f' % (lab, 100 * ((a < k).mean() / (b < k).mean() - 1), 100 * ((a[bi] < k).mean(1) / (b[bi] < k).mean(1) - 1).std())
                o += ' | ranking error %+5.1f +- %.1f' % (100 * (a.mean() / b.mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std())
            print(o, flush=True)
