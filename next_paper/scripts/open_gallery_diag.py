"""Why does GALA collapse with distractors (R3) while SCORE degrades gracefully? Diagnosis + injective transport.

Setup = CVPR_paper/scripts/regimes.py R4 ("R3" in the paper): 80-repetition queries of the 200 test concepts, gallery =
the 200 test images + D distractors (one image per training concept, same permutation), cross-fold GALA params.
Methods (top-1 / top-5):
  plain, score (our SCORE reimplementation, label_free.score_deploy)
  gala_bal   published loop, balanced Sinkhorn (columns forced to mass 1, distractors included)
  gala_free  row-softmax plan (no column constraint)
  gala_inj   injective plan: rows = 1, columns <= 1 (KL projection: scale a column down only if it exceeds 1)
  oracle_sup GALA fitted and scored on the true 200 images (= R1)
  oracle_fit GALA fitted on the true 200 images, scored against the full gallery (isolates retrieval competition)
Diagnostics: plan mass on true pairs / on distractors (first and last iteration), distance of the GALA rotation to
the oracle rotation and to the identity (||R - R'||_F / sqrt(2d), 1 = orthogonal), SCORE landmark count / precision.
usage: open_gallery_diag.py --arm atm128_vc --seed 3301 [--subjects ...]
"""
import argparse, csv, os, sys
from types import SimpleNamespace

import numpy as np

R = '/nasbrain/p20fores/Neurobridge_SSL'
for p in (R, f'{R}/scripts/things_eeg/tta_rebuttal', f'{R}/CVPR_paper/scripts'):
    sys.path.insert(0, p)
from shared import encode_repetition_blocks, fit_saw_transform, load_subject_context, score_features  # noqa: E402
from run_partial_coverage import damped_assign, topk_acc  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes  # noqa: E402
from label_free import csls, score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402

D_LIST = [0, 200, 400, 800, 1654]
C = f'{R}/results/things_eeg/cvpr/next_paper/open_gallery'


def capacity_assign(s, tau, iters):
    k = np.exp((s - s.max(1, keepdims=True)) / max(tau, 1e-8))
    for _ in range(max(1, int(iters))):
        k /= k.sum(1, keepdims=True)
        k *= np.minimum(1.0, 1.0 / np.clip(k.sum(0, keepdims=True), 1e-12, None))
    return k


def plan(s, p, mode):
    if mode == 'inj':
        return capacity_assign(s, p.sinkhorn_tau, p.sinkhorn_iters)
    return damped_assign(s, p.sinkhorn_tau, p.sinkhorn_iters, 1.0 if mode == 'bal' else 0.0)


def gala(q, gal, p, mode):
    t, _ = fit_saw_transform(q, p)
    s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    rot, plans = np.eye(t.shape[1], dtype=np.float32), []
    for _ in range(max(1, int(p.soft_procrustes_steps))):
        a = plan(s, p, mode)
        plans.append(a)
        m = fit_soft_assignment_procrustes(t, gal, a, power=p.soft_procrustes_power,
                                           normalize_inputs=p.soft_procrustes_normalize_inputs)
        if m is None:
            break
        t, rot = apply_orthogonal_map(t, m), rot @ m
        s = score_features(t, gal, use_csls=p.use_csls, csls_k=p.csls_k)
    return s, rot, plans, t


def mass(a, n=200):
    tot = a.sum()
    return a[np.arange(n), np.arange(n)].sum() / tot, a[:, n:].sum() / tot


def landmarks(q, g, k=10, eps=1e-6):  # SCORE's mutual-NN landmarks after its moment matching (score_fit, Eq. 3-4)
    qt = (q - q.mean(0)) / (q.std(0) + eps) * (g.std(0) + eps) + g.mean(0)
    s = csls(qt, g, k)
    fwd, bwd = s.argmax(1), s.argmax(0)
    mut = np.flatnonzero(bwd[fwd] == np.arange(len(q)))
    return len(mut), 100 * (fwd[mut] == mut).mean() if len(mut) else np.nan


def embeddings(arm, seed, sid):
    f = f'{C}/cache/{arm}_s{seed}/sub-{sid:02d}.npz'
    if os.path.exists(f):
        z = np.load(f)
        return z['q'], z['g'], z['d']
    rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
    _, ea, ds, mods = load_subject_context(f'{R}/results/things_eeg/cvpr/loso/{arm}/seed{seed}', rt, sid, average=False)
    (q,), g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
    raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy')).astype(np.float32)
    raw = raw[:, 0] if raw.ndim == 4 else raw.reshape(1654, -1, raw.shape[-1])[:, 0]
    d = project_raw_images(ea, mods, raw[np.random.default_rng(sid).permutation(1654)])  # as regimes.py
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, q=q, g=g, d=d)
    return q, g, d


def rdist(a, b):
    return float(np.linalg.norm(a - b) / np.sqrt(2 * a.shape[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    os.makedirs(C, exist_ok=True)
    out = open(f'{C}/{a.arm}_s{a.seed}.csv', 'w', newline='')
    w = csv.writer(out)
    w.writerow(['subject', 'D', 'method', 'metric', 'value'])
    y = np.arange(200)
    for sid in a.subjects:
        p = subject_params(params_csv, sid)
        q, g, dis = embeddings(a.arm, a.seed, sid)
        s_or, r_or, _, t_or = gala(q, g, p, 'bal')
        for D in D_LIST:
            gal = np.concatenate([g, dis[:D]], 0)
            res = {'plain': score_features(q, gal), 'score': score_deploy(q, gal),
                   'oracle_sup': s_or, 'oracle_fit': score_features(t_or, gal, use_csls=p.use_csls, csls_k=p.csls_k)}
            for mode in ('bal', 'free', 'inj'):
                s, rot, plans, _ = gala(q, gal, p, mode)
                res[f'gala_{mode}'] = s
                for tag, pl in (('first', plans[0]), ('last', plans[-1])):
                    tp, dm = mass(pl)
                    w.writerow([sid, D, f'gala_{mode}', f'true_mass_{tag}', tp])
                    w.writerow([sid, D, f'gala_{mode}', f'distractor_mass_{tag}', dm])
                w.writerow([sid, D, f'gala_{mode}', 'rot_to_oracle', rdist(rot, r_or)])
                w.writerow([sid, D, f'gala_{mode}', 'rot_to_identity', rdist(rot, np.eye(len(rot)))])
            n, prec = landmarks(q, gal)
            w.writerow([sid, D, 'score', 'landmarks', n])
            w.writerow([sid, D, 'score', 'landmark_precision', prec])
            for m, s in res.items():
                acc = topk_acc(s, y)
                w.writerow([sid, D, m, 'top1', acc[1]])
                w.writerow([sid, D, m, 'top5', acc[5]])
        w.writerow([sid, 0, 'oracle', 'rot_to_identity', rdist(r_or, np.eye(len(r_or)))])
        out.flush()
        print(sid, 'done', flush=True)
    out.close()


if __name__ == '__main__':
    main()
