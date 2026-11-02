"""E4.1 streaming regime (R2) on one model: menu size K x repetitions per query, transductive on the buffer.

Same regime as the rebuttal's acausal_menu_grid (run_streaming_refit.py --acausal --csls --geodesic): a fixed K-item
menu (random per subject), queries are q-repetition averages (80/q disjoint blocks per concept), they arrive one at a
time in random order, and at 25 checkpoints SAGE is re-fitted on the buffer and the buffer is scored (CSLS).
Two changes for the CVPR protocol (checklist D15/D17):
  * calibration hyper-parameters per subject come from --params_csv (the loso_fold_cv rows of
    run_honest_tta_selection.py, i.e. chosen on the OTHER folds), not the old test-tuned defaults;
  * a prequential score: the trials that arrived since the previous checkpoint, decoded by the fit that first
    included them ("<arm>@new"), i.e. what the user would have seen at arrival time;
  * a strict-causal score ("<arm>@prev"): the same trials decoded by the map frozen at the PREVIOUS checkpoint (fitted
    before they arrived; CSLS hub term from that fit's buffer only), plain cosine for the first chunk.
Each subject is encoded once per q; every K reuses it.
usage: stream_grid.py --source_run_dir R --params_csv C --output_dir O [--subjects ...] [--menus ...] [--reps ...]
"""
import argparse, csv, os, sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
from shared import TTAParams, apply_saw_transform, cosine_scores, encode_repetition_blocks, load_subject_context, normalize_rows  # noqa: E402
from module.util import apply_orthogonal_map  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from run_streaming_refit import build_pool, fit_calibration_rho, geodesic_blend, schur_form  # noqa: E402
from label_free import csls_ref, score_fit  # noqa: E402


def subject_params(params_csv, sid):
    rows = [r for r in csv.DictReader(open(params_csv)) if r['protocol'] == 'loso_fold_cv' and int(r['subject_id']) == sid]
    assert len(rows) == 1, (params_csv, sid)
    r = rows[0]
    return replace(TTAParams(), saw_shrink=float(r['saw_shrink']), sinkhorn_tau=float(r['sinkhorn_tau']),
                   soft_procrustes_steps=int(r['soft_procrustes_steps']), soft_procrustes_power=float(r['soft_procrustes_power']))


def frozen_scorer(mapf, fit_x, gallery, k, a=1.0):
    """Scores of any queries under a map fitted on fit_x: (1 - a) cos + a CSLS, the CSLS gallery-side hub term taken
    from the mapped fit buffer only. On fit_x itself this is the usual transductive score."""
    ref = mapf(fit_x)
    def f(x):
        z = mapf(x)
        c = cosine_scores(z, gallery)
        return c if a == 0 else (1 - a) * c + a * csls_ref(z, ref, gallery, k)
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_run_dir', required=True)
    p.add_argument('--params_csv', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--menus', nargs='+', type=int, default=[25, 50, 100, 150, 200])
    p.add_argument('--reps', nargs='+', type=int, default=[5, 10, 20, 40])
    p.add_argument('--alphas', nargs='*', type=float, default=[0.5, 1.0])
    p.add_argument('--ramps', nargs='*', default=[], help='E4.2 damping arms cap:const, alpha = cap*u/(u+const), u = t/K')
    p.add_argument('--score_only', action='store_true', help='arms plain + SCORE (deployment step re-fitted on the buffer) only')
    p.add_argument('--fullramps', nargs='*', default=[], help='same alpha, but it also damps SAW whitening (feature blend) '
                   'and CSLS (score blend (1-a) cos + a CSLS): a = 0 is plain, a = 1 is SAGE')
    p.add_argument('--support', action='store_true', help='E4.7: + full1_3s / full1_3sf = full1_3 fitted on the SCORE '
                   'top-1 support of the buffer only (balanced / free marginal), scored on the full menu')
    p.add_argument('--n_checkpoints', type=int, default=25)
    p.add_argument('--stream_seeds', type=int, default=3)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'stream_rows.csv')
    done = set()
    if os.path.exists(out):  # resume: subjects already written are skipped
        done = {int(r['subject']) for r in csv.DictReader(open(out))}
    f = open(out, 'a', newline='')
    w = csv.writer(f)
    if not done:
        w.writerow(['subject', 'menu', 'reps', 'seed', 't', 'coverage_U', 'method', 'top1', 'top5'])
    for sid in args.subjects:
        if sid in done:
            continue
        params = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, eval_args, dataset, modules = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        rows = []
        for q in args.reps:
            blocks, gallery_full, _ = encode_repetition_blocks(eval_args, modules, dataset, block_sizes=[q] * (80 // q), seed=1000 + sid)
            pool_full, labels_full = build_pool(blocks)
            for K in args.menus:
                keep = np.sort(np.random.default_rng(500 + sid).choice(len(gallery_full), K, replace=False))
                remap = np.full(len(gallery_full), -1, dtype=np.int64)
                remap[keep] = np.arange(K)
                mask = np.isin(labels_full, keep)
                pool, labels, gallery = pool_full[mask], remap[labels_full[mask]], gallery_full[keep]
                every = max(1, len(pool) // args.n_checkpoints)
                checkpoints = [every * (i + 1) for i in range(args.n_checkpoints) if every * (i + 1) <= len(pool)]
                for seed in range(args.stream_seeds):
                    order = np.random.default_rng(10_000 * sid + seed).permutation(len(pool))
                    prev, frozen = 0, {}
                    for t in checkpoints:
                        buf = order[:t]
                        new = slice(prev, t)  # positions in buf of the trials that arrived since the last fit
                        prev = t
                        y = labels[buf]
                        fits = {'plain': lambda x: cosine_scores(x, gallery)}
                        if args.score_only:
                            fits['score'] = frozen_scorer(score_fit(pool[buf], gallery), pool[buf], gallery, 10)
                        else:
                            cal = fit_calibration_rho(pool[buf], gallery, params, rho=1.0)
                            form = schur_form(cal['orthogonal_map'])
                            saw = lambda x, st=cal['saw_stats']: apply_saw_transform(x, st)
                            arms = [(f'sage_a{a:g}', a) for a in args.alphas]
                            for r in args.ramps:  # label-free: the strength depends on the buffer-to-menu ratio only
                                cap, const = map(float, r.split(':'))
                                arms.append((f'ramp{cap:g}_{const:g}', cap * (t / K) / (t / K + const)))
                            for name, a in arms:
                                m = lambda x, R=geodesic_blend(form, a): apply_orthogonal_map(saw(x), R)
                                fits[name] = frozen_scorer(m, pool[buf], gallery, params.csls_k)
                            for r in args.fullramps:
                                cap, const = map(float, r.split(':'))
                                a = cap * (t / K) / (t / K + const)
                                m = lambda x, a=a, R=geodesic_blend(form, a): apply_orthogonal_map(
                                    (1 - a) * normalize_rows(x) + a * normalize_rows(saw(x)), R)
                                fits[f'full{cap:g}_{const:g}'] = frozen_scorer(m, pool[buf], gallery, params.csls_k, a)
                            if args.support:  # E4.7: SAGE fitted only on the menu items SCORE's map puts first for some query
                                sup = np.unique(frozen_scorer(score_fit(pool[buf], gallery), pool[buf], gallery, 10)(pool[buf]).argmax(1))
                                a = (t / K) / (t / K + 3)
                                for tag, rho in (('s', 1.0), ('sf', 0.0)):
                                    cs = fit_calibration_rho(pool[buf], gallery[sup], params, rho=rho)
                                    m = lambda x, a=a, st=cs['saw_stats'], R=geodesic_blend(schur_form(cs['orthogonal_map']), a): apply_orthogonal_map(
                                        (1 - a) * normalize_rows(x) + a * normalize_rows(apply_saw_transform(x, st)), R)
                                    fits[f'full1_3{tag}'] = frozen_scorer(m, pool[buf], gallery, params.csls_k, a)
                        cov = len(np.unique(y))
                        x_new = pool[buf[new]]
                        for name, fit in fits.items():
                            sc = fit(pool[buf])
                            acc = topk_acc(sc, y)
                            rows.append((sid, K, q, seed, t, cov, name, acc[1], acc[5]))
                            acc = topk_acc(sc[new], y[new])
                            rows.append((sid, K, q, seed, t, cov, f'{name}@new', acc[1], acc[5]))
                            acc = topk_acc(frozen.get(name, fits['plain'])(x_new), y[new])
                            rows.append((sid, K, q, seed, t, cov, f'{name}@prev', acc[1], acc[5]))
                        frozen = fits
            print(f'subject {sid:02d} reps {q} done', flush=True)
        w.writerows(rows)
        f.flush()
    f.close()


if __name__ == '__main__':
    main()
