"""Frozen TSConv-IV33/ATM-IV28 fusion; source-only alignment and selection."""
import argparse
import csv
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ensemble_experiments.compact_followup.teachers import Teachers
from ensemble_experiments.compact_valcon.train import OUTPUT as PREVIOUS, dataset, split_indices
from ensemble_experiments.mechanism_study.common import DUMPS, REPO, unit, row_z, write_json

OUTPUT = REPO / 'results/things_eeg/spherical_fusion_20260909'
WEIGHTS = [0., .25, .5, .75, 1.]


def procrustes(a, b):
    """Map row vectors B to A, without centering or changing inner products."""
    u, singular, vt = np.linalg.svd(b.astype(np.float64).T @ a.astype(np.float64))
    return u @ vt, singular


def slerp(a, b, t):
    if t == 0: return a
    if t == 1: return b
    cosine = np.clip((a*b).sum(-1, keepdims=True), -1., 1.)
    if np.any(cosine < -1+1e-7):
        raise ValueError('Near-antipodal interpolation requires an explicit convention')
    angle = np.arccos(cosine)
    denom = np.sin(angle).clip(1e-12)
    curved = np.sin((1-t)*angle)/denom*a + np.sin(t*angle)/denom*b
    return unit(np.where(cosine > 1-1e-7, (1-t)*a+t*b, curved))


def scores(q, v, rotation):
    a, b = q
    x, y = v
    b, y = b @ rotation, y @ rotation
    aa, bb, ab, ba = a@x.T, b@y.T, a@y.T, b@x.T
    out = {'solo_ts': aa, 'solo_atm': bb, 'cross_ts_atm': ab,
           'cross_atm_ts': ba, 'cross_terms_only': (ab+ba)/2,
           'four_terms': (aa+bb+ab+ba)/4}
    for t in WEIGHTS:
        out[f'raw_{t:g}'] = (1-t)*aa+t*bb
        out[f'rowz_{t:g}'] = (1-t)*row_z(aa)+t*row_z(bb)
        out[f'slerp_{t:g}'] = slerp(a,b,t) @ slerp(x,y,t).T
        out[f'nlerp_{t:g}'] = unit((1-t)*a+t*b) @ unit((1-t)*x+t*y).T
    return out


def accuracy(score, target):
    return float(np.mean(score.argmax(-1) == target)*100)


@torch.inference_mode()
def extract(teacher, data, indices):
    es = [[], []]
    keys, subjects = [], []
    for batch in DataLoader(Subset(data, indices), batch_size=200, shuffle=False):
        features = teacher(batch[0].cuda(), batch[1].cuda(), batch[3].cuda())
        for m, (e, _) in enumerate(features): es[m].append(unit(e.cpu().numpy()))
        keys.extend(zip(batch[4].tolist(), batch[5].tolist()))
        subjects.extend(batch[3].tolist())
    return np.stack([np.concatenate(e) for e in es]), np.array(keys), np.array(subjects)


@torch.inference_mode()
def project_images(teacher, raw):
    parts = [[], []]
    for start in range(0, len(raw), 512):
        x = torch.from_numpy(raw[start:start+512]).cuda()
        parts[0].append(unit(teacher.ts.image_heads[0](x[:,:3200]).cpu().numpy()))
        parts[1].append(unit(teacher.image_head(x[:,3200:]).cpu().numpy()))
    return np.stack([np.concatenate(p) for p in parts])


def validation(q, v, query_ids):
    # Every validation image is a query once per source subject. Final panel is
    # padded with earlier candidates, not additional queries, to retain 200-way.
    permutation = np.random.default_rng(3300).permutation(v.shape[1])
    for start in range(0, len(permutation), 200):
        active = permutation[start:start+200]
        gallery = np.concatenate([active, permutation[:200-len(active)]])
        rows = np.flatnonzero(np.isin(query_ids, active))
        targets = np.array([{int(k): j for j,k in enumerate(gallery)}[int(k)]
                            for k in query_ids[rows]])
        yield q[:,rows], v[:,gallery], targets


def run(subject):
    out = OUTPUT / f'sub-{subject:02d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists(): return
    tick = time.monotonic()
    split = json.loads((PREVIOUS/f'runs/single/seed3300/sub-{subject:02d}/split.json').read_text())
    teacher = Teachers(subject, split).cuda().eval()
    data = dataset(split['training_subjects'], True)
    tr, va, held = split_indices(data)
    assert held == split['held_concepts'] and len(tr) == split['train_items']
    nobjects, nimages, dim = data.image_features.shape
    allkeys = np.array([(o,i) for o in range(nobjects) for i in range(nimages)])
    v = project_images(teacher, data.image_features.reshape(-1, dim)).astype(np.float64)
    fit = ~np.isin(allkeys[:,0], held)
    rotation, singular = procrustes(v[0,fit], v[1,fit])
    np.testing.assert_allclose(rotation.T@rotation, np.eye(128), atol=1e-10)
    valkeys = allkeys[~fit]
    lookup = {tuple(k):i for i,k in enumerate(valkeys)}
    print('EXTRACT_VALIDATION', subject, len(va), flush=True)
    q, keys, subjects = extract(teacher, data, va)
    query_ids = np.array([lookup[tuple(k)] for k in keys])
    assert len(set(zip(subjects, query_ids))) == len(va)
    np.savez_compressed(out/'geometry.npz', rotation=rotation, singular_values=singular,
                        validation_eeg=q, validation_images=v[:,~fit],
                        query_ids=query_ids, subjects=subjects, image_keys=valkeys)
    counts = {}
    for qp, vp, target in validation(q.astype(np.float64), v[:,~fit], query_ids):
        for name, score in scores(qp, vp, rotation).items():
            counts[name] = counts.get(name, 0) + int((score.argmax(-1)==target).sum())
    valacc = {k:100*n/len(va) for k,n in counts.items()}
    # Resolve ties toward 50/50, then lower ATM weight. Lock before test loading.
    selected = {family: min(WEIGHTS, key=lambda t: (-valacc[f'{family}_{t:g}'], abs(t-.5), t))
                for family in ['raw', 'rowz', 'slerp', 'nlerp']}
    write_json(out/'selection.json', dict(weights=selected, validation_top1=valacc,
               fit_images=int(fit.sum()), validation_queries=len(va),
               image_alignment_cosine_train=float((v[0,fit]*(v[1,fit]@rotation)).sum(-1).mean()),
               image_alignment_cosine_validation=float((v[0,~fit]*(v[1,~fit]@rotation)).sum(-1).mean()),
               provenance=teacher.provenance))
    del data, teacher
    torch.cuda.empty_cache()
    with np.load(PREVIOUS/f'runs/single/seed3300/sub-{subject:02d}/test_scores.npz') as f:
        tq, tv = [unit(f[k].astype(np.float64)) for k in ['eeg_0','image_0']]
        testkeys = np.stack([f['object'], f['image_idx']], axis=1)
    with np.load(DUMPS/f'atm_iv_valcon-sub{subject:02d}.npz') as f:
        aq, av = [unit(f[k].astype(np.float64)) for k in ['eeg','image']]
        np.testing.assert_array_equal(testkeys, np.stack([f['object'],f['image_idx']],axis=1))
    test = scores(np.stack([tq,aq]), np.stack([tv,av]), rotation)
    names = ['solo_ts','solo_atm','cross_ts_atm','cross_atm_ts','cross_terms_only',
             'four_terms','raw_0.5','rowz_0.5','slerp_0.5','nlerp_0.5']
    reported = {k:test[k] for k in names}
    reported.update({f'{k}_selected':test[f'{k}_{t:g}'] for k,t in selected.items()})
    np.testing.assert_allclose(test['slerp_0.5'],test['nlerp_0.5'],atol=1e-12)
    np.savez_compressed(out/'test_scores.npz', **reported, keys=testkeys)
    result = dict(subject=subject, seconds=time.monotonic()-tick, selected_weights=selected,
                  top1={k:accuracy(s,np.arange(200)) for k,s in reported.items()})
    write_json(out/'complete.json', result)
    print('COMPLETE', json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(1,11)))
    args = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(source_sha256=digest, weights=WEIGHTS, alignment='uncentered orthogonal B->A',
                    selection='source concept validation top1, fixed 200-way galleries', seed=3300)
    if (OUTPUT/'manifest.json').exists():
        assert json.loads((OUTPUT/'manifest.json').read_text()) == manifest
    write_json(OUTPUT/'manifest.json', manifest)
    torch.set_num_threads(4)
    for subject in args.subjects:
        assert 1 <= subject <= 10
        run(subject)
        completed = [json.loads(f.read_text()) for f in sorted(OUTPUT.glob('sub-*/complete.json'))]
        rows = [dict(method=k, folds=len(completed),
                     mean_top1=float(np.mean([r['top1'][k] for r in completed])))
                for k in completed[0]['top1']]
        with (OUTPUT/'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print('ALL_REQUESTED_FOLDS_COMPLETE', flush=True)


if __name__ == '__main__': main()
