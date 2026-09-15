"""Aggregate completed folds only; paired contrasts use identical subjects."""
import csv
import fcntl
import json
import numpy as np
from ensemble_experiments.mechanism_study.common import write_json, load_member, row_z
from .train import OUTPUT
from .models import ARMS


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(p.read_text()) for p in sorted((OUTPUT/'runs').glob('*/seed3300/sub-*/complete.json'))]
        for row in rows:
            # Archive-derived rescue comparison, not a new selection rule.
            subject = row['subject']
            a, _, ka = load_member('atm_iv_valcon', subject)
            b, _, kb = load_member('iv33g_valcon', subject)
            p = OUTPUT/'runs'/row['arm']/'seed3300'/f'sub-{subject:02d}'/'test_scores.npz'
            with np.load(p) as f:
                keys = np.stack([f['object'], f['image_idx']], axis=1)
                np.testing.assert_array_equal(keys, ka); np.testing.assert_array_equal(keys, kb)
                correct = row_z(f['scores']).mean(0).argmax(1) == np.arange(200)
            ref = (row_z(a)+row_z(b)).argmax(1) == np.arange(200)
            solo = (a.argmax(1) == np.arange(200)) | (b.argmax(1) == np.arange(200))
            row['reference_top1'] = float(ref.mean()*100)
            row['delta_reference_pp'] = row['top1']-row['reference_top1']
            row['reference_correct_retained'] = int((correct & ref).sum())
            row['reference_correct_total'] = int(ref.sum())
            row['reference_all_wrong_rescues_retained'] = int((correct & ref & ~solo).sum())
            row['reference_all_wrong_rescues_total'] = int((ref & ~solo).sum())
        write_json(OUTPUT/'folds.json', rows)
        summary = []
        for arm in ARMS:
            rs = [r for r in rows if r['arm'] == arm]
            if not rs: continue
            summary.append(dict(arm=arm, folds=len(rs), **{k:float(np.mean([r[k] for r in rs]))
                for k in ['top1','top5','parameters','training_seconds','validation_seconds',
                          'epochs_completed','selected_epoch','delta_reference_pp']}))
        write_json(OUTPUT/'summary.json', summary)
        if summary:
            tmp = OUTPUT/'summary.tmp.csv'
            with tmp.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
            tmp.replace(OUTPUT/'summary.csv')
        paired = []
        for first, second in [('dual_head','single'),('hybrid','single'),('hybrid','dual_head')]:
            a = {r['subject']:r for r in rows if r['arm']==first}
            b = {r['subject']:r for r in rows if r['arm']==second}
            subjects = sorted(a.keys() & b.keys())
            if subjects:
                delta = [a[s]['top1']-b[s]['top1'] for s in subjects]
                paired.append(dict(contrast=f'{first}-{second}', subjects=subjects,
                    differences_pp=delta, mean_delta_pp=float(np.mean(delta))))
        write_json(OUTPUT/'paired.json', paired)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
