"""Fixed comparisons against the completed compact baseline and 40.20% pair."""
import csv
import fcntl
import json
import numpy as np
from ensemble_experiments.compact_valcon.train import OUTPUT as PREVIOUS
from ensemble_experiments.mechanism_study.common import write_json, load_member, row_z
from .train import OUTPUT
from .models import ARMS


def main():
    lock=(OUTPUT/'summary.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[]
    for path in sorted((OUTPUT/'runs').glob('*/seed3300/sub-*/complete.json')):
        r=json.loads(path.read_text());s=r['subject']
        baseline=json.loads((PREVIOUS/'runs/dual_head/seed3300'/f'sub-{s:02d}'/'complete.json').read_text())
        atm,_,keys=load_member('atm_iv_valcon',s)
        with np.load(PREVIOUS/'runs/single/seed3300'/f'sub-{s:02d}'/'test_scores.npz') as f:ts=f['scores'][0]
        with np.load(path.parent/'test_scores.npz') as f:
            np.testing.assert_array_equal(keys,np.stack([f['object'],f['image_idx']],axis=1))
            correct=row_z(f['scores']).mean(0).argmax(1)==np.arange(200)
        teacher=(row_z(atm)+row_z(ts)).argmax(1)==np.arange(200)
        r.update(baseline_top1=baseline['top1'],delta_baseline_pp=r['top1']-baseline['top1'],
                 reference_top1=float(teacher.mean()*100),delta_reference_pp=r['top1']-float(teacher.mean()*100),
                 reference_correct_retained=int((correct&teacher).sum()),reference_correct_total=int(teacher.sum()))
        rows.append(r)
    write_json(OUTPUT/'folds.json',rows);summary=[]
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm]
        if not rs:continue
        summary.append(dict(arm=arm,folds=len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in
            ['top1','top5','parameters','training_seconds','teacher_seconds','validation_seconds',
             'epochs_completed','selected_epoch','delta_baseline_pp','delta_reference_pp']}))
    write_json(OUTPUT/'summary.json',summary)
    if summary:
        tmp=OUTPUT/'summary.tmp.csv'
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        tmp.replace(OUTPUT/'summary.csv')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
