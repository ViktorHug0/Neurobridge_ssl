"""Read-only score diagnostics; hybrid comparisons are post-hoc, not model selection."""
import json
from pathlib import Path
import numpy as np
from ensemble_experiments.mechanism_study.common import REPO, load_member, row_z, margin_metrics, write_json


def describe(scores):
    z = row_z(np.asarray(scores, dtype=np.float64))
    truth = np.arange(200)
    predictions = z.argmax(-1)
    correct = predictions == truth
    fused = z.mean(0).argmax(-1) == truth
    both_wrong = ~correct.any(0)
    wrong = z.copy()
    wrong[:, truth, truth] = -np.inf
    margins = z[:,truth,truth] - wrong.max(-1)
    out = margin_metrics(scores)
    out.update(solo_top1=(100*correct.mean(1)).tolist(),
        both_correct=int(correct.all(0).sum()), only_a=int((correct[0]&~correct[1]).sum()),
        only_b=int((correct[1]&~correct[0]).sum()), both_wrong=int(both_wrong.sum()),
        lost_available=int((correct.any(0)&~fused).sum()),
        same_wrong=int(((predictions[0]==predictions[1])&both_wrong).sum()),
        score_correlation=float(np.corrcoef(z.reshape(2,-1))[0,1]),
        margin_correlation=float(np.corrcoef(margins)[0,1]))
    assert int(fused.sum()) == 200-out['both_wrong']-out['lost_available']+out['all_wrong_rescues']
    return out, fused


def main():
    rows=[]
    for subject in range(1,6):
        atm,_,keys=load_member('atm_iv_valcon',subject)
        path=REPO/f'results/things_eeg/tsconv_bb128_valcon_20260911/runs/single/seed3300/sub-{subject:02d}/test_scores.npz'
        with np.load(path) as f:
            np.testing.assert_array_equal(keys,np.stack([f['object'],f['image_idx']],1))
            reference=np.stack([f['scores'][0],atm])
        stats, ref_correct=describe(reference)
        rows.append(dict(arm='Reference',subject=subject,**stats))
        for arm in ['G2','C0','C1','C2']:
            root='fusion_three_20260915' if arm=='G2' else 'fusion_c012_20260915'
            folder=REPO/f'results/things_eeg/{root}/runs/{arm}/sub-{subject:02d}'
            with np.load(folder/'test_scores.npz') as f:
                np.testing.assert_array_equal(keys,np.stack([f['object'],f['image_idx']],1))
                scores=f['scores'].copy()
            stats,correct=describe(scores)
            saved=json.loads((folder/'complete.json').read_text())
            np.testing.assert_allclose(stats['top1'],saved['top1'],atol=1e-8)
            stats.update(lost_vs_reference=int((ref_correct&~correct).sum()),
                gained_vs_reference=int((correct&~ref_correct).sum()),
                hybrid_ref_ts_new_b=describe(np.stack([reference[0],scores[1]]))[0]['top1'],
                hybrid_new_a_ref_atm=describe(np.stack([scores[0],reference[1]]))[0]['top1'])
            rows.append(dict(arm=arm,subject=subject,**stats))
    means={}
    for arm in ['Reference','G2','C0','C1','C2']:
        group=[r for r in rows if r['arm']==arm]
        means[arm]={k:np.mean([r[k] for r in group],axis=0).tolist()
                    for k in group[0] if k not in ('arm','subject')}
    output=REPO/'results/things_eeg/fusion_c012_20260915/score_analysis.json'
    write_json(output,dict(subjects=list(range(1,6)),means=means,folds=rows,
        caveats=['One seed, five subjects; subjects share the same 200 test images.',
                 'Reference and shared models have different checkpoint selection policies.',
                 'Hybrid scores are exploratory held-out-test diagnostics, not validated new methods.',
                 'Count metrics in means are average counts per 200-query fold.']))
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
