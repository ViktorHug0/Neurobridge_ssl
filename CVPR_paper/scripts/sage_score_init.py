import sys, numpy as np, pandas as pd
sys.path.insert(0,'scripts/things_eeg/tta_rebuttal'); sys.path.insert(0,'CVPR_paper/scripts')
from shared import apply_tta_calibration, fit_tta_calibration, score_features
from label_free import score_fit, score_deploy
from stream_grid import subject_params
rows=[]
for arm in ['atm128_s3300','atm_s3300','rmix_s3300','sqf_s3300','tsconv_s3300','score_orig_s3300']:
    P=f'results/things_eeg/cvpr/tta_screen_loso/{arm}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    for sid in range(1,11):
        z=np.load(f'results/things_eeg/cvpr/emb_cache/{arm}_b/sub-{sid:02d}.npz'); q,g=z['q'].astype(np.float32),z['g'].astype(np.float32)
        p=subject_params(P,sid); y=np.arange(200)
        acc=lambda s:100*(np.asarray(s).argmax(1)==y).mean()
        sage=acc(score_features(apply_tta_calibration(q,fit_tta_calibration(q,g,p)),g,use_csls=True,csls_k=p.csls_k))
        qs=score_fit(q,g)(q).astype(np.float32)   # SCORE-mapped queries as SAGE's starting point
        sage_si=acc(score_features(apply_tta_calibration(qs,fit_tta_calibration(qs,g,p)),g,use_csls=True,csls_k=p.csls_k))
        rows.append(dict(arm=arm,subject=sid,score=acc(score_deploy(q,g)),sage=sage,sage_scoreinit=sage_si))
    print(pd.DataFrame(rows).query('arm==@arm').mean(numeric_only=True).round(2).to_dict(),arm,flush=True)
pd.DataFrame(rows).to_csv('results/things_eeg/cvpr/analysis/sage_score_init.csv',index=False)
