import json, numpy as np
from pathlib import Path
D=Path('results/things_eeg/synthetic_subjects/ensemble_screen/dumps')
MAN=json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
names,mats=[],[]
for n in sorted(MAN):
    per=[]; ok=True
    for s in range(1,11):
        p=D/f'{n}-sub{s:02d}.npz'
        if not p.exists(): ok=False;break
        d=np.load(p); e=d['eeg'].astype(np.float64); i=d['image'].astype(np.float64)
        e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12); i/=np.maximum(np.linalg.norm(i,axis=1,keepdims=True),1e-12)
        per.append((e@i.T).astype(np.float32))
    if ok: names.append(n); mats.append(np.stack(per))
S=np.stack(mats); N=len(names); tgt=np.arange(200)
Z=(S-S.mean(3,keepdims=True))/(S.std(3,keepdims=True)+1e-12)
f=Z.reshape(N,-1); f=(f-f.mean(1,keepdims=True))/(f.std(1,keepdims=True)+1e-12)
C=(f@f.T)/f.shape[1]
off=C[np.triu_indices(N,1)]
print(f'score-matrix correlation over {N} members, {len(off)} pairs:')
print(f'  min {off.min():.3f}  p10 {np.percentile(off,10):.3f}  median {np.median(off):.3f}  p90 {np.percentile(off,90):.3f}  max {off.max():.3f}')
hit=np.stack([(S[k].argmax(2)==tgt[None,:]) for k in range(N)]).reshape(N,-1).astype(float)
h=(hit-hit.mean(1,keepdims=True))/(hit.std(1,keepdims=True)+1e-12)
Ch=(h@h.T)/h.shape[1]; offh=Ch[np.triu_indices(N,1)]
print(f'error(correctness) correlation:')
print(f'  min {offh.min():.3f}  p10 {np.percentile(offh,10):.3f}  median {np.median(offh):.3f}  p90 {np.percentile(offh,90):.3f}  max {offh.max():.3f}')
best=['atm_iv','ge100','tsconv_eva','tsconv_vith']
idx=[names.index(b) for b in best]
print('\nthe winning k=4 combo, pairwise:')
for a in range(4):
    for b in range(a+1,4):
        print(f'  {best[a]:12s} vs {best[b]:12s}  score-corr {C[idx[a],idx[b]]:.3f}   error-corr {Ch[idx[a],idx[b]]:.3f}')
