"""Plot recorded G2 metrics without imputing early-stopped folds."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]/'results/things_eeg/fusion_three_20260915'
OUT=ROOT/'plots'
OUT.mkdir(exist_ok=True)
histories=[json.loads((ROOT/f'runs/G2/sub-{s:02d}/epochs.json').read_text()) for s in range(1,6)]
lengths=[len(h) for h in histories]
common=min(lengths); maximum=max(lengths)
values=np.full((5,maximum,4),np.nan)
for s,h in enumerate(histories):
    for r in h:
        values[s,r['epoch']-1]=[*r['train_loss'],np.mean(r['train_loss']),r['validation']['top1']]
mean=np.nanmean(values,axis=0); sd=np.nanstd(values,axis=0,ddof=1)
n=np.isfinite(values[:,:,0]).sum(0); epochs=np.arange(1,maximum+1)
with (OUT/'g2_epoch_means.csv').open('w') as handle:
    writer=csv.writer(handle)
    writer.writerow(['epoch','n_folds','train_loss_a','train_loss_b','train_loss_mean','validation_top1','train_loss_sd','validation_top1_sd'])
    for i in range(maximum):writer.writerow([i+1,int(n[i]),*mean[i],sd[i,2],sd[i,3]])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,5.4))
for ax,col,color,title,ylabel in zip(axes,[2,3],['#3266ad','#18876e'],
        ['Training loss','Validation retrieval accuracy'],['Contrastive loss (mean of two branches)','Fused top-1 accuracy (%)']):
    ax.fill_between(epochs,mean[:,col]-sd[:,col],mean[:,col]+sd[:,col],color=color,alpha=.13,label='±1 SD across available folds')
    ax.plot(epochs[:common],mean[:common,col],color=color,lw=2.3,label='Mean of all 5 folds')
    ax.plot(epochs[common-1:],mean[common-1:,col],color=color,lw=2,ls='--',label='Mean of remaining folds')
    ax.axvline(common,color='gray',lw=1,ls=':')
    ax.axvspan(common,maximum,color='gray',alpha=.07)
    ax.set(title=title,xlabel='Epoch',ylabel=ylabel,xlim=(1,maximum))
    ax.grid(alpha=.15);ax.legend(fontsize=8,loc='best')
fig.suptitle('G2 · original shared-temporal-stem experiment · LOSO subjects 1–5',fontweight='bold')
fig.text(.5,.105,'Epochs 1–89: n=5  |  90–91: n=4  |  92–100: n=3. No extrapolation or carry-forward.',ha='center',fontsize=10)
fig.text(.5,.052,'Not recorded per epoch: train accuracy, validation loss, test loss or test accuracy.\nSelected-checkpoint mean test top-1: 40.5% (different selected epochs; not an epoch curve).',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.17,1,.94))
fig.savefig(OUT/'g2_epoch_curves.png',dpi=180)
fig.savefig(OUT/'g2_epoch_curves.pdf')
print('Fold lengths:',lengths)
print(OUT/'g2_epoch_curves.png')
