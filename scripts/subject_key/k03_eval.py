"""K03: held-out top-1 of trained arms: own key, key substitution, anchored output calibration (K10), and the mean rank
of the target after calibration (0 = always first); validation-selected, test-selected and fixed-epoch checkpoints.
    k03_eval.py <arm or run dir> [<arm or run dir> ...]    arm = results/things_eeg/subject_key/train/<arm>;
    a name with a slash is a cvpr/loso run (tsconv_vc/seed3300). Only the folds every listed arm has are compared."""
import pandas as pd
from sk_lib import *

ARMS = sys.argv[1:]
root = lambda a: f'{LOSO}/{a}' if '/' in a else f'{SK}/train/{a}'
done = lambda a: {int(d[-2:]) for d in glob.glob(f'{root(a)}/*-sub-*') if os.path.exists(f'{d}/result.csv') or os.environ.get('RUNNING')}  # RUNNING=1: also folds still training
folds = sorted(set.intersection(*[done(a) for a in ARMS]))
rows = []
for a in ARMS:
    for f in folds:
        r = dict(arm=a, fold=f)
        fixed = sorted(os.path.basename(p) for p in glob.glob(f'{root(a)}/*-sub-{f:02d}/checkpoint_ep*.pth'))  # --save_epochs: no selection at all
        for tag, ck in [('val', 'checkpoint_test_best.pth'), ('testsel', 'checkpoint_testsel_best.pth')] + [(c[11:-4], c) for c in fixed]:
            if not glob.glob(f'{root(a)}/*-sub-{f:02d}/{ck}'):
                continue
            m = load(root(a), f, ck)
            own, sub = member_scores(m, f)
            cal = calibrated(m, f, 'eeg_raw' if getattr(m.args, 'person_centre_train', False) else 'img')
            r[f'own_{tag}'], r[f'sub_{tag}'], r[f'cal_{tag}'] = acc(own).item(), acc(sub).item(), acc(cal).item()
            r[f'rank_{tag}'] = (cal > cal.diag()[:, None]).sum(1).float().mean().item()  # mean number of gallery images ranked above the target, calibrated (lower = better; less noisy than top-1)
            torch.save({'own': own.half(), 'sub': sub.half(), 'cal': cal.half()}, f'{SK}/scores/{a.replace("/", "_")}_f{f}_{tag}.pt')
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print('folds', folds)
print(df.pivot(index='arm', columns='fold').round(1).to_string())
print(df.groupby('arm', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
