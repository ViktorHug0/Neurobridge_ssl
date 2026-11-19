"""K24: score a multi-space encoder (train.py --aux_targets <dirs> --aux_dim 128: one trunk, the main head plus one
linear head per extra image space, trained together). A query is scored in every space (each calibrated, K10) and the
cosine matrices are averaged. Columns: main head alone / every extra head alone / all heads; top-1 and mean rank.
References on the same folds: the single-space ATM (atm128_vc seed 3300) and, where cached, the distilled linear heads
of K21 on that frozen ATM (spaces L28 + bigG + L20).
    k24_multispace.py <arm> [<arm> ...]     RUNNING=1: also folds still training"""
import pandas as pd
from sk_lib import *
from mi_lib import _DS

rank = lambda S: (S > S.diag()[:, None]).sum(1).float().mean().item()
root = lambda a: f'{SK}/train/{a}'
rows = []
for arm in sys.argv[1:]:
    for d in sorted(glob.glob(f'{root(arm)}/*-sub-*')):
        fold = int(d[-2:])
        if not (os.path.exists(f'{d}/result.csv') or os.environ.get('RUNNING')):
            continue
        for ck in sorted(os.path.basename(p) for p in glob.glob(f'{d}/checkpoint_*.pth')):
            tag = {'checkpoint_test_best.pth': 'val', 'checkpoint_testsel_best.pth': 'testsel'}.get(ck, ck[11:-4])
            m = load(root(arm), fold, ck)
            aux = torch.load(f'{d}/{ck}', map_location='cpu', weights_only=False)['aux_state_dict']
            pre = {}
            hook = (m.enc.head if hasattr(m.enc, 'head') else m.enc.proj_eeg).register_forward_hook(lambda mod, i, o: pre.setdefault('x', []).append(i[0].detach().float().cpu()))
            S = [calibrated(m, fold)]  # runs the 200 test queries through the trunk once
            P = [member_scores(m, fold)[0]]  # the same heads WITHOUT re-centring (the strictly inductive score)
            hook.remove()
            f = torch.cat(pre['x'])[:200]
            for k, feat_dir in enumerate(m.args.aux_targets):
                bank = {s: nz(torch.tensor(np.load(f'{feat_dir}/image_{s}.npy').astype(np.float32)).flatten(0, -2), dim=1) for s in ('train', 'test')}
                g = {s: nz(v @ aux['img'][f'{k}.weight'].T + aux['img'][f'{k}.bias'], dim=1) for s, v in bank.items()}
                e = nz(f @ aux['heads'][f'{k}.weight'].T + aux['heads'][f'{k}.bias'], dim=1)
                S.append(nz(e - e.mean(0) + g['train'].mean(0), dim=1) @ g['test'].T)
                P.append(e @ g['test'].T)
            r = dict(arm=arm, fold=fold, ckpt=tag, main=acc(S[0]).item(), **{f'head{k + 1}': acc(s).item() for k, s in enumerate(S[1:])},
                     all=acc(sum(S)).item(), all_rank=rank(sum(S)), main_rank=rank(S[0]), main_plain=acc(P[0]).item(), all_plain=acc(sum(P)).item(), first3=acc(sum(S[:3])).item(), first3_rank=rank(sum(S[:3])))  # first3 = main + the first two extra heads
            rows.append(r)
            print({k: (round(v, 2) if not isinstance(v, str) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
folds = sorted(df.fold.unique())
ref = []
for fold in folds:
    _DS.pop('img_dir', None)  # the loop above may have left another image space in mi_lib's arrays
    s = calibrated(load('atm128_vc/seed3300', fold), fold)
    r = dict(fold=fold, single_atm=acc(s).item(), single_atm_rank=rank(s))
    p = f'{SK}/cache/heads_kd_atm128_vc_seed3300_f{fold}.pt'
    if os.path.exists(p):
        h = torch.load(p).float()[[0, 2, 3]].sum(0)
        r.update(distilled3=acc(h).item(), distilled3_rank=rank(h))
    ref.append(r)
pd.set_option('display.width', 220)
print('folds', folds)
print(df.groupby(['arm', 'ckpt'], sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
print(pd.DataFrame(ref).mean().drop('fold').round(2).to_string())
