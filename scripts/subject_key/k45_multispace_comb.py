"""K45: the directly trained multi-head ATM (train.py --aux_targets ... --aux_dim 128, K24) with the in-trial key (K36).
Pure inductive, no re-centring: attention map of the raw trial, readout fed x - k + K, k = K (plain) or the in-trial key
(comb). Main head alone and the mean of the cosine matrices of all heads; validation-selected checkpoint.
    k45_multispace_comb.py <arm> [<arm> ...]"""
import pandas as pd
from sk_lib import *

rows = []
for arm in sys.argv[1:]:
    for d in sorted(glob.glob(f'{SK}/train/{arm}/2*-sub-*')):
        if not os.path.exists(f'{d}/result.csv'):
            continue
        fold = int(d[-2:])
        m = load(f'{SK}/train/{arm}', fold)
        aux = torch.load(f'{d}/checkpoint_test_best.pth', map_location='cpu', weights_only=False)['aux_state_dict']
        pre = []
        m.enc.proj_eeg.register_forward_hook(lambda mod, i, o: pre.append(i[0].detach().float().cpu()))
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        per = (x - K).view(200, 63, 5, 50).mean(2)
        att = atm_emb(m, x, fold)[1]
        r = dict(arm=arm, fold=fold)
        for name, k in (('plain', K), ('comb', K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5))):
            pre.clear()
            S = [nz(atm_emb(m, x - k + K, fold, att)[0], dim=1) @ g.T]
            f = torch.cat(pre)
            for j, feat_dir in enumerate(m.args.aux_targets):
                bank = nz(torch.tensor(np.load(f'{feat_dir}/image_test.npy').astype(np.float32)).flatten(0, -2), dim=1)
                S.append(nz(f @ aux['heads'][f'{j}.weight'].T + aux['heads'][f'{j}.bias'], dim=1) @ nz(bank @ aux['img'][f'{j}.weight'].T + aux['img'][f'{j}.bias'], dim=1).T)
            r[f'{name} main'], r[f'{name} all'] = acc(S[0]).item(), acc(sum(S)).item()
        rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k45_multispace_comb.csv', index=False)
print(df.groupby('arm', sort=False).agg(['mean', 'count']).drop(columns='fold').round(2).T.to_string())
