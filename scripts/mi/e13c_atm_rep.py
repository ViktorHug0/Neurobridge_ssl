"""Replication of the targeted key substitution on other ATM checkpoints (label-free, held-out subject):
attention map from the subject's OWN key, readout fed the mean training key (K) or the midpoint (half)."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e10_keysub import train_key
from e13_atm_attention import emb, score

if __name__ == '__main__':
    rows = []
    for arm in sys.argv[1].split(','):
        for fold in range(1, 11):
            d = glob.glob(f'{LOSO}/{arm}/*-sub-{fold:02d}')
            if not d or not os.path.exists(d[0] + '/checkpoint_test_best.pth'):
                continue
            m = GM(d[0])
            g = img_emb(m)
            x = T(eeg(fold))
            mu = x.mean(0, keepdim=True)
            K = torch.stack([train_key(t) for t in m.train_subs]).mean(0, keepdim=True)
            A = emb(m, mu, fold)[1]
            for name, key, att in (('own', mu, None), ('K everywhere', K, None), ('half everywhere', (mu + K) / 2, None),
                                   ('K readout, own attention', K, A), ('half readout, own attention', (mu + K) / 2, A)):
                rows.append(dict(arm=arm, fold=fold, cond=name, **score(emb(m, x - mu + key, fold, att)[0], g)))
            print(arm, fold, flush=True)
            pd.DataFrame(rows).to_csv(f'{OUT}/e13c_atm_rep{os.environ.get("TAG", "")}.csv', index=False)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index='arm', columns='cond', values='plain_cosine', sort=False).round(1).to_string())
    print(df.pivot_table(index='arm', columns='cond', values='full_sattc', sort=False).round(1).to_string())
    w = df.pivot_table(index=['arm', 'fold'], columns='cond', values='plain_cosine')
    print((w['K readout, own attention'] - w['own']).unstack('fold').round(1).to_string())
