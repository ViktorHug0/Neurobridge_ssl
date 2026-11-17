"""ATM targeted substitution with the new subject's key estimated from DISJOINT unlabelled data: the mean ERP of its
training-image trials (never the test trials). x' = x - m_u(train) + K for the readout, attention map from m_u(train)."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e10_keysub import train_key
from e13_atm_attention import emb, score

if __name__ == '__main__':
    rows = []
    for arm in sys.argv[1].split(','):
        for fold in range(1, 11):
            m = GM(glob.glob(f'{LOSO}/{arm}/*-sub-{fold:02d}')[0])
            g = img_emb(m)
            x = T(eeg(fold))
            mu_te, mu_tr = x.mean(0, keepdim=True), train_key(fold)[None]
            K = torch.stack([train_key(t) for t in m.train_subs]).mean(0, keepdim=True)
            for src, mu in (('test trials', mu_te), ('train-image trials', mu_tr)):
                A = emb(m, mu, fold)[1]
                rows.append(dict(arm=arm, fold=fold, key_from=src, own=score(emb(m, x, fold)[0], g)['plain_cosine'],
                                 **{'targeted ' + k: v for k, v in score(emb(m, x - mu + K, fold, A)[0], g).items()},
                                 **{'half ' + k: v for k, v in score(emb(m, x - mu + (mu + K) / 2, fold, A)[0], g).items()}))
            print(arm, fold, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e13d_key_source.csv', index=False)
    print(df.groupby(['arm', 'key_from'], sort=False).mean(numeric_only=True).drop(columns='fold').round(1).to_string())
