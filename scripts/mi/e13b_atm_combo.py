"""ATM, held-out subject: which key should feed the attention (spatial filter) and which the downstream readout?
input key in {own, half, K} x attention map computed from the key alone in {own, half, K}  (K = mean training key,
half = midpoint). Plain cosine and full_sattc."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e10_keysub import train_key
from e13_atm_attention import emb, score

if __name__ == '__main__':
    rows = []
    for fold in range(1, 11):
        m = GM(glob.glob(f'{LOSO}/atm128/seed3300/*-sub-{fold:02d}')[0])
        g = img_emb(m)
        x = T(eeg(fold))
        mu = x.mean(0, keepdim=True)
        K = torch.stack([train_key(t) for t in m.train_subs]).mean(0, keepdim=True)
        keys = {'own': mu, 'half': (mu + K) / 2, 'K': K}
        A = {k: emb(m, v, fold)[1] for k, v in keys.items()}
        for kin, kv in keys.items():
            for ka, av in A.items():
                rows.append(dict(fold=fold, input_key=kin, attention_key=ka, **score(emb(m, x - mu + kv, fold, av)[0], g)))
        print(fold, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e13b_atm_combo.csv', index=False)
    print(df.pivot_table(index='input_key', columns='attention_key', values=['plain_cosine', 'full_sattc'], sort=False).round(1).to_string())
