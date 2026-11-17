"""Inductive score ensemble (mean of cosine score matrices) with and without the test-time key substitution.
Members: ATM-128 (targeted: own-key attention, K readout), TSConv and Squeezeformer (key = midpoint to K)."""
import itertools
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e10_keysub import train_key
from e13_atm_attention import emb

if __name__ == '__main__':
    SEED = sys.argv[1] if len(sys.argv) > 1 else 'seed3300'
    ARMS = ['atm128', 'tsconv', 'sqf', 'eegproject']
    nz = torch.nn.functional.normalize
    rows = []
    for fold in range(1, 11):
        S = {}
        for arm in ARMS:
            d = glob.glob(f'{LOSO}/{arm}/{SEED}/*-sub-{fold:02d}') or glob.glob(f'{LOSO}/{arm}/seed3300/*-sub-{fold:02d}')
            m = GM(d[0])
            g = img_emb(m)
            x = T(eeg(fold))
            mu = x.mean(0, keepdim=True)
            K = torch.stack([train_key(t) for t in m.train_subs]).mean(0, keepdim=True)
            S[arm, 'own'] = nz(m.embed(x, fold), dim=1) @ g.T
            if arm == 'atm128':
                S[arm, 'sub'] = nz(emb(m, x - mu + K, fold, emb(m, mu, fold)[1])[0], dim=1) @ g.T
            else:
                S[arm, 'sub'] = nz(m.embed(x + 0.5 * (K - mu), fold), dim=1) @ g.T
        y = torch.arange(200)
        for k in range(1, len(ARMS) + 1):
            for combo in itertools.combinations(ARMS, k):
                for mode in ('own', 'sub'):
                    sc = sum(S[a, mode] for a in combo)
                    rows.append(dict(fold=fold, members='+'.join(combo), k=k, mode=mode, acc=100 * (sc.argmax(1) == y).float().mean().item()))
        print(fold, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e16_ensemble_{SEED}.csv', index=False)
    print(df.pivot_table(index=['k', 'members'], columns='mode', values='acc').round(2).to_string())
