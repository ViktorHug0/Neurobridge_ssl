"""Does the key substitution hold when the test trials are noisy (few repetitions averaged)?
Query = mean of the first r repetitions of each test image, r in {1, 4, 80}; the subject key is always the mean of
all its test trials. Plain cosine, held-out subject."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e10_keysub import train_key
from e13_atm_attention import emb

if __name__ == '__main__':
    rows = []
    for arm in sys.argv[1].split(','):
        for fold in range(1, 11):
            m = GM(glob.glob(f'{LOSO}/{arm}/*-sub-{fold:02d}')[0])
            g = img_emb(m)
            reps = eeg(fold, 'test', average=False)  # (200, 80, 63, 250)
            mu = T(eeg(fold)).mean(0, keepdim=True)
            K = torch.stack([train_key(t) for t in m.train_subs]).mean(0, keepdim=True)
            atm = m.args.eeg_encoder_type == 'ATM'
            A = emb(m, mu, fold)[1] if atm else None
            for r in (1, 4, 80):
                x = T(reps[:, :r].astype(np.float32).mean(1))
                row = dict(arm=arm, fold=fold, reps=r, own=top1(m.embed(x, fold), g), half=top1(m.embed(x + 0.5 * (K - mu), fold), g))
                if atm:
                    row['K readout, own attention'] = top1(emb(m, x - mu + K, fold, A)[0], g)
                rows.append(row)
            print(arm, fold, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e14_reps.csv', index=False)
    print(df.groupby(['arm', 'reps'], sort=False).mean(numeric_only=True).drop(columns='fold').round(1).to_string())
