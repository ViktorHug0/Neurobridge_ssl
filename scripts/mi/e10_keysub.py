"""Test-time key substitution on the held-out subject (frozen model, label-free):
    x' = x - m_u + lam * (K - m_u) ... i.e. key = m_u + lam (K - m_u),  K = canonical key.
  K_test  : mean over training subjects of their mean ERP on the 200 test images (needs the other subjects' test EEG)
  K_train : mean over training subjects of their mean ERP on the TRAINING images (deployable: training data only)
  ens9    : average of the embeddings obtained under each of the 9 training keys; conf_pick: the single training key
            with the highest mean top-1 cosine (label-free); best1: best single training key (oracle)
  n       : m_u estimated from only n of the held-out subject's test trials (random subset), lam = 1, K_train
"""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO

ARCHS = sys.argv[1].split(',') if len(sys.argv) > 1 else []
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else list(range(1, 11))
KTR = {}


def train_key(s):
    if s not in KTR:
        p = f'{OUT}/keys_train_sub{s:02d}.npy'
        if not os.path.exists(p):
            np.save(p, np.asarray(eeg(s, 'train'), dtype=np.float32).mean(0))
        KTR[s] = T(np.load(p))
    return KTR[s]


if __name__ == '__main__':
    rows = []
    for arch in ARCHS:
        for fold in FOLDS:
            d = glob.glob(f'{LOSO}/{arch}/seed3300/*-sub-{fold:02d}') if arch not in RUNS else [run_dir(arch, fold)]
            m = GM(d[0])
            g = img_emb(m)
            x = T(eeg(fold))
            mu = x.mean(0, keepdim=True)
            Kte = torch.stack([T(eeg(t)).mean(0) for t in m.train_subs]).mean(0)
            Ktr = torch.stack([train_key(t) for t in m.train_subs]).mean(0)
            r = dict(arch=arch, fold=fold, own=top1(m.embed(x, fold), g))
            for lam in (0.5, 1.0, 1.5):
                r[f'Ktest_{lam}'] = top1(m.embed(x + lam * (Kte - mu), fold), g)
                r[f'Ktrain_{lam}'] = top1(m.embed(x + lam * (Ktr - mu), fold), g)
            nz = torch.nn.functional.normalize
            per_key = [nz(m.embed(x - mu + T(eeg(t)).mean(0), fold), dim=1) for t in m.train_subs]  # 9 training keys
            r['ens9'] = top1(sum(per_key), g)  # key ensemble: average the embeddings obtained under each training key
            r['ens9+K'] = top1(sum(per_key) + 9 * nz(m.embed(x - mu + Kte, fold), dim=1), g)
            r['best1'] = max(top1(e, g) for e in per_key)  # oracle single training key
            conf = torch.stack([(e @ g.T).max(1).values.mean() for e in per_key])  # label-free: mean top-1 cosine
            r['conf_pick'] = top1(per_key[int(conf.argmax())], g)
            rng = np.random.RandomState(0)
            for n in (5, 20, 50):
                acc = []
                for _ in range(3):
                    mh = x[rng.choice(200, n, replace=False)].mean(0, keepdim=True)
                    acc.append(top1(m.embed(x - mh + Ktr, fold), g))
                r[f'n{n}'] = np.mean(acc)
            rows.append(r)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(f'{OUT}/e10_keysub{os.environ.get("TAG", "")}.csv', index=False)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).groupby('arch', sort=False).mean(numeric_only=True).drop(columns='fold').round(1))
