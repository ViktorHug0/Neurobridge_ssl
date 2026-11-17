"""Three follow-ups on test-time key substitution (held-out subject, frozen models, K = mean training key from
TRAINING images):
  a. does it add to the transductive heads (module.transductive.infer: csls / sinkhorn / full_sattc)?
  b. is the per-fold gain predicted by how far the held-out key is from the training keys (label-free)?
  c. which time window of the key matters: substitute the key only inside one window."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e10_keysub import train_key
from module.transductive import infer

ARCHS = sys.argv[1].split(',')
WINS = {'0-100ms': (0, 25), '100-200ms': (25, 50), '200-400ms': (50, 100), '400-1000ms': (100, 250)}
rows = []
for arch in ARCHS:
    for fold in range(1, 11):
        m = GM(glob.glob(f'{LOSO}/{arch}/seed3300/*-sub-{fold:02d}')[0])
        g = img_emb(m)
        x = T(eeg(fold))
        mu = x.mean(0, keepdim=True)
        keys = torch.stack([train_key(t) for t in m.train_subs])
        K = keys.mean(0)
        r = dict(arch=arch, fold=fold)
        # b. distance of the held-out key to the centroid, in units of the training keys' mean distance to it
        r['key_dist'] = ((mu[0] - K).norm() / (keys - K).flatten(1).norm(dim=1).mean()).item()
        r['key_nn_dist'] = ((keys - mu).flatten(1).norm(dim=1).min() / (keys - K).flatten(1).norm(dim=1).mean()).item()
        embs = {'own': m.embed(x, fold), 'sub0.5': m.embed(x + 0.5 * (K - mu), fold), 'sub1': m.embed(x + (K - mu), fold)}
        for name, e in embs.items():
            for head in ('plain_cosine', 'csls', 'sinkhorn', 'full_sattc'):
                S = torch.as_tensor(np.asarray(infer(e.numpy(), g.numpy(), head)))
                r[f'{name}|{head}'] = 100 * (S.argmax(1) == torch.arange(200)).float().mean().item()
        for wn, (a, b) in WINS.items():
            d = torch.zeros_like(mu)
            d[..., a:b] = (K - mu)[..., a:b]
            r[f'win {wn}'] = top1(m.embed(x + d, fold), g)
        rows.append(r)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/e11_tta_dist{os.environ.get("TAG", "")}.csv', index=False)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.groupby('arch', sort=False).mean(numeric_only=True).drop(columns='fold').round(1).T.to_string())
df['gain'] = df['sub1|plain_cosine'] - df['own|plain_cosine']
print('corr(gain, key_dist) per arch:', df.groupby('arch').apply(lambda d: d.gain.corr(d.key_dist)).round(2).to_dict())
