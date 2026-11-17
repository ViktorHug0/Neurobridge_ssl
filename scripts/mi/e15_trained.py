"""Evaluate the trained key arms (results/things_eeg/mi/train/<arm>) on the held-out subject: plain cosine and
full_sattc, with the input prepared as in training (--subject_key demean: x - m_u; split: [x - m_u, m_u])."""
import pandas as pd
from mi_lib import *
from e09_arch import GM
from module.transductive import infer

rows = []
for arm in sys.argv[1].split(','):
    for d in sorted(glob.glob(f'{OUT}/train/{arm}/*-sub-*')):
        if not os.path.exists(d + '/result.csv'):
            continue
        m = GM(d)
        fold = m.test_sub
        g = img_emb(m)
        x = T(eeg(fold))
        mu = x.mean(0, keepdim=True)
        mode = getattr(m.args, 'subject_key', 'keep')
        xin = {'keep': x, 'swap': x, 'demean': x - mu, 'split': torch.stack([x - mu, mu.expand_as(x)], 1)}[mode]
        e = m.embed(xin, fold)
        r = dict(arm=arm, fold=fold, result_csv=pd.read_csv(d + '/result.csv')['best top1 acc'][0], epoch=pd.read_csv(d + '/result.csv')['best epoch'][0])
        for head in ('plain_cosine', 'full_sattc'):
            S = torch.as_tensor(np.asarray(infer(e.numpy(), g.numpy(), head)))
            r[head] = 100 * (S.argmax(1) == torch.arange(200)).float().mean().item()
        rows.append(r)
        print(r, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e15_trained{os.environ.get("TAG", "")}.csv', index=False)
print(df.groupby('arm').mean(numeric_only=True).round(2))
