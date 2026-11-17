"""Does transductive adaptation (full_sattc: whitening + CSLS + Sinkhorn + Procrustes on the 200 unlabelled test
trials) replace the key lookup? Seen subjects under their own key vs a wrong key vs no key, and the held-out
subject, each scored with plain cosine and with full_sattc. If TTA brings seen/wrong-key back to seen/own-key,
the network's key-conditional decoder and test-time adaptation are two routes to the same subject-specific map."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from module.transductive import infer

ARCHS = sys.argv[1].split(',')
rows = []
for arch in ARCHS:
    for fold in range(1, 11):
        m = GM(glob.glob(f'{LOSO}/{arch}/seed3300/*-sub-{fold:02d}')[0])
        g = img_emb(m)
        xs = {s: T(eeg(s)) for s in range(1, 11)}
        mu = {s: xs[s].mean(0, keepdim=True) for s in xs}
        seen = [m.train_subs[(fold * 3 + j * 3) % 9] for j in range(2)]
        for s in [fold] + seen:
            others = [t for t in m.train_subs if t != s]
            conds = {'own': xs[s], 'wrong': xs[s] - mu[s] + mu[others[fold % len(others)]], 'nokey': xs[s] - mu[s]}
            for cn, x in conds.items():
                e = m.embed(x, s)
                r = dict(arch=arch, fold=fold, sub=s, held=s == fold, key=cn)
                for head in ('plain_cosine', 'full_sattc'):
                    S = torch.as_tensor(np.asarray(infer(e.numpy(), g.numpy(), head)))
                    r[head] = 100 * (S.argmax(1) == torch.arange(200)).float().mean().item()
                rows.append(r)
        print(arch, fold, flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/e12_tta_key{os.environ.get("TAG", "")}.csv', index=False)
df = pd.DataFrame(rows)
print(df.pivot_table(index=['arch', 'held'], columns='key', values=['plain_cosine', 'full_sattc'], sort=False).round(1).to_string())
