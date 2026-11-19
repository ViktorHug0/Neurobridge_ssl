"""K04: does the key carry the subject's spatial mixing? A 63x63 electrode operator estimated from keys alone
(label-free):  W = argmin |W m_from - m_to|^2 + lam |W - I|^2,  lam = a * trace(m_from m_from^T) / 63.
 (1) seen subjects (the model has a decoder behind each key): trials of s re-mixed toward another training subject t.
       own: x_s | wrong: c_s + m_t | op: W c_s + m_t  (does re-mixing the CONTENT recover what the wrong key lost?)
 (2) held-out subject: toward the mean training key K.
       own | sub (the usual substitution) | op: W c_u + K   (ATM: attention map from the subject's own key in sub and op)
Usage: k04_keyop.py <run> [folds]"""
import pandas as pd
from sk_lib import *

RUN = sys.argv[1]
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else list(range(1, 11))
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def op(m_from, m_to, a):
    A, B = m_from[0].double(), m_to[0].double()  # (63, 250)
    lam = a * torch.trace(A @ A.T) / 63
    eye = torch.eye(63, dtype=torch.float64)
    return ((B @ A.T + lam * eye) @ torch.linalg.inv(A @ A.T + lam * eye)).float()


rows = []
for fold in FOLDS:
    m = load(RUN, fold)
    g = img_emb(m)
    atm = m.args.eeg_encoder_type == 'ATM'
    xs = {s: T(eeg(s)) for s in range(1, 11)}
    mu = {s: xs[s].mean(0, keepdim=True) for s in xs}
    K = mean_key(m)

    def emb(x, sub, key_for_attention=None):
        if atm and key_for_attention is not None:
            return atm_emb(m, x, sub, atm_emb(m, key_for_attention, sub)[1])[0]
        return m.embed(x, sub)

    # (1) seen subjects
    for j in range(3):
        s = m.train_subs[(fold * 3 + j * 3) % 9]
        others = [t for t in m.train_subs if t != s]
        t = others[(fold + j) % len(others)]
        c = xs[s] - mu[s]
        r = dict(fold=fold, part='seen', own=top1(emb(xs[s], s), g), wrong=top1(emb(c + mu[t], s), g))
        for a in ALPHAS:
            r[f'op{a}'] = top1(emb(torch.einsum('ij,njt->nit', op(mu[s], mu[t], a), c) + mu[t], s), g)
        rows.append(r)
    # (2) held-out subject
    c = xs[fold] - mu[fold]
    r = dict(fold=fold, part='held', own=top1(emb(xs[fold], fold), g), wrong=top1(embed_sub(m, xs[fold], fold, mu[fold], K), g))
    for a in ALPHAS:
        r[f'op{a}'] = top1(emb(torch.einsum('ij,njt->nit', op(mu[fold], K, a), c) + (K if atm else mu[fold] + 0.5 * (K - mu[fold])), fold, mu[fold]), g)
    rows.append(r)
    print(fold, {k: round(v, 1) for k, v in r.items() if k != 'part'}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k04_keyop_{RUN.replace("/", "_")}.csv', index=False)
pd.set_option('display.width', 200)
print("columns: own | wrong = seen: wrong key, held: usual substitution | op<a> = content re-mixed by the key-derived operator")
print(df.groupby('part', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
