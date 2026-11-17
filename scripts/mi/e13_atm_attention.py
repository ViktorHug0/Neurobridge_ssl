"""Where does ATM use the subject key? ATM = one attention layer over the 63 electrode tokens, then a TSConv readout.
Attention patching: run one input while forcing the attention weights recorded from another input.
  own            x_s                       wrong           c_s + m_t
  wrong+A(own)   wrong-key input, attention weights of the own-key run (per image)
  own+A(wrong)   own-key input, attention weights of the wrong-key run
  own+A(key)     own-key input, ONE attention map for all images: the map of the key alone (m_s)
Each scored with plain cosine and full_sattc. Also: how image-invariant the attention map is
(correlation between the maps of two images of one subject vs of two subjects on one image)."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from module.transductive import infer
from module.eeg_encoder.atm import FullAttention

ARCH = sys.argv[1] if len(sys.argv) > 1 else 'atm128'
STATE = {'rec': None, 'force': None}
_orig = FullAttention.forward


def patched(self, queries, keys, values):
    E = queries.shape[-1]
    A = torch.softmax(torch.einsum("blhe,bshe->bhls", queries, keys) / E ** 0.5, dim=-1)
    if STATE['force'] is not None:
        A = STATE['force'].expand(len(A), -1, -1, -1) if len(STATE['force']) == 1 else STATE['force'][STATE['pos']:STATE['pos'] + len(A)]
        STATE['pos'] += len(A)
    if STATE['rec'] is not None:
        STATE['rec'].append(A)
    return torch.einsum("bhls,bshd->blhd", A, values).contiguous()


FullAttention.forward = patched


def emb(m, x, sub, force=None):
    STATE.update(rec=[], force=force, pos=0)
    e = m.embed(x, sub)
    A = torch.cat(STATE['rec'])
    STATE.update(rec=None, force=None)
    return e, A


def score(e, g):
    out = {}
    for head in ('plain_cosine', 'full_sattc'):
        S = torch.as_tensor(np.asarray(infer(e.numpy(), g.numpy(), head)))
        out[head] = 100 * (S.argmax(1) == torch.arange(200)).float().mean().item()
    return out


if __name__ == '__main__':
    rows, inv = [], []
    for fold in range(1, 11):
        m = GM(glob.glob(f'{LOSO}/{ARCH}/seed3300/*-sub-{fold:02d}')[0])
        g = img_emb(m)
        xs = {s: T(eeg(s)) for s in range(1, 11)}
        mu = {s: xs[s].mean(0, keepdim=True) for s in xs}
        maps = {}
        for s in [fold] + [m.train_subs[(fold * 3 + j * 3) % 9] for j in range(2)]:
            others = [t for t in m.train_subs if t != s]
            t = others[fold % len(others)]
            wrong = xs[s] - mu[s] + mu[t]
            e_own, A_own = emb(m, xs[s], s)
            e_wr, A_wr = emb(m, wrong, s)
            _, A_key = emb(m, mu[s], s)
            conds = {'own': e_own, 'wrong': e_wr, 'wrong+A(own)': emb(m, wrong, s, A_own)[0], 'own+A(wrong)': emb(m, xs[s], s, A_wr)[0],
                     'own+A(key)': emb(m, xs[s], s, A_key)[0]}
            for cn, e in conds.items():
                rows.append(dict(fold=fold, sub=s, held=s == fold, cond=cn, **score(e, g)))
            maps[s] = A_own.flatten(1)
            c = lambda a, b: torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            inv.append(dict(fold=fold, sub=s, held=s == fold, within_subject=np.mean([c(maps[s][i], maps[s][i + 100]) for i in range(100)]),
                            to_key_map=np.mean([c(maps[s][i], A_key.flatten()) for i in range(0, 200, 4)])))
        ks = list(maps)
        inv.append(dict(fold=fold, sub=0, held=False, across_subjects=np.mean([c(maps[ks[0]][i], maps[ks[1]][i]) for i in range(100)])))
        print(fold, flush=True)
    df, inv = pd.DataFrame(rows), pd.DataFrame(inv)
    df.to_csv(f'{OUT}/e13_atm_attention_{ARCH}.csv', index=False)
    inv.to_csv(f'{OUT}/e13_atm_attmaps_{ARCH}.csv', index=False)
    print(df.pivot_table(index=['held', 'cond'], values=['plain_cosine', 'full_sattc'], sort=False).round(1).to_string())
    print(inv.drop(columns=['fold', 'sub']).groupby('held').mean().round(3))
