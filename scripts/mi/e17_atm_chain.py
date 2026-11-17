"""ATM: where along the network is the key used? Seen subject under a WRONG key; restore the right key's mean
activation at one sub-stage and continue (as E07-C for TSConv). The accuracy after restoring at stage j is what the
model gets when every nonlinearity before j saw the wrong key and every one after j the right key."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO

ARCH = sys.argv[1] if len(sys.argv) > 1 else 'atm128'


def chain(m, sid):
    e = m.enc
    L = e.encoder.encoder.attn_layers[0]
    ts = e.enc_eeg[0].tsconv
    ffn = lambda x: L.norm2(x + L.conv2(L.activation(L.conv1(x.transpose(-1, 1)))).transpose(-1, 1))
    return [('embed', lambda x: e.encoder.enc_embedding(x, sid[:len(x)])), ('attention', lambda x: x + L.attention(x)),
            ('norm1', L.norm1), ('ffn+norm2', ffn), ('enc_norm', lambda x: e.encoder.encoder.norm(x)[:, :63]),
            ('t_lin', lambda x: ts[:3](x.unsqueeze(1))), ('t_act', ts[3]), ('s_lin', ts[4:6]), ('s_act', ts[6]),
            ('bb', lambda x: e.enc_eeg[0].projection(x).flatten(1)), ('fc1', e.proj_eeg[0]), ('res', e.proj_eeg[1]),
            ('ln', torch.nn.Sequential(*list(e.proj_eeg)[2:])), ('align', m.eeg_proj)]


@torch.no_grad()
def run(m, x, sid, start=0, keep=False):
    acts = []
    for name, f in chain(m, sid)[start:]:
        x = f(x)
        acts.append(x)
    return acts if keep else x


if __name__ == '__main__':
    rows = []
    for fold in range(1, 11):
        m = GM(glob.glob(f'{LOSO}/{ARCH}/seed3300/*-sub-{fold:02d}')[0])
        g = img_emb(m)
        xs = {s: T(eeg(s)) for s in range(1, 11)}
        mu = {s: xs[s].mean(0, keepdim=True) for s in xs}
        names = [n for n, _ in chain(m, None)]
        for s in [m.train_subs[(fold * 3 + j * 3) % 9] for j in range(3)]:
            sid = torch.full((200,), s, dtype=torch.long)
            t = [o for o in m.train_subs if o != s][fold % 8]
            right = run(m, xs[s], sid, keep=True)
            wrong = run(m, xs[s] - mu[s] + mu[t], sid, keep=True)
            rows.append(dict(fold=fold, sub=s, stage='(own key)', acc=top1(right[-1], g)))
            rows.append(dict(fold=fold, sub=s, stage='(wrong key)', acc=top1(wrong[-1], g)))
            for j, n in enumerate(names):
                a = wrong[j] - wrong[j].mean(0, keepdim=True) + right[j].mean(0, keepdim=True)
                rows.append(dict(fold=fold, sub=s, stage=n, acc=top1(run(m, a, sid, start=j + 1) if j + 1 < len(names) else a, g)))
        print(fold, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/e17_atm_chain_{ARCH}.csv', index=False)
    print(df.groupby('stage', sort=False).acc.mean().round(1).to_string())
