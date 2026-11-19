"""K06: is ATM's dissociation (attention keeps the new subject's own key, the rest takes the canonical key) a
property of attention in general? Token encoder rmix (TokenREVE: 63 electrode tokens + 5 time tokens, 6 blocks).
Two passes per trial: pass A on the input that sets the ATTENTION weights of chosen blocks, pass B on the input
that flows through values / feed-forward / readout with those weights forced.
  rows: stream input (own key, midpoint, K); columns: where the attention weights come from.
Usage: k06_token_targeted.py [run]   (default rmix/seed3300, test-selected: no validation-selected rmix exists yet)"""
import pandas as pd
from sk_lib import *
from module.eeg_encoder.alljoined_encoders import _SwitchBlock
from module.eeg_encoder.model import run_eeg_backbone

RUN = sys.argv[1] if len(sys.argv) > 1 else 'rmix/seed3300'
ST = {'rec': None, 'force': None, 'i': 0, 'blocks': ()}


def fwd(self, x):
    b, n, d = x.shape
    xn = self.n1(x)
    q, k, v = (t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for t in self.qkv(xn).chunk(3, -1))
    A = torch.softmax(q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5, -1)
    if ST['force'] is not None and ST['i'] in ST['blocks']:
        A = ST['force'][ST['i']]
    if ST['rec'] is not None:
        ST['rec'].append(A)
    ST['i'] += 1
    x = x + self.out((A @ v).transpose(1, 2).reshape(b, n, d))
    return x + self._ffn(x, self.n2, self.ff1, self.ff2)


_SwitchBlock.forward = fwd


@torch.no_grad()
def embed2(m, x_att, x_stream, sub, blocks):
    out = []
    for i in range(0, len(x_att), 100):
        sid = torch.full((len(x_att[i:i + 100]),), sub, device=DEV, dtype=torch.long)
        ST.update(rec=[], force=None, i=0)
        run_eeg_backbone(m.enc, m.args, x_att[i:i + 100].to(DEV), sid)
        ST.update(force=ST['rec'], rec=None, i=0, blocks=blocks)
        out.append(m.eeg_proj(run_eeg_backbone(m.enc, m.args, x_stream[i:i + 100].to(DEV), sid)).float().cpu())
    ST.update(force=None)
    return torch.cat(out)


rows = []
for fold in range(1, 11):
    m = load(RUN, fold)
    g = img_emb(m)
    x = T(eeg(fold)); mu = x.mean(0, keepdim=True); K = mean_key(m)
    inp = {'own': x, 'half': x + 0.5 * (K - mu), 'K': x - mu + K}
    for stream in inp:
        r = dict(fold=fold, stream=stream, same=top1(embed2(m, inp[stream], inp[stream], fold, ()), g))
        if stream != 'own':
            r.update(att_own=top1(embed2(m, x, inp[stream], fold, range(6)), g),
                     att_own_0_2=top1(embed2(m, x, inp[stream], fold, range(3)), g),
                     att_own_3_5=top1(embed2(m, x, inp[stream], fold, range(3, 6)), g))
        else:
            r.update(att_K=top1(embed2(m, inp['K'], x, fold, range(6)), g))
        rows.append(r)
    print(fold, [{k: round(v, 1) if not isinstance(v, str) else v for k, v in r.items() if k != 'fold'} for r in rows[-3:]], flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k06_token_targeted_{RUN.replace("/", "_")}.csv', index=False)
print("same = attention from the stream input itself; att_own = attention weights of the own-key pass (all blocks / blocks 0-2 / 3-5); att_K = own-key stream with K-key attention")
print(df.groupby('stream', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
