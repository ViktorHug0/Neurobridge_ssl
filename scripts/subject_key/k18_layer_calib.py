"""K18: is the output the best place to re-centre? ATM, new subject: at one stage of the network the subject's mean
activation is replaced by the training subjects' mean activation at that stage (1,500 training images, own keys,
4-repetition averages), the rest of the network runs on, and the output is scored plain and after the usual output
calibration (image anchor). Stages as in scripts/mi/e17_atm_chain.py.
Usage: k18_layer_calib.py <run>"""
import pandas as pd
from sk_lib import *
from e17_atm_chain import chain, run

RUN = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
IM = np.sort(np.random.default_rng(0).choice(16540, 1500, replace=False))
rows = []
for fold in range(1, 11):
    m = load(RUN, fold)
    g, anchor = img_emb(m), img_emb(m, 'train').mean(0)
    names = [n for n, _ in chain(m, None)]
    ref, n = None, 0
    for s in m.train_subs:
        xs = T(eeg(s, 'train')[IM])
        for i in range(0, len(xs), 100):
            acts = run(m, xs[i:i + 100].to(DEV), torch.full((100,), s, dtype=torch.long, device=DEV), keep=True)
            sums = [a.sum(0).cpu() for a in acts]
            ref = sums if ref is None else [r + a for r, a in zip(ref, sums)]
            n += 100
    ref = [r / n for r in ref]
    x = T(eeg(fold)).to(DEV)
    sid = torch.full((200,), fold, dtype=torch.long, device=DEV)
    acts = run(m, x, sid, keep=True)
    cal = lambda e: nz(e.cpu(), dim=1) - nz(e.cpu(), dim=1).mean(0) + anchor
    rows.append(dict(fold=fold, stage='(none)', hidden=top1(acts[-1].cpu(), g), hidden_out=top1(cal(acts[-1]), g)))
    for j, name in enumerate(names):
        a = acts[j] - acts[j].mean(0, keepdim=True) + ref[j].to(DEV)
        e = run(m, a, sid, start=j + 1) if j + 1 < len(names) else a
        rows.append(dict(fold=fold, stage=name, hidden=top1(e.cpu(), g), hidden_out=top1(cal(e), g)))
    print(fold, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k18_layer_calib.csv', index=False)
print(df.groupby('stage', sort=False)[['hidden', 'hidden_out']].mean().round(2).to_string())
