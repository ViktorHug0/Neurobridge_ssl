"""K51: the in-trial simulated re-centring (K46-K50, paired estimator, J = 16) on any trained model. Pure inductive.
For each query x: k = K + comb(x - K);  o = mean_400 e(K + c) + mean_16 [e(k + c_j) - e(K + c_j)];  score e(x) - o + anchor.
c = 400 key-free training contents (9-subject averages), anchor = mean training-image embedding of the head's space.
Heads: the model's own, and with train.py --aux_targets --aux_dim (K24) every auxiliary head; "all" = mean of the cosine matrices.
Columns: plain / centre (pure inductive) / cal (true mean of the 200 queries: test data required).
    k51_centre_any.py <run> [<run> ...]      run = cvpr/loso/<arm>/<seed> or an arm of results/things_eeg/subject_key/train"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 400, replace=False))
rows = []
for RUN in sys.argv[1:]:
    root = RUN if '/seed' in RUN else f'{SK}/train/{RUN}'
    for d in sorted(glob.glob(f'{root if os.path.isabs(root) else f"{LOSO}/{root}"}/2*-sub-*')):
        if not os.path.exists(f'{d}/result.csv'):
            continue
        fold = int(d[-2:])
        m = load(root, fold)
        aux = torch.load(f'{d}/checkpoint_test_best.pth', map_location='cpu', weights_only=False).get('aux_state_dict')
        pre = []
        if aux:
            (m.enc.head if hasattr(m.enc, 'head') else m.enc.proj_eeg).register_forward_hook(lambda mod, i, o: pre.append(i[0].detach().float().cpu()))
        G, anc = [img_emb(m)], [img_emb(m, 'train').mean(0)]
        for j, feat_dir in enumerate(m.args.aux_targets if aux else []):
            bank = {s: nz(torch.tensor(np.load(f'{feat_dir}/image_{s}.npy').astype(np.float32)).flatten(0, -2), dim=1) for s in ('train', 'test')}
            g = {s: nz(v @ aux['img'][f'{j}.weight'].T + aux['img'][f'{j}.bias'], dim=1) for s, v in bank.items()}
            G.append(g['test']); anc.append(g['train'].mean(0))

        def heads(v):
            pre.clear(); e = m.embed(v, fold); f = torch.cat(pre) if aux else None
            return [nz(e, dim=1)] + [nz(f @ aux['heads'][f'{j}.weight'].T + aux['heads'][f'{j}.bias'], dim=1) for j in range(len(G) - 1)]

        x, K = T(eeg(fold)), mean_key(m)
        c = torch.stack([T(eeg(s, 'train')[IM]) - train_key(s) for s in m.train_subs]).mean(0)
        comb = lambda d: (lambda p: (p - p.mean(-1, keepdim=True)).repeat(1, 1, 5))(d.unflatten(-1, (5, 50)).mean(-2))
        k = K + comb(x - K)
        E, eK = heads(x), heads(K + c)
        idx = torch.stack([torch.randperm(400, generator=torch.Generator().manual_seed(i))[:16] for i in range(200)])
        sim = heads((k[:, None] + c[idx]).flatten(0, 1))               # 200 x 16 simulated trials
        O = [b.mean(0) + (s.view(200, 16, -1) - b[idx]).mean(1) for s, b in zip(sim, eK)]
        S = {'plain': [e @ g.T for e, g in zip(E, G)], 'centre': [nz(e - o + a, dim=1) @ g.T for e, o, a, g in zip(E, O, anc, G)],
             'cal': [nz(e - e.mean(0) + a, dim=1) @ g.T for e, a, g in zip(E, anc, G)]}
        r = dict(run=RUN, fold=fold, **{f'{n} main': acc(s[0]).item() for n, s in S.items()})
        if aux:
            r.update({f'{n} all': acc(sum(s)).item() for n, s in S.items()})
        rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k51_centre_any_{os.environ.get("TAG", "x")}.csv', index=False)
print(df.groupby('run', sort=False).agg(['mean', 'count']).drop(columns='fold').round(2).T.to_string())
