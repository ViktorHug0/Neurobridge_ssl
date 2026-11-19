"""K21: one frozen trunk, several linear heads: variants of K20. Trunk = $TRUNK (default atm128_vc/seed3300; the 1440-d input of its head).
Heads are ridge regressions (lambda 0.1 x mean eigenvalue) from the trunk feature of the clean training trials to
  kd   the teacher member's EEG embedding of the same trial (teacher file t6), or
  img  that member's embedding of the trial's IMAGE (no teacher EEG at all: a linear probe into the member's image space).
Each head is scored against its member's image embeddings of the 200 test images, calibrated (image anchor) or not.
Reported: every head, and the mean of the cosine matrices over subsets of heads; top-1 and mean rank of the target.
Caches: results/things_eeg/subject_key/cache/{trunk_<trunk>,gte_t6,heads_<kind>_<trunk>}_f<fold>.pt
Usage: k21_heads.py [folds]"""
import pandas as pd
from sk_lib import *
from module.eeg_encoder.model import run_eeg_backbone

FOLDS = [int(f) for f in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 5, 7]
TRUNK = os.environ.get('TRUNK', 'atm128_vc/seed3300')
TN = TRUNK.replace('/', '_')
TEACHER = os.environ.get('TEACHER', 't6')  # t9 = t6 + ATM teachers on CLIP-H, DINOv2 L40, InternViT L44
SUF = '' if TEACHER == 't6' else '_' + TEACHER
SETS = {'all6': range(6), 'atm4': range(4), 'spaces3': (0, 2, 3), 'spaces3+labram': (0, 2, 3, 4), 'own': (0,)}
if TEACHER == 't9':
    SETS = {'spaces3': (0, 2, 3), 'spaces6': (0, 2, 3, 6, 7, 8), '+cliph': (0, 2, 3, 6), '+dino40': (0, 2, 3, 7), '+iv44': (0, 2, 3, 8), 'all9': range(9)}
os.makedirs(f'{SK}/cache', exist_ok=True)
rank = lambda S: (S > S.diag()[:, None]).sum(1).float().mean().item()
rows = []
for fold in FOLDS:
    t = torch.load(f'{SK}/teachers/{TEACHER}_f{fold}.pt', map_location='cpu')
    pg, pt = f'{SK}/cache/gte_{TEACHER}_f{fold}.pt', f'{SK}/cache/trunk_{TN}_f{fold}.pt'
    if not os.path.exists(pg):
        Gte, teacher = [], []
        for run in t['runs']:  # one member on the GPU at a time
            mm = load(run, fold); Gte.append(img_emb(mm, 'test')); teacher.append(calibrated(mm, fold)); del mm; torch.cuda.empty_cache()
        torch.save({'Gte': Gte, 'teacher': teacher}, pg)
    c = torch.load(pg); Gte, teacher = c['Gte'], c['teacher']
    if not os.path.exists(pt):
        m = load(TRUNK if '/seed' in TRUNK else f'{SK}/train/{TRUNK}', fold)  # a cvpr/loso run or an arm of this folder
        pre = {}
        (m.enc.head if hasattr(m.enc, 'head') else m.enc.proj_eeg).register_forward_hook(lambda mod, i, o: pre.__setitem__('x', i[0].detach().half().cpu()))

        @torch.no_grad()
        def feats(x, s):
            out = []
            for i in range(0, len(x), 50):  # small batches: a training run shares the GPU
                xb = x[i:i + 50].to(DEV)
                run_eeg_backbone(m.enc, m.args, xb, torch.full((len(xb),), s, device=DEV)); out.append(pre['x'])
            return torch.cat(out)

        torch.save({'tr': torch.cat([feats(T(eeg(s, 'train')), s) for s in t['subjects']]), 'te': feats(T(eeg(fold)), fold),
                    'own': calibrated(m, fold)}, pt)
        del m; torch.cuda.empty_cache()
    c = torch.load(pt)
    X, Xte = c['tr'].double(), c['te'].double()
    mu = X.mean(0); X -= mu; Xte -= mu
    XtX = X.T @ X
    A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
    img_of = torch.arange(len(X)) % 16540
    r = dict(fold=fold, own_head=acc(c['own']).item(), teacher=acc(sum(teacher)).item(), teacher_rank=rank(sum(teacher)), own_rank=rank(c['own']))
    for kind in ('kd', 'img')[:1 if SUF else 2]:
        S = {True: [], False: []}
        for k in range(len(t['eeg'])):
            Y = (t['eeg'][k] if kind == 'kd' else t['img'][k][img_of]).double()
            e = nz((Xte @ (A @ (X.T @ (Y - Y.mean(0)))) + Y.mean(0)).float(), dim=1)
            S[False].append(e @ Gte[k].T)
            S[True].append(nz(e - e.mean(0) + t['img'][k].float().mean(0), dim=1) @ Gte[k].T)
            r[f'{kind}_head{k}'] = acc(S[True][-1]).item()
        for name, idx in SETS.items():
            r[f'{kind}_{name}'] = acc(sum(S[True][k] for k in idx)).item()
            r[f'{kind}_{name}_rank'] = rank(sum(S[True][k] for k in idx))
            r[f'{kind}_{name}_nocal'] = acc(sum(S[False][k] for k in idx)).item()
        torch.save(torch.stack(S[True]).half(), f'{SK}/cache/heads_{kind}_{TN}{SUF}_f{fold}.pt')
    rows.append(r)
    print({k: round(v, 1) for k, v in r.items() if 'head' not in k or k == 'own_head'}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k21_heads_{TN}{SUF}.csv', index=False)
print(df.mean().round(2).to_string())
