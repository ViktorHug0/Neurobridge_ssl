"""Is the subject key specific to TSConv? Key-swap test on the 9 architectures of results/things_eeg/cvpr/loso
(same recipe: SubjectMix, InternViT L28, LOSO, seed 3300). Per fold: the held-out subject and 3 seen subjects,
each under its own key / two other training keys / the mean training key / no key."""
import pandas as pd
from mi_lib import *
import train as train_mod
from module.eeg_encoder.model import build_eeg_encoder, run_eeg_backbone

ARCHS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['tsconv', 'atm128', 'eegproject', 'eegnet', 'conformer', 'sqf', 'rmix', 'reve6', 'labram']
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 and sys.argv[2][0].isdigit() else list(range(1, 11))
LOSO = os.path.join(REPO, 'results/things_eeg/cvpr/loso')
argv, sys.argv = sys.argv, ['train.py']
DEFAULTS = vars(train_mod.parse_args())
sys.argv = argv


class GM:
    def __init__(self, d, ckpt='checkpoint_test_best.pth', channels=63):
        from types import SimpleNamespace
        c = json.load(open(f'{d}/train_config.json'))
        c['feature_dim'] = int(str(c['feature_dim']).split(',')[0])
        self.args = a = SimpleNamespace(**{**DEFAULTS, **c})
        ck = torch.load(f'{d}/{ckpt}', map_location=DEV, weights_only=False)
        img_dim = img_raw().shape[1]
        bb = a.eeg_backbone_dim or img_dim
        self.enc = build_eeg_encoder(a, bb, 250, channels)
        self.eeg_proj = build_projector(a.projector, bb, a.feature_dim, activation=a.projector_activation)
        self.img_proj = build_projector(a.projector, img_dim, a.feature_dim, activation=a.projector_activation)
        self.enc.load_state_dict(ck['model_state_dict'])
        self.eeg_proj.load_state_dict(ck['eeg_projector_state_dict'])
        self.img_proj.load_state_dict(ck['img_projector_state_dict'])
        for mod in (self.enc, self.eeg_proj, self.img_proj):
            mod.to(DEV).eval().requires_grad_(False)
        self.test_sub, self.train_subs = c['test_subject_ids'][0], c['train_subject_ids']

    @torch.no_grad()
    def embed(self, x, sub):
        out = []
        for i in range(0, len(x), 100):
            xb = x[i:i + 100].to(DEV)
            sid = torch.full((len(xb),), sub, device=DEV, dtype=torch.long)
            out.append(self.eeg_proj(run_eeg_backbone(self.enc, self.args, xb, sid)).float().cpu())
        return torch.cat(out)


if __name__ == '__main__':
    rows = []
    for arch in ARCHS:
        for fold in FOLDS:
            d = glob.glob(f'{LOSO}/{arch}/seed3300/*-sub-{fold:02d}')
            if not d or not os.path.exists(d[0] + '/checkpoint_test_best.pth'):
                continue
            m = GM(d[0])
            g = img_emb(m)
            xs = {s: T(eeg(s)) for s in range(1, 11)}
            mu = {s: xs[s].mean(0, keepdim=True) for s in xs}
            tm = torch.stack([mu[t] for t in m.train_subs]).mean(0)
            ref = pd.read_csv(d[0] + '/result.csv')['best top1 acc'][0]
            seen = [m.train_subs[(fold * 3 + j * 3) % 9] for j in range(3)]
            for s in [fold] + seen:
                c = xs[s] - mu[s]
                others = [t for t in m.train_subs if t != s]
                o1, o2 = others[fold % len(others)], others[(fold + 4) % len(others)]
                r = dict(arch=arch, fold=fold, sub=s, held=s == fold, ref=ref, own=top1(m.embed(xs[s], s), g),
                         other=np.mean([top1(m.embed(c + mu[o], s), g) for o in (o1, o2)]),
                         trainmean=top1(m.embed(c + tm, s), g), zero=top1(m.embed(c, s), g))
                rows.append(r)
            print(arch, fold, {k: round(v, 1) for k, v in rows[-4].items() if k in ('ref', 'own', 'other', 'trainmean', 'zero')}, flush=True)
            pd.DataFrame(rows).to_csv(f'{OUT}/e09_arch{os.environ.get("TAG", "")}.csv', index=False)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(df.groupby(['arch', 'held'], sort=False)[['ref', 'own', 'other', 'trainmean', 'zero']].mean().round(1))
