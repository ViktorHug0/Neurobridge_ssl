"""Challenge-regime helpers (NeurIPS Track 1 sealed phase) for the MI experiments.
Frozen cv4 members trained with --holdout_images 5000, scored like logs/holdout_eval.py: single 32x120 trials, per-64-batch
scaling, repeats averaged within (person, image), cosine top-5 against a 5,000-image gallery, r = 1 / 3 / 4.
A "cell" = a set of people x a set of 5,000 training-set images:
    HELD = the 5,000 images no ho5k run trained on;  SEEN = 5,000 of the 11,540 trained images."""
import glob, json, os, sys
from types import SimpleNamespace
import numpy as np, torch

R = '/nasbrain/p20fores/Neurobridge_SSL'
sys.path.insert(0, R)
from neurips_challenge.tta_cache import DATA, FOLDS, ROOT  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset  # noqa: E402
from module.eeg_encoder.model import run_eeg_backbone  # noqa: E402

OUT = f'{R}/results/things_eeg/mi/nc'
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda:0')
HELD = np.sort(np.random.default_rng(0).choice(16540, 5000, replace=False))  # = train.py held_out()
SEEN = np.sort(np.random.default_rng(1).choice(np.setdiff1d(np.arange(16540), HELD), 5000, replace=False))
MEMBERS = {'d6': ('reve_d6_ho5k_long_optx_things', 'checkpoint_ep10.pth'), 'rmix': ('rmix_reg_ho5k_long_optx_things', 'checkpoint_ep10.pth'),
           'sqfx': ('sqf_dbn_ho5k_long_optx_things', 'checkpoint_test_best.pth'), 'sqf': ('sqf_dbn_ho5k_long_things', 'checkpoint_ep32.pth'),
           'd6m': ('reve_d6_ho5k_mid_optx_things', 'checkpoint_ep10.pth'), 'rmixm': ('rmix_reg_ho5k_mid_optx_things', 'checkpoint_ep10.pth'),
           'scr': ('reve_scratch_ho5k_long_optx_things', 'checkpoint_ep10.pth')}
# ens5 at the sub-6 budgets: cached clean-proxy outputs logs/holdout/f<k>/<name>.npy (d6, rmix, SQF-X, REVE-scratch, SQF)
ENS5 = {1: ('reve_d6_ho5k_long_optx_things__ep10', 'rmix_reg_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         2: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         3: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_mid_optx_things', 'sqf_dbn_ho5k_long_things__ep32')}
TARGETS = np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy', mmap_mode='r').reshape(16540, 1536)
nz = lambda x: x / x.norm(dim=1, keepdim=True)


def gallery(images=HELD, big=False):
    g = nz(torch.tensor(np.asarray(TARGETS[images])).float())
    if big:  # + 9.4k never-trained THINGS images
        g = torch.cat([g, nz(torch.tensor(np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/things_extra.npy')).float())])
    if big == 2:  # + 25.6k images outside THINGS (COCO + ImageNet val, ny01): a 40k gallery with the sealed mix (at most 9.4k THINGS images are left, E00)
        g = torch.cat([g, nz(torch.tensor(np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/open_gallery45k.npy')).float())[:40000 - len(g)]])
    return g.to(DEV)


def cell(people, images=HELD, seed=7):
    """X (n, 32, 120) float32, subj (n,), img (n,) = position in `images`; rows permuted so 64-batches mix people."""
    X, subj, img = [], [], []
    for s in people:
        x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)[images])
        gi, rep = np.nonzero(np.abs(x).sum((2, 3)) > 0)  # missing repeats are all-zero padding
        X.append(x[gi, rep]); subj += [s] * len(gi); img += gi.tolist()
    o = np.random.default_rng(seed).permutation(len(img))
    return np.concatenate(X)[o], np.array(subj)[o], np.array(img)[o]


def key(s):
    """the person's image-independent mean ERP (32, 120), from every training-set trial (cached)."""
    p = f'{OUT}/key_sub{s:02d}.npy'
    if not os.path.exists(p):
        x = np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)
        tot, n = np.zeros((32, 120)), 0
        for b in range(0, len(x), 8192):
            c = np.asarray(x[b:b + 8192]); ok = np.abs(c).sum((1, 2)) > 0
            tot += c[ok].sum(0, dtype=np.float64); n += ok.sum()
        np.save(p, (tot / n).astype(np.float32))
    return torch.tensor(np.load(p))


def load(member, k=1, ck=None):
    """frozen member -> f(X, delta=None): outputs (n, 1536) on DEV; delta(a, b) is added to input rows a:b."""
    recipe, ck0 = MEMBERS.get(member, (member, 'checkpoint_test_best.pth'))
    ck = ck or ck0
    d = os.path.dirname(sorted(glob.glob(f'{ROOT}/{recipe}/fold{k}/*/*/train_config.json'))[-1])
    cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
    ckpt = torch.load(f'{d}/{ck}', map_location=DEV, weights_only=False)
    model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [FOLDS[k][0]]), DEV, 120)

    @torch.no_grad()
    def f(X, delta=None, bs=int(os.environ.get('BS', 512))):  # BS=128: ST-EEGFormer next to a training on a shared 10 GB GPU
        out = []
        for b in range(0, len(X), bs):
            x = torch.as_tensor(np.asarray(X[b:b + bs])).to(DEV)
            if delta is not None:
                x = x + delta(b, b + bs)
            out.append(eproj(run_eeg_backbone(model, cfg, x, torch.full((len(x),), 10 ** 6, device=DEV))).float().cpu())  # outputs stay on the CPU: the GPU is shared with training runs
        return torch.cat(out)
    f.model, f.eproj, f.cfg = model, eproj, cfg
    return f


def scaled(P):
    return torch.cat([P[b:b + 64] / P[b:b + 64].norm(dim=1).mean() for b in range(0, len(P), 64)])


def rank(q, t, gal, legacy=False):
    """number of gallery items scoring above the target. The target's own score is read from the same matrix product:
    recomputing it as a separate dot product (legacy=True, as logs/holdout_eval.py does) differs in the last float bits
    and counts the target above itself for part of the queries."""
    q, t, out = nz(q.float()), torch.as_tensor(t), []
    for b in range(0, len(q), 4096):
        c, tt = q[b:b + 4096].to(DEV), t[b:b + 4096].to(DEV)
        S = c @ gal.T
        tgt = (c * gal[tt]).sum(1, keepdim=True) if legacy else S.gather(1, tt[:, None])
        out.append((S > tgt).sum(1).cpu())
    return torch.cat(out)


def score(y, subj, img, gal, hits=False):
    """top-5 % at r = 1, 3, 4 (r >= 3 on the (person, image) groups with 4 repeats). hits=True: also the r=3 / r=4 hit vectors."""
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4]))
    t1 = torch.as_tensor(img); t4 = t1[g4[:, 0]]
    y = scaled(y.cpu())
    h1, h3, h4 = rank(y, t1, gal) < 5, rank(y[g4[:, :3]].mean(1), t4, gal) < 5, rank(y[g4].mean(1), t4, gal) < 5
    out = dict(r1=100 * h1.float().mean().item(), r3=100 * h3.float().mean().item(), r4=100 * h4.float().mean().item())
    return (out, dict(r3=h3.cpu().numpy(), r4=h4.cpu().numpy(), subj=np.asarray(subj)[g4[:, 0].numpy()])) if hits else out


def paired(ha, hb, n=2000, seed=0):
    """relative gain (%) of hit vector ha over hb and its bootstrap SE (groups resampled)."""
    i = np.random.default_rng(seed).integers(0, len(ha), (n, len(ha)))
    g = 100 * (ha[i].mean(1) / hb[i].mean(1) - 1)
    return 100 * (ha.mean() / hb.mean() - 1), g.std()


def cohort(name, n=6000):
    """exported external cohort (nq06 / nq10), known time shift applied -> X, subj, img (rows of its targets.npz), gallery, r=3 groups (every 3 trials of a (person, image)); rows permuted so 64-batches mix people"""
    D = f'/nasbrain/p20fores/{name}'; dd = [np.load(p) for p in sorted(glob.glob(f'{D}/export/sub-*.npz'))]
    X = torch.tensor(np.concatenate([d['X'][:n] for d in dd]).astype(np.float32))[:, :, (torch.arange(120) - {'ds004018': 5, 'ds007964': 4}[name]).clamp(0, 119)]
    img = np.concatenate([d['img'][:n] for d in dd]); subj = np.concatenate([[j] * len(d['img'][:n]) for j, d in enumerate(dd)]); o = np.random.default_rng(0).permutation(len(X)); X, img, subj = X[o], img[o], subj[o]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    return X, subj, img, nz(torch.tensor(np.load(f'{D}/targets.npz')['F'])).to(DEV), torch.as_tensor(np.array([v[i:i + 3] for v in groups.values() for i in range(0, len(v) - 2, 3)]))
