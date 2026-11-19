"""Shared helpers of the subject-key experiments (scripts/subject_key, log KEY_LOG.md). Builds on scripts/mi:
x = m_s + c  with  m_s = the subject's image-independent mean ERP (the "key").
Substitution used everywhere (label-free): ATM = attention map from the subject's own key, readout fed the mean training
key K (from training images); other encoders = key moved halfway to K."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mi'))
_argv, sys.argv = sys.argv, sys.argv[:1]  # the scripts/mi modules read sys.argv at import
from mi_lib import *  # noqa: F401,F403,E402
from mi_lib import _DS  # noqa: E402
from e09_arch import GM, LOSO  # noqa: E402
from e10_keysub import train_key  # noqa: E402
from e13_atm_attention import emb as atm_emb  # noqa: E402  (patches FullAttention.forward: record / force the map)
sys.argv = _argv

SK = os.path.join(REPO, 'results/things_eeg/subject_key')
# member pools of the score ensembles (K01, K12)
POOLS = {
    'vc': ['atm128_vc/seed3300', 'atm128_vc/seed3301', 'atm128_vc/seed3302', 'tsconv_vc/seed3300', 'labram_vc/seed3300',
           'eegproject_vc/seed3300', 'conformer_vc/seed3300'],
    'vc_img': ['atm128_vc/seed3300', 'atm128_vc/seed3302', 'atm128_bigg27_vc/seed3300', 'atm128_iv20_vc/seed3300', 'labram_vc/seed3300', 'tsconv_vc/seed3300'],  # + ATM on other image spaces (K12 only)
    'testsel': ['atm128/seed3300', 'tsconv/seed3300', 'sqf/seed3300', 'rmix/seed3300', 'labram/seed3300', 'eegproject/seed3300'],
}
nz = torch.nn.functional.normalize


def load(run, fold, ckpt='checkpoint_test_best.pth'):
    """run = a directory holding one run per fold (cvpr/loso/<arm>/<seed> or an absolute path).
    In validation-concept runs checkpoint_test_best.pth is the VALIDATION-selected epoch and checkpoint_testsel_best.pth the test-selected one."""
    root = run if os.path.isabs(run) else f'{LOSO}/{run}'
    d = glob.glob(f'{root}/*-sub-{fold:02d}')[0]
    feat = json.load(open(f'{d}/train_config.json'))['image_feature_dir'].rstrip('/')
    if _DS.get('img_dir') != feat:  # mi_lib's image arrays are those of the model being loaded (arms differ in image space)
        for split in ('train', 'test'):
            x = np.load(f'{feat}/image_{split}.npy')
            _DS[('img', split)] = x.reshape(-1, x.shape[-1])
        _DS['img_dir'] = feat
    return GM(d, ckpt)


def mean_key(m):
    return torch.stack([train_key(t) for t in m.train_subs]).mean(0, keepdim=True)


def embed_sub(m, x, sub, mu, K, lam=0.5):
    """Embedding of trials x of a subject whose key is mu, under the substitution."""
    if m.args.eeg_encoder_type == 'ATM':
        return atm_emb(m, x - mu + K, sub, atm_emb(m, mu, sub)[1])[0]
    return m.embed(x + lam * (K - mu), sub)


def member_scores(m, fold):
    """Cosine score matrices (200 queries x 200 images) of the held-out subject: own key / substituted.
    A model trained without the key (train.py --subject_key demean | split) has one way to be fed: both entries are it."""
    g, x = img_emb(m), T(eeg(fold))
    mu, K = x.mean(0, keepdim=True), mean_key(m)
    mode = getattr(m.args, 'subject_key', 'keep')
    if mode in ('demean', 'split'):
        s = nz(m.embed(x - mu if mode == 'demean' else torch.stack([x - mu, mu.expand_as(x)], 1), fold), dim=1) @ g.T
        return s, s
    return nz(m.embed(x, fold), dim=1) @ g.T, nz(embed_sub(m, x, fold, mu, K), dim=1) @ g.T


def calibrated(m, fold, anchor='img'):
    """Score matrix after the anchored output calibration (KEY_LOG K10): the subject's normalised embeddings are shifted
    so that their mean is the mean image embedding of the training images (anchor='img'), or, for a model trained with
    --person_centre_train (centred in the raw output space around the training subjects' mean), the mean raw embedding
    of the training subjects on 1,500 training images (anchor='eeg_raw')."""
    g, x = img_emb(m), T(eeg(fold))
    mode = getattr(m.args, 'subject_key', 'keep')
    mu = x.mean(0, keepdim=True)
    feed = lambda v, k: v - k if mode == 'demean' else torch.stack([v - k, k.expand_as(v)], 1) if mode == 'split' else v
    e = m.embed(feed(x, mu), fold)
    if anchor == 'eeg_raw':
        im = np.sort(np.random.default_rng(0).choice(16540, 1500, replace=False))
        ref = torch.cat([m.embed(feed(T(eeg(s, 'train')[im]), train_key(s)[None]), s) for s in m.train_subs])
        return nz(e - e.mean(0) + ref.mean(0), dim=1) @ g.T
    e = nz(e, dim=1)
    return nz(e - e.mean(0) + img_emb(m, 'train').mean(0), dim=1) @ g.T


def acc(S):
    return 100 * (S.argmax(-1) == torch.arange(S.shape[-1])).float().mean(-1)


def key_estimators(m, K, reps=80, n_img=400):
    """In-trial estimators of a subject's key deviation: name -> function of d = x - K, shape (n, 63, 250) (KEY_LOG K36, K46).
      comb  the 200 ms-periodic part of d without its temporal mean (no parameter)
      time  Wiener filter over time, shared by the electrodes
      kron  Wiener filter with separable (electrodes x time) covariances
    Wiener statistics from the training subjects only: key deviations around K against the content of a training trial
    at `reps` repetitions (signal part from the agreement of two 2-repetition halves, noise part scaled 1 / reps)."""
    im = np.sort(np.random.default_rng(0).choice(16540, n_img, replace=False))
    cov = lambda a, b, ax: torch.einsum('nct,ncu->tu' if ax == 't' else 'nct,ndt->cd', a, b) / len(a)
    psd = lambda S: (lambda w, V: V * w.clamp(min=0) @ V.T)(*torch.linalg.eigh(S))
    S = {ax: dict(d=0, s=0, h=0) for ax in 'tc'}
    for s in m.train_subs:
        d = (train_key(s) - K[0]).double()[None]
        h = T(eeg(s, 'train', average=False)[im]).double()
        h1, h2 = h[:, :2].mean(1) - train_key(s), h[:, 2:].mean(1) - train_key(s)
        for ax in 'tc':
            S[ax]['d'] += cov(d, d, ax) / len(m.train_subs)
            S[ax]['s'] += (cov(h1, h2, ax) + cov(h2, h1, ax)) / (2 * len(m.train_subs))
            S[ax]['h'] += (cov(h1, h1, ax) + cov(h2, h2, ax)) / (2 * len(m.train_subs))
    Sd = {ax: S[ax]['d'] for ax in 'tc'}
    Sc = {ax: psd(S[ax]['s']) + 2 * psd(S[ax]['h'] - S[ax]['s']) / reps for ax in 'tc'}
    eye = lambda M: 1e-6 * M.trace() / len(M) * torch.eye(len(M), dtype=torch.double)
    At = (Sd['t'] @ torch.linalg.inv(Sd['t'] + Sc['t'] + eye(Sd['t']))).float().T
    W, U, lam = {}, {}, {}
    for ax in 'tc':  # whiten the content, diagonalise the key deviations: gain = l_c l_t / (l_c l_t + tr_d / tr_c)
        w, V = torch.linalg.eigh(Sc[ax] + eye(Sc[ax]))
        W[ax] = (V * w.rsqrt() @ V.T, V * w.sqrt() @ V.T)
        lam[ax], U[ax] = torch.linalg.eigh(W[ax][0] @ Sd[ax] @ W[ax][0])
    G = torch.outer(lam['c'], lam['t']).clamp(min=0)
    G = (G / (G + Sd['c'].trace() / Sc['c'].trace())).float()
    Lc, Lt = (U['c'].T @ W['c'][0]).float(), (W['t'][0] @ U['t']).float()  # into the joint basis
    Rc, Rt = (W['c'][1] @ U['c']).float(), (U['t'].T @ W['t'][1]).float()  # and back

    def comb(d):
        per = d.unflatten(-1, (5, 50)).mean(-2)
        return (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)
    return {'comb': comb, 'time': lambda d: d @ At, 'kron': lambda d: Rc @ (G * (Lc @ d @ Lt)) @ Rt}
