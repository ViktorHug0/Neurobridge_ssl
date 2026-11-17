"""P06 is THINGS-EEG1 undecodable by our members, or only mis-timed? (D03 applied to the cohort that failed in October)
October: pooling the 48 THINGS-EEG1 people gave nothing and EEG1-only models were at chance on Alljoined; that was before the
5-sample lag between THINGS-EEG2 and Alljoined was known, and the members halve at a 50 ms shift.
A few EEG1 people are rebuilt from raw with the October export (grader chain, 32 Alljoined electrodes, -0.2..0.8 s, 120 Hz) and
sent, zero-shot, through a fold-1 member trained on Alljoined + lag-aligned THINGS-EEG2, at input shifts s (content moved
later by s samples, edge replicated). Never-trained images (HELD), 5k gallery: single-trial top-5 and P(target > random image),
and the average of the people's outputs per image. Reference: the fold's new Alljoined people, one trial each.
    np06_eeg1_lag.py <dir with the recovered export scripts> <EEG1 subjects, e.g. 45,10,25,33> [member recipe:checkpoint ...]"""
from nc_lib import *

sys.path.insert(0, sys.argv[1]); subs = [int(a) for a in sys.argv[2].split(',')]
members = [a.split(':') for a in sys.argv[3:]] or [['mi_rmix_ho5k_lag5', 'checkpoint_ep10.pth']]
X = {}
for s in subs:
    p = f'{OUT}/eeg1_sub{s:02d}_held.npz'
    if not os.path.exists(p):
        import export_things_eeg1_to_alljoined as ex
        x, present = ex.load(s, ex.eeg2_image_index(ex.IMAGES)[0]); np.savez(p, x=x[HELD].astype(np.float16), present=present[HELD])
    d = np.load(p); X[s] = (torch.tensor(d['x'].astype(np.float32)), torch.tensor(d['present'])); print('EEG1 sub-%02d: %d of 5000 held images present, mean|x| %.3f' % (s, d['present'].sum(), np.abs(d['x'][d['present']][:500].astype(np.float32)).mean()), flush=True)
gal = gallery(); t = torch.arange(120)


def stats(y, img):
    r = rank(scaled(y), img, gal).float(); return 100 * (r < 5).float().mean().item(), 1 - r.mean().item() / 4999


for recipe, ck in members:
    f = load(recipe, 1, ck); print(recipe, ck)
    Xa, sa, ia = cell(FOLDS[1]); y = f(Xa, bs=1000); n = torch.bincount(torch.as_tensor(ia), minlength=5000)
    first = {}  # one trial per (person, image): the EEG1 regime
    for i, kk in enumerate(zip(sa.tolist(), ia.tolist())):
        first.setdefault(kk, i)
    idx = torch.as_tensor(list(first.values())); yy, ii = scaled(y)[idx], torch.as_tensor(ia)[idx]
    m4 = torch.as_tensor(sa)[idx] != FOLDS[1][-1]; pooled = torch.zeros(5000, 1536).index_add_(0, ii[m4], yy[m4])
    print('  reference, new Alljoined people: single trial top-5 %.3f %%, P %.4f | 4 people averaged per image: top-5 %.3f %%, P %.4f' % (*stats(y, torch.as_tensor(ia)), *stats(pooled, torch.arange(5000))), flush=True)
    for s_ in range(-14, 19, 2):
        g = (t - s_).clamp(0, 119); pooled, cnt, line = torch.zeros(5000, 1536), torch.zeros(5000), []
        for s, (x, pr) in X.items():
            img = torch.nonzero(pr)[:, 0]; y = f(x[pr][:, :, g], bs=1000); line.append('%.2f / %.3f' % stats(y, img))
            pooled.index_add_(0, img, scaled(y)); cnt[img] += 1
        print('  shift %+3d samples (%+4.0f ms): per person top-5 %% / P: %s | %d people averaged: top-5 %.3f %%, P %.4f' % (s_, 1000 * s_ / 120, '  '.join(line), len(X), *stats(pooled[cnt > 0], torch.nonzero(cnt > 0)[:, 0])), flush=True)
