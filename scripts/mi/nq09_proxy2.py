"""Q09 second clean proxy for a cohort on THINGS images (ds007964): the 5,000 held-out images are drawn outside the cohort's
image set (train.py --holdout_avoid), so the cohort trains with all its trials. Held-out people x these never-trained images
x 4 repeats, 64-trial batches mixing the 5 people, scaled outputs; top-5 % at 5k (r=3, r=4) and with the never-trained THINGS
distractors that are not cohort images (about 14k, r=3). Members alone, paired against the first one (no ensemble swap: the
other members were trained under the first holdout).
    nq09_proxy2.py <fold> <recipe:epoch> [<recipe:epoch> ...]"""
from nc_lib import *

T = np.load('/nasbrain/p20fores/ds007964/targets.npz'); pool = np.setdiff1d(np.arange(16540), (T['obj'] * 10 + T['image'])[T['obj'] < 1654])
HELD2 = np.sort(np.random.default_rng(0).choice(pool, 5000, replace=False))  # = train.py held_out() with --holdout_avoid
D = f'{R}/data/things_eeg/image_feature/dinov2-giant_track1'; mine = set(T['names'].tolist())
keep = [os.path.basename(l.strip()) not in mine for l in open(f'{D}/things_extra_names.txt')]
gals = {'5k': gallery(HELD2), '14k': torch.cat([gallery(HELD2), nz(torch.tensor(np.load(f'{D}/things_extra.npy')[keep]).float()).to(DEV)])}
k = int(sys.argv[1]); X, subj, img = cell(FOLDS[k], HELD2); base = None
for a in sys.argv[2:]:
    recipe, e = a.split(':'); p = f'{OUT}/hold2_f{k}_{recipe}_ep{int(e):02d}_raw.npy'
    if not os.path.exists(p):
        d = os.path.dirname(sorted(glob.glob(f'{ROOT}/{recipe}/fold{k}/*/*/train_config.json'))[-1])
        ck = f'checkpoint_ep{int(e):02d}.pth'; np.save(p, load(recipe, k, ck if os.path.exists(f'{d}/{ck}') else 'checkpoint_test_best.pth')(torch.as_tensor(X), bs=1000).cpu().numpy())
    y = scaled(torch.as_tensor(np.load(p)).float()); res = {g: score(y, subj, img, gal, hits=True) for g, gal in gals.items()}
    hits = {'5k r3': res['5k'][1]['r3'], '5k r4': res['5k'][1]['r4'], '14k r3': res['14k'][1]['r3']}; base = base or hits
    print('%-34s @%-2s ' % (recipe, e) + ' | '.join('%s %.3f (%+.1f +- %.1f %%)' % (n, 100 * h.mean(), *paired(h, base[n])) for n, h in hits.items()), flush=True)
