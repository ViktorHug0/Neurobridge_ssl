"""Q07 are ds007964 people decodable by our members, zero-shot, and at which time shift? (nq03 for a cohort on THINGS images)
Exported people (nq06) through a fold-1 member, input shifted by s samples (content later by s, edge replicated).
Gallery = the 1,900 images of the cohort. Single trial, and the (up to 3) trials of a (person, image) averaged.
    nq07_ds007964_zeroshot.py <sub,sub,...> [member recipe:checkpoint] [shifts, e.g. -8:9:2]"""
from nc_lib import *

D = os.environ.get('COHORT', '/nasbrain/p20fores/ds007964'); G = nz(torch.tensor(np.load(f'{D}/targets.npz')['F'])).to(DEV)
recipe, ck = (sys.argv[2] if len(sys.argv) > 2 else 'mi_rmix_ho5k_lag5:checkpoint_ep10.pth').split(':'); f = load(recipe, 1, ck)
a, b, c = (int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else '-8:9:2').split(':')); t = torch.arange(120)


def stats(y, img):
    S = nz(y.float()).to(DEV) @ G.T; tg = S.gather(1, img.to(DEV)[:, None]); return 100 * ((S > tg).sum(1) < 5).float().mean().item(), (S < tg).float().mean().item()


print(recipe, ck, '| chance: top-5 0.26 %, P .5')
for s_ in range(a, b, c):
    line = []
    for s in sys.argv[1].split(','):
        d = np.load(f'{D}/export/sub-{s}.npz'); img = torch.tensor(d['img']); y = scaled(f(torch.tensor(d['X'].astype(np.float32))[:, :, (t - s_).clamp(0, 119)], bs=1000))
        m = torch.zeros(len(G), 1536).index_add_(0, img, y); line.append('%s: %.2f %% / %.3f, averaged %.2f %% / %.3f' % (s, *stats(y, img), *stats(m, torch.arange(len(G)))))
    print('  shift %+3d (%+4.0f ms) top-5 / P(target > random image): %s' % (s_, 1000 * s_ / 120, '   '.join(line)), flush=True)
