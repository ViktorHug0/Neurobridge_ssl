"""Q03 are MSS people decodable by our members, zero-shot, and at which time shift? (D03 / P06 applied to MSS)
Exported sessions (nq02) through a fold-1 member, input shifted by s samples (content later by s, edge replicated).
Gallery = the 10,000 MSS images (grader targets from nq01). MSS sequences hold 20 images of ONE class, so a trial's epoch
also contains same-class neighbours: P(target > random image) mixes class and image information; P(target > random image
of the SAME class) is the clean image-level read (chance .5 for both).
    nq03_mss_zeroshot.py <sub-ses,sub-ses,...> [member recipe:checkpoint] [shifts, e.g. -12:13:2]"""
from nc_lib import *

D = '/nasbrain/p20fores/mss_ds005589'; T = np.load(f'{D}/targets.npz'); G = nz(torch.tensor(T['F'])).to(DEV); cls = torch.tensor(T['cls']).to(DEV)
sess = sys.argv[1].split(','); recipe, ck = (sys.argv[2] if len(sys.argv) > 2 else 'mi_rmix_ho5k_lag5:checkpoint_ep10.pth').split(':')
a, b, c = (int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else '-12:13:2').split(':')); t = torch.arange(120)
data = {s: np.load(f'{D}/export/sub-{s.split("-")[0]}_ses-{s.split("-")[1]}.npz') for s in sess}
f = load(recipe, 1, ck); same = cls[:, None] == cls[None]; gen = torch.Generator(device=DEV).manual_seed(0)


def stats(y, img):
    q = nz(y.float()).to(DEV); img = img.to(DEV); S = q @ G.T; tg = S.gather(1, img[:, None]); m = same[img]; m[torch.arange(len(img)), img] = False
    return 100 * ((S > tg).sum(1) < 5).float().mean().item(), (S < tg).float().mean().item(), (((S < tg) & m).sum(1) / m.sum(1)).mean().item()


print(recipe, ck, '| chance: top-5 0.05 %, P .5')
for s_ in range(a, b, c):
    g = (t - s_).clamp(0, 119); line = []; pooled, cnt = torch.zeros(10000, 1536), torch.zeros(10000)
    for s, d in data.items():
        x = torch.tensor(d['X'].astype(np.float32))[:, :, g]; img = torch.tensor(d['img']); y = scaled(f(x, bs=1000))
        line.append('%s: %.2f %% / %.3f / %.3f' % (s, *stats(y, img))); pooled.index_add_(0, img, y); cnt.index_add_(0, img, torch.ones(len(img)))
    print('  shift %+3d (%+4.0f ms) single trial top-5 / P(any image) / P(same class): %s | all trials of an image averaged (%.1f per image): %.2f %% / %.3f / %.3f'
          % (s_, 1000 * s_ / 120, '   '.join(line), cnt[cnt > 0].mean(), *stats(pooled[cnt > 0], torch.nonzero(cnt > 0)[:, 0])), flush=True)
