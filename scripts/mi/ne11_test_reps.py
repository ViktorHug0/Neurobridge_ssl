"""E11 why are the 200 test-image trials harder than held-out training-split images? r=1 top-5 of ens5 (REVE@10, fold k)
on the test trials by repetition index (repetition suppression / session effect vs concept novelty), 5.2k gallery."""
from nc_lib import *
k = int(sys.argv[1]); L = ('long', 'mid')[k != 1]
names = [f'reve_d6_ho5k_{L}_optx_things_ep10', f'rmix_reg_ho5k_{L}_optx_things_ep10', 'sqf_dbn_ho5k_long_optx_things_test_best', f'reve_scratch_ho5k_{L}_optx_things_' + ('ep10' if k == 1 else 'test_best'), 'sqf_dbn_ho5k_long_things_ep32']
subj = np.repeat(FOLDS[k], 16000); img = np.tile(np.repeat(np.arange(200), 80), 5); rep = np.tile(np.arange(80), 1000)
o = np.random.default_rng(11).permutation(80000); subj, img, rep = subj[o], img[o], rep[o]
y = sum(scaled(torch.as_tensor(np.load(f'{OUT}/test_f{k}_{n}.npy')).float()) for n in names)
T = nz(torch.tensor(np.load(f'{R}/data/things_eeg/image_feature/dinov2-giant_track1/image_test.npy').reshape(200, 1536)).float()).to(DEV)
h = (rank(y, torch.as_tensor(img), torch.cat([T, gallery()])) < 5).numpy()
print('r=1 top-5 %% overall %.3f; by repetition index (bins of 10): %s' % (100 * h.mean(), ' '.join('%.2f' % (100 * h[(rep >= a) & (rep < a + 10)].mean()) for a in range(0, 80, 10))))
print('per person:', ' '.join('%d: %.2f' % (s, 100 * h[subj == s].mean()) for s in FOLDS[k]))
pi = np.array([h[img == i].mean() for i in range(200)]); print('per image: %d of 200 images never hit; top 20 images carry %.0f %% of hits' % ((pi == 0).sum(), 100 * np.sort(pi)[-20:].sum() / pi.sum()))
