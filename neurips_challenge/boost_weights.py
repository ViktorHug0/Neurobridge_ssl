"""AUG_LOG boosting: per training trial, how well the base members rank the TRUE image: p_true = 1 - rank / n_images over
the whole training gallery (1 = top), from teachers/fold<k>_4m.npy -> fold<k>_4m_ptrue.npy (N x 1 float32;
train.py --boost_gamma reads a 1-column --distill_targets as p_true, weight (1 - p_true)^gamma = (rank / n)^gamma).
(Softmax probabilities were flat: median 0.005, mean (1-p)^2 0.983 -> no reweighting.)
Row order = train.py's (subject, object, image, rep): true image = (row // reps) % n_images.
    boost_weights.py <fold> [tau=0.02] [--image]   (--image: each trial gets its IMAGE's mean hardness over all its trials
                                                  -> fold<k>_4m_pimg.npy; per-trial hardness is mostly single-trial noise)"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

IMAGE = '--image' in sys.argv
argv = [a for a in sys.argv if a != '--image']
K, TAU, BATCH = argv[1], float(argv[2]) if len(argv) > 2 else 0.02, 512
R = 'results/alljoined/track1/cv4/teachers'
T = np.load(f'{R}/fold{K}_4m.npy', mmap_mode='r')
G = np.load('data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy').reshape(-1, 1536)
n_img = len(G)
reps = len(T) // (25 * n_img)
assert reps * 25 * n_img == len(T), (len(T), n_img)
g = F.normalize(torch.as_tensor(G, device='cuda').float(), dim=1)
out = np.empty((len(T), 1), np.float32)
for b in range(0, len(T), 8192):
    t = F.normalize(torch.as_tensor(np.asarray(T[b:b + 8192]), device='cuda').float(), dim=1)
    logits = t @ g.T / TAU
    true = torch.as_tensor((np.arange(b, b + len(t)) // reps) % n_img, device='cuda')
    rank = (logits > logits.gather(1, true[:, None])).sum(1, keepdim=True).float()
    out[b:b + len(t)] = (1 - rank / n_img).cpu().numpy()
if IMAGE:  # mean over every trial (all subjects, repeats) of the same image
    img = (np.arange(len(out)) // reps) % n_img
    out[:, 0] = (np.bincount(img, out[:, 0], n_img) / np.bincount(img, minlength=n_img))[img]
np.save(f'{R}/fold{K}_4m_' + ('pimg' if IMAGE else 'ptrue') + '.npy', out)
print(f'fold {K}: {len(out)} rows, reps {reps}; rank/n quantiles 10/50/90 %: {np.percentile(1 - out, [10, 50, 90]).round(3)}; '
      f'mean weight (1-p)^2 {((1 - out) ** 2).mean():.3f}')
