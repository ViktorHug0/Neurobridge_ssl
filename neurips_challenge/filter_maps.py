"""SUBJECT_LOG D9: structure of the per-person 40x40 filter-mixing maps with 11 people per base model (D5 had 5).
ftor_s11_hconv fine-tunes only TSConv's 1x1 `projection` conv of the s11 tsc_dbn40_things split model on each unseen
person's own training trials; dM_s = M_s - M0 (1600-d). Per split (one filter basis):
  gain     top-5 of the adapted model vs the base model on that person
  |dM|/|M0|, rank: energy share of dM's top 1 / 4 / 8 singular directions
  shared   |mean dM|^2 / mean |dM|^2  (chance 1/n for unrelated maps)
  LOO      share of a held-out person's dM inside the top-j PCs of the OTHER people's dMs, same split, vs the same
           projection onto the other split's people (unrelated filter basis = null with matched statistics)
    filter_maps.py
"""
import glob, json
import numpy as np
import torch

R = 'results/alljoined/track1'
SPLITS = {1: [2, 3, 4, 5, 7, 11, 14, 15, 16, 17, 20], 2: [1, 2, 3, 6, 8, 9, 10, 12, 13, 18, 19]}
W = lambda path: torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']['projection.weight'].squeeze().double().numpy()


def pc_share(x, others, j):
    """share of |x|^2 inside the top-j principal directions (uncentred) of the rows of `others`"""
    V = np.linalg.svd(others, full_matrices=False)[2][:j]
    return float(np.sum((V @ x) ** 2) / np.sum(x ** 2))


def main():
    dM, info = {}, {}
    for k, subs in SPLITS.items():
        base = glob.glob(f'{R}/s11/tsc_dbn40_things/fold{k}/tsc/2026*')[0]
        M0 = W(f'{base}/checkpoint_sealed_best.pth')
        b_score = json.load(open(f'{base}/track1_score.json'))['subject_agg_by_subject']
        rows = []
        for s in subs:
            run = glob.glob(f'{R}/cv4/ftor_s11_hconv/fold{k}/s{s}/tsc/2026*/track1_score.json')
            if not run:
                continue
            d = W(run[0].replace('track1_score.json', 'checkpoint_sealed_best.pth')) - M0
            sv = np.linalg.svd(d, compute_uv=False) ** 2
            rows.append(d.ravel())
            info[(k, s)] = dict(gain=100 * (json.load(open(run[0]))['subject_agg']['top5'] - b_score[str(s)]['top5']),
                                rel=np.linalg.norm(d) / np.linalg.norm(M0), top=[sv[:j].sum() / sv.sum() for j in (1, 4, 8)])
        dM[k] = np.array(rows)
        n = len(rows)
        if n < 3:
            print(f'split {k}: {n} people, skipped'); continue
        g = [info[(k, s)] for s in subs if (k, s) in info]
        print(f'split {k}: {n} people  gain {np.mean([x["gain"] for x in g]):+.1f} ({sum(x["gain"] > 0 for x in g)}/{n})  '
              f'|dM|/|M0| {np.mean([x["rel"] for x in g]):.2f}  top1/4/8 energy ' +
              ' / '.join(f'{100 * np.mean([x["top"][i] for x in g]):.0f}%' for i in range(3)))
        D = dM[k]
        print(f'  shared |mean|^2/mean|dM|^2 {np.sum(D.mean(0) ** 2) / np.mean(np.sum(D ** 2, 1)):.3f} (chance {1 / n:.3f})')
        C = D / np.linalg.norm(D, axis=1, keepdims=True); cs = C @ C.T
        print(f'  pairwise cosine mean {cs[np.triu_indices(n, 1)].mean():.3f}  max {cs[np.triu_indices(n, 1)].max():.3f}')
    if all(len(dM[k]) >= 3 for k in SPLITS):
        for k in SPLITS:
            D, other = dM[k], dM[3 - k]
            for j in (1, 2, 4, 8):
                if j >= len(D):
                    continue
                same = np.mean([pc_share(D[i], np.delete(D, i, 0), j) for i in range(len(D))])
                null = np.mean([pc_share(D[i], other, j) for i in range(len(D))])
                print(f'  LOO split {k} top-{j} PCs of the others: same basis {100 * same:.1f}%  other split (null) {100 * null:.1f}%')


if __name__ == '__main__':
    x = np.array([1., 0, 0]); O = np.array([[2., 0, 0], [0, 1, 0]])
    assert abs(pc_share(x, O, 1) - 1) < 1e-9 and abs(pc_share(np.array([0., 0, 1]), O, 2)) < 1e-9
    main()
