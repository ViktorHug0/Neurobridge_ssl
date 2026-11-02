"""Are R3 distractors harder because they are TRAINING images (the encoder was trained to pull EEG towards them)?
ValCon encoders never used the 165 held-out validation concepts as targets (train.py: permutation of the sorted
concepts with the fixed seed 20260822, first 10%). Two distractor pools of equal size, pushed through the
subject's frozen image projector:
  novel   all 10 images of the 165 held-out concepts (1650 images, never targets), shuffled
  train   image 0 of 1650 non-held training concepts (targets during training), shuffled
Methods per D in {0, 200, 400, 800, 1650}: plain, SCORE, GALA, fusion G+3C1, oracle fit (true support, full gallery).
usage: novel_distractors.py --seed 3301
"""
import argparse, csv, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from open_gallery_diag import C, R  # noqa: E402
from open_gallery_methods import loop, mm, nrm  # noqa: E402
from agnostic import zrow  # noqa: E402
from shared import fit_saw_transform, load_subject_context, score_features  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402

HELD = np.sort(np.random.default_rng(20260822).permutation(np.arange(1654))[:165])


def pools(arm, seed, sid):
    f = f'{C}/cache/{arm}_s{seed}/sub-{sid:02d}.npz'
    z = dict(np.load(f))
    if 'novel' not in z:
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, _, mods = load_subject_context(f'{R}/results/things_eeg/cvpr/loso/{arm}/seed{seed}', rt, sid, average=False)
        raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy'), mmap_mode='r')
        rng = np.random.default_rng(7000 + sid)
        nov = np.asarray(raw[HELD], np.float32).reshape(-1, raw.shape[-1])[rng.permutation(1650)]
        rest = np.setdiff1d(np.arange(1654), HELD)
        trn = np.asarray(raw[rest[rng.permutation(len(rest))[:1650]], 0], np.float32)
        z['novel'], z['train'] = project_raw_images(ea, mods, nov), project_raw_images(ea, mods, trn)
        np.savez(f, **z)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='atm128_vc')
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    params_csv = f'{R}/results/things_eeg/cvpr/tta_screen_loso/{a.arm}_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'
    w = csv.writer(open(f'{C}/novel_{a.arm}_s{a.seed}.csv', 'w', newline=''))
    w.writerow(['subject', 'pool', 'D', 'method', 'top1'])
    y = np.arange(200)
    for sid in range(1, 11):
        p = subject_params(params_csv, sid)
        z = pools(a.arm, a.seed, sid)
        q, g = z['q'].astype(np.float32), z['g'].astype(np.float32)
        q_saw = nrm(fit_saw_transform(q, p)[0])
        t_or = loop(q_saw, g, p)[1]
        for pool in ('novel', 'train'):
            for D in (0, 200, 400, 800, 1650):
                if pool == 'train' and D == 0:
                    continue
                gal = np.concatenate([g, z[pool][:D].astype(np.float32)], 0)
                sg = loop(q_saw, gal, p)[0]
                res = {'plain': score_features(q, gal), 'score': score_deploy(q, gal), 'gala': sg,
                       'fuse3': zrow(sg) + 3 * zrow(loop(nrm(mm(q, gal)), gal, p, lam=1.0)[0]),
                       'oracle_fit': score_features(t_or, gal, use_csls=p.use_csls, csls_k=p.csls_k)}
                for m, s in res.items():
                    w.writerow([sid, pool, D, m, topk_acc(s, y)[1]])
        print(sid, flush=True)


if __name__ == '__main__':
    main()
