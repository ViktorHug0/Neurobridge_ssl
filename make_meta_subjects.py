"""Build "meta-subjects": EEG1 subjects pooled in groups of 4 by averaging same-stimulus trials.

Trades cohort size for per-trial SNR. EEG1 shows each training image once per subject, and the
subject-count scaling curve on EEG1 was nearly flat (4.0 -> 7.7 over N=1..8 vs EEG2's 7.0 -> 35.6),
which points at per-subject SNR rather than cohort size as the binding constraint. Averaging 4
subjects on the same stimulus is the direct test: 4x the trials per image, 1/4 the subjects.

  36 good subjects -> 9 meta-subjects of 4. Two different shufflings -> 18 meta-subjects total,
  which is what makes meta-subject-to-meta-subject variability measurable.

Training side is clean: every subject's training array is aligned to the same 16,540 EEG2 training
images, so averaging is elementwise. Trials that a subject never saw were filled with that
subject's concept-mean by preprocess_eeg1.py and are flagged in `missing_mask`; those are excluded
from the average so a surrogate never dilutes real trials.

Test side needs care: subjects saw DIFFERENT validation images (no image is held by all 36, and a
random group of 4 shares only ~70 of the 200). Restricting to a group's intersection would give
each meta-subject its own 59-84-way retrieval task and make the accuracies incomparable -- which
would defeat the point. Instead every meta-subject keeps the SAME 200 images (the order of
val_image_set/order.txt, so the existing 200-row feature bank applies unchanged) and each image is
averaged over whichever group members saw it. Averaging depth therefore varies per image (mean
~3.1 of 4); per-meta-subject depth is reported in groups.json so it can be checked as a covariate.

Usage:  python make_meta_subjects.py --seed 0 --out_dir <dir>
"""
import argparse
import json
import os

import numpy as np

SRC = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Preprocessed_data_250Hz"
ORDER = "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg1/val_image_set/order.txt"


def load(sub, split):
    return np.load(f"{SRC}/sub-{sub:02d}/preprocessed_eeg_{split}.npy", allow_pickle=True).item()


def build_train(group):
    """(16540, 1, 63, 250) averaged over the members that genuinely saw each image."""
    total = count = None
    for sub in group:
        d = load(sub, "training")
        x, seen = d["preprocessed_eeg_data"], ~d["missing_mask"]
        if total is None:
            total = np.zeros_like(x, dtype=np.float64)
            count = np.zeros(x.shape[0], dtype=np.int32)
            meta = {"ch_names": d["ch_names"], "times": d["times"]}
        total[seen] += x[seen]
        count += seen
        del d, x
    np.divide(total, np.maximum(count, 1)[:, None, None, None], out=total)
    return total.astype(np.float32), count, meta


def build_test(group, order):
    """(200, 12, 63, 250) on the shared image order, averaged over the members that saw each."""
    row = {f: i for i, f in enumerate(order)}
    total = count = None
    for sub in group:
        d = load(sub, "test")
        x = d["preprocessed_eeg_data"]
        idx = [row[f] for f in d["image_files"].tolist()]
        if total is None:
            total = np.zeros((len(order),) + x.shape[1:], dtype=np.float64)
            count = np.zeros(len(order), dtype=np.int32)
        total[idx] += x
        count[idx] += 1
        del d, x
    np.divide(total, np.maximum(count, 1)[:, None, None, None], out=total)
    return total.astype(np.float32), count


def main(args):
    good = [int(x) for x in open(args.subjects_file).read().split()]
    order = open(ORDER).read().split()
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(good)
    groups = [sorted(int(s) for s in shuffled[i * args.group_size:(i + 1) * args.group_size])
              for i in range(args.n_groups)]

    manifest = {"seed": args.seed, "source": SRC, "groups": {}}
    for g, group in enumerate(groups, start=1):
        out = os.path.join(args.out_dir, f"sub-{g:02d}")
        os.makedirs(out, exist_ok=True)
        train, train_depth, meta = build_train(group)
        np.save(f"{out}/preprocessed_eeg_training.npy",
                {"preprocessed_eeg_data": train, "missing_mask": train_depth == 0, **meta})
        del train
        test, test_depth = build_test(group, order)
        np.save(f"{out}/preprocessed_eeg_test.npy",
                {"preprocessed_eeg_data": test, "image_files": np.array(order), **meta})
        del test
        manifest["groups"][f"sub-{g:02d}"] = {
            "members": group,
            "train_mean_depth": round(float(train_depth.mean()), 3),
            "test_mean_depth": round(float(test_depth.mean()), 3),
            "test_images_unseen_by_group": int((test_depth == 0).sum()),
        }
        print(f"sub-{g:02d} <- {group}  train_depth={train_depth.mean():.2f} "
              f"test_depth={test_depth.mean():.2f} unseen={(test_depth == 0).sum()}", flush=True)

    with open(os.path.join(args.out_dir, "groups.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {args.n_groups} meta-subjects -> {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--subjects_file", default="/nasbrain/p20fores/Neurobridge_SSL/eeg1_good_subjects.txt")
    p.add_argument("--n_groups", type=int, default=9)
    p.add_argument("--group_size", type=int, default=4)
    main(p.parse_args())
