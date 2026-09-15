"""Image features for THINGS-EEG1's validation images, and per-subject feature dirs.

EEG1's 200 validation images are in neither EEG2 split, so no features exist for them. They are
also the only EEG1 images with repetitions (12 vs 1), which is what makes an intra-subject test
set worth running at all.

Each subject saw a different ~160 of the 200 (10 subjects share only 19), so test sets cannot be
shared: every subject gets its own feature directory holding its 150 rows, with image_train.npy
symlinked to the shared EEG2 bank (EEG1's training array is aligned to EEG2's training images).

  stage  -> unpack the 200 images into an extract_feature.py-shaped image set
  slice  -> cut the 200-row bank into per-subject image_test.npy, ordered to match each
            subject's preprocessed_eeg_test.npy["image_files"]
"""
import argparse
import os

import numpy as np

# data/images_THINGS.zip is password-protected; the THINGS-MEG copy of the same 22,248-image pool
# is already unpacked and holds all 200 validation images.
POOL = "/nasbrain/p20fores/Neurobridge_SSL/data/things_meg/image_set/object_images"
EEG2_BANK = ("/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/"
             "InternViT-6B_layer28_mean_8bit")


def concept_of(fname):
    return fname.rsplit(".", 1)[0].rsplit("_", 1)[0]


def val_union(eeg_dirs):
    """The union of validation images, read back from the exported test sets (single source of
    truth: preprocess_eeg1.py already pinned each subject's row order)."""
    files = set()
    for d in eeg_dirs:
        p = os.path.join(d, "preprocessed_eeg_test.npy")
        if os.path.isfile(p):
            files |= set(np.load(p, allow_pickle=True).item()["image_files"].tolist())
    return sorted(files)


def stage(args):
    dirs = sorted(os.path.join(args.eeg_dir, d) for d in os.listdir(args.eeg_dir) if d.startswith("sub-"))
    images = val_union(dirs)
    if not images:
        raise SystemExit(f"no preprocessed_eeg_test.npy found under {args.eeg_dir}")
    def link(fname, d):
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, fname)
        if not os.path.exists(out):
            os.symlink(os.path.join(POOL, concept_of(fname), fname), out)

    test_root = os.path.join(args.image_set_dir, "test_images")
    for i, fname in enumerate(images):
        link(fname, os.path.join(test_root, f"{i:04d}_{concept_of(fname)}"))
    # extract_feature.py always walks training_images too; give it one cheap concept whose
    # output we discard, rather than re-extracting all 16,540 EEG2 training images.
    for fname in images[:10]:
        link(fname, os.path.join(args.image_set_dir, "training_images", "0000_dummy"))
    with open(os.path.join(args.image_set_dir, "order.txt"), "w") as fh:
        fh.write("\n".join(images))
    print(f"staged {len(images)} validation images -> {test_root}")


def slice_per_subject(args):
    order = open(os.path.join(args.image_set_dir, "order.txt")).read().split()
    row_of = {f: i for i, f in enumerate(order)}
    bank = np.load(os.path.join(args.feature_dir, "image_test.npy"))  # (n_val, 1, dim)
    if bank.shape[0] != len(order):
        raise SystemExit(f"feature bank has {bank.shape[0]} rows, staged order has {len(order)}")

    n = 0
    for sub in sorted(d for d in os.listdir(args.eeg_dir) if d.startswith("sub-")):
        p = os.path.join(args.eeg_dir, sub, "preprocessed_eeg_test.npy")
        if not os.path.isfile(p):
            continue
        files = np.load(p, allow_pickle=True).item()["image_files"].tolist()
        out_dir = os.path.join(args.out_dir, sub)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "image_test.npy"), bank[[row_of[f] for f in files]])
        link = os.path.join(out_dir, "image_train.npy")
        if not os.path.islink(link) and not os.path.isfile(link):
            os.symlink(os.path.join(EEG2_BANK, "image_train.npy"), link)
        n += 1
    print(f"wrote per-subject feature dirs for {n} subjects -> {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage_name", choices=["stage", "slice"])
    p.add_argument("--eeg_dir", default="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Preprocessed_data_250Hz")
    p.add_argument("--image_set_dir", default="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg1/val_image_set")
    p.add_argument("--feature_dir", default="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg1/image_feature/InternViT-6B_layer28_mean_8bit")
    p.add_argument("--out_dir", default="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg1/image_feature/InternViT-6B_layer28_mean_8bit/per_subject")
    args = p.parse_args()
    (stage if args.stage_name == "stage" else slice_per_subject)(args)
