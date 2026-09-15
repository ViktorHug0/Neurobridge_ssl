"""Rebuild meta-subject TEST sets on the stimuli shared by all 4 group members.

The union test set built by make_meta_subjects.py keeps all 200 validation images and averages each
over whichever members saw it. That has two defects the intersection removes:

  * 1-10 images per meta-subject were seen by NO member -- those gallery rows are all zeros.
  * test averaging depth is 3.0 while training depth is 3.95, so the model is scored on inputs
    shallower than anything it trained on.

Here every image is averaged over exactly 4 members, matching training. Each group's intersection
differs (64-78 images on shuffle 0), so galleries are subsampled to a fixed K: the retrieval task
is then literally K-way for every meta-subject and the accuracies are comparable with no
difficulty correction. Different meta-subjects still get different IMAGES, which adds some
gallery-composition variance -- read the spread with that in mind.

Training files are symlinked, not rebuilt: the training side is unchanged, so the models trained by
run_meta_subjects.sh are re-scored as-is with no retraining.

Usage:  python make_meta_test_intersection.py --shuffle 0 --k 60
"""
import argparse
import json
import os

import numpy as np

SRC = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Preprocessed_data_250Hz"
META = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Meta_subjects"
BANK = ("/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg1/image_feature/"
        "InternViT-6B_layer28_mean_8bit")
ORDER = "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg1/val_image_set/order.txt"


def member_test(sub):
    d = np.load(f"{SRC}/sub-{sub:02d}/preprocessed_eeg_test.npy", allow_pickle=True).item()
    return d, {f: i for i, f in enumerate(d["image_files"].tolist())}


def main(args):
    groups = json.load(open(f"{META}/shuffle{args.shuffle}/groups.json"))["groups"]
    order = open(ORDER).read().split()
    bank_row = {f: i for i, f in enumerate(order)}
    bank = np.load(f"{BANK}/image_test.npy")
    out_root = f"{META}/shuffle{args.shuffle}_isect"
    feat_root = f"{BANK}/isect_shuffle{args.shuffle}"

    # Fail before writing anything if K is too large for any group.
    members = {g: v["members"] for g, v in groups.items()}
    shared = {}
    for g, subs in members.items():
        sets = [set(np.load(f"{SRC}/sub-{s:02d}/preprocessed_eeg_test.npy",
                            allow_pickle=True).item()["image_files"].tolist()) for s in subs]
        shared[g] = sorted(set.intersection(*sets))
        print(f"{g} {subs} intersection {len(shared[g])}", flush=True)
    tight = {g: len(v) for g, v in shared.items() if len(v) < args.k}
    if tight:
        raise SystemExit(f"--k {args.k} exceeds the intersection of {tight}")

    rng = np.random.default_rng(args.seed)
    manifest = {"shuffle": args.shuffle, "k": args.k, "seed": args.seed, "groups": {}}
    for g, subs in members.items():
        images = sorted(rng.choice(shared[g], size=args.k, replace=False).tolist())
        total = None
        for s in subs:
            d, row = member_test(s)
            x = d["preprocessed_eeg_data"]
            if total is None:
                total = np.zeros((args.k,) + x.shape[1:], dtype=np.float64)
                meta = {"ch_names": d["ch_names"], "times": d["times"]}
            total += x[[row[f] for f in images]]
            del d, x
        out = os.path.join(out_root, g)
        os.makedirs(out, exist_ok=True)
        np.save(f"{out}/preprocessed_eeg_test.npy",
                {"preprocessed_eeg_data": (total / len(subs)).astype(np.float32),
                 "image_files": np.array(images), **meta})
        link = f"{out}/preprocessed_eeg_training.npy"
        if not os.path.exists(link):
            os.symlink(f"{META}/shuffle{args.shuffle}/{g}/preprocessed_eeg_training.npy", link)

        fdir = os.path.join(feat_root, g)
        os.makedirs(fdir, exist_ok=True)
        np.save(f"{fdir}/image_test.npy", bank[[bank_row[f] for f in images]])
        flink = f"{fdir}/image_train.npy"
        if not os.path.exists(flink):
            os.symlink(f"{BANK}/image_train.npy", flink)

        manifest["groups"][g] = {"members": subs, "intersection": len(shared[g]), "k": args.k}
        print(f"{g} <- {subs}  {args.k}-way from {len(shared[g])} shared images", flush=True)

    with open(os.path.join(out_root, "groups.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {len(members)} intersection test sets -> {out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shuffle", type=int, required=True)
    p.add_argument("--k", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
