"""Caption a test EEG trial with the caption of its nearest TRAINING image. No learned map.

Every mapped route so far (Q-Former 32x768, CLIP 1024) collapses because the map is fit on image
embeddings and EEG is off-manifold. This skips the map entirely and stays in the one space we know
works -- the 512-d shared space, where this EEG hits 81% top1 -- asking only: do the concepts the
EEG lands near carry the right words?

The bank is the 16540 TRAINING captions. Training concepts (1654) are disjoint from the 200 test
concepts, so a retrieved caption can never be the reference: unlike caption_eval's `retrieval` arm,
this cannot copy its own answer and has to generalise.

  .venv/bin/python caption_nn.py
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import caption_bridge as cb
from clip_space_probe import clip_feats

CKPT = "results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01"
OUT = "data/things_eeg/captions"


class Args:
    input, checkpoint_dir, feature_dim = "proj", CKPT, 512


def write(path, concepts, caps):
    with open(path, "w") as f:
        for c, cap in zip(concepts, caps):
            f.write(json.dumps({"concept": c, "captions": [cap]}) + "\n")
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    # clip = the recon encoder's space (raw ViT-H, identity projector), the EEG->image->caption route
    p.add_argument("--space", default="internvit", choices=["internvit", "clip", "enigma"])
    p.add_argument("--enigma_dir",
                   default="/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01",
                   help="--space enigma: read ENIGMA's own predicted ViT-H embeddings from here")
    p.add_argument("--npz", default=f"{OUT}/eeg_sub01.npz")
    p.add_argument("--tag", default="", help="suffix for the output jsonl")
    args = p.parse_args()

    if args.space == "enigma":
        # ENIGMA regresses RAW ViT-H image embeddings, so the bank is the raw ViT-H training
        # features -- not our shared space, and not the ridge-mapped one the `clip` arm uses.
        tr = np.load("data/things_eeg/image_feature/ViT-H-14_final/image_train.npy")
        xtr = F.normalize(torch.from_numpy(tr.reshape(-1, tr.shape[-1])).float(), dim=-1)
    elif args.space == "clip":
        xtr = clip_feats("training")
    else:
        xtr = F.normalize(cb.embeddings("training", Args), dim=-1)
    bank = cb.load_captions("training")
    assert len(bank) == len(xtr), f"{len(bank)} captions vs {len(xtr)} training embeddings"

    concepts = sorted(os.listdir(os.path.join(cb.ROOT, "test_images")))
    if args.space == "enigma":
        # dir name is category_num, which is the sorted-concept index on the test split
        eeg = torch.stack([torch.load(os.path.join(args.enigma_dir, str(i), "predicted_embeds.pt"),
                                      weights_only=False) for i in range(len(concepts))])
        d = {"eeg": eeg.numpy(),
             "image": np.load("data/things_eeg/image_feature/ViT-H-14_final/image_test.npy"
                              ).reshape(len(concepts), -1)}
    else:
        d = np.load(args.npz)
        assert (d["object"] == np.arange(len(d["eeg"]))).all(), "npz rows are not in concept order"

    for name, v in [("eeg", d["eeg"]), ("img", d["image"])]:  # img = this method's own ceiling
        q = F.normalize(torch.from_numpy(v).float(), dim=-1)
        nn = (q @ xtr.T).argmax(1)
        write(f"{OUT}/test.nn_{name}{args.tag}.jsonl", concepts, [bank[j] for j in nn.tolist()])
        for c, j in list(zip(concepts, nn.tolist()))[:6]:
            print(f"  {c:28s} {bank[j][:88]}")
        print()


if __name__ == "__main__":
    main()
