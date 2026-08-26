"""Run a trained ENIGMA encoder over a whole split and save its predicted ViT-H embeddings.

`recon_inference.py` is hard-coded to split="test" and only writes per-concept files. The bridge
needs the TRAINING split too (next_experiments.md E1: fit the bridge on real EEG predictions
instead of on real image embeddings plus Gaussian noise).

Rows come out in stimulus_walk order -- sorted concept dirs, sorted files within them -- which is
what qformer_<split>.npy and caption_bridge.embeddings() use. Asserted against image_path, not
assumed.

  cd /nasbrain/p20fores/ENIGMA && .venv-enigma/bin/python <this> --split training
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nasbrain/p20fores/ENIGMA")
from source.dataset import EEGDataset          # noqa: E402
from source.models import ENIGMA               # noqa: E402

ROOT = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"
OUT = "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/enigma"


def walk(split):
    d = os.path.join(ROOT, f"{split}_images")
    return [os.path.join(d, c, f)
            for c in sorted(os.listdir(d))
            for f in sorted(os.listdir(os.path.join(d, c)))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["training", "test"], required=True)
    p.add_argument("--subject", default="sub-01")
    p.add_argument("--ckpt", default="/nasbrain/p20fores/ENIGMA/train_logs/ENIGMA_repro_sub01/last.pth")
    p.add_argument("--config", default="things_eeg2_local")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=512)
    a = p.parse_args()

    ds = EEGDataset(config_name=a.config, subjects=[a.subject],
                    split="test" if a.split == "test" else "train")
    paths = walk(a.split)
    assert ds.stim_df["image_path"].tolist() == paths, "dataset order != stimulus walk order"

    model = ENIGMA(ds.eeg_data.shape[-2], ds.eeg_data.shape[-1], subjects=[a.subject], embed_dim=1024)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False))
    model.cuda().eval()

    out = np.zeros((len(ds.eeg_data), 1024), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(out), a.batch_size):
            eeg = ds.eeg_data[i: i + a.batch_size].cuda()
            out[i: i + len(eeg)] = model(eeg, [a.subject] * len(eeg)).cpu().numpy()
    path = a.out or os.path.join(OUT, f"enigma_pred_{a.split}_{a.subject}.npy")
    np.save(path, out)
    print(f"wrote {path} {out.shape}")


if __name__ == "__main__":
    main()
