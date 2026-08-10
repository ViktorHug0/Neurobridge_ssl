"""Phase 1b: train the EEG->VAE-latent low-level decoder for one subject.

Reuses EEGPreImageDataset for EEG loading (data paths/channels taken from a trained recon
checkpoint's train_config), and regresses to the precomputed blurred-stimulus VAE latents.
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from module.dataset import EEGPreImageDataset
from module.lowlevel import EEGLatentDecoder
from train import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="recon checkpoint (for data paths/channels)")
    ap.add_argument("--latents", default="data/things_eeg/vae_latents/train.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = SimpleNamespace(**json.load(open(os.path.join(args.checkpoint_dir, "train_config.json"))))
    seed_everything(2025)
    # ConvTranspose2d has no deterministic cuDNN kernel; seed_everything's determinism throws
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    ds = EEGPreImageDataset(
        [cfg.train_subject_ids[0]], cfg.eeg_data_dir, cfg.selected_channels, cfg.time_window,
        cfg.image_feature_dir, "", False, [], average=True, train=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    lat = torch.from_numpy(np.load(args.latents).astype(np.float32))  # (obj,img,4,64,64) on CPU
    ch, T = ds.channels_num, ds.num_sample_points  # actual data dims (not len(selected_channels))
    model = EEGLatentDecoder(channels_num=ch, eeg_sample_points=T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = torch.nn.MSELoss()

    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for batch in loader:
            eeg = batch[0].to(device).float()
            target = lat[batch[4], batch[5]].to(device)  # index CPU latents -> GPU
            loss = mse(model(eeg), target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(eeg); n += len(eeg)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep+1}/{args.epochs} mse {tot/n:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "channels": ch, "T": T}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
