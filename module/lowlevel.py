"""Phase 1b: low-level decoder. EEG -> SDXL VAE latent (4x64x64), for an img2img init that
injects coarse layout/colour before IP-Adapter refines the high-level semantics (ATM/MindEye).
"""
import torch.nn as nn

from module.eeg_encoder.model import TSConv


class EEGLatentDecoder(nn.Module):
    def __init__(self, channels_num=17, eeg_sample_points=250, feat_dim=1024):
        super().__init__()
        self.backbone = TSConv(feature_dim=feat_dim, eeg_sample_points=eeg_sample_points,
                               channels_num=channels_num)
        self.fc = nn.Linear(feat_dim, 256 * 8 * 8)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),  # 8->16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),    # 16->32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),     # 32->64
            nn.Conv2d(32, 4, 3, 1, 1),                                               # ->4x64x64
        )

    def forward(self, eeg):
        f = self.backbone(eeg)                 # (B, feat_dim)
        return self.up(self.fc(f).view(-1, 256, 8, 8))  # (B, 4, 64, 64)
