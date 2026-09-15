"""EEG foundation-model encoders (LaBraM, CBraMod) wrapped for the alignment pipeline.

Reconstructed 2026-09-15 from the surviving artifacts of the July 2026 vxam-Q1 experiment,
whose original source was lost while `scripts/` was gitignored (see PROTECTED.md). The
architecture is not guesswork: `results/things_eeg/foundation/*/`'s `model_state_dict` pins it
exactly — 221 parameters, zero shape mismatches against
`Labram(n_times=200, n_chans=63, n_outputs=0, patch_size=200)` plus a `head` Linear(200, 1024).
Recipe from the surviving `train_config.json`: alpha 0.8 CLIP+MSE, lr 2e-3 -> 1e-5 cosine with
5 warmup epochs, weight decay 0.05, 50 epochs, full fine-tuning (not frozen, not LoRA).

Pretrained weights come from the Hugging Face Hub ids `braindecode/labram-pretrained` and
`braindecode/cbramod-pretrained`, both already in the local HF cache.

The one thing the saved weights do NOT determine is how a 250-sample/250 Hz THINGS-EEG-2 epoch
was reduced to the 200 samples the foundation models consume. `--fm_resample` selects it:
`resample` (default) treats both as the same one-second window and interpolates 250 Hz -> 200 Hz,
which is what LaBraM's 200-sample (1 s at 200 Hz) patch size expects; `crop` takes the first 200
samples (0.8 s). See `scripts/things_eeg/verify_fm_reconstruction.py` for the empirical check
against the recorded per-fold accuracies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# THINGS-EEG-2 preprocessed montage, in channel order. All 63 map case-insensitively onto
# braindecode's 128-entry LABRAM_CHANNEL_ORDER, so LaBraM can select the matching position
# embeddings for this subset.
THINGS_EEG2_CH_NAMES = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4',
    'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
]

# Both models were run with a one-second window at their own pretraining rate.
FM_N_TIMES = 200
FM_PATCH_SIZE = 200
FM_EMBED_DIM = 200

_HUB_IDS = {
    'LaBraM': 'braindecode/labram-pretrained',
    'CBraMod': 'braindecode/cbramod-pretrained',
}


class FoundationEncoder(nn.Module):
    """A pretrained EEG foundation model plus a linear head onto the backbone width.

    Forward maps ``(batch, channels, time)`` to ``(batch, feature_dim)``. The whole stack is
    trainable: the July experiment full-fine-tuned the foundation model, which is what the
    literature does (frozen / linear-probe underperforms, see AdaBrain-Bench arXiv 2507.09882).
    """

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        channels_num: int,
        pretrained: bool = True,
        resample: str = 'resample',
        ch_names: list[str] | None = None,
    ):
        super().__init__()
        if model_name not in _HUB_IDS:
            raise ValueError(f"Unknown foundation model {model_name!r}; expected one of {sorted(_HUB_IDS)}")
        if resample not in ('resample', 'crop'):
            raise ValueError(f"resample must be 'resample' or 'crop', got {resample!r}")

        self.model_name = model_name
        self.resample = resample
        self.ch_names = list(ch_names) if ch_names is not None else list(THINGS_EEG2_CH_NAMES)
        if len(self.ch_names) != channels_num:
            raise ValueError(
                f"{model_name} needs one channel name per channel: got {channels_num} channels "
                f"but {len(self.ch_names)} names. Pass --fm_ch_names for a non-default montage."
            )

        self.fm = self._build(model_name, channels_num, pretrained)
        self.head = nn.Linear(FM_EMBED_DIM, feature_dim)

    @staticmethod
    def _build(model_name: str, channels_num: int, pretrained: bool) -> nn.Module:
        # Imported lazily: braindecode pulls in mne/mne-bids, so an unrelated run should not
        # pay for it (nor fail on it).
        from braindecode.models import CBraMod, Labram

        if model_name == 'LaBraM':
            cls, kwargs = Labram, dict(n_outputs=0)  # n_outputs=0 -> final_layer is nn.Identity
            # LaBraM's Hub config carries its own 128-entry chs_info, which conflicts with any
            # n_chans we would pass, so the donor is built from that config untouched.
            donor_kwargs: dict = {}
        else:
            # CBraMod always builds a LazyLinear task head from n_outputs; return_encoder_output
            # replaces it with Identity instead, which is why the saved CBraMod weights stop at
            # fm.proj_out with no head at all. Its Hub config is all-null, so the donor needs the
            # same explicit dims as the target.
            cls, kwargs = CBraMod, dict(return_encoder_output=True)
            donor_kwargs = dict(n_times=FM_N_TIMES, n_chans=channels_num,
                                patch_size=FM_PATCH_SIZE, **kwargs)
        model = cls(
            n_times=FM_N_TIMES,
            n_chans=channels_num,
            patch_size=FM_PATCH_SIZE,
            **kwargs,
        )
        if pretrained:
            pre = cls.from_pretrained(_HUB_IDS[model_name], **donor_kwargs)
            # The Hub checkpoints are built for their own n_times, so the temporal embedding and
            # any other length-dependent buffer will not match a 200-sample window. Take every
            # parameter whose shape agrees and leave the rest at init.
            target = model.state_dict()
            donor = {
                k: v for k, v in pre.state_dict().items()
                if k in target and target[k].shape == v.shape
            }
            missing = [k for k in target if k not in donor]
            model.load_state_dict(donor, strict=False)
            if missing:
                print(
                    f"[{model_name}] loaded {len(donor)}/{len(target)} pretrained tensors; "
                    f"randomly initialised: {missing}"
                )
        return model

    def _to_fm_window(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce a (batch, channels, time) epoch to the model's FM_N_TIMES samples."""
        n_time = x.shape[-1]
        if n_time == FM_N_TIMES:
            return x
        if self.resample == 'crop':
            if n_time < FM_N_TIMES:
                raise ValueError(f"cannot crop {n_time} samples to {FM_N_TIMES}")
            return x[..., :FM_N_TIMES]
        # Same one-second window, resampled to the model's rate.
        return F.interpolate(x, size=FM_N_TIMES, mode='linear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_fm_window(x)
        if self.model_name == 'LaBraM':
            # Without ch_names LaBraM requires the full 128-channel canonical order.
            feats = self.fm(x, ch_names=self.ch_names)
        else:
            feats = self.fm(x)
        if feats.dim() > 2:  # pool any residual token axis
            feats = feats.mean(dim=tuple(range(1, feats.dim() - 1)))
        return self.head(feats)


def is_foundation_encoder(name: str) -> bool:
    return name in _HUB_IDS


def build_foundation_encoder(
    model_name: str,
    feature_dim: int,
    channels_num: int,
    pretrained: bool = True,
    resample: str = 'resample',
    ch_names: list[str] | None = None,
) -> FoundationEncoder:
    return FoundationEncoder(
        model_name=model_name,
        feature_dim=feature_dim,
        channels_num=channels_num,
        pretrained=pretrained,
        resample=resample,
        ch_names=ch_names,
    )
