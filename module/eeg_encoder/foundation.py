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
    'REVE': 'brain-bzh/reve-base',
}

# LaBraM/CBraMod emit FM_EMBED_DIM-wide features; REVE's transformer is 512-wide.
_EMBED_DIMS = {'LaBraM': FM_EMBED_DIM, 'CBraMod': FM_EMBED_DIM, 'REVE': 512}

# REVE was pretrained on z-scored EEG clipped at 15 SD (braindecode reve.py docstring). The
# grader serves RobustScaler/clamp-20 tensors, so we restandardise per trial -- identically at
# train and inference, so this cannot reintroduce a train/test mismatch.
REVE_CLIP_SD = 15.0


class ReveFlattenHead(nn.Module):
    """REVE's own fine-tuning head (reve_eeg classifier.py, pooling "no"): one learned query
    attention-pools the last-layer tokens, the pooled token is concatenated with every token,
    then flatten -> RMSNorm -> dropout -> Linear. Unlike a token mean it keeps the channel layout."""

    def __init__(self, n_tokens: int, embed_dim: int, feature_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * embed_dim ** -0.5)
        width = (n_tokens + 1) * embed_dim
        self.norm = nn.RMSNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(width, feature_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, tokens, E)
        scores = self.query @ tokens.transpose(1, 2) / tokens.shape[-1] ** 0.5
        pooled = torch.softmax(scores, dim=-1) @ tokens  # (B, 1, E)
        x = torch.cat([pooled, tokens], dim=1).flatten(1)
        return self.linear(self.dropout(self.norm(x)))


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
        input_normalization: str = 'trial_zscore',
        head: str = 'mean',
        head_dropout: float = 0.0,
    ):
        super().__init__()
        if input_normalization not in ('trial_zscore', 'none'):
            raise ValueError(
                f"input_normalization must be 'trial_zscore' or 'none', got {input_normalization!r}"
            )
        self.input_normalization = input_normalization
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

        self.fm = self._build(model_name, channels_num, pretrained, self.ch_names)
        self.head_type = head
        if head == 'mean':
            self.head = nn.Linear(_EMBED_DIMS[model_name], feature_dim)
        elif head == 'flatten' and model_name == 'REVE':
            size, overlap = self.fm.patch_size, self.fm.patch_overlap
            n_patches = (FM_N_TIMES - size) // (size - overlap) + 1  # Tensor.unfold's count
            self.head = ReveFlattenHead(channels_num * n_patches, _EMBED_DIMS[model_name],
                                        feature_dim, head_dropout)
        else:
            raise ValueError(f"head must be 'mean', or 'flatten' for REVE; got {head!r}")

    def unfreeze_last_k(self, k: int) -> int:
        """Train only the last `k` backbone layers plus the head; freeze everything before.

        A frozen probe is not a real option here: on this benchmark REVE head-only reaches 0.064
        val batch top-5 against 0.499 for full fine-tuning, matching AdaBrain-Bench (arXiv
        2507.09882). So the knob worth having is how much of the stack to open. REVE is 22 blocks
        of 3,140,096 parameters, so k=4 is 13.3M trainable (19%) against 70.1M for a full tune.
        """
        if self.model_name != 'REVE':
            raise NotImplementedError(
                f"unfreeze_last_k is only wired for REVE's transformer.layers, not {self.model_name}"
            )
        layers = self.fm.transformer.layers
        if not 0 < k <= len(layers):
            raise ValueError(f"k must be in 1..{len(layers)}, got {k}")
        for parameter in self.fm.parameters():
            parameter.requires_grad_(False)
        for block in layers[len(layers) - k:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _build(model_name: str, channels_num: int, pretrained: bool, ch_names: list[str]) -> nn.Module:
        # Imported lazily: braindecode pulls in mne/mne-bids, so an unrelated run should not
        # pay for it (nor fail on it).
        from braindecode.models import Labram

        if model_name == 'REVE':
            # REVE resolves each channel to a 3D coordinate from its position bank, which is what
            # makes it montage-agnostic. n_outputs=2 is a dummy head we bypass in forward().
            from braindecode.models import REVE

            kwargs = dict(
                n_outputs=2,
                n_chans=channels_num,
                n_times=FM_N_TIMES,
                chs_info=[{'ch_name': name} for name in ch_names],
            )
            return REVE.from_pretrained(_HUB_IDS['REVE'], **kwargs) if pretrained else REVE(**kwargs)

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
            from braindecode.models import CBraMod  # absent from some braindecode versions

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
        if self.model_name == 'REVE':
            if self.input_normalization == 'trial_zscore':
                # REVE's pretraining convention (braindecode reve.py). Applied identically at
                # train and inference, so it adds no mismatch -- but it is NOT what the 0.29
                # checkpoint did: that run relied on the offline StandardScaler alone. Use
                # 'none' to reproduce it.
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
                x = ((x - mean) / std).clamp(-REVE_CLIP_SD, REVE_CLIP_SD)
            # return_output bypasses the dummy classification head; last layer -> (B, tokens, 512)
            feats = self.fm(x, return_output=True)[-1]
            if self.head_type == 'flatten':
                return self.head(feats)
        elif self.model_name == 'LaBraM':
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
    input_normalization: str = 'trial_zscore',
    head: str = 'mean',
    head_dropout: float = 0.0,
) -> FoundationEncoder:
    return FoundationEncoder(
        model_name=model_name,
        feature_dim=feature_dim,
        channels_num=channels_num,
        pretrained=pretrained,
        resample=resample,
        ch_names=ch_names,
        input_normalization=input_normalization,
        head=head,
        head_dropout=head_dropout,
    )
