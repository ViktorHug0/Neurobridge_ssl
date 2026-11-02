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
    'EEGPT': 'braindecode/eegpt-pretrained',
}
# EEGPT was pretrained at 250 Hz with 64-sample patches (stride 32): 256 samples = 7 patches.
EEGPT_N_TIMES = 256

# LaBraM/CBraMod emit FM_EMBED_DIM-wide features; REVE's transformer is 512-wide.
_EMBED_DIMS = {'LaBraM': FM_EMBED_DIM, 'CBraMod': FM_EMBED_DIM, 'REVE': 512,
               'EEGPT': 4 * 512}  # EEGPT: 4 summary tokens x 512, averaged over patches

# REVE was pretrained on z-scored EEG clipped at 15 SD (braindecode reve.py docstring). The
# grader serves RobustScaler/clamp-20 tensors, so we restandardise per trial -- identically at
# train and inference, so this cannot reintroduce a train/test mismatch.
REVE_CLIP_SD = 15.0


class _DropPath(nn.Module):
    """Stochastic depth for a residual branch whose caller adds the skip (x = branch(x) + x)."""
    def __init__(self, branch, p):
        super().__init__()
        self.branch, self.p = branch, p

    def forward(self, x):
        y = self.branch(x)
        if not self.training or self.p == 0:
            return y
        keep = (torch.rand(y.shape[0], *([1] * (y.dim() - 1)), device=y.device) >= self.p).to(y.dtype)
        return y * keep / (1 - self.p)


class _ChanLoRAPatchEmbed(nn.Module):
    """REVE's shared patch embedding plus a per-electrode low-rank correction: patches (B, C, n, P) ->
    base(p) + (p @ A[c]) @ B. A starts at zero, so training starts from the shared embedding."""
    def __init__(self, base, n_chans, patch_size, embed_dim, rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.zeros(n_chans, patch_size, rank))
        self.B = nn.Parameter(torch.randn(rank, embed_dim) / rank ** 0.5)

    def forward(self, p):
        return self.base(p) + torch.einsum('bcnp,cpr->bcnr', p, self.A) @ self.B


class _ResidualMLP(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.f = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.f(x)


class _AttnStatHead(nn.Module):
    """Squeezeformer's pooling on foundation-model tokens: softmax token weights, weighted mean and std, linear."""
    def __init__(self, width, feature_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.linear = nn.Linear(2 * width, feature_dim)

    def forward(self, tokens):  # (B, T, E)
        w = self.score(tokens).softmax(dim=1)
        mean = (w * tokens).sum(1)
        std = (w * (tokens - mean[:, None]).square()).sum(1).clamp_min(1e-6).sqrt()
        return self.linear(torch.cat([mean, std], -1))


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
        drop_path: float = 0.0,
        arch_kwargs: dict | None = None,
    ):
        super().__init__()
        # arch_kwargs: constructor overrides (depth, width, patch size...) for a from-scratch model
        arch_kwargs = dict(arch_kwargs or {})
        # final_norm False drops the backbone's last LayerNorm before the head (REVE's return_output never had one);
        # popped first because it changes no weight shape, so it is valid on a pretrained model too
        final_norm = arch_kwargs.pop('final_norm', True)
        resized = bool(arch_kwargs)
        if arch_kwargs and pretrained:
            raise ValueError('arch_kwargs change the architecture: only valid with pretrained=False')
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

        if model_name == 'EEGPT':
            # EEGPT's channel vocabulary lacks AFz/FT9/FT10/TP9/TP10: feed it the other 58.
            from braindecode.models.eegpt import CHANNEL_DICT
            keep = [i for i, n in enumerate(self.ch_names) if n.upper() in CHANNEL_DICT]
            self.register_buffer('eegpt_keep', torch.tensor(keep), persistent=False)
        # Wrapper-level crossovers (LaBraM, scratch only): the input's own rate (n_times), a learned
        # 63 -> K spatial mix feeding K "virtual electrode" tokens, TSConv's band-power branch in the head.
        self.n_times = arch_kwargs.get('n_times', FM_N_TIMES)
        virtual = arch_kwargs.pop('virtual_channels', 0)
        power = arch_kwargs.pop('power_branch', False)
        self.spatial = nn.Conv1d(channels_num, virtual, 1, bias=False) if virtual else None
        self.fm_names = self.ch_names[:virtual] if virtual else self.ch_names
        self.power = nn.Sequential(nn.Conv1d(channels_num, 40, 25, bias=False), nn.BatchNorm1d(40),
                                   ) if power else None
        self.power_pool, self.power_dropout = nn.AvgPool1d(25, 2), nn.Dropout(0.15)
        self.fm = self._build(model_name, len(self.fm_names), pretrained, self.fm_names, drop_path, arch_kwargs)
        if not final_norm:
            assert isinstance(self.fm.norm, nn.LayerNorm), f'{model_name} has no final LayerNorm to drop'
            self.fm.norm = nn.Identity()
        self.head_type = head
        if head == 'mean':
            width = _EMBED_DIMS[model_name]
            if resized:  # a resized model: LaBraM's token width follows its tokenizer, so measure it
                self.head = nn.Identity()
                with torch.no_grad():
                    width = self.forward(torch.zeros(2, channels_num, FM_N_TIMES)).shape[-1]
            self.head = nn.Linear(width, feature_dim)
        elif head == 'attnstat':  # SQF-style pooling: attention-weighted mean + std over tokens -> linear
            width = _EMBED_DIMS[model_name]
            self.head = _AttnStatHead(width, feature_dim)
        elif head == 'mlp':  # mean-pooled tokens -> the TSConv/SQF head trunk: Linear, residual GELU MLP, LayerNorm,
            # then a free linear lift (output norm unconstrained)
            width = _EMBED_DIMS[model_name]
            self.head = nn.Sequential(nn.Linear(width, width), _ResidualMLP(width, head_dropout or 0.5),
                                      nn.LayerNorm(width), nn.Linear(width, feature_dim))
        elif head == 'flatten' and model_name == 'REVE':
            size, overlap = self.fm.patch_size, self.fm.patch_overlap
            n_patches = (FM_N_TIMES - size) // (size - overlap) + 1  # Tensor.unfold's count
            self.head = ReveFlattenHead(channels_num * n_patches, _EMBED_DIMS[model_name],
                                        feature_dim, head_dropout)
            if pretrained:  # as the authors do: start the query from REVE's pretrained cls_query_token
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                self.head.query.data.copy_(
                    load_file(hf_hub_download(_HUB_IDS['REVE'], 'model.safetensors'))['cls_query_token'])
        elif head == 'flatten' and model_name == 'LaBraM':
            # patch_size == FM_N_TIMES: one token per channel
            self.head = ReveFlattenHead(channels_num, _EMBED_DIMS[model_name], feature_dim, head_dropout)
        elif head == 'multilayer' and model_name == 'LaBraM':
            # The usual pooled last layer plus the channel-token means of the 3 blocks before it.
            self._taps = []
            for block in self.fm.blocks[-4:-1]:
                block.register_forward_hook(lambda _m, _i, out: self._taps.append(out[:, 1:].mean(1)))
            width = 4 * _EMBED_DIMS[model_name]
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(head_dropout), nn.Linear(width, feature_dim))
        else:
            raise ValueError(f"head must be 'mean', 'mlp', 'flatten' (REVE/LaBraM) or 'multilayer' (LaBraM); got {head!r} (or 'attnstat')")

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
    def _build(model_name: str, channels_num: int, pretrained: bool, ch_names: list[str],
               drop_path: float = 0.0, arch_kwargs: dict | None = None) -> nn.Module:
        # Imported lazily: braindecode pulls in mne/mne-bids, so an unrelated run should not
        # pay for it (nor fail on it).
        from braindecode.models import Labram

        if model_name == 'REVE':
            # REVE resolves each channel to a 3D coordinate from its position bank, which is what
            # makes it montage-agnostic. n_outputs=2 is a dummy head we bypass in forward().
            from braindecode.models import REVE

            arch_kwargs = dict(arch_kwargs or {})
            chan_lora = arch_kwargs.pop('chan_lora', 0)  # ours, not a REVE argument: per-electrode patch correction rank
            kwargs = {**dict(
                n_outputs=2,
                n_chans=channels_num,
                n_times=FM_N_TIMES,
                chs_info=[{'ch_name': name} for name in ch_names],
            ), **(arch_kwargs or {})}
            fm = REVE.from_pretrained(_HUB_IDS['REVE'], **kwargs) if pretrained else REVE(**kwargs)
            if drop_path > 0:  # stochastic depth on every residual branch, rate linear in depth (DeiT)
                layers = fm.transformer.layers
                for i, pair in enumerate(layers):
                    for j in range(2):
                        pair[j] = _DropPath(pair[j], drop_path * i / max(1, len(layers) - 1))
            if chan_lora:
                fm.to_patch_embedding = _ChanLoRAPatchEmbed(fm.to_patch_embedding, channels_num, fm.patch_size,
                                                            fm.embed_dim, chan_lora)
            return fm

        if model_name == 'EEGPT':
            from braindecode.models import EEGPT
            from braindecode.models.eegpt import CHANNEL_DICT

            names = [n.upper() for n in ch_names if n.upper() in CHANNEL_DICT]
            model = EEGPT(n_chans=len(names), chs_info=[{'ch_name': n} for n in names],
                          n_times=EEGPT_N_TIMES, sfreq=250, chan_proj_type='none',
                          return_encoder_output=True)
            if pretrained:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file

                donor = load_file(hf_hub_download(_HUB_IDS['EEGPT'], 'model.safetensors'))
                target = model.state_dict()
                keep = {k: v for k, v in donor.items() if k in target and target[k].shape == v.shape
                        and k != 'chans_id'}
                model.load_state_dict(keep, strict=False)
                print(f"[EEGPT] loaded {len(keep)}/{len(target)} pretrained tensors; "
                      f"left at init: {[k for k in target if k not in keep]}")
            return model

        if model_name == 'LaBraM':
            # n_outputs=0 -> final_layer is nn.Identity; drop_path is LaBraM's fine-tuning stochastic depth
            cls, kwargs = Labram, dict(n_outputs=0, drop_path_prob=drop_path)
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
        model = cls(**{'n_times': FM_N_TIMES, 'n_chans': channels_num, 'patch_size': FM_PATCH_SIZE,
                       **kwargs, **(arch_kwargs or {})})
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
        """Reduce a (batch, channels, time) epoch to the model's n_times samples (FM_N_TIMES by default)."""
        n_time = x.shape[-1]
        if n_time == self.n_times:
            return x
        if self.resample == 'crop':
            if n_time < self.n_times:
                raise ValueError(f"cannot crop {n_time} samples to {self.n_times}")
            return x[..., :self.n_times]
        # Same one-second window, resampled to the model's rate.
        return F.interpolate(x, size=self.n_times, mode='linear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == 'EEGPT':
            x = F.interpolate(x[:, self.eegpt_keep], size=EEGPT_N_TIMES, mode='linear', align_corners=False)
            return self.head(self.fm(x).mean(dim=1).flatten(1))  # (B, patches, 4, 512) -> (B, 2048)
        x = self._to_fm_window(x)
        power = None
        if self.power is not None:  # on the model-rate window, so its width is fixed
            power = self.power_pool(self.power(x).square()).clamp_min(1e-6).log().flatten(1)
            power = self.power_dropout(power)
        if self.spatial is not None:
            x = self.spatial(x)
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
            if self.head_type in ('flatten', 'attnstat'):
                return self.head(feats)
        elif self.model_name == 'LaBraM':
            # Without ch_names LaBraM requires the full 128-channel canonical order.
            if self.head_type == 'flatten':
                return self.head(self.fm(x, ch_names=self.ch_names, return_patch_tokens=True))
            if self.head_type == 'multilayer':
                # ponytail: taps via forward hooks; fine while the module is never deepcopied
                feats = self.fm(x, ch_names=self.ch_names)
                taps, self._taps[:] = list(self._taps), []
                return self.head(torch.cat([feats, *taps], dim=-1))
            feats = self.fm(x, ch_names=self.fm_names)
        else:
            feats = self.fm(x)
        if feats.dim() > 2:  # pool any residual token axis
            feats = feats.mean(dim=tuple(range(1, feats.dim() - 1)))
        if power is not None:
            feats = torch.cat([feats, power], dim=-1)
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
    drop_path: float = 0.0,
    arch_kwargs: dict | None = None,
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
        drop_path=drop_path,
        arch_kwargs=arch_kwargs,
    )
