"""Five-member ensemble for Track 1 EEG-to-image retrieval (local grader top-5 0.4821).

Members (Neurobridge_SSL runs, local grader top-5 each):
  reve_r1    REVE-base, mean head                          0.4063
  reve_r12   REVE-base, mean head, head/backbone lr split   0.3969
  reve_r123  REVE-base, REVE query-pool + flatten head      0.3717
  sqf        TSConv-Squeezeformer, trunk 512                0.3727
  sqf_power  TSConv-Squeezeformer + band-power branch       0.3759

Every member's prediction is L2-normalised and the five are averaged with equal weights; the
grader's cosine ranking of that mean equals the ranking of the summed member cosines. REVE
members were trained on the 120 Hz warm-up tensors linearly stretched to 200 samples; the
Squeezeformers consume the 120 samples directly.

Channel order: the grader serves channels in the BrainVision recording order (Fp1, F3, F7, ...)
while every member was trained on the 10-20 order below. forward() permutes by name, which
REVE alone would tolerate (it reads positions from chs_info) but the Squeezeformers' spatial
convolutions would not.
"""

import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.models import REVE
from benchmark_utils.base_solver import CompetSolver


TRAIN_CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4',
    'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
]


# ---- REVE members ------------------------------------------------------------------------

class ReveFlattenHead(nn.Module):
    """One learned query attention-pools the tokens; [pooled, tokens] -> RMSNorm -> Linear."""

    def __init__(self, n_tokens, embed_dim, feature_dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        width = (n_tokens + 1) * embed_dim
        self.norm = nn.RMSNorm(width)
        self.linear = nn.Linear(width, feature_dim)

    def forward(self, tokens):
        scores = self.query @ tokens.transpose(1, 2) / tokens.shape[-1] ** 0.5
        pooled = torch.softmax(scores, dim=-1) @ tokens
        return self.linear(self.norm(torch.cat([pooled, tokens], dim=1).flatten(1)))


class FineTunedReve(nn.Module):
    def __init__(self, n_chans, n_outputs, chs_info, flatten_head):
        super().__init__()
        self.backbone = REVE(n_chans=n_chans, n_times=200, n_outputs=2, chs_info=chs_info)
        self.flatten_head = flatten_head
        # REVE patches 200 samples with 20 overlap -> one patch per channel.
        self.probe = (ReveFlattenHead(n_chans, 512, n_outputs) if flatten_head
                      else nn.Linear(512, n_outputs))

    def forward(self, x):
        if x.shape[-1] != 200:
            x = F.interpolate(x, size=200, mode="linear", align_corners=False)
        tokens = self.backbone(x, return_output=True)[-1]
        return self.probe(tokens if self.flatten_head else tokens.mean(dim=1))


# ---- TSConv-Squeezeformer members (ensemble_experiments/architectures/ortho_encoders.py) ---

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


def _head(in_dim, feature_dim, dropout=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, feature_dim),
        ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(feature_dim, feature_dim), nn.Dropout(dropout))),
        nn.LayerNorm(feature_dim),
    )


class _FusedTemporalConvPool(nn.Module):
    """Conv25 -> AvgPool fused into one strided convolution."""

    def __init__(self, out_channels, temporal_kernel, pool_kernel, pool_stride):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 1, 1, temporal_kernel))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.pool_stride = pool_stride
        self.effective_kernel = temporal_kernel + pool_kernel - 1
        self.register_buffer("box", torch.full((1, 1, pool_kernel), 1.0 / pool_kernel), persistent=False)

    def forward(self, x):
        weight = F.conv1d(self.weight.squeeze(2), self.box, padding=self.box.shape[-1] - 1).unsqueeze(2)
        return F.conv2d(x, weight, self.bias, stride=(1, self.pool_stride))


class _ConformerFeedForward(nn.Module):
    def __init__(self, d_model, expansion, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, expansion * d_model), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(expansion * d_model, d_model), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class _ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, drop):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)
        z = F.glu(self.pointwise_in(z), dim=1)
        z = F.silu(self.bn(self.depthwise(z)))
        return self.drop(self.pointwise_out(z).transpose(1, 2))


class _FastSelfAttention(nn.Module):
    def __init__(self, d_model, heads, drop):
        super().__init__()
        self.heads, self.head_dim = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))


class _FastSqueezeformerBlock(nn.Module):
    def __init__(self, d_model, heads, conv_kernel, expansion, drop):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _FastSelfAttention(d_model, heads, drop)
        self.conv = _ConformerConvModule(d_model, conv_kernel, drop)
        self.ffn = _ConformerFeedForward(d_model, expansion, drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return self.final_norm(x)


class FastTSConvSqueezeformer(nn.Module):
    def __init__(self, feature_dim, eeg_sample_points, channels_num, power_branch,
                 d_model=160, depth=3, heads=4, conv_kernel=15, expansion=2, drop=0.15,
                 stem_filters=40, pool_kernel=25, pool_stride=2, backbone_dim=512):
        super().__init__()
        self.n_times = eeg_sample_points
        self.temporal = _FusedTemporalConvPool(stem_filters, 25, pool_kernel, pool_stride)
        max_tokens = (eeg_sample_points - self.temporal.effective_kernel) // pool_stride + 1
        self.stem_after_pool = nn.Sequential(
            nn.BatchNorm2d(stem_filters), nn.ELU(),
            nn.Conv2d(stem_filters, stem_filters, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(stem_filters), nn.ELU(), nn.Dropout(drop),
        )
        self.token_projection = nn.Sequential(nn.Linear(stem_filters, d_model), nn.LayerNorm(d_model))
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        self.blocks = nn.ModuleList([
            _FastSqueezeformerBlock(d_model, heads, conv_kernel, expansion, drop) for _ in range(depth)
        ])
        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1),
        )
        # Band-power branch: 63->40 Conv1d (25 taps) -> BN -> square -> pool -> log.
        self.power_branch = nn.Sequential(
            nn.Conv1d(channels_num, stem_filters, 25, bias=False), nn.BatchNorm1d(stem_filters),
        ) if power_branch else None
        self.power_pool = nn.AvgPool1d(pool_kernel, pool_stride)
        power_dim = stem_filters * ((eeg_sample_points - 24 - pool_kernel) // pool_stride + 1) if power_branch else 0
        self.head = _head(2 * d_model + power_dim, backbone_dim)
        self.head.append(nn.Linear(backbone_dim, feature_dim))

    def forward(self, x):
        if x.shape[-1] != self.n_times:
            x = F.interpolate(x, size=self.n_times, mode="linear", align_corners=False)
        z = self.temporal(x.unsqueeze(1))
        z = self.stem_after_pool(z).squeeze(2).transpose(1, 2)
        z = self.token_projection(z) + self.position[:, :z.shape[1]]
        for block in self.blocks:
            z = block(z)
        weights = self.pool_score(z).softmax(dim=1)
        mean = (weights * z).sum(dim=1)
        variance = (weights * (z - mean.unsqueeze(1)).square()).sum(dim=1)
        pooled = torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)
        if self.power_branch is not None:
            power = self.power_pool(self.power_branch(x).square())
            pooled = torch.cat([pooled, power.clamp_min(1e-6).log().flatten(1)], dim=-1)
        return self.head(pooled)


# ---- ensemble ----------------------------------------------------------------------------

class Ensemble(nn.Module):
    def __init__(self, members, served_channels):
        super().__init__()
        self.members = nn.ModuleDict(members)
        served = [name.lower() for name in served_channels]
        order = torch.tensor([served.index(name.lower()) for name in TRAIN_CHANNELS])
        self.register_buffer("order", order, persistent=False)

    def forward(self, x):
        x = x.float()[:, self.order]
        preds = [F.normalize(member(x), dim=-1) for member in self.members.values()]
        return torch.stack(preds).mean(dim=0)

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        return self(x)


class Solver(CompetSolver):
    name = "REVE-Squeezeformer-Ensemble5"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        submission_dir = meta["submission_dir"]
        os.environ["REVE_POSITIONS_PATH"] = str(submission_dir)
        n_chans, n_outputs = meta["n_chans"], meta["n_outputs"]
        # Members see TRAIN_CHANNELS order (Ensemble.forward permutes), so REVE's positions do too.
        chs_info = [{"ch_name": name} for name in TRAIN_CHANNELS]
        reve = lambda flatten: FineTunedReve(n_chans, n_outputs, chs_info, flatten)
        # Squeezeformers were trained on 120-sample (120 Hz) epochs.
        sqf = lambda power: FastTSConvSqueezeformer(n_outputs, 120, n_chans, power)
        model = Ensemble({
            "reve_r1": reve(False),
            "reve_r12": reve(False),
            "reve_r123": reve(True),
            "sqf": sqf(False),
            "sqf_power": sqf(True),
        }, [info["ch_name"] for info in meta["chs_info"]])
        state = torch.load(submission_dir / "weights.pt", map_location=meta["device"], weights_only=True)
        model.load_state_dict(state)
        return model.to(meta["device"]).eval()
