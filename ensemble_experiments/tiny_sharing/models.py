"""Ten paired TinyTSConv/TinyATM configurations for the controlled sharing study.

Isolated from train.py on purpose (tiny_sharing_plan.md sec. 9): nothing here is
imported by the running Nash/IV28 jobs.

Branch A = TS route (raw EEG time axis, image target InternViT33).
Branch B = ATM route (attention front, image target InternViT28).

C0        native independent pair (module/eeg_encoder verbatim)
C1        common bridge interface, everything untied
C2..C7    C1 plus tied groups T / S / R / N
C8/C9     attention-after-stem topology, untied (C8) and shared stem (C9)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.eeg_encoder.model import TSConv_parameterizable
from module.eeg_encoder.atm.atm import ATMS, Config, iTransformer

CHANNELS = 63
TIME = 250
ALIGN_DIM = 128
IMAGE_DIM = 3200
READOUT_DIM = 256
POOLED_POSITIONS = 8
BRIDGE_WIDTH = 12
D_MODEL = 112
STEM_POSITIONS = 36  # (250 - 25 + 1 - 51) // 5 + 1

# Native SubjectEmbedding sends the WHOLE batch to its shared token as soon as any
# id is out of range, so training composition (subject 10 present) and evaluation
# composition (one inner subject) would silently take different paths. Every
# forward, in every condition and phase, is pinned to the unknown/shared token by
# passing an out-of-range id. subject_embedding.weight is then dead weight and is
# reported as unreachable rather than trainable. gate.py audits the other case.
UNKNOWN_SUBJECT_ID = 10


class DeterministicAdaptivePool(nn.Module):
    """Adaptive (1, 8) averaging without CUDA's atomic-add backward kernel."""
    def forward(self, x):
        width = x.shape[-1]
        return torch.cat([
            x[..., (i * width) // POOLED_POSITIONS:
              math.ceil((i + 1) * width / POOLED_POSITIONS)].mean(
                  dim=(-2, -1), keepdim=True)
            for i in range(POOLED_POSITIONS)
        ], dim=-1)

CONFIG_IDS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

# tie groups per config: T temporal conv, S spatial conv + 1x1 projection,
# R the two readout dense matrices, N conv BN state + readout LayerNorm affine.
TIE_GROUPS = {
    "C1": (),
    "C2": ("T",),
    "C3": ("S",),
    "C4": ("R",),
    "C5": ("T", "S"),
    "C6": ("T", "S", "R"),
    "C7": ("T", "S", "R", "N"),
}

# module-construction seeds, so matched conditions share private initializations
SEED_STEM = 0
SEED_FRONT = 1
SEED_ALIGN_A = 2
SEED_ALIGN_B = 3
SEED_IMG_A = 4
SEED_IMG_B = 5
SEED_TOPO_STEM = 6
SEED_TOPO_ATTN = 7
SEED_TAIL = 8


def _seeded(seed, build):
    """Build a module under a fixed global RNG so matched conditions share inits."""
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        return build()
    finally:
        torch.random.set_rng_state(state)


def _dropout(x, p, training, gen):
    """Dropout drawn from an explicit per-branch stream (plan sec. 5)."""
    if not training or p <= 0:
        return x
    if gen is None:
        return F.dropout(x, p, True)
    keep = torch.empty_like(x).bernoulli_(1.0 - p, generator=gen)
    return x * keep / (1.0 - p)


class PairedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d plus the C7 pooled-moment path.

    ``pooled(xa, xb)`` normalizes both branches with equally weighted pooled
    moments, so neither the TS 36-position nor the ATM 8-position tensor
    dominates, and updates the running state once. Convention: population
    moments for normalization, Bessel-corrected pooled variance for
    running_var (PyTorch's own convention); eps/momentum are the module's
    defaults (1e-5 / 0.1).
    """

    def pooled(self, xa, xb):
        if not self.training:
            return self(xa), self(xb)
        dims = (0, 2, 3)
        ma, mb = xa.mean(dims), xb.mean(dims)
        va = xa.var(dims, unbiased=False)
        vb = xb.var(dims, unbiased=False)
        mean = 0.5 * (ma + mb)
        var = 0.5 * (va + vb) + 0.25 * (ma - mb) ** 2

        with torch.no_grad():
            n = xa[:, 0].numel() + xb[:, 0].numel()
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(
                self.momentum * var * n / max(n - 1, 1)
            )
            self.num_batches_tracked += 1

        shape = (1, -1, 1, 1)
        scale = (var + self.eps).rsqrt().view(shape)
        w, b = self.weight.view(shape), self.bias.view(shape)
        return tuple((x - mean.view(shape)) * scale * w + b for x in (xa, xb))


class Readout(nn.Module):
    """Linear(96,256) -> residual(GELU, Linear(256,256), Dropout) -> LayerNorm."""

    def __init__(self, in_dim, dim, dropout):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.p = dropout
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, gen=None):
        h = self.lin1(x)
        return self.norm(h + _dropout(self.lin2(F.gelu(h)), self.p, self.training, gen))


class BridgeTail(nn.Module):
    """spatial conv -> BN -> ELU -> dropout -> 1x1 -> pool8 -> readout(256)."""

    def __init__(self, width=BRIDGE_WIDTH, dropout=0.5):
        super().__init__()
        self.spatial = nn.Conv2d(width, width, (CHANNELS, 1))
        self.bn = PairedBatchNorm2d(width)
        self.p = dropout
        self.projection = nn.Conv2d(width, width, (1, 1))
        self.readout = Readout(POOLED_POSITIONS * width, READOUT_DIM, dropout)

    def to_bn(self, h):
        return self.spatial(h)

    def after_bn(self, h, gen=None):
        h = self.projection(_dropout(F.elu(h), self.p, self.training, gen))
        h = DeterministicAdaptivePool()(h)
        # position-major, then feature channel, in BOTH branches
        flat = h.squeeze(2).permute(0, 2, 1).reshape(h.shape[0], -1)
        return self.readout(flat, gen)


class BridgeStem(nn.Module):
    """temporal conv 1->12 (1,25) -> avgpool (1,51)/5 -> BN (ELU applied outside)."""

    def __init__(self, width=BRIDGE_WIDTH):
        super().__init__()
        self.temporal = nn.Conv2d(1, width, (1, 25))
        self.pool = nn.AvgPool2d((1, 51), (1, 5))
        self.bn = PairedBatchNorm2d(width)

    def to_bn(self, x):
        return self.pool(self.temporal(x))


class AttentionFront(nn.Module):
    """Native ATM attention front, with the 63 electrode rows made explicit.

    iTransformer.forward keeps ``[:, :63]`` of a 64-row stack whose row 0 is the
    subject token, i.e. it feeds the token and drops the last electrode. Here we
    take rows 1:64 so tied spatial filters see consistent row identities. This is
    an architectural bridge change, NOT an effect of sharing (plan sec. 2).
    """

    def __init__(self, legacy_crop=False):
        super().__init__()
        self.net = iTransformer(
            Config(CHANNELS, d_model=D_MODEL, n_heads=4, e_layers=1,
                   d_ff=D_MODEL, dropout=0.25)
        )
        self.legacy_crop = legacy_crop

    def forward(self, eeg):
        ids = torch.full((eeg.shape[0],), UNKNOWN_SUBJECT_ID,
                         device=eeg.device, dtype=torch.long)
        tokens = self.net.enc_embedding(eeg, None, ids)
        tokens, _ = self.net.encoder(tokens, attn_mask=None)
        rows = tokens[:, :CHANNELS] if self.legacy_crop else tokens[:, 1:CHANNELS + 1]
        return rows.unsqueeze(1)


class ElectrodeAttentionResidual(nn.Module):
    """C8/C9 attention AFTER the stem, over 63 electrode tokens of 12x36.

    Not the native ATM front: no subject token here. That topology difference is
    identical in C8 and C9, which is the whole point of having C8.
    """

    def __init__(self, width=BRIDGE_WIDTH, positions=STEM_POSITIONS):
        super().__init__()
        desc = width * positions
        self.shape = (width, positions)
        self.norm = nn.LayerNorm(desc)
        self.down = nn.Linear(desc, D_MODEL)
        self.encoder = nn.TransformerEncoderLayer(
            D_MODEL, 4, dim_feedforward=D_MODEL, dropout=0.25,
            activation="gelu", batch_first=True,
        )
        self.up = nn.Linear(D_MODEL, desc)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, stem):
        b = stem.shape[0]
        desc = stem.permute(0, 2, 1, 3).reshape(b, CHANNELS, -1)
        h = self.up(self.encoder(self.down(self.norm(desc))))
        h = h.reshape(b, CHANNELS, *self.shape).permute(0, 2, 1, 3)
        return stem + torch.tanh(self.gamma) * h


class PairedModel(nn.Module):
    """One condition: two EEG branches, two private linear image projectors."""

    def __init__(self, config, master_seed=3300, atm_seed=4300):
        super().__init__()
        if config not in CONFIG_IDS:
            raise ValueError(f"unknown config {config}")
        self.config = config
        self.ties = TIE_GROUPS.get(config, ())
        self.shared_stem = config == "C9"
        s = master_seed

        if config == "C0":
            self._build_native(s, atm_seed)
        elif config in TIE_GROUPS:
            self._build_bridge(s)
        else:
            self._build_topology(s)

        backbone_b = 128 if config == "C0" else READOUT_DIM
        self.align_a = _seeded(s + SEED_ALIGN_A, lambda: nn.Linear(READOUT_DIM, ALIGN_DIM))
        self.align_b = _seeded(s + SEED_ALIGN_B, lambda: nn.Linear(backbone_b, ALIGN_DIM))
        self.img_a = _seeded(s + SEED_IMG_A, lambda: nn.Linear(IMAGE_DIM, ALIGN_DIM))
        self.img_b = _seeded(s + SEED_IMG_B, lambda: nn.Linear(IMAGE_DIM, ALIGN_DIM))
        self._gens = None

    # ---------------- builders ----------------

    def _build_native(self, s, atm_seed):
        self.ts = _seeded(s, lambda: TSConv_parameterizable(
            feature_dim=READOUT_DIM, eeg_sample_points=TIME, channels_num=CHANNELS,
            temporal_filters=10, temporal_kernel=25, pool_kernel=51, pool_stride=5,
            spatial_filters=10, projection_filters=10,
        ))
        self.atm = _seeded(atm_seed, lambda: ATMS(
            channels_num=CHANNELS, feature_dim=128, eeg_sample_points=TIME,
            d_model=D_MODEL, n_heads=4, e_layers=1, d_ff=D_MODEL,
            temporal_filters=12, temporal_kernel=25, pool_kernel=51, pool_stride=5,
            spatial_filters=12, projection_filters=12,
        ))

    def _build_bridge(self, s):
        # untied copies start from the SAME template values the tied variant uses,
        # so tying is never confounded with a different initialization (plan sec. 5)
        stem = _seeded(s + SEED_STEM, BridgeStem)
        tail = _seeded(s + SEED_TAIL, BridgeTail)
        self.stem_a, self.stem_b = copy.deepcopy(stem), copy.deepcopy(stem)
        self.tail_a, self.tail_b = copy.deepcopy(tail), copy.deepcopy(tail)

        if "T" in self.ties:
            self.stem_b.temporal = self.stem_a.temporal
        if "S" in self.ties:
            self.tail_b.spatial = self.tail_a.spatial
            self.tail_b.projection = self.tail_a.projection
        if "R" in self.ties:
            # R is the two dense matrices only; the LayerNorm belongs to N
            self.tail_b.readout.lin1 = self.tail_a.readout.lin1
            self.tail_b.readout.lin2 = self.tail_a.readout.lin2
        if "N" in self.ties:
            self.stem_b.bn = self.stem_a.bn
            self.tail_b.bn = self.tail_a.bn
            self.tail_b.readout.norm = self.tail_a.readout.norm

        self.front_b = _seeded(s + SEED_FRONT, AttentionFront)

    def _build_topology(self, s):
        stem = _seeded(s + SEED_TOPO_STEM, BridgeStem)
        tail = _seeded(s + SEED_TAIL, BridgeTail)
        self.stem_a = stem
        # C8's second copy starts from exactly the tensors C9's single stem holds
        self.stem_b = stem if self.shared_stem else copy.deepcopy(stem)
        self.tail_a, self.tail_b = copy.deepcopy(tail), copy.deepcopy(tail)
        self.attn_b = _seeded(s + SEED_TOPO_ATTN, ElectrodeAttentionResidual)

    # ---------------- per-branch dropout streams ----------------

    def _gen(self, index, device):
        if self._gens is None:
            self._gens = [torch.Generator(device=device).manual_seed(9100 + i)
                          for i in range(2)]
        return self._gens[index]

    # ---------------- forwards ----------------

    def forward(self, eeg_a, eeg_b):
        """Returns (align_a, align_b), the two 128-D EEG embeddings."""
        if self.config == "C0":
            ids = torch.full((eeg_b.shape[0],), UNKNOWN_SUBJECT_ID,
                             device=eeg_b.device, dtype=torch.long)
            ha, hb = self.ts(eeg_a), self.atm(eeg_b, ids)
        elif self.config in TIE_GROUPS:
            ha, hb = self._forward_bridge(eeg_a, eeg_b)
        else:
            ha, hb = self._forward_topology(eeg_a, eeg_b)
        return self.align_a(ha), self.align_b(hb)

    def _forward_bridge(self, eeg_a, eeg_b):
        ga, gb = self._gen(0, eeg_a.device), self._gen(1, eeg_a.device)
        xa = self.stem_a.to_bn(eeg_a.unsqueeze(1))
        xb = self.stem_b.to_bn(self.front_b(eeg_b))
        tie_n = "N" in self.ties

        if tie_n:
            xa, xb = self.stem_a.bn.pooled(xa, xb)
        else:
            xa, xb = self.stem_a.bn(xa), self.stem_b.bn(xb)

        xa = self.tail_a.to_bn(F.elu(xa))
        xb = self.tail_b.to_bn(F.elu(xb))

        if tie_n:
            xa, xb = self.tail_a.bn.pooled(xa, xb)
        else:
            xa, xb = self.tail_a.bn(xa), self.tail_b.bn(xb)

        return self.tail_a.after_bn(xa, ga), self.tail_b.after_bn(xb, gb)

    def _forward_topology(self, eeg_a, eeg_b):
        ga, gb = self._gen(0, eeg_a.device), self._gen(1, eeg_a.device)
        if self.shared_stem:
            # C9: the duplicate stem forward is genuinely removed
            act = F.elu(self.stem_a.bn(self.stem_a.to_bn(eeg_a.unsqueeze(1))))
            act_a = act_b = act
        else:
            act_a = F.elu(self.stem_a.bn(self.stem_a.to_bn(eeg_a.unsqueeze(1))))
            act_b = F.elu(self.stem_b.bn(self.stem_b.to_bn(eeg_b.unsqueeze(1))))
        ha = self.tail_a.after_bn(self.tail_a.bn(self.tail_a.to_bn(act_a)), ga)
        hb = self.attn_b(act_b)
        hb = self.tail_b.after_bn(self.tail_b.bn(self.tail_b.to_bn(hb)), gb)
        return ha, hb

    def project_images(self, feat_a, feat_b):
        return self.img_a(feat_a), self.img_b(feat_b)


def _unique(params):
    seen, out = {}, []
    for p in params:
        if id(p) not in seen:
            seen[id(p)] = True
            out.append(p)
    return out


# Registered but never reached by our forward. The shared/unknown token IS reached
# (see UNKNOWN_SUBJECT_ID), so it is deliberately not in this list.
DEAD_MARKS = ("subject_wise_linear", "subject_embedding.subject_embedding",
              "subject_embedding.mask_embedding", "temporal_embedding", "mask_token")


def parameter_report(model):
    """Registered / unique / EEG-side / image-side / unreachable counts."""
    named = list(model.named_parameters(remove_duplicate=False))
    eeg = [(n, p) for n, p in named if not n.startswith(("img_a", "img_b"))]
    img = [p for n, p in named if n.startswith(("img_a", "img_b"))]
    unique_all = _unique(p for _, p in named)
    return {
        "registered_total": sum(p.numel() for _, p in named),
        "unique_total": sum(p.numel() for p in unique_all),
        "eeg_registered": sum(p.numel() for _, p in eeg),
        "eeg_unique": sum(p.numel() for p in _unique(p for _, p in eeg)),
        "image_side": sum(p.numel() for p in img),
        "eeg_unreachable": sum(
            p.numel() for n, p in eeg if any(m in n for m in DEAD_MARKS)
        ),
        "shared_parameters": (
            sum(p.numel() for _, p in named) - sum(p.numel() for p in unique_all)
        ),
    }


def tied_tensor_names(model):
    """Map id(param) -> the >1 registered names that alias it (manifest field)."""
    groups = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    return {names[0]: names for names in groups.values() if len(names) > 1}
