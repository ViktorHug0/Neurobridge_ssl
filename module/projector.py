import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUGeluReparam(nn.Module):
    """Forward ReLU(z); backward via GELU (Sparse CLIP dead-neuron trick, Eq. 7)."""

    def forward(self, x):
        return F.relu(x).detach() + F.gelu(x) - F.gelu(x).detach()


class TopKActivation(nn.Module):
    """Keep only the top-k activations per sample; zero the rest (Sparse CLIP TopK ablation)."""

    def __init__(self, k: int):
        super().__init__()
        if k <= 0:
            raise ValueError(f"TopKActivation requires k > 0, got {k}")
        self.k = int(k)

    def forward(self, x):
        k = min(self.k, x.shape[-1])
        if k <= 0:
            return x
        topk = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk.indices, topk.values)
        return out


def _build_out_activation(activation: str, topk: int = 0) -> nn.Module:
    if activation == 'none':
        return nn.Identity()
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'relu_gelu':
        return ReLUGeluReparam()
    if activation == 'topk':
        return TopKActivation(topk)
    raise ValueError(f"Unsupported projector activation: {activation}")


class ProjectorLinear(nn.Module):
    def __init__(self, input_dim, output_dim, activation='none', topk=0):
        super(ProjectorLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.out_activation = _build_out_activation(activation, topk=topk)

    def forward(self, x):
        return self.out_activation(self.linear(x))


class ProjectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, activation='none', topk=0):
        super(ProjectorMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.out_activation = _build_out_activation(activation, topk=topk)

    def forward(self, x):
        return self.out_activation(self.mlp(x))


class ProjectorDirect(nn.Module):
    def __init__(self, activation='none', topk=0):
        super(ProjectorDirect, self).__init__()
        self.out_activation = _build_out_activation(activation, topk=topk)

    def forward(self, x):
        return self.out_activation(x)


class SubspaceProjector(nn.Module):
    """EEG side of --projector subspace: k predicted coefficients -> target mean + top-k principal directions of
    the TRAINING-image targets (fixed buffers: fitted once by train.py, stored in the checkpoint). Retrieval of
    clean DINOv2 test targets survives a 64-PC projection intact (top-5 100%), while the other ~1470 output
    dimensions only give an EEG regressor room to fit noise."""

    def __init__(self, k, out_dim, centered=False):
        # centered=True ('subspace_c'): output only the principal-direction part. The target mean (norm ~5.1)
        # outweighs a typical image's deviation (~3.4), so with it every cosine is dominated by the shared mean
        # (tsc_sub64 fold 1: -7.5 vs TSConv); centered keeps the clean-target ceiling (64 PCs: top-5 100%).
        super().__init__()
        self.centered = centered
        self.register_buffer('mean', torch.zeros(out_dim))
        self.register_buffer('basis', torch.zeros(k, out_dim))

    @torch.no_grad()
    def fit(self, targets):
        t = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, self.mean.numel())
        mu = t.mean(0)
        _, _, vt = torch.linalg.svd(t - mu, full_matrices=False)
        self.mean.copy_(mu.to(self.mean.device))
        self.basis.copy_(vt[: self.basis.shape[0]].to(self.basis.device))

    def forward(self, z):
        return z @ self.basis if self.centered else self.mean + z @ self.basis
