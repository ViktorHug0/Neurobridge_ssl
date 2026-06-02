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
