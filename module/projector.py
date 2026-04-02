import torch.nn as nn
import torch.nn.functional as F


class ProjectorLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectorLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class ProjectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectorMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class ProjectorDirect(nn.Module):
    def __init__(self):
        super(ProjectorDirect, self).__init__()

    def forward(self, x):
        return x


class ResidualAdapter(nn.Module):
    """Small CLAP-style residual adapter on top of aligned features."""
    def __init__(self, d_in, d_mid, alpha_inference=1.0):
        super().__init__()
        self.alpha_inference = alpha_inference
        self.linear1 = nn.Linear(d_in, d_mid)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(d_mid, d_in, bias=False)
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x):
        alpha = 1.0 if self.training else self.alpha_inference
        h = self.linear2(self.act(self.linear1(x)))
        return x + alpha * h