import torch.nn as nn


def _build_out_activation(activation: str) -> nn.Module:
    if activation == 'none':
        return nn.Identity()
    if activation == 'relu':
        return nn.ReLU()
    raise ValueError(f"Unsupported projector activation: {activation}")


class ProjectorLinear(nn.Module):
    def __init__(self, input_dim, output_dim, activation='none'):
        super(ProjectorLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.out_activation = _build_out_activation(activation)

    def forward(self, x):
        return self.out_activation(self.linear(x))


class ProjectorDirect(nn.Module):
    def __init__(self, activation='none'):
        super(ProjectorDirect, self).__init__()
        self.out_activation = _build_out_activation(activation)

    def forward(self, x):
        return self.out_activation(x)
