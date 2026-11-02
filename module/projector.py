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


def build_projector(projector_type, input_dim, output_dim, activation='none'):
    activation = (activation or 'none').lower()
    if activation not in ('none', 'relu'):
        raise ValueError(f"Unsupported projector activation: {activation}")
    if projector_type == 'bottleneck':  # EEG side: encoder emits a low-rank code (--eeg_backbone_dim r), lifted
        # linearly to the target space; image/text side: identity, so outputs stay in the grader's DINOv2 space
        if input_dim == output_dim:
            return ProjectorDirect(activation=activation)
        return nn.Linear(input_dim, output_dim)
    if projector_type == 'direct':
        if input_dim != output_dim:
            raise ValueError(
                f"ProjectorDirect requires input_dim == output_dim, got {input_dim} != {output_dim}."
            )
        return ProjectorDirect(activation=activation)
    if projector_type == 'linear':
        return ProjectorLinear(input_dim, output_dim, activation=activation)
    raise ValueError(f"Unsupported projector type: {projector_type}")
