import torch
from torch import nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def grad_reverse(x, coeff=1.0):
    return _GradReverse.apply(x, coeff)


class SubjectClassifier(nn.Module):
    def __init__(self, input_dim, num_subjects):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_subjects)

    def forward(self, x):
        return self.linear(x)


def decorrelation_loss(z_content, z_style):
    zc = z_content - z_content.mean(dim=0, keepdim=True)
    zs = z_style - z_style.mean(dim=0, keepdim=True)
    cross_cov = (zc.T @ zs) / max(1, zc.shape[0] - 1)
    return (cross_cov ** 2).mean()


def _covariance(z):
    z = z - z.mean(dim=0, keepdim=True)
    return (z.T @ z) / max(1, z.shape[0] - 1)


def coral_subject_loss(z_content, subject_ids):
    unique_subjects = torch.unique(subject_ids)
    if unique_subjects.numel() < 2:
        return z_content.new_tensor(0.0)

    covs = []
    for sid in unique_subjects:
        z_sid = z_content[subject_ids == sid]
        if z_sid.shape[0] < 2:
            continue
        covs.append(_covariance(z_sid))

    if len(covs) < 2:
        return z_content.new_tensor(0.0)

    loss = z_content.new_tensor(0.0)
    pairs = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + torch.mean((covs[i] - covs[j]) ** 2)
            pairs += 1
    return loss / pairs
