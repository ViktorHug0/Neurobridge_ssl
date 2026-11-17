import torch


def shift(x, n):
    """x(t) -> x(t - n): later by n samples (n < 0: earlier); edges replicate."""
    if n == 0:
        return x
    return torch.cat([x[..., :1].expand(*x.shape[:-1], n), x[..., :-n]], -1) if n > 0 else torch.cat([x[..., -n:], x[..., -1:].expand(*x.shape[:-1], -n)], -1)
