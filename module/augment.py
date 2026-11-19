"""EEG augmentations: trial-level transforms (numpy, applied in the dataset) and batch-level
cross-subject mixing (torch, applied in the training loop)."""
import random

import numpy as np
import torch


class RandomGaussianNoise:
    """
    Adds random Gaussian noise to the EEG signal.
    std indicates the standard deviation of the noise.
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, size=eeg_data.shape)
        return eeg_data + noise


class RandomSmooth:
    """
    A simple smoothing operation, which can be understood as a simple convolution / moving average along the time axis.
    kernel_size indicates the size of the moving average kernel.
    """
    def __init__(self, kernel_size=5, smooth_prob=0.5):
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        ch, time_len = eeg_data.shape
        smoothed = np.copy(eeg_data)
        # Ensure kernel size is odd
        for c in range(ch):
            if np.random.rand() < self.smooth_prob:
                for t in range(time_len):
                    left = max(0, t - self.kernel_size // 2)
                    right = min(time_len, t + self.kernel_size // 2 + 1)
                    smoothed[c, t] = np.mean(eeg_data[c, left:right])
        return smoothed


def cross_subject_stimulus_mix(
    features, object_indices, image_indices, subject_ids, alpha=1.0, mixup_type='pairwise',
    return_mixup_metadata=False,
):
    if features.shape[0] < 2:
        if return_mixup_metadata:
            return features, {'partner_indices': None, 'mixed_mask': None}
        return features

    obj = object_indices.detach().cpu().tolist()
    img = image_indices.detach().cpu().tolist()
    sid = subject_ids.detach().cpu().tolist()

    groups = {}
    for i, (o, im) in enumerate(zip(obj, img)):
        groups.setdefault((int(o), int(im)), []).append(i)

    mixed = features.clone()
    partner_indices = torch.arange(features.shape[0], device=features.device)
    mixed_mask = torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
    lam_all = torch.ones(features.shape[0], device=features.device)  # weight of the row's own trial (pairwise mixing)
    dist_alpha = max(float(alpha), 1e-3)

    for indices in groups.values():
        group_size = len(indices)
        if group_size < 2:
            continue

        group_idx = torch.tensor(indices, device=features.device, dtype=torch.long)
        group_features = features[group_idx]

        if mixup_type == 'group':
            concentration = torch.full((group_size,), dist_alpha, device=features.device, dtype=torch.float32)
            weights = torch.distributions.Dirichlet(concentration).sample((group_size,)).to(group_features.dtype)
            mixed_group = torch.einsum('ab,b...->a...', weights, group_features)
        else:
            group_subject_ids = [sid[i] for i in indices]
            if len(set(group_subject_ids)) == group_size:
                order = torch.randperm(group_size, device=features.device)
                offset = random.randint(1, group_size - 1)
                partner_pos = torch.empty(group_size, device=features.device, dtype=torch.long)
                partner_pos[order] = torch.roll(order, shifts=offset)
            else:
                partner_pos = []
                for pos, subject_id in enumerate(group_subject_ids):
                    candidates = [j for j, sid_j in enumerate(group_subject_ids) if j != pos and sid_j != subject_id]
                    if not candidates:
                        candidates = [j for j in range(group_size) if j != pos]
                    partner_pos.append(random.choice(candidates))
                partner_pos = torch.tensor(partner_pos, device=features.device, dtype=torch.long)

            partner_indices[group_idx] = group_idx[partner_pos]
            mixed_mask[group_idx] = True
            concentration = torch.full((group_size,), dist_alpha, device=features.device, dtype=torch.float32)
            beta = torch.distributions.Beta(concentration, concentration)
            lam = beta.sample().to(group_features.dtype)
            lam_all[group_idx] = lam.float()
            lam = lam.view(*([group_size] + [1] * (group_features.dim() - 1)))
            mixed_group = lam * group_features + (1.0 - lam) * group_features[partner_pos]

        mixed[group_idx] = mixed_group

    if return_mixup_metadata:
        return mixed, {'partner_indices': partner_indices, 'mixed_mask': mixed_mask, 'lam': lam_all}
    return mixed


DONORS = {}  # --op_donors path -> (K, C, C) operators, loaded once


def cross_subject_operator_transfer(eeg_batch, subject_ids, shrink=0.05, eps=1e-8, boundary=0, donors=''):
    """do(subject := t): re-render a trial through another subject's channel-mixing operator.

    Model the sensor signal as x_s = A_s . (sources), so the subject-specific nuisance is the
    channel mixing A_s, observable through the channel covariance R_s. Mapping
    x_s -> R_t^(1/2) R_s^(-1/2) x_s swaps the recording operator while leaving the stimulus-driven
    sources untouched, so the image label is preserved exactly -- unlike convex mixup, which also
    blends the signals themselves.

    R_s is estimated per batch (~113 trials x 250 samples per subject, plenty for 63x63), which
    also injects a little estimation noise as free regularization. One target is drawn per source
    subject per batch; over an epoch all ordered pairs get covered.
    """
    channels = eeg_batch.shape[-2]
    uniq = torch.unique(subject_ids)
    if uniq.numel() < 2:
        return eeg_batch

    eye = torch.eye(channels, device=eeg_batch.device, dtype=torch.float64)

    def _spd(mat):
        evals, evecs = torch.linalg.eigh(mat)
        return evals.clamp_min(eps), evecs

    sqrt_ops, inv_sqrt_ops = {}, {}
    for s in uniq.tolist():
        flat = eeg_batch[subject_ids == s].reshape(-1, channels, eeg_batch.shape[-1]).to(torch.float64)
        cov = torch.einsum('nct,ndt->cd', flat, flat) / (flat.shape[0] * flat.shape[-1])
        cov = (1.0 - shrink) * cov + shrink * eye * (torch.diagonal(cov).mean())
        evals, evecs = _spd(cov)
        sqrt_ops[s] = (evecs * evals.sqrt()) @ evecs.T
        inv_sqrt_ops[s] = (evecs * evals.rsqrt()) @ evecs.T

    out = eeg_batch.clone()
    donors = DONORS.setdefault(donors, torch.as_tensor(np.load(donors)['S'])) if donors else None
    for s, j in zip(uniq.tolist(), draw_op_partners(subject_ids, boundary)):
        # donors (K, C, C): R^(1/2) of people of other cohorts (scripts/mi/nq08_op_donors.py). ponytail: half of the partners are donors, no knob
        target = donors[random.randrange(len(donors))].to(eye) if donors is not None and random.random() < 0.5 else None
        if target is None and s == j:
            continue
        mask = subject_ids == s
        transfer = (sqrt_ops[j] if target is None else target) @ inv_sqrt_ops[s]
        out[mask] = torch.einsum('cd,...dt->...ct', transfer.to(eeg_batch.dtype), eeg_batch[mask]).to(out.dtype)  # autocast einsum returns bf16
    return out


FUNC_OPS = None


def functional_operator_transfer(eeg_batch, subject_ids, ops_paths):
    """do(subject := t) with decoder-fitted operators: x_s -> M_t^-1 M_s x_s.

    M_s (channels x channels) is the spatial re-mix that best aligns person s to a model that never trained on it
    (scripts/mi/ne19_fit_operators.py, MI_NEURIPS_LOG E17-E19), so M_t^-1 M_s re-renders s as t along the directions
    that matter to the decoder. People without an operator are left alone; the target is any other person whose
    operator is well conditioned.
    """
    global FUNC_OPS
    if FUNC_OPS is None:
        ops = {}
        for path in ops_paths:
            ops.update(torch.load(path))
        # targets need a well-conditioned operator: inverting a near-singular one would blow one direction up
        FUNC_OPS = {s: m.double() for s, m in ops.items()}, {s: torch.linalg.inv(m.double()) for s, m in ops.items() if torch.linalg.cond(m.double()) < 10}
    fwd, inv = FUNC_OPS
    out = eeg_batch.clone()
    for s in torch.unique(subject_ids).tolist():
        if s not in fwd:
            continue
        t = random.choice([j for j in inv if j != s])
        mask = subject_ids == s
        transfer = (inv[t] @ fwd[s]).to(eeg_batch.device, eeg_batch.dtype)
        out[mask] = torch.einsum('cd,...dt->...ct', transfer, eeg_batch[mask]).to(out.dtype)
    return out


def draw_op_partners(subject_ids, boundary=0):
    """One random partner per subject in the batch (same headset only when boundary > 0: THINGS-EEG2 ids >= boundary)."""
    uniq = torch.unique(subject_ids).tolist()
    if boundary:
        return [random.choice([j for j in uniq if (j >= boundary) == (s >= boundary)]) for s in uniq]
    return [random.choice(uniq) for _ in uniq]
