import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        init_temperature,
        alpha,
        eeg_l2norm: bool,
        img_l2norm: bool,
        learnable: bool,
        is_softplus: bool,
        logit_scale_max=None,
        mse_on_raw: bool = False,
        asymmetric: bool = False,
    ):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
        self.mse_on_raw = mse_on_raw  # ENIGMA: MSE on unnormalized target (learns CLIP magnitude)
        # NeuralBench's ClipLoss(symmetric=False) scores only EEG->image, the direction the
        # grader actually evaluates. Averaging both directions is this repo's default.
        self.asymmetric = asymmetric
        
        self.is_softplus = is_softplus
        self.logit_scale_max = logit_scale_max
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature), requires_grad=learnable)
        self.softplus = nn.Softplus()

    def _get_logit_scale(self):
        if self.is_softplus:
            scale = self.softplus(self.logit_scale)
        else:
            scale = torch.exp(self.logit_scale)
        if self.logit_scale_max is not None:
            scale = scale.clamp(max=float(self.logit_scale_max))
        return scale

    def _normalize_inputs(self, eeg_feature, image_feature):
        if self.eeg_l2norm:
            eeg_feature = F.normalize(eeg_feature, p=2, dim=1)
        if self.img_l2norm:
            image_feature = F.normalize(image_feature, p=2, dim=1)
        return eeg_feature, image_feature

    @staticmethod
    def _multi_positive_cross_entropy(logits, positive_mask, row_weights=None):
        valid_rows = positive_mask.any(dim=1)
        if not torch.any(valid_rows):
            return logits.new_tensor(0.0)

        log_probs = torch.log_softmax(logits, dim=1)
        positives = positive_mask[valid_rows]
        
        # Avoid nan from 0 * -inf by using torch.where to mask out non-positives
        positive_log_probs = torch.where(positives, log_probs[valid_rows], torch.zeros_like(log_probs[valid_rows]))
        
        loss_per_row = -positive_log_probs.sum(dim=1) / positives.sum(dim=1).clamp_min(1)
        if row_weights is None:
            return loss_per_row.mean()
        weights = row_weights[valid_rows].to(dtype=loss_per_row.dtype)
        return (loss_per_row * weights).sum() / weights.sum().clamp_min(1e-8)

    def multi_positive_pair_loss(
        self, query_feature, key_feature, positive_mask,
        query_scale=None, sample_weights=None,
    ):
        if self.eeg_l2norm:
            query_feature = F.normalize(query_feature, p=2, dim=1)
        if self.img_l2norm:
            key_feature = F.normalize(key_feature, p=2, dim=1)
        logit_scale = self._get_logit_scale()
        logits = torch.matmul(query_feature, key_feature.T) * logit_scale
        if query_scale is not None:
            logits_qk = logits * query_scale.unsqueeze(1)
        else:
            logits_qk = logits
        loss_qk = self._multi_positive_cross_entropy(
            logits_qk, positive_mask, sample_weights
        )
        loss_kq = self._multi_positive_cross_entropy(
            logits.T, positive_mask.T, sample_weights
        )
        return (loss_qk + loss_kq) / 2

    @staticmethod
    def rsa_loss(eeg_feature, image_feature, same_image_mask):
        """1 - RSA: Pearson correlation between the off-diagonal EEG-EEG and image-image cosine matrices of the batch,
        both clouds mean-centred first (the CVPR E8.4 diagnostic, error_decomposition.py `rsa`). Pairs showing the
        same image (same_image_mask, incl. the diagonal) are dropped: with grouped batches they would dominate."""
        ze = F.normalize(eeg_feature - eeg_feature.mean(0), p=2, dim=1)
        zi = F.normalize((image_feature - image_feature.mean(0)).detach(), p=2, dim=1)
        keep = ~same_image_mask
        a, b = (ze @ ze.T)[keep], (zi @ zi.T)[keep]
        a, b = a - a.mean(), b - b.mean()
        return 1 - (a * b).sum() / (a.norm() * b.norm() + 1e-8)

    @staticmethod
    def isometry_loss(eeg_feature, image_idx, subject_id):
        """Mean over source-subject pairs of the residual of the best rotation between their embeddings of the batch's
        shared images (CVPR E8.11), split-half: R = U V^T from the SVD of A1^T B1 on half of the images (detached), scored
        as 1 - tr(R^T A2^T B2) / (|A2| |B2|) on the other half, both ways. A and B are row-normalised then centred.
        In-sample it would be near zero for any pair: a batch holds fewer shared images (~113) than dimensions."""
        subs = subject_id.unique().tolist()
        if len(subs) < 2:
            return eeg_feature.new_zeros(())
        rows = {}
        for s in subs:
            idx = (subject_id == s).nonzero(as_tuple=True)[0]
            rows[s] = {int(im): int(r) for im, r in zip(image_idx[idx].tolist(), idx.tolist())}  # ponytail: last row wins on duplicate images
        common = sorted(set.intersection(*(set(r) for r in rows.values())))
        if len(common) < 3:
            return eeg_feature.new_zeros(())
        z = F.normalize(eeg_feature, p=2, dim=1)
        clouds = {s: z[[rows[s][im] for im in common]] for s in subs}
        clouds = {s: c - c.mean(0) for s, c in clouds.items()}
        perm = torch.randperm(len(common), device=eeg_feature.device)
        halves = (perm[: len(common) // 2], perm[len(common) // 2:])
        losses = []
        for i, s in enumerate(subs):
            for t in subs[i + 1:]:
                for fit, ev in (halves, halves[::-1]):
                    a, b = clouds[s][fit], clouds[t][fit]
                    u, _, vt = torch.linalg.svd((a.T @ b).detach().float())
                    r = (u @ vt).to(a.dtype)
                    a, b = clouds[s][ev], clouds[t][ev]
                    losses.append(1 - (a @ r * b).sum() / (a.norm() * b.norm() + 1e-8))
        return torch.stack(losses).mean()

    @staticmethod
    def collapse_penalty(eeg_feature, pr_floor):
        """relu(1/PR - 1/pr_floor) of the row-normalised, centred batch EEG cloud, PR = tr(C)^2 / tr(C^2) its
        participation ratio (effective dimension). Guards the E8.11 geometric losses, which both collapse the cloud."""
        z = F.normalize(eeg_feature, p=2, dim=1)
        z = z - z.mean(0)
        c = z.T @ z / z.shape[0]
        return F.relu((c * c).sum() / torch.diagonal(c).sum().pow(2) - 1.0 / pr_floor)

    def forward(self, eeg_feature, image_feature, eeg_confidence=None):
        eeg_raw, image_raw = eeg_feature, image_feature
        eeg_feature, image_feature = self._normalize_inputs(eeg_feature, image_feature)

        # Calculate similarity matrix (N x N)
        logit_scale = self._get_logit_scale()
        similarity_matrix_ie = torch.matmul(eeg_feature, image_feature.T) * logit_scale
        if eeg_confidence is not None:
            similarity_matrix_ie_q = similarity_matrix_ie * eeg_confidence.unsqueeze(1)
        else:
            similarity_matrix_ie_q = similarity_matrix_ie

        # Construct labels
        labels = torch.arange(eeg_feature.shape[0], device=eeg_feature.device)

        # Calculate two parts of the loss
        loss_eeg_ie = self.criterion_cls(similarity_matrix_ie_q, labels)
        loss_img_ie = self.criterion_cls(similarity_matrix_ie.T, labels)

        if self.alpha != 1.0:
            if self.mse_on_raw:
                loss_mse = self.criterion_mse(eeg_raw, image_raw)
            else:
                loss_mse = self.criterion_mse(eeg_feature, image_feature)

        loss_contrastive = (
            loss_eeg_ie if self.asymmetric else (loss_eeg_ie + loss_img_ie) / 2
        )

        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive

        return loss
