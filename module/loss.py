import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        init_temperature,
        alpha,
        beta,
        eeg_l2norm: bool,
        img_l2norm: bool,
        text_l2norm: bool,
        learnable: bool,
        is_softplus: bool,
        logit_scale_max=None,
        mse_on_raw: bool = False,
        asymmetric: bool = False,
        soft_target_tau: float = 0.0,
        soft_mix_beta: float = 0.0,
    ):
        super(ContrastiveLoss, self).__init__()
        # > 0: EEG->image targets are softmax(image-image cosine / tau) instead of one-hot, so
        # near-identical images (and same-image duplicates in the batch) are not hard negatives.
        self.soft_target_tau = soft_target_tau
        # > 0 (NeuroCLIP): target = (1 - beta) one-hot + beta softmax(image-image cos / tau), a mild similarity smoothing
        self.soft_mix_beta = soft_mix_beta
        self.alpha = alpha
        self.beta = beta
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
        self.text_l2norm = text_l2norm
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

    def _normalize_inputs(self, eeg_feature, image_feature, text_feature=None):
        if self.eeg_l2norm:
            eeg_feature = F.normalize(eeg_feature, p=2, dim=1)
        if image_feature is not None and self.img_l2norm:
            image_feature = F.normalize(image_feature, p=2, dim=1)
        if text_feature is not None and self.text_l2norm:
            text_feature = F.normalize(text_feature, p=2, dim=1)
        return eeg_feature, image_feature, text_feature

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
        self, query_feature, key_feature, positive_mask, key_is_text=False,
        query_scale=None, sample_weights=None,
    ):
        if self.eeg_l2norm:
            query_feature = F.normalize(query_feature, p=2, dim=1)
        if key_is_text:
            if self.text_l2norm:
                key_feature = F.normalize(key_feature, p=2, dim=1)
        elif self.img_l2norm:
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

    def multi_positive_row_losses(self, query_feature, key_feature, positive_mask):
        """Per-row (query->key) multi-positive CE, unreduced. Used for per-environment risks."""
        if self.eeg_l2norm:
            query_feature = F.normalize(query_feature, p=2, dim=1)
        if self.img_l2norm:
            key_feature = F.normalize(key_feature, p=2, dim=1)
        logits = torch.matmul(query_feature, key_feature.T) * self._get_logit_scale()
        log_probs = torch.log_softmax(logits, dim=1)
        positive_log_probs = torch.where(positive_mask, log_probs, torch.zeros_like(log_probs))
        return -positive_log_probs.sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)

    def brain_to_brain_loss(self, eeg_feature, positive_mask):
        """Same-stimulus cross-subject EEG<->EEG InfoNCE.

        Every batch holds `samples_per_image` subjects viewing the SAME image, so the only
        common cause of agreement between two rows is the stimulus. Self-pairs are removed
        from both the positives and the denominator, so a row cannot win by matching itself.
        """
        z = F.normalize(eeg_feature, p=2, dim=1)
        logits = torch.matmul(z, z.T) * self._get_logit_scale()
        eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(eye, float('-inf'))
        return self._multi_positive_cross_entropy(logits, positive_mask & ~eye)

    def relational_loss(self, eeg_feature, image_feature, temperature=0.07):
        """RKD-style: EEG-EEG similarity structure should match image-image structure.

        Predicts *relative* geometry instead of absolute coordinates, so anything that acts
        as a per-subject rotation of the embedding space drops out of the objective.
        """
        ze = F.normalize(eeg_feature, p=2, dim=1)
        zi = F.normalize(image_feature, p=2, dim=1)
        eye = torch.eye(ze.shape[0], dtype=torch.bool, device=ze.device)
        se = torch.matmul(ze, ze.T).masked_fill(eye, float('-inf')) / temperature
        si = torch.matmul(zi, zi.T).masked_fill(eye, float('-inf')) / temperature
        log_q = F.log_softmax(se, dim=1)
        log_p = F.log_softmax(si.detach(), dim=1)
        # KL row-wise, with the self-position dropped *after* the product: exp(-inf)*(-inf - -inf)
        # is nan, so it has to be overwritten rather than multiplied away.
        kl = (log_p.exp() * (log_p - log_q)).masked_fill(eye, 0.0)
        return kl.sum(dim=1).mean()

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

    @staticmethod
    def mk_mmd(x, y, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
        """Multi-kernel RBF MMD between the EEG and image point clouds in the shared space.

        SAMGA's coarse stage: shrink the cross-modal distribution gap before asking the space to
        be instance-discriminative. Bandwidth is the median pairwise distance of the batch.
        """
        z = torch.cat([F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)], dim=0)
        d2 = torch.cdist(z, z).pow(2)
        bandwidth = d2.detach().median().clamp_min(1e-6)
        kernel = sum(torch.exp(-d2 / (bandwidth * s)) for s in scales) / len(scales)
        n = x.shape[0]
        return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2.0 * kernel[:n, n:].mean()

    def forward(self, eeg_feature, image_feature, text_feature, eeg_confidence=None):
        eeg_raw, image_raw = eeg_feature, image_feature
        eeg_feature, image_feature, text_feature = self._normalize_inputs(
            eeg_feature, image_feature, text_feature if self.beta != 1.0 else None
        )

        # Calculate similarity matrix (N x N)
        logit_scale = self._get_logit_scale()
        similarity_matrix_ie = torch.matmul(eeg_feature, image_feature.T) * logit_scale
        if eeg_confidence is not None:
            similarity_matrix_ie_q = similarity_matrix_ie * eeg_confidence.unsqueeze(1)
        else:
            similarity_matrix_ie_q = similarity_matrix_ie
        if self.beta != 1.0:
            similarity_matrix_te = torch.matmul(eeg_feature, text_feature.T) * logit_scale
            if eeg_confidence is not None:
                similarity_matrix_te_q = similarity_matrix_te * eeg_confidence.unsqueeze(1)
            else:
                similarity_matrix_te_q = similarity_matrix_te

        # Construct labels
        labels = torch.arange(eeg_feature.shape[0], device=eeg_feature.device)

        # Calculate two parts of the loss
        if self.soft_target_tau > 0:
            with torch.no_grad():
                target = F.normalize(image_feature, dim=1)
                target = torch.softmax(target @ target.T / self.soft_target_tau, dim=1)
                if self.soft_mix_beta > 0:
                    target = (1 - self.soft_mix_beta) * torch.eye(len(target), device=target.device) + self.soft_mix_beta * target
            loss_eeg_ie = -(target * F.log_softmax(similarity_matrix_ie_q, dim=1)).sum(1).mean()
        else:
            loss_eeg_ie = self.criterion_cls(similarity_matrix_ie_q, labels)
        loss_img_ie = self.criterion_cls(similarity_matrix_ie.T, labels)
        if self.beta != 1.0:
            loss_eeg_te = self.criterion_cls(similarity_matrix_te_q, labels)
            loss_img_te = self.criterion_cls(similarity_matrix_te.T, labels)
            
        if self.alpha != 1.0:
            if self.mse_on_raw:
                loss_mse = self.criterion_mse(eeg_raw, image_raw)
            else:
                loss_mse = self.criterion_mse(eeg_feature, image_feature)
        
        # Total loss is the average
        if self.beta != 1.0:
            loss_contrastive_ie = (loss_eeg_ie + loss_img_ie) / 2
            loss_contrastive_te = (loss_eeg_te + loss_img_te) / 2
            loss_contrastive = self.beta * loss_contrastive_ie + (1 - self.beta) * loss_contrastive_te
        else:
            loss_contrastive = (
                loss_eeg_ie if self.asymmetric else (loss_eeg_ie + loss_img_ie) / 2
            )
        
        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive
        
        return loss
