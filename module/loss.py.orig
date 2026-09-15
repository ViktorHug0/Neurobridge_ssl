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
    ):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
        self.text_l2norm = text_l2norm
        self.mse_on_raw = mse_on_raw  # ENIGMA: MSE on unnormalized target (learns CLIP magnitude)
        
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
    def _multi_positive_cross_entropy(logits, positive_mask):
        valid_rows = positive_mask.any(dim=1)
        if not torch.any(valid_rows):
            return logits.new_tensor(0.0)

        log_probs = torch.log_softmax(logits, dim=1)
        positives = positive_mask[valid_rows]
        
        # Avoid nan from 0 * -inf by using torch.where to mask out non-positives
        positive_log_probs = torch.where(positives, log_probs[valid_rows], torch.zeros_like(log_probs[valid_rows]))
        
        loss_per_row = -positive_log_probs.sum(dim=1) / positives.sum(dim=1).clamp_min(1)
        return loss_per_row.mean()

    def multi_positive_pair_loss(self, query_feature, key_feature, positive_mask, key_is_text=False, query_scale=None):
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
        loss_qk = self._multi_positive_cross_entropy(logits_qk, positive_mask)
        loss_kq = self._multi_positive_cross_entropy(logits.T, positive_mask.T)
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
            loss_contrastive = (loss_eeg_ie + loss_img_ie) / 2
        
        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive
        
        return loss