import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class ContrastiveLoss(nn.Module):
    soft_label, soft_tau, gce_q = 0.0, 0.05, 0.0  # set from train.py (--soft_label, --soft_tau, --gce_q); EEG -> image direction only
    bank, bank_soft, bank_sep = None, False, 0.0  # --open_bank [--open_bank_soft | --open_bank_sep L]: unit embeddings of images with no EEG in the EEG -> image loss

    def __init__(
        self,
        init_temperature,
        alpha,
        eeg_l2norm: bool,
        img_l2norm: bool,
        learnable: bool,
        is_softplus: bool,
        logit_scale_max=None,
        asymmetric: bool = False,
    ):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
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

    def multi_positive_pair_loss(self, query_feature, key_feature, positive_mask):
        if self.eeg_l2norm:
            query_feature = F.normalize(query_feature, p=2, dim=1)
        if self.img_l2norm:
            key_feature = F.normalize(key_feature, p=2, dim=1)
        logit_scale = self._get_logit_scale()
        logits = torch.matmul(query_feature, key_feature.T) * logit_scale
        loss_qk = self._multi_positive_cross_entropy(
            logits, positive_mask
        )
        loss_kq = self._multi_positive_cross_entropy(
            logits.T, positive_mask.T
        )
        return (loss_qk + loss_kq) / 2

    def forward(self, eeg_feature, image_feature):
        eeg_feature, image_feature = self._normalize_inputs(eeg_feature, image_feature)

        # Calculate similarity matrix (N x N)
        logit_scale = self._get_logit_scale()
        similarity_matrix_ie = torch.matmul(eeg_feature, image_feature.T) * logit_scale

        # Construct labels
        labels = torch.arange(eeg_feature.shape[0], device=eeg_feature.device)
        if self.soft_label:  # SoftCLIP: part of the target mass goes to the batch images that resemble the trial's own image
            k = F.normalize(image_feature, dim=1)
            eye = torch.eye(len(k), device=k.device)  # the own image is left out of the softmax: the share a goes to OTHER images, by similarity
            labels = (1 - self.soft_label) * eye + self.soft_label * torch.softmax(k @ k.T / self.soft_tau - 1e9 * eye, 1)
        extra = 0.0
        if self.bank is not None and self.bank_sep:  # a softmax over the bank ALONE (bank images in the batch's softmax give a domain shortcut: Y02, obs run):
            kb = F.normalize(image_feature, dim=1) @ self.bank.T  # the trial's similarity profile over the bank should be its image's profile
            extra = self.bank_sep * self.criterion_cls(eeg_feature @ self.bank.T * logit_scale, torch.softmax(kb.float() / self.soft_tau, 1))
        elif self.bank is not None:  # open bank in the batch's softmax (MI_NEURIPS_LOG Y02): every trial is also scored against images that nobody saw
            assert self.asymmetric and self.alpha == 1.0 and self.soft_label and not self.gce_q
            kb = k @ self.bank.T; dup = kb > 0.999  # a bank copy of the trial's own image is neither a negative nor a neighbour
            logits = torch.cat([similarity_matrix_ie, (eeg_feature @ self.bank.T * logit_scale).masked_fill(dup, -1e9)], 1)
            if self.bank_soft:  # the soft share a goes to the batch AND bank images that resemble the own image
                eye = torch.cat([eye, torch.zeros_like(kb)], 1)
                labels = (1 - self.soft_label) * eye + self.soft_label * torch.softmax(torch.cat([k @ k.T, kb.masked_fill(dup, -1e9)], 1) / self.soft_tau - 1e9 * eye, 1)
            else:  # bank images are plain negatives
                labels = torch.cat([labels, torch.zeros_like(kb)], 1)
            return self.criterion_cls(logits, labels)
        if self.gce_q:  # generalised cross-entropy (1 - p^q) / q: the gradient of a trial is scaled by p^q, so trials with no readable signal weigh less (q -> 0 is the cross-entropy)
            p = torch.softmax(similarity_matrix_ie, 1).diagonal()
            return ((1 - p ** self.gce_q) / self.gce_q).mean()

        # Calculate two parts of the loss
        loss_eeg_ie = self.criterion_cls(similarity_matrix_ie, labels)
        loss_img_ie = self.criterion_cls(similarity_matrix_ie.T, labels)

        if self.alpha != 1.0:
            loss_mse = self.criterion_mse(eeg_feature, image_feature)

        loss_contrastive = (
            loss_eeg_ie if self.asymmetric else (loss_eeg_ie + loss_img_ie) / 2
        )

        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive

        return loss + extra


def build_image_positive_mask(object_indices, image_indices):
    same_object = object_indices.unsqueeze(1) == object_indices.unsqueeze(0)
    same_image = image_indices.unsqueeze(1) == image_indices.unsqueeze(0)
    return same_object & same_image


def _compute_cross_modal_loss_impl(
    criterion,
    eeg_feature,
    image_feature,
    positive_mask,
    use_multi_positive,
):
    if use_multi_positive:
        loss_contrastive = criterion.multi_positive_pair_loss(
            eeg_feature,
            image_feature,
            positive_mask,
        )

        if criterion.alpha != 1.0:
            eeg_for_mse, image_for_mse = criterion._normalize_inputs(eeg_feature, image_feature)
            loss_mse = criterion.criterion_mse(eeg_for_mse, image_for_mse)
            return criterion.alpha * loss_contrastive + (1 - criterion.alpha) * loss_mse
        return loss_contrastive

    return criterion(eeg_feature, image_feature)


def compute_cross_modal_loss(
    criterion,
    eeg_feature,
    image_feature,
    positive_mask,
    use_multi_positive,
):
    """Run contrastive geometry in FP32 even when encoder AMP is active."""
    if not torch.is_autocast_enabled('cuda'):
        return _compute_cross_modal_loss_impl(
            criterion, eeg_feature, image_feature,
            positive_mask, use_multi_positive,
        )

    with torch.autocast(device_type='cuda', enabled=False):
        return _compute_cross_modal_loss_impl(
            criterion,
            eeg_feature.float(),
            image_feature.float(),
            positive_mask,
            use_multi_positive,
        )


def compute_subject_mixup_regularization(mixed_eeg_feature, original_eeg_feature, partner_indices, mixed_mask):
    if partner_indices is None or mixed_mask is None or not torch.any(mixed_mask):
        return mixed_eeg_feature.new_tensor(0.0)
    mixed_subset = mixed_eeg_feature[mixed_mask]
    original_subset = original_eeg_feature[mixed_mask]
    partner_subset = original_eeg_feature[partner_indices[mixed_mask]]
    return (
        torch.norm(mixed_subset - original_subset, p=2, dim=1).sum()
        + torch.norm(mixed_subset - partner_subset, p=2, dim=1).sum()
    ) / mixed_eeg_feature.shape[0]
