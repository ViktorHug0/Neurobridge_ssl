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

        return loss
