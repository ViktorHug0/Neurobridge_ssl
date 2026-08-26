import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)


class EEGNet(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()

        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (channels_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                nn.Dropout2d(0.5)
            )
        
        # Use a dummy tensor to pass through the backbone to calculate the flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels_num, eeg_sample_points)
            out = self.backbone(dummy)
            embedding_dim = out.shape[1] * out.shape[2] * out.shape[3]
        
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.backbone(x)
        x = x.view(x.size(0), -1) 
        x = self.project(x)
        return x

class EEGProject(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.input_dim = eeg_sample_points * channels_num

        self.model = nn.Sequential(nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.3),
            )),
            nn.LayerNorm(feature_dim))
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class TSConv(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        
        emb_size = 40
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))
        
        embedding_dim = (math.ceil((((eeg_sample_points - 25) + 1) - 51) / 5.) + 1) * 40
        self.proj_eeg = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5),
            )),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x: Tensor, return_intermediate=False, temporal_augmentation=None):
        x = x.unsqueeze(dim=1)
        x_temp = self.tsconv[:4](x)
        if temporal_augmentation is not None:
            x_temp = temporal_augmentation(x_temp)
        x_spat = self.tsconv[4:](x_temp)
        x_proj = self.projection(x_spat)
        x_flat = x_proj.view(x_proj.size(0), -1)
        x_out = self.proj_eeg(x_flat)
        if return_intermediate:
            return {
                'temporal': x_temp.view(x_temp.size(0), -1),
                'spatial': x_spat.view(x_spat.size(0), -1),
                'backbone': x_flat,
                'output': x_out
            }
        return x_out


def _make_activation(name):
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported TSConv activation: {name}")


class DomainBatchNorm2d(nn.Module):
    """Two-domain BatchNorm with a shared surrounding feature extractor.

    Subject IDs below ``boundary`` use the primary (EEG2) statistics; IDs at or
    above it use the auxiliary (EEG1) statistics.  At inference, omitting IDs is
    deliberately equivalent to the primary path.
    """

    def __init__(self, num_features, boundary):
        super().__init__()
        if boundary <= 0:
            raise ValueError("DomainBatchNorm2d boundary must be positive")
        self.boundary = int(boundary)
        self.primary = nn.BatchNorm2d(num_features)
        self.auxiliary = nn.BatchNorm2d(num_features)

    def forward(self, features, subject_ids=None):
        if subject_ids is None:
            return self.primary(features)
        if subject_ids.ndim != 1 or subject_ids.shape[0] != features.shape[0]:
            raise ValueError("subject_ids must have one entry per feature row")
        auxiliary_mask = subject_ids >= self.boundary
        if not torch.any(auxiliary_mask):
            return self.primary(features)
        if torch.all(auxiliary_mask):
            return self.auxiliary(features)
        output = torch.empty_like(features)
        output[~auxiliary_mask] = self.primary(features[~auxiliary_mask])
        output[auxiliary_mask] = self.auxiliary(features[auxiliary_mask])
        return output


class SubjectBatchNorm2d(nn.Module):
    """Source-person BatchNorm random effects plus a trained population path.

    Index zero is reserved for virtual subjects and unseen-person inference.
    During evaluation every row is routed through index zero regardless of the
    dataset's numeric subject ID.
    """

    def __init__(self, num_features, max_subject_id):
        super().__init__()
        if max_subject_id <= 0:
            raise ValueError("SubjectBatchNorm2d max_subject_id must be positive")
        self.max_subject_id = int(max_subject_id)
        self.paths = nn.ModuleList([
            nn.BatchNorm2d(num_features)
            for _ in range(self.max_subject_id + 1)
        ])

    def forward(self, features, subject_ids=None):
        if subject_ids is None or not self.training:
            return self.paths[0](features)
        if subject_ids.ndim != 1 or subject_ids.shape[0] != features.shape[0]:
            raise ValueError("subject_ids must have one entry per feature row")
        if torch.any(subject_ids < 0) or torch.any(subject_ids > self.max_subject_id):
            raise ValueError(
                f"subject BN IDs must lie in [0, {self.max_subject_id}]"
            )
        output = torch.empty_like(features)
        for subject_id in torch.unique(subject_ids).tolist():
            mask = subject_ids == int(subject_id)
            output[mask] = self.paths[int(subject_id)](features[mask])
        return output


class TSConv_parameterizable(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        eeg_sample_points=250,
        channels_num=63,
        temporal_filters=40,
        temporal_kernel=25,
        pool_kernel=51,
        pool_stride=5,
        spatial_filters=40,
        projection_filters=40,
        activation="elu",
        dropout=0.5,
        head_dropout=None,
        batch_norm=True,
        conv_bias=True,
        domain_batch_norm_boundary=0,
        subject_batch_norm_max_id=0,
        subject_adapter_rank=0,
        subject_adapter_max_id=32,
        subject_adapter_scale=0.1,
        subject_adapter_virtual_prob=0.0,
        subject_adapter_virtual_scale=1.0,
        subject_adapter_source_ids=None,
        subject_adapter_inferred=False,
        mutual_reconstruction_hidden=0,
        cohort_reconstruction_hidden=0,
    ):
        super().__init__()

        if temporal_kernel <= 0 or pool_kernel <= 0 or pool_stride <= 0:
            raise ValueError("TSConv kernel and stride parameters must be positive.")
        if temporal_filters <= 0 or spatial_filters <= 0 or projection_filters <= 0:
            raise ValueError("TSConv filter counts must be positive.")

        head_dropout = dropout if head_dropout is None else head_dropout
        layers = [
            nn.Conv2d(1, temporal_filters, (1, temporal_kernel), (1, 1), bias=conv_bias),
            nn.AvgPool2d((1, pool_kernel), (1, pool_stride)),
        ]
        self.domain_batch_norm_boundary = int(domain_batch_norm_boundary)
        self.subject_batch_norm_max_id = int(subject_batch_norm_max_id)
        if self.domain_batch_norm_boundary > 0 and self.subject_batch_norm_max_id > 0:
            raise ValueError("domain and subject BatchNorm routing are mutually exclusive")
        self.subject_adapter_rank = int(subject_adapter_rank)
        self.subject_adapter_max_id = int(subject_adapter_max_id)
        self.subject_adapter_scale = float(subject_adapter_scale)
        self.subject_adapter_virtual_prob = float(subject_adapter_virtual_prob)
        self.subject_adapter_virtual_scale = float(subject_adapter_virtual_scale)
        self.subject_adapter_inferred = bool(subject_adapter_inferred)
        self.subject_adapter_source_ids = tuple(
            int(value) for value in (subject_adapter_source_ids or ())
        )
        self.mutual_reconstruction_hidden = int(mutual_reconstruction_hidden)
        self.cohort_reconstruction_hidden = int(cohort_reconstruction_hidden)
        if batch_norm:
            layers.append(
                SubjectBatchNorm2d(temporal_filters, self.subject_batch_norm_max_id)
                if self.subject_batch_norm_max_id > 0
                else (
                    DomainBatchNorm2d(temporal_filters, self.domain_batch_norm_boundary)
                    if self.domain_batch_norm_boundary > 0
                    else nn.BatchNorm2d(temporal_filters)
                )
            )
        layers.append(_make_activation(activation))
        layers.append(nn.Conv2d(temporal_filters, spatial_filters, (channels_num, 1), (1, 1), bias=conv_bias))
        if batch_norm:
            layers.append(
                SubjectBatchNorm2d(spatial_filters, self.subject_batch_norm_max_id)
                if self.subject_batch_norm_max_id > 0
                else (
                    DomainBatchNorm2d(spatial_filters, self.domain_batch_norm_boundary)
                    if self.domain_batch_norm_boundary > 0
                    else nn.BatchNorm2d(spatial_filters)
                )
            )
        layers.extend([
            _make_activation(activation),
            nn.Dropout(dropout),
        ])

        self.tsconv = nn.Sequential(*layers)
        if self.subject_adapter_rank > 0:
            if self.subject_adapter_max_id <= 0:
                raise ValueError("subject_adapter_max_id must be positive")
            self.subject_adapter_embedding = nn.Embedding(
                self.subject_adapter_max_id + 1,
                self.subject_adapter_rank,
                padding_idx=0,
            )
            self.subject_adapter_projection = nn.Linear(
                self.subject_adapter_rank,
                2 * temporal_filters * channels_num,
                bias=False,
            )
            nn.init.normal_(self.subject_adapter_embedding.weight, std=0.02)
            with torch.no_grad():
                self.subject_adapter_embedding.weight[0].zero_()
            nn.init.zeros_(self.subject_adapter_projection.weight)
            self.subject_adapter_shape = (temporal_filters, channels_num)
            if self.subject_adapter_inferred:
                router_hidden = max(32, 8 * self.subject_adapter_rank)
                self.subject_adapter_router = nn.Sequential(
                    nn.Linear(2 * temporal_filters, router_hidden),
                    nn.GELU(),
                    nn.LayerNorm(router_hidden),
                )
                self.subject_adapter_code_head = nn.Linear(
                    router_hidden, self.subject_adapter_rank
                )
                self.subject_adapter_class_head = nn.Linear(
                    router_hidden, self.subject_adapter_max_id + 1
                )
                self._subject_adapter_router_codes = None
                self._subject_adapter_router_logits = None
        if self.mutual_reconstruction_hidden > 0:
            if self.subject_adapter_rank <= 0:
                raise ValueError("mutual reconstruction requires a subject adapter embedding")
            self.mutual_reconstruction_decoder = nn.Sequential(
                nn.Linear(
                    feature_dim + self.subject_adapter_rank,
                    self.mutual_reconstruction_hidden,
                ),
                nn.GELU(),
                nn.Linear(
                    self.mutual_reconstruction_hidden,
                    channels_num * eeg_sample_points,
                ),
            )
            self.mutual_reconstruction_shape = (channels_num, eeg_sample_points)
        if self.cohort_reconstruction_hidden > 0:
            self.cohort_reconstruction_decoder = nn.Sequential(
                nn.Linear(feature_dim, self.cohort_reconstruction_hidden),
                nn.GELU(),
                nn.Linear(
                    self.cohort_reconstruction_hidden,
                    channels_num * eeg_sample_points,
                ),
            )
            self.cohort_reconstruction_shape = (channels_num, eeg_sample_points)
        self.projection = nn.Conv2d(spatial_filters, projection_filters, (1, 1), stride=(1, 1), bias=conv_bias)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels_num, eeg_sample_points)
            embedding_dim = self.projection(self.tsconv(dummy)).view(1, -1).shape[1]

        self.proj_eeg = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(head_dropout),
            )),
            nn.LayerNorm(feature_dim),
        )

    @staticmethod
    def _apply_layers(layers, features, subject_ids):
        for layer in layers:
            if isinstance(layer, (DomainBatchNorm2d, SubjectBatchNorm2d)):
                features = layer(features, subject_ids)
            else:
                features = layer(features)
        return features

    def forward(
        self, x: Tensor, return_intermediate=False, temporal_augmentation=None,
        subject_ids=None,
    ):
        x = x.unsqueeze(dim=1)
        # tsconv layers: 0:conv, 1:pool, 2:bn, 3:act, 4:conv, 5:bn, 6:act, 7:dropout
        x_temp = self._apply_layers(self.tsconv[:4], x, subject_ids)
        if self.subject_adapter_rank > 0 and (
            subject_ids is not None or self.subject_adapter_inferred
        ):
            if subject_ids is not None and (
                subject_ids.ndim != 1 or subject_ids.shape[0] != x_temp.shape[0]
            ):
                raise ValueError("subject_ids must have one entry per feature row")
            if self.subject_adapter_inferred:
                # Infer a continuous person coordinate from the observation itself,
                # so the adapter remains defined for a genuinely unseen person.
                means = x_temp.mean(dim=(2, 3))
                log_stds = x_temp.var(
                    dim=(2, 3), unbiased=False
                ).clamp_min(1e-6).sqrt().log()
                router_features = self.subject_adapter_router(
                    torch.cat([means, log_stds], dim=1)
                )
                adapter_codes = self.subject_adapter_code_head(router_features)
                self._subject_adapter_router_codes = adapter_codes
                self._subject_adapter_router_logits = self.subject_adapter_class_head(
                    router_features
                )
            else:
                adapter_ids = subject_ids.to(device=x_temp.device, dtype=torch.long)
                if not self.training:
                    # The held person's ID has no learned meaning.  Inference uses
                    # the population path trained under padding ID zero.
                    adapter_ids = torch.zeros_like(adapter_ids)
                if torch.any(adapter_ids.abs() > self.subject_adapter_max_id):
                    raise ValueError(
                        f"absolute subject adapter IDs must not exceed {self.subject_adapter_max_id}"
                    )
                virtual_mask = adapter_ids < 0
                lookup_ids = adapter_ids.clamp_min(0)
                adapter_codes = self.subject_adapter_embedding(lookup_ids)
                if torch.any(virtual_mask):
                    adapter_codes = adapter_codes.clone()
                    adapter_codes[virtual_mask] = 0
                    if self.training and self.subject_adapter_virtual_prob > 0:
                        source_ids = torch.tensor(
                            self.subject_adapter_source_ids,
                            device=x_temp.device,
                            dtype=torch.long,
                        )
                        source_codes = self.subject_adapter_embedding(source_ids).detach()
                        mean = source_codes.mean(dim=0)
                        centered = source_codes - mean
                        covariance = centered.T @ centered / max(source_codes.shape[0] - 1, 1)
                        covariance = covariance + 1e-6 * torch.eye(
                            covariance.shape[0], device=covariance.device,
                            dtype=covariance.dtype,
                        )
                        chol = torch.linalg.cholesky(covariance)
                        for identity in torch.unique(adapter_ids[virtual_mask].abs()):
                            identity_mask = adapter_ids == -identity
                            if torch.rand((), device=x_temp.device) < self.subject_adapter_virtual_prob:
                                draw = mean + self.subject_adapter_virtual_scale * (
                                    torch.randn_like(mean) @ chol.T
                                )
                                adapter_codes[identity_mask] = draw
            adapter = self.subject_adapter_projection(adapter_codes)
            filters, channels = self.subject_adapter_shape
            adapter = adapter.view(x_temp.shape[0], 2, filters, channels, 1)
            gamma, beta = adapter[:, 0], adapter[:, 1]
            scale = self.subject_adapter_scale
            x_temp = x_temp * (1.0 + scale * torch.tanh(gamma)) + scale * beta
        if temporal_augmentation is not None:
            x_temp = temporal_augmentation(x_temp)
        x_spat = self._apply_layers(self.tsconv[4:], x_temp, subject_ids)
        x_proj = self.projection(x_spat)
        x_flat = x_proj.view(x_proj.size(0), -1)
        x_out = self.proj_eeg(x_flat)
        if return_intermediate:
            return {
                'temporal': x_temp.view(x_temp.size(0), -1),
                'spatial': x_spat.view(x_spat.size(0), -1),
                'backbone': x_flat,
                'output': x_out
            }
        return x_out

    def subject_adapter_router_loss(self, subject_ids):
        """Anchor the amortized style code with observable source identity."""
        if not self.subject_adapter_inferred:
            raise RuntimeError("inferred subject adapter is disabled")
        logits = self._subject_adapter_router_logits
        if logits is None or logits.shape[0] != subject_ids.shape[0]:
            raise RuntimeError("router loss must follow the matching encoder forward pass")
        labels = subject_ids.to(device=logits.device, dtype=torch.long)
        valid = (labels > 0) & (labels <= self.subject_adapter_max_id)
        if not torch.any(valid):
            return logits.sum() * 0.0, 0
        return nn.functional.cross_entropy(logits[valid], labels[valid]), int(valid.sum())

    def decode_subject_reconstruction(self, content, target_subject_ids):
        """Render population content using a learned source-person random effect."""
        if self.mutual_reconstruction_hidden <= 0:
            raise RuntimeError("mutual reconstruction decoder is disabled")
        target_subject_ids = target_subject_ids.to(device=content.device, dtype=torch.long)
        if torch.any(target_subject_ids <= 0) or torch.any(
            target_subject_ids > self.subject_adapter_max_id
        ):
            raise ValueError("reconstruction targets must be learned source subject IDs")
        style = self.subject_adapter_embedding(target_subject_ids)
        reconstruction = self.mutual_reconstruction_decoder(
            torch.cat([content, style], dim=1)
        )
        return reconstruction.view(content.shape[0], *self.mutual_reconstruction_shape)

    def decode_cohort_reconstruction(self, content):
        """Decode the exact-stimulus population-average EEG target."""
        if self.cohort_reconstruction_hidden <= 0:
            raise RuntimeError("cohort reconstruction decoder is disabled")
        reconstruction = self.cohort_reconstruction_decoder(content)
        return reconstruction.view(content.shape[0], *self.cohort_reconstruction_shape)


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        channels_num,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=51,
        pool_time_stride=5,
        dropout=0.5,
    ):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(n_filters_time),
            nn.ELU(),
            nn.AvgPool2d((1, pool_time_length), (1, pool_time_stride)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=1)
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads,
        dropout=0.5,
        forward_expansion=4,
        forward_dropout=None,
    ):
        super().__init__()
        forward_dropout = dropout if forward_dropout is None else forward_dropout
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(forward_dropout),
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(forward_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attention_dropout(attn_output)
        x = x + self.feed_forward(self.norm2(x))
        return x


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        emb_size,
        num_layers=2,
        num_heads=10,
        dropout=0.5,
        forward_expansion=4,
        forward_dropout=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerEncoderBlock(
                emb_size=emb_size,
                num_heads=num_heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                forward_dropout=forward_dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.output = nn.Linear(input_dim, feature_dim)
        self.norm = nn.BatchNorm1d(feature_dim)
        self.activation = nn.ELU()

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        x = self.flatten(x)
        x = self.activation(self.norm(self.output(x)))
        return x


class EEGConformer(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        eeg_sample_points=250,
        channels_num=63,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=51,
        pool_time_stride=5,
        num_layers=2,
        num_heads=10,
        transformer_dropout=0.5,
        forward_expansion=4,
    ):
        super().__init__()

        if n_filters_time <= 0:
            raise ValueError("n_filters_time must be positive.")
        if filter_time_length <= 0 or pool_time_length <= 0 or pool_time_stride <= 0:
            raise ValueError("filter_time_length, pool_time_length, and pool_time_stride must be positive.")
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        if n_filters_time % num_heads != 0:
            raise ValueError(
                f"n_filters_time ({n_filters_time}) must be divisible by num_heads ({num_heads})."
            )

        self.patch_embedding = _PatchEmbedding(
            channels_num=channels_num,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            dropout=transformer_dropout,
        )
        self.transformer_encoder = _TransformerEncoder(
            emb_size=n_filters_time,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=transformer_dropout,
            forward_expansion=forward_expansion,
            forward_dropout=transformer_dropout,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels_num, eeg_sample_points)
            patch_tokens = self.patch_embedding(dummy)
            self.pos_encoder = PositionalEncoding(n_filters_time, patch_tokens.shape[1])
            token_embeddings = self.transformer_encoder(self.pos_encoder(patch_tokens))
            flattened_dim = token_embeddings.reshape(1, -1).shape[1]

        self.classification_head = _ClassificationHead(
            input_dim=flattened_dim,
            feature_dim=feature_dim,
        )

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x, return_features=return_features)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class EEGTransformer(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        d_model = 128
        nhead = 8
        num_layers = 4
        dim_feedforward = 512
        dropout = 0.1
        
        # Project input (channels) -> embedding dimension
        self.input_proj = nn.Linear(channels_num, d_model)
        # Positional encoding across time dimension
        self.pos_encoder = PositionalEncoding(d_model, eeg_sample_points)
        # Transformer encoder (batch_first=True for [B, S, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Final projection to desired output dimension
        self.fc_out = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG data tensor of shape [batch_size, channels_num, seq_len].
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Rearrange to [batch_size, seq_len, channels_num]
        x = x.permute(0, 2, 1)
        # Project to embedding dimension
        x = self.input_proj(x)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer encoding
        x = self.transformer_encoder(x)
        # Pool across time (mean pooling)
        x = x.mean(dim=1)  # [batch_size, d_model]
        # Final feature projection
        x = self.fc_out(x)  # [batch_size, output_dim]
        return x

if __name__ == "__main__":
    # Example usage
    eeg_sample_points = 250
    channels_num = 17
    feature_dim = 1024
    model = EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    
    # Create a dummy EEG input tensor with shape (batch_size, channels_num, eeg_sample_points)
    batch_size = 8
    dummy_eeg_input = torch.randn(batch_size, channels_num, eeg_sample_points)
    
    # Forward pass through the model
    output = model(dummy_eeg_input)
    print(output.shape)  # Expected output shape: (batch_size, feature_dim)

class TSConv30(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.tsconv30 = nn.Sequential(
            nn.Conv2d(1, 40, (1, 30), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        
        emb_size = 40
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))
        
        embedding_dim = (math.ceil((((eeg_sample_points - 30) + 1) - 51) / 5.) + 1) * 40
        self.proj_eeg = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5),
            )),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x:Tensor):
        x = x.unsqueeze(dim=1)
        x = self.tsconv30(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        x = self.proj_eeg(x)
        return x
