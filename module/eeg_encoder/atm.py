"""ATM (ATMS): iTransformer channel attention + ShallowNet-style conv readout.

Only the parts of the Time-Series-Library layers that ATMS reaches are kept: inverted value
embedding, one full-attention encoder layer, no masks, no time marks. Module and attribute
names are unchanged so existing checkpoints load strictly.
"""
import math
from math import sqrt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from module.eeg_encoder.model import DomainBatchNorm2d


class PositionalEmbedding(nn.Module):  # never read (no time marks) but its buffer is in every checkpoint
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))


class TimeFeatureEmbedding(nn.Module):  # never called, kept for checkpoint keys
    def __init__(self, d_model, freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)


class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super(SubjectEmbedding, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))  # Shared token for unknown subjects
        self.mask_embedding = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding

    def forward(self, subject_ids):
        if subject_ids[0] is None or torch.any(subject_ids >= self.subject_embedding.num_embeddings):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, freq='h', dropout=0.1, num_subjects=None):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        self.mask_token = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding

    def forward(self, x, subject_ids=None):
        x = self.value_embedding(x)
        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(subject_ids)  # (batch_size, 1, d_model)
            x = torch.cat([subject_emb, x], dim=1)  # prepend along the token axis: (batch_size, seq_len + 1, d_model)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        E = queries.shape[-1]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = torch.softmax(1. / sqrt(E) * scores, dim=-1)
        if getattr(self, 'forced', None) is not None:  # ATMS split-key forward: the map computed from the subject key
            A = self.forced
        if getattr(self, 'record', False):
            self.last = A
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)
        out = self.inner_attention(queries, keys, values).view(B, L, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = x + self.dropout(self.attention(x))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Config:
    def __init__(
        self,
        channels_num,
        d_model=250,
        n_heads=4,
        e_layers=1,
        d_ff=256,
        dropout=0.25,
        seq_len=250,
    ):
        self.seq_len = seq_len             # Sequence length (= EEG samples per epoch)
        self.d_model = d_model             # Model dimension
        self.freq = 'h'                    # Time frequency
        self.dropout = dropout             # Dropout rate
        self.n_heads = n_heads             # Number of attention heads
        self.e_layers = e_layers           # Number of encoder layers
        self.d_ff = d_ff                   # Feedforward network dimension
        self.activation = 'gelu'           # Activation function
        self.enc_in = channels_num         # Encoder input dimension (example value)


class iTransformer(nn.Module):
    def __init__(self, configs, num_subjects=10):
        super(iTransformer, self).__init__()
        self.configs = configs
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.freq, configs.dropout, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, subject_ids)
        enc_out = self.encoder(enc_out)
        return enc_out[:, :self.configs.enc_in, :]


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        configs=None,
        temporal_filters=40,
        temporal_kernel=25,
        pool_kernel=51,
        pool_stride=5,
        spatial_filters=40,
        projection_filters=40,
        dropout=0.5,
    ):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, temporal_kernel), stride=(1, 1)),
            nn.AvgPool2d((1, pool_kernel), (1, pool_stride)),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Conv2d(temporal_filters, spatial_filters, (configs.enc_in, 1), stride=(1, 1)),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(spatial_filters, projection_filters, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, configs=None, **patch_kwargs):
        super().__init__(
            PatchEmbedding(configs=configs, **patch_kwargs),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(
        self,
        channels_num=63,
        feature_dim=1024,
        eeg_sample_points=250,
        d_model=250,
        n_heads=4,
        e_layers=1,
        d_ff=256,
        attention_dropout=0.25,
        temporal_filters=40,
        temporal_kernel=25,
        pool_kernel=51,
        pool_stride=5,
        spatial_filters=40,
        projection_filters=40,
        conv_dropout=0.5,
        subject_token=True,
        backbone_dim=0,
        domain_bn_boundary=0,
    ):
        super(ATMS, self).__init__()
        # e_layers == 0 keeps the per-channel temporal embedding and the conv readout but drops
        # the channel transformer (attention + FFN): the "ATM without attention" ablation.
        if d_model <= 0 or d_ff <= 0 or e_layers < 0 or n_heads <= 0:
            raise ValueError("ATM dimensions and head count must be positive, layer count >= 0")
        conv_width = d_model - temporal_kernel + 1
        pooled_width = (conv_width - pool_kernel) // pool_stride + 1
        if pooled_width <= 0:
            raise ValueError(
                "ATM d_model is too short for the requested temporal convolution and pool"
            )
        subjects_num=2
        default_config = Config(
            channels_num,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=attention_dropout,
            # The iTransformer value embedding is Linear(seq_len, d_model) over each channel's
            # time series, so seq_len must be the actual epoch length. It was pinned at 250,
            # which crashes on any other sampling rate; 250-sample runs are unaffected.
            seq_len=eeg_sample_points,
        )
        # The subject token is prepended, then enc_out[:, :enc_in] keeps it plus the first 62
        # channels, so O2's own token never reaches the conv readout (it leaks in only through
        # attention: 5x less output sensitivity than with the token dropped). Its table is also indexed 0-9 while
        # subject ids are 1-10, so training only ever saw the shared fallback token. The
        # Codabench grader calls predict(X) with no subject ids anyway; subject_token=False
        # drops the token and keeps all 63 channels.
        self.encoder = iTransformer(default_config, num_subjects=10 if subject_token else None)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, eeg_sample_points) for _ in range(subjects_num)])
        self.enc_eeg = Enc_eeg(
            configs=default_config,
            temporal_filters=temporal_filters,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            spatial_filters=spatial_filters,
            projection_filters=projection_filters,
            dropout=conv_dropout,
        )
        tsconv = self.enc_eeg[0].tsconv
        if domain_bn_boundary > 0:  # per-dataset statistics in the conv stem's two BatchNorms (cf. FastTSConvSqueezeformer)
            for k in (2, 5):
                tsconv[k] = DomainBatchNorm2d(tsconv[k].num_features, domain_bn_boundary)
        embedding_dim = pooled_width * projection_filters
        # backbone_dim > 0 mirrors TSConv_parameterizable's trunk: the residual head runs at
        # backbone_dim and a final Linear maps to feature_dim (the grader's 1536-D space).
        trunk_dim = backbone_dim or feature_dim
        self.proj_eeg = Proj_eeg(embedding_dim=embedding_dim, proj_dim=trunk_dim)
        if trunk_dim != feature_dim:
            self.proj_eeg.append(nn.Linear(trunk_dim, feature_dim))

    def forward(self, x:Tensor, subject_ids=None):
        x = x.squeeze(1)
        if x.dim() == 4:  # (B, 2, C, T) = [content, key] (train.py --subject_key split; scripts/mi/MI_LOG.md E13):
            # the electrode attention map is computed from the subject key alone, then applied to the key-free content
            fa = self.encoder.encoder.attn_layers[0].attention.inner_attention
            fa.record = True
            self.encoder(x[:, 1], subject_ids)
            fa.forced, fa.record = fa.last, False
            x = self.encoder(x[:, 0], subject_ids)
            fa.forced = fa.last = None
        else:
            x = self.encoder(x, subject_ids)
        if any(isinstance(layer, DomainBatchNorm2d) for layer in self.enc_eeg[0].tsconv):
            z = x.unsqueeze(1)
            for layer in self.enc_eeg[0].tsconv:
                z = layer(z, subject_ids) if isinstance(layer, DomainBatchNorm2d) else layer(z)
            eeg_embedding = self.enc_eeg[1](self.enc_eeg[0].projection(z))
        else:
            eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out
