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
    
    def forward(self, x:Tensor):
        x = x.unsqueeze(dim=1)
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        x = self.proj_eeg(x)
        return x


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
        if batch_norm:
            layers.append(nn.BatchNorm2d(temporal_filters))
        layers.append(_make_activation(activation))
        layers.append(nn.Conv2d(temporal_filters, spatial_filters, (channels_num, 1), (1, 1), bias=conv_bias))
        if batch_norm:
            layers.append(nn.BatchNorm2d(spatial_filters))
        layers.extend([
            _make_activation(activation),
            nn.Dropout(dropout),
        ])

        self.tsconv = nn.Sequential(*layers)
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

    def forward(self, x: Tensor):
        x = x.unsqueeze(dim=1)
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        x = self.proj_eeg(x)
        return x


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