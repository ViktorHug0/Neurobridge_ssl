"""AUG_LOG 2026-10-29 decorrelated-encoder search: a recurrent member, an inductive bias neither the REVE family (patch
transformer) nor TSC/SQF (temporal-then-spatial convolution) has.
GRUEncoder: per-trial z-score -> spatial filters -> depthwise temporal smoothing -> per-dataset BatchNorm -> BiGRU over
time -> attention mean+std pooling -> Linear(feature_dim)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.eeg_encoder.model import DomainBatchNorm2d


class GRUEncoder(nn.Module):
    def __init__(self, feature_dim, eeg_sample_points, channels_num, n_filters=64, hidden=256, layers=2,
                 temporal_kernel=5, dropout=0.2, domain_bn_boundary=0, cell='gru', bidirectional=True):
        super().__init__()
        self.spatial = nn.Conv1d(channels_num, n_filters, 1)
        self.temporal = nn.Conv1d(n_filters, n_filters, temporal_kernel, padding=temporal_kernel // 2, groups=n_filters)
        self.domain = domain_bn_boundary > 0
        self.bn = (DomainBatchNorm2d(n_filters, domain_bn_boundary, norm=nn.BatchNorm1d) if self.domain
                   else nn.BatchNorm1d(n_filters))
        rnn = {'gru': nn.GRU, 'lstm': nn.LSTM}[cell]
        self.gru = rnn(n_filters, hidden, layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        out = hidden * (2 if bidirectional else 1)
        self.score = nn.Linear(out, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * out), nn.Dropout(dropout), nn.Linear(2 * out, feature_dim))

    def forward(self, x, subject_ids=None):
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)
        h = self.temporal(self.spatial(x))
        h = self.bn(h, subject_ids) if self.domain else self.bn(h)
        h, _ = self.gru(F.gelu(h).transpose(1, 2))  # (batch, time, 2 hidden)
        w = self.score(h).softmax(dim=1)
        mean = (w * h).sum(1)
        std = (w * (h - mean.unsqueeze(1)).square()).sum(1).clamp_min(1e-6).sqrt()
        return self.head(torch.cat([mean, std], -1))


if __name__ == '__main__':
    x, sid = torch.randn(6, 32, 120), torch.tensor([1, 2, 3, 21, 22, 25])
    for kw in ({}, {'cell': 'lstm'}, {'bidirectional': False}):
        m = GRUEncoder(1536, 120, 32, domain_bn_boundary=21, **kw)
        out = m(x, sid)
        assert out.shape == (6, 1536)
        out.sum().backward()
        m.eval()
        assert torch.allclose(m(x, sid), m(4 * x - 1, sid), atol=1e-4), 'scale/offset-free input'
        print('GRUEncoder', kw, 'ok', f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M')
