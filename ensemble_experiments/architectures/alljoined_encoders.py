"""_SwitchBlock: the transformer block TokenREVE (hybrid_encoders.py) is built from."""
import torch
from torch import nn


class _SwitchBlock(nn.Module):
    """Pre-norm transformer block that is either REVE's (norm='rms', bias=False, ff='geglu') or
    nn.TransformerEncoderLayer's (norm='layer', bias=True, ff='relu', norm_first, no dropout), or anything between.
    Same parameter layout as both (fused qkv, out, ff1, ff2), so weights copy across exactly."""

    def __init__(self, d, heads, norm, bias, ff, ff_mult):
        super().__init__()
        from braindecode.models.reve import RMSNorm
        make_norm = (lambda: RMSNorm(d)) if norm == 'rms' else (lambda: nn.LayerNorm(d))
        self.heads, self.geglu = heads, ff == 'geglu'
        hidden = int(d * 2.66) if self.geglu else ff_mult * d  # REVE: mlp_dim_ratio 2.66
        self.n1, self.n2 = make_norm(), make_norm()
        self.qkv, self.out = nn.Linear(d, 3 * d, bias=bias), nn.Linear(d, d, bias=bias)
        self.ff1, self.ff2 = nn.Linear(d, 2 * hidden if self.geglu else hidden, bias=bias), nn.Linear(hidden, d, bias=bias)
        self.attn_mask, self.gate = None, None  # optional (HISTORY.md R-series round 5): boolean attention mask, output gate
        self.type_ffn, self.split = None, 0     # optional (round 6): a second FFN (n2, ff1, ff2) for tokens >= split

    def forward(self, x):
        b, n, d = x.shape
        xn = self.n1(x)
        q, k, v = (t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for t in self.qkv(xn).chunk(3, -1))
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask).transpose(1, 2).reshape(b, n, d)
        if self.gate is not None:  # gated attention: per-token, per-channel sigmoid gate on the attention output
            a = a * torch.sigmoid(self.gate(xn))
        x = x + self.out(a)
        if self.type_ffn is not None:  # token-type-specific feed-forward experts, attention stays joint
            t = self.type_ffn
            return torch.cat([x[:, :self.split] + self._ffn(x[:, :self.split], self.n2, self.ff1, self.ff2),
                              x[:, self.split:] + self._ffn(x[:, self.split:], t['n2'], t['ff1'], t['ff2'])], 1)
        return x + self._ffn(x, self.n2, self.ff1, self.ff2)

    def _ffn(self, x, n2, ff1, ff2):
        h = ff1(n2(x))
        if self.geglu:
            h, gate = h.chunk(2, -1)
            h = nn.functional.gelu(gate) * h
        else:
            h = nn.functional.relu(h)
        return ff2(h)
