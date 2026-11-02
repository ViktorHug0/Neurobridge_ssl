"""Encoders designed from the Alljoined cross-subject findings (neurips_challenge/CV4_PROTOCOL.md).

- Conv models (TSConv) mix all electrodes with spatial filters and are robust to a new recording's scale.
- Electrode-token transformers (LaBraM, REVE, ATM) are stronger on most new subjects but collapse on quiet
  recordings (sub-15).
- TSConv + LaBraM as an ensemble beat either alone by ~3-4 points, because they fail on different subjects.

ScaleRobustChannelTransformer: one token per electrode from its per-trial standardised waveform, plus its
log-RMS as a separate input, so the recording's overall scale cannot shift the token embeddings.
DualStreamNet: a TSConv stream and that token stream summed, trained jointly with stream dropout so each
stream must stand alone (an in-model version of the conv + transformer ensemble).
"""
import torch
from torch import nn


class ScaleRobustChannelTransformer(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32,
                 d_model=128, depth=4, heads=4, drop=0.1, ff_mult=2):
        super().__init__()
        self.embed = nn.Linear(eeg_sample_points, d_model)
        self.amplitude = nn.Linear(1, d_model)          # log-RMS of the electrode, kept but decoupled
        self.electrode = nn.Parameter(torch.zeros(1, channels_num, d_model))
        nn.init.trunc_normal_(self.electrode, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, drop, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feature_dim)

    def forward(self, x):  # (batch, channels, time)
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp_min(1e-4)
        tokens = self.embed((x - mean) / std) + self.amplitude(torch.log(std)) + self.electrode
        return self.head(self.norm(self.blocks(tokens)).mean(1))


class _ConvStream(nn.Module):
    """TSConv_parameterizable's 120 Hz geometry (temporal 25, pool 25/2, 40 filters)."""

    def __init__(self, feature_dim, eeg_sample_points, channels_num, filters=40, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, filters, (1, 25)), nn.AvgPool2d((1, 25), (1, 2)), nn.BatchNorm2d(filters), nn.ELU(),
            nn.Conv2d(filters, filters, (channels_num, 1)), nn.BatchNorm2d(filters), nn.ELU(), nn.Dropout(drop),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, 1, channels_num, eeg_sample_points)).numel()
        self.head = nn.Linear(flat, feature_dim)

    def forward(self, x):
        return self.head(self.net(x.unsqueeze(1)).flatten(1))


class DualStreamNet(nn.Module):
    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, stream_drop=0.25):
        super().__init__()
        self.conv = _ConvStream(feature_dim, eeg_sample_points, channels_num)
        self.tokens = ScaleRobustChannelTransformer(feature_dim, eeg_sample_points, channels_num)
        self.stream_drop = stream_drop

    def forward(self, x):
        a, b = self.conv(x), self.tokens(x)
        if self.training and self.stream_drop > 0:
            # per trial: with prob stream_drop keep only one (random) stream, at double weight
            r = torch.rand(x.shape[0], 1, device=x.device)
            keep_a = (r >= self.stream_drop / 2).float()
            keep_b = (r < self.stream_drop / 2).float() + (r >= self.stream_drop).float()
            return (keep_a * a + keep_b * b) * 2 / (keep_a + keep_b)
        return a + b


class _SourceRead(nn.Module):
    """Source latents cross-attend to the electrode tokens (pre-norm), then an MLP."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.q_norm, self.kv_norm, self.mlp_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, s, e):
        kv = self.kv_norm(e)
        s = s + self.attn(self.q_norm(s), kv, kv, need_weights=False)[0]
        return s + self.mlp(self.mlp_norm(s))


class SourceTokenTransformer(nn.Module):
    """Scratch REVE's ingredients (one token per electrode from its per-trial z-scored 1 s waveform, a deep
    pre-norm transformer, mean pooling) plus an explicit per-trial unmixing step: K learned source latents
    cross-attend to the electrode tokens and are then processed among themselves. Electrodes are a
    subject-specific projection of shared cortical sources, so the latents are where subject-invariant
    structure can live; the electrode -> source map is recomputed for every trial (attention), unlike the
    fixed spatial filters of conv models.
    """

    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, d_model=512, heads=8,
                 electrode_depth=6, source_depth=12, n_sources=64, read_every=4, clip_sd=15.0):
        super().__init__()
        self.clip_sd, self.read_every = clip_sd, read_every
        self.embed = nn.Linear(eeg_sample_points, d_model)
        self.electrode = nn.Parameter(torch.zeros(1, channels_num, d_model))
        self.sources = nn.Parameter(torch.zeros(1, n_sources, d_model))
        nn.init.trunc_normal_(self.electrode, std=0.02)
        nn.init.trunc_normal_(self.sources, std=0.02)
        block = lambda: nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, 0.0, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.electrode_blocks = nn.ModuleList(block() for _ in range(electrode_depth))
        self.source_blocks = nn.ModuleList(block() for _ in range(source_depth))
        self.reads = nn.ModuleList(_SourceRead(d_model, heads) for _ in range(-(-source_depth // read_every)))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feature_dim)

    def forward(self, x):  # (batch, channels, time)
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp_min(1e-5)
        e = self.embed(((x - mean) / std).clamp(-self.clip_sd, self.clip_sd)) + self.electrode  # REVE's input convention
        for blk in self.electrode_blocks:
            e = blk(e)
        s = self.sources.expand(x.shape[0], -1, -1)
        for i, blk in enumerate(self.source_blocks):
            if i % self.read_every == 0:
                s = self.reads[i // self.read_every](s, e)
            s = blk(s)
        return self.head(self.norm(s).mean(1))


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
        self.attn_mask, self.gate = None, None  # optional (ARCH_LOG R-series round 5): boolean attention mask, output gate
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


class BridgeTransformer(nn.Module):
    """ScaleRobustChannelTransformer -> scratch REVE, one switch at a time (what makes REVE better?).

    Defaults reproduce ScaleRobustChannelTransformer exactly (same state_dict, same init order); every switch
    flipped reproduces our REVE pipeline (FoundationEncoder 'REVE', trial_zscore, mean head) exactly. The
    self-check verifies both by weight copy. Switches, in bridge order:
      amp_input  log-RMS amplitude input (rct) / none (REVE discards scale)
      dropout    0.1 (rct) / 0 (REVE)
      block      'torch': nn.TransformerEncoder (LayerNorm, biases, ReLU ff_mult x, all layers cloned from one
                 init) + final LayerNorm / 'reve': REVE's backbone (RMSNorm, no biases, GEGLU 2.66x,
                 independent init), no final norm / 'switch': _SwitchBlock stack with the five parts of that swap
                 set one by one (norm, bias, ff, init, final_norm; defaults = REVE's)
      final_norm None: the block's own (torch: LayerNorm; switch/reve: none) / True / False forces it
      pos        'learned' electrode vectors / 'reve': LayerNorm(Fourier4D + MLP) of the electrode coordinates
      reve_input False: z-score the 120 samples / True: resample to 200, z-score, clip at 15 SD
    """

    def __init__(self, feature_dim=1536, eeg_sample_points=120, channels_num=32, ch_names=None, d_model=512,
                 depth=6, heads=8, ff_mult=4, amp_input=True, dropout=0.1, block='torch', pos='learned',
                 reve_input=False, norm='rms', bias=False, ff='geglu', init='independent', final_norm=None):
        super().__init__()
        self.amp_input, self.reve_input = amp_input, reve_input
        reve = None
        if block == 'reve' or pos == 'reve':
            from braindecode.models import REVE
            reve = REVE(n_outputs=2, n_chans=channels_num, n_times=200, chs_info=[{'ch_name': n} for n in ch_names],
                        embed_dim=d_model, depth=depth, heads=heads, head_dim=d_model // heads)
        self.embed = reve.to_patch_embedding[0] if reve_input and reve is not None else \
            nn.Linear(200 if reve_input else eeg_sample_points, d_model)
        if amp_input:
            self.amplitude = nn.Linear(1, d_model)
        if pos == 'learned':
            self.electrode = nn.Parameter(torch.zeros(1, channels_num, d_model))
            nn.init.trunc_normal_(self.electrode, std=0.02)
        else:
            from braindecode.models.reve import FourierEmb4D
            self.fourier4d, self.mlp4d, self.pos_ln = reve.fourier4d, reve.mlp4d, reve.ln
            self.register_buffer('positions', FourierEmb4D.add_time_patch(reve.default_pos.unsqueeze(0), 1))
        if block == 'switch':
            assert dropout == 0, "_SwitchBlock has no dropout"
            make = lambda: _SwitchBlock(d_model, heads, norm, bias, ff, ff_mult)
            if init == 'cloned':  # nn.TransformerEncoder's behaviour: every layer starts as a copy of one
                import copy
                first = make()
                self.blocks = nn.Sequential(*[copy.deepcopy(first) for _ in range(depth)])
            else:
                self.blocks = nn.Sequential(*[make() for _ in range(depth)])
            self.norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()
        elif block == 'torch':
            layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, dropout, batch_first=True,
                                               norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(d_model) if final_norm is not False else nn.Identity()
        else:
            assert dropout == 0, "REVE's backbone has no dropout"
            self.blocks, self.norm = reve.transformer, nn.Identity()
        self.head = nn.Linear(d_model, feature_dim)

    def forward(self, x):  # (batch, channels, time)
        if self.reve_input:  # the FoundationEncoder REVE path: resample, then z-score + clip
            x = nn.functional.interpolate(x, size=200, mode='linear', align_corners=False)
            mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp_min(1e-5)
            z = ((x - mean) / std).clamp(-15.0, 15.0)
        else:
            mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp_min(1e-4)
            z = (x - mean) / std
        tokens = self.embed(z)
        if self.amp_input:
            tokens = tokens + self.amplitude(torch.log(std))
        if hasattr(self, 'electrode'):
            tokens = tokens + self.electrode
        else:
            tokens = tokens + self.pos_ln(self.fourier4d(self.positions) + self.mlp4d(self.positions))
        return self.head(self.norm(self.blocks(tokens)).mean(1))


if __name__ == "__main__":  # self-check
    torch.manual_seed(0)
    x = torch.randn(8, 32, 120)
    ct = ScaleRobustChannelTransformer().eval()
    assert ct(x).shape == (8, 1536)
    ct.amplitude.weight.data.zero_(); ct.amplitude.bias.data.zero_()  # without the amplitude input: scale-free
    assert torch.allclose(ct(x), ct(3.0 * x), atol=1e-4), "token stream must ignore the recording's scale"
    net = DualStreamNet()
    assert net.train()(x).shape == (8, 1536) and net.eval()(x).shape == (8, 1536)
    net.train()(x).sum().backward()
    assert net.conv.head.weight.grad.abs().sum() > 0 and net.tokens.head.weight.grad.abs().sum() > 0
    st = SourceTokenTransformer(electrode_depth=1, source_depth=2, read_every=1).eval()
    assert st(x).shape == (8, 1536)
    assert torch.allclose(st(x), st(3.0 * x + 1.0), atol=1e-4), "per-electrode z-score: scale/offset-free"
    st.train()(x).sum().backward()
    assert st.sources.grad.abs().sum() > 0 and st.embed.weight.grad.abs().sum() > 0
    # BridgeTransformer endpoints. (1) defaults == ScaleRobustChannelTransformer: same state_dict, and the same
    # init under the same seed (so the rct_w512 runs are its default config)
    kw = dict(d_model=64, depth=2, heads=4, ff_mult=4)
    torch.manual_seed(1); rct = ScaleRobustChannelTransformer(**kw).eval()
    torch.manual_seed(1); br = BridgeTransformer(**kw).eval()
    assert all(torch.equal(a, b) for a, b in zip(rct.state_dict().values(), br.state_dict().values()))
    assert torch.allclose(rct(x), br(x), atol=1e-5), "default bridge must be the robust transformer"
    # (1b) final_norm=False on the torch block: same weights minus the LayerNorm, and its output skips that norm
    torch.manual_seed(1); nf = BridgeTransformer(**kw, final_norm=False).eval()
    assert isinstance(nf.norm, nn.Identity) and nf.head.weight.equal(br.head.weight)
    tok = torch.randn(2, 32, br.head.in_features)
    assert torch.allclose(nf.head(nf.blocks(tok).mean(1)), br.head(br.blocks(tok).mean(1)), atol=1e-5)
    # (2) all switches == REVE as FoundationEncoder runs it (resample -> z-score+clip -> REVE -> token mean -> head)
    names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
             'Fz', 'Cz', 'Pz', 'AF3', 'AF4', 'FC1', 'FC2', 'CP1', 'CP2', 'PO3', 'PO4', 'FC5', 'FC6', 'CP5', 'CP6', 'Oz']
    from braindecode.models import REVE
    ref = REVE(n_outputs=2, n_chans=32, n_times=200, chs_info=[{'ch_name': n} for n in names], embed_dim=64,
               depth=2, heads=4, head_dim=16).eval()
    ref_head = nn.Linear(64, 1536)
    br = BridgeTransformer(ch_names=names, d_model=64, depth=2, heads=4, amp_input=False, dropout=0.0,
                           block='reve', pos='reve', reve_input=True).eval()
    br.embed.load_state_dict(ref.to_patch_embedding[0].state_dict())
    for mine, theirs in ((br.mlp4d, ref.mlp4d), (br.pos_ln, ref.ln), (br.blocks, ref.transformer), (br.head, ref_head)):
        mine.load_state_dict(theirs.state_dict())
    z = nn.functional.interpolate(x, size=200, mode='linear', align_corners=False)
    z = ((z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True).clamp_min(1e-5)).clamp(-15, 15)
    want = ref_head(ref(z, return_output=True)[-1].mean(1))
    assert torch.allclose(br(x), want, atol=1e-4), "all-REVE bridge must be REVE"
    # (3) _SwitchBlock at its defaults == REVE's backbone; (4) at norm=layer, bias, relu, + final LayerNorm ==
    # nn.TransformerEncoder(norm_first, no dropout) + LayerNorm. Both by weight copy.
    tok = torch.randn(8, 32, 64)
    sw = BridgeTransformer(d_model=64, depth=2, heads=4, amp_input=False, dropout=0.0, block='switch').eval()
    for mine, (attn, ffn) in zip(sw.blocks, ref.transformer.layers):
        for m, t in ((mine.n1, attn.norm), (mine.qkv, attn.to_qkv), (mine.out, attn.to_out),
                     (mine.n2, ffn.net[0]), (mine.ff1, ffn.net[1]), (mine.ff2, ffn.net[3])):
            m.load_state_dict(t.state_dict())
    assert torch.allclose(sw.norm(sw.blocks(tok)), ref.transformer(tok), atol=1e-5), "switch block != REVE backbone"
    sw = BridgeTransformer(d_model=64, depth=2, heads=4, amp_input=False, dropout=0.0, block='switch', norm='layer',
                           bias=True, ff='relu', final_norm=True).eval()
    tb = BridgeTransformer(d_model=64, depth=2, heads=4, amp_input=False, dropout=0.0, block='torch').eval()
    for mine, t in zip(sw.blocks, tb.blocks.layers):
        mine.qkv.weight.data, mine.qkv.bias.data = t.self_attn.in_proj_weight.data, t.self_attn.in_proj_bias.data
        for m, s in ((mine.out, t.self_attn.out_proj), (mine.ff1, t.linear1), (mine.ff2, t.linear2),
                     (mine.n1, t.norm1), (mine.n2, t.norm2)):
            m.load_state_dict(s.state_dict())
    sw.norm.load_state_dict(tb.norm.state_dict())
    assert torch.allclose(sw.norm(sw.blocks(tok)), tb.norm(tb.blocks(tok)), atol=1e-5), "switch block != torch layer"
    cl = BridgeTransformer(d_model=64, depth=3, heads=4, amp_input=False, dropout=0.0, block='switch', init='cloned')
    ind = BridgeTransformer(d_model=64, depth=3, heads=4, amp_input=False, dropout=0.0, block='switch')
    assert torch.equal(cl.blocks[0].qkv.weight, cl.blocks[2].qkv.weight) and not torch.equal(
        ind.blocks[0].qkv.weight, ind.blocks[1].qkv.weight), "init='cloned' must start every layer identical"
    print("alljoined_encoders self-check ok")
