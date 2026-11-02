"""Any braindecode model as an EEG encoder (AUG_LOG 2026-10-29 decorrelated-encoder search): per-trial per-channel
z-score, then braindecode.models.<model>(n_chans, n_outputs=feature_dim, n_times, sfreq) -> (batch, feature_dim)."""
import torch.nn as nn


class Braindecode(nn.Module):
    def __init__(self, feature_dim, eeg_sample_points, channels_num, model='EEGNeX', sfreq=120, **kwargs):
        super().__init__()
        import braindecode.models as bd  # lazy: braindecode pulls in mne
        self.net = getattr(bd, model)(n_chans=channels_num, n_outputs=feature_dim, n_times=eeg_sample_points,
                                      sfreq=sfreq, **kwargs)

    def forward(self, x, subject_ids=None):
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)
        return self.net(x)


if __name__ == '__main__':
    import torch
    for name in ('EEGNeX', 'BIOT', 'EEGInceptionERP'):
        m = Braindecode(1536, 120, 32, model=name)
        x = torch.randn(4, 32, 120)
        assert m(x).shape == (4, 1536)
        assert torch.allclose(m.eval()(x), m(5 * x + 2), atol=1e-4), 'scale/offset-free input'
        print(name, 'ok')
