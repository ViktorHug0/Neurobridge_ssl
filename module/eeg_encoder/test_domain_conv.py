# Self-check for per-dataset BatchNorm routing (TSConv, Squeezeformer, ATM): .venv/bin/python module/eeg_encoder/test_domain_conv.py
import torch, sys
sys.path.insert(0, '.')
from module.eeg_encoder.model import TSConv_parameterizable, DomainBatchNorm2d
torch.manual_seed(0)
m = TSConv_parameterizable(feature_dim=1536, eeg_sample_points=120, channels_num=32, pool_kernel=25, pool_stride=2,
                           domain_batch_norm_boundary=21).eval()
assert sum(isinstance(b, DomainBatchNorm2d) for b in m.modules()) == 2
x = torch.randn(6, 32, 120); ids = torch.tensor([2, 22, 5, 30, 1, 21])
out = m(x, subject_ids=ids)
assert torch.allclose(out[ids < 21], m(x, subject_ids=None)[ids < 21], atol=1e-6)            # no IDs = primary
with torch.no_grad():
    for b in m.modules():
        if isinstance(b, DomainBatchNorm2d): b.auxiliary.running_mean.add_(5.0)
assert torch.allclose(out[ids < 21], m(x, subject_ids=ids)[ids < 21], atol=1e-6) and not torch.allclose(
    out[ids >= 21], m(x, subject_ids=ids)[ids >= 21])                                       # THINGS rows use auxiliary only
unk = torch.full_like(ids, 10 ** 6)  # how score_track1 marks unseen subjects: must take the primary path
assert torch.allclose(m(x, subject_ids=unk), m(x, subject_ids=None), atol=1e-6)
m.train(); m(x, subject_ids=ids).sum().backward()
assert all(b.auxiliary.weight.grad is not None for b in m.modules() if isinstance(b, DomainBatchNorm2d))
print('domain BatchNorm TSConv ok')

# Squeezeformer with per-dataset BatchNorm (stem + every conformer conv block)
from module.eeg_encoder.ortho_encoders import FastTSConvSqueezeformer
torch.manual_seed(0)
base = FastTSConvSqueezeformer(1536, 120, 32, pool_kernel=25, pool_stride=2)
dbn = FastTSConvSqueezeformer(1536, 120, 32, pool_kernel=25, pool_stride=2, domain_bn_boundary=21)
assert sum(isinstance(m, DomainBatchNorm2d) for m in dbn.modules()) == 2 + 3  # stem x2, one per block
assert not any(isinstance(m, DomainBatchNorm2d) for m in base.modules())       # default build unchanged
x2 = torch.randn(6, 32, 120)
dbn.train(); dbn(x2, subject_ids=ids).sum().backward()                          # both statistics paths train
assert all(m.auxiliary.weight.grad is not None for m in dbn.modules() if isinstance(m, DomainBatchNorm2d))
dbn.eval()
with torch.no_grad():
    for m in dbn.modules():
        if isinstance(m, DomainBatchNorm2d): m.auxiliary.running_mean.add_(5.0)
    assert torch.allclose(dbn(x2, subject_ids=torch.full_like(ids, 10 ** 6)), dbn(x2), atol=1e-6)  # unseen -> primary
    assert not torch.allclose(dbn(x2, subject_ids=ids)[ids >= 21], dbn(x2)[ids >= 21])          # THINGS -> auxiliary
print('domain BatchNorm squeezeformer ok')

# ATM with per-dataset BatchNorm (the conv stem's two BNs)
from module.eeg_encoder.atm.atm import ATMS
torch.manual_seed(0)
kw = dict(channels_num=32, feature_dim=1536, eeg_sample_points=120, subject_token=False, backbone_dim=1024, pool_kernel=25, pool_stride=2)
base, dbn = ATMS(**kw), ATMS(**kw, domain_bn_boundary=21)
assert sum(isinstance(m, DomainBatchNorm2d) for m in dbn.modules()) == 2
assert not any(isinstance(m, DomainBatchNorm2d) for m in base.modules())
dbn.train(); dbn(x2, ids).sum().backward()
assert all(m.auxiliary.weight.grad is not None for m in dbn.modules() if isinstance(m, DomainBatchNorm2d))
dbn.eval(); base.eval()
with torch.no_grad():
    sd = {k.replace('.primary.', '.'): v for k, v in dbn.state_dict().items() if '.auxiliary.' not in k}
    base.load_state_dict(sd)
    assert torch.allclose(dbn(x2, ids)[ids < 21], base(x2, ids)[ids < 21], atol=1e-5)          # Alljoined = plain ATM path
    for m in dbn.modules():
        if isinstance(m, DomainBatchNorm2d): m.auxiliary.running_mean.add_(5.0)
    assert torch.allclose(dbn(x2, torch.full_like(ids, 10 ** 6)), base(x2, ids), atol=1e-5)    # unseen -> primary
    assert not torch.allclose(dbn(x2, ids)[ids >= 21], base(x2, ids)[ids >= 21])             # THINGS -> auxiliary
print('domain BatchNorm ATM ok')
