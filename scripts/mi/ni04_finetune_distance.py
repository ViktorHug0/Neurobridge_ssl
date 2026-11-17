"""I07 is the LaBraM "fine-tuning" a fine-tuning or a retrain? Distance of the trained weights (fold 1, lag run) from the
public pretrained checkpoint, per group of layers: relative change |w - w0| / |w0| and cosine(w, w0). For scale, the
same numbers between the pretrained weights and a fresh random initialisation of the same architecture."""
import copy
from types import SimpleNamespace
from nc_lib import *
from module.eeg_encoder.model import build_eeg_encoder

f = load('mi_plab_ho5k_lag5', 1, 'checkpoint_test_best.pth')  # stop epoch 8 = the final checkpoint
cfg = copy.copy(f.cfg); ds = build_test_dataset(cfg, [FOLDS[1][0]])
dim = getattr(cfg, 'eeg_backbone_dim', 0) or ds.image_features.shape[-1]
pre = build_eeg_encoder(cfg, dim, 120, ds.channels_num).state_dict()  # constructor loads the public checkpoint
cfg2 = copy.copy(cfg); cfg2.fm_no_pretrained = True
rnd = build_eeg_encoder(cfg2, dim, 120, ds.channels_num).state_dict()
tr = {k: v.cpu() for k, v in f.model.state_dict().items()}
grp = lambda n: 'head / new layers' if not n.startswith('fm.') else 'temporal embedding (random at start)' if 'temporal_embedding' in n else \
    'patch embedding (conv front end)' if 'patch_embed' in n else 'positions / tokens' if any(t in n for t in ('position', 'cls_token', 'channel')) else \
    'blocks %s' % ('0-3' if int(n.split('blocks.')[1].split('.')[0]) < 4 else '4-7' if int(n.split('blocks.')[1].split('.')[0]) < 8 else '8-11') if 'blocks.' in n else 'other'
G = {}
for n, w0 in pre.items():
    if not torch.is_floating_point(w0) or w0.numel() < 2 or tr[n].shape != w0.shape:
        continue
    g = G.setdefault(grp(n), [0, 0, 0, 0, 0, 0, 0])
    w, r = tr[n].float().flatten(), rnd[n].float().flatten(); w0 = w0.float().flatten()
    g[0] += (w - w0).pow(2).sum(); g[1] += w0.pow(2).sum(); g[2] += (w * w0).sum(); g[3] += w.pow(2).sum(); g[4] += (r - w0).pow(2).sum(); g[5] += (r * w0).sum(); g[6] += r.pow(2).sum()
print('%-38s %-22s %-14s | %-26s %s' % ('layers', 'relative change', 'cosine', 'random init: rel. distance', 'cosine'))
for n, g in sorted(G.items()):
    print('%-38s %-22.2f %-14.3f | %-26.2f %.3f' % (n, (g[0] / g[1]).sqrt(), g[2] / (g[1] * g[3]).sqrt(), (g[4] / g[1]).sqrt(), g[5] / (g[1] * g[6]).sqrt().clamp_min(1e-9)))
print('names not grouped:', sorted({n for n in pre if grp(n) == 'other'})[:8])
print('per tensor, block 5 (cosine trained-pretrained | random-pretrained | relative change | elements):')
cos = lambda a, b: float((a.flatten() * b.flatten()).sum() / (a.norm() * b.norm()).clamp_min(1e-12))
for n, w0 in pre.items():
    if 'blocks.5.' in n and torch.is_floating_point(w0):
        print('  %-40s %.3f | %.3f | %.2f | %d' % (n.split('blocks.5.')[1], cos(tr[n].float(), w0.float()), cos(rnd[n].float(), w0.float()), float((tr[n].float() - w0.float()).norm() / w0.float().norm().clamp_min(1e-12)), w0.numel()))
big = [n for n, w0 in pre.items() if torch.is_floating_point(w0) and w0.dim() >= 2 and 'blocks.' in n]
a = torch.cat([tr[n].float().flatten() for n in big]); b = torch.cat([pre[n].float().flatten() for n in big]); c = torch.cat([rnd[n].float().flatten() for n in big])
print('all weight matrices of the 12 blocks: cosine trained-pretrained %.3f, random-pretrained %.3f, relative change %.2f, norm trained / pretrained %.2f' % (cos(a, b), cos(c, b), float((a - b).norm() / b.norm()), float(a.norm() / b.norm())))
