"""O01 a linear encoding model (image embedding -> EEG) and what it captures for never-seen images.
Motivation: distinct training images are the steepest lever measured (7.5k -> 11.5k images: +14..28 %) and all 30
training people saw the same 16.5k images. If the image-evoked part of a trial can be predicted for a new image, real
trials can be re-targeted to images nobody was recorded on (O02).
Per person p: ridge W_p from the unit-norm DINOv2-g embedding (centred) to the repeat-averaged response (32 x 120,
channels whitened with the person's trial covariance), fitted on the 11,540 trained images. On the 5,000 never-trained
images, 5k gallery, top-5 (chance 0.10 %), Gaussian matched filter (score = <x, W g> - |W g|^2 / 2):
  own     the person's own W_p, the person's r=3 mean: what a linear encoding model holds of that person's image code
  pop     new person, mean W of the Alljoined training people: the part of it that is shared
  half    reference without any image model: the person's other-repeat mean as the template of each image (within-person
          repeat matching, r=2 against r=2): how much image information the EEG holds at all
Per-cell correlations (all 3,840 cells, no selection): split-half of the 2-repeat means, and prediction vs 4-repeat mean.
    no01_encoding.py <fold> [ridge, default 1]"""
from nc_lib import *

k = int(sys.argv[1]); lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1.
NEW = FOLDS[k]; AJ = [s for s in range(1, 21) if s not in NEW][:8]; TH = [21, 22, 23, 24]
tr_img = np.setdiff1d(np.arange(16540), HELD)
G = nz(torch.tensor(np.asarray(TARGETS)).float()); mu = G[tr_img].mean(0); F = (G - mu).to(DEV)  # (16540, 1536)
A = torch.linalg.inv(F[tr_img].T @ F[tr_img] + lam * torch.eye(1536, device=DEV)) @ F[tr_img].T  # (1536, 11540)


def person(s):
    x = torch.tensor(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 32, 120)[:])
    ok = (x.abs().sum((2, 3)) > 0)
    c = torch.einsum('nct,ndt->cd', x[::40, 0], x[::40, 0]) / (len(x[::40]) * 120)  # channel covariance of single trials
    e, v = torch.linalg.eigh(c.double()); wh = (v * e.clamp_min(1e-6).rsqrt()) @ v.T  # C^-1/2
    x = torch.einsum('dc,nrct->nrdt', wh.float(), x).flatten(2)  # whitened, (16540, 4, 3840)
    okf = ok.float(); return x, ok, x.sum(1) / okf.sum(1, keepdim=True).clamp(min=1)


def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-9)).mean().item()


def top5(q, T, t):  # q (n, 3840) queries, T (5000, 3840) templates, t (n,) true rows
    T = T.to(DEV); bias = -.5 * T.pow(2).sum(1)
    S = torch.cat([q[b:b + 512].to(DEV) @ T.T + bias for b in range(0, len(q), 512)])
    return 100 * ((S > S.gather(1, t.to(DEV)[:, None])).sum(1) < 5).float().mean().item()


W = {}
for s in AJ + TH + NEW:
    x, ok, erp = person(s)
    full = ok[HELD].all(1); h = x[HELD][full]; t = torch.nonzero(full)[:, 0]
    q3 = h[:, :3].mean(1); b = erp[tr_img].mean(0)
    line = '%s %2d: split-half r %.4f' % ('NEW' if s in NEW else 'THINGS' if s > 20 else 'AJ ', s, corr(h[:, :2].mean(1), h[:, 2:].mean(1)))
    tmpl = torch.zeros(5000, 3840); tmpl[t] = h[:, 2:].mean(1) - b
    line += ' | half (no image model) top-5 %.2f %%' % top5((h[:, :2].mean(1) - b) * 2, tmpl * 2, t)  # x2: unit-variance noise per 2-repeat mean is 1/2
    if s not in NEW:
        w = (A @ (erp[tr_img] - b).to(DEV)).cpu(); W[s] = w
        pred = (F[HELD] @ w.to(DEV)).cpu()
        line += ' | own model: r(pred, 4-rep mean) %.4f, top-5 r=3 %.2f %%, r=1 %.2f %%' % (corr(pred[full], erp[HELD][full]), top5((q3 - b) * 3, pred, t), top5(h[:, 0] - b, pred, t))
    else:
        for name, grp in (('pop AJ', AJ), ('pop THINGS', TH)):
            pred = (F[HELD] @ torch.stack([W[p] for p in grp]).mean(0).to(DEV)).cpu()
            line += ' | %s: r %.4f, top-5 r=3 %.2f %%' % (name, corr(pred[full], erp[HELD][full]), top5((q3 - b) * 3, pred, t))
    print(line, flush=True); del x
