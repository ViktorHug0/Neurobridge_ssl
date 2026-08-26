"""Rank encoders by how well their embedding predicts the REAL image's Q-Former tokens.

E0 ranked encoders on cos(EEG prediction, image embedding), which is contaminated: train.py learns
the image projector too, so the target moves. At --alpha 0 the projector simply collapses the image
side onto the EEG and cosine hits 0.998 with 2% top-1. The target's own effective rank falls
monotonically well before that (95.3 -> 72.5), so the confound is present throughout, not just at
the degenerate end.

qformer_{training,test}.npy are cached from real images by BLIP-2's own frozen ViT-g + Q-Former and
depend on no encoder, so they are a fixed target. This fits the bridge's regression to them in
closed form -- for a linear map under MSE, training with additive isotropic input noise of variance
sigma^2 is ridge with lambda proportional to sigma^2 (Bishop 1995), so one eigendecomposition gives
the whole sigma sweep. Drops the bridge's CE term and the re-normalisation, so it screens rather
than selects.

  .venv/bin/python reconstruction_experiments/caption_ridge.py --enigma
  .venv/bin/python reconstruction_experiments/caption_ridge.py --npz <npz> --checkpoint_dir <ckpt> \
      --feature_dir <dir> --tag mse05
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

CAP = "data/things_eeg/captions"
ENIG = "data/things_eeg/enigma"
VITH = "data/things_eeg/image_feature/ViT-H-14_final"
# every encoder picked 100, the old grid's top -- a truncated grid reports floors, not optima
LAMBDAS = [10.0 ** k for k in range(-3, 9)]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def project(feature_dir, split, checkpoint_dir, feature_dim):
    f = np.load(os.path.join(feature_dir, f"image_{split}.npy"))
    x = torch.from_numpy(f.reshape(-1, f.shape[-1])).float()
    if checkpoint_dir:
        from module.projector import ProjectorLinear
        ck = torch.load(os.path.join(checkpoint_dir, "checkpoint_test_best.pth"),
                        map_location="cpu", weights_only=False)
        head = ProjectorLinear(x.shape[-1], feature_dim)
        head.load_state_dict(ck["img_projector_state_dict"])
        with torch.no_grad():
            x = head(x)
    return unit(x.numpy())


def save_bridge(path, w, ym, lam):
    """Write the ridge solution in caption_bridge.py's .pth format so --stage generate reads it.

    nn.Linear stores (out, in) and adds a bias, so W.T and the target mean drop straight in. This
    IS the --ce_weight 0 bridge: MSE on a linear map has a closed form, and ridge lambda is the
    input-noise sigma by the Bishop equivalence. Seconds instead of the ~2 h an SGD run costs,
    almost all of which is the CE term backpropagating through the frozen OPT.
    """
    torch.save({"state_dict": {"weight": torch.from_numpy(w.T.copy()).float(),
                               "bias": torch.from_numpy(ym.copy()).float()},
                "in_dim": w.shape[0], "n_tok": 32, "d_tok": 768, "lambda": lam}, path)
    print(f"wrote {path}  lambda={lam:g}  in_dim={w.shape[0]}")


def score(tag, x_tr, x_te, save=None, save_lams=()):
    """Fit image -> Q-Former on the training split, evaluate on EEG-predicted test embeddings."""
    y_tr = np.load(f"{CAP}/qformer_training.npy").reshape(len(x_tr), -1).astype(np.float32)
    y_te = np.load(f"{CAP}/qformer_test.npy").reshape(len(x_te), -1).astype(np.float32)
    ym = y_tr.mean(0)
    y_tr = y_tr - ym
    a = x_tr.T @ x_tr
    b = x_tr.T @ y_tr
    d, v = np.linalg.eigh(a)
    vb = v.T @ b
    # R^2 against the training mean, the same baseline caption_bridge.py prints before training
    denom = ((y_te - ym) ** 2).sum()
    best = None
    # save_lams may name values off the decade grid, so iterate over the union
    for lam in sorted(set(LAMBDAS) | set(save_lams)):
        w = v @ (vb / (d + lam)[:, None])
        if lam in save_lams:
            save_bridge(f"{save}_lam{lam:g}.pth", w, ym, lam)
        pred = x_te @ w + ym
        r2 = 1 - ((pred - y_te) ** 2).sum() / denom
        cos = float(np.mean(np.sum(unit(pred - ym) * unit(y_te - ym), axis=1)))
        if best is None or cos > best[2]:   # cos, not R2 -- see the header note
            best = (lam, r2, cos)
    print(f"{tag:14s} {best[0]:9.3g} {best[1]:8.4f} {best[2]:7.3f}"
          f"{'  <- lambda at grid edge' if best[0] in (LAMBDAS[0], LAMBDAS[-1]) else ''}")


# Rank on cos_qf, not R2. The two disagree and cos_qf is the one that reproduces the known
# cwBLEU ordering: ENIGMA scored 7.09 against base's 6.51, and cos_qf puts ENIGMA ahead (0.268 vs
# 0.238) while R2 inverts it (0.0533 vs 0.0560). R2 is scale-sensitive and the two spaces differ in
# width (1024-d ViT-H vs 512-d projected), so it is not comparable across them.
HEAD = f"{'encoder':14s} {'lambda*':>9s} {'R2':>8s} {'cos_qf':>7s}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enigma", action="store_true")
    p.add_argument("--npz", default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--feature_dir", default="data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit")
    p.add_argument("--feature_dim", type=int, default=512)
    p.add_argument("--tag", default=None)
    p.add_argument("--no_header", action="store_true")
    p.add_argument("--save", default=None, help="prefix; writes a bridge .pth per --save_lams")
    p.add_argument("--pred_npy", default=None,
                   help="fit on these predicted training embeddings instead of real image ones")
    p.add_argument("--save_lams", type=float, nargs="*", default=[],
                   help="lambdas to dump. Doubles as the sigma sweep: lambda ~ sigma^2")
    a = p.parse_args()
    if not a.no_header:
        print(HEAD)

    if a.enigma:
        x_tr = unit(np.load(a.pred_npy).astype(np.float32)) if a.pred_npy \
            else project(VITH, "train", None, 0)
        x_te = unit(np.load(f"{ENIG}/enigma_pred_test_sub-01.npy"))
        score("enigma", x_tr, x_te, a.save, a.save_lams)
    if a.npz:
        x_tr = project(a.feature_dir, "train", a.checkpoint_dir, a.feature_dim)
        d = np.load(a.npz)
        assert (d["object"] == np.arange(len(d["eeg"]))).all(), "npz rows are not in concept order"
        score(a.tag or os.path.basename(a.npz), x_tr, unit(d["eeg"].astype(np.float32)),
              a.save, a.save_lams)


if __name__ == "__main__":
    main()
