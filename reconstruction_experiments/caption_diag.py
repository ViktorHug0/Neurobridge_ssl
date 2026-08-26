"""Three numbers that decide whether an embedding space is worth a 2h bridge (next_experiments E0).

  cos       mean cos(EEG prediction, true image embedding). Sets sigma via cos ~ 1/sqrt(1+sigma^2),
            which is how ENIGMA's 0.36 gave 2.59.
  eff_rank  exp(entropy of the covariance spectrum). What matters is the ratio to the image side
            under one convention, not the absolute value: 0.40 for ENIGMA, 0.36 for the 81%-top-1
            InternViT encoder. Both predictions are substantially lower-rank than their targets.
  norm_cv   coefficient of variation of the predicted norm. Near zero means magnitude carries no
            information and both sides should be normalised.
  top1      200-way retrieval, printed only so the dissociation from cos is visible in one table.

  .venv/bin/python reconstruction_experiments/caption_diag.py --enigma
  .venv/bin/python reconstruction_experiments/caption_diag.py --npz feats/sub1.npz --checkpoint_dir <ckpt>
"""
import argparse
import os

import numpy as np
import torch

ENIG = "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/enigma"
VITH = "data/things_eeg/image_feature/ViT-H-14_final"


def eff_rank(x, sq=True):
    """exp(H(p)) over the normalised spectrum of the centred matrix.

    sq=True weights by s^2, the covariance eigenvalues, i.e. how many directions carry the
    variance. sq=False is Roy & Vetterli's original on s itself and reads ~3x higher on the same
    data, so the two are not comparable -- the project's earlier 34/17 vs 97/65 figures are the
    squared convention and everything here follows it.
    """
    s = np.linalg.svd(x - x.mean(0, keepdims=True), compute_uv=False)
    w = s ** 2 if sq else s
    p = w / w.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def report(name, pred, true, top1=True):
    pn = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    tn = true / np.linalg.norm(true, axis=1, keepdims=True)
    cos = float((pn * tn).sum(1).mean())
    norms = np.linalg.norm(pred, axis=1)
    acc = ""
    if top1 and len(pred) == len(true):
        # 200-way only makes sense when rows are one-per-concept
        hit = (pn @ tn.T).argmax(1) == np.arange(len(pn))
        acc = f"{100 * hit.mean():7.2f}"
    print(f"{name:34s} {cos:6.3f} {eff_rank(pred):8.1f} {eff_rank(true):9.1f} "
          f"{norms.mean():8.2f} {100 * norms.std() / norms.mean():7.2f} {acc:>7s} "
          f"{(1 / max(cos, 1e-6) ** 2 - 1) ** 0.5:8.2f}")


HEAD = (f"{'space':34s} {'cos':>6s} {'rank_p':>8s} {'rank_t':>9s} {'norm':>8s} "
        f"{'cv%':>7s} {'top1':>7s} {'sigma':>8s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enigma", action="store_true", help="ENIGMA predictions vs raw ViT-H")
    p.add_argument("--npz", default=None, help="evaluate.py --dump_npz, test split")
    p.add_argument("--checkpoint_dir", default=None, help="--npz: projects images to the shared space")
    p.add_argument("--feature_dim", type=int, default=512)
    p.add_argument("--feature_dir",
                   default="data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit",
                   help="--npz: image features the encoder was aligned to")
    p.add_argument("--name", default=None, help="row label")
    a = p.parse_args()
    print(HEAD)

    if a.enigma:
        for split in ("training", "test"):
            f = os.path.join(ENIG, f"enigma_pred_{split}_sub-01.npy")
            if not os.path.exists(f):
                print(f"{'enigma ' + split:34s} (not built yet)")
                continue
            pred = np.load(f)
            t = np.load(os.path.join(VITH, f"image_{'train' if split == 'training' else 'test'}.npy"))
            true = t.reshape(-1, t.shape[-1]).astype(np.float32)
            report(f"enigma {split}", pred, true, top1=(split == "test"))

    if a.npz:
        d = np.load(a.npz)
        pred = d["eeg"].astype(np.float32)
        t = np.load(os.path.join(a.feature_dir, "image_test.npy"))
        x = torch.from_numpy(t.reshape(-1, t.shape[-1])).float()
        from module.projector import ProjectorLinear
        ck = torch.load(os.path.join(a.checkpoint_dir, "checkpoint_test_best.pth"),
                        map_location="cpu", weights_only=False)
        head = ProjectorLinear(x.shape[-1], a.feature_dim)
        head.load_state_dict(ck["img_projector_state_dict"])
        with torch.no_grad():
            true = head(x).numpy()
        assert (d["object"] == np.arange(len(pred))).all(), "npz rows are not in concept order"
        report(a.name or os.path.basename(a.npz), pred, true)


if __name__ == "__main__":
    main()
