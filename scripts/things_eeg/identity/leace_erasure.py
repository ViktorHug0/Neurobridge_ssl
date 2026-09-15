"""LEACE subject-axis erasure on the alignment space (inductive, LOSO).

Fit the closed-form least-squares concept eraser (Belrose et al. 2023) on the
9 TRAIN subjects' alignment features -- the rank-8 between-subject-mean subspace
-- then apply it to the held-out subject's 200 query embeddings and measure the
200-way retrieval top1 delta. Inductive: the held-out subject is NOT in the fit
(honest zero-shot). The diagnostic measured align subject_frac ~= 0.029, so the
expected effect is small; this quantifies it and confirms the axis is removable
(train subject-probe BA should collapse toward chance = 1/9).

LEACE primitives vendored from fmscope (diagnostics/erasure.py). Model loading +
feature extraction + retrieval reuse evaluate.py / module.util unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluate import _encode_dataset_features, _load_json, _to_bool
from module.dataset import EEGPreImageDataset
from module.util import retrieve_all
from train import build_eeg_encoder, build_projector, seed_everything


# --------------------------------------------------------------------------- #
# LEACE primitives (vendored from fmscope diagnostics/erasure.py)             #
# --------------------------------------------------------------------------- #
def whiten(features, shrinkage=True):
    X = np.asarray(features, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    Sigma, _ = ledoit_wolf(Xc, assume_centered=True) if shrinkage else (Xc.T @ Xc / len(X), None)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)
    sq = np.sqrt(evals)
    smax = sq.max() if sq.size else 0.0
    pos = sq > 1e-8 * smax if smax > 0 else np.zeros_like(sq, bool)
    inv = np.where(pos, 1.0 / np.where(pos, sq, 1.0), 0.0)
    W = (evecs * inv) @ evecs.T
    W_plus = (evecs * sq) @ evecs.T
    return mu, Xc, W, W_plus


def subject_eraser(Xc, W, W_plus, subject):
    Xc = np.asarray(Xc, dtype=np.float64)
    pids = np.asarray(subject)
    n, d = Xc.shape
    subs = np.unique(pids)
    idx = {s: i for i, s in enumerate(subs)}
    Z = np.zeros((n, len(subs)))
    for i, s in enumerate(pids):
        Z[i, idx[s]] = 1.0
    Zc = Z - Z.mean(0)
    Sigma_XZ = Xc.T @ Zc / n
    U, s, _ = np.linalg.svd(W @ Sigma_XZ, full_matrices=False)
    r = int((s > 1e-6 * s.max()).sum()) if s.size and s.max() > 0 else 0
    Ur = U[:, :r]
    P_perp = np.eye(d) - W_plus @ (Ur @ Ur.T) @ W
    return P_perp, r


def apply_eraser(features, mu, P_perp):
    X = np.asarray(features, dtype=np.float64)
    return ((X - mu) @ P_perp.T + mu).astype(np.float32)


def subject_probe_ba(X, subject, cap=1500, n_splits=5, seed=42):
    """Linear subject-probe balanced accuracy (confirms the axis is removable)."""
    X = np.asarray(X, dtype=np.float64)
    pids = np.asarray(subject)
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(pids):
        i = np.where(pids == s)[0]
        keep.extend((rng.choice(i, cap, replace=False) if len(i) > cap else i).tolist())
    keep = np.array(keep)
    Xs, ys = X[keep], pids[keep]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bas = []
    for tr, te in skf.split(Xs, ys):
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Xs[tr]), ys[tr])
        bas.append(balanced_accuracy_score(ys[te], clf.predict(sc.transform(Xs[te]))))
    return float(np.mean(bas)), 1.0 / len(np.unique(ys))


# --------------------------------------------------------------------------- #
def load_run(checkpoint_dir, device_str):
    """Replicate evaluate.py loading; return (eval_args, modules, model, img_projector, device, unseen)."""
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_test_best.pth")
    train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_args = Namespace(**train_cfg)
    eval_args.device = device_str
    seed_everything(seed=train_cfg.get("seed", 2099))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    unseen = int(train_cfg.get("test_subject_ids", [1])[0])
    checkpoint = torch.load(ckpt_path, map_location=device)

    probe = EEGPreImageDataset(
        [unseen], eval_args.eeg_data_dir, eval_args.selected_channels, eval_args.time_window,
        eval_args.image_feature_dir, getattr(eval_args, "text_feature_dir", ""), False,
        getattr(eval_args, "aug_image_feature_dirs", []), True, False, None, False,
        _to_bool(getattr(eval_args, "image_test_aug", False)),
        _to_bool(getattr(eval_args, "eeg_test_aug", False)),
        _to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )
    img_dim = probe.image_features.shape[-1]
    backbone_dim = getattr(eval_args, "eeg_backbone_dim", 0) or img_dim
    act = getattr(eval_args, "projector_activation", "none")
    topk = getattr(eval_args, "projector_topk", 512)

    model = build_eeg_encoder(eval_args, backbone_dim, probe.num_sample_points, probe.channels_num).to(device)
    eeg_projector = build_projector(eval_args.projector, backbone_dim, eval_args.feature_dim, activation=act, topk=topk).to(device)
    img_projector = build_projector(eval_args.projector, img_dim, eval_args.feature_dim, activation=act, topk=topk).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    img_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    for m in (model, eeg_projector, img_projector):
        m.eval()
    return eval_args, {"eeg_projector": eeg_projector, "img_projector": img_projector}, model, img_projector, device, unseen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--output_dir", default="results/things_eeg/identity_diag")
    ap.add_argument("--output_name", default="leace")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fit_cap", type=int, default=2000, help="per-subject rows subsampled for the LEACE fit + probe")
    args = ap.parse_args()

    eval_args, modules, model, img_projector, device, unseen = load_run(args.checkpoint_dir, args.device)
    train_ids = [s for s in range(1, 11) if s != unseen]
    print(f"unseen=sub-{unseen} | fit LEACE on train subjects {train_ids}")

    # fit features: train subjects, align space, per-rep (subsampled)
    tr_eeg, _, tr_subj, _, _, _ = _encode_dataset_features(
        eval_args, modules, model, img_projector, device, train_ids, average=False)
    rng = np.random.default_rng(0)
    keep = np.concatenate([
        (lambda i: rng.choice(i, args.fit_cap, replace=False) if len(i) > args.fit_cap else i)(np.where(tr_subj == s)[0])
        for s in np.unique(tr_subj)])
    tr_eeg, tr_subj = tr_eeg[keep], tr_subj[keep]

    # retrieval features: held-out subject, averaged -> 200 queries + 200 image candidates
    te_eeg, te_img, _, _, _, _ = _encode_dataset_features(
        eval_args, modules, model, img_projector, device, [unseen], average=True)

    # fit LEACE (rank-8 between-train-subject-mean subspace)
    mu, Xc, W, W_plus = whiten(tr_eeg)
    P_perp, rank = subject_eraser(Xc, W, W_plus, tr_subj)

    # confirm the axis is removable: train subject-probe BA pre vs post
    ba_pre, chance = subject_probe_ba(tr_eeg, tr_subj)
    ba_post, _ = subject_probe_ba(apply_eraser(tr_eeg, mu, P_perp), tr_subj)

    # the actual question: held-out 200-way retrieval, plain vs erased
    _, c1_plain, n = retrieve_all(te_eeg, te_img, eval_mode="plain_cosine")
    _, c1_erased, _ = retrieve_all(apply_eraser(te_eeg, mu, P_perp), te_img, eval_mode="plain_cosine")
    top1_plain, top1_erased = c1_plain / n, c1_erased / n

    out = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir), "unseen_subject": unseen,
        "erased_subspace_rank": int(rank), "n_queries": int(n),
        "train_subject_probe_ba_pre": ba_pre, "train_subject_probe_ba_post": ba_post,
        "subject_probe_chance": chance,
        "top1_plain": top1_plain, "top1_erased": top1_erased,
        "top1_delta": top1_erased - top1_plain,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"[erasure] rank={rank}  train subj-probe BA {ba_pre:.3f} -> {ba_post:.3f} (chance {chance:.3f})")
    print(f"[retrieval sub-{unseen}] top1 plain={top1_plain*100:.2f}%  erased={top1_erased*100:.2f}%  "
          f"delta={100*(top1_erased-top1_plain):+.2f} pp")
    print(f"wrote {os.path.join(args.output_dir, args.output_name)}.json")


if __name__ == "__main__":
    main()
