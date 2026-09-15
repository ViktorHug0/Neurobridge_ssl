"""Identity-Trap diagnostics for the THINGS-EEG-2 cross-subject alignment space.

Two frozen-representation diagnostics, run on ONE LOSO checkpoint's embedding
space with all 10 subjects' 200-way test trials projected into it (the held-out
subject is the genuine unseen outlier; every other subject was seen in training):

  1. Variance decomposition -- crossed subject x class sum-of-squares fractions
     + a matched random-Gaussian null (how subject-dominated is the space?).
  2. WSCI (200-way generalization) -- inter-subject RSA (rotation-INVARIANT: is
     the class geometry shared up to any rotation?) and raw class-centroid cosine
     consistency (same-basis). gap = RSA - raw = the part a Procrustes/SAW map
     recovers.

`crossed_ss_fractions` and the Gaussian null are ported from fmscope
(github.com/Jimmy110101013/fmscope, diagnostics/variance.py + null_control.py);
model loading + feature extraction reuse evaluate.py unchanged.

Run:  see run_identity_diagnostics.sh   (or --selftest for the offline check)
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

# repo imports (evaluate.py already imports these from train.py)
from torch.utils.data import DataLoader

from evaluate import _load_json, _to_bool
from module.dataset import EEGPreImageDataset
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything


# --------------------------------------------------------------------------- #
# Variance decomposition  (vendored from fmscope, numpy-only)                  #
# --------------------------------------------------------------------------- #
def crossed_ss_fractions(features, subject, label):
    """Two-factor crossed sum-of-squares fractions against the grand mean.

    label_frac and subject_frac are computed INDEPENDENTLY (not a partition);
    for a crossed design (label varies within subject, as here) they sum to <=1
    and residual = 1 - both. Ported from fmscope diagnostics/variance.py.
    """
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)
    grand = f.mean(axis=0, keepdims=True)
    diff = f - grand
    ss_total = float((diff * diff).sum())

    def _between(groups):
        ss = 0.0
        for g in np.unique(groups):
            m = groups == g
            if m.sum() == 0:
                continue
            d = f[m].mean(0) - grand.squeeze()
            ss += float(m.sum()) * float((d * d).sum())
        return ss

    ss_label = _between(y)
    ss_subject = _between(s)
    t = max(ss_total, 1e-18)
    frac_label = ss_label / t
    frac_subject = ss_subject / t
    return {
        "label_frac": frac_label,
        "subject_frac": frac_subject,
        "residual_frac": max(1.0 - frac_label - frac_subject, 0.0),
        "SS_total": ss_total,
    }


def gaussian_null(subject, label, n, d, n_seeds=20, seed=0):
    """Matched random-Gaussian null: excess = real / null is what proves the
    subject dominance is FM structure, not ANOVA combinatorics (S vs L df)."""
    rng = np.random.default_rng(seed)
    subj_fracs, lab_fracs = [], []
    for _ in range(n_seeds):
        rand = rng.standard_normal((n, d))
        out = crossed_ss_fractions(rand, subject, label)
        subj_fracs.append(out["subject_frac"])
        lab_fracs.append(out["label_frac"])
    n_subj = int(np.unique(subject).size)
    n_lab = int(np.unique(label).size)
    return {
        "null_subject_frac_mean": float(np.mean(subj_fracs)),
        "null_label_frac_mean": float(np.mean(lab_fracs)),
        "df_pred_subject": (n_subj - 1) / max(n - 1, 1),  # closed-form (S-1)/(N-1)
        "df_pred_label": (n_lab - 1) / max(n - 1, 1),
        "n_seeds": n_seeds,
    }


def variance_decomposition(features, subject, label, unseen_subject, null_cap=20000, seed=0):
    real = crossed_ss_fractions(features, subject, label)
    excess_subject = (real["subject_frac"] / max(gaussian_null(subject, label, len(features), features.shape[1], seed=seed)["null_subject_frac_mean"], 1e-12))

    # matched excess: subsample rows (stratified by subject) so real & null share (N, group sizes)
    rng = np.random.default_rng(seed)
    if len(features) > null_cap:
        keep = []
        per = max(1, null_cap // int(np.unique(subject).size))
        for sid in np.unique(subject):
            idx = np.where(subject == sid)[0]
            keep.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
        keep = np.concatenate(keep)
    else:
        keep = np.arange(len(features))
    real_sub = crossed_ss_fractions(features[keep], subject[keep], label[keep])
    null = gaussian_null(subject[keep], label[keep], len(keep), features.shape[1], seed=seed)

    # per-subject offset from the grand mean (the "outlier" read: does the unseen subject sit apart?)
    grand = features.mean(0)
    per_subject_offset = {}
    for sid in np.unique(subject):
        c = features[subject == sid].mean(0)
        per_subject_offset[int(sid)] = float(np.linalg.norm(c - grand))

    return {
        "full": real,
        "matched_subsample": {
            "n": int(len(keep)),
            "real": real_sub,
            "null": null,
            "excess_subject": float(real_sub["subject_frac"] / max(null["null_subject_frac_mean"], 1e-12)),
            "excess_label": float(real_sub["label_frac"] / max(null["null_label_frac_mean"], 1e-12)),
        },
        "per_subject_offset_to_grand": per_subject_offset,
        "unseen_subject": int(unseen_subject),
    }


# --------------------------------------------------------------------------- #
# WSCI (200-way): RSA (rotation-invariant) + raw class-centroid cosine         #
# --------------------------------------------------------------------------- #
def _subject_class_centroids(features, subject, label):
    """Return (subject_ids, class_ids, C) with C[s, c] = mean embedding of
    subject s on class c, centred by that subject's mean over classes."""
    subs = np.unique(subject)
    classes = np.unique(label)
    d = features.shape[1]
    C = np.full((len(subs), len(classes), d), np.nan, dtype=np.float64)
    for i, sid in enumerate(subs):
        sm = subject == sid
        for j, c in enumerate(classes):
            m = sm & (label == c)
            if m.any():
                C[i, j] = features[m].mean(0)
        # centre out this subject's DC offset so we compare geometry, not location
        C[i] -= np.nanmean(C[i], axis=0, keepdims=True)
    return subs, classes, C


def _rank(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def wsci(features, subject, label, unseen_subject):
    subs, classes, C = _subject_class_centroids(features, subject, label)
    n_sub = len(subs)

    # --- raw same-basis cosine consistency: per class, mean pairwise cos across subjects ---
    Cn = C / np.clip(np.linalg.norm(C, axis=2, keepdims=True), 1e-12, None)
    per_class_cos = []
    for j in range(len(classes)):
        V = Cn[:, j, :]                       # (n_sub, d)
        valid = ~np.isnan(V).any(1)
        Vv = V[valid]
        if len(Vv) >= 2:
            G = Vv @ Vv.T
            per_class_cos.append(G[np.triu_indices(len(Vv), 1)].mean())
    raw_cos_mean = float(np.mean(per_class_cos)) if per_class_cos else float("nan")

    # --- RSA: per-subject 200x200 cosine RDM, Spearman-correlate across subject pairs ---
    rdm_ut = []                                # upper-triangle of each subject's RDM (ranked)
    iu = np.triu_indices(len(classes), 1)
    for i in range(n_sub):
        V = Cn[i]
        V = np.where(np.isnan(V), 0.0, V)
        R = V @ V.T
        rdm_ut.append(_rank(R[iu]))
    rdm_ut = np.stack(rdm_ut)                   # (n_sub, n_pairs), rank-transformed
    # Spearman(i,j) = Pearson of ranks
    rdm_ut = (rdm_ut - rdm_ut.mean(1, keepdims=True)) / np.clip(rdm_ut.std(1, keepdims=True), 1e-12, None)
    RSA = (rdm_ut @ rdm_ut.T) / rdm_ut.shape[1]
    rsa_pairs = RSA[np.triu_indices(n_sub, 1)]
    rsa_mean = float(rsa_pairs.mean())

    # per-subject: mean RSA / mean raw-cos to all OTHERS (flag the unseen subject)
    per_subject = {}
    for i, sid in enumerate(subs):
        others = [k for k in range(n_sub) if k != i]
        rsa_to_others = float(RSA[i, others].mean())
        raw_to_others = []
        for j in range(len(classes)):
            v = Cn[i, j]
            if np.isnan(v).any():
                continue
            for k in others:
                w = Cn[k, j]
                if not np.isnan(w).any():
                    raw_to_others.append(float(v @ w))
        per_subject[int(sid)] = {
            "rsa_to_others": rsa_to_others,
            "raw_cos_to_others": float(np.mean(raw_to_others)) if raw_to_others else float("nan"),
        }

    return {
        "raw_cos_mean": raw_cos_mean,       # same-basis: shared class directions?
        "rsa_mean": rsa_mean,               # rotation-invariant: shared geometry?
        "gap_rsa_minus_raw": rsa_mean - raw_cos_mean,  # what alignment recovers
        "n_subjects": n_sub,
        "n_classes": int(len(classes)),
        "per_subject_to_others": per_subject,
        "unseen_subject": int(unseen_subject),
    }


# --------------------------------------------------------------------------- #
# Embedding extraction (reuse evaluate.py loader)                             #
# --------------------------------------------------------------------------- #
def extract_embeddings(checkpoint_dir, subject_ids, device_str):
    """Return (align_feats, backbone_feats, subject, class, unseen_subject).

    One forward pass captures BOTH the pre-projector backbone (the fair analog
    of the paper's frozen features -- and the clean-xadv attachment point) and
    the post-projector alignment space (what retrieval actually scores).
    """
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_test_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint_test_best.pth in {checkpoint_dir}")
    train_cfg = _load_json(os.path.join(checkpoint_dir, "train_config.json"))
    eval_args = Namespace(**train_cfg)
    eval_args.device = device_str
    seed_everything(seed=train_cfg.get("seed", 2099))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    unseen = int(train_cfg.get("test_subject_ids", [subject_ids[0]])[0])

    checkpoint = torch.load(ckpt_path, map_location=device)

    dataset = EEGPreImageDataset(
        list(subject_ids), eval_args.eeg_data_dir, eval_args.selected_channels, eval_args.time_window,
        eval_args.image_feature_dir, getattr(eval_args, "text_feature_dir", ""), False,
        getattr(eval_args, "aug_image_feature_dirs", []), False, False, None, False,
        _to_bool(getattr(eval_args, "image_test_aug", False)),
        _to_bool(getattr(eval_args, "eeg_test_aug", False)),
        _to_bool(getattr(eval_args, "frozen_eeg_prior", False)),
    )
    img_dim = dataset.image_features.shape[-1]
    backbone_dim = getattr(eval_args, "eeg_backbone_dim", 0) or img_dim
    act = getattr(eval_args, "projector_activation", "none")
    topk = getattr(eval_args, "projector_topk", 512)

    model = build_eeg_encoder(eval_args, backbone_dim, dataset.num_sample_points, dataset.channels_num).to(device)
    eeg_projector = build_projector(eval_args.projector, backbone_dim, eval_args.feature_dim, activation=act, topk=topk).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    model.eval(); eeg_projector.eval()

    loader = DataLoader(dataset, batch_size=getattr(eval_args, "eval_batch_size", 512), shuffle=False,
                        num_workers=getattr(eval_args, "num_workers", 0))
    align_l, back_l, subj_l, obj_l = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg_batch = batch[0].to(device)
            sid = batch[3].to(device)
            back = run_eeg_backbone(model, eval_args, eeg_batch, sid)
            align = eeg_projector(back)
            back_l.append(back.cpu().numpy()); align_l.append(align.cpu().numpy())
            subj_l.append(batch[3].numpy()); obj_l.append(batch[4].numpy())
    return (np.concatenate(align_l).astype(np.float32),
            np.concatenate(back_l).astype(np.float32),
            np.concatenate(subj_l).astype(np.int64),
            np.concatenate(obj_l).astype(np.int64), unseen)


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #
def make_figure(var_out, wsci_out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    f = var_out["full"]
    ax1.bar(["variance"], [f["subject_frac"]], label="subject", color="#c0504d")
    ax1.bar(["variance"], [f["label_frac"]], bottom=[f["subject_frac"]], label="class", color="#4f81bd")
    ax1.bar(["variance"], [f["residual_frac"]], bottom=[f["subject_frac"] + f["label_frac"]], label="residual", color="#d9d9d9")
    nl = var_out["matched_subsample"]["null"]["null_subject_frac_mean"]
    ax1.axhline(nl, ls="--", color="k", lw=1)
    ax1.text(0, nl, f" Gaussian null subj={nl:.3f}", va="bottom", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("fraction of embedding variance")
    ax1.set_title(f"Variance decomposition\nsubj {f['subject_frac']:.2f} (×{var_out['matched_subsample']['excess_subject']:.0f} null) | "
                  f"class {f['label_frac']:.3f}")
    ax1.legend(fontsize=8, loc="upper right")

    labels = ["raw cos\n(same basis)", "RSA\n(rot-invariant)", "gap\n(=alignment)"]
    vals = [wsci_out["raw_cos_mean"], wsci_out["rsa_mean"], wsci_out["gap_rsa_minus_raw"]]
    ax2.bar(labels, vals, color=["#4f81bd", "#9bbb59", "#f79646"])
    ax2.axhline(0, color="k", lw=0.8)
    # overlay the unseen subject's to-others RSA as a marker
    u = wsci_out["unseen_subject"]
    ps = wsci_out["per_subject_to_others"]
    seen = [v["rsa_to_others"] for k, v in ps.items() if k != u]
    ax2.scatter([1] * len(seen), seen, color="gray", s=18, zorder=3, label="seen subj RSA→others")
    if u in ps:
        ax2.scatter([1], [ps[u]["rsa_to_others"]], color="red", s=48, zorder=4, marker="D", label=f"unseen (sub-{u})")
    ax2.set_title("WSCI (200-way)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Self-check (no framework): synthetic data with known structure               #
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    n_sub, n_cls, reps, d = 10, 200, 20, 64
    class_sig = rng.standard_normal((n_cls, d))          # SHARED class geometry across subjects
    subj_off = rng.standard_normal((n_sub, d)) * 3.0     # strong per-subject offset
    feats, subj, lab = [], [], []
    for s in range(n_sub):
        for c in range(n_cls):
            x = class_sig[c] + subj_off[s] + rng.standard_normal((reps, d)) * 0.5
            feats.append(x); subj += [s] * reps; lab += [c] * reps
    feats = np.concatenate(feats); subj = np.array(subj); lab = np.array(lab)

    v = variance_decomposition(feats, subj, lab, unseen_subject=0)
    w = wsci(feats, subj, lab, unseen_subject=0)
    # subject offset is strong => subject_frac dominates and beats the null by a lot
    assert v["full"]["subject_frac"] > v["full"]["label_frac"], v["full"]
    assert v["matched_subsample"]["excess_subject"] > 5, v["matched_subsample"]
    # class geometry is shared in the SAME basis (identical class_sig) => raw cos high, RSA high, small gap
    assert w["raw_cos_mean"] > 0.5, w
    assert w["rsa_mean"] > 0.8, w
    # rotate one subject's centroids => raw cos should drop but RSA (rotation-invariant) stay high
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    feats_rot = feats.copy()
    m0 = subj == 0
    feats_rot[m0] = feats[m0] @ Q
    w2 = wsci(feats_rot, subj, lab, unseen_subject=0)
    assert w2["per_subject_to_others"][0]["rsa_to_others"] > 0.6, "RSA must survive rotation"
    assert w2["per_subject_to_others"][0]["raw_cos_to_others"] < w["per_subject_to_others"][0]["raw_cos_to_others"], "raw cos must drop under rotation"
    print("selftest OK:",
          f"subj_frac={v['full']['subject_frac']:.3f} excess={v['matched_subsample']['excess_subject']:.1f} "
          f"raw={w['raw_cos_mean']:.3f} rsa={w['rsa_mean']:.3f} gap={w['gap_rsa_minus_raw']:.3f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint_dir", type=str, help="a single LOSO run dir (has checkpoint_test_best.pth + train_config.json)")
    ap.add_argument("--subject_ids", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--output_dir", type=str, default="results/things_eeg/identity_diag")
    ap.add_argument("--output_name", type=str, default="diag")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--null_cap", type=int, default=20000, help="rows for the matched real-vs-null excess (stratified by subject)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.checkpoint_dir:
        ap.error("--checkpoint_dir is required (or use --selftest)")

    align, backbone, subj, obj, unseen = extract_embeddings(args.checkpoint_dir, args.subject_ids, args.device)
    print(f"align {align.shape} | backbone {backbone.shape} | {len(np.unique(subj))} subjects | "
          f"{len(np.unique(obj))} classes | unseen=sub-{unseen}")

    layers = {"align": align, "backbone": backbone}
    out = {"checkpoint_dir": os.path.abspath(args.checkpoint_dir), "n_windows": int(align.shape[0])}
    for name, feats in layers.items():
        out[name] = {
            "embedding_dim": int(feats.shape[1]),
            "variance_decomposition": variance_decomposition(feats, subj, obj, unseen, null_cap=args.null_cap),
            "wsci": wsci(feats, subj, obj, unseen),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    # figure uses the align layer (the retrieval space)
    make_figure(out["align"]["variance_decomposition"], out["align"]["wsci"],
                os.path.join(args.output_dir, f"{args.output_name}.png"))

    for name in ("backbone", "align"):
        v = out[name]["variance_decomposition"]["full"]
        exc = out[name]["variance_decomposition"]["matched_subsample"]["excess_subject"]
        w = out[name]["wsci"]
        print(f"\n[{name:8s} variance] subject={v['subject_frac']:.3f}  class={v['label_frac']:.4f}  "
              f"residual={v['residual_frac']:.3f}  (subject ×{exc:.0f} over null)")
        print(f"[{name:8s} wsci    ] raw_cos={w['raw_cos_mean']:.3f}  RSA={w['rsa_mean']:.3f}  gap={w['gap_rsa_minus_raw']:.3f}")
    bsub = out["backbone"]["variance_decomposition"]["full"]["subject_frac"]
    asub = out["align"]["variance_decomposition"]["full"]["subject_frac"]
    print(f"\n>> clean-xadv premise: backbone subject_frac={bsub:.3f} vs align {asub:.3f} "
          f"({'backbone carries more subject -> a real target' if bsub > asub + 0.02 else 'backbone already subject-suppressed too -> low headroom'})")
    print(f"wrote {json_path} + .png")


if __name__ == "__main__":
    main()
