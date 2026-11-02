"""Score a Neurobridge checkpoint with the NeuralBench / Codabench Track-1 rules.

Three numbers come out of one pass over the **un-averaged** test epochs:

  per_epoch     cosine top-1/top-5 for every single EEG epoch      -> Codabench warm-up leaderboard
  subject_agg   predictions averaged within (subject, image) first -> NeuralBench headline metric
  instance_agg  predictions averaged across everything per image   -> NeuralBench diagnostic

The gallery is the set of unique target embeddings in the test split, exactly as
`neuralbench/callbacks.py:295` builds it and as the start-kit's `score_submission.py` does.
`train.py` cannot produce the per-epoch number: its test dataset hardcodes repetition averaging
(train.py:1932), which is the subject_agg protocol.

Ranking validity: the grader ranks a model's raw 1,536-D output against raw DINOv2 candidates,
so a checkpoint is only leaderboard-meaningful when both projectors are identities
(`--projector direct --feature_dim 1536`). Anything else is scored in the model's own alignment
space and flagged, because the grader cannot apply a learned image projector.

Everything is streamed one subject at a time: un-averaged THINGS-EEG-2 test data is ~1 GB per
subject, and per-epoch top-k needs only running hit counts, not 160k stored predictions.

Self-check: `python score_track1.py --self_test`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Run from anywhere: this script lives one level below the repo root it imports from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `module.dataset` and `train` are imported lazily inside the functions that need them.


# The grader serves 120 Hz, -0.2-0.8 s, RobustScaler+clamp tensors. Scoring on anything else
# measures a model on input it will never receive, so this is the default for every run.
OFFICIAL_EEG_DIR = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench"


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def topk_hits(preds: np.ndarray, labels: np.ndarray, gallery_normed: np.ndarray):
    """Cosine top-1/top-5 hit masks, identical to the start-kit's score_submission.py."""
    scores = _normalize(preds) @ gallery_normed.T
    order = np.argsort(scores, axis=1)[:, -5:]
    top1 = np.argmax(scores, axis=1) == labels
    top5 = np.any(order == labels[:, None], axis=1)
    return top1, top5


def topk_acc(preds: np.ndarray, labels: np.ndarray, gallery_normed: np.ndarray) -> dict:
    top1, top5 = topk_hits(preds, labels, gallery_normed)
    return {
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
        "n_queries": int(len(labels)),
    }


def build_test_dataset(cfg, subject_ids):
    from module.dataset import EEGPreImageDataset

    return EEGPreImageDataset(
        subject_ids,
        cfg.eeg_data_dir,
        cfg.selected_channels,
        cfg.time_window,
        cfg.image_feature_dir,
        getattr(cfg, "text_feature_dir", ""),
        False,
        [],
        False,  # average=False -> one item per repetition, the grader's unit
        False,  # random=False  -> deterministic (subject, object, image, rep) indexing
        None,
        False,
        False,
        False,
        False,
    )


def build_modules(cfg, checkpoint, dataset, device, n_times):
    from train import build_eeg_encoder, build_projector

    image_feature_dim = getattr(dataset, "target_dims", [dataset.image_features.shape[-1]])[0]  # aux targets: primary only
    backbone_dim = getattr(cfg, "eeg_backbone_dim", 0) or image_feature_dim
    feature_dim = int(cfg.feature_dim)

    model = build_eeg_encoder(cfg, backbone_dim, n_times, dataset.channels_num)
    eeg_projector = build_projector(cfg.projector, backbone_dim, feature_dim)
    img_projector = build_projector(cfg.projector, image_feature_dim, feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    eeg_projector.load_state_dict(checkpoint["eeg_projector_state_dict"])
    img_projector.load_state_dict(checkpoint["img_projector_state_dict"])
    for module in (model, eeg_projector, img_projector):
        module.to(device).eval()
    return model, eeg_projector, img_projector


@torch.no_grad()
def score(cfg, checkpoint, subject_ids, device, batch_size, num_workers) -> dict:
    native_n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
    probe = build_test_dataset(cfg, [subject_ids[0]])
    modules = build_modules(cfg, checkpoint, probe, device, native_n_times)
    del probe
    return score_modules(*modules, cfg, subject_ids, device, batch_size, num_workers)


FP_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30))  # rfft bins of a 120-sample trial at 120 Hz = Hz


def eeg_fingerprint(X, bands=FP_BANDS):
    """(B, 32, T) raw EEG -> (B, 656) label-free per-trial features: log channel variance, channel correlations
    (upper triangle), log band power per channel."""
    X = X.double(); b = X - X.mean(-1, keepdim=True)
    v = (b ** 2).mean(-1) + 1e-6
    c = torch.einsum('nct,ndt->ncd', b, b) / b.shape[-1] / torch.sqrt(v[:, :, None] * v[:, None, :])
    iu = torch.triu_indices(X.shape[1], X.shape[1], 1)
    P = torch.fft.rfft(b, dim=-1).abs() ** 2
    bp = torch.cat([torch.log(P[..., lo:hi].mean(-1) + 1e-6) for lo, hi in bands], 1)
    return torch.cat([torch.log(v), c[:, iu[0], iu[1]], bp], 1).float()


def fit_fingerprint_wccn(data_dir, subjects, shrink=0.3, n_per=3000, seed=0):
    """offline (TTA_LOG T4-T6): standardisation + within-person whitening (WCCN, all 656 fingerprint dims) from
    `subjects`' training-image trials. Same hardware only (Alljoined people, not THINGS-EEG2)."""
    Fs = []
    for s in subjects:  # per-subject fingerprints cached once: random reads of train.npy saturate the NAS
        cache = f'{data_dir}/.cache/fp_train_s{s:02d}_n{n_per}_seed{seed}_{"-".join(f"{lo}_{hi}" for lo, hi in FP_BANDS)}.npy'
        if not os.path.exists(cache):
            x = np.load(f'{data_dir}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)
            idx = np.sort(np.random.default_rng([seed, s]).choice(len(x), 4 * n_per, replace=False))
            xs = np.asarray(x[idx], np.float32); xs = xs[np.abs(xs).sum((1, 2)) > 0][:n_per]  # skip unseen-image padding
            np.save(cache + '.tmp.npy', eeg_fingerprint(torch.tensor(xs)).numpy()); os.replace(cache + '.tmp.npy', cache)
        Fs.append(np.load(cache))
    X = np.concatenate(Fs).astype(np.float64); m0, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - m0) / sd; y = np.repeat(np.arange(len(Fs)), [len(f) for f in Fs])
    M = np.stack([Xs[y == i].mean(0) for i in range(len(Fs))]); C = Xs - M[y]
    Sw = C.T @ C / len(C); Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    w, V = np.linalg.eigh(Sw)
    return {'m0': torch.tensor(m0, dtype=torch.float32), 'sd': torch.tensor(sd, dtype=torch.float32),
            'T': torch.tensor((V / np.sqrt(np.maximum(w, 1e-12))) @ V.T, dtype=torch.float32)}


def lw_shrink(Y, d):
    """uncentred Ledoit-Wolf whitening of one batch (TTA_LOG T23): batch direction i (eigenvalue lam_i of Y^T Y / B) is
    scaled by sqrt(d nu / ((1 - d) lam_i + d nu)), nu = mean eigenvalue over all D dims; d = 1 is the identity"""
    _, sv, Vt = torch.linalg.svd(Y, full_matrices=False); lam = sv ** 2 / len(Y); nu = lam.sum() / Y.shape[1]
    r = torch.sqrt(d * nu / ((1 - d) * lam + d * nu))
    return Y - ((Y @ Vt.T) * (1 - r)) @ Vt


def batch_transform_wknn(Ps, F, wccn, K=4):
    """The lean test-time adaptation of the best ensemble (TTA_LOG T5, T23, T39; AUG_LOG), inside one predict() batch,
    label-free and stateless: each member's output is divided by its batch mean norm and every trial is centred on the
    mean of its K most similar batch-mates in the within-person-whitened fingerprint space (cosine, batch-centred),
    keeping the batch mean; the members are summed, then ONE uncentred Ledoit-Wolf shrink with d = D / (D + B).
    Ps: list of member outputs (B, D); F: eeg_fingerprint of the batch; wccn: fit_fingerprint_wccn."""
    Z = ((F - wccn['m0']) / wccn['sd']) @ wccn['T']; Z = Z - Z.mean(0)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12); S = Z @ Z.T; S.fill_diagonal_(-9)
    W = torch.zeros_like(S).scatter_(1, S.topk(min(K, len(S) - 1), 1).indices, 1.) if len(S) > 1 else None
    out = 0
    for P in Ps:
        P = P.float() / P.float().norm(dim=1).mean()
        out = out + (P - W @ P / W.sum(1, keepdim=True) + P.mean(0, keepdim=True) if W is not None else P)
    return lw_shrink(out, out.shape[1] / (out.shape[1] + len(out))) if len(out) > 1 else out


@torch.no_grad()
def score_modules(model, eeg_projector, img_projector, cfg, subject_ids, device, batch_size,
                  num_workers, fraction=1.0) -> dict:
    """The grader pass on live modules; train.py calls it mid-run with fraction < 1."""
    from train import run_eeg_backbone

    native_n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
    gallery_normed = None
    n_top1 = n_top5 = n_epochs = 0

    for slot, subject_id in enumerate(subject_ids):
        dataset = build_test_dataset(cfg, [subject_id])
        if gallery_normed is None:
            # One subject builds the gallery; every subject shares the same 200 targets.
            n_objects = dataset.num_objects
            targets = torch.tensor(dataset.image_features[:, 0, :dataset.target_dims[0]], dtype=torch.float32, device=device)
            gallery_normed = _normalize(img_projector(targets).float().cpu().numpy())
            sums = np.zeros((len(subject_ids), n_objects, gallery_normed.shape[1]), dtype=np.float64)
            counts = np.zeros((len(subject_ids), n_objects), dtype=np.int64)
        if fraction < 1.0:
            # Seeded per subject: the same trials every call and every run, so estimates from
            # different epochs and runs are comparable.
            keep = np.random.default_rng(subject_id).choice(
                len(dataset), max(1, round(fraction * len(dataset))), replace=False)
            dataset = Subset(dataset, np.sort(keep).tolist())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for eeg, _, _, subject_batch, object_idx, _, _ in loader:
            eeg = eeg.to(device, non_blocking=True)
            if eeg.shape[-1] != native_n_times:
                # NOTE: stretches the served -0.2-0.8 s epoch onto a model trained on a
                # different window. Time alignment is NOT corrected -- see README.
                eeg = torch.nn.functional.interpolate(
                    eeg, size=native_n_times, mode="linear", align_corners=False
                )
            # Codabench calls predict(X) with no subject ids. An out-of-range id is how ATM's
            # SubjectEmbedding reaches its shared fallback token, i.e. what a submission sees.
            unknown = torch.full_like(subject_batch, 10**6)
            backbone = run_eeg_backbone(model, cfg, eeg, unknown.to(device))
            preds = eeg_projector(backbone).float().cpu().numpy()
            labels = object_idx.numpy()

            top1, top5 = topk_hits(preds, labels, gallery_normed)
            n_top1 += int(top1.sum())
            n_top5 += int(top5.sum())
            n_epochs += len(labels)

            np.add.at(sums[slot], labels, preds)
            np.add.at(counts[slot], labels, 1)
        if fraction == 1.0:
            print(f"subject {subject_id}: {n_epochs} epochs scored", flush=True)
        del dataset, loader

    seen = counts > 0
    subject_preds = (sums[seen] / counts[seen][:, None]).astype(np.float32)
    subject_labels = np.broadcast_to(np.arange(n_objects), counts.shape)[seen]

    instance_counts = counts.sum(axis=0)
    instance_seen = instance_counts > 0
    instance_preds = (
        sums.sum(axis=0)[instance_seen] / instance_counts[instance_seen][:, None]
    ).astype(np.float32)
    instance_labels = np.arange(n_objects)[instance_seen]

    return {
        "n_gallery": int(len(gallery_normed)),
        "per_epoch": {
            "top1": n_top1 / n_epochs,
            "top5": n_top5 / n_epochs,
            "n_queries": n_epochs,
        },
        "subject_agg": topk_acc(subject_preds, subject_labels, gallery_normed),
        "subject_agg_by_subject": {
            str(sid): topk_acc(sums[i][seen[i]] / counts[i][seen[i]][:, None],
                               np.arange(n_objects)[seen[i]], gallery_normed)
            for i, sid in enumerate(subject_ids) if seen[i].any()
        },
        "instance_agg": topk_acc(instance_preds, instance_labels, gallery_normed),
    }


def self_test() -> None:
    rng = np.random.default_rng(0)
    gallery = _normalize(rng.normal(size=(200, 32)))

    # A prediction equal to its own target must rank first.
    labels = np.arange(200)
    assert topk_acc(gallery.copy(), labels, gallery)["top1"] == 1.0

    # Pure noise sits near chance: 1/200 top-1, 5/200 top-5.
    noise = rng.normal(size=(20000, 32))
    chance = topk_acc(noise, rng.integers(0, 200, 20000), gallery)
    assert abs(chance["top1"] - 0.005) < 0.004, chance
    assert abs(chance["top5"] - 0.025) < 0.008, chance

    # Averaging repetitions must recover the target that single noisy epochs miss.
    signal = np.repeat(gallery, 40, axis=0)
    epochs = signal + rng.normal(scale=6.0, size=signal.shape)
    epoch_labels = np.repeat(labels, 40)
    per_epoch = topk_acc(epochs, epoch_labels, gallery)
    sums = np.zeros_like(gallery)
    np.add.at(sums, epoch_labels, epochs)
    averaged = topk_acc(sums / 40, labels, gallery)
    assert averaged["top5"] > 4 * per_epoch["top5"], (per_epoch, averaged)

    # batch_transform_wknn: finite for every batch size; a batch of 1 is the member's norm-scaled output
    xs, P = torch.randn(64, 32, 120), torch.randn(64, 16)
    wccn = {'m0': torch.zeros(656), 'sd': torch.ones(656), 'T': torch.eye(656)}
    for B in (1, 2, 64):
        y = batch_transform_wknn([P[:B], 2 * P[:B]], eeg_fingerprint(xs[:B]), wccn)
        assert torch.isfinite(y).all() and y.shape == (B, 16)
    assert torch.allclose(batch_transform_wknn([P[:1]], eeg_fingerprint(xs[:1]), wccn), P[:1] / P[:1].norm())
    print(f"self-test ok (per-epoch top5 {per_epoch['top5']:.3f} -> agg {averaged['top5']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", nargs="?")
    parser.add_argument("--test_subject_ids", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--checkpoint_name", default="checkpoint_test_best.pth",
                        help="e.g. checkpoint_sealed_best.pth; scores land in track1_score_sealed.json")
    parser.add_argument(
        "--eeg_data_dir",
        default=OFFICIAL_EEG_DIR,
        help="EEG served to the model; defaults to the official 120 Hz grader input. "
        "Pass 'native' to score on whatever the run trained on.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.checkpoint_dir is None:
        parser.error("checkpoint_dir is required unless --self_test")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.checkpoint_dir, "train_config.json")) as handle:
        cfg = SimpleNamespace(**json.load(handle))
    checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, args.checkpoint_name),
        map_location=device,
        weights_only=False,
    )
    if args.eeg_data_dir != "native":
        cfg.eeg_data_dir = args.eeg_data_dir
    subject_ids = args.test_subject_ids or cfg.test_subject_ids

    report = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "test_subject_ids": list(subject_ids),
        "eeg_data_dir": cfg.eeg_data_dir,
        "official_input": args.eeg_data_dir != "native",
        "leaderboard_valid": cfg.projector in ("direct", "subspace", "subspace_c"),  # both rank raw outputs in DINOv2 space
        **score(cfg, checkpoint, subject_ids, device, args.batch_size, args.num_workers),
    }
    if not report["leaderboard_valid"]:
        report["warning"] = (
            f"projector={cfg.projector!r}: scored in the model's own alignment space. "
            "The Codabench grader ranks raw 1536-D predictions against raw DINOv2 candidates "
            "and cannot apply a learned image projector, so per_epoch is NOT a leaderboard "
            "number for this checkpoint. Retrain with --projector direct --feature_dim 1536."
        )

    # Official and native scores are different measurements; never let one clobber the other.
    name = "track1_score.json" if report["official_input"] else "track1_score_native.json"
    tag = args.checkpoint_name.removeprefix("checkpoint_").removesuffix("_best.pth")
    if tag != "test":
        name = name.replace(".json", f"_{tag}.json")
        report["checkpoint_name"] = args.checkpoint_name
    with open(os.path.join(args.checkpoint_dir, name), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
