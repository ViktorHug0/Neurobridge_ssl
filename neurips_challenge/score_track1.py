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

# `module.dataset` and `train` are imported lazily inside the functions that need them, so the
# metric math above can be imported from the challenge venv by verify_grader.py.


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


def batch_transform(P, a=0.5, k=2, b=0.5, detach_dir=False):
    """Stateless post-processing of one predict() batch (ROUTING_LOG.md E5-E25): subtract a * the batch mean and
    shrink the centred batch's top-k principal directions by b. Defaults validated for the sealed condition (shuffled
    batches mixing participants): ensemble 33.63 -> 38.62 in grader_sim.py --shuffle --k 2 (official metric, 20/20).
    detach_dir: no gradient through the SVD (training)."""
    mu = P.mean(0, keepdim=True); C = P - mu
    if k and len(P) > k:
        V = torch.linalg.svd(C.float(), full_matrices=False)[2][:k].to(C.dtype)
        V = V.detach() if detach_dir else V
        C = C - b * (C @ V.T) @ V
    return C + (1 - a) * mu


FP_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30))  # rfft bins of a 120-sample trial at 120 Hz = Hz


def eeg_fingerprint(X, bands=FP_BANDS):
    """(B, 32, T) raw EEG -> (B, 656) label-free per-trial features: log channel variance, channel correlations
    (upper triangle), log band power per channel (= neighbour_lda.fingerprint, in torch)."""
    X = X.double(); b = X - X.mean(-1, keepdim=True)
    v = (b ** 2).mean(-1) + 1e-6
    c = torch.einsum('nct,ndt->ncd', b, b) / b.shape[-1] / torch.sqrt(v[:, :, None] * v[:, None, :])
    iu = torch.triu_indices(X.shape[1], X.shape[1], 1)
    P = torch.fft.rfft(b, dim=-1).abs() ** 2
    bp = torch.cat([torch.log(P[..., lo:hi].mean(-1) + 1e-6) for lo, hi in bands], 1)
    return torch.cat([torch.log(v), c[:, iu[0], iu[1]], bp], 1).float()


def batch_transform_fp(P, F, lda, p=6, c=1.0, a=0.5, k=3, b=0.5):
    """batch_transform + label-free subject routing (ROUTING_LOG E32-E37): project the trials' fingerprints F on
    an offline subject-discriminant (LDA) space `lda` = dict(m0, sd, W) fitted on training subjects' raw EEG,
    z-score within the batch, and regress the top-p within-batch fingerprint PCs out of the centred outputs (c),
    then shrink the top-k output directions by b and keep (1 - a) of the batch mean. Stateless. Defaults = E42
    (LDA on Alljoined + THINGS-EEG2 subjects): s11 exact 31.02 -> 36.95 vs 35.52 for batch_transform(k=2)."""
    mu = P.mean(0, keepdim=True); C = P - mu
    p = min(p, (len(P) - 1) // 8)  # small batches: p >= B-1 would regress out ALL trial variation (C -> 0)
    if p > 0:
        Fp = ((F - lda['m0']) / lda['sd']) @ lda['W']
        Fp = (Fp - Fp.mean(0)) / (Fp.std(0) + 1e-6)
        Z = torch.linalg.svd(Fp, full_matrices=False)[0][:, :p]
        C = C - c * Z @ (Z.T @ C)
    return batch_transform(C, a=0.0, k=k, b=b) + (1 - a) * mu  # C is centred: a=0 keeps it as is


RICH = ('fine', 'calpha', 'cbeta', 'cgamma', 'pgamma')  # TTA_LOG T9: identity groups appended to eeg_fingerprint


def identity_features(X, groups=RICH):
    """(B, 32, 120) raw EEG -> eeg_fingerprint (656) + the group_features.py identity groups, in torch (same maths):
    fine = log power in 2-Hz bins 1-45 Hz, c<band> = channel correlations of the band-passed trial, pgamma = log power
    30-58 Hz in 7-Hz bins."""
    out = [eeg_fingerprint(X)]
    b = X.double(); b = b - b.mean(-1, keepdim=True)
    Fq = torch.fft.rfft(b, dim=-1); P = Fq.abs() ** 2; f = torch.fft.rfftfreq(b.shape[-1], 1 / 120).to(b.device)
    iu = torch.triu_indices(b.shape[1], b.shape[1], 1)

    def corr(lo, hi):
        x = torch.fft.irfft(Fq * ((f >= lo) & (f < hi)), n=b.shape[-1], dim=-1)
        x = x - x.mean(-1, keepdim=True); v = (x ** 2).mean(-1) + 1e-9
        c = torch.einsum('nct,ndt->ncd', x, x) / x.shape[-1] / torch.sqrt(v[:, :, None] * v[:, None, :])
        return c[:, iu[0], iu[1]]
    g = {'fine': lambda: torch.log(torch.stack([P[..., lo:lo + 2].mean(-1) for lo in range(1, 45, 2)], -1) + 1e-6).flatten(1),
         'calpha': lambda: corr(8, 13), 'cbeta': lambda: corr(13, 30), 'cgamma': lambda: corr(30, 58),
         'pgamma': lambda: torch.log(torch.stack([P[..., lo:lo + 7].mean(-1) for lo in range(30, 58, 7)], -1) + 1e-6).flatten(1)}
    return torch.cat(out + [g[k]().float() for k in groups], 1)


FP_AB = slice(592, 656)  # eeg_fingerprint columns of the alpha + beta log power per channel (TTA_LOG T38: 64 dims)


def fit_fingerprint_wccn(data_dir, subjects, shrink=0.3, groups=(), cols=slice(None)):
    """offline (TTA_LOG T4-T6): standardisation + within-person whitening (WCCN, all fingerprint dims) from
    `subjects`' training-image fingerprint caches (fit_fingerprint_lda builds them; groups: + group_features.py
    caches, same order as identity_features). Same hardware only (Alljoined). cols: a subset of eeg_fingerprint's
    columns (FP_AB); the test-time fingerprint must then be sliced the same way."""
    one = lambda g, s: np.load(f'{data_dir}/.cache/gf_{g}_train_s{s:02d}.npy')
    Fs = [np.hstack([np.load(f'{data_dir}/.cache/fp_train_s{s:02d}_n3000_seed0_{"-".join(f"{lo}_{hi}" for lo, hi in FP_BANDS)}.npy')[:, cols]]
                    + [one(g, s) for g in groups]) for s in subjects]
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


def batch_transform_wknn(Ps, F, wccn, K=4, c=0.7, k=3, b=0.5, a=0.5, d=None):
    """label-free person grouping inside one predict() batch (TTA_LOG T5): each trial is centred on the mean of its K
    most similar batch-mates in the within-person-whitened fingerprint space (cosine, batch-centred), with strength c,
    then the batch's top-k output directions are shrunk by b and (1 - a) of the batch mean is kept. Ps: list of member
    outputs (B, D) (shared neighbours); returns the summed transformed output. cv4 ensemble 40.05 -> 42.15 (vs
    batch_transform_fp). Stateless; batches of 1 fall back to batch_transform.
    d set: ONE uncentred Ledoit-Wolf shrink of the SUMMED person-centred batch (mean kept) replaces k / b / a (TTA_LOG
    T23, T32: the shrink belongs on the ensemble output; d .93 c .85, REVE-d6-KD + SQF-KD + rmix-reg 44.12).
    d = 'auto': d = D / (D + B), i.e. shrink toward the mean batch eigenvalue (T39, no knob; use with c = 1)."""
    Z = ((F - wccn['m0']) / wccn['sd']) @ wccn['T']; Z = Z - Z.mean(0)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12); S = Z @ Z.T; S.fill_diagonal_(-9)
    out = 0
    for P in Ps:
        P = P.float() / P.float().norm(dim=1).mean()
        if len(P) < 2:
            out = out + batch_transform(P, a=a, k=k, b=b); continue
        W = torch.zeros_like(S).scatter_(1, S.topk(min(K, len(S) - 1), 1).indices, 1.); W.fill_diagonal_(0)
        mu = P.mean(0, keepdim=True)
        C = P - mu - c * (W @ P / W.sum(1, keepdim=True) - mu)
        out = out + (C + mu if d else batch_transform(C, a=0.0, k=k, b=b) + (1 - a) * mu)
    if d == 'auto':
        d = out.shape[1] / (out.shape[1] + len(out))
    return lw_shrink(out, d) if d and len(out) > 1 else out


def fit_fingerprint_lda(data_dir, subjects, n_per=3000, seed=0, bands=FP_BANDS):
    """offline: the subject-discriminant projection of eeg_fingerprint over `subjects`' training-image trials."""
    from neurips_challenge.neighbour_lda import lda
    Fs = []
    for s in subjects:  # per-subject fingerprints cached once: random reads of train.npy saturate the NAS
        cache = f'{data_dir}/.cache/fp_train_s{s:02d}_n{n_per}_seed{seed}_{"-".join(f"{lo}_{hi}" for lo, hi in bands)}.npy'
        if not os.path.exists(cache):
            x = np.load(f'{data_dir}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)
            idx = np.sort(np.random.default_rng([seed, s]).choice(len(x), 4 * n_per, replace=False))
            xs = np.asarray(x[idx], np.float32); xs = xs[np.abs(xs).sum((1, 2)) > 0][:n_per]  # skip unseen-image padding
            np.save(cache + '.tmp.npy', eeg_fingerprint(torch.tensor(xs), bands).numpy()); os.replace(cache + '.tmp.npy', cache)
        Fs.append(np.load(cache))
    Ftr = np.concatenate(Fs).astype(np.float64); m0, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    mu, W = lda((Ftr - m0) / sd, np.repeat(subjects, [len(f) for f in Fs]), dim=min(32, len(subjects) - 1))
    return {'bands': bands, 'm0': torch.tensor(m0 + mu * sd, dtype=torch.float32), 'sd': torch.tensor(sd, dtype=torch.float32),
            'W': torch.tensor(W, dtype=torch.float32)}


@torch.no_grad()
def score_modules(model, eeg_projector, img_projector, cfg, subject_ids, device, batch_size,
                  num_workers, fraction=1.0, return_preds=False) -> dict:
    """The grader pass on live modules; train.py calls it mid-run with fraction < 1.
    return_preds adds the per-(subject, image) mean predictions, the gallery and the mean per-epoch
    prediction norm (what an ensemble needs: the grader's repeat averaging is linear)."""
    from train import run_eeg_backbone

    native_n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
    gallery_normed = None
    n_top1 = n_top5 = n_epochs = 0
    norm_sum = 0.0

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
        transform = getattr(cfg, 'batch_transform', False)
        if transform:  # grader-like batches: 64 trials in random (≈ recording) order, then the batch transform
            dataset = Subset(dataset, np.random.default_rng(subject_id + 1).permutation(len(dataset)).tolist())
        loader = DataLoader(
            dataset, batch_size=64 if transform else batch_size, shuffle=False, num_workers=num_workers
        )
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
            preds = eeg_projector(backbone).float()
            preds = (batch_transform(preds) if transform else preds).cpu().numpy()
            labels = object_idx.numpy()

            top1, top5 = topk_hits(preds, labels, gallery_normed)
            n_top1 += int(top1.sum())
            n_top5 += int(top5.sum())
            n_epochs += len(labels)

            np.add.at(sums[slot], labels, preds)
            norm_sum += float(np.linalg.norm(preds, axis=1).sum())
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

    extra = {}
    if return_preds:
        extra = {"subject_mean_preds": {str(sid): (sums[i] / np.maximum(counts[i], 1)[:, None]).astype(np.float32)
                                        for i, sid in enumerate(subject_ids)},
                 "gallery_normed": gallery_normed.astype(np.float32), "mean_norm": norm_sum / n_epochs}
    return {
        **extra,
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

    # batch_transform: identity at a=b=0, and differentiable (train.py --batch_transform trains through it)
    x = torch.randn(40, 16, requires_grad=True)
    y = batch_transform(x, a=0.0, b=0.0, detach_dir=True)
    assert y.requires_grad and torch.allclose(y, x, atol=1e-5)
    # batch_transform_fp: with c=0 it is batch_transform(k=3)
    xs, P = torch.randn(64, 32, 120), torch.randn(64, 16)
    lda = {'m0': torch.zeros(656), 'sd': torch.ones(656), 'W': torch.randn(656, 8)}
    assert torch.allclose(batch_transform_fp(P, eeg_fingerprint(xs), lda, c=0.0), batch_transform(P, k=3), atol=1e-5)
    for B in (1, 2, 8, 16):  # tiny batches stay finite and keep per-trial differences
        y = batch_transform_fp(P[:B], eeg_fingerprint(xs[:B]), lda)
        assert torch.isfinite(y).all() and (B == 1 or (y - y.mean(0)).abs().sum() > 0)
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
                        help="e.g. checkpoint_testsel_best.pth (--save_testsel_checkpoint); "
                        "scores land in track1_score_testsel.json instead of track1_score.json")
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
