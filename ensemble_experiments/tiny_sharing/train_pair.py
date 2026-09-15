"""Train and evaluate one paired condition on one inner screening fold.

Phase B of tiny_sharing_plan.md: 8 training subjects, the inner subject held out
entirely from training and used only for checkpoint selection and diagnostics.
All selection is on inner validation; no outer EEG or scores are loaded here.
"""

import argparse
import hashlib
import json
import os
import time
import functools
import fcntl

import numpy as np
import torch
import torch.nn.functional as F

from module.loss import ContrastiveLoss
from ensemble_experiments.tiny_sharing import data as fold_data
from ensemble_experiments.tiny_sharing.models import (
    CONFIG_IDS, TIE_GROUPS, PairedModel, parameter_report, tied_tensor_names,
)

IMAGES_PER_BATCH = 128          # x 8 subjects = effective batch 1024
EPOCHS = 40
EARLY_WINDOW = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.5
TEMPERATURE = 0.07
GRAD_DIAG_EVERY = 25


def source_hash():
    """Any edit to the experiment code invalidates a resume (plan sec. 9)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            h.update(open(os.path.join(here, name), "rb").read())
    return h.hexdigest()[:16]


def atomic_save(value, path):
    torch.save(value, path + '.tmp')
    os.replace(path + '.tmp', path)


def atomic_json(value, path):
    with open(path + '.tmp', 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(path + '.tmp', path)


def locked_run(fn):
    @functools.wraps(fn)
    def wrapped(config, fold, out_dir, *args, **kwargs):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, '.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fn(config, fold, out_dir, *args, **kwargs)
    return wrapped


def export_artifacts(artifacts, out_dir):
    for name, value in artifacts.items():
        path = os.path.join(out_dir, name)
        if name.endswith('.json'):
            atomic_json(value, path)
        elif name.endswith('.pth'):
            atomic_save(value, path)
        else:
            with open(path + '.tmp', 'wb') as handle:
                np.savez_compressed(handle, **value)
            os.replace(path + '.tmp', path)


def make_criterion(device):
    # fixed temperature, softplus, EEG unnormalized in training, image normalized
    return ContrastiveLoss(
        init_temperature=TEMPERATURE, alpha=1.0, beta=1.0,
        eeg_l2norm=False, img_l2norm=True, text_l2norm=False,
        learnable=False, is_softplus=True,
    ).to(device)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def embed_validation(model, fold, chunk=200):
    model.eval()
    ea, eb = [], []
    with torch.no_grad():
        for start in range(0, fold.val_eeg.shape[0], chunk):
            block = fold.val_eeg[start:start + chunk]
            za, zb = model(block, block)
            ea.append(za)
            eb.append(zb)
        za, zb = torch.cat(ea), torch.cat(eb)
        ia, ib = model.project_images(fold.val_img_a.float(), fold.val_img_b.float())
    return (F.normalize(za, dim=1), F.normalize(zb, dim=1),
            F.normalize(ia, dim=1), F.normalize(ib, dim=1))


def validation_loss(model, fold, criterion):
    """Mean over the two branches of the grouped multi-positive loss."""
    model.eval()
    tot_a = tot_b = 0.0
    rows = 0
    with torch.no_grad():
        for chunk in fold.loss_chunks:
            idx = torch.from_numpy(chunk).to(fold.device)
            eeg = fold.val_eeg[idx]
            za, zb = model(eeg, eeg)
            ia, ib = model.project_images(
                fold.val_img_a[idx].float(), fold.val_img_b[idx].float()
            )
            mask = torch.eye(len(chunk), dtype=torch.bool, device=fold.device)
            la = criterion.multi_positive_pair_loss(za, ia, mask)
            lb = criterion.multi_positive_pair_loss(zb, ib, mask)
            tot_a += float(la) * len(chunk)
            tot_b += float(lb) * len(chunk)
            rows += len(chunk)
    return tot_a / rows, tot_b / rows


def _rowz(scores):
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True, correction=0).clamp_min(1e-8)


def panel_scores(model, fold):
    """Per-query 200-way score rows for both branches, every query exactly once."""
    za, zb, ia, ib = embed_validation(model, fold)
    out_a, out_b, correct = [], [], []
    for block, cand in fold.panels:
        q = torch.from_numpy(block).to(fold.device)
        c = torch.from_numpy(cand).to(fold.device)
        out_a.append(za[q] @ ia[c].T)
        out_b.append(zb[q] @ ib[c].T)
        correct.append(torch.arange(len(block), device=fold.device))
    return (torch.cat(out_a), torch.cat(out_b), torch.cat(correct),
            za, zb)


def topk_hits(scores, correct, k):
    return (scores.topk(k, dim=1).indices == correct[:, None]).any(1)


def _cka(x, y):
    """Linear CKA between two activation matrices (centered similarity)."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    num = (x.T @ y).pow(2).sum()
    den = (x.T @ x).norm() * (y.T @ y).norm()
    return float(num / den.clamp_min(1e-12))


def evaluate(model, fold):
    """The full measurement block of plan sec. 8, on inner validation."""
    sa, sb, correct, za, zb = panel_scores(model, fold)
    za_z, zb_z = _rowz(sa), _rowz(sb)
    fused = 0.5 * (za_z + zb_z)

    hits = {
        "ts": topk_hits(sa, correct, 1),
        "atm": topk_hits(sb, correct, 1),
        "fused": topk_hits(fused, correct, 1),
    }
    n = len(correct)
    res = {
        "n_queries": n,
        "top1_ts": float(hits["ts"].float().mean()),
        "top1_atm": float(hits["atm"].float().mean()),
        "top1_fused": float(hits["fused"].float().mean()),
        "top5_ts": float(topk_hits(sa, correct, 5).float().mean()),
        "top5_atm": float(topk_hits(sb, correct, 5).float().mean()),
        "top5_fused": float(topk_hits(fused, correct, 5).float().mean()),
    }
    res["gain_over_mean_branch"] = res["top1_fused"] - 0.5 * (res["top1_ts"] + res["top1_atm"])
    res["gain_over_best_branch"] = res["top1_fused"] - max(res["top1_ts"], res["top1_atm"])

    a, b, f = hits["ts"], hits["atm"], hits["fused"]
    res["overlap"] = {
        "both": int((a & b).sum()), "ts_only": int((a & ~b).sum()),
        "atm_only": int((~a & b).sum()), "neither": int((~a & ~b).sum()),
    }
    res["oracle_top1"] = float((a | b).float().mean())
    res["prediction_agreement"] = float(
        (sa.argmax(1) == sb.argmax(1)).float().mean()
    )
    res["fusion_rescues"] = int((~a & ~b & f).sum())
    res["fusion_losses"] = int(((a | b) & ~f).sum())

    # correct-vs-hardest-distractor margins under the same row-z convention
    for name, z in (("ts", za_z), ("atm", zb_z), ("fused", fused)):
        gold = z.gather(1, correct[:, None]).squeeze(1)
        masked = z.scatter(1, correct[:, None], float("-inf"))
        res[f"margin_{name}"] = float((gold - masked.max(1).values).mean())
    res["distractor_complementarity_bonus"] = (
        res["margin_fused"] - 0.5 * (res["margin_ts"] + res["margin_atm"])
    )

    res["cka_readout"] = _cka(za, zb)
    res["scores"] = (sa.cpu().numpy(), sb.cpu().numpy(), correct.cpu().numpy())
    return res


# --------------------------------------------------------------------------- #
# gradient diagnostics at candidate shared blocks
# --------------------------------------------------------------------------- #

def _diag_blocks(model):
    """Homologous candidate-shared blocks on each branch, per topology."""
    if model.config == "C0":
        # native pair: the two temporal convolutions are homologous in role only
        # (10 vs 12 filters, raw time vs learned latent axis), so no common
        # coordinate system exists and no cosine is reported below
        return {"T": ([model.ts.tsconv[0]], [model.atm.enc_eeg[0].tsconv[0]])}
    if model.config in TIE_GROUPS:
        return {
            "T": ([model.stem_a.temporal], [model.stem_b.temporal]),
            "S": ([model.tail_a.spatial, model.tail_a.projection],
                  [model.tail_b.spatial, model.tail_b.projection]),
            "R": ([model.tail_a.readout.lin1, model.tail_a.readout.lin2],
                  [model.tail_b.readout.lin1, model.tail_b.readout.lin2]),
        }
    return {"T": ([model.stem_a.temporal], [model.stem_b.temporal])}


def grad_diagnostics(model, loss_a, loss_b):
    out = {}
    for name, (mods_a, mods_b) in _diag_blocks(model).items():
        pa = [p for m in mods_a for p in m.parameters()]
        pb = [p for m in mods_b for p in m.parameters()]
        ga = torch.autograd.grad(loss_a, pa, retain_graph=True, allow_unused=True)
        gb = torch.autograd.grad(loss_b, pb, retain_graph=True, allow_unused=True)
        fa = torch.cat([g.flatten() for g in ga if g is not None]) if any(
            g is not None for g in ga) else None
        fb = torch.cat([g.flatten() for g in gb if g is not None]) if any(
            g is not None for g in gb) else None
        comparable = (
            fa is not None and fb is not None and fa.numel() == fb.numel()
        )
        out[name] = {
            "cosine": float(F.cosine_similarity(fa, fb, dim=0)) if comparable else None,
            "norm_ts": float(fa.norm()) if fa is not None else None,
            "norm_atm": float(fb.norm()) if fb is not None else None,
        }
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@locked_run
def run(config, fold, out_dir, epochs=EPOCHS, device="cuda:0", max_steps=None,
        stop_after_epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    torch.manual_seed(3300)
    model = PairedModel(config).to(dev)
    criterion = make_criterion(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    counts = parameter_report(model)
    manifest = {
        "config": config, "slot": fold.slot, "outer_subject": fold.outer,
        "inner_subject": fold.inner, "train_subjects": fold.train_subjects,
        "tie_groups": list(TIE_GROUPS.get(config, ())),
        "tied_tensor_names": tied_tensor_names(model),
        "held_concepts": int(len(fold.held)), "concept_seed": fold_data.CONCEPT_SEED,
        "gallery_seed": fold_data.GALLERY_SEED,
        "image_target_ts": fold_data.TARGET_A, "image_target_atm": fold_data.TARGET_B,
        "epochs": epochs, "images_per_batch": IMAGES_PER_BATCH,
        "max_steps": max_steps,
        "samples_per_image": fold.samples_per_image,
        "effective_batch": IMAGES_PER_BATCH * fold.samples_per_image,
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "scheduler": None, "temperature": TEMPERATURE, "temperature_learnable": False,
        "mixup": {"type": "pairwise", "alpha": MIXUP_ALPHA, "prob": 1.0},
        "dtype": "fp32", "master_seed": 3300, "atm_branch_seed": 4300,
        "subject_token_policy": "unknown/shared token in every batch and phase",
        "bn": {"eps": 1e-5, "momentum": 0.1,
               "c7_pooled": "(va+vb)/2 + (ma-mb)^2/4, equal branch weight"},
        "source_hash": source_hash(),
        "parameters": counts,
        "optimizer_param_tensors": len(params),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            previous = json.load(handle)
        if previous != manifest:
            raise RuntimeError(f"Incompatible existing manifest: {out_dir}; use a new result root")
    atomic_json(manifest, manifest_path)

    ckpt_path = os.path.join(out_dir, "last.pth")
    history, start_epoch = [], 0
    best = {"mean": (1e9, -1), "ts": (1e9, -1), "atm": (1e9, -1), "mean20": (1e9, -1)}
    artifacts = {}
    step_times, total_train, peak_gpu = [], 0.0, 0
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if state.get("source_hash") == manifest["source_hash"]:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            if state.get("branch_rng"):
                model._gen(0, dev)   # materialize the per-branch streams
                for g, saved in zip(model._gens, state["branch_rng"]):
                    g.set_state(saved)
            history, best, start_epoch = state["history"], state["best"], state["epoch"]
            artifacts = state['artifacts']
            export_artifacts(artifacts, out_dir)
            step_times, total_train = state['step_times'], state['total_train']
            peak_gpu = state['peak_gpu']
            print(f"[{config}] resumed at epoch {start_epoch}", flush=True)
        else:
            raise RuntimeError(f"Source changed; refusing stale resume: {out_dir}")

    steps = fold.steps_per_epoch(IMAGES_PER_BATCH)
    if max_steps is not None:
        steps = min(steps, max_steps)
    torch.cuda.reset_peak_memory_stats(dev)

    for epoch in range(start_epoch, epochs):
        model.train()
        order = fold.epoch_key_order(epoch)
        epoch_start = time.time()
        run_a = run_b = 0.0
        diags = []
        for step in range(steps):
            keys = order[step * IMAGES_PER_BATCH:(step + 1) * IMAGES_PER_BATCH]
            eeg, ta, tb, mask = fold.batch(keys, epoch, step, MIXUP_ALPHA)
            # diagnostic steps are excluded from the reported step time, so the
            # diagnostics' overhead is never charged to the main timing
            diag_step = step % GRAD_DIAG_EVERY == 0
            torch.cuda.synchronize(dev)
            t0 = time.time()
            za, zb = model(eeg, eeg)
            ia, ib = model.project_images(ta, tb)
            la = criterion.multi_positive_pair_loss(za, ia, mask)
            lb = criterion.multi_positive_pair_loss(zb, ib, mask)
            if not torch.isfinite(la + lb):
                raise FloatingPointError(f'{config}: nonfinite training loss')
            if diag_step:
                diags.append(grad_diagnostics(model, la, lb))
            optimizer.zero_grad(set_to_none=True)
            ((la + lb) / 2).backward()
            optimizer.step()
            torch.cuda.synchronize(dev)
            if not diag_step:
                step_times.append(time.time() - t0)
            run_a += float(la)
            run_b += float(lb)
        train_time = time.time() - epoch_start
        total_train += train_time

        va, vb = validation_loss(model, fold, criterion)
        acc = evaluate(model, fold)
        if not np.isfinite([va, vb]).all():
            raise FloatingPointError(f'{config}: nonfinite validation loss')
        row = {
            "epoch": epoch, "train_loss_ts": run_a / steps, "train_loss_atm": run_b / steps,
            "val_loss_ts": va, "val_loss_atm": vb, "val_loss_mean": 0.5 * (va + vb),
            "epoch_seconds": train_time,
            "top1_ts": acc["top1_ts"], "top1_atm": acc["top1_atm"],
            "top1_fused": acc["top1_fused"], "oracle_top1": acc["oracle_top1"],
            "grad_diagnostics": {
                block: {key: float(np.mean(values)) if values else None
                        for key in diags[0][block]
                        for values in [[d[block][key] for d in diags if d[block][key] is not None]]}
                for block in diags[0]
            } if diags else None,
            "grad_diagnostic_samples": len(diags),
        }
        history.append(row)
        print(
            f"[{config} slot{fold.slot}] ep{epoch:02d} "
            f"val {row['val_loss_mean']:.4f} "
            f"top1 ts {acc['top1_ts']*100:.2f} atm {acc['top1_atm']*100:.2f} "
            f"fused {acc['top1_fused']*100:.2f} ({train_time:.1f}s)",
            flush=True,
        )

        for key, value in (("mean", row["val_loss_mean"]), ("ts", va), ("atm", vb)):
            if value >= best[key][0]:
                continue
            best[key] = (value, epoch)
            if key == "mean":
                artifacts['best_mean.pth'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                artifacts['scores_best_mean.npz'] = dict(
                    ts=acc['scores'][0], atm=acc['scores'][1], correct=acc['scores'][2])
                metrics = {k: v for k, v in acc.items() if k != "scores"}
                metrics["selected_epoch"] = epoch
                artifacts['metrics_best_mean.json'] = metrics
            else:
                # branch-best readout: each branch's own scores at its own best
                # epoch, so the independent-checkpoint fusion needs no extra run
                artifacts[f'scores_best_{key}.npz'] = dict(
                    scores=acc["scores"][0 if key == "ts" else 1],
                    correct=acc["scores"][2], epoch=epoch,
                )
        if epoch < EARLY_WINDOW and row["val_loss_mean"] < best["mean20"][0]:
            best["mean20"] = (row["val_loss_mean"], epoch)
            artifacts['metrics_best_mean_20.json'] = (
                {k: v for k, v in acc.items() if k != "scores"} | {"selected_epoch": epoch})

        peak_gpu = max(peak_gpu, int(torch.cuda.max_memory_allocated(dev)))
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "artifacts": artifacts, "step_times": step_times,
            "total_train": total_train, "peak_gpu": peak_gpu,
            "branch_rng": [g.get_state() for g in (model._gens or [])],
            "history": history, "best": best,
            "epoch": epoch + 1, "source_hash": manifest["source_hash"],
        }, ckpt_path)
        export_artifacts(artifacts, out_dir)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            return None

    warm = step_times[len(step_times) // 4:]
    summary = {
        **manifest,
        "history": history,
        "selected_epoch_common": best["mean"][1],
        "selected_epoch_common_within20": best["mean20"][1],
        "branch_best_epoch_ts": best["ts"][1],
        "branch_best_epoch_atm": best["atm"][1],
        "steps_per_epoch": steps,
        "total_steps": steps * epochs,
        "warm_step_seconds": float(np.median(warm)) if warm else None,
        "mean_epoch_seconds": total_train / max(epochs, 1),
        "total_train_seconds": total_train,
        "peak_gpu_bytes": peak_gpu,
    }
    atomic_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--configs", nargs="+", default=CONFIG_IDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--result-root", default=os.path.join(
        fold_data.REPO, "results/things_eeg/tiny_sharing/screen"))
    args = ap.parse_args()

    for slot in args.slots:
        print(f"=== loading slot {slot} ===", flush=True)
        fold = fold_data.FoldData(slot, torch.device(args.device))
        for config in args.configs:
            out = os.path.join(args.result_root, f"slot{slot}", config)
            if os.path.isfile(os.path.join(out, "summary.json")):
                with open(os.path.join(out, "summary.json")) as handle:
                    completed = json.load(handle)
                if completed['source_hash'] != source_hash() or completed['epochs'] != args.epochs:
                    raise RuntimeError(f'Incompatible completed run: {out}')
                print(f"[{config} slot{slot}] done; skipping", flush=True)
                continue
            run(config, fold, out, epochs=args.epochs, device=args.device)
        del fold
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
