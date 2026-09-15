#!/usr/bin/env python3
"""TTA diagnostic experiments: permutation null, bootstrap stability,
generalization split (rotation vs identity blend), PCA trajectory, and geodesic convergence."""
import argparse, json, os, sys
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import expm, logm
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.dataset import EEGPreImageDataset
from module.util import (
    apply_orthogonal_map, fit_soft_assignment_procrustes,
    sinkhorn_normalize, topk,
)
from train import build_eeg_encoder, build_projector, run_eeg_backbone, seed_everything

# ── tiny helpers ──────────────────────────────────────────────────────
def _to_bool(v, d=False):
    if v is None: return d
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() in {"1","true","yes"}
    return bool(v)

def _load_json(p):
    if not os.path.isfile(p): return {}
    with open(p) as f: return json.load(f)

def _norm(x, eps=1e-12):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)

def _cos(a, b): return (_norm(a) @ _norm(b).T).astype(np.float32)

def _csls(sim, k=10):
    n_q, n_c = sim.shape
    k = max(1, min(k, n_q, n_c))
    rx = np.partition(sim, n_c-k, axis=1)[:, -k:].mean(1, keepdims=True)
    ry = np.partition(sim, n_q-k, axis=0)[-k:, :].mean(0, keepdims=True)
    return (2*sim - rx - ry).astype(np.float32)

def _scores(q, c, csls_k=1):
    s = _cos(q, c)
    return _csls(s, csls_k) if csls_k > 0 else s

def _top1(scores):
    _, c1 = topk(scores, 1)
    return c1 / scores.shape[0]

# ── whitening ─────────────────────────────────────────────────────────
def _fit_whiten(feats, shrink=0.85, eps=1e-6):
    mu = feats.mean(0, keepdims=True)
    c = feats - mu
    cov = (c.T @ c / max(feats.shape[0]-1, 1)).astype(np.float32)
    d = cov.shape[0]
    cov = (1-shrink)*cov + shrink*(np.trace(cov)/d)*np.eye(d, dtype=np.float32) + eps*np.eye(d, dtype=np.float32)
    ev, U = np.linalg.eigh(cov)
    W = U @ np.diag(np.clip(ev, eps, None)**-0.5).astype(np.float32) @ U.T
    return mu.astype(np.float32), W.astype(np.float32)

def _apply_whiten(feats, mu, W):
    return _norm((feats - mu) @ W)

# ── full pipeline (returns R* and optionally per-step Rs) ─────────────
def _full_pipeline(E, V, p, track=False):
    """Run whiten→CSLS→Sinkhorn→Procrustes loop.
    Returns (transformed_E, R_cumulative [, list_of_per_step_R])."""
    mu, W = _fit_whiten(E, shrink=p.saw_shrink)
    E = _apply_whiten(E, mu, W)
    R = np.eye(E.shape[1], dtype=np.float32)
    Rs = [R.copy()] if track else None
    for _ in range(p.steps):
        s = _scores(E, V, p.csls_k)
        P = sinkhorn_normalize(s, tau=p.tau, num_iters=p.sink_iters)
        dR = fit_soft_assignment_procrustes(E, V, P, power=p.power)
        if dR is None: break
        E = apply_orthogonal_map(E, dR)
        R = (R @ dR).astype(np.float32)
        if track: Rs.append(R.copy())
    return E, R, mu, W, Rs

def _full_pipeline_simple(E, V, p):
    E_t, R, *_ = _full_pipeline(E, V, p)
    return E_t, R

# ── geodesic distance on SO(d) ────────────────────────────────────────
def _geodesic(A, B):
    """‖log(A^T B)‖_F / √2"""
    M = A.T @ B
    L = logm(M.astype(np.complex128))
    return float(np.linalg.norm(L.real)) / np.sqrt(2)


def _rotation_blend_identity_geodesic(R, alpha, L_log_precomp=None):
    """Geodesic on SO(d) from I toward R: R^α ≃ exp(α log R). alpha=0→I, alpha=1→R.

    Uses one matrix log per split via L_log_precomp when provided (recommended).
    Re-orthogonalises after exp() to compensate for drift / branch choices.
    """
    R = np.asarray(R, dtype=np.float64)
    d = R.shape[0]
    eps = 1e-10
    if alpha <= eps:
        return np.eye(d, dtype=np.float32)
    if alpha >= 1.0 - eps:
        return R.astype(np.float32)
    if L_log_precomp is None:
        L_log = np.real(logm(R.astype(np.complex128)))
    else:
        L_log = L_log_precomp
    Rd = np.real(expm(L_log.astype(np.complex128) * float(alpha))).astype(np.float64)
    U, _, Vt = np.linalg.svd(Rd)
    Z = U @ Vt
    if np.linalg.det(Z) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        Z = (U @ Vt).astype(np.float32)
    else:
        Z = Z.astype(np.float32)
    return Z

# ── data loading (mirrors existing scripts) ───────────────────────────
def _find_ckpt(src, sid):
    sfx = f"-sub-{int(sid):02d}"
    hits = [os.path.join(src, n) for n in os.listdir(src)
            if n.endswith(sfx) and os.path.isdir(os.path.join(src, n))]
    if not hits: raise FileNotFoundError(f"No ckpt for sub {sid} in {src}")
    return max(hits, key=os.path.getmtime)

def _encode(src, sid, device="cuda:0", batch_size=1024, num_workers=4):
    ckpt_dir = _find_ckpt(src, sid)
    tcfg = _load_json(os.path.join(ckpt_dir, "train_config.json"))
    ecfg = _load_json(os.path.join(ckpt_dir, "evaluate_config.json"))
    m = {**tcfg, **ecfg, "test_subject_id": int(sid), "device": device,
         "eval_batch_size": batch_size, "num_workers": num_workers}
    ea = SimpleNamespace(**m)
    avg = _to_bool(getattr(ea, "data_average", True), True)
    ds = EEGPreImageDataset(
        [int(sid)], getattr(ea, "eeg_data_dir", ""), getattr(ea, "selected_channels", []), getattr(ea, "time_window", [0, 250]),
        getattr(ea, "image_feature_dir", ""), getattr(ea, "text_feature_dir", ""),
        False, [], avg, False, None, False, False, False,
        _to_bool(getattr(ea,"frozen_eeg_prior",False)))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ckpt_dir,"checkpoint_test_best.pth"), map_location=dev)
    sp = ds.num_sample_points; cn = ds.channels_num
    ifd = ds.image_features.shape[-1]
    bfd = getattr(ea, "eeg_backbone_dim", 0) or getattr(ea, "backbone_feature_dim", 0) or ifd
    model = build_eeg_encoder(ea, bfd, sp, cn).to(dev)
    ip = build_projector(ea.projector, ifd, ea.feature_dim).to(dev)
    ep = build_projector(ea.projector, bfd, ea.feature_dim).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    ep.load_state_dict(ck["eeg_projector_state_dict"])
    ip.load_state_dict(ck["img_projector_state_dict"])
    model.eval(); ep.eval(); ip.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eeg_l, img_l = [], []
    with torch.no_grad():
        for b in loader:
            eb = run_eeg_backbone(model, ea, b[0].to(dev), b[3].to(dev))
            eeg_l.append(ep(eb).cpu().numpy())
            img_l.append(ip(b[1].to(dev)).cpu().numpy())
    return np.concatenate(eeg_l), np.concatenate(img_l)


# ══════════════════════════════════════════════════════════════════════
# Experiment 1: Permutation null test
# ══════════════════════════════════════════════════════════════════════
def exp_permutation_null(E, V, p, n_perm=100, seed=0):
    E_t, _ = _full_pipeline_simple(E, V, p)
    s = _scores(E_t, V, p.csls_k)
    if p.final_sinkhorn: s = sinkhorn_normalize(s, tau=p.tau, num_iters=p.sink_iters)
    A_star = _top1(s)

    rng = np.random.default_rng(seed)
    A_perm = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(V.shape[0])
        V_pi = V[perm]
        E_t_pi, _ = _full_pipeline_simple(E, V_pi, p)
        s_pi = _scores(E_t_pi, V_pi, p.csls_k)
        if p.final_sinkhorn: s_pi = sinkhorn_normalize(s_pi, tau=p.tau, num_iters=p.sink_iters)
        # ground truth under permuted labels: target i→perm[i], but since we
        # score E_t_pi vs V_pi which is already reordered, targets are diagonal
        A_perm[k] = _top1(s_pi)
    pval = float((A_perm >= A_star).mean())
    return {"A_star": A_star, "A_perm": A_perm, "p_value": pval}


# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Bootstrap stability
# ══════════════════════════════════════════════════════════════════════
def exp_bootstrap_stability(E, V, p, n_boot=200, m=150, seed=0):
    _, R_star, *_ = _full_pipeline(E, V, p)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(E.shape[0], m, replace=False)
        _, R_b, *_ = _full_pipeline(E[idx], V[idx], p)
        deltas[b] = _geodesic(R_star, R_b)
    return {"mean_delta": float(deltas.mean()), "std_delta": float(deltas.std()), "deltas": deltas}


# ══════════════════════════════════════════════════════════════════════
# Experiment 3: Generalization 100-100 split (blend eval rotation with identity)
# ══════════════════════════════════════════════════════════════════════
def exp_generalization_split(E, V, p, blend_alphas=(0.0, 0.25, 0.5, 0.75, 1.0), n_splits=50, seed=0):
    """Held-out accuracy vs geodesic blend R^alpha between identity and fitted R."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    half = n // 2
    blend_alphas = tuple(float(a) for a in sorted(set(blend_alphas)))
    n_alpha = len(blend_alphas)
    sattc_bins = [[] for _ in range(n_alpha)]
    baseline_accs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        fit_idx, eval_idx = perm[:half], perm[half:2*half]
        mu, W = _fit_whiten(E[fit_idx], shrink=p.saw_shrink)
        E_fit = _apply_whiten(E[fit_idx], mu, W)
        V_fit = V[fit_idx]
        R = np.eye(E_fit.shape[1], dtype=np.float32)
        cur = E_fit.copy()
        for _step in range(p.steps):
            s = _scores(cur, V_fit, p.csls_k)
            P = sinkhorn_normalize(s, tau=p.tau, num_iters=p.sink_iters)
            dR = fit_soft_assignment_procrustes(cur, V_fit, P, power=p.power)
            if dR is None: break
            cur = apply_orthogonal_map(cur, dR)
            R = (R @ dR).astype(np.float32)
        R64 = np.asarray(R, dtype=np.float64)
        L_log = np.real(logm(R64.astype(np.complex128)))

        E_eval_w = _apply_whiten(E[eval_idx], mu, W)
        V_eval = V[eval_idx]
        for j, alpha in enumerate(blend_alphas):
            Ra = _rotation_blend_identity_geodesic(R64, alpha, L_log_precomp=L_log)
            E_eval_aligned = _norm(E_eval_w @ Ra)
            s_eval = _scores(E_eval_aligned, V_eval, p.csls_k)
            if p.final_sinkhorn: s_eval = sinkhorn_normalize(s_eval, tau=p.tau, num_iters=p.sink_iters)
            sattc_bins[j].append(_top1(s_eval))
        s_base = _cos(E[eval_idx], V_eval)
        baseline_accs.append(_top1(s_base))

    out = {
        "blend_alphas": list(blend_alphas),
        "baseline_mean": float(np.mean(baseline_accs)),
        "baseline_std": float(np.std(baseline_accs)),
    }
    out["sattc_mean"], out["sattc_std"] = [], []
    for j in range(n_alpha):
        arr = np.asarray(sattc_bins[j], dtype=np.float64)
        out["sattc_mean"].append(float(arr.mean()))
        out["sattc_std"].append(float(arr.std()))
    if 1.0 in blend_alphas:
        ia = blend_alphas.index(1.0)
        out["sattc_mean_full"] = out["sattc_mean"][ia]
        out["sattc_std_full"] = out["sattc_std"][ia]
    return out


# ══════════════════════════════════════════════════════════════════════
# Experiment 4: PCA trajectory visualisation
# ══════════════════════════════════════════════════════════════════════
def exp_pca_trajectory(E, V, p, out_dir, seed=0, n_perm_vis=1):
    mu_w, W = _fit_whiten(E, shrink=p.saw_shrink)
    E0 = _apply_whiten(E, mu_w, W)
    # fit PCA on [E0, V]
    combined = np.vstack([E0, V])
    pca_mean = combined.mean(0)
    _, _, Vt = np.linalg.svd(combined - pca_mean, full_matrices=False)
    P = Vt[:2].T  # (d, 2)

    # collect snapshots
    snapshots = [("t=0", E0.copy(), np.eye(E0.shape[1], dtype=np.float32))]
    cur = E0.copy()
    for t in range(p.steps):
        s = _scores(cur, V, p.csls_k)
        Pk = sinkhorn_normalize(s, tau=p.tau, num_iters=p.sink_iters)
        dR = fit_soft_assignment_procrustes(cur, V, Pk, power=p.power)
        if dR is None: break
        cur = apply_orthogonal_map(cur, dR)
        if t < 2 or t == p.steps - 1:
            snapshots.append((f"t={t+1}", cur.copy(), Pk.copy()))

    # permutation null panel
    rng = np.random.default_rng(seed)
    perm = rng.permutation(V.shape[0])
    V_pi = V[perm]
    E_perm, _ = _full_pipeline_simple(E, V_pi, p)
    snapshots.append(("perm-null", E_perm, None))

    n_panels = len(snapshots)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5*n_panels, 4), dpi=120)
    if n_panels == 1: axes = [axes]
    n = E0.shape[0]
    cmap = plt.cm.tab20(np.linspace(0, 1, min(n, 20)))
    colors = [cmap[i % len(cmap)] for i in range(n)]

    V_proj = (V - pca_mean) @ P
    for ax, (title, E_snap, Pk_snap) in zip(axes, snapshots):
        E_proj = (E_snap - pca_mean) @ P
        for i in range(n):
            ax.scatter(*E_proj[i], marker='o', color=colors[i], s=20, alpha=0.7, zorder=3)
            ax.scatter(*V_proj[i], marker='^', color=colors[i], s=20, alpha=0.7, zorder=3)
            if Pk_snap is not None:
                w = float(Pk_snap[i, i]) if Pk_snap.shape[0] > i else 0
                ax.plot([E_proj[i,0], V_proj[i,0]], [E_proj[i,1], V_proj[i,1]],
                        color=colors[i], alpha=min(1.0, w*3), linewidth=0.5, zorder=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    legend_elements = [Line2D([0],[0],marker='o',color='gray',label='EEG',markersize=5,linestyle='None'),
                       Line2D([0],[0],marker='^',color='gray',label='Image',markersize=5,linestyle='None')]
    axes[-1].legend(handles=legend_elements, fontsize=7, loc='upper right')
    fig.suptitle("PCA trajectory: EEG→Image alignment", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_trajectory.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved pca_trajectory.png")


# ══════════════════════════════════════════════════════════════════════
# Experiment 5: Geodesic distance curves
# ══════════════════════════════════════════════════════════════════════
def exp_geodesic_curves(all_subjects_EV, p, out_dir):
    """all_subjects_EV: list of (subject_id, E, V)"""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    all_steps, all_cumul = [], []
    for sid, E, V in all_subjects_EV:
        _, _, _, _, Rs = _full_pipeline(E, V, p, track=True)
        if Rs is None or len(Rs) < 2: continue
        steps = [_geodesic(Rs[t], Rs[t+1]) for t in range(len(Rs)-1)]
        cumul = [_geodesic(Rs[0], Rs[t]) for t in range(len(Rs))]
        iters_s = np.arange(len(steps))
        iters_c = np.arange(len(cumul))
        ax.plot(iters_s, steps, '-', color='tab:blue', alpha=0.25, linewidth=0.8)
        ax.plot(iters_c, cumul, '-', color='tab:orange', alpha=0.25, linewidth=0.8)
        # pad to uniform length for mean
        max_len = p.steps + 1
        s_pad = np.full(max_len, np.nan); s_pad[:len(steps)] = steps
        c_pad = np.full(max_len+1, np.nan); c_pad[:len(cumul)] = cumul
        all_steps.append(s_pad); all_cumul.append(c_pad)

    if all_steps:
        mean_s = np.nanmean(all_steps, axis=0)
        mean_c = np.nanmean(all_cumul, axis=0)
        ax.plot(np.arange(len(mean_s)), mean_s, '-', color='tab:blue', linewidth=2.5, label='step size (mean)')
        ax.plot(np.arange(len(mean_c)), mean_c, '-', color='tab:orange', linewidth=2.5, label='cumul. displacement (mean)')
    ax.set_xlabel("Iteration"); ax.set_ylabel("Geodesic distance on SO(d)")
    ax.legend(fontsize=8); ax.set_title("Geodesic convergence dynamics")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "geodesic_curves.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved geodesic_curves.png")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run_dir", type=str,
        default="results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260426-015522/param_k30_pool51_do050_featdim64_seed3300")
    ap.add_argument("--subjects", nargs="+", type=int, default=[1,2,3,4,5,6,7,8,9,10])
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=3300)
    # pipeline hypers
    ap.add_argument("--saw_shrink", type=float, default=0.65)
    ap.add_argument("--csls_k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--sink_iters", type=int, default=10)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--final_sinkhorn", action="store_true")
    # experiment toggles
    ap.add_argument("--skip_perm", action="store_true")
    ap.add_argument("--skip_bootstrap", action="store_true")
    ap.add_argument("--skip_generalization", action="store_true")
    ap.add_argument("--skip_pca", action="store_true")
    ap.add_argument("--skip_geodesic", action="store_true")
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--n_splits", type=int, default=50)
    ap.add_argument(
        "--blend_alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Geodesic blends exp(α log R): α=0→I, α=1→R when evaluating split B.",
    )
    args = ap.parse_args()

    if args.output_dir is None:
        from datetime import datetime
        tag = f"tta_diagnostics_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        args.output_dir = os.path.join(REPO_ROOT, "results", "things_eeg", "inter-subjects", tag)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    p = SimpleNamespace(saw_shrink=args.saw_shrink, csls_k=args.csls_k, tau=args.tau,
                        sink_iters=args.sink_iters, steps=args.steps, power=args.power,
                        final_sinkhorn=args.final_sinkhorn)

    # encode all subjects once
    subjects_data = []
    for sid in args.subjects:
        print(f"Encoding subject {sid}...")
        E, V = _encode(args.source_run_dir, sid, device=args.device)
        subjects_data.append((sid, E, V))

    results = {}

    # ── Exp 1: permutation null ───────────────────────────────────────
    if not args.skip_perm:
        print("\n=== Permutation Null Test ===")
        for sid, E, V in subjects_data:
            r = exp_permutation_null(E, V, p, n_perm=args.n_perm, seed=args.seed+sid)
            print(f"  sub-{sid:02d}: A*={r['A_star']:.4f}  p={r['p_value']:.6f}  "
                  f"perm_mean={r['A_perm'].mean():.4f}±{r['A_perm'].std():.4f}")
            results.setdefault("permutation_null", {})[sid] = {
                "A_star": r["A_star"], "p_value": r["p_value"],
                "perm_mean": float(r["A_perm"].mean()), "perm_std": float(r["A_perm"].std()),
            }

    # ── Exp 2: bootstrap stability ────────────────────────────────────
    if not args.skip_bootstrap:
        print("\n=== Bootstrap Stability ===")
        for sid, E, V in subjects_data:
            r = exp_bootstrap_stability(E, V, p, n_boot=args.n_boot, seed=args.seed+sid)
            print(f"  sub-{sid:02d}: δ = {r['mean_delta']:.6f} ± {r['std_delta']:.6f}")
            results.setdefault("bootstrap_stability", {})[sid] = {
                "mean_delta": r["mean_delta"], "std_delta": r["std_delta"],
            }

    # ── Exp 3: generalization split ───────────────────────────────────
    if not args.skip_generalization:
        print("\n=== Generalization Split ===")
        for sid, E, V in subjects_data:
            r = exp_generalization_split(
                E, V, p, blend_alphas=args.blend_alphas, n_splits=args.n_splits,
                seed=args.seed + sid)
            blends = ", ".join(
                f"α={a:g}:{m:.4f}±{s:.4f}"
                for a, m, s in zip(r["blend_alphas"], r["sattc_mean"], r["sattc_std"])
            )
            print(f"  sub-{sid:02d}: baseline={r['baseline_mean']:.4f}±{r['baseline_std']:.4f}")
            print(f"         {blends}")
            row = {
                "blend_alphas": r["blend_alphas"],
                "sattc_mean": r["sattc_mean"],
                "sattc_std": r["sattc_std"],
                "baseline_mean": r["baseline_mean"],
                "baseline_std": r["baseline_std"],
            }
            if "sattc_mean_full" in r:
                row["sattc_mean_full"] = r["sattc_mean_full"]
                row["sattc_std_full"] = r["sattc_std_full"]
            results.setdefault("generalization_split", {})[sid] = row

    # ── Exp 4: PCA trajectory (first subject only for viz) ────────────
    if not args.skip_pca:
        print("\n=== PCA Trajectory ===")
        sid, E, V = subjects_data[0]
        exp_pca_trajectory(E, V, p, args.output_dir, seed=args.seed)

    # ── Exp 5: geodesic curves ────────────────────────────────────────
    if not args.skip_geodesic:
        print("\n=== Geodesic Curves ===")
        exp_geodesic_curves(subjects_data, p, args.output_dir)

    # save json summary
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
