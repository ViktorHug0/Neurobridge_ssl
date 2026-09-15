#!/usr/bin/env python3
"""
Load sub-01/test.npy only, pick 9 random test trials, and save seven figures:
three P5 grids (regular, overlay, pairwise average), three 16x16 seaborn heatmaps
(raw, raw with ~50% of cells randomly scaled by ±20%, and Sinkhorn of the raw matrix),
and one 9x9 seaborn heatmap (low background with three diagonal 3x3 high blocks).

Fast path: --build-cache writes a small .npz next to the data (or use --cache-path).

Requires seaborn for the heatmap; use `conda install seaborn` or `python3 -m pip install seaborn`
if `pip install` hits PEP 668.

Run from repo root:
  python3 scripts/things_eeg/plot_random_eeg_grid.py --build-cache --eeg_data_dir /path/to/preprocessed_eeg
  python3 scripts/things_eeg/plot_random_eeg_grid.py --eeg_data_dir /path/to/preprocessed_eeg --out ./figure.jpg
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

try:
    import seaborn as sns
except ModuleNotFoundError as _e:
    raise SystemExit(
        "Missing dependency: seaborn (required for the 16x16 heatmaps).\n\n"
        "Your `pip` may be blocked by PEP 668 on the system Python. Install into the "
        "same environment as the `python` you use to run this script, for example:\n\n"
        "  conda install seaborn\n"
        "  # or:\n"
        "  python3 -m pip install seaborn\n\n"
        "Avoid bare `pip install` if it targets a different interpreter than `python3`."
    ) from _e

CHANNEL_NAME = "P5"
TIME_START_S = 0.2
TIME_END_S = 0.6
LINE_WIDTH = 2.4
SINKHORN_ITERS = 5
# sub-01 test.npy only; bump when cache schema changes
CACHE_VERSION = 2
SUBJECT_ID = 1
TRAIN = False


def _subject_dir(eeg_data_dir: str, subject_id: int = SUBJECT_ID) -> str:
    return os.path.join(eeg_data_dir, f"sub-{subject_id:02d}")


def _resolve_eeg_file(subject_dir: str, train: bool) -> str:
    if train:
        candidates = ["train.npy", "preprocessed_eeg_training.npy"]
    else:
        candidates = ["test.npy", "preprocessed_eeg_test.npy"]
    for name in candidates:
        p = os.path.join(subject_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not find {'train' if train else 'test'} EEG file in '{subject_dir}'. "
        f"Tried: {', '.join(candidates)}"
    )


def _load_eeg_container(path: str):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    return obj


def _standardize_eeg_array(eeg_obj, train: bool, num_images_per_object: int = 10) -> np.ndarray:
    if isinstance(eeg_obj, dict):
        if "preprocessed_eeg_data" in eeg_obj:
            x = eeg_obj["preprocessed_eeg_data"]
        else:
            raise KeyError(f"Unsupported EEG dict keys: {list(eeg_obj.keys())}")
    else:
        x = eeg_obj
    if not isinstance(x, np.ndarray):
        raise TypeError(f"EEG data must be ndarray after load, got {type(x)}")
    if x.ndim == 5:
        return x
    if train and x.ndim == 4:
        n = x.shape[0]
        if n % num_images_per_object != 0:
            raise ValueError(f"Cannot reshape training EEG {x.shape}")
        num_objects = n // num_images_per_object
        return x.reshape(num_objects, num_images_per_object, x.shape[1], x.shape[2], x.shape[3])
    if (not train) and x.ndim == 4:
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
    raise ValueError(f"Unsupported EEG array shape: {x.shape}")


def _load_info(eeg_data_dir: str) -> dict:
    p = os.path.join(eeg_data_dir, "info.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _default_cache_path(eeg_data_dir: str) -> str:
    cache_dir = os.path.join(eeg_data_dir, ".cache")
    return os.path.join(
        cache_dir,
        f"plot_random_eeg_{CHANNEL_NAME.lower()}_sub{SUBJECT_ID:02d}_test_{TIME_START_S}_{TIME_END_S}v{CACHE_VERSION}.npz",
    )


def _time_vector_full(info: dict, n_t: int) -> np.ndarray:
    times = info.get("times")
    if times is not None:
        t = np.asarray(times, dtype=np.float64)
        if t.size != n_t:
            t = np.linspace(float(t[0]), float(t[-1]), n_t)
        return t
    sfreq = float(info.get("sfreq", 250.0))
    return np.arange(n_t, dtype=np.float64) / sfreq


def _time_window_mask(t: np.ndarray) -> np.ndarray:
    win = (t >= TIME_START_S) & (t <= TIME_END_S)
    if not np.any(win):
        raise SystemExit(
            f"No samples in [{TIME_START_S}, {TIME_END_S}] s; time axis is "
            f"[{float(t.min()):.4f}, {float(t.max()):.4f}]"
        )
    return win


def _resolve_p5_index(raw, info: dict, eeg_data_dir: str, x_shape: tuple[int, ...]) -> int:
    ch_names: list[str] | None = None
    if isinstance(raw, dict) and raw.get("ch_names") is not None:
        ch_names = list(raw["ch_names"])
    if ch_names is None:
        cn = info.get("ch_names")
        if cn is not None:
            ch_names = list(cn)
    if ch_names is None:
        raise SystemExit(
            f"Cannot resolve channel {CHANNEL_NAME!r}: no 'ch_names' in EEG file dict "
            f"or {eeg_data_dir}/info.json"
        )
    if CHANNEL_NAME not in ch_names:
        raise SystemExit(f"Channel {CHANNEL_NAME!r} not in ch_names (have {len(ch_names)} channels).")
    ch_i = ch_names.index(CHANNEL_NAME)
    if ch_i >= x_shape[3]:
        raise SystemExit(f"Channel index {ch_i} out of range for EEG shape {x_shape}")
    return ch_i


def _npz_scalar_str(z, key: str) -> str:
    v = z[key]
    a = np.asarray(v)
    if a.shape == ():
        return str(a.item())
    return str(a)


def _npz_scalar_int(z, key: str) -> int:
    return int(np.asarray(z[key]).item())


def _npz_scalar_float(z, key: str) -> float:
    return float(np.asarray(z[key]).item())


def _try_load_snippet_cache(
    cache_path: str,
    eeg_data_dir: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.isfile(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    try:
        if _npz_scalar_int(z, "cache_version") != CACHE_VERSION:
            return None
        if _npz_scalar_str(z, "eeg_data_dir") != eeg_data_dir:
            return None
        if _npz_scalar_int(z, "subject_id") != SUBJECT_ID:
            return None
        if bool(_npz_scalar_int(z, "train")) != TRAIN:
            return None
        if _npz_scalar_str(z, "channel") != CHANNEL_NAME:
            return None
        if _npz_scalar_float(z, "time_start") != TIME_START_S or _npz_scalar_float(z, "time_end") != TIME_END_S:
            return None
        snippets = np.asarray(z["snippets"], dtype=np.float32)
        t_plot = np.asarray(z["t_plot"], dtype=np.float64)
        if snippets.ndim != 2 or t_plot.ndim != 1:
            return None
        if snippets.shape[1] != t_plot.shape[0]:
            return None
    finally:
        z.close()
    return snippets, t_plot


def _save_snippet_cache(
    cache_path: str,
    snippets: np.ndarray,
    t_plot: np.ndarray,
    eeg_data_dir: str,
) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    np.savez_compressed(
        tmp,
        snippets=np.asarray(snippets, dtype=np.float32),
        t_plot=np.asarray(t_plot, dtype=np.float64),
        eeg_data_dir=eeg_data_dir,
        subject_id=SUBJECT_ID,
        train=int(TRAIN),
        channel=CHANNEL_NAME,
        time_start=TIME_START_S,
        time_end=TIME_END_S,
        cache_version=CACHE_VERSION,
    )
    os.replace(tmp, cache_path)


def _load_sub01_test(eeg_data_dir: str):
    """Load standardized EEG from sub-01 test.npy only."""
    sd = _subject_dir(eeg_data_dir)
    if not os.path.isdir(sd):
        raise SystemExit(f"Missing subject directory: {sd}")
    eeg_path = _resolve_eeg_file(sd, train=TRAIN)
    info = _load_info(eeg_data_dir)
    raw = _load_eeg_container(eeg_path)
    x = _standardize_eeg_array(raw, train=TRAIN)
    if x.ndim != 5:
        raise SystemExit(f"Unexpected EEG shape after standardize: {x.shape}")
    ch_i = _resolve_p5_index(raw, info, eeg_data_dir, x.shape)
    return x, ch_i, info


def _build_snippet_cache(
    *,
    eeg_data_dir: str,
    cache_path: str,
    n_samples: int,
    seed: int | None,
) -> None:
    rng_np = np.random.default_rng(seed)
    x, ch_i, info = _load_sub01_test(eeg_data_dir)

    n_obj, n_img, n_rep, _n_ch, n_t = x.shape
    t_full = _time_vector_full(info, n_t)
    win = _time_window_mask(t_full)
    t_plot = t_full[win]

    flat_n = n_obj * n_img * n_rep
    n_samples = min(n_samples, flat_n)
    idx = rng_np.choice(flat_n, size=n_samples, replace=False)
    unr = np.stack(np.unravel_index(idx, (n_obj, n_img, n_rep)), axis=1)
    rows = [
        x[int(io), int(ii), int(ir), ch_i].astype(np.float32, copy=False)[win]
        for io, ii, ir in unr
    ]
    snippets = np.stack(rows, axis=0)
    _save_snippet_cache(cache_path, snippets, t_plot, eeg_data_dir)
    out_uri = Path(cache_path).as_uri()
    print(f"Saved {n_samples} snippets to:\n{cache_path}\n{out_uri}")


def _center_signal(y: np.ndarray) -> np.ndarray:
    y0 = np.asarray(y, dtype=np.float64)
    return y0 - float(np.mean(y0))


def _style_axis(ax, spine_lw: float) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.52, linewidth=0.85, zorder=0)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.tick_params(
        axis="both",
        which="major",
        labelleft=False,
        labelbottom=False,
        length=0,
        width=0,
    )
    ax.minorticks_off()
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)


def _new_figure() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8, 8),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.12},
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.06)
    return fig, axes


def _random_derangement(n: int, rng: np.random.Generator) -> list[int]:
    perm = list(rng.permutation(n))
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def _draw_mixed_dashed_line(ax, t: np.ndarray, y: np.ndarray, color_a, color_b, linewidth: float) -> None:
    dash = (0, (6, 6))
    dash_shifted = (6, (6, 6))
    ax.plot(t, y, color=color_a, linewidth=linewidth, linestyle=dash, zorder=2)
    ax.plot(t, y, color=color_b, linewidth=linewidth, linestyle=dash_shifted, zorder=2)


def _output_paths(
    out_arg: str | None,
) -> tuple[
    str | None, str | None, str | None, str | None, str | None, str | None, str | None
]:
    if out_arg is None:
        return None, None, None, None, None, None, None
    out_path = os.path.abspath(os.path.expanduser(out_arg))
    p = Path(out_path)
    return (
        out_path,
        str(p.with_name(f"{p.stem}_overlay{p.suffix}")),
        str(p.with_name(f"{p.stem}_average{p.suffix}")),
        str(p.with_name(f"{p.stem}_heatmap{p.suffix}")),
        str(p.with_name(f"{p.stem}_heatmap_perturbed{p.suffix}")),
        str(p.with_name(f"{p.stem}_heatmap_sinkhorn{p.suffix}")),
        str(p.with_name(f"{p.stem}_block_9x9{p.suffix}")),
    )


def _perturb_cells_pm20(
    raw: np.ndarray,
    rng: np.random.Generator,
    *,
    frac: float = 0.5,
    rel: float = 0.2,
) -> np.ndarray:
    """Copy `raw`; randomly pick `frac` of cells and multiply by (1+rel) or (1-rel)."""
    out = np.asarray(raw, dtype=np.float64).copy()
    n = out.size
    k = max(1, int(round(n * frac)))
    flat_idx = rng.choice(n, size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k)
    flat = out.ravel()
    new_vals = flat[flat_idx] * (1.0 + rel * signs)
    flat[flat_idx] = np.maximum(new_vals, 0.0)
    return out


def _sinkhorn_scale(K: np.ndarray, n_iter: int) -> np.ndarray:
    """
    Alternating row/column normalization (Sinkhorn) on a positive matrix.
    Each iteration: scale rows to sum to 1, then columns to sum to 1.
    """
    A = np.asarray(K, dtype=np.float64)
    A = np.maximum(A, 1e-12)
    for _ in range(n_iter):
        A = A / (A.sum(axis=1, keepdims=True) + 1e-15)
        A = A / (A.sum(axis=0, keepdims=True) + 1e-15)
    return A


def _plot_heatmap_fig(data: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.08)
    sns.heatmap(
        data,
        ax=ax,
        cmap="viridis",
        cbar=True,
        cbar_ax=cax,
        square=True,
        linewidths=0.55,
        linecolor="0.25",
        annot=False,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.axis("off")
    cax.set_xlabel("")
    cax.set_ylabel("")
    cax.set_title("")
    cax.tick_params(which="both", length=0, width=0, labelleft=False, labelright=False)
    cax.minorticks_off()
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    return fig


def _build_9x9_block_diagonal_matrix(rng: np.random.Generator) -> np.ndarray:
    """9x9: U(0,0.1) off-diagonal blocks; three consecutive 3x3 diagonal blocks in [0.9, 1]."""
    m = rng.uniform(0.0, 0.3, size=(9, 9))
    for b in range(3):
        lo, hi = b * 3, (b + 1) * 3
        m[lo:hi, lo:hi] = rng.uniform(0.70, 1.0, size=(3, 3))
    return m


def _plot_9x9_block_heatmap_fig(data: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        data,
        ax=ax,
        cmap="viridis",
        cbar=False,
        square=True,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.4,
        linecolor="0.32",
    )
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def _plot_random_heatmaps(rng: np.random.Generator) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    raw = rng.random((16, 16))
    raw[0, 0] = 1.0
    raw[0, 1] = 0.0
    raw[0, 2] = 0.7
    raw[1, 0] = 0.1
    raw[1, 1] = 0.8
    raw[1, 2] = 0.3
    raw[2, 0] = 0.05
    raw[2, 1] = 0.1
    raw[2, 2] = 0.05
    raw[0, -1] = 1
    raw[1, -1] = 0.6
    raw[2, -1] = 0.1
    raw[-1, 0] = 0
    raw[-1, 1] = 0.1
    raw[-1, 2] = 1
    raw[-1, -1] = 0.4
    perturbed = _perturb_cells_pm20(raw, rng, frac=0.5, rel=0.2)
    sink = _sinkhorn_scale(raw, SINKHORN_ITERS)
    return _plot_heatmap_fig(raw), _plot_heatmap_fig(perturbed), _plot_heatmap_fig(sink)


def _save_or_show_figures(
    *,
    fig_regular: plt.Figure,
    fig_overlay: plt.Figure,
    fig_average: plt.Figure,
    fig_heatmap: plt.Figure,
    fig_heatmap_perturbed: plt.Figure,
    fig_heatmap_sinkhorn: plt.Figure,
    fig_9x9: plt.Figure,
    args: argparse.Namespace,
) -> None:
    out_regular, out_overlay, out_average, out_heatmap, out_pert, out_sk, out_9x9 = _output_paths(
        args.out
    )
    if out_regular is not None:
        figures = [
            ("regular", fig_regular, out_regular),
            ("overlay", fig_overlay, out_overlay),
            ("average", fig_average, out_average),
            ("heatmap", fig_heatmap, out_heatmap),
            ("heatmap_perturbed", fig_heatmap_perturbed, out_pert),
            ("heatmap_sinkhorn", fig_heatmap_sinkhorn, out_sk),
            ("block_9x9", fig_9x9, out_9x9),
        ]
        for _label, fig, path in figures:
            assert path is not None
            fig.savefig(path, dpi=150, bbox_inches="tight")
        print(
            "Saved figures to:\n"
            f"{out_regular}\n{Path(out_regular).as_uri()}\n"
            f"{out_overlay}\n{Path(out_overlay).as_uri()}\n"
            f"{out_average}\n{Path(out_average).as_uri()}\n"
            f"{out_heatmap}\n{Path(out_heatmap).as_uri()}\n"
            f"{out_pert}\n{Path(out_pert).as_uri()}\n"
            f"{out_sk}\n{Path(out_sk).as_uri()}\n"
            f"{out_9x9}\n{Path(out_9x9).as_uri()}"
        )
    else:
        print("No file saved (--out not set); opening the regular plot window.")
        plt.close(fig_overlay)
        plt.close(fig_average)
        plt.close(fig_heatmap)
        plt.close(fig_heatmap_perturbed)
        plt.close(fig_heatmap_sinkhorn)
        plt.close(fig_9x9)
        plt.show()
    plt.close(fig_regular)
    if plt.fignum_exists(fig_overlay.number):
        plt.close(fig_overlay)
    if plt.fignum_exists(fig_average.number):
        plt.close(fig_average)
    if plt.fignum_exists(fig_heatmap.number):
        plt.close(fig_heatmap)
    if plt.fignum_exists(fig_heatmap_perturbed.number):
        plt.close(fig_heatmap_perturbed)
    if plt.fignum_exists(fig_heatmap_sinkhorn.number):
        plt.close(fig_heatmap_sinkhorn)
    if plt.fignum_exists(fig_9x9.number):
        plt.close(fig_9x9)


def _plot_grids(
    *,
    t: np.ndarray,
    ys: list[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    cmap = plt.get_cmap("tab10")
    colors = [cmap((8 + 1) % 10) for i in range(9)]
    spine_lw = float(plt.rcParams.get("axes.linewidth", 0.8)) * 2.0
    ys_centered = [_center_signal(y) for y in ys]
    pair_idx = _random_derangement(len(ys_centered), rng)

    fig_regular, axes_regular = _new_figure()
    fig_overlay, axes_overlay = _new_figure()
    fig_average, axes_average = _new_figure()
    fig_heatmap, fig_heatmap_perturbed, fig_heatmap_sinkhorn = _plot_random_heatmaps(rng)
    fig_9x9 = _plot_9x9_block_heatmap_fig(_build_9x9_block_diagonal_matrix(rng))

    for i, ax in enumerate(axes_regular.flat):
        _style_axis(ax, spine_lw)
        ax.plot(t, ys_centered[i], color=colors[i], linewidth=LINE_WIDTH, zorder=2)

    for i, ax in enumerate(axes_overlay.flat):
        j = pair_idx[i]
        y_a = ys_centered[i]
        y_b = ys_centered[j]
        _style_axis(ax, spine_lw)
        ax.plot(t, y_a, color=colors[i], linewidth=LINE_WIDTH, zorder=2)
        ax.plot(t, y_b, color=colors[j], linewidth=LINE_WIDTH, zorder=2)

    for i, ax in enumerate(axes_average.flat):
        j = pair_idx[i]
        y_a = ys_centered[i]
        y_b = ys_centered[j]
        y_avg = 0.5 * (y_a + y_b)
        _style_axis(ax, spine_lw)
        _draw_mixed_dashed_line(ax, t, y_avg, colors[i], colors[j], linewidth=LINE_WIDTH)

    _save_or_show_figures(
        fig_regular=fig_regular,
        fig_overlay=fig_overlay,
        fig_average=fig_average,
        fig_heatmap=fig_heatmap,
        fig_heatmap_perturbed=fig_heatmap_perturbed,
        fig_heatmap_sinkhorn=fig_heatmap_sinkhorn,
        fig_9x9=fig_9x9,
        args=args,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eeg_data_dir",
        type=str,
        default="./things_eeg/data/preprocessed_eeg",
        help="Directory containing sub-01/… with test.npy",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Base path for saved figures: regular, *_overlay, *_average, *_heatmap, "
        "*_heatmap_perturbed, *_heatmap_sinkhorn, *_block_9x9. "
        "Omit to open only the regular plot interactively.",
    )
    p.add_argument(
        "--build-cache",
        action="store_true",
        help="Load sub-01 test.npy once, extract P5 snippets into <eeg_data_dir>/.cache/ "
        f"({TIME_START_S}–{TIME_END_S} s; count set by --cache-n).",
    )
    p.add_argument(
        "--cache-n",
        type=int,
        default=400,
        help="Number of snippets to store when using --build-cache (default: 400).",
    )
    p.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Override path for snippet .npz (default: <eeg_data_dir>/.cache/...).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read snippet cache; always load sub-01 test.npy from disk.",
    )
    args = p.parse_args()

    eeg_data_dir = os.path.abspath(args.eeg_data_dir)
    cache_path = args.cache_path or _default_cache_path(eeg_data_dir)

    if args.build_cache:
        _build_snippet_cache(
            eeg_data_dir=eeg_data_dir,
            cache_path=cache_path,
            n_samples=max(1, int(args.cache_n)),
            seed=args.seed,
        )
        return

    rng_np = np.random.default_rng(args.seed)

    if not args.no_cache:
        loaded = _try_load_snippet_cache(cache_path, eeg_data_dir)
        if loaded is not None:
            snippets, t = loaded
            n = snippets.shape[0]
            pick = rng_np.choice(n, size=9, replace=n < 9)
            ys = [snippets[i] for i in pick]
            _plot_grids(t=t, ys=ys, args=args, rng=rng_np)
            return

    x, ch_i, info = _load_sub01_test(eeg_data_dir)
    n_obj, n_img, n_rep, _n_ch, n_t = x.shape
    flat_n = n_obj * n_img * n_rep
    idx = rng_np.choice(flat_n, size=9, replace=flat_n < 9)
    unraveled = np.stack(np.unravel_index(idx, (n_obj, n_img, n_rep)), axis=1)

    t_full = _time_vector_full(info, n_t)
    win = _time_window_mask(t_full)
    t = t_full[win]

    ys = [x[io, ii, ir, ch_i].astype(np.float32, copy=False)[win] for io, ii, ir in unraveled]
    _plot_grids(t=t, ys=ys, args=args, rng=rng_np)

    if not args.no_cache:
        print(
            "Tip: precompute snippets for faster plots:\n"
            f"  python3 scripts/things_eeg/plot_random_eeg_grid.py --build-cache "
            f"--eeg_data_dir {eeg_data_dir}"
        )


if __name__ == "__main__":
    main()
