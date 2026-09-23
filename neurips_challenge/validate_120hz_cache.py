"""Check build_120hz_cache.py against NeuralBench's own dataloader.

Plausible output is not evidence. This instantiates the real Track-1 test loader and asks whether
the epochs it yields appear, element for element, in the array we built. If our onset arithmetic,
baseline window, clamp or channel permutation is off by anything, the match rate collapses.

Run with the CHALLENGE venv (it needs neuralbench/neuralset):

    /nasbrain/p20fores/Neurips_challenge/.venv/bin/python neurips_challenge/validate_120hz_cache.py \
        --built_dir <dir written by build_120hz_cache.py> --subjects 1

Our array must be built with --window_start -0.2 --duration 1.0 for this to mean anything: that is
the window the official loader serves. A pass tells you the epoching machinery is faithful; it says
nothing about which window you should ultimately train on.

Expected match rate: our built subjects / 10. With one subject built, ~10% of their epochs should
match and the rest should not, so both a false pass and a false fail are visible.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def official_test_loader(batch_size: int, data_overrides: dict | None = None):
    from exca import ConfDict
    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.main import Experiment
    from neuralbench.registry import DEFAULTS_DIR, load_yaml_config

    import track1_eeg_image  # noqa: F401 - registers Gifford2022Local

    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    config["wandb_config"] = None
    grid = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))
    grid["seed"] = [33]
    configs = prepare_task_configs(
        config, grid, "eeg", "image",
        use_task_grid=False, debug=False, force=False, prepare=False, download=False,
        models=["track1_eegnet"], datasets=[None],
    )
    cfg = configs[0]
    # The default Track-1 config is the 120 Hz warm-up preset. Overriding data.neuro lets the same
    # element-for-element check cover a cache built from any other EegExtractor preset.
    for key, value in (data_overrides or {}).items():
        if key == "neuro":
            cfg["data"]["neuro"].update(value)
        else:
            cfg["data"][key] = value
    cfg["data"]["batch_size"] = batch_size
    cfg["data"]["num_workers"] = 4
    cfg["wandb_config"] = None
    return Experiment.model_validate(cfg).data.prepare()["test"]


def signature(epoch: np.ndarray) -> bytes:
    """Cheap key for candidate lookup: one channel, coarsely rounded."""
    return np.round(epoch[0], 3).astype(np.float32).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--built_dir", required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--data_overrides",
        default=None,
        help='JSON merged into the loader\'s data config, e.g. \'{"neuro": {"frequency": 200.0, '
             '"filter": [0.5, 99.5], "notch_filter": null, "baseline": null, '
             '"scaler": "StandardScaler", "clamp": 15.0}}\'',
    )
    args = parser.parse_args()
    overrides = json.loads(args.data_overrides) if args.data_overrides else None

    info = json.load(open(os.path.join(args.built_dir, "info.json")))
    if abs(info["window_start"] + 0.2) > 1e-9 or abs(info["duration"] - 1.0) > 1e-9:
        raise SystemExit(
            f"built with window_start={info['window_start']} duration={info['duration']}; "
            "the official loader serves -0.2/1.0, so this comparison would be meaningless"
        )
    nb_channels = info["ch_names"]

    # Index every epoch we built, keyed on a coarse signature.
    table: dict[bytes, list[np.ndarray]] = {}
    for subject in args.subjects:
        path = os.path.join(args.built_dir, f"sub-{subject:02d}", "test.npy")
        array = np.load(path, mmap_mode="r")
        flat = np.asarray(array).reshape(-1, array.shape[-2], array.shape[-1])
        for epoch in flat:
            table.setdefault(signature(epoch), []).append(epoch)
        print(f"indexed sub-{subject:02d}: {len(flat)} epochs")
    print(f"signature table: {len(table)} keys")

    loader = official_test_loader(args.batch_size, overrides)
    their_channels = [
        name for name, _ in sorted(
            loader.dataset.extractors["neuro"]._channels.items(), key=lambda kv: kv[1]
        )
    ]
    permutation = [their_channels.index(name) for name in nb_channels]
    print(f"channel permutation built: {len(permutation)} channels")

    seen = matched = 0
    worst = 0.0
    for batch_idx, batch in enumerate(loader, start=1):
        neuro = batch.data["neuro"].float().numpy()[:, permutation, :]
        for epoch in neuro:
            seen += 1
            for candidate in table.get(signature(epoch), ()):
                difference = float(np.abs(candidate - epoch).max())
                if difference <= args.tolerance:
                    matched += 1
                    worst = max(worst, difference)
                    break
        if batch_idx >= args.max_batches:
            break

    rate = matched / seen if seen else 0.0
    expected = len(args.subjects) / 10
    print(f"\nepochs checked : {seen}")
    print(f"matched        : {matched}  ({rate:.1%})")
    print(f"expected       : ~{expected:.1%} (built {len(args.subjects)}/10 subjects)")
    print(f"worst abs diff on matches: {worst:.3e}")

    if matched == 0:
        raise SystemExit("FAIL: no epoch matched - the epoching is wrong, not merely imprecise")
    if rate < 0.5 * expected:
        raise SystemExit(f"FAIL: match rate {rate:.1%} far below the expected {expected:.1%}")
    print("\nPASS: our epochs reproduce NeuralBench's dataloader output.")


if __name__ == "__main__":
    main()
