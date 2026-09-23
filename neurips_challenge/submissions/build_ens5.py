"""Pack the 5-member ensemble: weights.pt for ens5/submission.py, member-by-member parity check.

Parity: every packaged member must reproduce, on real official test EEG, the prediction of the
model train.py builds from the same checkpoint (the one score_track1.py graded).
"""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from neurips_challenge.score_track1 import build_modules, build_test_dataset, OFFICIAL_EEG_DIR  # noqa: E402

RUNS = REPO / "results/things_eeg/neurips_track1"
MEMBERS = {  # packaged name -> run dir
    "reve_r1": RUNS / "reve/r1_in120_bf16/20260923-173716-REVE",
    "reve_r12": RUNS / "reve/r12_in120_lr_bf16/20260923-173243-REVE",
    "reve_r123": RUNS / "reve/r123_in120_lr_head_bf16/20260923-173240-REVE",
    "sqf": RUNS / "recipe/sqf_trunk512_oc50/OrthoFastTSConvSqueezeformer/20260923-161353-OrthoFastTSConvSqueezeformer",
    "sqf_power": RUNS / "recipe/sqf_trunk512_oc50_power/OrthoFastTSConvSqueezeformer/20260923-193847-OrthoFastTSConvSqueezeformer",
}
OUT = Path(__file__).resolve().parent / "ens5"


def packaged_key(member, key):
    if member.startswith("reve"):  # FoundationEncoder: fm.* -> backbone.*, head.* -> probe.*
        key = "backbone." + key[3:] if key.startswith("fm.") else "probe." + key[len("head."):]
    return f"members.{member}.{key}"


def load_submission():
    package = types.ModuleType("benchmark_utils")
    base = types.ModuleType("benchmark_utils.base_solver")
    base.CompetSolver = type("CompetSolver", (), {})
    package.base_solver = base
    sys.modules.setdefault("benchmark_utils", package)
    sys.modules.setdefault("benchmark_utils.base_solver", base)
    spec = importlib.util.spec_from_file_location("ens5_submission", OUT / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    device = torch.device("cuda")
    state, originals, eeg = {}, {}, None
    for member, run_dir in MEMBERS.items():
        cfg = SimpleNamespace(**json.load(open(run_dir / "train_config.json")))
        cfg.eeg_data_dir = OFFICIAL_EEG_DIR
        checkpoint = torch.load(run_dir / "checkpoint_test_best.pth", map_location="cpu", weights_only=False)
        assert not checkpoint["eeg_projector_state_dict"], "direct projector expected"
        for key, value in checkpoint["model_state_dict"].items():
            state[packaged_key(member, key)] = value.float().contiguous()
        dataset = build_test_dataset(cfg, [1])
        if eeg is None:
            eeg = torch.stack([torch.as_tensor(dataset[i][0]) for i in range(0, 16000, 125)]).float().to(device)
        n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
        model, projector, _ = build_modules(cfg, checkpoint, dataset, device, n_times)
        x = eeg if eeg.shape[-1] == n_times else torch.nn.functional.interpolate(
            eeg, size=n_times, mode="linear", align_corners=False)
        originals[member] = projector(model(x)).float()
        del model, projector, dataset
        print(f"{member}: {len(checkpoint['model_state_dict'])} tensors", flush=True)

    torch.save(state, OUT / "weights.pt")
    submission = load_submission()
    from module.eeg_encoder.foundation import THINGS_EEG2_CH_NAMES
    meta = {"submission_dir": OUT, "device": "cuda", "n_chans": 63, "n_times": 120, "n_outputs": 1536,
            "chs_info": [{"ch_name": n} for n in THINGS_EEG2_CH_NAMES]}
    ensemble = submission.Solver().load_model(meta)  # load_state_dict is strict
    for member, module in ensemble.members.items():
        diff = (module(eeg) - originals[member]).abs().max().item()
        scale = originals[member].abs().max().item()
        print(f"parity {member}: max |diff| {diff:.2e} (output scale {scale:.2f})", flush=True)
        assert diff < 1e-3 * scale, member
    params = sum(p.numel() for p in ensemble.parameters())
    print(f"ensemble parameters: {params / 1e6:.1f}M; weights.pt {os.path.getsize(OUT / 'weights.pt') / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
