"""Does a held-out subject's collapse come from its input scale? Rescore a checkpoint on one subject with the
EEG rescaled at input: a global gain, or per-trial RMS normalisation to a target RMS (keeps the channel profile).

    diag_input_scale.py <run_dir> --subjects 15 14 --gains 1 1.35 1.8 2.5 --rms_targets 1.1
Prints sealed-rule (subject-aggregated) top-5 per (subject, transform). Uses the last-epoch checkpoint.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402
from neurips_challenge.score_track1 import build_modules, build_test_dataset, score_modules  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--subjects", type=int, nargs="+", default=[15, 14])
p.add_argument("--gains", type=float, nargs="*", default=[1.0])
p.add_argument("--rms_targets", type=float, nargs="*", default=[])
p.add_argument("--eeg_data_dir", default="/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench")
args = p.parse_args()

cfg = SimpleNamespace(**json.load(open(os.path.join(args.run_dir, "train_config.json"))))
cfg.eeg_data_dir = args.eeg_data_dir  # the run's staged /tmp copy is gone
device = torch.device("cuda:0")
ckpt = torch.load(os.path.join(args.run_dir, "checkpoint_test_best.pth"), map_location=device, weights_only=False)
n_times = int(cfg.time_window[1]) - int(cfg.time_window[0])
mods = build_modules(cfg, ckpt, build_test_dataset(cfg, [args.subjects[0]]), device, n_times)

orig = train.run_eeg_backbone
transforms = [(f"gain {g:g}", lambda x, g=g: x * g) for g in args.gains]
transforms += [(f"rms->{t:g}", lambda x, t=t: x * (t / x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)))
               for t in args.rms_targets]
for s in args.subjects:
    for name, fn in transforms:
        train.run_eeg_backbone = lambda model, c, eeg, sid, fn=fn: orig(model, c, fn(eeg), sid)
        r = score_modules(*mods, cfg, [s], device, 512, 0)
        print(f"{os.path.basename(args.run_dir)} sub {s} {name}: top5 {100 * r['subject_agg']['top5']:.1f}", flush=True)
train.run_eeg_backbone = orig
