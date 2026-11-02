"""Uniform weight soup: average several runs' best checkpoints into one run dir, then grade it.

    soup.py <out_dir> <run_dir> [<run_dir> ...]

The runs must share an architecture and init (same --seed); only float tensors are averaged
(BatchNorm running stats included, num_batches_tracked taken from the first run).
"""

import json
import os
import shutil
import subprocess
import sys

import torch

out_dir, run_dirs = sys.argv[1], sys.argv[2:]
assert len(run_dirs) >= 2, "a soup needs at least two runs"
states = [torch.load(os.path.join(d, "checkpoint_test_best.pth"), map_location="cpu", weights_only=False)
          for d in run_dirs]
soup = states[0]
for key, value in soup["model_state_dict"].items():
    if value.is_floating_point():
        soup["model_state_dict"][key] = sum(s["model_state_dict"][key].float() for s in states).div(
            len(states)).to(value.dtype)
os.makedirs(out_dir, exist_ok=True)
torch.save(soup, os.path.join(out_dir, "checkpoint_test_best.pth"))
shutil.copy(os.path.join(run_dirs[0], "train_config.json"), out_dir)
with open(os.path.join(out_dir, "soup_members.json"), "w") as handle:
    json.dump(run_dirs, handle, indent=1)
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "score_track1.py"), out_dir], check=True)
