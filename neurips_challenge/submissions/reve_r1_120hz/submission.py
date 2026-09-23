"""Full-fine-tuned REVE submission for Track 1 EEG-to-image retrieval.

Neurobridge_SSL run reve/r1_in120_bf16 (local grader top-5 0.4063): REVE-base fine-tuned on
the official 120 Hz tensors, linearly interpolated to REVE's 200 samples exactly as below,
so training and inference see the same input.
"""

import os

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.models import REVE
from benchmark_utils.base_solver import CompetSolver


class FineTunedReve(nn.Module):
    """Fine-tuned REVE encoder and its 512-to-1536 retrieval head."""

    def __init__(self, n_chans, n_outputs, chs_info):
        super().__init__()
        self.backbone = REVE(
            n_chans=n_chans,
            n_times=200,
            n_outputs=2,
            chs_info=chs_info,
        )
        self.probe = nn.Linear(512, n_outputs)

    def forward(self, x):
        x = x.float()
        # Trained on the 120 Hz warm-up tensors stretched to REVE's 200-sample window.
        if x.shape[-1] != 200:
            x = F.interpolate(x, size=200, mode="linear", align_corners=False)
        encoded = self.backbone(x, return_output=True)[-1]
        return self.probe(encoded.mean(dim=1))

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        return self(x)


class Solver(CompetSolver):
    name = "Fine-Tuned-REVE-120Hz"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        submission_dir = meta["submission_dir"]
        os.environ["REVE_POSITIONS_PATH"] = str(submission_dir)
        model = FineTunedReve(
            n_chans=meta["n_chans"],
            n_outputs=meta["n_outputs"],
            chs_info=meta["chs_info"],
        )
        state = torch.load(
            submission_dir / "weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        model.load_state_dict(state)
        return model.to(meta["device"]).eval()
