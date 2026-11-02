"""Scoring probe for Track 1 warm-up (update 1): a fixed, untrained random linear map EEG -> 1536-D.

It carries no image information, so both pooled and subject-aggregated top-5 should land at chance (5/200 = 0.025).
If the leaderboard shows ~1.0 for this, the subject-aggregated metric is broken, not saturated.
"""

import torch

from benchmark_utils.base_solver import CompetSolver


class RandomProjection(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("W", torch.randn(n_in, n_out, generator=g) / n_in ** 0.5)

    @torch.inference_mode()
    def predict(self, x):
        return x.float().flatten(1) @ self.W


class Solver(CompetSolver):
    name = "Probe-Random-Projection"

    def load_model(self, meta):
        return RandomProjection(meta["n_chans"] * meta["n_times"], meta["n_outputs"]).to(meta["device"])
