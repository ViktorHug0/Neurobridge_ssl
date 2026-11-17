"""Track 1 (sealed phase, 32-ch Alljoined): ensemble of EEG-to-DINOv2 encoders.

Members (TorchScript: EEG encoder + projector, traced from Neurobridge_SSL runs on all 20 Alljoined people + THINGS-EEG2
sub-21..30 + 10 test reps per person; members.pt lists the files and their source checkpoints).

predict(X) uses the current batch only (no state across calls, any batch size, mixed people): channels are permuted by
name to the training order, each member's output is divided by its mean norm over the batch, and the members are
summed (times an optional per-member weight from members.pt, default 1). When members.pt names a detector, the batch is
first aligned in time (align(): a per-batch shift of a few samples, from the batch alone; with a 'reader' in members.pt, only when two
independent estimates of the delay agree). No test-time adaptation of the outputs: with the few repeats per image of the sealed set it costs accuracy (cv4, 3 repeats:
14.11 vs 12.46 top-5 on a 200-image gallery; it only pays off from ~8 repeats on).
"""

import torch
from benchmark_utils.base_solver import CompetSolver


class Ensemble:
    def __init__(self, submission_dir, served_channels, device):
        aux = torch.load(submission_dir / 'members.pt', map_location=device, weights_only=True)
        self.members = [torch.jit.load(str(submission_dir / f), map_location=device).eval() for f in aux['members']]
        self.weights = aux.get('weights', [1.0] * len(self.members))
        self.amp = set(aux.get('amp', []))  # members run in half precision (the large transformers: same output to 5 digits, half the time)
        served = [name.lower() for name in served_channels]
        self.order = torch.tensor([served.index(name.lower()) for name in aux['channels']], device=device)
        # optional latency alignment (members.pt 'detector'): see align()
        self.detector = torch.jit.load(str(submission_dir / aux['detector']), map_location=device).eval() if 'detector' in aux else None
        self.shifts, self.bias, self.gate = aux.get('shifts', []), aux.get('bias', 0), aux.get('gate', 0.0)
        # optional trained latency reader (members.pt 'reader'): align() then acts only when it agrees with the detector
        self.reader = torch.jit.load(str(submission_dir / aux['reader']), map_location=device).eval() if 'reader' in aux else None
        self.target, self.agree, self.offset = aux.get('target', 0.25), aux.get('agree', 2.5), aux.get('offset', 0.9)

    def align(self, X):
        """Latency alignment from the current batch alone (no labels, no state). A recording set-up can delay the EEG by a
        few samples relative to the training cohort (THINGS-EEG2 is 5 samples off Alljoined-1.6M), which costs a decoder
        trained at one timing most of its accuracy. The detector member is run on the batch shifted by each candidate
        (edge samples replicated); its mean output norm peaks at the timing it was trained on. The vertex of the parabola
        through the three points around the maximum, plus a calibration constant, is the shift applied to the batch.
        Batches under 32 trials are left as they are (the estimate needs the average over the batch)."""
        B, T = X.shape[0], X.shape[-1]
        if self.detector is None or B < 32:
            return X
        t = torch.arange(T, device=X.device)
        views = torch.cat([X[:, :, (t - c).clamp(0, T - 1)] for c in self.shifts])
        n = self.detector(views).float().norm(dim=1).view(len(self.shifts), B).mean(1)
        if (n.max() / n.mean() - 1).item() < self.gate:  # no clear peak over the candidate shifts (atypical people in a one-person batch): leave the batch as it is
            return X
        j = min(max(int(n.argmax()), 1), len(self.shifts) - 2)
        left, mid, right = n[j - 1].item(), n[j].item(), n[j + 1].item()
        curvature = left - 2 * mid + right
        vertex = self.shifts[j] + (0.5 * (left - right) / curvature * (self.shifts[1] - self.shifts[0]) if curvature < 0 else 0.0)
        c = int(round(min(max(vertex, self.shifts[0]), self.shifts[-1]) + self.bias))
        if self.reader is not None:
            # A small network trained to read the delay of a trial (17 classes, -8..+8 samples) gives a second estimate:
            # the vertex of its mean log-probability over the batch. It is more precise than the detector on people
            # recorded like the training cohort, and wrong on other protocols; the detector is the opposite. The batch
            # is corrected with the reader's value only when both agree, and left as it is otherwise.
            lp = torch.log_softmax(self.reader(X).float(), 1).mean(0)
            k = min(max(int(lp.argmax()), 1), lp.numel() - 2)
            left, mid, right = lp[k - 1].item(), lp[k].item(), lp[k + 1].item()
            curvature = left - 2 * mid + right
            delay = k - lp.numel() // 2 + (0.5 * (left - right) / curvature if curvature < 0 else 0.0)
            if abs(-vertex - self.offset - delay) > self.agree:
                return X
            c = -int(round(delay - self.target))
        return X[:, :, (t - c).clamp(0, T - 1)] if c else X

    @torch.inference_mode()
    def predict(self, X):
        X = self.align(X.float()[:, self.order])
        outs = []
        for i, member in enumerate(self.members):
            with torch.autocast(X.device.type, dtype=torch.float16, enabled=i in self.amp):
                outs.append(member(X).float())
        return sum(w * P / P.norm(dim=1).mean() for w, P in zip(self.weights, outs))


class Solver(CompetSolver):
    name = "Neurobridge-ensemble"

    def load_model(self, meta):
        return Ensemble(meta["submission_dir"], meta["ch_names"], meta["device"])
