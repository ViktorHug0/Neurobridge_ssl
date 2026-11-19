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
        self.members, self.weights, self.amp = [], [], set()  # amp: members run in half precision (the large transformers: same output to 5 digits, half the time)
        for i, f in enumerate(aux['members']):
            try:  # a member that does not load in the grader's runtime is left out rather than failing the whole run
                self.members.append(torch.jit.load(str(submission_dir / f), map_location=device).eval())
            except Exception as e:
                print(f'member {f} left out: {e}'); continue
            self.weights.append(aux.get('weights', [1.0] * len(aux['members']))[i])
            if i in aux.get('amp', []):
                self.amp.add(len(self.members) - 1)
        served = [name.lower() for name in served_channels]
        self.order = torch.tensor([served.index(name.lower()) for name in aux['channels']], device=device)
        # optional latency alignment (members.pt 'detector'): see align()
        self.detector = self._load(submission_dir, aux.get('detector'), device)
        self.shifts, self.bias, self.gate = aux.get('shifts', []), aux.get('bias', 0), aux.get('gate', 0.0)
        self.cap, self.reject = aux.get('cap', 99), aux.get('reject', False)  # |correction| above cap: clipped to it, or (reject) no correction
        self.dead = aux.get('dead', 0)  # |correction| below dead: not applied (one-sample proposals at zero delay are mostly noise)
        # each trial's output scaled by its own norm^(pow-1): the grader averages raw outputs over repeats, so a pow above 1 lets the
        # trials the ensemble answers with more conviction weigh more in that average (per trial: no batch statistic, no state)
        self.pow = aux.get('pow', 1.0)
        # optional trained latency reader (members.pt 'reader'): align() then acts only when it agrees with the detector
        self.reader = self._load(submission_dir, aux.get('reader'), device)
        self.target, self.agree, self.offset = aux.get('target', 0.25), aux.get('agree', 2.5), aux.get('offset', 0.9)
        # TorchScript optimises a graph after its first profiled calls (outputs then change by ~1e-4): run every network twice on
        # zeros here, so that every real batch goes through the same optimised graph whatever its position in the stream
        z = torch.zeros(64, len(aux['channels']), 120, device=device)
        with torch.inference_mode():
            for _ in range(2):
                for i, net in enumerate(self.members + [n for n in (self.detector, self.reader) if n is not None]):
                    try:
                        with torch.autocast(z.device.type, dtype=torch.float16, enabled=i in self.amp):
                            net(z if net is not self.detector else z.repeat(len(self.shifts), 1, 1))
                    except Exception:
                        pass

    @staticmethod
    def _load(submission_dir, f, device):  # optional alignment network: None if absent or not loadable here (no alignment then)
        try:
            return torch.jit.load(str(submission_dir / f), map_location=device).eval() if f else None
        except Exception as e:
            print(f'{f} left out: {e}'); return None

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
        if not torch.isfinite(n).all() or (n.max() / n.mean() - 1).item() < self.gate:  # no clear peak over the candidate shifts (atypical people in a one-person batch): leave the batch as it is
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
        if abs(c) < self.dead:
            return X
        if abs(c) > self.cap:
            c = 0 if self.reject else (self.cap if c > 0 else -self.cap)
        return X[:, :, (t - c).clamp(0, T - 1)] if c else X

    @torch.inference_mode()
    def predict(self, X):
        X = torch.nan_to_num(X.float()[:, self.order], nan=0.0, posinf=0.0, neginf=0.0)
        try:  # the alignment is a correction, half precision an acceleration: if either fails here, do without it
            X = self.align(X)
        except Exception:  # this batch only: nothing is remembered across predict() calls
            pass
        outs = []
        for i, member in enumerate(self.members):
            try:
                with torch.autocast(X.device.type, dtype=torch.float16, enabled=i in self.amp):
                    outs.append(member(X).float())
            except Exception:
                outs.append(None)
        # one synchronisation per batch: only if some output is missing or not finite, redo those members in full precision
        if any(P is None for P in outs) or not torch.isfinite(torch.stack([P.sum() for P in outs])).all():
            for i, member in enumerate(self.members):
                if outs[i] is None or not torch.isfinite(outs[i]).all():
                    try:
                        outs[i] = torch.nan_to_num(member(X).float(), nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        outs[i] = None  # unusable in this runtime: left out of this batch
        out = sum(w * P / P.norm(dim=1).mean().clamp_min(1e-12) for w, P in zip(self.weights, outs) if P is not None)
        return out if self.pow == 1 else out * out.norm(dim=1, keepdim=True).clamp_min(1e-12) ** (self.pow - 1)


class Solver(CompetSolver):
    name = "Neurobridge-ensemble"

    def load_model(self, meta):
        return Ensemble(meta["submission_dir"], meta["ch_names"], meta["device"])
