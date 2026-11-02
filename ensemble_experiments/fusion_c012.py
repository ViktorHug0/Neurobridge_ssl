"""Matched C0/C1/C2 follow-up; reuse the unchanged G2 recipe and checkpoint policy."""
import argparse
import hashlib
import json
import fcntl
from pathlib import Path
import torch
from torch import nn
from ensemble_experiments.fusion_three import train as trainer
from ensemble_experiments.fusion_three.models import Decoder, CorrectedLinear
from ensemble_experiments.full_sharing_v2.models import FullPair, seeded

ARMS = ('C0', 'C1', 'C2')
DESCRIPTIONS = {
    'C0': 'Matched F4 control: private dense readouts, G2 validation/selection recipe',
    'C1': 'G2 rank16 first dense corrections; fully private residual dense layers',
    'C2': 'G2 with rank32 private corrections on both dense layers',
}
OUTPUT = trainer.REPO/'results/things_eeg/fusion_c012_20260915'
original_hashes = trainer.source_hashes


class CDecoder(Decoder):
    def __init__(self, variant):
        nn.Module.__init__(self)
        assert variant in ARMS
        self.arm = 'G2'  # Inherit the common F4 two-path forward/loss implementation.
        self.variant = variant
        self.pair = FullPair('F4')
        if variant != 'C0':
            first = self.pair.readout_a[0]
            residual = self.pair.readout_a[1].f[1]
            rank = 32 if variant == 'C2' else 16
            for j, readout in enumerate((self.pair.readout_a, self.pair.readout_b)):
                readout[0] = seeded(5100+j, lambda: CorrectedLinear(first, rank))
                if variant == 'C2':
                    readout[1].f[1] = seeded(5200+j, lambda: CorrectedLinear(residual, rank))

    def shared_blocks(self):
        blocks = dict(stem=list(self.pair.stem_a.parameters()))
        if self.variant != 'C0':
            blocks['readout'] = list(self.pair.readout_a[0].shared.parameters())
            if self.variant == 'C2':
                blocks['readout'] += list(self.pair.readout_a[1].f[1].shared.parameters())
        return blocks


def source_hashes():
    hashes = original_hashes()
    path = Path(__file__).resolve()
    hashes[str(path.relative_to(trainer.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def summarize():
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(p.read_text()) for p in sorted((OUTPUT/'runs').glob('C*/sub-*/complete.json'))]
        means = {}
        for arm in ARMS:
            subset = [r for r in rows if r['arm'] == arm]
            if subset:
                means[arm] = dict(n=len(subset), top1=sum(r['top1'] for r in subset)/len(subset))
        trainer.write_json(OUTPUT/'summary.json', dict(completed=len(rows), expected=15, means=means, folds=rows))


def preflight():
    torch.manual_seed(3300)
    eeg = torch.randn(4,63,250,device='cuda')
    subjects = torch.arange(1,5,device='cuda')
    images = torch.randn(4,6400,device='cuda')
    reference = Decoder('G2').cuda().eval()
    with torch.no_grad():
        expected = reference.encode(eeg, subjects)
    sizes = {}
    for arm in ARMS:
        model = CDecoder(arm).cuda().eval()
        with torch.no_grad():
            actual = model.encode(eeg, subjects)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert a.shape == (4,128)
        p = model.pair
        assert p.stem_a is p.stem_b
        if arm == 'C0':
            assert p.readout_a[0] is not p.readout_b[0]
        else:
            assert p.readout_a[0].shared is p.readout_b[0].shared
        if arm == 'C1':
            assert p.readout_a[1].f[1] is not p.readout_b[1].f[1]
        model.train()
        losses = model.losses(model(eeg,images,subjects),torch.eye(4,device='cuda',dtype=torch.bool))
        trainer.gradients(model, losses)
        torch.stack(losses).mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        sizes[arm] = model.size()
        print('PREFLIGHT',arm,sizes[arm],flush=True)
        del model
    trainer.write_json(OUTPUT/'preflight/passed.json',dict(passed=True,sources=source_hashes(),sizes=sizes,
        checks=['exact initial G2/F4 equivalence','sharing aliases','128D outputs','GPU backward and shared gradient diagnostics']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    trainer.Decoder = CDecoder
    trainer.DESCRIPTIONS = DESCRIPTIONS
    trainer.OUTPUT = OUTPUT
    trainer.ARMS = ARMS
    trainer.source_hashes = source_hashes
    trainer.summarize = summarize
    trainer.deterministic()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        preflight()
    else:
        assert args.arm
        if args.arm == 'C2':
            previous = trainer.REPO/'results/things_eeg/fusion_three_20260915/runs/G2'
            assert all((previous/f'sub-{s:02d}/complete.json').exists() for s in range(1,6)), 'G2 must finish first'
        trainer.worker(args.arm)
