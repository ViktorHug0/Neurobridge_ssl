"""Shape/param/speed self-check for the orthogonal encoders.

Run: .venv/bin/python ensemble_experiments/architectures/smoke.py [--cuda] [ENCODER ...]
Asserts every encoder maps (B, 63, 250) -> (B, 1024) with finite grads.
"""
import sys, time
import torch

sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from ensemble_experiments.architectures.ortho_encoders import _REGISTRY, build_ortho_encoder

BB, C, T = 1024, 63, 250


def main():
    cuda = '--cuda' in sys.argv
    dev = 'cuda' if cuda else 'cpu'
    B = 256 if cuda else 8
    requested = [arg for arg in sys.argv[1:] if arg != '--cuda']
    names = requested or list(_REGISTRY)
    unknown = sorted(set(names) - set(_REGISTRY))
    if unknown:
        raise ValueError(f"unknown encoders: {unknown}")
    x = torch.randn(B, C, T, device=dev)
    print(f"{'encoder':16s} {'params':>10s} {'ms/step':>9s}")
    for name in names:
        m = build_ortho_encoder(name, BB, T, C).to(dev)
        y = m(x)
        assert y.shape == (B, BB), f"{name}: {tuple(y.shape)}"
        y.square().mean().backward()
        gs = [p.grad for p in m.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gs), f"{name}: bad grad"
        n = sum(p.numel() for p in m.parameters())
        reps = 5 if cuda else 1
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            m.zero_grad(set_to_none=True)
            m(x).square().mean().backward()
        if cuda:
            torch.cuda.synchronize()
        print(f"{name:16s} {n:10,d} {(time.time()-t0)/reps*1e3:9.1f}")
    print("OK")


if __name__ == '__main__':
    main()
