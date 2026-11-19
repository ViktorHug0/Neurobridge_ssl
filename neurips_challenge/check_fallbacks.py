"""Self-check of submission.py's fallbacks (run on a GPU node): a member that cannot be loaded is left out, a member whose fp16
output overflows is re-run in fp32, a NaN in the input does not reach the output.  .venv/bin/python neurips_challenge/check_fallbacks.py"""
import sys, types, tempfile, importlib.util, torch
from pathlib import Path
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object
sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
spec = importlib.util.spec_from_file_location('sub', Path(__file__).with_name('submission.py')); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
T = Path(tempfile.mkdtemp()); torch.manual_seed(0)


def mk(scale):
    l = torch.nn.Linear(32 * 120, 16)
    with torch.no_grad():
        l.weight.mul_(scale)
    return torch.jit.trace(torch.nn.Sequential(torch.nn.Flatten(), l), torch.randn(2, 32, 120))


mk(1.).save(str(T / 'good.pt')); mk(1e4).save(str(T / 'huge.pt')); (T / 'bad.pt').write_text('not a model'); ch = [f'c{i}' for i in range(32)]
torch.save({'members': ['good.pt', 'huge.pt', 'bad.pt'], 'channels': ch, 'amp': [0, 1], 'weights': [1., 1., 1.]}, T / 'members.pt')
e = S.Ensemble(T, ch[::-1], torch.device('cuda')); assert len(e.members) == 2 and e.amp == {0, 1}
x = torch.randn(64, 32, 120, device='cuda') * 50; x[3, 5, 7] = float('nan'); y = e.predict(x); assert torch.isfinite(y).all()
xx = torch.nan_to_num(x[:, e.order]); g, h = (torch.jit.load(str(T / f)).cuda() for f in ('good.pt', 'huge.pt'))
with torch.autocast('cuda', dtype=torch.float16):
    assert not torch.isfinite(h(xx)).all()  # the overflow really happens in fp16
assert torch.allclose(y, sum(P / P.norm(dim=1).mean() for P in (g(xx).float(), h(xx).float())), rtol=1e-2, atol=1e-3)
# alignment: an unloadable detector is dropped at load; a detector that fails when called skips the alignment for that batch only
aux = torch.load(T / 'members.pt'); aux.update(detector='bad.pt', shifts=list(range(-8, 9, 2)), bias=1.5, gate=0.1); torch.save(aux, T / 'members.pt')
assert S.Ensemble(T, ch, torch.device('cuda')).detector is None
torch.jit.trace(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16 * 120, 4)), torch.randn(2, 16, 120)).save(str(T / 'wrong.pt'))
aux['detector'] = 'wrong.pt'; torch.save(aux, T / 'members.pt'); e2 = S.Ensemble(T, ch[::-1], torch.device('cuda'))
assert e2.detector is not None and torch.equal(e2.predict(x), y) and e2.detector is not None  # same output as without alignment, nothing remembered
print('fallbacks OK')
