"""check_stateless.py <package dir>: predict() of a real package keeps nothing across calls. The same batch gives the same output
on its first call, after other batches (one of them delayed, one of another size) and after itself."""
import sys, types, json, importlib.util, numpy as np, torch
from pathlib import Path
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object
sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
from neurips_challenge.tta_cache import DATA
out = Path(sys.argv[1]); spec = importlib.util.spec_from_file_location('submission', out / 'submission.py'); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
m = sub.Solver.load_model(None, {'submission_dir': out, 'ch_names': json.load(open(f'{DATA}/info.json'))['ch_names'], 'device': torch.device('cuda:0')})
x = torch.tensor(np.load(f'{DATA}/sub-01/test.npy', mmap_mode='r')[:200, 0, 0]).cuda(); A = x[:64]
before = {k: v for k, v in vars(m).items() if k not in ('members', 'detector', 'reader')}
y1 = m.predict(A); m.predict(x[64:128, :, (torch.arange(120, device='cuda') - 4).clamp(0, 119)]); m.predict(x[128:165]); y2 = m.predict(A); y3 = m.predict(A)
same = all(before[k] is v or (torch.equal(before[k], v) if torch.is_tensor(v) else before[k] == v) for k, v in vars(m).items() if k in before)
print('first call = later calls: %s / %s (max rel. diff %.1e) | attributes untouched: %s' % (torch.equal(y1, y2), torch.equal(y2, y3), ((y1 - y2).norm() / y1.norm()).item(), same))
assert torch.equal(y1, y2) and torch.equal(y2, y3) and same
