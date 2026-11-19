"""time_align.py <old package> <new package>: ms per 64-batch of two packages and of the new package's alignment step alone, in
alternating blocks on the same GPU (one package loaded at a time: the shared 10 GB card cannot hold two next to a training)."""
import sys, types, json, time, importlib.util, numpy as np, torch
from pathlib import Path
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object
sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
from neurips_challenge.tta_cache import DATA
ch = json.load(open(f'{DATA}/info.json'))['ch_names']; x = torch.tensor(np.load(f'{DATA}/sub-01/test.npy', mmap_mode='r')[:64, 0, 0]).cuda()
def load(d):
    spec = importlib.util.spec_from_file_location('s%d' % time.time_ns(), Path(d) / 'submission.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Solver.load_model(None, {'submission_dir': Path(d), 'ch_names': ch, 'device': torch.device('cuda:0')})
def t(f, n=30):
    with torch.inference_mode():
        for _ in range(5): f()
        out = []
        for _ in range(n):
            torch.cuda.synchronize(); s = time.time(); f(); torch.cuda.synchronize(); out.append(1000 * (time.time() - s))
    return out
R = {'old predict': [], 'new predict': [], 'new alignment step': []}
for _ in range(2):
    m = load(sys.argv[1]); R['old predict'] += t(lambda: m.predict(x).cpu()); del m; torch.cuda.empty_cache()
    m = load(sys.argv[2]); xo = x.float()[:, m.order]; R['new predict'] += t(lambda: m.predict(x).cpu()); R['new alignment step'] += t(lambda: m.align(xo)); del m; torch.cuda.empty_cache()
for k, v in R.items(): print('%-20s median %.1f ms (25-75 %%: %.1f-%.1f)' % (k, np.median(v), *np.percentile(v, [25, 75])), flush=True)
print('alignment share of the new predict: %.0f %%; new / old: %.3f' % (100 * np.median(R['new alignment step']) / np.median(R['new predict']), np.median(R['new predict']) / np.median(R['old predict'])))
