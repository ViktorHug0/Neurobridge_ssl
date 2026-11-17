import sys, time, types, importlib.util, json, numpy as np, torch
from pathlib import Path
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object
sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
DATA = json.load(open('/nasbrain/p20fores/Neurobridge_SSL/neurips_challenge/submissions/.data_path')) if 0 else None
from neurips_challenge.tta_cache import DATA
ch = json.load(open(f'{DATA}/info.json'))['ch_names']; x = torch.tensor(np.load(f'{DATA}/sub-01/test.npy', mmap_mode='r')[:64, 0, 0]).cuda()
for out in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location('submission', Path(out) / 'submission.py'); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
    m = sub.Solver.load_model(None, {'submission_dir': Path(out), 'ch_names': ch, 'device': torch.device('cuda:0')})
    with torch.no_grad():
        for _ in range(10): m.predict(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(100): m.predict(x).cpu()
        torch.cuda.synchronize(); print(out.split('/')[-1], '%.1f ms per 64-batch' % (10 * (time.time() - t)), flush=True)
    del m; torch.cuda.empty_cache()
