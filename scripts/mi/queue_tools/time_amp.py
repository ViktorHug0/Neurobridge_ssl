"""time_amp.py <package dir>: per-member time of a package's predict at fp32 and under fp16 / bf16 autocast, and the output change."""
import sys, time, json, numpy as np, torch
from pathlib import Path
from neurips_challenge.tta_cache import DATA
out = Path(sys.argv[1]); aux = torch.load(out / 'members.pt', weights_only=True)
x = torch.tensor(np.load(f'{DATA}/sub-01/test.npy', mmap_mode='r')[:64, 0, 0]).cuda()
def t(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); s = time.time()
    for _ in range(n): y = f()
    torch.cuda.synchronize(); return 1000 * (time.time() - s) / n, y.float()
tot = np.zeros(3)
with torch.inference_mode():
    for f in aux['members']:
        m = torch.jit.load(str(out / f), map_location='cuda').eval(); r = []
        a, ya = t(lambda: m(x)); r.append(a); msg = ''
        for dt in (torch.float16, torch.bfloat16):
            def g():
                with torch.autocast('cuda', dtype=dt): return m(x)
            try: b, yb = t(g); r.append(b); msg += ' | %s %.1f ms, cos %.5f' % (str(dt)[6:], b, torch.cosine_similarity(ya, yb).mean())
            except Exception as e: r.append(a); msg += ' | %s FAILED %s' % (str(dt)[6:], str(e)[:60])
        tot += r; print('%-18s fp32 %.1f ms%s' % (f, a, msg), flush=True); del m
print('members total: fp32 %.1f, fp16 %.1f, bf16 %.1f ms' % tuple(tot))
