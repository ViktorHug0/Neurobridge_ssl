import json, time, torch, sys
sys.path.insert(0, '/nasbrain/p20fores/Neurobridge_SSL')
from module.eeg_encoder.foundation import FM_N_TIMES
from braindecode.models import REVE
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']
for hub in ('brain-bzh/reve-base', 'brain-bzh/reve-large'):
    fm = REVE.from_pretrained(hub, n_outputs=2, n_chans=32, n_times=FM_N_TIMES, chs_info=[{'ch_name': c} for c in CH]).cuda()
    n = sum(p.numel() for p in fm.parameters()) / 1e6; opt = torch.optim.AdamW(fm.parameters(), lr=1e-4)
    x = torch.randn(64, 32, FM_N_TIMES, device='cuda'); torch.cuda.reset_peak_memory_stats()
    for i in range(6):
        if i == 2: torch.cuda.synchronize(); t = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = fm(x, return_output=True) if 'return_output' in fm.forward.__code__.co_varnames else fm(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        y.float().pow(2).mean().backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    print(hub, '%.0f M params' % n, '| %.0f ms per step (batch 64)' % ((time.time() - t) / 4 * 1000), '| peak %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    del fm, opt, x, y; torch.cuda.empty_cache()
