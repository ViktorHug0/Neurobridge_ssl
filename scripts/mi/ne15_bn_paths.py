"""E15 the two BatchNorm paths of the SQF members (per-dataset statistics): new Alljoined people decoded through the
Alljoined statistics (deployed), through the THINGS statistics, and both averaged; without and with the THINGS lag.
How far apart are the two paths once the datasets are time-aligned? fold 1, never-trained images, 5k, exact ranks."""
from nc_lib import *
F = f'{R}/logs/holdout/f1'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']; X = np.load(f'{F}/X.npy', mmap_mode='r'); gal = gallery()
for name in ('sqf_dbn_ho5k_long_optx_things', 'mi_sqfx_ho5k_lag5'):
    f = load(name, 1)
    @torch.no_grad()
    def run(sid):
        return torch.cat([f.eproj(run_eeg_backbone(f.model, f.cfg, torch.as_tensor(np.asarray(X[b:b + 512])).to(DEV), torch.full((len(X[b:b + 512]),), sid, device=DEV))).float().cpu() for b in range(0, len(X), 512)])
    ya, yt = scaled(run(10 ** 6)), scaled(run(21))
    (sa, ha), (s2, h2), (sb, hb) = score(ya, subj, img, gal, hits=True), score(yt, subj, img, gal, hits=True), score(ya + yt, subj, img, gal, hits=True)
    bn = [(m.primary, m.auxiliary) for m in f.model.modules() if hasattr(m, 'auxiliary')]
    d = np.mean([((p.running_mean - a.running_mean).norm() / p.running_mean.norm().clamp_min(1e-6)).item() for p, a in bn]); v = np.mean([(p.running_var / a.running_var).log().abs().mean().item() for p, a in bn])
    print('%-32s Alljoined stats %s | THINGS stats %s (%+.1f +- %.1f %%) | both averaged %s (%+.1f +- %.1f %%) | cos(outputs) %.3f | %d BN layers: mean shift %.2f, |log var ratio| %.2f' % (
        name, ' / '.join('%.3f' % x for x in sa.values()), ' / '.join('%.3f' % x for x in s2.values()), *paired(h2['r3'], ha['r3']), ' / '.join('%.3f' % x for x in sb.values()), *paired(hb['r3'], ha['r3']),
        (nz(ya[:20000]) * nz(yt[:20000])).sum(1).mean(), len(bn), d, v), flush=True)
    del f; torch.cuda.empty_cache()
