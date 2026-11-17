"""Package an ensemble as a Codabench Track-1 submission folder: neurips_challenge/submission.py, one TorchScript file
per member (encoder + EEG projector traced on CUDA with the unknown-subject id, as tta_cache.py runs them) and
members.pt (member files, training channel order, source checkpoints).
    package_submission.py <out_dir> <run_dir> [...] [--checkpoint checkpoint_test_best.pth]
    package_submission.py --verify <out_dir> <fold>   # the cv4 fold stream, served in a shuffled channel order, through
        # Solver.load_model + predict at batch 64 and 37: pooled subject-aggregated top-5 over disjoint 3-repeat groups on
        # the 200-image gallery (logs/sealed_ens_search.py's sealed-like metric, same groups)
"""
import argparse, json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.score_track1 import build_modules, build_test_dataset  # noqa: E402
from neurips_challenge.tta_cache import DATA, FOLDS, OUT  # noqa: E402
from module.eeg_encoder.model import run_eeg_backbone  # noqa: E402

CHANNELS = json.load(open(f'{DATA}/info.json'))['ch_names']
DEV = torch.device('cuda:0')


class Member(torch.nn.Module):
    def __init__(self, model, eeg_projector, cfg):
        super().__init__(); self.model, self.eeg_projector, self.cfg = model, eeg_projector, cfg

    def forward(self, x):
        return self.eeg_projector(run_eeg_backbone(self.model, self.cfg, x, torch.full((x.shape[0],), 10 ** 6, device=x.device)))


@torch.no_grad()
def export(out, run_dirs, checkpoint, detector=False, bias=1, gate=0.0, amp=()):
    out.mkdir(parents=True)
    x = torch.tensor(np.load(f'{DATA}/sub-01/test.npy', mmap_mode='r')[:64, 0, 0]).to(DEV)  # (64, 32, 120) real trials
    files = []
    for i, d in enumerate(run_dirs):
        cfg = SimpleNamespace(**json.load(open(f'{d}/train_config.json'))); cfg.eeg_data_dir = DATA
        assert cfg.selected_channels == CHANNELS and list(cfg.time_window) == [0, 120], d
        ckpt = torch.load(f'{d}/{checkpoint}', map_location=DEV, weights_only=False)
        model, eproj, _ = build_modules(cfg, ckpt, build_test_dataset(cfg, [1]), DEV, 120)
        for m in model.modules():  # braindecode metadata properties raise ValueError when unset, and trace hasattr()s them
            if getattr(m, '_sfreq', 0) is None:
                m._sfreq = 120.0
            if getattr(m, '_chs_info', 0) is None:
                m._chs_info = [{'ch_name': c} for c in CHANNELS]
        eager = Member(model, eproj, cfg).eval()
        traced = torch.jit.trace(eager, x)
        for b in (64, 37, 1):  # the trace must not bake in the batch size
            err = (traced(x[:b]) - eager(x[:b])).abs().max().item() / eager(x[:b]).abs().max().item()
            assert err < 1e-3, f'{d}: traced != eager at batch {b} (rel err {err:.2e})'
        files.append(f'member{i}_{cfg.output_name}.pt'); traced.save(str(out / files[-1]))
        print(f'{files[-1]} <- {d}/{checkpoint}', flush=True)
        del model, eproj, eager, traced; torch.cuda.empty_cache()
    aux = {'members': files, 'channels': CHANNELS, 'sources': [f'{d}/{checkpoint}' for d in run_dirs]}
    if detector:  # the LAST run dir is the latency detector of submission.py align(), not an ensemble member
        aux.update(members=files[:-1], sources=aux['sources'][:-1], detector=files[-1], detector_source=aux['sources'][-1], shifts=list(range(-8, 9, 2)), bias=bias, gate=gate)
    if amp:  # indices of the members submission.py runs under fp16 autocast
        aux['amp'] = list(amp)
    torch.save(aux, out / 'members.pt')
    shutil.copy(Path(__file__).with_name('submission.py'), out / 'submission.py')


@torch.no_grad()
def verify(out, k, delay=0):
    import importlib.util, types
    stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object  # benchopt is not in this venv:
    sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})  # check_submission.sh runs the real one
    spec = importlib.util.spec_from_file_location('submission', out / 'submission.py')
    sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
    c = np.load(f'{OUT}/fold{k}.npz')
    raw = {s: np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120) for s in FOLDS[k]}
    X = torch.as_tensor(np.stack([raw[s][i] for s, i in zip(c['subj'], c['idx'])]))
    served = list(np.random.default_rng(0).permutation(CHANNELS))  # the grader's order differs from training's
    X = X[:, [CHANNELS.index(ch) for ch in served]]
    if delay:
        X = X[:, :, (torch.arange(120) - delay).clamp(0, 119)]
    model = sub.Solver.load_model(None, {'submission_dir': out, 'ch_names': served, 'device': DEV})
    groups = {}
    for i, key in enumerate(zip(c['subj'], c['img'])):
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(k)
    idx = torch.as_tensor(np.concatenate([rng.permutation(v)[:len(v) // 3 * 3].reshape(-1, 3) for v in groups.values()]))
    t = torch.as_tensor(c['img'][idx[:, 0].numpy()])
    n = lambda x: x / x.norm(dim=1, keepdim=True)
    gal = n(torch.as_tensor(c['G']).float())
    for bs in (64, 37):
        y = torch.cat([model.predict(X[b:b + bs].to(DEV)).float().cpu() for b in range(0, len(X), bs)])
        q = n(y[idx].mean(1))  # the scorer averages raw predictions within (participant, image), then normalises
        r = (lambda S: S > S.gather(1, t[:, None]))(q @ gal.T).sum(1)  # target scored in the same matmul (no rounding self-count)
        print(f'fold {k}, batch {bs}: 3-repeat subject-aggregated top-5 {100 * (r < 5).float().mean():.2f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path); ap.add_argument('runs', nargs='*')
    ap.add_argument('--checkpoint', default='checkpoint_test_best.pth')
    ap.add_argument('--verify', type=int, help='cv4 fold to replay through the packaged Solver')
    ap.add_argument('--detector', action='store_true', help='the last run dir is the latency detector (submission.py align()), not a member')
    ap.add_argument('--bias', type=float, default=1, help='calibration constant of the alignment (scripts/mi/nr05_autolag.py: the norm of rmix peaks one sample early on new Alljoined people)')
    ap.add_argument('--gate', type=float, default=0.0, help='alignment acts only if the detector curve peaks at least this far above its mean (scripts/mi S06b: 0.10)')
    ap.add_argument('--amp', type=int, nargs='*', default=[], help='indices (order of the run dirs) of the members run in half precision')
    ap.add_argument('--delay', type=int, default=0, help='--verify: delay the replayed stream by this many samples (tests the alignment)')
    a = ap.parse_args()
    if a.runs:
        export(a.out, a.runs, a.checkpoint, a.detector, a.bias, a.gate, a.amp)
    if a.verify:
        verify(a.out, a.verify, a.delay)
