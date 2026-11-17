"""Mechanistically targeted test-time adaptation: adapt ONE small subject-specific component by self-training on the
transductive assignment (pseudo-labels = argmax of full_sattc on the 200 unlabelled test trials, refreshed every
round); everything else frozen. Labels are used for reporting only.
  MODE=att    ATM: the electrode attention map (4 x 64 x 64 logits, initialised from the map of the subject's key)
  MODE=inmix  any encoder: a 63 x 63 electrode re-mixing of the input (identity at init)
  MODE=intime control: a temporal FIR filter per electrode at the input (63 x 21 taps)
  MODE=out    control: a linear map on the output embedding (identity at init)
  HEAD=gala   pseudo-labels and reported score from CVPR sage_plus with the subject's cross-fold parameters"""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO
from e13_atm_attention import emb, STATE
from module.transductive import infer
from module.eeg_encoder.model import run_eeg_backbone

ARM = sys.argv[1] if len(sys.argv) > 1 else 'atm128/seed3300'
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(1, 11)
ROUNDS, STEPS, LR, TAU = int(os.environ.get('ROUNDS', 12)), 10, float(os.environ.get('LR', 0.05)), 20.0
MODE, HEAD = os.environ.get('MODE', 'att'), os.environ.get('HEAD', 'sattc')
NQ = int(os.environ.get('NQ', 200))  # number of test trials available (the first NQ images); the gallery keeps all 200
NDIS = int(os.environ.get('NDIS', 0))  # open gallery: add NDIS distractor images (training images) to the 200 candidates
HEADNAME = os.environ.get('HEADNAME', 'full_sattc')  # module.transductive.infer method used as the head
PL = os.environ.get('PL', 'head')  # pseudo-label source: head (the transductive head) | plain (plain cosine argmax)
nz = torch.nn.functional.normalize
TARGET = torch.arange(200)
if DEV == 'cuda':  # shares the GPU with a training run: never take more than ~2 GB
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEMFRAC', 0.22)))
if HEAD == 'gala':
    sys.path.insert(0, os.path.join(REPO, 'CVPR_paper/scripts'))
    from sage_plus import sage_plus
    from stream_grid import subject_params
    PARAMS = os.path.join(REPO, os.environ['PARAMS'])


def scores(e, g, fold):
    e, g = e.detach().cpu().numpy(), g.cpu().numpy()
    S = sage_plus(e, g, subject_params(PARAMS, fold)) if HEAD == 'gala' else infer(e, g, HEADNAME)
    S = torch.as_tensor(np.asarray(S))
    return 100 * (S.argmax(1) == TARGET[:len(S)]).float().mean().item(), S


if __name__ == '__main__':
    rows = []
    for fold in FOLDS:
        m = GM(glob.glob(f'{LOSO}/{ARM}/*-sub-{fold:02d}')[0])
        g = img_emb(m).to(DEV)
        if NDIS:
            dis = img_emb(m, 'train')[torch.randperm(16540, generator=torch.Generator().manual_seed(0))[:NDIS]]
            g = torch.cat([g, dis.to(DEV)])
        if os.environ.get('PERM'):  # sanity check: shuffle the gallery order, so that no step can profit from query i <-> gallery i
            perm = torch.randperm(200, generator=torch.Generator().manual_seed(fold))
            g, TARGET = g[perm.to(DEV)], torch.argsort(perm)
        x = T(eeg(fold))[:NQ].to(DEV)
        sid = torch.full((NQ,), fold, dtype=torch.long, device=DEV)
        base = lambda xin: m.eeg_proj(run_eeg_backbone(m.enc, m.args, xin, sid))
        if MODE in ('att', 'att+inmix'):
            P = torch.log(emb(m, x.mean(0, keepdim=True).cpu(), fold)[1] + 1e-8).clone().requires_grad_(True)
            Q = torch.zeros(63, 63, device=DEV, requires_grad=MODE == 'att+inmix')

            def forward():
                STATE.update(rec=None, force=torch.softmax(P, -1), pos=0)
                e = base(x + torch.einsum('bct,cd->bdt', x, Q))
                STATE.update(force=None)
                return e
        elif MODE == 'inmix':
            P = torch.zeros(63, 63, device=DEV, requires_grad=True)
            forward = lambda: base(x + torch.einsum('bct,cd->bdt', x, P))
        elif MODE == 'intime':  # control: one temporal FIR filter per electrode (63 x 21 taps, identity at init)
            P = torch.zeros(63, 1, 21, device=DEV, requires_grad=True)
            forward = lambda: base(x + torch.nn.functional.conv1d(x, P, padding=10, groups=63))
        else:
            with torch.no_grad():
                e0 = base(x)
            P = torch.zeros(e0.shape[1], e0.shape[1], device=DEV, requires_grad=True)
            forward = lambda: e0 + e0 @ P
        opt = torch.optim.Adam([{'params': [P]}] + ([{'params': [Q], 'lr': 0.003}] if MODE == 'att+inmix' else []), lr=LR)
        for r in range(ROUNDS + 1):
            with torch.no_grad():
                e = forward()
            tta, S = scores(e, g, fold)
            rows.append(dict(arm=ARM, mode=MODE, fold=fold, round=r, tta=tta, plain=100 * ((nz(e, dim=1) @ g.T).argmax(1).cpu() == TARGET[:NQ]).float().mean().item()))
            print(rows[-1], flush=True)
            if r == ROUNDS:
                break
            pl = (S.argmax(1).to(DEV) if PL == 'head' else (nz(e, dim=1) @ g.T).argmax(1))
            for _ in range(STEPS):
                loss = torch.nn.functional.cross_entropy(TAU * nz(forward(), dim=1) @ g.T, pl)
                opt.zero_grad(); loss.backward(); opt.step()
        pd.DataFrame(rows).to_csv(f'{OUT}/e20_att_tta{os.environ.get("TAG", "")}.csv', index=False)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index='round', columns='fold', values='tta').loc[[0, ROUNDS // 2, ROUNDS]].round(1).to_string())
    print(df.groupby('round')[['tta', 'plain']].mean().round(2).T.to_string())
