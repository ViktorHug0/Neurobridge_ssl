"""P04 last-layer retraining on people the body never saw (DFR, Kirichenko et al. 2023, applied to M02).
M02: a trained person's advantage is a private code read by the late feed-forward layers and the head. Those layers were
fitted on trained people only, so they have never read "a new person seen through this body". Test: keep the fold model's
body frozen and retrain its late part on 3 of the fold's 5 HELD-OUT people (their 11,540 trained images, same loss), then
score the 2 other held-out people on never-trained images (5k, r=3; and the single-trial P(target > random image)).
Control: the same retraining on 3 people the body WAS trained on (more steps on few people, nothing new for the head).
    part: head | last (block 5 + head) | ff (feed-forward of blocks 3-5 + head) | all (full fine-tune, control)
    np04_late_retrain.py <recipe> <fold> <checkpoint> [parts, default head,last,ff,all]"""
import copy
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]; parts = (sys.argv[4] if len(sys.argv) > 4 else 'head,last,ff,all').split(',')
f = load(recipe, k, ck); m = f.model; init = copy.deepcopy(m.state_dict()); gal = gallery(); L = len(m.blocks)
E = nz(torch.tensor(np.asarray(TARGETS)).float()).to(DEV); tr = np.setdiff1d(np.arange(16540), HELD)
NEW = FOLDS[k]; OLD = FOLDS[k % 4 + 1][:3]; STEPS = (100, 300, 1000, 2000)
SEL = {'head': lambda n: n.startswith('head'), 'last': lambda n: n.startswith(('head', f'blocks.{L - 1}.')),
       'ff': lambda n: n.startswith('head') or any(n.startswith(f'blocks.{b}.ff') for b in range(L - 3, L)), 'all': lambda n: True}


def evaluate(cells):
    out = {}
    for name, (X, subj, img) in cells.items():
        y = f(X, bs=1000); sc, h = score(y, subj, img, gal, hits=True)
        out[name] = (sc['r3'], h['r3'], 1 - rank(scaled(y), img, gal).float().mean().item() / 4999)
    return out


def retrain(people, part, cells, base):
    m.load_state_dict(init); X, _, img = cell(people, tr); X = torch.as_tensor(X); tgt = torch.as_tensor(tr[img])
    for n, p in m.named_parameters():
        p.requires_grad_(SEL[part](n))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05); g = torch.Generator().manual_seed(0)
    for step in range(1, STEPS[-1] + 1):
        i = torch.randint(0, len(X), (128,), generator=g)
        y = f.eproj(run_eeg_backbone(m, f.cfg, X[i].to(DEV), torch.full((128,), 10 ** 6, device=DEV))).float()
        loss = torch.nn.functional.cross_entropy(y @ E[tgt[i]].T, torch.arange(128, device=DEV))  # EEG -> image only, unit-norm targets, scale 1: the members' loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step in STEPS:
            ev = evaluate(cells)
            print('    %-5s on %-12s step %4d loss %.3f | ' % (part, people, step, loss.item()) + ' | '.join('%s top-5 %.3f (%+.1f +- %.1f %%), P %.4f (%+.4f)' % (n, v[0], *paired(v[1], base[n][1]), v[2], v[2] - base[n][2]) for n, v in ev.items()), flush=True)
    for p in m.parameters():
        p.requires_grad_(False)


for A, B in ((NEW[:3], NEW[3:]), (NEW[2:], NEW[:2])):
    cells = {'new people %s' % B: cell(B)}; m.load_state_dict(init); base = evaluate(cells)
    print('retrain on %s / test on %s | intact: ' % (A, B) + ' | '.join('%s top-5 %.3f, P(target > random) %.4f' % (n, v[0], v[2]) for n, v in base.items()), flush=True)
    for part in parts:
        retrain(A, part, cells, base)
        retrain(OLD, part, cells, base)  # control: people the body was trained on
