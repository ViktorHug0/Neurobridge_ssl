"""E3.1: SATTC's own test-time calibration (their `evaluate_model`, SAW -> Ada-CSLS -> structural expert -> PoE)
on precomputed query/gallery embeddings. Nothing of theirs is re-ported: we import run_sattc_loso.py and call it.

  validate  their saved sub-XX_final.pth + their test set; evaluate_model on the real model, then again through the
            pass-through wrapper on the embeddings it produced; both must equal their logged final-epoch row.
  ours      atm31_valcon (or --run_dir) 80-repetition queries; plain cosine and SATTC per subject -> CSV.

Run from anywhere with PYTHONPATH=$SATTC/.deps (wandb stub):
  .venv/bin/python CVPR_paper/scripts/sattc_on_embeddings.py validate --subjects sub-01
  .venv/bin/python CVPR_paper/scripts/sattc_on_embeddings.py ours --output_csv results/.../on_atm31_valcon.csv
"""
import argparse, copy, csv, glob, os, sys
from types import SimpleNamespace

import numpy as np
import torch

SATTC = '/nasbrain/p20fores/SATTC-CVPR2026'
RUN_DIR = SATTC + '/Retrieval/models/contrast/run_sattc_loso_sattc/across/SATTC'  # one timestamp dir per fold
DATA = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
LOG_DIR = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/cvpr/competitors/sattc/own_conditions_seed3300/run_sattc_loso_sattc'
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(SATTC, 'Retrieval'))
import run_sattc_loso as R  # noqa: E402


class _Grab(Exception):
    pass


def sattc_config(seed=3300):
    """Their final-stage config: their argparse + normalisation (run up to set_seed), then lines 3209-3221/3613-3623."""
    def grab(_seed):  # ponytail: frame grab, because their parser lives inside _run_training_pipeline
        raise _Grab(sys._getframe(1).f_locals['args'])
    orig, R.set_seed = R.set_seed, grab
    argv, sys.argv = sys.argv, ['run_sattc_loso.py', '--seed', str(seed), '--data_path', DATA]  # as sattc_repro.sbatch
    try:
        R._run_training_pipeline()
        raise RuntimeError('set_seed was not reached')
    except _Grab as g:
        args = g.args[0]
    finally:
        R.set_seed, sys.argv = orig, argv
    orig(seed)
    ids = [R.extract_id_from_string(s) for s in args.subjects]
    base_slots = max(len(args.subjects), max(ids) + 1)
    use_unk = args.test_subject_ids in {'unk_avg', 'unk_learned'} or args.subject_dropout_p > 0
    args.use_subject_unk, args._subject_slots = use_unk, base_slots + (1 if use_unk else 0)
    cfg = copy.deepcopy(args)
    cfg.stage = 'final'
    cfg.enable_mnn_precheck = False  # diagnostic only (writes plots/CSVs, never touches the logits)
    return cfg


def prime_whitening(cfg, train_text, train_img):
    """What train_model's first call caches on the config: global ZCA stats of the TRAIN-split candidates."""
    R._resolve_auto_whitening_state(cfg, 'text', train_text.float())
    R._resolve_auto_whitening_state(cfg, 'img', train_img.float())


class PassThrough(torch.nn.Module):
    """evaluate_model reads forward(x, subject_ids), .logit_scale and .loss_func; there is no .encoder."""
    def __init__(self, logit_scale=None):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.zeros([]) if logit_scale is None else logit_scale.detach().clone())
        self.loss_func = R.ClipLoss()

    def forward(self, x, subject_ids=None):
        return x


def sattc_eval(cfg, sub, q, g, text_g, model=None, device='cpu'):
    """One 200-way evaluate_model call; the loader is one batch in their test tuple format."""
    labels = torch.arange(len(q))
    loader = [(torch.as_tensor(q).float(), labels, None, torch.as_tensor(text_g).float(), None, torch.as_tensor(g).float())]
    _, res, _ = R.evaluate_model(sub, (model or PassThrough()).to(device), loader, torch.device(device), torch.as_tensor(text_g).float(),
                                 torch.as_tensor(g).float(), [200], cfg)
    return 100 * res[200]['top1'], 100 * res[200]['top5']


def validate(subjects, device='cpu'):
    from torch.utils.data import DataLoader
    feats = torch.load(os.path.join(SATTC, 'features/ViT-H-14_features_train.pt'), map_location='cpu')
    for sub in subjects:
        (ckpt,) = glob.glob(os.path.join(RUN_DIR, '*', f'{sub}_final.pth'))
        log = list(csv.DictReader(open(os.path.join(LOG_DIR, f'final_SATTC_{sub}.csv'))))[-1]
        cfg = sattc_config()
        prime_whitening(cfg, feats['text_features'], feats['img_features'])
        model = R.SATTC(num_subjects=cfg._subject_slots, use_subject_unk=cfg.use_subject_unk,
                        encoder_choice=cfg.chose_eeg_encoder)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        ds = R.EEGDataset(cfg.data_path, subjects=[sub], train=False)
        loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False)
        seen = []
        hook = model.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
        model.to(device)
        _, res, _ = R.evaluate_model(sub, model, loader, torch.device(device), ds.text_features, ds.img_features,
                                     [200], copy.deepcopy(cfg))
        hook.remove()
        real = (100 * res[200]['top1'], 100 * res[200]['top5'])
        wrapped = sattc_eval(copy.deepcopy(cfg), sub, torch.cat(seen), ds.img_features, ds.text_features, model=PassThrough(model.logit_scale), device=device)
        logged = (100 * float(log['test_accuracy']), 100 * float(log['top5_acc']))
        print(f'{sub} epoch {log["epoch"]}: logged {logged}  their-model {real}  wrapper {wrapped}', flush=True)


def ours(run_dir, output_csv, subjects, gallery_whiten=True):
    sys.path.insert(0, os.path.join(REPO, 'scripts/things_eeg/tta_rebuttal'))
    from shared import encode_repetition_blocks, load_subject_context
    os.environ.setdefault('SAGE_CHECKPOINT_NAME', 'checkpoint_testsel_best.pth')
    rows = []
    for sid in subjects:
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, eval_args, dataset, modules = load_subject_context(os.path.abspath(run_dir), rt, sid, average=False)
        (q,), g, _ = encode_repetition_blocks(eval_args, modules, dataset, block_sizes=[80], seed=3300)
        # their whitening stats come from the train-split candidates: our image projector on all 16,540 train images
        raw = np.load(os.path.join(eval_args.image_feature_dir, 'image_train.npy')).astype(np.float32)
        with torch.no_grad():
            g_train = modules['img_projector'](torch.from_numpy(raw.reshape(-1, *raw.shape[2:]))).float()  # [N*10, D] or SAMGA [N*10, K, D]
        cfg = sattc_config()
        prime_whitening(cfg, g_train, g_train)  # text branch never enters the ranking; given the image stats
        # their encoder is trained against whitened targets, ours is not: the paper reports SATTC with this off (2026-10-25)
        cfg.global_auto_whiten = gallery_whiten
        s = torch.nn.functional.normalize(torch.from_numpy(q), dim=1) @ torch.nn.functional.normalize(torch.from_numpy(g), dim=1).T
        top = s.topk(5, dim=1).indices
        y = torch.arange(len(q))
        plain = (100 * (top[:, 0] == y).float().mean().item(), 100 * (top == y[:, None]).any(1).float().mean().item())
        sat = sattc_eval(cfg, f'sub-{sid:02d}', q, g, g)
        rows += [{'subject': sid, 'method': 'plain', 'top1': plain[0], 'top5': plain[1]},
                 {'subject': sid, 'method': 'sattc', 'top1': sat[0], 'top5': sat[1]}]
        print(sid, 'plain', plain, 'sattc', sat, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject', 'method', 'top1', 'top5'])
        w.writeheader()
        w.writerows(rows)
    for m in ('plain', 'sattc'):
        print(m, 'mean top1 %.2f top5 %.2f' % tuple(np.mean([[r['top1'], r['top5']] for r in rows if r['method'] == m], 0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['validate', 'ours'])
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--run_dir', default=os.path.join(REPO, 'results/things_eeg/honest_ensemble/atm31_valcon/seed3300'))
    p.add_argument('--output_csv', default=os.path.join(REPO, 'results/things_eeg/cvpr/competitors/sattc/on_atm31_valcon.csv'))
    p.add_argument('--no_gallery_whiten', action='store_true', help="their --disable_global_auto_whiten")
    p.add_argument('--device', default='cpu', help='validate only; their run evaluated on cuda (TF32 cuDNN convs)')
    a = p.parse_args()
    if a.mode == 'validate':
        validate(a.subjects or ['sub-01'], a.device)
    else:
        ours(a.run_dir, a.output_csv, [int(s.split('-')[-1]) for s in a.subjects] if a.subjects else list(range(1, 11)),
             not a.no_gallery_whiten)
