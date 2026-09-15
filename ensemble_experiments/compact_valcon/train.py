"""Resumable, concept-validation-only training for the compact decoder study."""
import argparse
import csv
import fcntl
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from module.dataset import EEGPreImageDataset
from module.sampler import GroupedImageBatchSampler
from train import _GroupedSubset, seed_everything, cross_subject_stimulus_mix, build_image_positive_mask
from ensemble_experiments.mechanism_study.common import EEG_DIR, TARGETS, write_json, row_z, margin_metrics
from ensemble_experiments.mechanism_study.train_pair import atomic_save
from .models import CompactDecoder, ARMS

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/things_eeg/compact_valcon_20260908'


def split_indices(data):
    groups = data.get_image_group_indices()
    concepts = sorted({key[0] for key in groups})
    count = min(max(1, int(np.floor(len(concepts) * 0.1))), len(concepts) - 1)
    held = set(np.random.default_rng(20260822).permutation(concepts)[:count].tolist())
    train, val = [], []
    for (obj, _), indices in groups.items():
        (val if obj in held else train).extend(indices)
    return train, val, sorted(held)


def dataset(subjects, training):
    # Load a SINGLE primary target, then concatenate raw IV28 ourselves. Dataset's
    # auxiliary loader normalizes secondary blocks, unlike the historical solos.
    data = EEGPreImageDataset(subjects, train=training, eeg_data_dir=EEG_DIR,
        selected_channels=[], time_window=[0, 250], image_feature_dir=TARGETS['33'],
        text_feature_dir='', image_aug=False, aug_image_feature_dirs=[], average=True, _random=False)
    extra = np.load(Path(TARGETS['28']) / ('image_train.npy' if training else 'image_test.npy'))
    data.image_features = np.concatenate([data.image_features, extra], axis=-1).astype(np.float32)
    data.target_dims = [3200, 3200]
    return data


@torch.inference_mode()
def validation_loss(model, loader, device):
    model.eval()
    total = 0.
    for batch in loader:
        eeg, images, objects, ids = [batch[i].to(device, non_blocking=True) for i in [0, 1, 4, 5]]
        total += model.loss(model(eeg, images), build_image_positive_mask(objects, ids)).item()
    # Historical train.py averages batch losses equally (including final partial).
    return total / len(loader)


@torch.inference_mode()
def test_scores(model, loader, device):
    model.eval()
    es, ims = [[] for _ in model.targets], [[] for _ in model.targets]
    objects, ids = [], []
    for batch in loader:
        features = model(batch[0].to(device), batch[1].to(device))
        for m, (e, i) in enumerate(features):
            es[m].append(F.normalize(e, dim=-1).cpu().numpy())
            ims[m].append(F.normalize(i, dim=-1).cpu().numpy())
        objects.extend(batch[4].tolist()); ids.extend(batch[5].tolist())
    es, ims = [np.concatenate(a) for a in es], [np.concatenate(a) for a in ims]
    scores = np.stack([e @ i.T for e, i in zip(es, ims)])
    assert scores.shape[1:] == (200, 200)
    assert len(set(zip(objects, ids))) == 200
    dump = dict(scores=scores, object=np.array(objects), image_idx=np.array(ids))
    for m, (e, i) in enumerate(zip(es, ims)):
        dump[f'eeg_{m}'] = e; dump[f'image_{m}'] = i
    z = row_z(scores).mean(0)
    metrics = margin_metrics(scores)
    metrics['top5'] = float((np.argsort(z, axis=1)[:, -5:] == np.arange(200)[:, None]).any(1).mean() * 100)
    metrics['solo_top1'] = [float((s.argmax(1) == np.arange(200)).mean() * 100) for s in scores]
    return metrics, dump


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(rng):
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--subject', type=int, choices=range(1, 11), required=True)
    p.add_argument('--seed', type=int, default=3300)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--output', type=Path, default=OUTPUT / 'runs')
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--gradient-rule', choices=['mean', 'nash'], default='mean')
    args = p.parse_args()
    if args.gradient_rule == 'nash':
        if args.arm != 'dual_head' or args.output == OUTPUT / 'runs':
            raise ValueError('Nash requires dual_head and a separate output root')
        from ensemble_experiments.nash_direction import backward, actual_update
    if args.max_batches and args.output == OUTPUT / 'runs':
        raise ValueError('Smoke runs require a separate --output')
    out = args.output / args.arm / f'seed{args.seed}' / f'sub-{args.subject:02d}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'complete.json').exists():
        print('ALREADY_COMPLETE', out, flush=True); return
    config = {**vars(args), 'output': str(args.output), 'batch_size': 1024, 'effective_batch_size': 1017,
              'mixup_type': 'pairwise', 'mixup_alpha': 0.5, 'learning_rate': 3e-4,
              'weight_decay': 1e-4, 'selection': 'minimum mean-head source-concept validation loss',
              'val_concept_ratio': 0.1, 'val_concept_seed': 20260822}
    if args.gradient_rule == 'mean':
        config.pop('gradient_rule')  # Preserve existing mean-run resume configs.
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Refusing configuration change in existing run')
    write_json(out / 'config.json', config)
    sources = [s for s in range(1, 11) if s != args.subject]
    data = dataset(sources, True)
    tr, va, held = split_indices(data)
    write_json(out / 'split.json', dict(training_subjects=sources, outer_subject=args.subject,
        held_concepts=held, train_items=len(tr), validation_items=len(va)))
    train_data = _GroupedSubset(data, tr)
    sampler = GroupedImageBatchSampler(train_data, 1024, samples_per_image=9, seed=args.seed)
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Subset(data, va), batch_size=200, shuffle=False, num_workers=0,
        pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    seed_everything(args.seed)
    model = CompactDecoder(args.arm, data.channels_num).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    write_json(out / 'model_size.json', model.size())
    best, bad, start, selected_epoch = float('inf'), 0, 1, 0
    best_weights = None
    records = []
    last = out / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for value in optimizer.state.values():
            for k, v in value.items():
                if isinstance(v, torch.Tensor): value[k] = v.cuda()
        restore_rng(state['rng'])
        best, bad, start = state['best'], state['bad'], state['epoch'] + 1
        selected_epoch, records = state['selected_epoch'], state['records']
        best_weights = state['best_model']
        atomic_save(out / 'best.pth', dict(model=best_weights, epoch=selected_epoch, val_loss=best))
        # A partial epoch is replayed from last.pth; records are checkpoint-authoritative.
    seed_everything(args.seed + 10000) if start == 1 else None
    print('START', args.arm, args.subject, model.size(), 'steps', len(loader), flush=True)
    for epoch in range(start, args.epochs + 1):
        if bad >= args.patience: break
        sampler.epoch = epoch - 1
        random.seed(args.seed + epoch)
        # Isolate data/mixup randomness from architecture/dropout RNG consumption.
        mix_rng = torch.Generator(device='cuda').manual_seed(args.seed + epoch)
        torch.cuda.reset_peak_memory_stats()
        model.train(); total = 0.; steps = 0
        diagnostics, diagnostic_counts = {}, {}
        torch.cuda.synchronize(); tick = time.monotonic()
        for batch in loader:
            eeg, images, subjects, objects, ids = [batch[i].cuda(non_blocking=True) for i in [0, 1, 3, 4, 5]]
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.set_rng_state(mix_rng.get_state())
                eeg = cross_subject_stimulus_mix(eeg, objects, ids, subjects, alpha=0.5, mixup_type='pairwise')
                mix_rng.set_state(torch.cuda.get_rng_state())
            positives = build_image_positive_mask(objects, ids)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                features = model(eeg, images)
            if args.gradient_rule == 'nash':
                loss, measures, task_grads = backward(model, features, positives)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                measure_actual = steps % 10 == 0
                before = (torch.cat([p.detach().flatten() for p in model.backbone.parameters()])
                          if measure_actual else None)
                optimizer.step()
                if measure_actual: measures.update(actual_update(model, before, task_grads))
                for key,value in measures.items():
                    diagnostics[key] = diagnostics.get(key, 0.) + value.detach()
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1
                del task_grads, before, measures
            else:
                loss = model.loss(features, positives)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                loss.backward(); optimizer.step()
            total += loss.item(); steps += 1
            if args.max_batches and steps >= args.max_batches: break
        torch.cuda.synchronize(); train_seconds = time.monotonic() - tick
        vt = time.monotonic(); val = validation_loss(model, val_loader, 'cuda:0')
        if not np.isfinite(val): raise FloatingPointError('Non-finite validation loss')
        if val < best:
            best, bad, selected_epoch = val, 0, epoch
            best_weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_save(out / 'best.pth', dict(model=best_weights, epoch=epoch, val_loss=val))
        else: bad += 1
        row = dict(epoch=epoch, train_loss=total / steps, validation_loss=val, steps=steps,
            train_seconds=train_seconds, validation_seconds=time.monotonic()-vt,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        if diagnostics:
            row['gradient_diagnostics'] = {k:float(v/diagnostic_counts[k]) for k,v in diagnostics.items()}
        records.append(row)
        atomic_save(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng(),
            epoch=epoch, selected_epoch=selected_epoch, best=best, best_model=best_weights,
            bad=bad, records=records))
        write_json(out / 'epochs.json', records)
        print(json.dumps(row), flush=True)
    selected = torch.load(out / 'best.pth', map_location='cuda:0', weights_only=False)
    model.load_state_dict(selected['model'])
    # No outer EEG or labels have been loaded before checkpoint selection finishes.
    test = dataset([args.subject], False)
    metrics, dump = test_scores(model, DataLoader(test, batch_size=200, shuffle=False), 'cuda:0')
    np.savez_compressed(out / 'test_scores.npz', **dump)
    result = dict(arm=args.arm, subject=args.subject, seed=args.seed, selected_epoch=selected['epoch'],
                  epochs_completed=len(records), validation_loss=selected['val_loss'], **metrics,
                  **model.size(), training_seconds=sum(r['train_seconds'] for r in records),
                  validation_seconds=sum(r['validation_seconds'] for r in records))
    write_json(out / 'complete.json', result)
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__': main()
