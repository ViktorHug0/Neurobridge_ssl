"""Controlled paired decoders with unseen-subject AND unseen-concept validation.

The outer subject is evaluated only after checkpoint selection. A second,
disjoint 200-concept source gallery is exported for mechanism replication.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from module.dataset import EEGPreImageDataset
from module.eeg_encoder.model import TSConv_parameterizable
from module.eeg_encoder.atm.atm import ATMS
from module.projector import ProjectorLinear
from module.loss import ContrastiveLoss
from module.sampler import GroupedImageBatchSampler
from train import _GroupedSubset, cross_subject_stimulus_mix, build_image_positive_mask, seed_everything
from ensemble_experiments.decorrelated_models.losses import deployed_ensemble_contrastive_loss
from .common import EEG_DIR, TARGETS, OUTPUT, margin_metrics, row_z, write_json

ARMS = {
    'geometry28': dict(encoders=['ATM', 'TSConv'], targets=['28', '28']),
    'geometry33': dict(encoders=['ATM', 'TSConv'], targets=['33', '33']),
    'temporal': dict(encoders=['TSConv', 'TSConv'], targets=['28', '28'], windows=[[0, 75], [75, 150]]),
    'independent_targets': dict(encoders=['TSConv', 'TSConv'], targets=['28', '33']),
    'shared_stem': dict(encoders=['TSConv', 'TSConv'], targets=['28', '33'], sharing='stem'),
    'shared_full': dict(encoders=['TSConv', 'TSConv'], targets=['28', '33'], sharing='full'),
    'shared_full_matched': dict(encoders=['TSConv', 'TSConv'], targets=['28', '33'], sharing='full', match_parameters=True),
    'joint_b030': dict(encoders=['ATM', 'TSConv'], targets=['28', '28'], beta=0.3),
    'frozen_b000': dict(encoders=['ATM', 'TSConv'], targets=['28', '28'], freeze=True, beta=0.),
    'frozen_b030': dict(encoders=['ATM', 'TSConv'], targets=['28', '28'], freeze=True, beta=0.3),
}


class Pair(nn.Module):
    def __init__(self, spec, dims, channels=63, samples=250, seed=3300, width=128):
        super().__init__()
        self.spec = spec
        self.dims = dims
        self.sharing = spec.get('sharing', 'none')
        self.encoders = nn.ModuleList()
        self.eeg_heads = nn.ModuleList()
        self.image_heads = nn.ModuleList()
        self.criteria = nn.ModuleList()
        for m, encoder_type in enumerate(spec['encoders']):
            # Identical seeds across targets/windows isolate those changes.
            seed_everything(seed)
            if self.sharing != 'full' or m == 0:
                encoder = (ATMS(feature_dim=width, eeg_sample_points=samples, channels_num=channels)
                    if encoder_type == 'ATM' else TSConv_parameterizable(
                        feature_dim=width, eeg_sample_points=samples, channels_num=channels,
                        temporal_kernel=30, pool_kernel=51, pool_stride=5, dropout=0.5))
                self.encoders.append(encoder)
            self.eeg_heads.append(ProjectorLinear(width, 128))
            self.image_heads.append(ProjectorLinear(dims[m], 128))
            self.criteria.append(ContrastiveLoss(init_temperature=0.07, alpha=1., beta=1.,
                eeg_l2norm=False, img_l2norm=True, text_l2norm=False, learnable=False, is_softplus=True))
        if self.sharing == 'stem':
            self.stem = self.encoders[0].tsconv[:4]
            for encoder in self.encoders:
                encoder.tsconv = encoder.tsconv[4:]
        masks = torch.ones(2, 1, 1, samples)
        for m, (start, stop) in enumerate(spec.get('windows', [[0, samples]] * 2)):
            masks[m].zero_()
            masks[m, :, :, start:stop] = 1
        self.register_buffer('masks', masks)

    def freeze_first(self):
        for module in (self.encoders[0], self.eeg_heads[0], self.image_heads[0], self.criteria[0]):
            module.requires_grad_(False)
            module.eval()

    def forward(self, eeg, images, subjects):
        blocks = images.split(self.dims, dim=-1)
        shared = None
        if self.sharing == 'stem':
            shared = self.stem(eeg.unsqueeze(1))
        elif self.sharing == 'full':
            shared = self.encoders[0](eeg)
        result = []
        for m, encoder_type in enumerate(self.spec['encoders']):
            if self.sharing == 'full':
                hidden = shared
            elif self.sharing == 'stem':
                encoder = self.encoders[m]
                hidden = encoder.proj_eeg(encoder.projection(encoder.tsconv(shared)).flatten(1))
            else:
                x = eeg * self.masks[m]
                hidden = self.encoders[m](x, subjects) if encoder_type == 'ATM' else self.encoders[m](x)
            # Normalize BOTH raw image blocks, avoiding asymmetric auxiliary normalization.
            result.append((self.eeg_heads[m](hidden), self.image_heads[m](F.normalize(blocks[m], dim=-1))))
        return result


def make_pair(spec, dims, channels, seed):
    width = 128
    if spec.get('match_parameters'):
        base = dict(spec, sharing='none', match_parameters=False)
        count = lambda model: sum(p.numel() for p in model.parameters())
        target = count(Pair(base, dims, channels, seed=seed))
        p128 = count(Pair(spec, dims, channels, seed=seed, width=128))
        p129 = count(Pair(spec, dims, channels, seed=seed, width=129))
        # TSConv's single backbone has one width-squared layer; heads are affine in width.
        a = p129 - p128 - (129 ** 2 - 128 ** 2)
        c = p128 - 128 ** 2 - a * 128
        width = round((-a + np.sqrt(a * a - 4 * (c - target))) / 2)
    return Pair(spec, dims, channels, seed=seed, width=width), width


def loaders(args, spec):
    val_subject = args.subject % 10 + 1
    sources = [s for s in range(1, 11) if s not in [args.subject, val_subject]]
    kwargs = dict(eeg_data_dir=EEG_DIR, selected_channels=[], time_window=[0, 250],
        image_feature_dir=','.join(TARGETS[t] for t in spec['targets']),
        text_feature_dir='', image_aug=False, aug_image_feature_dirs=[], average=True, _random=False)
    data = EEGPreImageDataset(sources, train=True, **kwargs)
    val_data = EEGPreImageDataset([val_subject], train=True, **kwargs)
    concepts = np.random.default_rng(20260907).permutation(data.num_objects)
    validation, probe = concepts[:200].tolist(), concepts[200:400].tolist()
    excluded = set(validation + probe)
    groups = data.get_image_group_indices()
    train_indices = [i for (obj, _), indices in groups.items() if obj not in excluded for i in indices]
    train_data = _GroupedSubset(data, train_indices)
    sampler = GroupedImageBatchSampler(train_data, args.batch_size, samples_per_image=8, seed=7330)
    train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    def gallery(which):
        indices = [int(c) * val_data.num_images_per_object for c in which]
        return DataLoader(Subset(val_data, indices), batch_size=200, shuffle=False, num_workers=0)
    protocol = dict(train_subjects=sources, validation_subject=val_subject, outer_subject=args.subject,
        validation_concepts=validation, probe_concepts=probe, training_concepts=sorted(set(concepts.tolist()) - excluded),
        selected_image_per_concept=0, selection='minimum 200-way fused source-validation cross entropy',
        outer_test_policy='evaluate only after checkpoint selection; no refit',
        note='eight-source-subject, concept-disjoint comparison; not directly comparable to nine-source historical scores')
    return data, train_loader, gallery(validation), gallery(probe), kwargs, protocol


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    eegs, imgs = [[], []], [[], []]
    objects, indices = [], []
    for batch in loader:
        features = model(batch[0].to(device), batch[1].to(device), batch[3].to(device))
        for m, (eeg, image) in enumerate(features):
            eegs[m].append(F.normalize(eeg, dim=-1).cpu().numpy())
            imgs[m].append(F.normalize(image, dim=-1).cpu().numpy())
        objects.extend(batch[4].tolist())
        indices.extend(batch[5].tolist())
    eegs, imgs = [np.concatenate(x) for x in eegs], [np.concatenate(x) for x in imgs]
    scores = np.stack([e @ i.T for e, i in zip(eegs, imgs)])
    z = row_z(scores.astype(np.float64)).mean(0)
    loss = F.cross_entropy(torch.from_numpy(z), torch.arange(len(z))).item()
    dump = dict(scores=scores, a_eeg=eegs[0], b_eeg=eegs[1], a_image=imgs[0], b_image=imgs[1],
        object=np.array(objects), image_idx=np.array(indices))
    return loss, margin_metrics(scores), dump


def atomic_save(path, payload):
    temporary = path.with_suffix('.tmp.pth')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--seed', type=int, default=3300)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--output', type=Path, default=OUTPUT / 'controlled')
    p.add_argument('--max-batches', type=int, default=0, help='smoke only; use a separate output directory')
    args = p.parse_args()
    spec = ARMS[args.arm]
    out = args.output / args.arm / f'seed{args.seed}' / f'sub-{args.subject:02d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('complete', out, flush=True)
        return
    config = {**vars(args), 'output': str(args.output), 'spec': spec}
    config_path = out / 'train_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed in an existing run directory')
    write_json(config_path, config)
    data, train_loader, val_loader, probe_loader, kwargs, protocol = loaders(args, spec)
    write_json(out / 'split.json', protocol)
    model, width = make_pair(spec, data.target_dims, data.channels_num, args.seed)
    if spec.get('freeze'):
        reference = args.output / 'geometry28' / f'seed{args.seed}' / f'sub-{args.subject:02d}'
        if not (reference / 'complete.json').exists():
            raise RuntimeError(f'Frozen reference not completed: {reference}')
        if json.loads((reference / 'split.json').read_text()) != protocol:
            raise ValueError('Frozen reference uses different source splits')
        state = torch.load(reference / 'best.pth', map_location='cpu', weights_only=False)['model']
        current = model.state_dict()
        for key in current:
            if any(key.startswith(prefix) for prefix in ['encoders.0.', 'eeg_heads.0.', 'image_heads.0.', 'criteria.0.']):
                current[key] = state[key]
        model.load_state_dict(current)
        model.freeze_first()
    model.to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    seed_everything(7330)
    start, best = 1, float('inf')
    last_path = out / 'last.pth'
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for state_values in optimizer.state.values():
            for key, value in state_values.items():
                if isinstance(value, torch.Tensor):
                    state_values[key] = value.to(args.device)
        start, best = state['epoch'] + 1, state['best']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        train_loader.batch_sampler.epoch = start - 1
    metadata = dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), backbone_width=width)
    write_json(out / 'model_size.json', metadata)
    print('START', args.arm, args.subject, args.seed, metadata, flush=True)
    for epoch in range(start, args.epochs + 1):
        tick = time.monotonic()
        model.train()
        if spec.get('freeze'):
            model.freeze_first()
        loss_sum, steps = 0., 0
        for batch in train_loader:
            eeg, images, subjects, objects, image_ids = [batch[i].to(args.device, non_blocking=True) for i in [0, 1, 3, 4, 5]]
            eeg = cross_subject_stimulus_mix(eeg, objects, image_ids, subjects, alpha=0.5, mixup_type='pairwise')
            positive = build_image_positive_mask(objects, image_ids)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                features = model(eeg, images, subjects)
            # Contrastive losses and cosine geometry remain FP32 for all arms.
            features = [(e.float(), i.float()) for e, i in features]
            losses = [criterion.multi_positive_pair_loss(e, i, positive) for criterion, (e, i) in zip(model.criteria, features)]
            loss = losses[1] if spec.get('freeze') else sum(losses)
            if spec.get('beta', 0):
                scores = [F.normalize(e, dim=-1) @ F.normalize(i, dim=-1).T for e, i in features]
                fusion, _ = deployed_ensemble_contrastive_loss(*scores, positive, objects, image_ids)
                loss = loss + spec['beta'] * fusion
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            steps += 1
            if args.max_batches and steps >= args.max_batches:
                break
        validation_loss, validation_metrics, _ = evaluate(model, val_loader, args.device)
        if validation_loss < best:
            best = validation_loss
            atomic_save(out / 'best.pth', dict(model=model.state_dict(), epoch=epoch, validation_loss=best))
        row = dict(epoch=epoch, train_loss=loss_sum / steps, validation_loss=validation_loss,
            validation_top1=validation_metrics['top1'], seconds=time.monotonic() - tick,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        with (out / 'epochs.csv').open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        atomic_save(last_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best=best,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        print(json.dumps(row), flush=True)
    selected = torch.load(out / 'best.pth', map_location=args.device, weights_only=False)
    model.load_state_dict(selected['model'])
    test_data = EEGPreImageDataset([args.subject], train=False, **kwargs)
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False, num_workers=0)
    results = {}
    for name, loader in [('validation', val_loader), ('probe', probe_loader), ('test', test_loader)]:
        loss, metrics, dump = evaluate(model, loader, args.device)
        results[name] = dict(loss=loss, **metrics)
        np.savez_compressed(out / f'{name}_scores.npz', **dump)
    row = dict(arm=args.arm, subject=args.subject, seed=args.seed, epoch=selected['epoch'],
        **results['test'], **metadata, selection_protocol='source_subject_and_concepts')
    with (out / 'result.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    write_json(out / 'complete.json', results)
    print('COMPLETE', out, results, flush=True)


if __name__ == '__main__':
    main()
