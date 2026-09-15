"""Same-device end-to-end training-step and EEG-inference microbenchmark."""
import argparse
import gc
import json
import time
import torch
import torch.nn.functional as F
from train import seed_everything, build_image_positive_mask, cross_subject_stimulus_mix
from ensemble_experiments.mechanism_study.common import write_json
from .models import CompactDecoder, ARMS
from .train import OUTPUT


def measure(arm, precision, steps=12):
    seed_everything(3300)
    model = CompactDecoder(arm).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n = 1017
    eeg = torch.randn(n, 63, 250, device='cuda')
    images = F.normalize(torch.randn(n//9, 6400, device='cuda'), dim=-1).repeat_interleave(9, 0)
    objects = torch.arange(n//9, device='cuda').repeat_interleave(9)
    ids = torch.zeros_like(objects); subjects = torch.arange(1, 10, device='cuda').repeat(n//9)
    positives = build_image_positive_mask(objects, ids)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for step in range(steps+3):
        torch.cuda.synchronize(); t = time.monotonic()
        mixed = cross_subject_stimulus_mix(eeg, objects, ids, subjects, alpha=.5, mixup_type='pairwise')
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            features = model(mixed, images)
        loss = model.loss(features, positives)
        if not torch.isfinite(loss): raise FloatingPointError('benchmark loss')
        loss.backward(); opt.step()
        torch.cuda.synchronize()
        if step >= 3: timings.append(time.monotonic()-t)
    peak = torch.cuda.max_memory_allocated()/2**30
    model.eval()
    inference = []
    with torch.inference_mode():
        for _ in range(12):
            torch.cuda.synchronize(); t = time.monotonic()
            model.encode(eeg[:200])
            torch.cuda.synchronize(); inference.append(time.monotonic()-t)
    return dict(arm=arm, precision=precision, gpu=torch.cuda.get_device_name(), **model.size(),
                training_step_seconds=sum(timings)/len(timings),
                inference_200_seconds=sum(inference[2:])/10, peak_allocated_gib=peak,
                batch_size=n, steps=steps, last_loss=loss.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--steps', type=int, default=12)
    args = p.parse_args()
    rows = []
    for arm in (*ARMS, 'reference_atm'):
        try:
            row = measure(arm, args.precision, args.steps)
        except torch.cuda.OutOfMemoryError as exc:
            row = dict(arm=arm, precision=args.precision, error='CUDA OOM', detail=str(exc))
        rows.append(row)
        write_json(OUTPUT / f'benchmark_{args.precision}.json', rows)
        print(json.dumps(row), flush=True)
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__': main()
