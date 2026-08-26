"""Dump plain-cosine score inputs (eeg/image features + labels) for every candidate run."""
import json, os, subprocess, sys, time

MAN = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
DUMP = 'results/things_eeg/synthetic_subjects/ensemble_screen/dumps'
EVAL = 'results/things_eeg/synthetic_subjects/ensemble_screen/eval'
names = sys.argv[1:] or sorted(MAN)
os.makedirs(DUMP, exist_ok=True)
for name in names:
    for s in range(1, 11):
        out = f'{DUMP}/{name}-sub{s:02d}.npz'
        if os.path.exists(out):
            continue
        ck = MAN[name]['folds'][str(s)]
        t = time.time()
        r = subprocess.run([sys.executable, 'evaluate.py', '--checkpoint_dir', ck,
                            '--output_dir', EVAL, '--output_name', f'{name}-sub{s:02d}',
                            '--eval_mode', 'plain_cosine', '--test_subject_id', str(s),
                            '--device', 'cuda:0', '--batch_size', '32', '--num_workers', '0',
                            '--dump_npz', out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f'FAIL {name} sub{s:02d}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"}', flush=True)
            break  # whole family is broken; skip its remaining folds
        print(f'ok {name} sub{s:02d} {time.time()-t:.0f}s', flush=True)
