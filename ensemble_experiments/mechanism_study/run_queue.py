"""Single-GPU persistent queue with per-job logs, status, and resumable tasks."""
import fcntl
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import torch
from .common import REPO, OUTPUT, write_json
from .train_pair import ARMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    expected = 'RTX 3090' if args.worker == 0 else 'RTX 3080'
    if torch.cuda.device_count() != 1 or expected not in torch.cuda.get_device_name(0):
        raise RuntimeError(f'Worker {args.worker} requires only {expected} visible')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / f'queue_worker{args.worker}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = OUTPUT / 'logs'; logs.mkdir(exist_ok=True)
    module = 'ensemble_experiments.mechanism_study.'
    jobs = []
    # Round-robin subjects/arms gives an early complete fold across mechanisms;
    # there is no score-dependent promotion or pruning.
    for subject in range(args.worker + 1, 11, 3):
        jobs.append((f'repetitions_sub{subject:02d}', [sys.executable, '-m', module + 'repetitions', '--subject', str(subject)]))
        for seed in [3300]:
            for arm in ARMS:
                jobs.append((f'{arm}_s{seed}_sub{subject:02d}', [sys.executable, '-m', module + 'train_pair',
                    '--arm', arm, '--subject', str(subject), '--seed', str(seed)]))
    manifest = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(0),
        jobs=[dict(name=n, command=c) for n, c in jobs],
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((REPO / 'ensemble_experiments/mechanism_study').glob('*.py')) +
            [REPO / p for p in ['train.py', 'module/dataset.py', 'module/loss.py', 'module/eeg_encoder/model.py', 'module/eeg_encoder/atm/atm.py']]})
    write_json(OUTPUT / f'queue_manifest_worker{args.worker}.json', manifest)
    cpu = None
    if args.worker == 0:
        cpu_log = (logs / 'score_analysis.log').open('a')
        cpu = subprocess.Popen([sys.executable, '-u', '-m', module + 'score_analysis'], cwd=REPO, stdout=cpu_log, stderr=subprocess.STDOUT)
    failures, done = [], []
    try:
        for name, command in jobs:
            started = time.time()
            with (logs / f'{name}.log').open('a') as log:
                print('START', name, flush=True)
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                write_json(OUTPUT / f'status_worker{args.worker}.json', dict(state='running', current=name, pid=child.pid,
                    hostname=os.uname().nodename, gpu=torch.cuda.get_device_name(0), worker=args.worker,
                    started=started, completed=len(done), total=len(jobs), failures=failures, analysis_pid=cpu.pid if cpu else None))
                rc = child.wait()
            event = dict(name=name, returncode=rc, seconds=time.time() - started)
            with (OUTPUT / f'events_worker{args.worker}.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            (failures if rc else done).append(name)
            print('END', event, flush=True)
            # Each failure is explicit; independent later experiments still run.
            if rc and len(failures) >= 3 and all(x in failures for x, _ in jobs[max(0, jobs.index((name, command))-2):jobs.index((name, command))+1]):
                raise RuntimeError('Three consecutive failures; stopping to avoid wasting GPU time')
            # Serialize summaries across NAS workers, keeping outputs consistent.
            with (OUTPUT / 'summary.lock').open('w') as summary_lock, (logs / 'summaries.log').open('a') as log:
                fcntl.flock(summary_lock, fcntl.LOCK_EX)
                summary_rc = subprocess.call([sys.executable, '-m', module + 'summarize'], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if summary_rc:
                print('WARNING: summary failed; inspect summaries.log', flush=True)
        cpu_rc = cpu.wait() if cpu else 0
        write_json(OUTPUT / f'status_worker{args.worker}.json', dict(state='finished' if not failures and cpu_rc == 0 else 'finished_with_errors',
            completed=len(done), total=len(jobs), failures=failures, analysis_returncode=cpu_rc))
    except BaseException as error:
        write_json(OUTPUT / f'status_worker{args.worker}.json', dict(state='stopped', error=str(error), completed=len(done), total=len(jobs), failures=failures))
        raise


if __name__ == '__main__':
    main()
