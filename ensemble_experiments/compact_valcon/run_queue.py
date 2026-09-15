"""Three fixed, balanced workers with persistent task logs and source guards."""
import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from .models import ARMS
from .train import ROOT, OUTPUT
from ensemble_experiments.mechanism_study.common import write_json


def now(): return datetime.now(timezone.utc).isoformat()


def sources():
    paths = [p for p in (ROOT/'ensemble_experiments/compact_valcon').iterdir()
             if p.suffix in ['.py','.md','.sh','.sbatch']]
    paths += [ROOT/p for p in ['train.py','module/dataset.py','module/loss.py','module/sampler.py',
        'module/projector.py','module/util.py','module/eeg_encoder/model.py',
        'ensemble_experiments/mechanism_study/train_pair.py','ensemble_experiments/mechanism_study/common.py']]
    paths += list((ROOT/'module/eeg_encoder/atm').rglob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', type=int, choices=[0,1,2])
    args = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT/'manifest.json'
    if args.prepare:
        if plan_path.exists(): raise RuntimeError('Manifest already frozen')
        if not json.loads((OUTPUT/'resume_test.json').read_text())['passed']:
            raise RuntimeError('Resume test failed')
        benchmark = json.loads((OUTPUT/'benchmark_fp32.json').read_text())
        by_arm = {r['arm']:r for r in benchmark}
        for arm in ARMS:
            if 'error' in by_arm[arm]: raise RuntimeError(f'Benchmark failed: {arm}')
            if by_arm[arm]['parameters'] > by_arm['single']['parameters']*1.05:
                raise RuntimeError('Parameter budget failed')
        # Max(TSConv, ATM) on this card, same batch/precision, no parallelism credit.
        available = [by_arm[a]['training_step_seconds'] for a in ['single','reference_atm'] if 'error' not in by_arm[a]]
        for arm in ARMS:
            if by_arm[arm]['training_step_seconds'] > 1.2*max(available):
                raise RuntimeError(f'Training time budget failed: {arm}')
        tasks = [dict(arm=arm, subject=s, worker=(s-1+i)%3) for s in range(1,11) for i,arm in enumerate(ARMS)]
        write_json(plan_path, dict(created=now(), seed=3300, precision='fp32', epochs=100, patience=20,
            mixup='pairwise', alignment_dim=128, tasks=tasks, source_sha256=sources(),
            benchmark=benchmark, historical_reference=41.4,
            historical_caveat='historical TSConv member uses group mixup and fd512; not a matched recipe control'))
        (OUTPUT/'logs').mkdir(exist_ok=True)
        print('PREPARED', plan_path); return
    if args.worker is None: p.error('--worker is required unless --prepare')
    lock = (OUTPUT/f'worker{args.worker}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(plan_path.read_text())
    tasks = [t for t in manifest['tasks'] if t['worker']==args.worker]
    failures = []; completed = 0
    status_path = OUTPUT/f'status_worker{args.worker}.json'
    status = dict(worker=args.worker, host=socket.gethostname(), pid=os.getpid(),
                  slurm_job=os.environ.get('SLURM_JOB_ID'), started=now(), total=len(tasks))
    for task in tasks:
        if sources() != manifest['source_sha256']:
            write_json(status_path, dict(**status, state='source_changed', updated=now(), failures=failures))
            raise RuntimeError('Source hashes changed since launch; stopping before next task')
        log_path = OUTPUT/'logs'/f"{task['arm']}-sub-{task['subject']:02d}.log"
        command = [sys.executable, '-u', '-m', 'ensemble_experiments.compact_valcon.train',
                   '--arm', task['arm'], '--subject', str(task['subject'])]
        tick = time.monotonic()
        with log_path.open('a', buffering=1) as log:
            log.write(f'\nQUEUE START {now()} {command}\n')
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write_json(status_path, dict(**status, state='running', task=task, child_pid=child.pid,
                                        completed=completed, updated=now(), failures=failures))
            code = child.wait()
        event = dict(**task, exit_code=code, seconds=time.monotonic()-tick, finished=now())
        with (OUTPUT/f'events_worker{args.worker}.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        if code: failures.append(event)
        else: completed += 1
        print(json.dumps(event), flush=True)
        with (OUTPUT/'logs'/f'summary_worker{args.worker}.log').open('a') as log:
            result = subprocess.run([sys.executable,'-m','ensemble_experiments.compact_valcon.summarize'],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: print('WARNING: summary failed; inspect summary log', flush=True)
        if len(failures)>=3: break
    write_json(status_path, dict(**status, state='finished' if not failures else 'finished_with_failures',
                                completed=completed, failures=failures, updated=now()))


if __name__ == '__main__': main()
