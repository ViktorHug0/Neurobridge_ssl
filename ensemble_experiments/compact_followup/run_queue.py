"""One locked queue per arm: electrodes on513, distillation on an allocated A100."""
import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime,timezone
from ensemble_experiments.compact_valcon.run_queue import sources as old_sources
from ensemble_experiments.mechanism_study.common import write_json
from .teachers import teacher_paths,sha256
from .models import ARMS
from .train import ROOT,OUTPUT


def now():return datetime.now(timezone.utc).isoformat()


def sources():
    result=old_sources()
    for p in (ROOT/'ensemble_experiments/compact_followup').iterdir():
        if p.suffix in ['.py','.md','.sh','.sbatch']:result[str(p.relative_to(ROOT))]=sha256(p)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--resplit',action='store_true');p.add_argument('--arm',choices=ARMS);args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True);manifest_path=OUTPUT/'manifest.json'
    if args.resplit:
        manifest=json.loads(manifest_path.read_text())
        backup=OUTPUT/'manifest_single_gpu.json'
        if backup.exists():raise RuntimeError('Resplit already recorded')
        write_json(backup,manifest)
        manifest.update(source_sha256=sources(),gpus=2,
            tasks=[dict(subject=s,arm=a) for a in ARMS for s in range(1,11)],
            execution=dict(electrode='sl-tp-br-513 / RTX3080',distill='Slurm / A100 / 40GB host RAM / 8 CPUs'),
            resplit_at=now(),host=None)
        write_json(manifest_path,manifest)
        (OUTPUT/'status.json').rename(OUTPUT/'status_before_split.json')
        print('RESPLIT: separate arm queues; prior manifest/status preserved');return
    if args.prepare:
        if manifest_path.exists():raise RuntimeError('Manifest already frozen')
        assert json.loads((OUTPUT/'resume_test.json').read_text())['passed']
        assert json.loads((OUTPUT/'teacher_audit.json').read_text())['passed']
        benchmark=json.loads((OUTPUT/'benchmark.json').read_text())
        for r in benchmark:
            assert r['parameters']<=3131081*1.05
            assert r['training_step_seconds']<=.2817838339251466*1.20
        teachers={str(s):{str(path):sha256(path) for path in teacher_paths(s)} for s in range(1,11)}
        write_json(manifest_path,dict(created=now(),source_sha256=sources(),teachers=teachers,
            tasks=[dict(subject=s,arm=a) for a in ARMS for s in range(1,11)],benchmark=benchmark,
            seed=3300,alignment_dim=128,mixup='pairwise',reference_top1=40.2,kd_weight=1.0,
            kd_temperature=1.0,epochs=100,patience=20,gpus=2,
            execution=dict(electrode='sl-tp-br-513 / RTX3080',distill='Slurm / A100 / 40GB host RAM / 8 CPUs')))
        (OUTPUT/'logs').mkdir(exist_ok=True);print('PREPARED',manifest_path);return
    if args.arm is None:p.error('--arm is required')
    if args.arm=='electrode' and socket.gethostname()!='sl-tp-br-513':raise RuntimeError('Electrode queue must run on513')
    lock=(OUTPUT/f'queue_{args.arm}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(manifest_path.read_text());failures=[];completed=0
    tasks=[t for t in manifest['tasks'] if t['arm']==args.arm]
    status_path=OUTPUT/f'status_{args.arm}.json'
    status=dict(host=socket.gethostname(),pid=os.getpid(),slurm_job=os.environ.get('SLURM_JOB_ID'),
                started=now(),total=len(tasks),arm=args.arm)
    for task in tasks:
        if sources()!=manifest['source_sha256']:
            write_json(status_path,dict(**status,state='source_changed',completed=completed,failures=failures))
            raise RuntimeError('Source changed; refusing next task')
        if task['arm']=='distill':
            for path,digest in manifest['teachers'][str(task['subject'])].items():
                from pathlib import Path
                if sha256(Path(path))!=digest:raise RuntimeError('Teacher checkpoint changed')
        log_path=OUTPUT/'logs'/f"{task['arm']}-sub-{task['subject']:02d}.log"
        command=[sys.executable,'-u','-m','ensemble_experiments.compact_followup.train',
                 '--arm',task['arm'],'--subject',str(task['subject'])]
        tick=time.monotonic()
        with log_path.open('a',buffering=1) as log:
            log.write(f'\nQUEUE START {now()} {command}\n')
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write_json(status_path,dict(**status,state='running',task=task,child_pid=child.pid,
                completed=completed,updated=now(),failures=failures))
            code=child.wait()
        event=dict(**task,exit_code=code,seconds=time.monotonic()-tick,finished=now())
        with (OUTPUT/f'events_{args.arm}.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if code:failures.append(event)
        else:completed+=1
        print(json.dumps(event),flush=True)
        with (OUTPUT/'logs/summary.log').open('a') as log:
            result=subprocess.run([sys.executable,'-m','ensemble_experiments.compact_followup.summarize'],
                                  cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:print('WARNING summary failed; inspect summary.log',flush=True)
        if len(failures)>=3:break
    write_json(status_path,dict(**status,state='finished' if not failures else 'finished_with_failures',
        completed=completed,failures=failures,updated=now()))


if __name__=='__main__':main()
