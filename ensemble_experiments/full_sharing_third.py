"""Accelerate the frozen two-worker grid without altering running code/manifests.

Work backwards through the last five tasks of each queue. The original workers
skip completed folds. Yield before either original worker reaches that tail,
leaving resumable checkpoints for them; never race their per-fold locks.
"""
import fcntl
import json
import os
import socket
import subprocess
import sys
import time

import torch
from ensemble_experiments.full_sharing_v2.queue import baseline_complete, summary
from ensemble_experiments.full_sharing_v2.train import OUTPUT, REPO, sources
from ensemble_experiments.mechanism_study.common import write_json


def approaching():
    for worker in (0, 1):
        status = json.loads((OUTPUT / f'status_worker{worker}.json').read_text())
        # Tail starts at index 10. Yield while the predecessor still runs.
        if status['state'] == 'complete' or status.get('completed_in_queue', 0) >= 9:
            return True
    return False


def main():
    baseline_complete()
    plan = json.loads((OUTPUT / 'manifest.json').read_text())
    assert sources() == plan['sources']
    tasks = []
    for worker in (0, 1):
        tasks.extend([t for t in plan['tasks'] if t['worker'] == worker][10:])
    tasks.sort(key=lambda t: (t['subject'], t['config']), reverse=True)
    write_json(OUTPUT / 'third_worker_plan.json', dict(tasks=tasks, strategy='reverse tail; yield at original queue index 9'))
    device = str(torch.cuda.get_device_properties(0).uuid)
    with (OUTPUT / f'device-{device}.lock').open('a') as gpu, (OUTPUT / 'worker2.lock').open('a') as lock:
        for handle in (gpu, lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = 0
        for task in tasks:
            assert sources() == plan['sources']
            if approaching():
                break
            c, s = task['config'], task['subject']
            out = OUTPUT / f'runs/{c}/seed3300/sub-{s:02d}'
            if (out / 'complete.json').exists():
                continue
            status = dict(state='running', task=task, host=socket.gethostname(),
                          pid=os.getpid(), completed=completed, total=len(tasks), updated=time.time())
            write_json(OUTPUT / 'status_worker2.json', status)
            with (OUTPUT / f'{c}-sub-{s:02d}.log').open('a', buffering=1) as log:
                child = subprocess.Popen([sys.executable, '-u', '-m',
                    'ensemble_experiments.full_sharing_v2.train', '--config', c,
                    '--subject', str(s)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                yielded = False
                while child.poll() is None:
                    if approaching():
                        child.terminate()
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        yielded = True
                        break
                    time.sleep(1)
            if yielded:
                break
            if child.returncode:
                write_json(OUTPUT / 'status_worker2.json', dict(status, state='failed', exit_code=child.returncode))
                raise RuntimeError(f'{c} subject {s} failed')
            completed += 1
            summary()
        write_json(OUTPUT / 'status_worker2.json', dict(state='finished', completed=completed,
                   note='Any remaining tail folds stay assigned to the original workers.'))


if __name__ == '__main__':
    main()
