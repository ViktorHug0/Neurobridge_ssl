"""ARCH_LOG watchdog: every 10 min, cut h* screens that are clearly losing to hbase32_things (TSConv-bn64 on the same
code path) at matched epochs: >= 10 epochs and both the mean and the last-5 delta <= -1.5. Removes the recipe from
align_queue.txt (atomically, before the kill, so no worker re-launches it), kills its train.py on the worker jobs,
and appends a line to ARCH_LOG.md.
    arch_autokill.py <job id> [<job id> ...]
"""
import glob, json, os, subprocess, sys, time
import numpy as np

ROOT = '/nasbrain/p20fores/Neurobridge_SSL'
CV4 = f'{ROOT}/results/alljoined/track1/cv4'
Q = f'{ROOT}/neurips_challenge/overnight/align_queue.txt'


def curve(f):
    return {r['epoch']: np.mean(list(r['by_subject'].values())) for r in map(json.loads, open(f))}  # already in %


def verdict(c, cb):
    ep = sorted(set(c) & set(cb))
    d = np.array([c[e] - cb[e] for e in ep])
    return len(d), (d.mean() if len(d) else 0.0), (d[-5:].mean() if len(d) else 0.0)


def main(jobs):
    base = curve(glob.glob(f'{CV4}/hbase32_things/fold1/*/2026*/sealed_per_epoch.jsonl')[0])
    # scale check on real curves: hbase32 vs the original TSC-bn64 run is -0.6 over 40 epochs (ARCH_LOG)
    ref = curve(glob.glob(f'{CV4}/tsc_dbn40_bn64_things/fold1/*/2026*/sealed_per_epoch.jsonl')[0])
    assert abs(verdict(base, ref)[1]) < 3, 'curve scale is wrong'
    done = set()
    while True:
        for f in glob.glob(f'{CV4}/h*_things/fold1/*/2026*/sealed_per_epoch.jsonl'):
            recipe = f.split('/')[-5]
            queued = {l.split()[0] for l in open(Q) if l.strip() and not l.startswith('#')}
            if recipe in done or recipe not in queued or recipe.startswith('hbase') or time.time() - os.path.getmtime(f) > 1200 or \
                    glob.glob(os.path.dirname(f) + '/track1_score.json'):
                continue  # not running
            n, mean, last5 = verdict(curve(f), base)
            if n >= 10 and mean <= -1.5 and last5 <= -1.5:
                lines = [l for l in open(Q) if l.split()[:1] != [recipe]]
                open(Q + '.new', 'w').writelines(lines); os.replace(Q + '.new', Q)
                for j in jobs:
                    subprocess.run(['timeout', '60', 'srun', f'--jobid={j}', '--overlap', '-n', '1', '-c', '1', '--mem=200M',
                                    'bash', '-c', 'for p in $(pgrep -x python); do ps -ww -o args= -p $p | grep -q "cv4/$0/" '
                                    '&& kill $p; done; true', recipe], capture_output=True)  # python only: pkill -f also hits srun
                with open(f'{ROOT}/neurips_challenge/ARCH_LOG.md', 'a') as log:
                    log.write(f'- [autokill {time.strftime("%H:%M")}] {recipe}: {n} ep, mean {mean:+.2f}, last5 {last5:+.2f} '
                              'vs hbase32 -> KILLED\n')
                done.add(recipe)
                print(f'killed {recipe} {n} {mean:+.2f} {last5:+.2f}', flush=True)
        time.sleep(600)


if __name__ == '__main__':
    assert verdict({1: 10, 2: 12}, {1: 11, 2: 15}) == (2, -2.0, -2.0)
    main(sys.argv[1:])
