"""Exit 1 if an arm's mean over the folds run so far is below a floor.

Usage: abort_check.py <run_dir> [ignored]

The floor lives here, keyed on how many folds are in, so it can be revised while
arms are running -- this file is re-read on every call, whereas editing the
driver shell script under a running bash is not safe.

Why these values. The original plan scaled the floor by the TSConv baseline's
per-fold profile ({1:50.5, 2:43.0, 3:27.0, 4:29.5, 5:29.5, ...}, mean 40.17 over
folds 1-3). The first ortho results show that profile does not transfer: `mixer`
scored 30.0 on fold 1 (baseline 50.5) but 38.5 on fold 2 (baseline 43.0). These
architectures are markedly flatter across subjects than TSConv, so a
baseline-proportional floor would kill arms that are on track for >30 overall.

With a flat profile the running mean is the estimator, and per-fold spread is
large (~7 points sd, so ~4 points se at n=3). Against the >30 target that makes
25.0 the right disaster floor at fold 3 -- it still kills a riemann-class failure
(which sat near 5) within one fold -- and 28.0 the right floor at fold 5, where
the estimate has tightened.
"""
import csv, glob, os, sys

FLOORS = {3: 25.0, 5: 28.0}

run_dir = sys.argv[1]
values = [float(list(csv.DictReader(open(f)))[0]['best top1 acc'])
          for f in sorted(glob.glob(os.path.join(run_dir, '*sub-*', 'result.csv')))]
mean = sum(values) / len(values)
floor = FLOORS.get(len(values), 0.0)
ok = mean >= floor
arm = os.path.basename(os.path.dirname(run_dir))
print(f'[abort-check] {arm} folds 1-{len(values)} mean={mean:.2f} floor={floor:.2f} '
      f'{"CONTINUE" if ok else "ABORT"}', flush=True)
sys.exit(0 if ok else 1)
