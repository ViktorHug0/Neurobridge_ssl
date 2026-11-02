"""One line per in-flight / recent alignment screen: matched-epoch mean delta vs its baseline (kill rule:
<= -1.5 over >= 6 epochs; seed-noise sd ~1.7 per epoch).   align_running.py"""
import glob, json, os, time
import numpy as np

ROOT = 'results/alljoined/track1/cv4'
BASE = {'tsc': 'tsc_dbn40_things', 'sqf': 'sqf_dbn_things', 'reve': 'reve_d6_things', 'stf': 'sqf_dbn_things', 'sqfmix': 'sqf_dbn_things', 'sta': 'tsc_dbn40_bn64_things', 'etok': 'tsc_dbn40_bn64_things', 'hfa': 'tsc_dbn40_bn64_things', 'hta': 'tsc_dbn40_bn64_things', 'hvs': 'tsc_dbn40_bn64_things', 'hds': 'tsc_dbn40_bn64_things', 'hdp': 'tsc_dbn40_bn64_things', **dict.fromkeys(['rvir', 'rmix', 'rmix12', 'rmix40', 'rmixflat', 'rmixd8', 'rmixd12', 'rmix', 'rmix_split', 'rmix_ov', 'rmix_ms', 'rmix_tsc', 'rmix_tsc9', 'rmix_sqf', 'rmix_bip', 'rmix_gate', 'rmix_reg', 'rmix_ep', 'rmix_tffn', 'rmix_apool', 'rmix_filt', 'rmix_cemb', 'rmix_union', 'rtime', 'rtime12', 'rtime3', 'relec'], 'reve_d6w256_things')}


def curve(f):
    return {r['epoch']: np.mean(list(r['by_subject'].values())) for r in map(json.loads, open(f))}


now = time.time()
for f in sorted(glob.glob(f'{ROOT}/*/fold*/*/2026*/sealed_per_epoch.jsonl')):
    if now - os.path.getmtime(f) > 3 * 3600:
        continue
    recipe, fold = f.split('/')[4], f.split('/')[5]
    base = 'tsc_dbn40_bn64_things' if recipe.startswith('tsc_dbn40_bn64') else BASE.get(recipe.split('_')[0], 'tsc_dbn40_bn64_things' if recipe.startswith('h') else None)  # stacked on the adopted head
    # ARCH_LOG hybrids: also vs hbase32_things, TSConv-bn64 re-implemented on the same code path
    for base in [base] + (['hbase32_things'] if recipe.startswith('h') and not recipe.startswith('hbase') else []):
        if base is None or recipe == base:
            continue
        b = glob.glob(f'{ROOT}/{base}/{fold}/*/2026*/sealed_per_epoch.jsonl')
        if not b:
            continue
        c, cb = curve(f), curve(b[0])
        ep = sorted(set(c) & set(cb))
        d = np.array([c[e] - cb[e] for e in ep])
        flag = 'KILL?' if len(d) >= 6 and d.mean() <= -1.5 else ('good' if len(d) >= 6 and d.mean() >= 2 else '')
        print(f'{recipe:28s} {fold}: {len(d):2d} ep, mean {d.mean():+5.2f}, last5 {d[-5:].mean():+5.2f}  {flag}' + (' (vs hbase32)' if base == 'hbase32_things' else ''))
