"""Score every best-4 expert (per LOSO fold) on EVERY subject's 200-image test set.

Fold k's expert was trained on subjects != k. Its scores on subject j != k are
'seen subject, unseen image'; on subject k they are 'unseen subject, unseen image'
(the existing benchmark dumps). Both are needed to build router training data.
"""
import glob, json, os, subprocess, sys
MAN = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
E = ['atm_iv', 'ge100', 'tsconv_eva', 'tsconv_vith']
OUT = 'results/things_eeg/synthetic_subjects/router/dumps'
EV = 'results/things_eeg/synthetic_subjects/router/eval'
os.makedirs(OUT, exist_ok=True)
for n in E:
    for fold in range(1, 11):
        ck = MAN[n]['folds'][str(fold)]
        for sub in range(1, 11):
            p = f'{OUT}/{n}-f{fold:02d}-s{sub:02d}.npz'
            if os.path.exists(p):
                continue
            r = subprocess.run([sys.executable, 'evaluate.py', '--checkpoint_dir', ck,
                                '--output_dir', EV, '--output_name', f'{n}-f{fold:02d}-s{sub:02d}',
                                '--eval_mode', 'plain_cosine', '--test_subject_id', str(sub),
                                '--device', 'cuda:0', '--batch_size', '32', '--num_workers', '0',
                                '--dump_npz', p], capture_output=True, text=True)
            if r.returncode:
                print(f'FAIL {n} f{fold} s{sub}: {r.stderr.strip().splitlines()[-1]}', flush=True)
        print(f'done {n} fold {fold}', flush=True)
