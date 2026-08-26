import glob, os, subprocess, sys
D='results/things_eeg/synthetic_subjects/ensemble_screen/dumps'
E='results/things_eeg/synthetic_subjects/ensemble_screen/eval'
ARMS={'sm_eva':'eva_s3300','sm_vith':'vith_s3300','sm_eva1':'eva_s3301','sm_eva2':'eva_s3302'}
for tag,arm in ARMS.items():
    root=glob.glob(f'results/things_eeg/synthetic_subjects/shrunk_ensemble/{arm}/seed*')[0]
    for s in range(1,11):
        out=f'{D}/{tag}-sub{s:02d}.npz'
        if os.path.exists(out): continue
        ck=glob.glob(f'{root}/*-sub-{s:02d}')[0]
        r=subprocess.run([sys.executable,'evaluate.py','--checkpoint_dir',ck,'--output_dir',E,
            '--output_name',f'{tag}-sub{s:02d}','--eval_mode','plain_cosine','--test_subject_id',str(s),
            '--device','cuda:0','--batch_size','32','--num_workers','0','--dump_npz',out],
            capture_output=True,text=True)
        print(('ok ' if r.returncode==0 else 'FAIL ')+f'{tag} sub{s:02d}'+('' if r.returncode==0 else ' '+r.stderr.strip().splitlines()[-1]),flush=True)
