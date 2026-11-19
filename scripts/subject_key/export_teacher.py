"""Teacher file for train.py --kd_teacher: for one fold and a list of finished runs (the members), the L2-normalised
embedding of every clean training trial (9 training subjects x 16,540 images, 4-repetition averages, subject-major
order) and of every training image, float16.
    export_teacher.py <name> <fold> <run> [<run> ...]   -> results/things_eeg/subject_key/teachers/<name>_f<fold>.pt"""
from sk_lib import *

NAME, FOLD, RUNS = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
out = f'{SK}/teachers/{NAME}_f{FOLD}.pt'
os.makedirs(os.path.dirname(out), exist_ok=True)
E, G = [], []
for run in RUNS:
    m = load(run, FOLD)
    assert m.test_sub == FOLD
    E.append(torch.cat([nz(m.embed(T(eeg(s, 'train')), s), dim=1).half() for s in m.train_subs]))
    G.append(img_emb(m, 'train').half())
    print(run, tuple(E[-1].shape), flush=True)
torch.save({'subjects': m.train_subs, 'runs': RUNS, 'eeg': E, 'img': G}, out)
print('saved', out)
