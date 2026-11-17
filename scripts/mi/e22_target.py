"""A5. Target side: which images / categories transfer to an unseen subject?
Per test image: hit rate when the subject is held out vs when it is seen (plain cosine; TSConv, ATM, sqf, EEGProject,
10 folds), by THINGS high-level category (category_mat_manual.tsv, 27 categories), and against two properties of the
image target: its mean cosine to the other 199 gallery targets (typicality) and to its nearest gallery neighbour."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO

IMG = '/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set'
test = [d.split('_', 1)[1] for d in sorted(os.listdir(f'{IMG}/test_images'))]
train = [d.split('_', 1)[1] for d in sorted(os.listdir(f'{IMG}/training_images'))]
allc = sorted(train + test)
cat = pd.read_csv('/nasbrain/p20fores/ds004212/sourcedata/category_mat_manual.tsv', sep='\t')
assert len(allc) == len(cat) == 1854
tcat = cat.iloc[[allc.index(c) for c in test]].reset_index(drop=True)

hit = {}
for arch in ('tsconv', 'atm128', 'sqf', 'eegproject'):
    H = np.zeros((10, 10, 200))  # fold, subject, image
    for fold in range(1, 11):
        m = GM(glob.glob(f'{LOSO}/{arch}/seed3300/*-sub-{fold:02d}')[0])
        g = img_emb(m)
        for s in range(1, 11):
            e = torch.nn.functional.normalize(m.embed(T(eeg(s)), s), dim=1)
            H[fold - 1, s - 1] = ((e @ g.T).argmax(1) == torch.arange(200)).numpy()
        if fold == 1:
            S = g @ g.T
            typ = ((S.sum(1) - 1) / 199).numpy()
            nn_ = S.fill_diagonal_(-1).max(1).values.numpy()
    held = np.stack([H[f, f] for f in range(10)]).mean(0)
    seen = np.stack([H[f, [s for s in range(10) if s != f]].mean(0) for f in range(10)]).mean(0)
    hit[arch] = (held, seen, typ, nn_)
    print(arch, 'held %.1f seen %.1f | corr(held, seen) over images %.2f | corr(held, typicality) %.2f, corr(seen, typicality) %.2f | '
          'corr(held, nn sim) %.2f, corr(seen, nn sim) %.2f' % (100 * held.mean(), 100 * seen.mean(), np.corrcoef(held, seen)[0, 1],
          np.corrcoef(held, typ)[0, 1], np.corrcoef(seen, typ)[0, 1], np.corrcoef(held, nn_)[0, 1], np.corrcoef(seen, nn_)[0, 1]), flush=True)
held = np.mean([h[0] for h in hit.values()], 0)
seen = np.mean([h[1] for h in hit.values()], 0)
df = pd.DataFrame(dict(concept=test, held=held, seen=seen, ratio=held / np.clip(seen, 1e-3, None)))
rows = []
for c in tcat.columns:
    idx = tcat[c].values == 1
    if idx.sum() >= 5:
        rows.append(dict(category=c, n=int(idx.sum()), held=100 * held[idx].mean(), seen=100 * seen[idx].mean(), ratio=held[idx].mean() / seen[idx].mean()))
none = tcat.values.sum(1) == 0
rows.append(dict(category='(no category)', n=int(none.sum()), held=100 * held[none].mean(), seen=100 * seen[none].mean(), ratio=held[none].mean() / seen[none].mean()))
out = pd.DataFrame(rows).sort_values('ratio')
print(out.round(2).to_string(index=False))
print('images never retrieved when held out: %d ; always (>= 0.75): %d' % ((held == 0).sum(), (held >= 0.75).sum()))
print(df.sort_values('held').head(8).round(2).to_string(index=False)); print(df.sort_values('held').tail(8).round(2).to_string(index=False))
df.to_csv(f'{OUT}/e22_target_images.csv', index=False); out.to_csv(f'{OUT}/e22_target_categories.csv', index=False)
