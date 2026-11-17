"""Appendix C (2026-11-16): calibration gain across encoders and image targets in the batch evaluation, with the final
GALA (column masses estimated from the batch, pseudo-count 1) without the prior (user), 16 iterations from the identity.
Settings selected cross-fold on the batch evaluation: per subject, the setting with the best mean top-1 over the other
nine subjects. Grid: rho x tau x power of select_stream.py for d = 128; the reduced grid of the stream selection of the
other encoders for d = 512 (1.5 s per fit).
Writes results/things_eeg/cvpr/analysis/encoder_gain_final.csv (arm, subject, m_plain, m_gala, rsa) and prints the
mean per arm. make_encoder_figure.py reads it.
usage: appendix_c.py [--workers 3]
"""
import argparse, itertools, os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
C = 'results/things_eeg/cvpr/emb_cache'
A = 'results/things_eeg/cvpr/analysis'
ARMS = ['atm128_valcon_s3300', 'atm128_vc_s3301', 'atm128_vc_s3302', 'tsconv_vc_s3300', 'eegnet_vc_s3300', 'eegproject_vc_s3300',
        'conformer_vc_s3300', 'labram_vc_s3300', 'atm128_iv20_vc_s3300', 'atm128_iv44_vc_s3300', 'atm128_dino40_vc_s3300',
        'atm128_cliph_vc_s3300', 'atm128_bigg27_vc_s3300']
FULL = list(itertools.product((0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94), (0.03, 0.05, 0.07, 0.1, 0.14, 0.2), (1.0, 1.2, 1.5)))
SMALL = list(itertools.product((0.8, 0.9, 0.94), (0.03, 0.05, 0.07, 0.1, 0.14), (1.0, 1.2)))


def job(arg):
    import gala
    arm, sid = arg
    z = np.load(f'{C}/{arm}_b/sub-{sid:02d}.npz')
    q, g = z['q'].astype(np.float64), z['g'].astype(np.float64)
    grid = FULL if q.shape[1] <= 128 else SMALL
    return [(arm, sid, *p, 100 * (gala.fit(q, g, *p, freq=1.0, shrink=False)(q).argmax(1) == np.arange(len(q))).mean()) for p in grid]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    raw = f'{A}/encoder_gain_final_grid.csv'
    if not os.path.exists(raw):
        with Pool(a.workers) as pool:
            out = [r for rs in pool.imap_unordered(job, [(arm, s) for arm in ARMS for s in range(1, 11)]) for r in rs]
        pd.DataFrame(out, columns=['arm', 'subject', 'rho', 'tau', 'gamma', 'top1']).to_csv(raw, index=False)
    d = pd.read_csv(raw)
    old = pd.concat([pd.read_csv(f'{A}/encoder_gain_diagnostics{x}_vc.csv') for x in ('', '_targets')]).set_index(['arm', 'subject'])
    rows = []
    for arm, da in d.groupby('arm'):
        t = da.pivot_table(index=['rho', 'tau', 'gamma'], columns='subject', values='top1')
        for s in t.columns:
            pick = t.drop(columns=s).mean(1).idxmax()
            rows.append((arm, s, old.loc[(arm, s), 'm_plain'], t.loc[pick, s], old.loc[(arm, s), 'rsa'], *pick))
    r = pd.DataFrame(rows, columns=['arm', 'subject', 'm_plain', 'm_gala', 'rsa', 'rho', 'tau', 'gamma'])
    r.to_csv(f'{A}/encoder_gain_final.csv', index=False)
    m = r.groupby('arm')[['m_plain', 'm_gala', 'rsa']].mean()
    m['fixed'] = 100 * (m.m_gala - m.m_plain) / (100 - m.m_plain)
    print(m.round(3).to_string())
