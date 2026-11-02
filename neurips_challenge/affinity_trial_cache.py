"""Per-trial caches (trial_cache.cache_run) for affinity models s<i>: 5 training subjects, 15 unseen test subjects
each -- the only models with enough unseen subjects to study batches mixing N = 1..15 participants.
    affinity_trial_cache.py 1 2 3 4"""
import glob, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_cache import cache_run  # noqa: E402

for i in sys.argv[1:]:
    d = glob.glob(f'results/alljoined/track1/affinity/s{i}/reve/2026*')[0]
    cache_run(d, torch.device('cuda:0'))
