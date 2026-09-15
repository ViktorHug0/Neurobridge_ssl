"""Shared definitions for the September 2026 mechanism study."""
from pathlib import Path
import json
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / 'results/things_eeg/ensemble_mechanism_20260907'
DUMPS = REPO / 'results/things_eeg/synthetic_subjects/ensemble_screen/dumps'
EEG_DIR = '/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
FEATURES = REPO / 'data/things_eeg/image_feature'
TARGETS = {str(k): str(FEATURES / f'InternViT-6B_layer{k}_mean_8bit') for k in (28, 33)}
COMMITTEES = {
    'seed3': ['p3300', 'p3301', 'p3302'],
    'diverse3': ['atm_iv', 'tsconv_eva', 'tsconv_vith'],
    'quartet': ['atm_vith', 'atm_iv_group_e75', 'iv33_group_e75', 'sqf28'],
    'valcon4': ['atm_iv_valcon', 'atm33_valcon', 'iv33g_valcon', 'tsconv_bigg_valcon'],
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def row_z(x):
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True).clip(1e-8)


def load_member(name, subject):
    with np.load(DUMPS / f'{name}-sub{subject:02d}.npz') as f:
        eeg, image = unit(f['eeg'].astype(np.float64)), unit(f['image'].astype(np.float64))
        keys = np.stack([f['object'], f['image_idx']], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError(f'duplicate candidate identities: {name}, subject {subject}')
    return eeg @ image.T, image @ image.T, keys


def margin_metrics(scores):
    """All arrays are [member, query, candidate], with diagonal truth."""
    z = row_z(np.asarray(scores, dtype=np.float64))
    q = np.arange(z.shape[1])
    pos = z[:, q, q]
    wrong = z.copy()
    wrong[:, q, q] = -np.inf
    fused = z.mean(0)
    fw = fused.copy()
    fw[q, q] = -np.inf
    individual = pos - wrong.max(-1)
    bonus = wrong.max(-1).mean(0) - fw.max(-1)
    margin = fused[q, q] - fw.max(-1)
    np.testing.assert_allclose(margin, individual.mean(0) + bonus, atol=1e-10)
    assert bonus.min() > -1e-10
    correct = z.argmax(-1) == q
    ok = fused.argmax(-1) == q
    return {
        'top1': float(ok.mean() * 100),
        'mean_solo_top1': float(correct.mean() * 100),
        'oracle_top1': float(correct.any(0).mean() * 100),
        'mean_individual_margin': float(individual.mean()),
        'mean_ensemble_margin': float(margin.mean()),
        'distractor_bonus': float(bonus.mean()),
        'all_wrong_rescues': int((ok & ~correct.any(0)).sum()),
        'bonus_crossings': int((ok & (individual.mean(0) <= 0)).sum()),
    }
