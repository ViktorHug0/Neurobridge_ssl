"""Average plain-cosine score matrices exported by evaluate.py."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _scores(path):
    data = np.load(path)
    eeg = data['eeg'].astype(np.float64)
    image = data['image'].astype(np.float64)
    eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    return eeg @ image.T, data['object'], data['image_idx']


def _accuracy(scores):
    target = np.arange(len(scores))
    top1 = float(np.mean(scores.argmax(axis=1) == target) * 100.0)
    top5_idx = np.argpartition(scores, -5, axis=1)[:, -5:]
    top5 = float(np.mean(np.any(top5_idx == target[:, None], axis=1)) * 100.0)
    return top1, top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-dir', required=True)
    parser.add_argument('--members', nargs='+', default=['group', 'pair'])
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows = []
    for subject in args.subjects:
        member_scores = []
        reference_labels = None
        row = {'subject': subject}
        for member in args.members:
            path = Path(args.dump_dir) / f'{member}-sub{subject:02d}.npz'
            scores, objects, images = _scores(path)
            labels = np.stack([objects, images], axis=1)
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise ValueError(f'Query order differs in {path}')
            member_scores.append(scores)
            row[f'{member}_top1'], row[f'{member}_top5'] = _accuracy(scores)
        ensemble = np.mean(member_scores, axis=0)
        row['ensemble_top1'], row['ensemble_top5'] = _accuracy(ensemble)
        rows.append(row)

    frame = pd.DataFrame(rows)
    average = {'subject': 'Average'}
    for column in frame.columns[1:]:
        average[column] = frame[column].mean()
    frame = pd.concat([frame, pd.DataFrame([average])], ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
